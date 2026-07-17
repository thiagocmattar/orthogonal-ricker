"""Report 05 post-hoc clipping reductions and renderers.

The inputs are deliberately close to saved ``clipping_frontier.jsonl`` rows:

``site_sweeps[architecture_id][site]`` is a list of three mappings, one per
method, with ``method`` (AdamW, OR, or OL1) and ``rows``.  ``joint_sweeps``
has the same method mappings directly under each architecture ID.  The joint
rows must contain the direct logical-product counters written by clipping
sweeps run with ``--measure-zero-products``.

Every validation-loss change is referenced to threshold 0 from the same
sweep.  This is scientifically important because the exact-product diagnostic
forces eager attention and its absolute threshold-0 loss must not be compared
with a loss measured by a different evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_exp.plot_api import DOUBLE_COLUMN_WIDTH_INCHES
from paper_exp.plot_style import SeriesStyle, report04_method_style


REPORT05_CLIPPING_ARCHITECTURES = (
    ("one_relu", "One-ReLU"),
    ("three_relu", "Three-ReLU"),
    ("six_relu_pre", "Six-ReLU PRE"),
    ("six_relu_post", "Six-ReLU POST"),
)
REPORT05_CLIPPING_METHODS = ("AdamW", "OR", "OL1")
REPORT05_CLIPPING_SITES = (
    ("attention_inputs", "Attention input"),
    ("mlp_inputs", "MLP input"),
    ("mlp_hiddens", "MLP hidden"),
    ("query_gate_outputs", "Q gate"),
    ("key_gate_outputs", "K gate"),
    ("value_gate_outputs", "V gate"),
)
REPORT05_ACTIVE_CLIPPING_SITES = {
    "one_relu": ("mlp_hiddens",),
    "three_relu": ("attention_inputs", "mlp_inputs", "mlp_hiddens"),
    "six_relu_pre": tuple(site for site, _label in REPORT05_CLIPPING_SITES),
    "six_relu_post": tuple(site for site, _label in REPORT05_CLIPPING_SITES),
}


_METHOD_STYLES: dict[str, SeriesStyle] = {
    "AdamW": report04_method_style("three_relu_adamw"),
    "OR": report04_method_style("three_relu_or"),
    "OL1": report04_method_style("three_relu_ol1"),
}


def _reduce_report05_site_clipping_frontiers(
    site_sweeps: Mapping[
        str,
        Mapping[str, Sequence[Mapping[str, Any]]],
    ],
) -> tuple[dict[str, Any], ...]:
    """Reduce exact-zero site sweeps to within-sweep loss changes."""

    _require_architecture_keys(site_sweeps, context="site clipping")
    site_labels = dict(REPORT05_CLIPPING_SITES)
    reduced_architectures: list[dict[str, Any]] = []
    for architecture_id, architecture_label in REPORT05_CLIPPING_ARCHITECTURES:
        supplied_sites = site_sweeps[architecture_id]
        required_sites = REPORT05_ACTIVE_CLIPPING_SITES[architecture_id]
        supplied_keys = set(supplied_sites)
        required_keys = set(required_sites)
        missing = sorted(required_keys - supplied_keys)
        unexpected = sorted(supplied_keys - required_keys)
        if missing or unexpected:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unexpected:
                detail.append("unexpected " + ", ".join(unexpected))
            raise ValueError(
                f"Invalid site clipping inputs for {architecture_id}: "
                + "; ".join(detail)
                + "."
            )

        reduced_sites: dict[str, Any] = {}
        for site in required_sites:
            method_items = _method_items(
                supplied_sites[site],
                context=f"{architecture_id}/{site}",
            )
            reduced_series = []
            for method in REPORT05_CLIPPING_METHODS:
                rows = method_items[method].get("rows")
                if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
                    raise ValueError(
                        f"Clipping rows for {architecture_id}/{site}/{method} must be a sequence."
                    )
                points, baseline_loss = _relative_threshold_points(
                    rows,
                    fraction=lambda row, selected_site=site: _site_exact_zero_fraction(
                        row,
                        selected_site,
                    ),
                    context=f"{architecture_id}/{site}/{method}",
                )
                reduced_series.append(
                    {
                        "method": method,
                        "baseline_loss": baseline_loss,
                        "points": points,
                    }
                )
            reduced_sites[site] = {
                "site": site,
                "site_label": site_labels[site],
                "series": tuple(reduced_series),
            }
        reduced_architectures.append(
            {
                "architecture_id": architecture_id,
                "architecture_label": architecture_label,
                "sites": reduced_sites,
            }
        )
    return tuple(reduced_architectures)


def _reduce_report05_model_matmul_frontiers(
    joint_sweeps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    """Reduce direct logical-product joint sweeps to model-level frontiers."""

    _require_architecture_keys(joint_sweeps, context="joint clipping")
    reduced_architectures: list[dict[str, Any]] = []
    for architecture_id, architecture_label in REPORT05_CLIPPING_ARCHITECTURES:
        method_items = _method_items(
            joint_sweeps[architecture_id],
            context=f"{architecture_id}/joint",
        )
        reduced_series = []
        for method in REPORT05_CLIPPING_METHODS:
            rows = method_items[method].get("rows")
            if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
                raise ValueError(
                    f"Joint clipping rows for {architecture_id}/{method} must be a sequence."
                )
            points, baseline_loss = _relative_threshold_points(
                rows,
                fraction=_direct_model_avoidable_fraction,
                context=f"{architecture_id}/joint/{method}",
            )
            reduced_series.append(
                {
                    "method": method,
                    "baseline_loss": baseline_loss,
                    "points": points,
                }
            )
        reduced_architectures.append(
            {
                "architecture_id": architecture_id,
                "architecture_label": architecture_label,
                "series": tuple(reduced_series),
            }
        )
    return tuple(reduced_architectures)


def _relative_threshold_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: Any,
    context: str,
) -> tuple[tuple[dict[str, Any], ...], float]:
    """Return finite threshold points relative to this sweep's threshold 0."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("mode") not in (None, "threshold"):
            continue
        threshold = _finite_float(row.get("threshold"))
        validation_loss = _finite_float(row.get("validation_loss"))
        if threshold is None or validation_loss is None:
            continue
        fraction_value = float(fraction(row))
        if not math.isfinite(fraction_value) or not -1e-12 <= fraction_value <= 1.0 + 1e-12:
            raise ValueError(
                f"Exact-zero fraction for {context} must be finite and between 0 and 1."
            )
        validation_tokens = _positive_int(row.get("validation_tokens"), context=context)
        normalized.append(
            {
                "threshold": threshold,
                "fraction": min(1.0, max(0.0, fraction_value)),
                "validation_loss": validation_loss,
                "validation_tokens": validation_tokens,
            }
        )
    if not normalized:
        raise ValueError(f"No finite threshold rows for {context}.")

    thresholds = [point["threshold"] for point in normalized]
    duplicates = sorted(
        threshold
        for threshold in set(thresholds)
        if thresholds.count(threshold) > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate clipping thresholds for {context}: "
            + ", ".join(f"{threshold:g}" for threshold in duplicates)
            + "."
        )
    baseline_points = [point for point in normalized if abs(point["threshold"]) <= 1e-12]
    if len(baseline_points) != 1:
        raise ValueError(
            f"Expected exactly one threshold-0 row for {context}; found {len(baseline_points)}."
        )
    validation_token_counts = {point["validation_tokens"] for point in normalized}
    if len(validation_token_counts) != 1:
        raise ValueError(f"Validation token count changes within clipping sweep {context}.")

    baseline_loss = float(baseline_points[0]["validation_loss"])
    points = tuple(
        {
            "threshold": float(point["threshold"]),
            "x_percent": 100.0 * float(point["fraction"]),
            "loss_change": float(point["validation_loss"]) - baseline_loss,
            "validation_loss": float(point["validation_loss"]),
            "validation_tokens": int(point["validation_tokens"]),
            "is_threshold_zero": abs(float(point["threshold"])) <= 1e-12,
        }
        for point in sorted(normalized, key=lambda point: float(point["threshold"]))
    )
    return points, baseline_loss


def _site_exact_zero_fraction(row: Mapping[str, Any], site: str) -> float:
    by_site = row.get("site_achieved_sparsity")
    if isinstance(by_site, Mapping) and site in by_site:
        value = _finite_float(by_site[site])
        if value is not None:
            return value
    sites = row.get("sites")
    if isinstance(sites, Sequence) and not isinstance(sites, str | bytes):
        if list(sites) == [site]:
            value = _finite_float(row.get("achieved_sparsity"))
            if value is not None:
                return value
    raise ValueError(f"Clipping row does not contain an exact-zero fraction for site {site!r}.")


def _direct_model_avoidable_fraction(row: Mapping[str, Any]) -> float:
    zero_count = _nonnegative_int(row.get("block_zero_product_count"))
    model_count = _positive_int(
        row.get("model_matmul_product_count"),
        context="direct model-matmul frontier",
    )
    if zero_count is None:
        raise ValueError(
            "Joint clipping row is missing block_zero_product_count; use a sweep run "
            "with --measure-zero-products."
        )
    if zero_count > model_count:
        raise ValueError("Logical zero-product count exceeds the model product denominator.")
    direct_fraction = zero_count / model_count
    stored_fraction = row.get("potentially_avoidable_model_matmul_fraction")
    if stored_fraction is not None:
        stored = _finite_float(stored_fraction)
        if stored is None or not math.isclose(
            stored,
            direct_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Stored potentially avoidable model fraction disagrees with direct counters."
            )
    return direct_fraction


def _method_items(
    items: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(items, str | bytes):
        raise ValueError(f"Method sweeps for {context} must be a sequence.")
    grouped: dict[str, list[Mapping[str, Any]]] = {
        method: [] for method in REPORT05_CLIPPING_METHODS
    }
    unexpected: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            unexpected.append(type(item).__name__)
            continue
        method = _method_name(item)
        if method not in grouped:
            unexpected.append(str(method))
            continue
        grouped[method].append(item)
    missing = [method for method, matches in grouped.items() if not matches]
    duplicates = [method for method, matches in grouped.items() if len(matches) > 1]
    if missing or duplicates or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if duplicates:
            detail.append("duplicate " + ", ".join(duplicates))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"Invalid method cohort for {context}: " + "; ".join(detail) + ".")
    return {method: grouped[method][0] for method in REPORT05_CLIPPING_METHODS}


def _method_name(item: Mapping[str, Any]) -> str:
    method = str(item.get("method") or "").strip()
    if method:
        return method
    label = str(item.get("label") or "").strip()
    for candidate in REPORT05_CLIPPING_METHODS:
        if label == candidate or label.endswith(" " + candidate):
            return candidate
    return label


def _require_architecture_keys(inputs: Mapping[str, Any], *, context: str) -> None:
    required = {architecture_id for architecture_id, _label in REPORT05_CLIPPING_ARCHITECTURES}
    supplied = set(inputs)
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - required)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"Invalid Report 05 {context} architectures: " + "; ".join(detail) + ".")


def _plot_report05_site_clipping_frontiers(
    reduced: Sequence[Mapping[str, Any]],
) -> Figure:
    """Render a four-by-six shared-scale site-specific frontier grid."""

    architectures = _ordered_reduced_architectures(reduced, require_sites=True)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=6,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 8.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    all_loss_changes = [
        float(point["loss_change"])
        for architecture in architectures
        for site_payload in architecture["sites"].values()
        for series in site_payload["series"]
        for point in series["points"]
    ]
    y_limits = _loss_change_limits(all_loss_changes)
    validation_tokens = _single_validation_token_count(architectures)
    site_labels = dict(REPORT05_CLIPPING_SITES)

    for column, (site, site_label) in enumerate(REPORT05_CLIPPING_SITES):
        axes[0, column].set_title(site_label, fontsize=8.0, pad=5.0)
        for row, architecture in enumerate(architectures):
            ax = axes[row, column]
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.65, zorder=1)
            # Small shared padding keeps measurements at exactly 0% or 100%
            # from being hidden by the axes spines.
            ax.set_xlim(-1.5, 101.5)
            ax.set_ylim(*y_limits)
            ax.set_xticks((0.0, 50.0, 100.0))
            ax.tick_params(
                axis="both",
                labelsize=8.0,
                labelbottom=row == len(architectures) - 1,
                labelleft=column == 0,
            )
            if row == len(architectures) - 1:
                # Avoid the adjacent ``100``/``0`` labels merging between the
                # six narrow shared-axis panels.  The common scale remains
                # explicit at the outer bounds and at 50% in every panel.
                tick_labels = ax.get_xticklabels()
                if column == 0:
                    tick_labels[-1].set_visible(False)
                elif column == len(REPORT05_CLIPPING_SITES) - 1:
                    tick_labels[0].set_visible(False)
                else:
                    tick_labels[0].set_visible(False)
                    tick_labels[-1].set_visible(False)
            site_payload = architecture["sites"].get(site)
            if site_payload is None:
                ax.set_facecolor("#F4F4F4")
                ax.grid(False)
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#777777",
                )
            else:
                for series in site_payload["series"]:
                    method = str(series["method"])
                    points = series["points"]
                    style = _METHOD_STYLES[method]
                    ax.plot(
                        [float(point["x_percent"]) for point in points],
                        [float(point["loss_change"]) for point in points],
                        color=style.color,
                        marker=style.marker,
                        linestyle=style.linestyle,
                        linewidth=1.05,
                        markersize=2.4,
                        zorder=2,
                    )
            if column == 0:
                ax.text(
                    0.03,
                    0.94,
                    str(architecture["architecture_label"]),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.0,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
                    zorder=4,
                )

    fig.suptitle("Site-Specific Post-Hoc Clipping Frontiers", y=0.985)
    fig.text(
        0.5,
        0.955,
        (
            f"{validation_tokens:,} validation tokens per point; all panels share axes; "
            "loss change is relative to threshold 0 within the same sweep"
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.113,
        "Achieved exact zeros at the clipped site (%)",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    fig.text(
        0.013,
        0.52,
        "Validation loss change from threshold 0",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.0,
    )
    fig.legend(
        handles=_method_legend_handles(),
        labels=list(REPORT05_CLIPPING_METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.058),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=2.6,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Exact zero means x == 0 with no tolerance. Lines connect measured thresholds.\n"
            "Logical zeros are not measured dense-kernel speedup."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.080, right=0.985, top=0.910, bottom=0.155, hspace=0.25, wspace=0.16)
    return fig


def _plot_report05_model_matmul_frontiers(
    reduced: Sequence[Mapping[str, Any]],
) -> Figure:
    """Render model-level exact logical-product frontiers on shared axes."""

    architectures = _ordered_reduced_architectures(reduced, require_sites=False)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 7.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = tuple(axes[:, 0])
    all_points = [
        point
        for architecture in architectures
        for series in architecture["series"]
        for point in series["points"]
    ]
    x_values = [float(point["x_percent"]) for point in all_points]
    loss_changes = [float(point["loss_change"]) for point in all_points]
    x_upper = _percentage_upper_limit(x_values)
    y_limits = _loss_change_limits(loss_changes)
    validation_tokens = _single_validation_token_count(architectures)

    for ax, architecture in zip(flat_axes, architectures, strict=True):
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.75, zorder=1)
        for series in architecture["series"]:
            method = str(series["method"])
            points = series["points"]
            style = _METHOD_STYLES[method]
            ax.plot(
                [float(point["x_percent"]) for point in points],
                [float(point["loss_change"]) for point in points],
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                linewidth=1.35,
                markersize=3.5,
                label=method,
                zorder=2,
            )
            threshold_zero = next(point for point in points if point["is_threshold_zero"])
            ax.scatter(
                [float(threshold_zero["x_percent"])],
                [0.0],
                marker="*",
                s=62,
                facecolor=style.color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        ax.set_title(
            str(architecture["architecture_label"]),
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=3.0,
        )
        ax.set_xlim(0.0, x_upper)
        ax.set_ylim(*y_limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=(1, 2, 2.5, 5, 10)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10)))
        ax.tick_params(labelsize=8.0)

    fig.suptitle("Post-Hoc Clipping vs Potentially Avoidable Model Matmul Products", y=0.985)
    fig.text(
        0.5,
        0.950,
        (
            f"{validation_tokens:,} validation tokens per point; direct actual-operand counters.\n"
            "Loss change is relative to threshold 0 within each eager-attention sweep."
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.text(
        0.5,
        0.115,
        "Potentially avoidable model matmul products (%)",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    fig.text(
        0.025,
        0.525,
        "Validation loss change from threshold 0",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.0,
    )
    semantic_handle = Line2D(
        [],
        [],
        marker="*",
        color="none",
        markerfacecolor="#555555",
        markeredgecolor="white",
        markersize=8.5,
        label="Threshold 0",
    )
    fig.legend(
        handles=[*_method_legend_handles(), semantic_handle],
        labels=[*REPORT05_CLIPPING_METHODS, "Threshold 0"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.062),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=2.6,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Numerator: exact-zero logical products in QKV, valid-causal QK, valid-causal PV, W_O, W1, and W2.\n"
            "Denominator adds the LM head; this is an opportunity proxy, not speedup."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.090, right=0.985, top=0.895, bottom=0.160, hspace=0.22)
    return fig


def _ordered_reduced_architectures(
    reduced: Sequence[Mapping[str, Any]],
    *,
    require_sites: bool,
) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        architecture_id: []
        for architecture_id, _label in REPORT05_CLIPPING_ARCHITECTURES
    }
    for architecture in reduced:
        architecture_id = str(architecture.get("architecture_id") or "")
        if architecture_id not in grouped:
            raise ValueError(f"Unexpected reduced architecture: {architecture_id!r}.")
        grouped[architecture_id].append(architecture)
    missing = [architecture_id for architecture_id, matches in grouped.items() if not matches]
    duplicates = [architecture_id for architecture_id, matches in grouped.items() if len(matches) > 1]
    if missing or duplicates:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if duplicates:
            detail.append("duplicate " + ", ".join(duplicates))
        raise ValueError("Invalid reduced Report 05 clipping cohort: " + "; ".join(detail) + ".")
    ordered = tuple(grouped[architecture_id][0] for architecture_id in grouped)
    key = "sites" if require_sites else "series"
    for architecture in ordered:
        if key not in architecture:
            raise ValueError(
                f"Reduced architecture {architecture.get('architecture_id')!r} is missing {key!r}."
            )
    return ordered


def _single_validation_token_count(architectures: Sequence[Mapping[str, Any]]) -> int:
    token_counts: set[int] = set()
    for architecture in architectures:
        if "sites" in architecture:
            all_series = (
                series
                for site_payload in architecture["sites"].values()
                for series in site_payload["series"]
            )
        else:
            all_series = iter(architecture["series"])
        for series in all_series:
            for point in series["points"]:
                token_counts.add(int(point["validation_tokens"]))
    if len(token_counts) != 1:
        raise ValueError(
            "Report 05 clipping figures require one shared validation-token count; found "
            + ", ".join(f"{count:,}" for count in sorted(token_counts))
            + "."
        )
    return next(iter(token_counts))


def _method_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=_METHOD_STYLES[method].color,
            marker=_METHOD_STYLES[method].marker,
            linestyle=_METHOD_STYLES[method].linestyle,
            linewidth=1.35,
            markersize=4.0,
            label=method,
        )
        for method in REPORT05_CLIPPING_METHODS
    ]


def _loss_change_limits(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("A clipping figure requires at least one finite loss change.")
    minimum = min(0.0, min(finite))
    maximum = max(0.0, max(finite))
    span = max(maximum - minimum, abs(maximum), abs(minimum), 1e-3)
    return minimum - 0.04 * span, maximum + 0.08 * span


def _percentage_upper_limit(values: Sequence[float]) -> float:
    maximum = max((float(value) for value in values if math.isfinite(float(value))), default=0.0)
    padded = max(5.0, 1.06 * maximum)
    return 5.0 * math.ceil(padded / 5.0)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted >= 0 else None


def _positive_int(value: Any, *, context: str) -> int:
    converted = _nonnegative_int(value)
    if converted is None or converted <= 0:
        raise ValueError(f"Expected a positive integer for {context}; got {value!r}.")
    return converted
