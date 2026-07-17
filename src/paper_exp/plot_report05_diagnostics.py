"""Report 05 activation-propagation and distribution renderers.

This module is intentionally presentation-local.  It accepts already loaded
JSON payloads and checkpoint summaries, performs the scientific reductions,
and returns Matplotlib figures.  Result discovery, file I/O, export, and CLI
wiring remain in :mod:`paper_exp.plots`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from paper_exp.plot_api import DOUBLE_COLUMN_WIDTH_INCHES, PublicationProfile
from paper_exp.plot_common import _histogram_nonzero_density, _trimmed_decimal_tick
from paper_exp.plot_style import SeriesStyle, report04_method_style


REPORT05_HEATMAP_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=8.8,
    min_text_points=5.3,
)


@dataclass(frozen=True)
class _StageSpec:
    name: str
    label: str
    emphasized: bool = False


@dataclass(frozen=True)
class _ArchitectureSpec:
    architecture_id: str
    title: str
    runs: tuple[tuple[str, str, str], ...]
    stages: tuple[_StageSpec, ...]
    separator_after: tuple[int, ...]
    has_post_qkv_gates: bool = False


_METHOD_STYLES: dict[str, SeriesStyle] = {
    "AdamW": report04_method_style("Three-ReLU AdamW"),
    "OR": report04_method_style("Three-ReLU OR"),
    "OL1": report04_method_style("Three-ReLU OL1"),
}

_COMMON_ATTENTION_START = (
    _StageSpec("residual_input", "Block input H(l)"),
    _StageSpec("attention_layernorm_raw", "Attention LN"),
)
_DENSE_QKV_PATH = (
    _StageSpec("query_projection_output", "Raw Q after QKV"),
    _StageSpec("key_projection_output", "Raw K after QKV"),
    _StageSpec("value_projection_output", "Raw V after QKV"),
    _StageSpec("query_qk_input", "Q operand of QK"),
    _StageSpec("key_qk_input", "K operand of QK"),
    _StageSpec("value_pv_input", "V operand of PV"),
)
_ATTENTION_END = (
    _StageSpec("attention_probabilities", "P after softmax"),
    _StageSpec("attention_context", "C = PV"),
    _StageSpec("attention_output", "O after Wout"),
)
_ONE_RELU_MLP = (
    _StageSpec("mlp_layernorm_raw", "MLP LN"),
    _StageSpec("mlp_w1_preactivation", "U after W1"),
    _StageSpec("mlp_hidden_relu", "Hidden ReLU -> W2", True),
    _StageSpec("mlp_output", "M after W2"),
    _StageSpec("residual_output", "Block output H(l+1)"),
)
_THREE_RELU_MLP = (
    _StageSpec("mlp_layernorm_raw", "MLP LN"),
    _StageSpec("mlp_input_relu", "MLP ReLU -> W1", True),
    _StageSpec("mlp_w1_preactivation", "U after W1"),
    _StageSpec("mlp_hidden_relu", "Hidden ReLU -> W2", True),
    _StageSpec("mlp_output", "M after W2"),
    _StageSpec("residual_output", "Block output H(l+1)"),
)


REPORT05_DIAGNOSTIC_ARCHITECTURES: dict[str, _ArchitectureSpec] = {
    "one_relu": _ArchitectureSpec(
        architecture_id="one_relu",
        title="One-ReLU (MLP hidden)",
        runs=(
            ("AdamW", "One-ReLU AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            (
                "OR",
                "One-ReLU OR",
                "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
            ),
            (
                "OL1",
                "One-ReLU OL1",
                "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5",
            ),
        ),
        stages=(*_COMMON_ATTENTION_START, *_DENSE_QKV_PATH, *_ATTENTION_END, *_ONE_RELU_MLP),
        separator_after=(1, 10, 14),
    ),
    "three_relu": _ArchitectureSpec(
        architecture_id="three_relu",
        title="Three-ReLU",
        runs=(
            (
                "AdamW",
                "Three-ReLU AdamW",
                "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass",
            ),
            (
                "OR",
                "Three-ReLU OR",
                "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
            ),
            (
                "OL1",
                "Three-ReLU OL1",
                "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5",
            ),
        ),
        stages=(
            *_COMMON_ATTENTION_START,
            _StageSpec("attention_input_relu", "Attention ReLU -> QKV", True),
            *_DENSE_QKV_PATH,
            *_ATTENTION_END,
            *_THREE_RELU_MLP,
        ),
        separator_after=(1, 11, 16),
    ),
    "six_relu_pre": _ArchitectureSpec(
        architecture_id="six_relu_pre",
        title="Six-ReLU PRE (Q/K gate before RoPE)",
        runs=(
            (
                "AdamW",
                "Six-ReLU PRE AdamW",
                "107-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-adamw-full-pass",
            ),
            (
                "OR",
                "Six-ReLU PRE OR",
                "108-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
            ),
            (
                "OL1",
                "Six-ReLU PRE OL1",
                "109-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-l1-full-pass-w5",
            ),
        ),
        stages=(
            *_COMMON_ATTENTION_START,
            _StageSpec("attention_input_relu", "Attention ReLU -> QKV", True),
            _StageSpec("query_projection_output", "Raw Q after QKV"),
            _StageSpec("key_projection_output", "Raw K after QKV"),
            _StageSpec("value_projection_output", "Raw V after QKV"),
            _StageSpec("query_gate_output", "Q ReLU before RoPE", True),
            _StageSpec("key_gate_output", "K ReLU before RoPE", True),
            _StageSpec("value_gate_output", "V ReLU -> PV", True),
            _StageSpec("query_qk_input", "Q after RoPE -> QK"),
            _StageSpec("key_qk_input", "K after RoPE -> QK"),
            *_ATTENTION_END,
            *_THREE_RELU_MLP,
        ),
        separator_after=(1, 13, 18),
        has_post_qkv_gates=True,
    ),
    "six_relu_post": _ArchitectureSpec(
        architecture_id="six_relu_post",
        title="Six-ReLU POST (Q/K gate after RoPE)",
        runs=(
            (
                "AdamW",
                "Six-ReLU POST AdamW",
                "110-pythia-14m-minipile-post-qkv-relu-qk-post-rope-adamw-full-pass",
            ),
            (
                "OR",
                "Six-ReLU POST OR",
                "111-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
            ),
            (
                "OL1",
                "Six-ReLU POST OL1",
                "112-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-l1-full-pass-w5",
            ),
        ),
        stages=(
            *_COMMON_ATTENTION_START,
            _StageSpec("attention_input_relu", "Attention ReLU -> QKV", True),
            _StageSpec("query_projection_output", "Raw Q after QKV"),
            _StageSpec("key_projection_output", "Raw K after QKV"),
            _StageSpec("value_projection_output", "Raw V after QKV"),
            _StageSpec("query_gate_input", "Q after RoPE"),
            _StageSpec("key_gate_input", "K after RoPE"),
            _StageSpec("query_gate_output", "Q ReLU -> QK", True),
            _StageSpec("key_gate_output", "K ReLU -> QK", True),
            _StageSpec("value_gate_output", "V ReLU -> PV", True),
            *_ATTENTION_END,
            *_THREE_RELU_MLP,
        ),
        separator_after=(1, 13, 18),
        has_post_qkv_gates=True,
    ),
}


def _architecture_spec(architecture_id: str) -> _ArchitectureSpec:
    try:
        return REPORT05_DIAGNOSTIC_ARCHITECTURES[architecture_id]
    except KeyError as exc:
        valid = ", ".join(REPORT05_DIAGNOSTIC_ARCHITECTURES)
        raise ValueError(f"Unknown Report 05 architecture {architecture_id!r}; expected one of: {valid}.") from exc


def _select_architecture_methods(
    payload: dict[str, Any],
    architecture_id: str,
) -> tuple[dict[str, Any], ...]:
    """Select the three matched methods by immutable training config ID."""

    spec = _architecture_spec(architecture_id)
    methods = list(payload.get("methods", []))
    selected: list[dict[str, Any]] = []
    for _method, display_label, config_id in spec.runs:
        matches = [item for item in methods if str(item.get("config_id")) == config_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one propagation method for {display_label} ({config_id}); "
                f"found {len(matches)}."
            )
        selected.append(matches[0])
    return tuple(selected)


def _propagation_matrix(
    method: dict[str, Any],
    stages: tuple[_StageSpec, ...],
    *,
    num_layers: int,
) -> list[list[float]]:
    """Pool direct integer counts by stage; never average layer percentages."""

    lookup = {
        (str(row.get("name")), int(row.get("layer"))): row
        for row in method.get("activations", [])
    }
    matrix: list[list[float]] = []
    for stage in stages:
        values: list[float] = []
        pooled_zero = 0
        pooled_total = 0
        for layer in range(num_layers):
            row = lookup.get((stage.name, layer))
            if row is None:
                raise ValueError(
                    f"Missing propagation row activations/{stage.name}/layer_{layer} "
                    f"for {method.get('config_id')!r}."
                )
            if row.get("available", True) is False:
                reason = row.get("unavailable_reason") or "unspecified"
                raise ValueError(
                    f"Propagation stage {stage.name!r} is unavailable for "
                    f"{method.get('config_id')!r}: {reason}."
                )
            zero_count = int(row.get("zero_count") or 0)
            total = int(row.get("total") or 0)
            if total <= 0:
                raise ValueError(
                    f"Propagation row activations/{stage.name}/layer_{layer} has no denominator."
                )
            values.append(100.0 * zero_count / total)
            pooled_zero += zero_count
            pooled_total += total
        values.append(100.0 * pooled_zero / pooled_total)
        matrix.append(values)
    return matrix


def _plot_report05_propagation_heatmaps(
    payload: dict[str, Any],
    architecture_id: str,
) -> Figure:
    """Render one compact 1x3 exact-zero propagation figure."""

    spec = _architecture_spec(architecture_id)
    methods = _select_architecture_methods(payload, architecture_id)
    num_layers = int(methods[0].get("num_layers") or 0)
    if num_layers != 6 or any(int(method.get("num_layers") or 0) != num_layers for method in methods):
        raise ValueError("Report 05 propagation heatmaps require six consistent Pythia-14M layers.")

    matrices = [
        _propagation_matrix(method, spec.stages, num_layers=num_layers)
        for method in methods
    ]
    height = min(8.25, 2.25 + 0.255 * len(spec.stages))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, height),
        sharey=False,
    )
    column_labels = [f"L{layer}" for layer in range(num_layers)] + ["All"]
    image = None
    for method_index, (ax, matrix, run_spec) in enumerate(
        zip(axes, matrices, spec.runs, strict=True)
    ):
        method_name = run_spec[0]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
        ax.set_title(f"({chr(ord('a') + method_index)}) {method_name}", fontsize=8.2, pad=4)
        _format_compact_heatmap(
            ax,
            image,
            matrix,
            stages=spec.stages,
            column_labels=column_labels,
            show_row_labels=method_index == 0,
            separator_after=spec.separator_after,
        )

    if image is None:
        raise ValueError("Report 05 propagation payload has no plottable methods.")
    colorbar_axis = fig.add_axes([0.928, 0.205, 0.014, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Exact zeros (%)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.5, length=2)

    validation_tokens = int(payload.get("validation_tokens") or 0)
    fig.suptitle(
        f"Where Exact Zeros Persist Through Pythia-14M Blocks: {spec.title}",
        y=0.985,
        fontsize=10.5,
    )
    fig.text(
        0.59,
        0.947,
        f"Direct counts over {validation_tokens:,} validation tokens; All = count-weighted pool over L0-L5",
        ha="center",
        va="top",
        fontsize=7.0,
    )
    fig.text(
        0.59,
        0.040,
        "Exact zero: computed value == 0 (no tolerance). Each cell is zero values / produced values.",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.265, right=0.905, top=0.895, bottom=0.115, wspace=0.10)
    return fig


def _format_compact_heatmap(
    ax: Axes,
    image: Any,
    matrix: list[list[float]],
    *,
    stages: tuple[_StageSpec, ...],
    column_labels: list[str],
    show_row_labels: bool,
    separator_after: tuple[int, ...],
) -> None:
    ax.grid(False)
    ax.set_xticks(range(len(column_labels)), column_labels, fontsize=6.4)
    ax.set_yticks(range(len(stages)))
    if show_row_labels:
        ax.set_yticklabels([stage.label for stage in stages], fontsize=6.8)
        for text, stage in zip(ax.get_yticklabels(), stages, strict=True):
            if stage.emphasized:
                text.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([index + 0.5 for index in range(len(column_labels) - 1)], minor=True)
    ax.set_yticks([index + 0.5 for index in range(len(stages) - 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.38, alpha=0.40)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(len(column_labels) - 1.5, color="white", linewidth=1.25, alpha=0.95)
    for row_index in separator_after:
        ax.axhline(row_index + 0.5, color="white", linewidth=1.15, alpha=0.95)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            red, green, blue, _alpha = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                column_index,
                row_index,
                _compact_percent_label(value),
                ha="center",
                va="center",
                fontsize=5.4,
                color="black" if luminance >= 0.56 else "white",
            )


def _compact_percent_label(percent: float) -> str:
    if percent == 0.0:
        return "0"
    if percent < 0.1:
        return "<.1"
    if percent >= 99.95:
        return "100"
    return f"{percent:.1f}"


def _histogram_payload_for_site(
    payloads: dict[str, dict[str, Any]],
    site: str,
) -> dict[str, Any]:
    matches = [payload for payload in payloads.values() if site in set(payload.get("sites", []))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one activation-histogram payload for {site!r}; found {len(matches)}.")
    return matches[0]


def _histogram_method_by_config(
    payload: dict[str, Any],
    config_id: str,
) -> dict[str, Any]:
    matches = [item for item in payload.get("methods", []) if str(item.get("config_id")) == config_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one histogram method for {config_id!r}; found {len(matches)}.")
    return matches[0]


def _pooled_activation_distribution(
    payloads: dict[str, dict[str, Any]],
    *,
    site: str,
    config_id: str,
) -> dict[str, Any]:
    """Pool six layer histograms before deriving the conditional density."""

    payload = _histogram_payload_for_site(payloads, site)
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if len(edges) < 2:
        raise ValueError(f"Histogram payload for {site!r} has no bin edges.")
    method = _histogram_method_by_config(payload, config_id)
    by_name = {str(layer.get("name")): layer for layer in method.get("layers", [])}
    layers = []
    for layer_index in range(6):
        name = f"{site}.layer_{layer_index}"
        if name not in by_name:
            raise ValueError(f"Missing histogram layer {name!r} for {config_id!r}.")
        layers.append(by_name[name])

    bin_count = len(edges) - 1
    counts = [0] * bin_count
    total = 0
    exact_zero_count = 0
    outside = 0
    for layer in layers:
        layer_counts = [int(value) for value in layer.get("counts", [])]
        if len(layer_counts) != bin_count:
            raise ValueError(f"Histogram layer for {site!r} has {len(layer_counts)} bins; expected {bin_count}.")
        counts = [left + right for left, right in zip(counts, layer_counts, strict=True)]
        total += int(layer.get("total") or 0)
        exact_zero_count += int((layer.get("threshold_hits") or {}).get("0") or 0)
        outside += int(layer.get("underflow") or 0) + int(layer.get("overflow") or 0)
    pooled = {
        "counts": counts,
        "total": total,
        "threshold_hits": {"0": exact_zero_count},
    }
    densities, _exact_zero_fraction = _histogram_nonzero_density(pooled, edges)
    return {
        "site": site,
        "centers": [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)],
        "densities": densities,
        "range": (edges[0], edges[-1]),
        "total": total,
        "outside_fraction": outside / total if total else 0.0,
        "validation_tokens": int(payload.get("validation_tokens") or 0),
    }


def _weight_series_for_run(
    weight_series: list[dict[str, Any]],
    *,
    architecture_id: str,
    method: str,
    display_label: str,
    config_id: str,
) -> dict[str, Any]:
    by_config = [item for item in weight_series if str(item.get("config_id")) == config_id]
    if len(by_config) == 1:
        return by_config[0]
    if len(by_config) > 1:
        raise ValueError(f"Duplicate weight series for {config_id!r}.")

    aliases = {display_label}
    if architecture_id == "one_relu":
        aliases.add(f"MLP-ReLU {method}")
    by_label = [item for item in weight_series if str(item.get("label")) in aliases]
    if len(by_label) != 1:
        raise ValueError(
            f"Expected exactly one weight series for {display_label} ({config_id}); found {len(by_label)}."
        )
    return by_label[0]


def _plot_report05_architecture_distributions(
    histogram_payloads: dict[str, dict[str, Any]],
    weight_series: list[dict[str, Any]],
    architecture_id: str,
) -> Figure:
    """Render pooled activation and downstream learned-weight densities."""

    spec = _architecture_spec(architecture_id)
    activation_specs: list[tuple[str, str]] = [
        ("attention_inputs", "Attention input -> QKV"),
        ("mlp_inputs", "MLP input -> W1"),
        ("mlp_hiddens", "MLP hidden -> W2"),
    ]
    if spec.has_post_qkv_gates:
        q_suffix = "RoPE" if architecture_id == "six_relu_pre" else "QK"
        activation_specs.extend(
            [
                ("query_gate_outputs", f"Q gate -> {q_suffix}"),
                ("key_gate_outputs", f"K gate -> {q_suffix}"),
                ("value_gate_outputs", "V gate -> PV"),
            ]
        )

    rows = 3 if spec.has_post_qkv_gates else 2
    height = 7.15 if rows == 3 else 5.25
    fig, axes = plt.subplots(rows, 3, figsize=(DOUBLE_COLUMN_WIDTH_INCHES, height), squeeze=False)
    legend_handles: dict[str, Any] = {}
    validation_tokens: set[int] = set()
    maximum_outside = 0.0

    for panel_index, (site, title) in enumerate(activation_specs):
        ax = axes.flat[panel_index]
        distributions: list[tuple[str, dict[str, Any]]] = []
        for method, _display_label, config_id in spec.runs:
            distribution = _pooled_activation_distribution(
                histogram_payloads,
                site=site,
                config_id=config_id,
            )
            distributions.append((method, distribution))
            validation_tokens.add(int(distribution["validation_tokens"]))
            maximum_outside = max(maximum_outside, float(distribution["outside_fraction"]))
        _plot_density_panel(
            ax,
            distributions,
            title=title,
            is_weight=False,
            legend_handles=legend_handles,
            symlog_x=site in {"query_gate_outputs", "key_gate_outputs", "value_gate_outputs"},
        )
        if panel_index % 3 == 0:
            ax.set_ylabel("Density given x != 0 (log)", fontsize=8.0)

    weight_row = rows - 1
    weight_specs = (
        ("qkv", "QKV weights"),
        ("w1", "W1 weights"),
        ("w2", "W2 weights"),
    )
    for column, (group_id, title) in enumerate(weight_specs):
        ax = axes[weight_row, column]
        distributions = []
        for method, display_label, config_id in spec.runs:
            item = _weight_series_for_run(
                weight_series,
                architecture_id=architecture_id,
                method=method,
                display_label=display_label,
                config_id=config_id,
            )
            group = (item.get("weight_groups") or {}).get(group_id)
            if not isinstance(group, dict):
                raise ValueError(f"Missing weight group {group_id!r} for {display_label}.")
            distributions.append(
                (
                    method,
                    {
                        "centers": [float(value) for value in group.get("centers", [])],
                        "densities": [float(value) for value in group.get("densities", [])],
                        "range": tuple(float(value) for value in group.get("range", ())),
                    },
                )
            )
        _plot_density_panel(ax, distributions, title=title, is_weight=True, legend_handles=legend_handles)
        if column == 0:
            ax.set_ylabel("Weight density (log)", fontsize=8.0)

    if len(validation_tokens) != 1:
        raise ValueError(f"Activation histogram payloads disagree on validation tokens: {sorted(validation_tokens)}.")
    token_count = next(iter(validation_tokens))
    fig.suptitle(
        f"Activation and Downstream Weight Distributions: {spec.title}",
        y=0.985,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.952,
        (
            f"Activations pool six layers and {token_count:,} validation tokens; "
            "weights pool final-checkpoint tensors over six layers"
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8.0,
    )
    footer = (
        "Activation curves condition on x != 0; exact-zero atoms are intentionally omitted.\n"
        f"Maximum stored under/overflow mass: {100.0 * maximum_outside:.3f}%."
    )
    if spec.has_post_qkv_gates:
        footer += "\nQ/K/V gates feed RoPE, QK, or PV; these consumers have no learned weight tensor."
    fig.text(0.5, 0.062, footer, ha="center", va="bottom", fontsize=8.0)
    fig.subplots_adjust(
        left=0.125,
        right=0.985,
        top=0.895,
        bottom=0.205 if rows == 2 else 0.165,
        hspace=0.42,
        wspace=0.30,
    )
    return fig


def _plot_density_panel(
    ax: Axes,
    distributions: list[tuple[str, dict[str, Any]]],
    *,
    title: str,
    is_weight: bool,
    legend_handles: dict[str, Any],
    symlog_x: bool = False,
) -> None:
    positive = [
        float(value)
        for _method, distribution in distributions
        for value in distribution["densities"]
        if float(value) > 0.0 and math.isfinite(float(value))
    ]
    if not positive:
        raise ValueError(f"Density panel {title!r} contains no positive finite density.")
    y_min = max(min(positive) * 0.65, max(positive) * 1e-6, 1e-9)
    y_max = max(positive) * 1.45
    reference_range: tuple[float, float] | None = None
    for method, distribution in distributions:
        centers = [float(value) for value in distribution["centers"]]
        densities = [
            float(value) if float(value) > 0.0 and math.isfinite(float(value)) else math.nan
            for value in distribution["densities"]
        ]
        if len(centers) != len(densities):
            raise ValueError(f"Density panel {title!r} has mismatched centers and densities.")
        limits = tuple(float(value) for value in distribution["range"])
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise ValueError(f"Density panel {title!r} has invalid x range {limits!r}.")
        if reference_range is None:
            reference_range = (limits[0], limits[1])
        elif (limits[0], limits[1]) != reference_range:
            raise ValueError(f"Density panel {title!r} mixes incompatible x ranges.")
        style = _METHOD_STYLES[method]
        (line,) = ax.step(
            centers,
            densities,
            where="mid",
            color=style.color,
            linestyle=style.linestyle,
            linewidth=max(style.linewidth, 1.15),
            label=method,
        )
        legend_handles.setdefault(method, line)
    assert reference_range is not None
    ax.axvline(0.0, color="#555555", linewidth=0.65, alpha=0.65)
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(*reference_range)
    if symlog_x:
        ax.set_xscale("symlog", linthresh=0.5, linscale=0.7)
        if reference_range[1] >= 100.0:
            ax.set_xticks((0.0, 1.0, 10.0, 100.0))
    ax.set_title(title, fontsize=8.2, pad=3)
    if is_weight:
        x_label = "Weight value"
    elif symlog_x:
        x_label = "Activation value (symlog x)"
    else:
        x_label = "Activation value"
    ax.set_xlabel(x_label, fontsize=8.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
    ax.tick_params(axis="both", labelsize=8.0)
    ax.grid(alpha=0.20)


def _plot_report05_one_relu_propagation_heatmaps(payload: dict[str, Any]) -> Figure:
    return _plot_report05_propagation_heatmaps(payload, "one_relu")


def _plot_report05_three_relu_propagation_heatmaps(payload: dict[str, Any]) -> Figure:
    return _plot_report05_propagation_heatmaps(payload, "three_relu")


def _plot_report05_six_relu_pre_propagation_heatmaps(payload: dict[str, Any]) -> Figure:
    return _plot_report05_propagation_heatmaps(payload, "six_relu_pre")


def _plot_report05_six_relu_post_propagation_heatmaps(payload: dict[str, Any]) -> Figure:
    return _plot_report05_propagation_heatmaps(payload, "six_relu_post")


def _plot_report05_one_relu_distributions(
    histogram_payloads: dict[str, dict[str, Any]],
    weight_series: list[dict[str, Any]],
) -> Figure:
    return _plot_report05_architecture_distributions(histogram_payloads, weight_series, "one_relu")


def _plot_report05_three_relu_distributions(
    histogram_payloads: dict[str, dict[str, Any]],
    weight_series: list[dict[str, Any]],
) -> Figure:
    return _plot_report05_architecture_distributions(histogram_payloads, weight_series, "three_relu")


def _plot_report05_six_relu_pre_distributions(
    histogram_payloads: dict[str, dict[str, Any]],
    weight_series: list[dict[str, Any]],
) -> Figure:
    return _plot_report05_architecture_distributions(histogram_payloads, weight_series, "six_relu_pre")


def _plot_report05_six_relu_post_distributions(
    histogram_payloads: dict[str, dict[str, Any]],
    weight_series: list[dict[str, Any]],
) -> Figure:
    return _plot_report05_architecture_distributions(histogram_payloads, weight_series, "six_relu_post")
