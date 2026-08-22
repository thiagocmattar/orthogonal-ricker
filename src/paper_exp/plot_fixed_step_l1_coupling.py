"""Plot how FFN-only L1 pressure redistributes activation near-zero mass."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_style import COLORBLIND_SAFE_COLORS, REPORT04_PLOT_STYLE


DEFAULT_HISTOGRAMS = Path(
    "results/304-pythia-14m-fixed-2048-ffn-l1-coupling-histograms/"
    "001-20260822-124622-92e26051/activation_histograms.json"
)
DEFAULT_OUTPUT = Path(
    "figures/119-pythia-14m-fixed-2048-mlp-l1-near-zero-coupling.pdf"
)
BASELINE_CONFIG_ID = "12-pythia-14m-minipile-adamw-fixed-2048"
THRESHOLD = 0.01
COHORT = (
    ("l1_naive", 0.05, "27-pythia-14m-minipile-l1-naive-fixed-2048-w0p05"),
    ("l1_naive", 0.15, "28-pythia-14m-minipile-l1-naive-fixed-2048-w0p15"),
    ("l1_naive", 0.5, "29-pythia-14m-minipile-l1-naive-fixed-2048-w0p5"),
    ("l1_naive", 1.0, "30-pythia-14m-minipile-l1-naive-fixed-2048-w1"),
    ("l1_naive", 2.0, "45-pythia-14m-minipile-l1-naive-fixed-2048-w2"),
    ("l1_naive", 5.0, "46-pythia-14m-minipile-l1-naive-fixed-2048-w5"),
    ("orthogonal_l1", 0.05, "31-pythia-14m-minipile-orthogonal-l1-fixed-2048-w0p05"),
    ("orthogonal_l1", 0.15, "32-pythia-14m-minipile-orthogonal-l1-fixed-2048-w0p15"),
    ("orthogonal_l1", 0.5, "33-pythia-14m-minipile-orthogonal-l1-fixed-2048-w0p5"),
    ("orthogonal_l1", 1.0, "34-pythia-14m-minipile-orthogonal-l1-fixed-2048-w1"),
    ("orthogonal_l1", 2.0, "47-pythia-14m-minipile-orthogonal-l1-fixed-2048-w2"),
    ("orthogonal_l1", 5.0, "48-pythia-14m-minipile-orthogonal-l1-fixed-2048-w5"),
)
METHOD_STYLES = {
    "l1_naive": ("L1N", COLORBLIND_SAFE_COLORS[1], "o", "-"),
    "orthogonal_l1": ("OL1", COLORBLIND_SAFE_COLORS[0], "s", "--"),
}
FIGURE_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=4.6,
    min_text_points=8.0,
)


@dataclass(frozen=True)
class CouplingPoint:
    """One pressure run expressed relative to the matched AdamW checkpoint."""

    method: str
    weight: float
    config_id: str
    label: str
    mlp_fraction: float
    attention_fraction: float
    delta_mlp_pp: float
    delta_attention_pp: float


@dataclass(frozen=True)
class CouplingSummary:
    """Pooled checkpoint measurements needed by the coupling figure."""

    threshold: float
    validation_tokens: int
    validation_sequences: int
    baseline_mlp_fraction: float
    baseline_attention_fraction: float
    points: tuple[CouplingPoint, ...]


def load_coupling_summary(
    path: str | Path,
    *,
    threshold: float = THRESHOLD,
) -> CouplingSummary:
    """Load and reduce the pinned schema-v2 activation diagnostic."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError("The coupling figure requires activation histogram schema version 2.")
    threshold_key = f"{threshold:g}"
    configured_thresholds = {f"{float(value):g}" for value in payload.get("thresholds", [])}
    if threshold_key not in configured_thresholds:
        raise ValueError(f"Activation diagnostic does not contain threshold {threshold_key}.")

    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Activation diagnostic has no method list.")
    by_config_id = {str(method.get("config_id")): method for method in methods}
    expected_ids = {BASELINE_CONFIG_ID, *(config_id for _, _, config_id in COHORT)}
    if set(by_config_id) != expected_ids:
        missing = sorted(expected_ids.difference(by_config_id))
        unexpected = sorted(set(by_config_id).difference(expected_ids))
        raise ValueError(
            "Unexpected FFN-only L1 coupling cohort: "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}."
        )

    baseline = by_config_id[BASELINE_CONFIG_ID]
    baseline_mlp, baseline_mlp_total = _pooled_threshold_fraction(
        baseline,
        site="mlp_hiddens",
        threshold_key=threshold_key,
    )
    baseline_attention, baseline_attention_total = _pooled_threshold_fraction(
        baseline,
        site="attention_outputs",
        threshold_key=threshold_key,
    )

    points = []
    for method, weight, config_id in COHORT:
        row = by_config_id[config_id]
        mlp_fraction, mlp_total = _pooled_threshold_fraction(
            row,
            site="mlp_hiddens",
            threshold_key=threshold_key,
        )
        attention_fraction, attention_total = _pooled_threshold_fraction(
            row,
            site="attention_outputs",
            threshold_key=threshold_key,
        )
        if mlp_total != baseline_mlp_total or attention_total != baseline_attention_total:
            raise ValueError(f"Mismatched activation denominator for {config_id}.")
        points.append(
            CouplingPoint(
                method=method,
                weight=weight,
                config_id=config_id,
                label=str(row.get("label") or config_id),
                mlp_fraction=mlp_fraction,
                attention_fraction=attention_fraction,
                delta_mlp_pp=100.0 * (mlp_fraction - baseline_mlp),
                delta_attention_pp=100.0 * (attention_fraction - baseline_attention),
            )
        )

    return CouplingSummary(
        threshold=threshold,
        validation_tokens=int(payload["validation_tokens"]),
        validation_sequences=int(payload["validation_sequences"]),
        baseline_mlp_fraction=baseline_mlp,
        baseline_attention_fraction=baseline_attention,
        points=tuple(points),
    )


def _pooled_threshold_fraction(
    method: dict[str, Any],
    *,
    site: str,
    threshold_key: str,
) -> tuple[float, int]:
    layers = [
        layer
        for layer in method.get("layers", [])
        if str(layer.get("name", "")).startswith(f"{site}.layer_")
    ]
    if not layers:
        raise ValueError(f"{method.get('label', 'method')} has no {site} measurements.")

    hits = 0
    total = 0
    for layer in layers:
        threshold_hits = layer.get("threshold_hits")
        if not isinstance(threshold_hits, dict) or threshold_key not in threshold_hits:
            raise ValueError(
                f"{method.get('label', 'method')} {layer.get('name')} has no "
                f"threshold {threshold_key}."
            )
        layer_hits = int(threshold_hits[threshold_key])
        layer_total = int(layer["total"])
        if layer_hits < 0 or layer_total <= 0 or layer_hits > layer_total:
            raise ValueError(f"Invalid threshold counts for {layer.get('name')}.")
        hits += layer_hits
        total += layer_total
    return hits / total, total


def pearson_correlation(points: Sequence[CouplingPoint]) -> float:
    """Return the descriptive Pearson correlation for one method path."""

    if len(points) < 2:
        raise ValueError("At least two points are required for a correlation.")
    xs = [point.delta_mlp_pp for point in points]
    ys = [point.delta_attention_pp for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        raise ValueError("Correlation is undefined for a constant coordinate.")
    return numerator / (x_scale * y_scale)


def build_figure(summary: CouplingSummary) -> Figure:
    """Build the single-panel FFN-only L1 coupling scatter plot."""

    figure, axis = plt.subplots(
        figsize=(FIGURE_PROFILE.width_inches, FIGURE_PROFILE.max_height_inches),
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.24)

    all_x = [0.0]
    all_y = [0.0]
    for method in METHOD_STYLES:
        method_points = sorted(
            (point for point in summary.points if point.method == method),
            key=lambda point: point.weight,
        )
        if len(method_points) != 6:
            raise ValueError(f"Expected six {method} pressure weights.")
        label, color, marker, linestyle = METHOD_STYLES[method]
        xs = [point.delta_mlp_pp for point in method_points]
        ys = [point.delta_attention_pp for point in method_points]
        all_x.extend(xs)
        all_y.extend(ys)
        correlation = pearson_correlation(method_points)
        axis.plot(
            xs,
            ys,
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            alpha=0.88,
            zorder=2,
        )
        axis.scatter(
            xs,
            ys,
            color=color,
            edgecolors=color,
            marker=marker,
            s=38,
            linewidths=0.9,
            label=f"{label} (Pearson r={correlation:.2f}, n=6)",
            zorder=3,
        )
        for index, point in enumerate(method_points):
            dx, dy = _annotation_offset(method, index)
            axis.annotate(
                rf"$\lambda={point.weight:g}$",
                (point.delta_mlp_pp, point.delta_attention_pp),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.0,
                ha="right" if dx < 0 else "left",
                va="top" if dy < 0 else "bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
                zorder=4,
            )

    axis.scatter(
        [0.0],
        [0.0],
        color="black",
        marker="D",
        s=34,
        linewidths=0.9,
        zorder=5,
    )
    axis.annotate(
        "AdamW reference",
        (0.0, 0.0),
        xytext=(6, -8),
        textcoords="offset points",
        fontsize=8.0,
        ha="left",
        va="top",
        zorder=5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55, zorder=1)
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.55, zorder=1)

    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    x_padding = max(0.06 * x_span, 1.5)
    y_padding = max(0.08 * y_span, 1.0)
    axis.set_xlim(min(all_x) - x_padding, max(all_x) + x_padding)
    axis.set_ylim(min(all_y) - y_padding, max(all_y) + y_padding)
    axis.set_xlabel("Increase in MLP-hidden near-zero mass (pp vs AdamW)")
    axis.set_ylabel("Change in attention-output near-zero mass (pp vs AdamW)")
    axis.set_title(
        "FFN-only L1 pressure: targeted versus untargeted near-zero mass",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    axis.legend(loc="upper right", frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)

    figure.text(
        0.13,
        0.91,
        "Pythia-14M | 2,048 steps | seed 0 | pressure-loss site: MLP hiddens",
        fontsize=8.5,
        ha="left",
        va="center",
    )
    figure.text(
        0.13,
        0.055,
        (
            f"Near-zero: |a| <= 0.01; direct pooled counts over six layers and "
            f"{summary.validation_tokens:,} validation tokens per checkpoint.\n"
            "Near-zero mass is not exact sparsity; one seed, no uncertainty estimate."
        ),
        fontsize=8.0,
        ha="left",
        va="bottom",
    )
    return figure


def _annotation_offset(method: str, index: int) -> tuple[float, float]:
    if method == "l1_naive":
        offsets = ((-4, 8), (4, 8), (4, 8), (4, -9), (4, -9), (-4, -9))
    else:
        offsets = ((-4, -9), (4, -9), (4, -9), (4, 8), (4, 8), (-4, 8))
    return offsets[index]


def generate_figure(
    histograms: str | Path = DEFAULT_HISTOGRAMS,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    save_png: bool = False,
) -> list[Path]:
    """Generate the reproducible PDF and optional PNG from saved counts."""

    summary = load_coupling_summary(histograms)
    return export_figure(
        lambda: build_figure(summary),
        output,
        save_png=save_png,
        style=REPORT04_PLOT_STYLE,
        profile=FIGURE_PROFILE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot MLP/attention near-zero coupling for the fixed-step FFN-only L1 sweep."
    )
    parser.add_argument("--histograms", default=str(DEFAULT_HISTOGRAMS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args(argv)

    summary = load_coupling_summary(args.histograms)
    outputs = export_figure(
        lambda: build_figure(summary),
        args.output,
        save_png=args.png,
        style=REPORT04_PLOT_STYLE,
        profile=FIGURE_PROFILE,
    )
    for path in outputs:
        print(path)
    for method, (label, _color, _marker, _linestyle) in METHOD_STYLES.items():
        selected = [point for point in summary.points if point.method == method]
        print(f"{label} Pearson r={pearson_correlation(selected):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
