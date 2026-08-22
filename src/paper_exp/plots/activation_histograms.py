"""Renderer for saved activation-histogram diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .export import DOUBLE_COLUMN_WIDTH_INCHES
from .histograms import bin_centers, pooled_histogram, validated_histogram_payload
from .style import series_style


def build_activation_histograms(payload: dict[str, Any]) -> Figure:
    """Build pooled conditional densities with a separate exact-zero atom."""

    edges, methods = validated_histogram_payload(payload)
    centers = bin_centers(edges)
    reductions = [
        pooled_histogram(method, edges, separate_zero=True)
        for method in methods
    ]

    figure, (atom_axis, density_axis) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.1),
        gridspec_kw={"width_ratios": (1.0, 2.4)},
    )
    labels = [
        str(method.get("label") or method.get("config_id") or "unnamed")
        for method in methods
    ]
    atom_values = [item["zero_fraction"] for item in reductions]
    atom_axis.bar(
        range(len(labels)),
        [100.0 * value for value in atom_values],
        color=[series_style(index).color for index in range(len(labels))],
    )
    atom_axis.set_title("Exact-zero atom")
    atom_axis.set_ylabel("Exact zeros (%)")
    atom_axis.set_xticks(range(len(labels)))
    atom_axis.set_xticklabels(labels, rotation=35, ha="right")
    atom_axis.set_ylim(bottom=0.0)

    for index, (label, reduction) in enumerate(zip(labels, reductions, strict=True)):
        style = series_style(index)
        density_axis.plot(
            centers,
            reduction["density"],
            label=label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, len(centers) // 10),
            markersize=3.0,
            linewidth=style.linewidth,
        )
    density_axis.set_title("Conditional density given nonzero")
    density_axis.set_xlabel("Activation value")
    density_axis.set_ylabel("Density")
    if any(value > 0.0 for item in reductions for value in item["density"]):
        density_axis.set_yscale("log")
    density_axis.legend(frameon=False)
    figure.suptitle(str(payload.get("plot_title") or "Activation distributions"))
    figure.text(
        0.5,
        0.01,
        (
            f"n={len(methods)} checkpoints; integer counts pooled across layers before "
            "normalization; out-of-range mass remains in the denominator"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return figure
