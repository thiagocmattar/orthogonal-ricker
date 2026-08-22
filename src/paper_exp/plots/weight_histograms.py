"""Renderer for saved weight-histogram diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .export import DOUBLE_COLUMN_WIDTH_INCHES
from .histograms import bin_centers, pooled_histogram, validated_histogram_payload
from .style import series_style


def build_weight_histograms(payload: dict[str, Any]) -> Figure:
    """Build pooled weight densities from one saved diagnostic artifact."""

    edges, methods = validated_histogram_payload(payload)
    centers = bin_centers(edges)
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.2))
    for index, method in enumerate(methods):
        reduction = pooled_histogram(method, edges, separate_zero=False)
        label = str(method.get("label") or method.get("config_id") or "unnamed")
        style = series_style(index)
        axis.plot(
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
    axis.set_title(str(payload.get("plot_title") or "Weight distributions"))
    axis.set_xlabel("Weight value")
    axis.set_ylabel("Density over stored total")
    has_in_range_mass = any(
        value > 0.0
        for line in axis.lines
        for value in line.get_ydata()
    )
    if has_in_range_mass:
        axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No in-range weight mass was recorded",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.02,
        (
            f"n={len(methods)} checkpoints; integer counts pooled across layers; "
            "out-of-range mass remains in the denominator"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    return figure
