"""Renderer for saved training and calibration diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .export import DOUBLE_COLUMN_WIDTH_INCHES
from .histograms import finite_number
from .style import series_style


def build_run_diagnostics(payload: dict[str, Any]) -> Figure:
    """Build a model-agnostic run diagnostic from events and terminal metrics."""

    train_events = payload["train_events"]
    validation_events = payload["validation_events"]
    metrics = payload["metrics"]
    manifest = payload["manifest"]

    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 6.3))
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    loss_axis = figure.add_subplot(grid[0, :])
    gradient_axis = figure.add_subplot(grid[1, 0])
    stats_axis = figure.add_subplot(grid[1, 1])

    loss_axis.plot(
        [row["tokens_seen"] for row in train_events],
        [row["train_loss"] for row in train_events],
        label="train",
        marker="o",
        markersize=2.5,
        linewidth=1.2,
    )
    if validation_events:
        loss_axis.plot(
            [row["tokens_seen"] for row in validation_events],
            [row["validation_loss"] for row in validation_events],
            label="validation",
            marker="s",
            markersize=3.0,
            linewidth=1.2,
        )
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Tokens seen")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend(frameon=False)

    plotted_norm = False
    for key, label, style_index in (
        ("grad_norm", "gradient", 0),
        ("weight_norm", "weight", 1),
    ):
        rows = [row for row in train_events if finite_number(row.get(key))]
        if not rows:
            continue
        style = series_style(style_index)
        gradient_axis.plot(
            [row["tokens_seen"] for row in rows],
            [row[key] for row in rows],
            label=label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markersize=2.5,
            linewidth=style.linewidth,
        )
        plotted_norm = True
    gradient_axis.set_title("Optimization norms")
    gradient_axis.set_xlabel("Tokens seen")
    gradient_axis.set_ylabel("L2 norm")
    if plotted_norm:
        gradient_axis.legend(frameon=False)
    else:
        gradient_axis.text(
            0.5,
            0.5,
            "No gradient or weight norms logged",
            transform=gradient_axis.transAxes,
            ha="center",
            va="center",
        )

    stats_axis.axis("off")
    stats_axis.set_title("Recorded run statistics")
    metric_prefix = "training" if manifest.get("mode") == "pretrain" else "calibration"
    statistic_keys = (
        (f"{metric_prefix}/tokens_per_second", "Average tokens/s"),
        (f"{metric_prefix}/peak_gpu_memory_mb", "Peak GPU allocated (MB)"),
        (f"{metric_prefix}/peak_gpu_reserved_mb", "Peak GPU reserved (MB)"),
        (f"{metric_prefix}/wall_seconds_train", "Train wall time (s)"),
        (f"{metric_prefix}/wall_seconds_total", "Total wall time (s)"),
        ("checkpoint/final_size_mb", "Final checkpoint (MB)"),
    )
    statistic_lines = [
        f"{label}: {_format_number(metrics[key])}"
        for key, label in statistic_keys
        if finite_number(metrics.get(key))
    ]
    statistic_lines.extend(
        (
            f"Train points: {len(train_events)}",
            f"Validation points: {len(validation_events)}",
        )
    )
    stats_axis.text(
        0.02,
        0.94,
        "\n".join(statistic_lines),
        transform=stats_axis.transAxes,
        ha="left",
        va="top",
        linespacing=1.5,
    )

    identity = " / ".join(
        str(value)
        for value in (manifest.get("config_id"), manifest.get("run_id"))
        if value
    )
    figure.suptitle(f"Run diagnostics{f' - {identity}' if identity else ''}", y=0.995)
    return figure


def _format_number(value: Any) -> str:
    numeric = float(value)
    if abs(numeric) >= 1000.0:
        return f"{numeric:,.0f}"
    return f"{numeric:.4g}"
