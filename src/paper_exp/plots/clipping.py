"""Renderer for saved post-hoc activation clipping frontiers."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .export import DOUBLE_COLUMN_WIDTH_INCHES
from .histograms import finite_number
from .style import series_style


def build_clipping_frontier(rows: list[dict[str, Any]]) -> Figure:
    """Build an exact-sparsity versus validation-loss clipping frontier."""

    if not rows:
        raise ValueError("Clipping artifact contains no frontier points.")
    validated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Clipping row {index} must be a JSON object.")
        achieved = row.get("achieved_sparsity")
        loss = row.get("validation_loss")
        if not finite_number(achieved) or not 0.0 <= float(achieved) <= 1.0:
            raise ValueError(
                f"Clipping row {index} achieved_sparsity must be finite and in [0, 1]."
            )
        if not finite_number(loss):
            raise ValueError(f"Clipping row {index} validation_loss must be finite.")
        validation_tokens = row.get("validation_tokens")
        if validation_tokens is not None and (
            isinstance(validation_tokens, bool)
            or not isinstance(validation_tokens, int)
            or validation_tokens <= 0
        ):
            raise ValueError(
                f"Clipping row {index} validation_tokens must be a positive integer."
            )
        validated_rows.append(row)
    validated_rows.sort(key=lambda row: float(row["achieved_sparsity"]))

    sparsity = [100.0 * float(row["achieved_sparsity"]) for row in validated_rows]
    losses = [float(row["validation_loss"]) for row in validated_rows]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.4))
    style = series_style(0)
    axis.plot(
        sparsity,
        losses,
        color=style.color,
        marker=style.marker,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        markersize=4.5,
    )
    axis.set_title("Post-hoc activation clipping frontier")
    axis.set_xlabel("Achieved exact-zero activation sparsity (%)")
    axis.set_ylabel("Validation loss")
    axis.set_xlim(left=0.0)

    for index, (x_value, y_value, row) in enumerate(
        zip(sparsity, losses, validated_rows, strict=True)
    ):
        axis.annotate(
            _clipping_label(row),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 6 if index % 2 == 0 else -12),
            fontsize=8,
        )

    span = max(losses) - min(losses)
    margin = max(0.15 * span, 1e-4)
    if min(losses) > 0.0:
        axis.set_ylim(min(losses) - margin, max(losses) + margin)
        scale_note = "validation-loss axis is zoomed"
    else:
        axis.set_ylim(bottom=0.0)
        scale_note = "validation-loss axis starts at zero"
    token_counts = {
        row["validation_tokens"]
        for row in validated_rows
        if row.get("validation_tokens") is not None
    }
    token_note = (
        f"; {next(iter(token_counts)):,} validation tokens/point"
        if len(token_counts) == 1
        else ""
    )
    axis.text(
        0.99,
        0.02,
        f"n={len(validated_rows)} sweep points{token_note}; {scale_note}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def _clipping_label(row: dict[str, Any]) -> str:
    if finite_number(row.get("threshold")):
        return f"t={float(row['threshold']):g}"
    if finite_number(row.get("quantile")):
        return f"q={float(row['quantile']):g}"
    if finite_number(row.get("rms_multiplier")):
        return f"r={float(row['rms_multiplier']):g}"
    return str(row.get("mode") or "clip")
