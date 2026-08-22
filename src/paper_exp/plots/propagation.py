"""Renderer and reductions for saved activation-propagation diagnostics."""

from __future__ import annotations

import math
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .export import DOUBLE_COLUMN_WIDTH_INCHES


def build_activation_propagation(payload: dict[str, Any]) -> Figure:
    """Build pooled activation and matmul exact-zero heatmaps."""

    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Activation propagation payload has no methods.")
    groups = (
        (
            "activations",
            payload.get("activation_stage_order"),
            payload.get("activation_stage_labels"),
            "Activation exact zeros (%)",
        ),
        (
            "matmuls",
            payload.get("matmul_stage_order"),
            payload.get("matmul_stage_labels"),
            "Logical zero\nproducts (%)",
        ),
    )
    if any(
        not isinstance(order, list)
        or not order
        or not isinstance(labels, dict)
        for _key, order, labels, _title in groups
    ):
        raise ValueError("Activation propagation payload is missing an explicit stage order.")

    method_labels = [
        str(method.get("label") or method.get("config_id") or "unnamed")
        for method in methods
    ]
    longest_order = max(len(order) for _key, order, _labels, _title in groups)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            DOUBLE_COLUMN_WIDTH_INCHES,
            min(8.6, max(5.0, 1.8 + 0.32 * longest_order)),
        ),
        gridspec_kw={"width_ratios": (2.2, 1.0)},
    )
    for axis, (row_key, order, labels, title) in zip(axes, groups, strict=True):
        matrix = np.asarray(
            [_pooled_fraction_row(method, row_key, order) for method in methods],
            dtype=float,
        ).T
        percent_matrix = np.ma.masked_invalid(100.0 * matrix)
        colormap = plt.get_cmap("cividis").with_extremes(bad="#D9D9D9")
        image = axis.imshow(
            percent_matrix,
            vmin=0.0,
            vmax=100.0,
            cmap=colormap,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(method_labels)))
        axis.set_yticks(range(len(order)))
        axis.set_xticklabels(
            method_labels,
            rotation=35,
            ha="right",
        )
        axis.set_yticklabels(
            [
                textwrap.fill(str(labels.get(str(stage), stage)), width=28)
                for stage in order
            ]
        )
        axis.grid(False)
        if matrix.size <= 80:
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if math.isfinite(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{100.0 * value:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value < 0.55 else "black",
                        )
                    else:
                        axis.text(
                            column_index,
                            row_index,
                            "N/A",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#333333",
                        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        colorbar.set_label("Percent")

    figure.suptitle(
        str(payload.get("plot_title") or "Exact-zero activation propagation"),
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"n={len(methods)} checkpoints; integer zero and total counts pooled across "
            "layers; logical opportunity is not realized kernel speedup"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    return figure


def _pooled_fraction_row(
    method: dict[str, Any],
    row_key: str,
    order: list[Any],
) -> list[float]:
    rows = method.get(row_key)
    if not isinstance(rows, list):
        raise ValueError(f"Propagation method {method.get('label')!r} is missing {row_key}.")
    counts = {str(stage): [0, 0] for stage in order}
    seen = {str(stage): False for stage in order}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Propagation rows must be JSON objects.")
        name = str(row.get("name"))
        if name not in counts:
            continue
        seen[name] = True
        zero_count = row.get("zero_count")
        total = row.get("total")
        if zero_count is None and total is None:
            continue
        if (
            zero_count is None
            or total is None
            or isinstance(zero_count, bool)
            or not isinstance(zero_count, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or zero_count < 0
            or total <= 0
            or zero_count > total
        ):
            raise ValueError(f"Invalid integer propagation counts for stage {name!r}.")
        counts[name][0] += zero_count
        counts[name][1] += total
    missing = [name for name, was_seen in seen.items() if not was_seen]
    if missing:
        raise ValueError(
            f"Propagation method {method.get('label')!r} is missing stages: {', '.join(missing)}."
        )
    return [
        zeros / total if total > 0 else math.nan
        for zeros, total in counts.values()
    ]
