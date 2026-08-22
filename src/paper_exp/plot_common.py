"""Presentation-neutral reductions shared by explicit figure families."""

from __future__ import annotations

import math
from typing import Any


def histogram_layer(method: dict[str, Any], layer_name: str) -> dict[str, Any]:
    """Return one named layer histogram or fail instead of substituting data."""

    for layer in method.get("layers", []):
        if layer.get("name") == layer_name:
            return layer
    raise ValueError(f"Missing histogram layer {layer_name!r} for {method.get('label')!r}.")


def histogram_density(layer: dict[str, Any], edges: list[float]) -> list[float]:
    """Convert integer bin counts to a density over the stored total."""

    counts, total, widths = _histogram_inputs(layer, edges)
    return [
        count / total / width if total > 0.0 else 0.0
        for count, width in zip(counts, widths, strict=True)
    ]


def histogram_nonzero_density(
    layer: dict[str, Any],
    edges: list[float],
) -> tuple[list[float], float]:
    """Separate an exact-zero atom from the conditional nonzero density."""

    counts, total, widths = _histogram_inputs(layer, edges)
    if total <= 0.0:
        return [0.0 for _ in counts], 0.0

    threshold_hits = layer.get("threshold_hits") or {}
    raw_exact_zero_count = threshold_hits.get("0", 0)
    if (
        isinstance(raw_exact_zero_count, bool)
        or not isinstance(raw_exact_zero_count, int)
        or raw_exact_zero_count < 0
    ):
        raise ValueError("Exact-zero count must be a nonnegative integer.")
    exact_zero_count = float(raw_exact_zero_count)
    if not 0.0 <= exact_zero_count <= total:
        raise ValueError("Exact-zero count must lie between zero and the histogram total.")

    if exact_zero_count:
        zero_bin = next(
            (
                index
                for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
                if left <= 0.0 < right or (index == len(counts) - 1 and right == 0.0)
            ),
            None,
        )
        if zero_bin is None:
            raise ValueError("Histogram range excludes zero despite a nonzero exact-zero count.")
        if counts[zero_bin] + 1e-9 < exact_zero_count:
            raise ValueError("Exact-zero count exceeds its containing histogram bin.")
        counts[zero_bin] -= exact_zero_count

    nonzero_total = total - exact_zero_count
    density = [
        count / nonzero_total / width if nonzero_total > 0.0 else 0.0
        for count, width in zip(counts, widths, strict=True)
    ]
    return density, exact_zero_count / total


def finite_number(value: Any) -> bool:
    """Return whether a value is a finite real scalar, excluding booleans."""

    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _histogram_inputs(
    layer: dict[str, Any],
    edges: list[float],
) -> tuple[list[float], float, list[float]]:
    raw_counts = layer.get("counts", [])
    if not isinstance(raw_counts, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_counts
    ):
        raise ValueError("Histogram counts must be nonnegative integers.")
    counts = [float(value) for value in raw_counts]
    if len(edges) != len(counts) + 1:
        raise ValueError("Histogram edges must contain exactly one more value than counts.")
    widths = [right - left for left, right in zip(edges[:-1], edges[1:], strict=True)]
    if any(not math.isfinite(width) or width <= 0.0 for width in widths):
        raise ValueError("Histogram bin edges must be finite and strictly increasing.")
    raw_total = layer.get("total")
    if raw_total is None:
        total = float(sum(raw_counts))
    elif (
        isinstance(raw_total, bool)
        or not isinstance(raw_total, int)
        or raw_total < 0
    ):
        raise ValueError("Histogram total must be a nonnegative integer.")
    else:
        total = float(raw_total)
    if total < sum(counts) - 1e-9:
        raise ValueError("Histogram total must be finite and cover every in-range count.")
    return counts, total, widths
