"""Presentation-neutral histogram reductions for explicit figure families."""

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


def validated_histogram_payload(
    payload: dict[str, Any],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Validate and return the common edges and methods of a histogram payload."""

    raw_edges = payload.get("bin_edges")
    methods = payload.get("methods")
    if not isinstance(raw_edges, list) or len(raw_edges) < 2:
        raise ValueError("Histogram payload has no usable bin edges.")
    edges = [float(value) for value in raw_edges]
    if any(not math.isfinite(value) for value in edges) or any(
        right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise ValueError("Histogram edges must be finite and strictly increasing.")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Histogram payload has no methods.")
    if not all(isinstance(method, dict) for method in methods):
        raise ValueError("Every histogram method must be a JSON object.")
    return edges, methods


def pooled_histogram(
    method: dict[str, Any],
    edges: list[float],
    *,
    separate_zero: bool,
) -> dict[str, Any]:
    """Pool stored integer histogram counts across a method's layers."""

    layers = method.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"Histogram method {method.get('label')!r} has no layers.")
    counts = [0] * (len(edges) - 1)
    total = 0
    exact_zeros = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Histogram layers must be JSON objects.")
        raw_counts = layer.get("counts")
        if not isinstance(raw_counts, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_counts
        ):
            raise ValueError("Histogram layer counts must be nonnegative integers.")
        if len(raw_counts) != len(counts):
            raise ValueError("Histogram layers do not share the stored bin edges.")
        counts = [
            pooled + value
            for pooled, value in zip(counts, raw_counts, strict=True)
        ]
        raw_total = layer.get("total")
        if (
            isinstance(raw_total, bool)
            or not isinstance(raw_total, int)
            or raw_total < sum(raw_counts)
        ):
            raise ValueError(
                "Histogram layer total must be an integer covering every in-range count."
            )
        total += raw_total
        threshold_hits = layer.get("threshold_hits") or {}
        raw_exact_zeros = threshold_hits.get("0", 0)
        if (
            isinstance(raw_exact_zeros, bool)
            or not isinstance(raw_exact_zeros, int)
            or raw_exact_zeros < 0
            or raw_exact_zeros > raw_total
        ):
            raise ValueError("Histogram exact-zero counts must be valid nonnegative integers.")
        exact_zeros += raw_exact_zeros

    pooled_layer = {
        "counts": counts,
        "total": total,
        "threshold_hits": {"0": exact_zeros},
    }
    if separate_zero:
        density, zero_fraction = histogram_nonzero_density(pooled_layer, edges)
    else:
        density = histogram_density(pooled_layer, edges)
        zero_fraction = 0.0
    return {"density": density, "zero_fraction": zero_fraction}


def bin_centers(edges: list[float]) -> list[float]:
    """Return the midpoint of each adjacent edge pair."""

    return [
        0.5 * (left + right)
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
