"""Small plotting helpers shared by explicit paper-figure families.

Keep this module limited to presentation-neutral helpers that already have
multiple callers. Scientific cohorts, reductions, labels, and axis choices
belong to the figure family that owns them.
"""

from __future__ import annotations

import math
from typing import Any


def _histogram_method(payload: dict[str, Any], label_prefix: str) -> dict[str, Any] | None:
    for method in payload.get("methods", []):
        if str(method.get("label", "")).startswith(label_prefix):
            return method
    return None


def _histogram_layer(method: dict[str, Any], layer_name: str) -> dict[str, Any]:
    for layer in method.get("layers", []):
        if layer.get("name") == layer_name:
            return layer
    raise ValueError(f"Missing histogram layer {layer_name!r} for {method.get('label')!r}.")


def _histogram_density(layer: dict[str, Any], edges: list[float]) -> list[float]:
    counts = [float(value) for value in layer.get("counts", [])]
    total = float(layer.get("total") or sum(counts) or 1.0)
    widths = [right - left for left, right in zip(edges[:-1], edges[1:], strict=True)]
    return [count / total / width if width > 0.0 else 0.0 for count, width in zip(counts, widths, strict=True)]


def _histogram_nonzero_density(
    layer: dict[str, Any],
    edges: list[float],
) -> tuple[list[float], float]:
    """Separate an exact-zero atom from the conditional nonzero density.

    Histogram bins have width, while an exact-zero probability mass does not.
    Returning the two measures separately prevents a bin-width-dependent spike
    at zero from being presented as an ordinary continuous density.
    """

    counts = [float(value) for value in layer.get("counts", [])]
    if len(edges) != len(counts) + 1:
        raise ValueError("Histogram edges must contain exactly one more value than counts.")
    total = float(layer.get("total") or sum(counts) or 0.0)
    if total <= 0.0:
        return [0.0 for _count in counts], 0.0

    threshold_hits = layer.get("threshold_hits") or {}
    exact_zero_count = float(threshold_hits.get("0") or 0.0)
    if exact_zero_count < 0.0 or exact_zero_count > total:
        raise ValueError("Exact-zero count must lie between zero and the histogram total.")

    if exact_zero_count > 0.0:
        zero_bin = next(
            (
                index
                for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
                if left <= 0.0 < right or (index == len(counts) - 1 and right == 0.0)
            ),
            None,
        )
        if zero_bin is None:
            raise ValueError("Histogram range does not contain zero despite a nonzero exact-zero count.")
        if counts[zero_bin] + 1e-9 < exact_zero_count:
            raise ValueError("Exact-zero count exceeds the count in the histogram bin containing zero.")
        counts[zero_bin] -= exact_zero_count

    nonzero_total = total - exact_zero_count
    widths = [right - left for left, right in zip(edges[:-1], edges[1:], strict=True)]
    densities = [
        count / nonzero_total / width if nonzero_total > 0.0 and width > 0.0 else 0.0
        for count, width in zip(counts, widths, strict=True)
    ]
    return densities, exact_zero_count / total


def _histogram_center_window_mass(layer: dict[str, Any], edges: list[float], *, threshold: float) -> float:
    counts = [float(value) for value in layer.get("counts", [])]
    total = float(layer.get("total") or sum(counts) or 1.0)
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    selected_counts = [count for count, center in zip(counts, centers, strict=True) if abs(center) <= threshold]
    if selected_counts:
        window_count = sum(selected_counts)
    else:
        window_count = sum(
            count * max(0.0, min(right, threshold) - max(left, -threshold)) / (right - left)
            for count, left, right in zip(counts, edges[:-1], edges[1:], strict=True)
            if right > left
        )
    return window_count / total if total > 0.0 else 0.0


def _trimmed_decimal_tick(value: float, _position: int) -> str:
    label = f"{value:.2f}".rstrip("0")
    if label.endswith("."):
        label += "0"
    return label


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))
