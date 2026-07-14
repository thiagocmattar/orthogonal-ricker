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
