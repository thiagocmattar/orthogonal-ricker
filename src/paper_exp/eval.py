from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compute_smoke_metrics(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "smoke/num_examples": len(predictions),
        "smoke/passed": len(predictions) > 0,
    }


def evaluate_predictions(predictions: Sequence[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if metric_name.strip().upper().startswith("TODO"):
        raise NotImplementedError("TODO: define the real evaluation metric before scoring predictions.")

    raise NotImplementedError(
        "TODO: implement task-specific evaluation once the paper metric and prediction format are known."
    )
