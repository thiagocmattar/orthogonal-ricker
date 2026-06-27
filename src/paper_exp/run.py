from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import yaml

from paper_exp.config import validate_config
from paper_exp.eval import compute_smoke_metrics
from paper_exp.utils import build_manifest, write_json, write_jsonl

RUN_SEQUENCE_RE = re.compile(r"^(\d{3})-")


def run_smoke(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_config(config, allow_todos=True)

    experiment_id, run_id, run_dir = create_run_dir(config, config_path, run_id=run_id)

    num_examples = min(config["run"]["max_examples"], 3)
    predictions = [
        {
            "id": index,
            "input": f"smoke input {index}",
            "prediction": "SMOKE_PREDICTION",
            "target": "SMOKE_TARGET",
        }
        for index in range(num_examples)
    ]
    metrics = compute_smoke_metrics(predictions)
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=run_id,
        command=command,
        mode="smoke",
        config_id=experiment_id,
        result_path=run_dir,
    )

    write_run_artifacts(run_dir, config=config, metrics=metrics, manifest=manifest, predictions=predictions)
    return run_dir


def run_baseline(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_config(config, allow_todos=False)
    raise NotImplementedError(
        "TODO: choose the random-initialized pretraining budget before enabling the baseline run."
    )


def create_run_dir(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    run_id: str | None = None,
) -> tuple[str, str, Path]:
    experiment_id = make_experiment_id(config_path)
    experiment_dir = make_experiment_dir(config, experiment_id)
    numbered_run_id = make_run_id(experiment_dir, suffix=run_id)
    run_dir = experiment_dir / numbered_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return experiment_id, numbered_run_id, run_dir


def make_experiment_id(config_path: str | Path) -> str:
    return Path(config_path).stem


def make_experiment_dir(config: dict[str, Any], experiment_id: str) -> Path:
    experiment_dir = Path(config["output"]["dir"]) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def make_run_id(experiment_dir: Path, *, suffix: str | None = None) -> str:
    sequence = next_run_sequence(experiment_dir)
    return f"{sequence:03d}-{suffix or make_run_suffix()}"


def make_run_suffix() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def next_run_sequence(experiment_dir: Path) -> int:
    existing: list[int] = []
    for path in experiment_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_SEQUENCE_RE.match(path.name)
        if match:
            existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


def write_run_artifacts(
    run_dir: Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
