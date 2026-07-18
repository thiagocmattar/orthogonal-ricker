from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: MappingLike) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def collect_git_commit(root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root) if root is not None else None,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect_git_dirty(root: str | Path | None = None) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(root) if root is not None else None,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def collect_gpu_info() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return rows


def collect_package_versions() -> dict[str, str]:
    packages = ("paper-exp", "PyYAML", "matplotlib", "numpy", "datasets", "transformers", "torch", "safetensors")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    run_id: str,
    command: str,
    mode: str,
    config_id: str | None = None,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "experiment_name": config["experiment_name"],
        "config_id": config_id or Path(config_path).stem,
        "run_id": run_id,
        "run_sequence": _run_sequence(run_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": collect_git_commit(Path.cwd()),
        "git_dirty": collect_git_dirty(Path.cwd()),
        "config_path": str(config_path),
        "result_path": str(result_path) if result_path is not None else None,
        "mode": mode,
        "model_provider": config["model"]["provider"],
        "model_name": config["model"]["name"],
        "model_architecture": config["model"].get("architecture"),
        "model_initialization": config["model"].get("initialization"),
        "dataset_name": config["data"]["name"],
        "dataset_split": config["data"]["split"],
        "metric": config["evaluation"]["metric"],
        "seed": config["run"]["seed"],
        "model_initialization_seed": config["run"].get(
            "model_initialization_seed", config["run"]["seed"]
        ),
        "data_order_seed": config["run"].get("data_order_seed", config["run"]["seed"]),
        "training_schedule_scheme": config["run"].get("training_schedule_scheme"),
        "training_schedule_hash": config["run"].get("training_schedule_hash"),
        "validation_partition": config.get("validation", {}).get("partition"),
        "validation_partition_hash": config.get("validation", {}).get("partition_hash"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "gpu_info": collect_gpu_info(),
        "package_versions": collect_package_versions(),
    }


MappingLike = dict[str, Any]


def _run_sequence(run_id: str) -> int | None:
    prefix = run_id.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else None
