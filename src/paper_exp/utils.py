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
    with Path(path).open("r", encoding="utf-8") as handle:
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


def collect_package_versions() -> dict[str, str]:
    packages = ("paper-exp", "PyYAML", "matplotlib")
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
        "config_path": str(config_path),
        "result_path": str(result_path) if result_path is not None else None,
        "mode": mode,
        "model_provider": config["model"]["provider"],
        "model_name": config["model"]["name"],
        "dataset_name": config["data"]["name"],
        "dataset_split": config["data"]["split"],
        "metric": config["evaluation"]["metric"],
        "seed": config["run"]["seed"],
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": collect_package_versions(),
    }


MappingLike = dict[str, Any]


def _run_sequence(run_id: str) -> int | None:
    prefix = run_id.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else None
