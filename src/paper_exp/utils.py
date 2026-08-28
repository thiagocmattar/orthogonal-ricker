from __future__ import annotations

import json
import os
import platform
import socket
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


def portable_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Represent a path relative to the working tree when possible."""

    resolved = Path(path).resolve()
    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    run_id: str,
    command: str,
    mode: str,
    config_id: str | None = None,
    result_path: str | Path | None = None,
    tranche_id: str | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve() if repository is not None else Path.cwd().resolve()
    manifest = {
        "experiment_name": config["experiment_name"],
        "tranche_id": tranche_id,
        "config_id": config_id or Path(config_path).stem,
        "run_id": run_id,
        "run_sequence": _run_sequence(run_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": collect_git_commit(root),
        "git_dirty": collect_git_dirty(root),
        "config_path": portable_path(config_path, root=root),
        "result_path": (
            portable_path(result_path, root=root) if result_path is not None else None
        ),
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
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
    identity = config.get("identity")
    if identity is not None:
        if not isinstance(identity, dict):
            raise RuntimeError("Config identity must be a mapping before manifest publication.")
        required_identity = (
            "group_id",
            "condition_fingerprint",
            "training_implementation_id",
        )
        missing = [field for field in required_identity if field not in identity]
        if missing:
            raise RuntimeError(
                "Config identity is incomplete before manifest publication: "
                + ", ".join(missing)
            )
        from paper_exp.design import complete_config_sha256

        manifest.update(
            {
                "case_group_id": identity["group_id"],
                "condition_fingerprint": identity["condition_fingerprint"],
                "training_implementation_id": identity[
                    "training_implementation_id"
                ],
                "config_sha256": complete_config_sha256(config),
            }
        )
    worker_assignment = collect_worker_assignment()
    if worker_assignment is not None:
        if worker_assignment["config_id"] != manifest["config_id"]:
            raise RuntimeError(
                "Parallel-worker config identity does not match the run manifest."
            )
        manifest["worker_assignment"] = worker_assignment
    return manifest


def collect_worker_assignment() -> dict[str, Any] | None:
    """Return explicit coordinator/slot provenance for an isolated worker."""

    names = {
        "launch_id": "PAPER_EXP_PARALLEL_LAUNCH_ID",
        "slot_id": "PAPER_EXP_WORKER_SLOT_ID",
        "config_id": "PAPER_EXP_WORKER_CONFIG_ID",
        "launch_position": "PAPER_EXP_WORKER_LAUNCH_POSITION",
        "launch_size": "PAPER_EXP_WORKER_LAUNCH_SIZE",
        "coordinator_pid": "PAPER_EXP_COORDINATOR_PID",
        "gpu_uuid": "PAPER_EXP_WORKER_GPU_UUID",
        "gpu_name": "PAPER_EXP_WORKER_GPU_NAME",
        "gpu_total_memory_bytes": "PAPER_EXP_WORKER_GPU_TOTAL_MEMORY_BYTES",
        "gpu_compute_capability": "PAPER_EXP_WORKER_GPU_COMPUTE_CAPABILITY",
        "torch_version": "PAPER_EXP_WORKER_TORCH_VERSION",
        "cuda_runtime_version": "PAPER_EXP_WORKER_CUDA_RUNTIME_VERSION",
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
    }
    values = {field: os.environ.get(name) for field, name in names.items()}
    worker_markers = {
        field: value
        for field, value in values.items()
        if names[field].startswith("PAPER_EXP_")
    }
    if not any(value is not None for value in worker_markers.values()):
        return None
    missing = sorted(field for field, value in values.items() if value is None)
    if missing:
        raise RuntimeError(
            "Incomplete parallel-worker environment: " + ", ".join(missing)
        )
    return {
        "launch_id": values["launch_id"],
        "slot_id": values["slot_id"],
        "config_id": values["config_id"],
        "launch_position": int(str(values["launch_position"])),
        "launch_size": int(str(values["launch_size"])),
        "coordinator_pid": int(str(values["coordinator_pid"])),
        "worker_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": values["cuda_visible_devices"],
        "gpu_uuid": values["gpu_uuid"],
        "gpu_name": values["gpu_name"],
        "gpu_total_memory_bytes": int(str(values["gpu_total_memory_bytes"])),
        "gpu_compute_capability": values["gpu_compute_capability"],
        "torch_version": values["torch_version"],
        "cuda_runtime_version": values["cuda_runtime_version"],
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
    }


MappingLike = dict[str, Any]


def _run_sequence(run_id: str) -> int | None:
    prefix = run_id.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else None
