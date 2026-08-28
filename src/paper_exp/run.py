from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, TextIO
from uuid import uuid4

import yaml

from paper_exp.config import validate_config, validate_smoke_config
from paper_exp.launch import EXPERIMENTS_DIR_NAME, SCAFFOLD_NAME_RE
from paper_exp.parallel import WorkerSlot
from paper_exp.utils import build_manifest, write_json, write_jsonl

RUN_SEQUENCE_RE = re.compile(r"^(\d{3})-")
CORE_RUN_ARTIFACTS: tuple[str, ...] = (
    "config.yaml",
    "manifest.json",
    "metrics.json",
    "predictions.jsonl",
)


@dataclass(frozen=True)
class RunHandle:
    """Immutable identity and launch-manifest snapshot for one run."""

    config_id: str
    run_id: str
    run_dir: Path
    tranche_id: str | None
    _config_json: str = field(repr=False)
    _launch_manifest_json: str = field(repr=False)

    @property
    def config(self) -> dict[str, Any]:
        """Return a detached copy of the launch-time config."""

        return json.loads(self._config_json)

    @property
    def launch_manifest(self) -> dict[str, Any]:
        """Return a detached copy of the launch-time manifest."""

        return json.loads(self._launch_manifest_json)


def start_run(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    mode: str,
    run_id: str | None = None,
    repository: str | Path | None = None,
) -> RunHandle:
    """Create a run and immediately persist its config and launch provenance."""

    config_snapshot = deepcopy(config)
    validate_config(config_snapshot, allow_todos=True)
    config_id, numbered_run_id, run_dir = create_run_dir(
        config_snapshot,
        config_path,
        run_id=run_id,
        repository=repository,
    )
    tranche_id = _tranche_id_from_run_dir(run_dir)
    _atomic_write_yaml(run_dir / "config.yaml", config_snapshot)
    manifest = build_manifest(
        config=config_snapshot,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode=mode,
        config_id=config_id,
        result_path=run_dir,
        tranche_id=tranche_id,
        repository=repository,
    )
    manifest["status"] = "running"
    manifest["started_at"] = manifest["timestamp"]

    _atomic_write_json(run_dir / "manifest.json", manifest)
    return RunHandle(
        config_id=config_id,
        run_id=numbered_run_id,
        run_dir=run_dir,
        tranche_id=tranche_id,
        _config_json=json.dumps(config_snapshot, sort_keys=True),
        _launch_manifest_json=json.dumps(manifest, sort_keys=True),
    )


def complete_run(
    run: RunHandle,
    *,
    metrics: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    manifest_updates: Mapping[str, Any] | None = None,
) -> Path:
    """Write result artifacts and publish the completed manifest last."""

    _require_running(run)
    manifest = _terminal_manifest(run, manifest_updates)
    manifest["status"] = "completed"

    _atomic_write_json(run.run_dir / "metrics.json", metrics)
    _atomic_write_jsonl(run.run_dir / "predictions.jsonl", predictions)
    manifest["finished_at"] = _utc_now()
    _atomic_write_json(run.run_dir / "manifest.json", manifest)
    return run.run_dir


def fail_run(
    run: RunHandle,
    error: BaseException,
    *,
    manifest_updates: Mapping[str, Any] | None = None,
) -> Path:
    """Record a terminal failure while preserving launch-time provenance."""

    _require_running(run)
    manifest = _terminal_manifest(run, manifest_updates)
    manifest["status"] = "failed"
    manifest["finished_at"] = _utc_now()
    manifest["failure"] = {
        "type": type(error).__qualname__,
        "message": str(error),
    }
    _atomic_write_json(run.run_dir / "manifest.json", manifest)
    return run.run_dir


@contextmanager
def run_lifecycle(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    mode: str,
    run_id: str | None = None,
    repository: str | Path | None = None,
) -> Iterator[RunHandle]:
    """Start a run and record any escaping exception without replacing it."""

    run = start_run(
        config,
        config_path=config_path,
        command=command,
        mode=mode,
        run_id=run_id,
        repository=repository,
    )
    try:
        yield run
    except BaseException as error:
        try:
            fail_run(run, error)
        except BaseException as recording_error:
            try:
                error.add_note(
                    "Additionally failed to record run failure: "
                    f"{type(recording_error).__qualname__}: {recording_error}"
                )
            except BaseException:
                pass
        raise
    else:
        status = _current_status(run)
        if status == "completed":
            return
        error = RuntimeError(
            f"Run lifecycle exited without completion; current status is {status!r}"
        )
        if status == "running":
            try:
                fail_run(run, error)
            except BaseException as recording_error:
                error.add_note(
                    "Additionally failed to record unterminated lifecycle: "
                    f"{type(recording_error).__qualname__}: {recording_error}"
                )
        raise error


def run_smoke(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
    worker_slots: Sequence[WorkerSlot[str]] = (),
    require_cuda: bool = False,
    allow_shared_gpu: bool = False,
) -> Path:
    from paper_exp.runner import RunnerError, parse_worker_slot

    if not isinstance(require_cuda, bool) or not isinstance(allow_shared_gpu, bool):
        raise TypeError("Smoke CUDA flags must be booleans.")
    requested_slots = tuple(worker_slots)
    if any(not isinstance(slot, WorkerSlot) for slot in requested_slots):
        raise ValueError("Smoke worker slots must be WorkerSlot values.")
    try:
        slots = tuple(
            parse_worker_slot(f"{slot.slot_id}={slot.payload}")
            for slot in requested_slots
        )
    except RunnerError as error:
        raise ValueError(f"Invalid smoke worker slot: {error}") from error
    if slots != requested_slots:
        raise ValueError("Smoke worker slots must use canonical string IDs and devices.")
    if slots and len(slots) != 2:
        raise ValueError("Concurrent smoke requires exactly two worker slots.")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("Smoke worker slot IDs must be distinct.")
    if not slots and (require_cuda or allow_shared_gpu):
        raise ValueError(
            "--require-cuda and --allow-shared-gpu require two worker slots."
        )
    devices = tuple(slot.payload for slot in slots)
    if len(set(devices)) != len(devices) and not allow_shared_gpu:
        raise ValueError(
            "Smoke CUDA devices must be distinct unless shared-GPU mode is explicit."
        )
    validate_smoke_config(config)
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode="smoke",
        run_id=run_id,
    ) as run:
        run_config = run.config
        num_examples = min(run_config["run"]["max_examples"], 3)
        predictions = [
            {
                "id": index,
                "input": f"smoke input {index}",
                "prediction": "SMOKE_PREDICTION",
                "target": "SMOKE_TARGET",
            }
            for index in range(num_examples)
        ]
        metrics = {
            "smoke/num_examples": len(predictions),
            "smoke/passed": bool(predictions),
        }
        manifest_updates: dict[str, Any] | None = None
        if slots:
            from paper_exp.infrastructure_smoke import REPORT_HASH_NAME
            from paper_exp.infrastructure_smoke import REPORT_NAME
            from paper_exp.infrastructure_smoke import (
                run_concurrent_infrastructure_smoke,
            )
            concurrent_root = run.run_dir / "concurrent"
            report = run_concurrent_infrastructure_smoke(
                slots,
                concurrent_root,
                require_cuda=require_cuda,
                allow_shared_gpu=allow_shared_gpu,
            )
            report_bytes = (concurrent_root / REPORT_NAME).read_bytes()
            report_sha256 = sha256(report_bytes).hexdigest()
            expected_sidecar = (report_sha256 + "\n").encode("ascii")
            if (concurrent_root / REPORT_HASH_NAME).read_bytes() != expected_sidecar:
                raise RuntimeError(
                    "Concurrent infrastructure smoke report checksum is invalid."
                )
            try:
                saved_report = json.loads(report_bytes)
            except (UnicodeError, ValueError) as error:
                raise RuntimeError(
                    "Concurrent infrastructure smoke report is malformed."
                ) from error
            if saved_report != report:
                raise RuntimeError(
                    "Concurrent infrastructure smoke report differs from its result."
                )
            if report.get("passed") is not True:
                raise RuntimeError("Concurrent infrastructure smoke did not pass.")
            metrics.update(
                {
                    "smoke/concurrent_passed": report["passed"],
                    "smoke/concurrent_attempts": len(report["attempts"]),
                }
            )
            manifest_updates = {
                "infrastructure_smoke": {
                    "report": f"concurrent/{REPORT_NAME}",
                    "sha256": report_sha256,
                    "require_cuda": require_cuda,
                    "allow_shared_gpu": allow_shared_gpu,
                    "worker_slots": {
                        slot.slot_id: slot.payload for slot in slots
                    },
                }
            }
        return complete_run(
            run,
            metrics=metrics,
            predictions=predictions,
            manifest_updates=manifest_updates,
        )


def create_run_dir(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    run_id: str | None = None,
    repository: str | Path | None = None,
) -> tuple[str, str, Path]:
    experiment_id = make_experiment_id(config_path)
    experiment_dir = make_experiment_dir(
        config,
        experiment_id,
        repository=repository,
    )
    numbered_run_id = make_run_id(experiment_dir, suffix=run_id)
    run_dir = experiment_dir / numbered_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return experiment_id, numbered_run_id, run_dir


def make_experiment_id(config_path: str | Path) -> str:
    return Path(config_path).stem


def make_experiment_dir(
    config: dict[str, Any],
    experiment_id: str,
    *,
    repository: str | Path | None = None,
) -> Path:
    output_dir = Path(config["output"]["dir"])
    if not output_dir.is_absolute() and repository is not None:
        output_dir = Path(repository).resolve() / output_dir
    experiment_dir = output_dir / experiment_id
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


def _tranche_id_from_run_dir(run_dir: Path) -> str | None:
    raw_dir = run_dir.parent.parent
    scaffold_dir = raw_dir.parent
    if (
        raw_dir.name != "raw"
        or scaffold_dir.parent.name != EXPERIMENTS_DIR_NAME
        or SCAFFOLD_NAME_RE.fullmatch(scaffold_dir.name) is None
    ):
        return None
    return scaffold_dir.name


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


def _terminal_manifest(
    run: RunHandle, updates: Mapping[str, Any] | None
) -> dict[str, Any]:
    manifest = deepcopy(dict(updates or {}))
    manifest.update(run.launch_manifest)
    return manifest


def _require_running(run: RunHandle) -> None:
    status = _current_status(run)
    if status != "running":
        raise RuntimeError(
            f"Run {run.run_dir} is already terminal with status {status!r}"
        )


def _current_status(run: RunHandle) -> str | None:
    manifest_path = run.run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"Cannot read current run manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Current run manifest is not an object: {manifest_path}")
    if manifest.get("config_id") != run.config_id or manifest.get("run_id") != run.run_id:
        raise RuntimeError(f"Current run manifest identity does not match {run.run_dir}")
    status = manifest.get("status")
    return status if isinstance(status, str) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    with _atomic_text_file(path) as handle:
        json.dump(dict(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _atomic_write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    with _atomic_text_file(path) as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with _atomic_text_file(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


@contextmanager
def _atomic_text_file(path: str | Path) -> Iterator[TextIO]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            yield handle
        temporary_path.replace(output_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
