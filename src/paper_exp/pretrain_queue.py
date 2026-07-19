from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Sequence
from uuid import uuid4

import yaml

from paper_exp.config import load_config
from paper_exp.run import CORE_RUN_ARTIFACTS, RUN_SEQUENCE_RE, make_experiment_id
from paper_exp.utils import collect_git_dirty, read_json


QUEUE_SCHEMA_VERSION = 1
DEFAULT_STATE_PATH = Path("run-logs/pretrain-queue-state.json")
DEFAULT_LOGS_DIR = Path("run-logs")


class PretrainQueueError(RuntimeError):
    """Raised when a queued launch cannot proceed without risking provenance."""


def run_pretrain_queue(
    config_paths: Sequence[str | Path],
    *,
    state_path: str | Path = DEFAULT_STATE_PATH,
    logs_dir: str | Path = DEFAULT_LOGS_DIR,
) -> dict[str, Any]:
    """Run committed pretraining configs sequentially in child processes.

    Saved run configs, manifests, metrics, events, and checkpoints remain the
    authority. The queue state only records orchestration progress.
    """

    if not config_paths:
        raise PretrainQueueError("The pretrain queue requires at least one --config.")

    repository = _git_root(Path.cwd())
    normalized_paths = [
        _normalize_config_path(path, repository) for path in config_paths
    ]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise PretrainQueueError("The pretrain queue contains duplicate config paths.")

    queue_state_path = _resolve_runtime_path(state_path, repository)
    queue_logs_dir = _resolve_runtime_path(logs_dir, repository)
    lock_path = repository / "tmp" / "pretrain-queue.lock"

    with _exclusive_lock(lock_path):
        state = _load_or_create_state(queue_state_path, normalized_paths)
        if state["status"] == "failed":
            raise PretrainQueueError(
                f"Queue state records a prior failure: {queue_state_path}. "
                "Use a new --state-path for an explicit retry."
            )

        queue_logs_dir.mkdir(parents=True, exist_ok=True)
        state["status"] = "running"
        state["updated_at"] = _utc_now()
        _atomic_write_json(queue_state_path, state)

        try:
            for index, config_path_text in enumerate(normalized_paths):
                item = state["items"][index]
                config_path = Path(config_path_text)
                config = load_config(config_path, allow_todos=False)
                experiment_dir = _experiment_dir(config, config_path, repository)

                completed_run = _inspect_existing_attempts(
                    config=config,
                    config_path=config_path,
                    experiment_dir=experiment_dir,
                    repository=repository,
                )
                if completed_run is not None:
                    item.update(
                        {
                            "status": "skipped",
                            "run_dir": str(completed_run),
                            "finished_at": _utc_now(),
                            "message": "Verified completed run already exists.",
                        }
                    )
                    state["current_index"] = index + 1
                    state["updated_at"] = _utc_now()
                    _atomic_write_json(queue_state_path, state)
                    continue

                if item.get("status") == "running":
                    raise PretrainQueueError(
                        f"Queue state has an unresolved running item for {config_path}. "
                        "Inspect its authoritative result manifest before retrying."
                    )

                _require_clean_git_tree(repository)
                _reject_running_pretrains(_output_roots(normalized_paths, repository))

                before = _attempt_directories(experiment_dir)
                log_token = f"{_timestamp_token()}-{uuid4().hex[:8]}"
                log_stem = f"{make_experiment_id(config_path)}-queue-{log_token}"
                stdout_log = queue_logs_dir / f"{log_stem}.log"
                stderr_log = queue_logs_dir / f"{log_stem}.err.log"
                command = [
                    sys.executable,
                    "-m",
                    "paper_exp.cli",
                    "pretrain",
                    "--config",
                    _child_config_path(config_path, repository),
                ]

                item.update(
                    {
                        "status": "running",
                        "started_at": _utc_now(),
                        "finished_at": None,
                        "run_dir": None,
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "returncode": None,
                        "message": None,
                    }
                )
                state["current_index"] = index
                state["updated_at"] = _utc_now()
                _atomic_write_json(queue_state_path, state)

                returncode = _run_child(
                    command,
                    cwd=repository,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )
                item["returncode"] = returncode
                new_attempts = _attempt_directories(experiment_dir) - before
                if len(new_attempts) != 1:
                    raise PretrainQueueError(
                        f"Expected exactly one new run directory for {config_path}, "
                        f"found {len(new_attempts)}."
                    )
                run_dir = next(iter(new_attempts))
                item["run_dir"] = str(run_dir)

                if returncode != 0:
                    manifest_status = _manifest_status(run_dir)
                    raise PretrainQueueError(
                        f"Pretraining child failed for {config_path} with exit code "
                        f"{returncode}; manifest status is {manifest_status!r}."
                    )

                _verify_completed_run(
                    run_dir,
                    config=config,
                    config_path=config_path,
                    repository=repository,
                )
                item.update(
                    {
                        "status": "completed",
                        "finished_at": _utc_now(),
                        "message": "Child exited successfully and artifacts verified.",
                    }
                )
                state["current_index"] = index + 1
                state["updated_at"] = _utc_now()
                _atomic_write_json(queue_state_path, state)

            state["status"] = "completed"
            state["finished_at"] = _utc_now()
            state["updated_at"] = state["finished_at"]
            state["current_index"] = len(state["items"])
            _atomic_write_json(queue_state_path, state)
            return state
        except BaseException as error:
            state["status"] = "failed"
            state["finished_at"] = _utc_now()
            state["updated_at"] = state["finished_at"]
            state["failure"] = {
                "type": type(error).__qualname__,
                "message": str(error),
            }
            current_index = state.get("current_index")
            if isinstance(current_index, int) and 0 <= current_index < len(state["items"]):
                item = state["items"][current_index]
                if item.get("status") == "running":
                    item["status"] = "failed"
                    item["finished_at"] = state["finished_at"]
                    item["message"] = str(error)
            _atomic_write_json(queue_state_path, state)
            raise


def _normalize_config_path(path: str | Path, repository: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    return str(candidate.resolve())


def _resolve_runtime_path(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository / candidate


def _child_config_path(config_path: Path, repository: Path) -> str:
    try:
        return str(config_path.relative_to(repository))
    except ValueError:
        return str(config_path)


def _experiment_dir(config: dict[str, Any], config_path: Path, repository: Path) -> Path:
    output_dir = Path(str(config["output"]["dir"]))
    if not output_dir.is_absolute():
        output_dir = repository / output_dir
    return output_dir / make_experiment_id(config_path)


def _output_roots(config_paths: Sequence[str], repository: Path) -> set[Path]:
    roots: set[Path] = set()
    for config_path_text in config_paths:
        config_path = Path(config_path_text)
        config = load_config(config_path, allow_todos=False)
        output_dir = Path(str(config["output"]["dir"]))
        roots.add(output_dir if output_dir.is_absolute() else repository / output_dir)
    return roots


def _inspect_existing_attempts(
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_dir: Path,
    repository: Path,
) -> Path | None:
    completed: list[Path] = []
    for run_dir in sorted(_attempt_directories(experiment_dir)):
        manifest_path = run_dir / "manifest.json"
        try:
            manifest = read_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as error:
            raise PretrainQueueError(
                f"Existing attempt has no readable manifest: {run_dir}."
            ) from error
        if not isinstance(manifest, dict):
            raise PretrainQueueError(f"Existing attempt manifest is not an object: {run_dir}.")
        if manifest.get("config_id") != experiment_dir.name or manifest.get("run_id") != run_dir.name:
            raise PretrainQueueError(f"Existing attempt manifest identity is inconsistent: {run_dir}.")

        _verify_saved_config(run_dir, config)
        status = manifest.get("status")
        if status == "running":
            raise PretrainQueueError(
                f"Existing attempt is still marked running: {run_dir}. "
                "Inspect process and artifacts; the queue will not relaunch blindly."
            )
        if status == "completed":
            _verify_completed_run(
                run_dir,
                config=config,
                config_path=config_path,
                repository=repository,
            )
            completed.append(run_dir)
            continue
        if status == "failed":
            if not manifest.get("finished_at") or not isinstance(manifest.get("failure"), dict):
                raise PretrainQueueError(f"Failed attempt manifest is inconsistent: {run_dir}.")
            continue
        raise PretrainQueueError(
            f"Existing attempt has unresolved or inconsistent status {status!r}: {run_dir}."
        )
    return completed[-1] if completed else None


def _verify_saved_config(run_dir: Path, config: dict[str, Any]) -> None:
    config_snapshot = run_dir / "config.yaml"
    try:
        saved = yaml.safe_load(config_snapshot.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PretrainQueueError(f"Cannot read saved config snapshot: {config_snapshot}.") from error
    if saved != config:
        raise PretrainQueueError(
            f"Saved config does not match the queued immutable config: {config_snapshot}."
        )


def _verify_completed_run(
    run_dir: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    repository: Path,
) -> None:
    missing = [name for name in (*CORE_RUN_ARTIFACTS, "events.jsonl") if not (run_dir / name).is_file()]
    if missing:
        raise PretrainQueueError(
            f"Completed attempt is missing required artifacts at {run_dir}: {', '.join(missing)}."
        )

    _verify_saved_config(run_dir, config)
    manifest = read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise PretrainQueueError(f"Completed manifest is not an object: {run_dir}.")
    if manifest.get("status") != "completed" or not manifest.get("finished_at"):
        raise PretrainQueueError(f"Attempt is not explicitly terminal-completed: {run_dir}.")
    if manifest.get("config_id") != make_experiment_id(config_path) or manifest.get("run_id") != run_dir.name:
        raise PretrainQueueError(f"Completed manifest identity is inconsistent: {run_dir}.")
    if manifest.get("mode") != "pretrain":
        raise PretrainQueueError(f"Completed attempt is not a pretraining run: {run_dir}.")
    if manifest.get("git_dirty") is not False or not str(manifest.get("git_commit") or "").strip():
        raise PretrainQueueError(f"Completed attempt lacks clean launch provenance: {run_dir}.")

    try:
        metrics = read_json(run_dir / "metrics.json")
    except (OSError, UnicodeError, ValueError) as error:
        raise PretrainQueueError(f"Cannot read completed metrics: {run_dir}.") from error
    if not isinstance(metrics, dict):
        raise PretrainQueueError(f"Completed metrics are not an object: {run_dir}.")
    training = config.get("training")
    if not isinstance(training, dict):
        raise PretrainQueueError(f"Queued pretraining config has no training section: {config_path}.")
    planned_steps = _required_int(training.get("max_steps"), "training.max_steps", config_path)
    completed_steps = _required_int(
        metrics.get("calibration/optimizer_steps"),
        "calibration/optimizer_steps",
        run_dir / "metrics.json",
    )
    manifest_training = manifest.get("training")
    if not isinstance(manifest_training, dict) or manifest_training.get("completed_steps") != completed_steps:
        raise PretrainQueueError(f"Completed-step metadata is inconsistent: {run_dir}.")
    if metrics.get("calibration/planned_optimizer_steps") != planned_steps:
        raise PretrainQueueError(f"Planned-step metadata is inconsistent: {run_dir}.")
    if training.get("max_wall_seconds") is None and completed_steps != planned_steps:
        raise PretrainQueueError(
            f"Completed attempt reached {completed_steps} of {planned_steps} required steps: {run_dir}."
        )

    validation = config.get("validation")
    if isinstance(validation, dict) and validation.get("enabled", False):
        validation_loss = metrics.get("calibration/validation_loss_final")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, (int, float))
            or not math.isfinite(float(validation_loss))
        ):
            raise PretrainQueueError(f"Completed validation loss is not finite: {run_dir}.")

    checkpoint = config.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("save_final", False):
        checkpoint_manifest = manifest.get("checkpoint")
        if not isinstance(checkpoint_manifest, dict) or checkpoint_manifest.get("saved") is not True:
            raise PretrainQueueError(f"Completed final checkpoint is not marked saved: {run_dir}.")
        checkpoint_path = Path(str(checkpoint_manifest.get("path") or ""))
        if not checkpoint_path.is_absolute():
            checkpoint_path = repository / checkpoint_path
        if not checkpoint_path.is_dir():
            raise PretrainQueueError(f"Completed final checkpoint is missing: {checkpoint_path}.")


def _required_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PretrainQueueError(f"{field} is not a non-negative integer in {source}.")
    return value


def _manifest_status(run_dir: Path) -> str | None:
    try:
        manifest = read_json(run_dir / "manifest.json")
    except (OSError, UnicodeError, ValueError):
        return None
    return manifest.get("status") if isinstance(manifest, dict) else None


def _attempt_directories(experiment_dir: Path) -> set[Path]:
    if not experiment_dir.is_dir():
        return set()
    attempts: set[Path] = set()
    for path in experiment_dir.iterdir():
        if not path.is_dir():
            continue
        if RUN_SEQUENCE_RE.match(path.name) is None:
            raise PretrainQueueError(f"Unrecognized run directory in experiment output: {path}.")
        attempts.add(path)
    return attempts


def _reject_running_pretrains(output_roots: set[Path]) -> None:
    for output_root in output_roots:
        if not output_root.is_dir():
            continue
        for manifest_path in output_root.glob("*/*/manifest.json"):
            try:
                manifest = read_json(manifest_path)
            except (OSError, UnicodeError, ValueError):
                continue
            if (
                isinstance(manifest, dict)
                and manifest.get("mode") == "pretrain"
                and manifest.get("status") == "running"
            ):
                raise PretrainQueueError(
                    f"Another pretraining attempt is marked running: {manifest_path.parent}."
                )


def _run_child(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("x", encoding="utf-8") as stdout_handle, stderr_log.open(
        "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )
    return completed.returncode


def _git_root(cwd: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PretrainQueueError("The pretrain queue must run inside a Git repository.") from error
    root = completed.stdout.strip()
    if not root:
        raise PretrainQueueError("Git did not return a repository root.")
    return Path(root).resolve()


def _require_clean_git_tree(repository: Path) -> None:
    dirty = collect_git_dirty(repository)
    if dirty is None:
        raise PretrainQueueError("Cannot determine Git working-tree state before launch.")
    if dirty:
        raise PretrainQueueError(
            f"Git working tree is dirty before launch: {repository}. Commit or remove changes first."
        )


def _load_or_create_state(state_path: Path, config_paths: Sequence[str]) -> dict[str, Any]:
    if state_path.exists():
        try:
            state = read_json(state_path)
        except (OSError, UnicodeError, ValueError) as error:
            raise PretrainQueueError(f"Cannot read queue state: {state_path}.") from error
        if not isinstance(state, dict) or state.get("schema_version") != QUEUE_SCHEMA_VERSION:
            raise PretrainQueueError(f"Queue state has an unsupported schema: {state_path}.")
        saved_paths = [item.get("config_path") for item in state.get("items", [])]
        if saved_paths != list(config_paths):
            raise PretrainQueueError(
                f"Queue state config order does not match this invocation: {state_path}."
            )
        return state

    created_at = _utc_now()
    state = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "queue_id": uuid4().hex,
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
        "finished_at": None,
        "current_index": 0,
        "items": [
            {
                "config_path": config_path,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "run_dir": None,
                "stdout_log": None,
                "stderr_log": None,
                "returncode": None,
                "message": None,
            }
            for config_path in config_paths
        ],
    }
    _atomic_write_json(state_path, state)
    return state


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise PretrainQueueError(
            f"Another pretrain queue owns the lock {lock_path}. "
            "If no queue is alive, inspect the state before removing a stale lock."
        ) from error
    try:
        payload = json.dumps({"pid": os.getpid(), "created_at": _utc_now()}) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
