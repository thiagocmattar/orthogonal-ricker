from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import yaml

from paper_exp.config import load_config
from paper_exp.run import CORE_RUN_ARTIFACTS, RUN_SEQUENCE_RE, make_experiment_id
from paper_exp.utils import collect_git_dirty, read_json


QUEUE_SCHEMA_VERSION = 1
DEFAULT_STATE_PATH = Path("run-logs/pretrain-queue-state.json")
DEFAULT_LOGS_DIR = Path("run-logs")
REVIEWED_RETRY_AUTHORIZATION_MARKER = "PRETRAIN_QUEUE_RETRY_AUTHORIZATION_V1"
RECOVERY_SCIENCE_CODE_PATHS = (
    "src/paper_exp/activation_pressure.py",
    "src/paper_exp/activations.py",
    "src/paper_exp/calibration.py",
    "src/paper_exp/config.py",
    "src/paper_exp/data.py",
    "src/paper_exp/eval.py",
    "src/paper_exp/modeling.py",
    "src/paper_exp/reproducibility.py",
    "src/paper_exp/run.py",
    "src/paper_exp/utils.py",
)
RECOVERY_REVIEWED_CODE_PATHS = (
    "src/paper_exp/cli.py",
    "src/paper_exp/pretrain_queue.py",
)
_RETRY_MARKER_RE = re.compile(
    rf"(?<!\S)({REVIEWED_RETRY_AUTHORIZATION_MARKER} "
    r"failed_attempt=(?P<failed_attempt>[0-9]{3}) "
    r"authorized_next_attempt=(?P<next_attempt>[0-9]{3}) "
    r"inventory_sha256=(?P<inventory>[0-9a-f]{64}) "
    r"predecessor_queue_id=(?P<queue_id>[0-9a-f]{32}) "
    r"predecessor_queue_sha256=(?P<queue_sha>[0-9a-f]{64}) "
    r"stdout_sha256=(?P<stdout_sha>[0-9a-f]{64}) "
    r"stderr_sha256=(?P<stderr_sha>[0-9a-f]{64}) "
    r"terminated_pid=(?P<pid>[1-9][0-9]*))(?!\S)"
)
_RETRY_MARKER_KEYS = (
    REVIEWED_RETRY_AUTHORIZATION_MARKER,
    "failed_attempt=",
    "authorized_next_attempt=",
    "inventory_sha256=",
    "predecessor_queue_id=",
    "predecessor_queue_sha256=",
    "stdout_sha256=",
    "stderr_sha256=",
    "terminated_pid=",
)
_FULL_RUN_ID_RE = re.compile(r"^[0-9]{3}-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")


class PretrainQueueError(RuntimeError):
    """Raised when a queued launch cannot proceed without risking provenance."""


def run_pretrain_queue(
    config_paths: Sequence[str | Path],
    *,
    state_path: str | Path = DEFAULT_STATE_PATH,
    logs_dir: str | Path = DEFAULT_LOGS_DIR,
    recovery_of_state_path: str | Path | None = None,
    reviewed_retry_run_id: str | None = None,
    confirm_reviewed_retry_process_exited: bool = False,
    reviewed_retry_terminated_pid: int | None = None,
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
        recovery = _reviewed_recovery_contract(
            repository=repository,
            config_paths=normalized_paths,
            state_path=queue_state_path,
            logs_dir=queue_logs_dir,
            predecessor_state_path=recovery_of_state_path,
            reviewed_retry_run_id=reviewed_retry_run_id,
            confirm_process_exited=confirm_reviewed_retry_process_exited,
            terminated_pid=reviewed_retry_terminated_pid,
        )
        authorized_running = recovery["authorized_running"] if recovery is not None else {}
        recovery_state = recovery["state"] if recovery is not None else None
        if recovery is not None:
            _require_clean_git_tree(repository)
            _revalidate_reviewed_recovery(recovery)
        state = _load_or_create_state(
            queue_state_path,
            normalized_paths,
            recovery=recovery_state,
        )
        if recovery_state is not None and state["queue_id"] == recovery_state["predecessor_queue_id"]:
            raise PretrainQueueError("Recovery queue id collided with its predecessor queue id.")
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
            encountered_authorized: set[tuple[str, str]] = set()
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
                    authorized_running=authorized_running,
                    encountered_authorized=encountered_authorized,
                    allow_completed=not bool(recovery_state),
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
                if recovery is not None:
                    _revalidate_reviewed_recovery(recovery)
                _reject_running_pretrains(
                    _output_roots(normalized_paths, repository),
                    authorized_running=authorized_running,
                )

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
                if recovery is not None:
                    _revalidate_reviewed_recovery(recovery)
                item["returncode"] = returncode
                new_attempts = _attempt_directories(experiment_dir) - before
                if len(new_attempts) != 1:
                    raise PretrainQueueError(
                        f"Expected exactly one new run directory for {config_path}, "
                        f"found {len(new_attempts)}."
                    )
                run_dir = next(iter(new_attempts))
                item["run_dir"] = str(run_dir)
                if recovery_state is not None:
                    expected_sequence = recovery_state["expected_new_attempts"].get(
                        make_experiment_id(config_path)
                    )
                    actual_sequence = int(run_dir.name.split("-", 1)[0])
                    if actual_sequence != expected_sequence:
                        raise PretrainQueueError(
                            f"Recovery created attempt {actual_sequence:03d} for {config_path}; "
                            f"expected {expected_sequence:03d}."
                        )

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

            if set(authorized_running) != encountered_authorized:
                raise PretrainQueueError(
                    "Recovery queue did not encounter every exact reviewed running attempt once."
                )
            if recovery is not None:
                _revalidate_reviewed_recovery(recovery)
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
    authorized_running: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    encountered_authorized: set[tuple[str, str]] | None = None,
    allow_completed: bool = True,
) -> Path | None:
    authorized_running = authorized_running or {}
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
            identity = (experiment_dir.name, run_dir.name)
            authorization = authorized_running.get(identity)
            if authorization is not None:
                _verify_authorized_running_attempt(
                    run_dir=run_dir,
                    manifest=manifest,
                    authorization=authorization,
                )
                if encountered_authorized is not None:
                    if identity in encountered_authorized:
                        raise PretrainQueueError(
                            f"Reviewed running attempt was encountered more than once: {run_dir}."
                        )
                    encountered_authorized.add(identity)
                continue
            raise PretrainQueueError(
                f"Existing attempt is still marked running: {run_dir}. "
                "Inspect process and artifacts; the queue will not relaunch blindly."
            )
        if status == "completed":
            if not allow_completed:
                raise PretrainQueueError(
                    f"Fresh recovery queue found a pre-existing completed attempt: {run_dir}."
                )
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


def _reject_running_pretrains(
    output_roots: set[Path],
    *,
    authorized_running: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> None:
    authorized_running = authorized_running or {}
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
                identity = (str(manifest.get("config_id") or ""), str(manifest.get("run_id") or ""))
                authorization = authorized_running.get(identity)
                if authorization is not None:
                    _verify_authorized_running_attempt(
                        run_dir=manifest_path.parent,
                        manifest=manifest,
                        authorization=authorization,
                    )
                    continue
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


def _load_or_create_state(
    state_path: Path,
    config_paths: Sequence[str],
    *,
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        if state.get("recovery") != recovery:
            raise PretrainQueueError(
                f"Queue state recovery lineage does not match this invocation: {state_path}."
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
        "recovery": dict(recovery) if recovery is not None else None,
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


def _reviewed_recovery_contract(
    *,
    repository: Path,
    config_paths: Sequence[str],
    state_path: Path,
    logs_dir: Path,
    predecessor_state_path: str | Path | None,
    reviewed_retry_run_id: str | None,
    confirm_process_exited: bool,
    terminated_pid: int | None,
) -> dict[str, Any] | None:
    run_id = str(reviewed_retry_run_id or "").strip()
    requested = (
        predecessor_state_path is not None
        or bool(run_id)
        or confirm_process_exited
        or terminated_pid is not None
    )
    if not requested:
        return None
    if (
        predecessor_state_path is None
        or not run_id
        or confirm_process_exited is not True
        or type(terminated_pid) is not int
        or terminated_pid <= 0
    ):
        raise PretrainQueueError(
            "A reviewed retry requires --recovery-of-state-path, exactly one "
            "--reviewed-retry-run-id, --reviewed-retry-terminated-pid, and "
            "--confirm-reviewed-retry-process-exited."
        )
    if _FULL_RUN_ID_RE.fullmatch(run_id) is None:
        raise PretrainQueueError("Reviewed retry run id is malformed.")
    if _process_exists(terminated_pid):
        raise PretrainQueueError(
            f"Reviewed terminated PID {terminated_pid} is still alive; retry is unsafe."
        )

    predecessor_path = _absolute_path(
        _resolve_runtime_path(predecessor_state_path, repository)
    )
    authoritative_repository = predecessor_path.parent.parent
    state_path = _absolute_path(state_path)
    logs_dir = _absolute_path(logs_dir)
    _require_no_link_like_components(predecessor_path, "predecessor queue state")
    _require_regular_file(predecessor_path, "predecessor queue state")
    if state_path.exists():
        raise PretrainQueueError("A reviewed recovery queue requires a fresh nonexistent state path.")
    if logs_dir.exists():
        raise PretrainQueueError("A reviewed recovery queue requires a fresh nonexistent logs directory.")
    try:
        predecessor_bytes = predecessor_path.read_bytes()
        predecessor = json.loads(predecessor_bytes)
    except (OSError, UnicodeError, ValueError) as error:
        raise PretrainQueueError(
            f"Cannot read failed predecessor queue state: {predecessor_path}."
        ) from error
    if not isinstance(predecessor, dict) or predecessor.get("schema_version") != QUEUE_SCHEMA_VERSION:
        raise PretrainQueueError("Failed predecessor queue has an unsupported schema.")
    if predecessor.get("status") != "failed":
        raise PretrainQueueError("A recovery queue requires an explicitly failed predecessor queue.")
    if predecessor.get("recovery") is not None:
        raise PretrainQueueError(
            "Recovery-of-recovery is intentionally unsupported. Preserve every attempt and obtain "
            "a new reviewed adjudication plus queue implementation before another retry."
        )
    items = predecessor.get("items")
    failed_index = predecessor.get("current_index")
    if (
        not isinstance(items, list)
        or isinstance(failed_index, bool)
        or not isinstance(failed_index, int)
        or not 0 <= failed_index < len(items)
        or not all(isinstance(item, dict) for item in items)
    ):
        raise PretrainQueueError("Failed predecessor queue has an inconsistent item ledger.")
    failed_items = [index for index, item in enumerate(items) if item.get("status") == "failed"]
    if failed_items != [failed_index]:
        raise PretrainQueueError("Failed predecessor queue does not have one exact fail-stop item.")
    if any(item.get("status") != "pending" for item in items[failed_index + 1 :]):
        raise PretrainQueueError("Failed predecessor queue launched work after its fail-stop item.")
    if any(
        item.get("status") != "completed"
        or type(item.get("returncode")) is not int
        or item.get("returncode") != 0
        or item.get("message") != "Child exited successfully and artifacts verified."
        for item in items[:failed_index]
    ):
        raise PretrainQueueError("Failed predecessor queue prefix is not exactly completed.")
    queue_id = str(predecessor.get("queue_id") or "")
    if re.fullmatch(r"[0-9a-f]{32}", queue_id) is None:
        raise PretrainQueueError("Failed predecessor queue id is malformed.")
    if (
        predecessor.get("finished_at") != predecessor.get("updated_at")
        or predecessor.get("finished_at") != items[failed_index].get("finished_at")
    ):
        raise PretrainQueueError("Failed predecessor queue terminal timestamps are inconsistent.")

    recovery_items = items[failed_index:]
    expected_ids = [Path(str(item.get("config_path") or "")).stem for item in recovery_items]
    actual_ids = [Path(path).stem for path in config_paths]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise PretrainQueueError(
            "Recovery configs must be the exact failed-and-pending suffix of the predecessor queue."
        )

    failed_item = items[failed_index]
    failed_run_dir = _absolute_path(Path(str(failed_item.get("run_dir") or "")))
    if not failed_run_dir.is_dir() or failed_run_dir.name != run_id:
        raise PretrainQueueError("Reviewed retry id does not identify the predecessor fail-stop run.")
    if type(failed_item.get("returncode")) is not int or failed_item["returncode"] == 0:
        raise PretrainQueueError("Predecessor fail-stop item does not record a nonzero child exit.")
    predecessor_failure = predecessor.get("failure")
    if (
        not isinstance(predecessor_failure, dict)
        or predecessor_failure.get("type") != "PretrainQueueError"
        or predecessor_failure.get("message") != failed_item.get("message")
    ):
        raise PretrainQueueError("Predecessor queue has no exact failure message.")
    if "manifest status is 'running'" not in str(failed_item.get("message") or ""):
        raise PretrainQueueError("Reviewed pretrain retry is only valid for a retained running manifest.")

    manifest = _read_json_mapping(failed_run_dir / "manifest.json", "reviewed retry manifest")
    launch_commit = str(manifest.get("git_commit") or "")
    if re.fullmatch(r"[0-9a-f]{40}", launch_commit) is None:
        raise PretrainQueueError("Reviewed retry manifest launch commit is malformed.")
    current_head = _git_head(repository)
    if not _git_is_ancestor(repository, launch_commit, current_head):
        raise PretrainQueueError("Recovery commit is not descended from the predecessor launch commit.")
    changed_science_paths = [
        path
        for path in RECOVERY_SCIENCE_CODE_PATHS
        if _git_blob_oid(repository, launch_commit, path)
        != _git_blob_oid(repository, current_head, path)
    ]
    if changed_science_paths:
        raise PretrainQueueError(
            "Recovery changed launch-critical science code: " + ", ".join(changed_science_paths)
        )

    config_guards: list[dict[str, Any]] = []
    loaded_configs: dict[str, dict[str, Any]] = {}
    protected_paths: list[tuple[Path, str]] = [
        (predecessor_path, "predecessor queue state"),
        (failed_run_dir, "retained running attempt"),
        (failed_run_dir.parent, "retained experiment directory"),
        (failed_run_dir.parent.parent, "retained output root"),
        (repository / ".git", "repository metadata"),
    ]
    for config_path_text, prior_item in zip(config_paths, recovery_items, strict=True):
        current_path = _absolute_path(Path(config_path_text))
        prior_path = _absolute_path(Path(str(prior_item.get("config_path") or "")))
        _require_no_link_like_components(current_path, "recovery config")
        _require_regular_file(current_path, "recovery config")
        if not _same_file(current_path, prior_path):
            raise PretrainQueueError(
                f"Recovery config path differs from its predecessor queue item: {current_path.stem}."
            )
        relative = _relative_to_repository(current_path, repository, "recovery config")
        try:
            current_raw = yaml.safe_load(current_path.read_text(encoding="utf-8"))
            launch_bytes = _git_file_bytes(repository, launch_commit, relative.as_posix())
            launch_raw = yaml.safe_load(launch_bytes.decode("utf-8"))
            current_config = load_config(current_path, allow_todos=False)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
            raise PretrainQueueError(
                f"Cannot compare recovery config with launch commit: {current_path}."
            ) from error
        if not isinstance(current_raw, dict) or current_raw != launch_raw:
            raise PretrainQueueError(
                f"Recovery config changed scientifically from launch commit {launch_commit}: {current_path.stem}."
            )
        loaded_configs[current_path.stem] = current_config
        config_guards.append(
            {
                "config_id": current_path.stem,
                "path": str(current_path),
                "sha256": _sha256_file(current_path),
                "launch_blob_sha256": hashlib.sha256(launch_bytes).hexdigest(),
            }
        )
        protected_paths.append((current_path, f"recovery config {current_path.stem}"))

    config_registry_path = repository / "docs" / "experimental-design" / "config-registry.yaml"
    config_registry = _read_yaml_mapping(
        config_registry_path,
        "config registry",
    )
    run_registry_path = repository / "docs" / "experimental-design" / "run-registry.yaml"
    run_registry = _read_yaml_mapping(run_registry_path, "run registry")
    config_records = config_registry.get("records")
    run_records = run_registry.get("records")
    if not isinstance(config_records, list) or not all(isinstance(row, dict) for row in config_records):
        raise PretrainQueueError("Config registry records are malformed.")
    if not isinstance(run_records, list) or not all(isinstance(row, dict) for row in run_records):
        raise PretrainQueueError("Run registry records are malformed.")
    for expected_id in expected_ids:
        config_rows = [row for row in config_records if row.get("config_id") == expected_id]
        if (
            len(config_rows) != 1
            or config_rows[0].get("config_status") != "ready"
            or config_rows[0].get("canonical_run_id") is not None
            or Path(str(config_rows[0].get("config_path") or "")).stem != expected_id
        ):
            raise PretrainQueueError(
                f"Recovery suffix config is not uniquely registered ready and noncanonical: {expected_id}."
            )
    config_id = expected_ids[0]
    rows = [
        row
        for row in run_records
        if row.get("config_id") == config_id and row.get("run_id") == failed_run_dir.name
    ]
    if len(rows) != 1:
        raise PretrainQueueError("Reviewed retry attempt is not uniquely registered.")
    suffix_rows = [row for row in run_records if row.get("config_id") in set(expected_ids)]
    if suffix_rows != rows:
        raise PretrainQueueError(
            "Recovery suffix contains a pre-existing run registration beyond the reviewed invalid attempt."
        )
    row = rows[0]
    inventory_sha256 = _attempt_inventory_sha256(failed_run_dir)
    predecessor_sha256 = hashlib.sha256(predecessor_bytes).hexdigest()
    attempt = int(failed_run_dir.name.split("-", 1)[0])
    next_attempt = attempt + 1
    marker = (
        f"{REVIEWED_RETRY_AUTHORIZATION_MARKER} failed_attempt={attempt:03d} "
        f"authorized_next_attempt={next_attempt:03d} inventory_sha256={inventory_sha256} "
        f"predecessor_queue_id={queue_id} predecessor_queue_sha256={predecessor_sha256}"
    )
    stdout_log = Path(str(failed_item.get("stdout_log") or ""))
    stderr_log = Path(str(failed_item.get("stderr_log") or ""))
    stdout_log = _absolute_path(stdout_log)
    stderr_log = _absolute_path(stderr_log)
    _require_no_link_like_components(stdout_log, "predecessor stdout log")
    _require_no_link_like_components(stderr_log, "predecessor stderr log")
    _require_regular_file(stdout_log, "predecessor stdout log")
    _require_regular_file(stderr_log, "predecessor stderr log")
    stdout_sha256 = _sha256_file(stdout_log)
    stderr_sha256 = _sha256_file(stderr_log)
    marker += f" stdout_sha256={stdout_sha256} stderr_sha256={stderr_sha256}"
    marker += f" terminated_pid={terminated_pid}"
    expected_result = repository / str(row.get("result_path") or "")
    if not _same_file(failed_run_dir, expected_result):
        raise PretrainQueueError("Reviewed retry registry result path differs from the retained attempt.")
    authoritative_result = authoritative_repository / str(row.get("result_path") or "")
    if not authoritative_result.is_dir() or not _same_file(
        failed_run_dir, authoritative_result
    ):
        raise PretrainQueueError(
            "Reviewed retry result does not resolve to the exact authoritative predecessor repository."
        )
    marker_fields = _parse_retry_authorization(str(row.get("notes") or ""))
    if (
        type(row.get("attempt")) is not int
        or row.get("attempt") != attempt
        or row.get("mode") != "pretrain"
        or row.get("lifecycle_status") != "partial"
        or row.get("evidence_status") != "invalid"
        or row.get("canonical") is not False
        or row.get("finished_at") is not None
        or row.get("git_dirty") is not False
        or row.get("failure_type") != "infrastructure_operator_process_misidentification"
        or not str(row.get("failure_message") or "").strip()
        or not str(row.get("reviewed_at") or "").strip()
        or row.get("artifact_manifest_uri") is not None
        or row.get("artifact_manifest_sha256") is not None
        or marker_fields["marker"] != marker
    ):
        raise PretrainQueueError("Reviewed retry registry row does not carry the exact authorization contract.")
    reviewed_at = _parse_timestamp(str(row["reviewed_at"]), "reviewed retry registry row")
    predecessor_finished_at = _parse_timestamp(
        str(predecessor["finished_at"]), "predecessor queue"
    )
    if reviewed_at < predecessor_finished_at:
        raise PretrainQueueError("Reviewed retry authorization predates the predecessor failure.")
    if (
        manifest.get("status") != "running"
        or manifest.get("mode") != "pretrain"
        or manifest.get("config_id") != config_id
        or manifest.get("run_id") != failed_run_dir.name
        or type(manifest.get("run_sequence")) is not int
        or manifest.get("run_sequence") != attempt
        or manifest.get("git_commit") != row.get("git_commit")
        or manifest.get("git_dirty") is not False
    ):
        raise PretrainQueueError("Reviewed retry running manifest differs from its registry authorization.")
    _verify_saved_config(failed_run_dir, loaded_configs[config_id])

    expected_attempts: dict[str, int] = {}
    experiment_guards: list[dict[str, Any]] = []
    for index, config_path_text in enumerate(config_paths):
        config_path = Path(config_path_text)
        config = loaded_configs[config_path.stem]
        experiment_dir = _experiment_dir(config, config_path, repository)
        attempts = sorted(_attempt_directories(experiment_dir), key=lambda path: path.name)
        expected_existing = [failed_run_dir] if index == 0 else []
        if len(attempts) != len(expected_existing) or any(
            not _same_file(actual, expected)
            for actual, expected in zip(attempts, expected_existing, strict=True)
        ):
            raise PretrainQueueError(
                f"Recovery preflight found an unexpected attempt inventory for {config_path.stem}."
            )
        if index == 0 and not _same_file(failed_run_dir.parent, experiment_dir):
            raise PretrainQueueError(
                "Retained fail-stop attempt is not inside the failed config's exact experiment directory."
            )
        authoritative_experiment = (
            authoritative_repository / "results" / config_path.stem
        )
        _require_no_link_like_components(
            authoritative_experiment, f"authoritative experiment {config_path.stem}"
        )
        _require_no_link_like_components(
            experiment_dir.parent, f"runner output root {config_path.stem}"
        )
        if not authoritative_experiment.is_dir() or not _same_file(
            experiment_dir, authoritative_experiment
        ):
            raise PretrainQueueError(
                f"Runner experiment does not resolve to the authoritative result target: {config_path.stem}."
            )
        external_runner = not _same_file(repository, authoritative_repository)
        if external_runner and not _is_link_like(experiment_dir):
            raise PretrainQueueError(
                f"External runner experiment is not the reviewed junction: {config_path.stem}."
            )
        experiment_guards.append(
            {
                "config_id": config_path.stem,
                "runner_experiment_path": str(_absolute_path(experiment_dir)),
                "authoritative_experiment_path": str(
                    _absolute_path(authoritative_experiment)
                ),
                "external_runner": external_runner,
            }
        )
        expected_attempts[config_path.stem] = next_attempt if index == 0 else 1
        protected_paths.extend(
            [
                (_absolute_path(experiment_dir), f"experiment directory {config_path.stem}"),
                (_absolute_path(experiment_dir.parent), f"output root {config_path.stem}"),
            ]
        )

    for item in items:
        for field in ("stdout_log", "stderr_log"):
            if item.get(field):
                protected_paths.append(
                    (_absolute_path(Path(str(item[field]))), f"predecessor {field}")
                )
        if item.get("run_dir"):
            protected_paths.append(
                (_absolute_path(Path(str(item["run_dir"]))), "predecessor run directory")
            )
    protected_paths.extend(
        [
            (config_registry_path, "config registry"),
            (run_registry_path, "run registry"),
        ]
    )
    _validate_recovery_destinations(
        state_path=state_path,
        logs_dir=logs_dir,
        predecessor_state_path=predecessor_path,
        protected_paths=protected_paths,
    )
    run_registry_sha256 = hashlib.sha256(run_registry_path.read_bytes()).hexdigest()
    config_registry_sha256 = hashlib.sha256(config_registry_path.read_bytes()).hexdigest()
    lineage = {
        "predecessor_state_path": str(predecessor_path),
        "predecessor_queue_id": queue_id,
        "predecessor_state_sha256": predecessor_sha256,
        "predecessor_failed_index": failed_index,
        "predecessor_failed_config_id": config_id,
        "predecessor_failed_run_id": failed_run_dir.name,
        "reviewed_retry_run_id": run_id,
        "launch_git_commit": launch_commit,
        "recovery_git_commit": current_head,
        "launch_science_blob_oids": {
            path: _git_blob_oid(repository, launch_commit, path)
            for path in RECOVERY_SCIENCE_CODE_PATHS
        },
        "reviewed_recovery_code_blob_oids": {
            path: {
                "launch": _git_blob_oid(repository, launch_commit, path),
                "recovery": _git_blob_oid(repository, current_head, path),
            }
            for path in RECOVERY_REVIEWED_CODE_PATHS
        },
        "config_guards": config_guards,
        "experiment_guards": experiment_guards,
        "authorization_config_registry_sha256": config_registry_sha256,
        "authorization_run_registry_sha256": run_registry_sha256,
        "predecessor_stdout_sha256": stdout_sha256,
        "predecessor_stderr_sha256": stderr_sha256,
        "retained_attempt_inventory_sha256": inventory_sha256,
        "retained_config_sha256": _sha256_file(failed_run_dir / "config.yaml"),
        "retained_manifest_sha256": _sha256_file(failed_run_dir / "manifest.json"),
        "retained_events_sha256": _sha256_file(failed_run_dir / "events.jsonl"),
        "terminated_pid": terminated_pid,
        "expected_new_attempts": expected_attempts,
    }
    authorization = {
        "run_dir": str(failed_run_dir.resolve()),
        "inventory_sha256": inventory_sha256,
        "manifest_sha256": _sha256_file(failed_run_dir / "manifest.json"),
        "config_id": config_id,
        "run_id": failed_run_dir.name,
        "config_registry_path": str(
            (repository / "docs" / "experimental-design" / "config-registry.yaml").resolve()
        ),
        "config_registry_sha256": config_registry_sha256,
        "run_registry_path": str(run_registry_path.resolve()),
        "run_registry_sha256": run_registry_sha256,
    }
    immutable = {
        "repository": str(repository),
        "recovery_git_commit": current_head,
        "recovery_code_blob_oids": {
            path: _git_blob_oid(repository, current_head, path)
            for path in RECOVERY_REVIEWED_CODE_PATHS
        },
        "terminated_pid": terminated_pid,
        "predecessor_state_path": str(predecessor_path),
        "predecessor_state_sha256": predecessor_sha256,
        "stdout_log_path": str(stdout_log),
        "stdout_log_sha256": stdout_sha256,
        "stderr_log_path": str(stderr_log),
        "stderr_log_sha256": stderr_sha256,
        "config_guards": config_guards,
        "experiment_guards": experiment_guards,
        "authorization": authorization,
    }
    recovery = {
        "state": lineage,
        "authorized_running": {(config_id, failed_run_dir.name): authorization},
        "immutable": immutable,
    }
    _revalidate_reviewed_recovery(recovery)
    return recovery


def _revalidate_reviewed_recovery(recovery: Mapping[str, Any]) -> None:
    immutable = recovery.get("immutable")
    if not isinstance(immutable, Mapping):
        raise PretrainQueueError("Reviewed retry immutable guard is malformed.")
    terminated_pid = immutable.get("terminated_pid")
    if type(terminated_pid) is not int or terminated_pid <= 0:
        raise PretrainQueueError("Reviewed retry terminated PID guard is malformed.")
    if _process_exists(terminated_pid):
        raise PretrainQueueError(
            f"Reviewed terminated PID {terminated_pid} reappeared; retry is unsafe."
        )

    repository = Path(str(immutable.get("repository") or ""))
    expected_commit = immutable.get("recovery_git_commit")
    if _git_head(repository) != expected_commit:
        raise PretrainQueueError("Recovery Git commit changed after retry authorization.")
    recovery_code_oids = immutable.get("recovery_code_blob_oids")
    if not isinstance(recovery_code_oids, Mapping) or any(
        _git_blob_oid(repository, str(expected_commit), path) != oid
        for path, oid in recovery_code_oids.items()
    ):
        raise PretrainQueueError("Reviewed recovery queue/CLI code changed after authorization.")

    guarded_files = (
        ("predecessor_state_path", "predecessor_state_sha256", "predecessor queue state"),
        ("stdout_log_path", "stdout_log_sha256", "predecessor stdout log"),
        ("stderr_log_path", "stderr_log_sha256", "predecessor stderr log"),
    )
    for path_key, sha_key, label in guarded_files:
        path = Path(str(immutable.get(path_key) or ""))
        _require_no_link_like_components(path, label)
        _require_regular_file(path, label)
        if _sha256_file(path) != immutable.get(sha_key):
            raise PretrainQueueError(f"{label.capitalize()} changed after retry authorization.")

    config_guards = immutable.get("config_guards")
    if not isinstance(config_guards, list) or not config_guards:
        raise PretrainQueueError("Reviewed retry config guards are malformed.")
    for guard in config_guards:
        if not isinstance(guard, Mapping):
            raise PretrainQueueError("Reviewed retry config guard is malformed.")
        path = Path(str(guard.get("path") or ""))
        _require_no_link_like_components(path, "recovery config")
        _require_regular_file(path, "recovery config")
        if _sha256_file(path) != guard.get("sha256"):
            raise PretrainQueueError(
                f"Recovery config changed after authorization: {guard.get('config_id')}."
            )

    experiment_guards = immutable.get("experiment_guards")
    if not isinstance(experiment_guards, list) or not experiment_guards:
        raise PretrainQueueError("Reviewed retry experiment guards are malformed.")
    for guard in experiment_guards:
        if not isinstance(guard, Mapping):
            raise PretrainQueueError("Reviewed retry experiment guard is malformed.")
        runner_experiment = Path(str(guard.get("runner_experiment_path") or ""))
        authoritative_experiment = Path(
            str(guard.get("authoritative_experiment_path") or "")
        )
        _require_no_link_like_components(
            authoritative_experiment,
            f"authoritative experiment {guard.get('config_id')}",
        )
        _require_no_link_like_components(
            runner_experiment.parent,
            f"runner output root {guard.get('config_id')}",
        )
        if not runner_experiment.is_dir() or not authoritative_experiment.is_dir():
            raise PretrainQueueError(
                f"Reviewed experiment binding disappeared: {guard.get('config_id')}."
            )
        if not _same_file(runner_experiment, authoritative_experiment):
            raise PretrainQueueError(
                f"Reviewed experiment junction target changed: {guard.get('config_id')}."
            )
        if guard.get("external_runner") is True and not _is_link_like(runner_experiment):
            raise PretrainQueueError(
                f"Reviewed experiment junction was replaced: {guard.get('config_id')}."
            )

    authorization = immutable.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PretrainQueueError("Reviewed retry authorization guard is malformed.")
    run_dir = Path(str(authorization.get("run_dir") or ""))
    manifest = _read_json_mapping(run_dir / "manifest.json", "reviewed retry manifest")
    _verify_authorized_running_attempt(
        run_dir=run_dir,
        manifest=manifest,
        authorization=authorization,
    )


def _parse_retry_authorization(notes: str) -> dict[str, str]:
    matches = list(_RETRY_MARKER_RE.finditer(notes))
    if len(matches) != 1 or any(notes.count(key) != 1 for key in _RETRY_MARKER_KEYS):
        raise PretrainQueueError(
            "Reviewed retry notes must contain one exact singleton authorization marker."
        )
    match = matches[0]
    fields = {key: str(value) for key, value in match.groupdict().items()}
    fields["marker"] = match.group(1)
    return fields


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PretrainQueueError(f"{label.capitalize()} timestamp is malformed.") from error
    if parsed.tzinfo is None:
        raise PretrainQueueError(f"{label.capitalize()} timestamp is not timezone-aware.")
    return parsed


def _git_head(repository: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PretrainQueueError("Cannot resolve the recovery Git commit.") from error
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PretrainQueueError("Recovery Git commit is malformed.")
    return commit


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise PretrainQueueError("Cannot verify recovery Git ancestry.") from error
    if completed.returncode not in (0, 1):
        raise PretrainQueueError("Cannot verify recovery Git ancestry.")
    return completed.returncode == 0


def _git_file_bytes(repository: Path, commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repository,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PretrainQueueError(
            f"Cannot read {relative_path} from launch commit {commit}."
        ) from error
    return completed.stdout


def _git_blob_oid(repository: Path, commit: str, relative_path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", f"{commit}:{relative_path}"],
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PretrainQueueError(
            f"Cannot resolve science blob {relative_path} at {commit}."
        ) from error
    oid = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", oid) is None:
        raise PretrainQueueError(f"Science blob id is malformed: {relative_path} at {commit}.")
    return oid


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _relative_to_repository(path: Path, repository: Path, label: str) -> Path:
    try:
        return _absolute_path(path).relative_to(_absolute_path(repository))
    except ValueError as error:
        raise PretrainQueueError(f"{label.capitalize()} is outside the recovery repository: {path}.") from error


def _require_no_link_like_components(path: Path, label: str) -> None:
    current = _absolute_path(path)
    while True:
        if current.is_symlink() or _is_link_like(current):
            raise PretrainQueueError(f"{label.capitalize()} has a link-like path component: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or _is_link_like(path):
        raise PretrainQueueError(f"{label.capitalize()} is missing or not a regular file: {path}.")
    try:
        link_count = path.stat().st_nlink
    except OSError as error:
        raise PretrainQueueError(f"Cannot inspect {label}: {path}.") from error
    if link_count != 1:
        raise PretrainQueueError(f"{label.capitalize()} is hard-linked and unsafe to authorize: {path}.")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(str(_absolute_path(left).resolve(strict=False)))
    right_text = os.path.normcase(str(_absolute_path(right).resolve(strict=False)))
    try:
        common = os.path.normcase(os.path.commonpath([left_text, right_text]))
    except ValueError:
        return False
    return common in (left_text, right_text)


def _validate_recovery_destinations(
    *,
    state_path: Path,
    logs_dir: Path,
    predecessor_state_path: Path,
    protected_paths: Sequence[tuple[Path, str]],
) -> None:
    runtime_parent = _absolute_path(predecessor_state_path).parent
    if state_path.parent != runtime_parent or logs_dir.parent != runtime_parent:
        raise PretrainQueueError(
            "Recovery state and logs must be fresh direct children of the predecessor queue directory."
        )
    _require_no_link_like_components(state_path, "recovery state path")
    _require_no_link_like_components(logs_dir, "recovery logs directory")
    if _paths_overlap(state_path, logs_dir):
        raise PretrainQueueError("Recovery state and logs paths must not overlap.")
    for protected_path, label in protected_paths:
        if _paths_overlap(state_path, protected_path) or _paths_overlap(logs_dir, protected_path):
            raise PretrainQueueError(
                f"Recovery state/log destination overlaps protected {label}: {protected_path}."
            )


def _verify_authorized_running_attempt(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    expected = Path(str(authorization["run_dir"]))
    config_registry_path = Path(str(authorization["config_registry_path"]))
    run_registry_path = Path(str(authorization["run_registry_path"]))
    _require_no_link_like_components(config_registry_path, "config registry")
    _require_no_link_like_components(run_registry_path, "run registry")
    _require_regular_file(config_registry_path, "config registry")
    _require_regular_file(run_registry_path, "run registry")
    if not _same_file(run_dir, expected):
        raise PretrainQueueError("Reviewed retry authorization resolved to a different run directory.")
    if (
        manifest.get("status") != "running"
        or manifest.get("mode") != "pretrain"
        or manifest.get("config_id") != authorization.get("config_id")
        or manifest.get("run_id") != authorization.get("run_id")
        or _sha256_file(run_dir / "manifest.json") != authorization.get("manifest_sha256")
        or _attempt_inventory_sha256(run_dir) != authorization.get("inventory_sha256")
        or _sha256_file(config_registry_path)
        != authorization.get("config_registry_sha256")
        or _sha256_file(run_registry_path)
        != authorization.get("run_registry_sha256")
    ):
        raise PretrainQueueError("Reviewed retry attempt changed after authorization.")


def _attempt_inventory_sha256(run_dir: Path) -> str:
    if not run_dir.is_dir() or _is_link_like(run_dir):
        raise PretrainQueueError(f"Reviewed retry attempt is missing or link-like: {run_dir}.")
    entries = sorted(run_dir.iterdir(), key=lambda value: value.name)
    if [path.name for path in entries] != ["config.yaml", "events.jsonl", "manifest.json"]:
        raise PretrainQueueError(
            "Reviewed running partial must retain exactly config.yaml, events.jsonl, and manifest.json."
        )
    files: list[dict[str, Any]] = []
    for path in entries:
        if _is_link_like(path):
            raise PretrainQueueError(f"Reviewed retry attempt contains a link: {path}.")
        if not path.is_file():
            raise PretrainQueueError(f"Reviewed retry attempt contains a non-regular entry: {path}.")
        if path.stat().st_nlink != 1:
            raise PretrainQueueError(f"Reviewed retry attempt contains a hard link: {path}.")
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise PretrainQueueError(f"Reviewed retry attempt has no durable files: {run_dir}.")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            open_process.restype = ctypes.c_void_p
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [ctypes.c_void_p]
            close_handle.restype = ctypes.c_int
            handle = open_process(0x1000, 0, pid)
            if handle:
                close_handle(handle)
                return True
            error_code = ctypes.get_last_error()
        except (AttributeError, OSError) as error:
            raise PretrainQueueError(f"Cannot inspect reviewed terminated PID {pid}.") from error
        if error_code == 87:  # ERROR_INVALID_PARAMETER: no process with this PID.
            return False
        if error_code == 5:  # ERROR_ACCESS_DENIED: conservatively treat as alive.
            return True
        raise PretrainQueueError(
            f"Cannot determine whether reviewed terminated PID {pid} exists (WinError {error_code})."
        )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    if path.is_symlink() or os.path.islink(path):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PretrainQueueError(f"Cannot read {label}: {path}.") from error
    if not isinstance(value, dict):
        raise PretrainQueueError(f"{label.capitalize()} is not an object: {path}.")
    return value


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise PretrainQueueError(f"Cannot read {label}: {path}.") from error
    if not isinstance(value, dict):
        raise PretrainQueueError(f"{label.capitalize()} is not an object: {path}.")
    return value


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
