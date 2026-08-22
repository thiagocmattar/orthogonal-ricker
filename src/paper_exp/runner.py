"""Sequential, fail-stop execution for ordered pretraining configs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Any, TextIO
from uuid import uuid4

import yaml

from paper_exp.config import load_config, validate_training_config
from paper_exp.data import metadata_matches_config, verify_token_cache
from paper_exp.run import CORE_RUN_ARTIFACTS, make_experiment_id
from paper_exp.utils import collect_git_commit, collect_git_dirty, read_json


RUNNER_SCHEMA_VERSION = 1
DEFAULT_STATE_PATH = Path("run-logs/runner-state.json")
DEFAULT_LOGS_DIR = Path("run-logs")
_FINISHED_ITEM_STATUSES = frozenset({"completed", "skipped"})
_IS_WINDOWS = sys.platform == "win32"


class RunnerError(RuntimeError):
    """Raised when ordered execution cannot continue safely."""


@dataclass(frozen=True)
class _PreparedConfig:
    path: Path
    display_path: str
    config_id: str
    config: dict[str, Any]
    experiment_dir: Path
    existing_run_names: frozenset[str]
    completed_run: Path | None
    planned_steps: int
    max_wall_seconds: float | None


@dataclass
class _ChildProcess:
    process: subprocess.Popen[Any]
    stdout_handle: TextIO
    stderr_handle: TextIO

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> int | None:
        return self.process.poll()

    def close(self) -> None:
        self.stdout_handle.close()
        self.stderr_handle.close()


def run_configs(
    config_paths: Sequence[str | Path],
    *,
    state_path: str | Path = DEFAULT_STATE_PATH,
    logs_dir: str | Path = DEFAULT_LOGS_DIR,
    repository: str | Path | None = None,
    poll_seconds: float = 1.0,
    command: str | None = None,
) -> dict[str, Any]:
    """Preflight and execute pretraining configs one at a time in order."""

    if not config_paths:
        raise RunnerError("run-configs requires at least one --config.")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0.0:
        raise RunnerError("poll_seconds must be finite and positive.")

    repository_path = (
        Path(repository).resolve()
        if repository is not None
        else _git_root(Path.cwd())
    )
    state_file = _resolve_runtime_path(state_path, repository_path)
    log_root = _resolve_runtime_path(logs_dir, repository_path)
    lock_path = repository_path / "tmp" / "experiment-runner.lock"

    with _exclusive_lock(lock_path) as runner_token:
        if state_file.exists():
            raise RunnerError(
                f"Runner state already exists; preserve it and choose a new --state path: "
                f"{state_file}"
            )
        _require_clean_git_tree(repository_path)
        _require_reviewed_plan(repository_path)
        launch_git_commit = _require_git_commit(repository_path)
        prepared = _preflight_configs(
            config_paths,
            repository_path,
            launch_git_commit=launch_git_commit,
        )
        state = _initial_state(
            prepared,
            repository=repository_path,
            state_path=state_file,
            logs_dir=log_root,
            command=command,
            git_commit=launch_git_commit,
        )
        _refresh_summary(state)
        _write_state(state_file, state)

        try:
            for index, selected in enumerate(prepared):
                item = state["items"][index]
                _require_clean_git_tree(repository_path)
                current_git_commit = _require_git_commit(repository_path)
                if current_git_commit != launch_git_commit:
                    raise RunnerError(
                        "Repository HEAD changed after runner preflight; refusing to continue."
                    )
                if selected.completed_run is not None:
                    _populate_completed_item(
                        item,
                        selected.completed_run,
                        status="skipped",
                    )
                    _refresh_summary(state)
                    _write_state(state_file, state)
                    continue

                _run_one_config(
                    state,
                    item,
                    selected,
                    repository=repository_path,
                    state_path=state_file,
                    logs_dir=log_root,
                    poll_seconds=poll_seconds,
                    runner_token=runner_token,
                )

            state["status"] = "completed"
            state["finished_at"] = _utc_now()
            _refresh_summary(state)
            _write_state(state_file, state)
            return state
        except BaseException as error:
            state["status"] = "failed"
            state["finished_at"] = _utc_now()
            state["failure"] = {
                "type": type(error).__qualname__,
                "message": str(error),
            }
            _refresh_summary(state)
            _write_state(state_file, state)
            raise


def read_runner_status(
    state_path: str | Path = DEFAULT_STATE_PATH,
    *,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Return refreshed runner status without modifying state or run artifacts."""

    repository_path = Path(repository).resolve() if repository is not None else Path.cwd().resolve()
    state_file = _resolve_runtime_path(state_path, repository_path)
    try:
        loaded = read_json(state_file)
    except (OSError, UnicodeError, ValueError) as error:
        raise RunnerError(f"Cannot read runner state: {state_file}") from error
    if not isinstance(loaded, dict) or loaded.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise RunnerError(f"Unsupported or malformed runner state: {state_file}")
    items = loaded.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise RunnerError(f"Runner state has a malformed item ledger: {state_file}")

    state = deepcopy(loaded)
    for item in state["items"]:
        if item.get("status") == "running":
            _refresh_item_from_artifacts(item)
            _refresh_item_timing(item)
    state["runner_process_alive"] = _process_exists(state.get("runner_pid"))
    _refresh_summary(state)
    return state


def format_runner_status(state: Mapping[str, Any]) -> str:
    """Format one concise status update including progress and ETC."""

    progress = state.get("progress") if isinstance(state.get("progress"), Mapping) else {}
    finished = int(progress.get("finished_configs") or 0)
    total = int(progress.get("total_configs") or len(state.get("items", [])))
    lines = [
        f"Runner {state.get('status', 'unknown')}: {finished}/{total} config(s) finished."
    ]
    if "runner_process_alive" in state:
        lines.append(f"Runner process alive: {state.get('runner_process_alive')}.")
    for label, key in (
        ("Completed", "completed_configs"),
        ("Failed", "failed_configs"),
        ("Remaining", "remaining_configs"),
    ):
        names = progress.get(key)
        if isinstance(names, list) and names:
            lines.append(f"{label}: {', '.join(str(name) for name in names)}.")
    active_config = progress.get("active_config")
    if active_config:
        active_pid = progress.get("active_pid")
        active_run_dir = progress.get("active_run_dir")
        current_step = progress.get("current_step")
        planned_steps = progress.get("planned_steps")
        step_text = "progress unavailable"
        if isinstance(current_step, int) and isinstance(planned_steps, int):
            percentage = 100.0 * current_step / planned_steps if planned_steps else 0.0
            step_text = f"step {current_step:,}/{planned_steps:,} ({percentage:.1f}%)"
        process_text = f"PID {active_pid}" if isinstance(active_pid, int) else "PID unavailable"
        run_text = f"run {active_run_dir}" if active_run_dir else "run not discovered yet"
        tokens_seen = progress.get("tokens_seen")
        token_text = (
            f"; {tokens_seen:,} tokens" if isinstance(tokens_seen, int) else ""
        )
        lines.append(
            f"Active: {active_config}; {process_text}; {run_text}; {step_text}{token_text}."
        )
        latest_parts: list[str] = []
        latest_event_at = progress.get("latest_event_at")
        if latest_event_at:
            latest_parts.append(f"event {latest_event_at}")
        for label, key in (
            ("task loss", "latest_task_loss"),
            ("pressure loss", "latest_pressure_loss"),
            ("augmented loss", "latest_augmented_loss"),
        ):
            value = progress.get(key)
            if isinstance(value, int | float) and math.isfinite(float(value)):
                latest_parts.append(f"{label} {float(value):.6g}")
        throughput = progress.get("recent_tokens_per_second")
        if isinstance(throughput, int | float) and math.isfinite(float(throughput)):
            latest_parts.append(f"recent throughput {float(throughput):,.1f} tokens/s")
        if latest_parts:
            lines.append("Latest: " + "; ".join(latest_parts) + ".")

        active_remaining = progress.get("active_estimated_remaining_seconds")
        active_completion = progress.get("active_estimated_completion_at")
        if isinstance(active_remaining, int | float) and math.isfinite(
            float(active_remaining)
        ):
            lines.append(
                f"Active-run estimate: {_format_duration(float(active_remaining))}; "
                f"ETC: {active_completion}."
            )

    remaining = progress.get("estimated_remaining_seconds")
    completion = progress.get("estimated_completion_at")
    if isinstance(remaining, int | float) and math.isfinite(float(remaining)):
        lines.append(
            f"Queue estimate: {_format_duration(float(remaining))}; ETC: {completion}."
        )
    elif state.get("status") == "completed":
        lines.append("Queue estimate: 0s; experiment complete.")
    else:
        lines.append("Queue estimate and ETC: unknown until measurable progress exists.")
    rate = progress.get("seconds_per_step")
    if isinstance(rate, int | float) and math.isfinite(float(rate)):
        basis = progress.get("rate_basis") or "observed runs"
        lines.append(
            f"Rate basis: {float(rate):.3f} s/step from {basis}; uncertainty is not quantified."
        )
    return "\n".join(lines)


def _preflight_configs(
    config_paths: Sequence[str | Path],
    repository: Path,
    *,
    launch_git_commit: str,
) -> list[_PreparedConfig]:
    normalized = [_resolve_config_path(path, repository) for path in config_paths]
    if len(set(normalized)) != len(normalized):
        raise RunnerError("run-configs contains duplicate config paths.")
    config_ids = [make_experiment_id(path) for path in normalized]
    if len(set(config_ids)) != len(config_ids):
        raise RunnerError("run-configs contains duplicate config IDs.")

    prepared: list[_PreparedConfig] = []
    for path in normalized:
        _require_tracked_config(repository, path)
        config = load_config(path, allow_todos=False)
        validate_training_config(config)
        training = config.get("training")
        if not isinstance(training, dict):
            raise RunnerError(f"Pretraining config has no training section: {path}")
        planned_steps = _positive_int(training.get("max_steps"), "training.max_steps", path)
        max_wall_seconds = _optional_positive_number(
            training.get("max_wall_seconds"),
            "training.max_wall_seconds",
            path,
        )
        config_id = make_experiment_id(path)
        preflight_token_caches(
            config,
            repository=repository,
            source=path,
        )
        output_root = require_results_output(config, repository=repository, source=path)
        experiment_dir = output_root / config_id
        attempts = _attempt_directories(experiment_dir)
        completed = _inspect_attempts(
            attempts,
            config=config,
            config_id=config_id,
            repository=repository,
            launch_git_commit=launch_git_commit,
        )
        prepared.append(
            _PreparedConfig(
                path=path,
                display_path=_display_path(path, repository),
                config_id=config_id,
                config=config,
                experiment_dir=experiment_dir,
                existing_run_names=frozenset(run.name for run in attempts),
                completed_run=completed,
                planned_steps=planned_steps,
                max_wall_seconds=max_wall_seconds,
            )
        )
    return prepared


def _inspect_attempts(
    attempts: Sequence[Path],
    *,
    config: dict[str, Any],
    config_id: str,
    repository: Path,
    launch_git_commit: str,
) -> Path | None:
    completed: list[Path] = []
    for run_dir in attempts:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RunnerError(f"Incomplete run attempt has no manifest: {run_dir}")
        try:
            manifest = read_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as error:
            raise RunnerError(f"Cannot read run manifest: {manifest_path}") from error
        if not isinstance(manifest, dict):
            raise RunnerError(f"Run manifest is not an object: {manifest_path}")
        if manifest.get("config_id") != config_id or manifest.get("run_id") != run_dir.name:
            raise RunnerError(f"Run manifest identity does not match its directory: {run_dir}")
        _verify_saved_config(run_dir, config)
        status = manifest.get("status")
        if status == "running":
            raise RunnerError(
                f"Existing attempt is still marked running; inspect it before relaunch: {run_dir}"
            )
        if status == "completed":
            completed_git_commit = _verify_completed_run(
                run_dir,
                config=config,
                config_id=config_id,
                repository=repository,
            )
            if completed_git_commit == launch_git_commit:
                completed.append(run_dir)
            continue
        if status == "failed":
            if (
                not str(manifest.get("finished_at") or "").strip()
                or not isinstance(manifest.get("failure"), dict)
            ):
                raise RunnerError(f"Failed attempt has inconsistent terminal metadata: {run_dir}")
            continue
        raise RunnerError(
            f"Run attempt has no supported terminal status and blocks relaunch: {run_dir}"
        )
    return completed[-1] if completed else None


def _verify_saved_config(run_dir: Path, config: dict[str, Any]) -> None:
    config_path = run_dir / "config.yaml"
    try:
        saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RunnerError(f"Cannot read saved config snapshot: {config_path}") from error
    if saved_config != config:
        raise RunnerError(f"Saved config does not match the queued config: {config_path}")


def _run_one_config(
    state: dict[str, Any],
    item: dict[str, Any],
    selected: _PreparedConfig,
    *,
    repository: Path,
    state_path: Path,
    logs_dir: Path,
    poll_seconds: float,
    runner_token: str,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    runner_label = str(state["runner_id"])[:8]
    item_number = int(item["index"]) + 1
    log_stem = f"{item_number:03d}-{selected.config_id}-{runner_label}"
    stdout_log = logs_dir / f"{log_stem}.log"
    stderr_log = logs_dir / f"{log_stem}.err.log"
    command = [
        sys.executable,
        "-m",
        "paper_exp.cli",
        "pretrain",
        "--config",
        selected.display_path,
        "--runner-token",
        runner_token,
    ]

    item.update(
        {
            "status": "running",
            "started_at": _utc_now(),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "child_command": command,
        }
    )
    state["status"] = "running"
    _refresh_summary(state)
    _write_state(state_path, state)

    child: _ChildProcess | Any | None = None
    try:
        child = _start_child(
            command,
            cwd=repository,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        item["pid"] = int(child.pid)
        _write_state(state_path, state)

        while True:
            _refresh_running_item(item, selected)
            _refresh_summary(state)
            _write_state(state_path, state)
            exit_code = child.poll()
            if exit_code is not None:
                break
            time.sleep(poll_seconds)

        item["exit_code"] = int(exit_code)
        _refresh_running_item(item, selected)
        if exit_code != 0:
            item["status"] = "failed"
            item["finished_at"] = _utc_now()
            item["error"] = f"Pretraining child exited with code {exit_code}."
            raise RunnerError(
                f"Pretraining child failed for {selected.display_path} with exit code "
                f"{exit_code}; see {stderr_log}."
            )

        run_dir_text = item.get("run_dir")
        if not isinstance(run_dir_text, str):
            raise RunnerError(
                f"Pretraining child created no discoverable run for {selected.display_path}."
            )
        run_dir = Path(run_dir_text)
        completed_git_commit = _verify_completed_run(
            run_dir,
            config=selected.config,
            config_id=selected.config_id,
            repository=repository,
        )
        if completed_git_commit != state["git_commit"]:
            raise RunnerError(
                f"Completed child launch Git commit differs from runner preflight: {run_dir}"
            )
        _populate_completed_item(item, run_dir, status="completed")
        _refresh_summary(state)
        _write_state(state_path, state)
    except BaseException:
        if item.get("status") == "running":
            item["status"] = "failed"
            item["finished_at"] = _utc_now()
        raise
    finally:
        if child is not None:
            child.close()


def _refresh_running_item(item: dict[str, Any], selected: _PreparedConfig) -> None:
    run_dir_text = item.get("run_dir")
    if not isinstance(run_dir_text, str):
        candidates = [
            run
            for run in _attempt_directories(selected.experiment_dir)
            if run.name not in selected.existing_run_names
        ]
        if len(candidates) > 1:
            raise RunnerError(
                f"Expected one new run for {selected.display_path}; found {len(candidates)}."
            )
        if candidates:
            item["run_dir"] = str(candidates[0])
            run_dir_text = str(candidates[0])

    if isinstance(run_dir_text, str):
        _refresh_item_from_artifacts(item)
    _refresh_item_timing(item)


def _refresh_item_timing(item: dict[str, Any]) -> None:
    started = _parse_timestamp(item.get("started_at"))
    if started is not None:
        item["elapsed_seconds"] = max(0.0, (_utc_datetime() - started).total_seconds())
    current_step = item.get("current_step")
    elapsed = item.get("elapsed_seconds")
    if isinstance(current_step, int) and current_step > 0 and isinstance(elapsed, int | float):
        item["seconds_per_step"] = float(elapsed) / current_step


def _refresh_item_from_artifacts(item: dict[str, Any]) -> None:
    run_dir_text = item.get("run_dir")
    if not isinstance(run_dir_text, str):
        return
    run_dir = Path(run_dir_text)
    latest_train = _latest_train_event(run_dir / "events.jsonl")
    if latest_train is not None:
        step = latest_train.get("step")
        tokens_seen = latest_train.get("tokens_seen")
        if isinstance(step, int) and not isinstance(step, bool):
            item["current_step"] = step
        if isinstance(tokens_seen, int) and not isinstance(tokens_seen, bool):
            item["tokens_seen"] = tokens_seen
        for item_key, event_key in (
            ("latest_task_loss", "task_loss"),
            ("latest_pressure_loss", "pressure_loss"),
            ("latest_augmented_loss", "augmented_loss"),
            ("latest_step_wall_seconds", "step_wall_seconds"),
        ):
            value = latest_train.get(event_key)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                item[item_key] = float(value)
        step_wall = item.get("latest_step_wall_seconds")
        if (
            isinstance(step, int)
            and step > 0
            and isinstance(tokens_seen, int)
            and tokens_seen > 0
            and isinstance(step_wall, int | float)
            and float(step_wall) > 0.0
        ):
            item["recent_tokens_per_second"] = (
                (tokens_seen / step) / float(step_wall)
            )
        try:
            item["latest_event_at"] = datetime.fromtimestamp(
                (run_dir / "events.jsonl").stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
        except OSError:
            pass

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (OSError, UnicodeError, ValueError):
            return
        if isinstance(manifest, dict):
            item["observed_run_status"] = manifest.get("status")
            if manifest.get("status") in {"completed", "failed"}:
                item["status"] = str(manifest["status"])
                if isinstance(manifest.get("finished_at"), str):
                    item["finished_at"] = manifest["finished_at"]


def _populate_completed_item(item: dict[str, Any], run_dir: Path, *, status: str) -> None:
    metrics = read_json(run_dir / "metrics.json")
    if not isinstance(metrics, dict):
        raise RunnerError(f"Completed metrics are not an object: {run_dir / 'metrics.json'}")
    completed_steps = metrics.get("training/optimizer_steps")
    planned_steps = metrics.get("training/planned_optimizer_steps")
    elapsed = metrics.get("training/wall_seconds_total")
    item.update(
        {
            "status": status,
            "run_dir": str(run_dir),
            "finished_at": _utc_now(),
            "current_step": int(completed_steps) if isinstance(completed_steps, int) else item.get("current_step"),
            "completed_steps": int(completed_steps) if isinstance(completed_steps, int) else None,
        }
    )
    if isinstance(planned_steps, int):
        item["planned_steps"] = planned_steps
    if isinstance(elapsed, int | float) and math.isfinite(float(elapsed)) and float(elapsed) >= 0.0:
        item["elapsed_seconds"] = float(elapsed)
        if isinstance(completed_steps, int) and completed_steps > 0:
            item["seconds_per_step"] = float(elapsed) / completed_steps


def _verify_completed_run(
    run_dir: Path,
    *,
    config: dict[str, Any],
    config_id: str,
    repository: Path,
) -> str:
    required_artifacts = (*CORE_RUN_ARTIFACTS, "events.jsonl")
    missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
    if missing:
        raise RunnerError(
            f"Completed run is missing required artifacts ({', '.join(missing)}): {run_dir}"
        )
    try:
        manifest = read_json(run_dir / "manifest.json")
        saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = read_json(run_dir / "metrics.json")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise RunnerError(f"Cannot verify completed run: {run_dir}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise RunnerError(f"Run does not have a completed terminal manifest: {run_dir}")
    if not str(manifest.get("finished_at") or "").strip():
        raise RunnerError(f"Completed run has no terminal timestamp: {run_dir}")
    if manifest.get("config_id") != config_id or manifest.get("run_id") != run_dir.name:
        raise RunnerError(f"Completed run identity is inconsistent: {run_dir}")
    if manifest.get("mode") != "pretrain":
        raise RunnerError(f"Completed run is not a pretraining run: {run_dir}")
    git_commit = str(manifest.get("git_commit") or "").strip()
    if manifest.get("git_dirty") is not False or re.fullmatch(r"[0-9a-f]{40,64}", git_commit) is None:
        raise RunnerError(f"Completed run lacks clean launch Git provenance: {run_dir}")
    if saved_config != config:
        raise RunnerError(
            f"Completed run config differs from the queued immutable config: {run_dir}"
        )
    if not isinstance(metrics, dict):
        raise RunnerError(f"Completed run metrics are not an object: {run_dir}")
    events = _read_jsonl_objects(run_dir / "events.jsonl")
    _read_jsonl_objects(run_dir / "predictions.jsonl")
    planned = config.get("training", {}).get("max_steps")
    completed_steps = metrics.get("training/optimizer_steps")
    planned_steps = metrics.get("training/planned_optimizer_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or completed_steps > planned
    ):
        raise RunnerError(f"Completed-step metadata is invalid: {run_dir}")
    if planned_steps != planned:
        raise RunnerError(f"Planned-step metadata is inconsistent: {run_dir}")
    manifest_training = manifest.get("training")
    if (
        not isinstance(manifest_training, dict)
        or manifest_training.get("completed_steps") != completed_steps
    ):
        raise RunnerError(f"Completed-step metadata is inconsistent: {run_dir}")
    if config.get("training", {}).get("max_wall_seconds") is None and completed_steps != planned:
        raise RunnerError(
            f"Completed run did not finish its configured step budget: {run_dir}"
        )
    train_steps = [
        row.get("step")
        for row in events
        if row.get("event") == "train"
    ]
    if completed_steps > 0 and not train_steps:
        raise RunnerError(f"Completed run has no recorded training event: {run_dir}")
    if any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in train_steps):
        raise RunnerError(f"Completed run has invalid training-event steps: {run_dir}")
    if train_steps and (train_steps != sorted(set(train_steps)) or train_steps[-1] > completed_steps):
        raise RunnerError(f"Completed run has inconsistent training-event steps: {run_dir}")

    validation = config.get("validation")
    if isinstance(validation, dict) and validation.get("enabled", False):
        validation_loss = metrics.get("training/validation_loss_final")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, int | float)
            or not math.isfinite(float(validation_loss))
        ):
            raise RunnerError(f"Completed validation loss is not finite: {run_dir}")

    checkpoint = config.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("save_final", False):
        checkpoint_manifest = manifest.get("checkpoint")
        if not isinstance(checkpoint_manifest, dict) or checkpoint_manifest.get("saved") is not True:
            raise RunnerError(f"Completed final checkpoint is not marked saved: {run_dir}")
        checkpoint_path_text = str(checkpoint_manifest.get("path") or "").strip()
        if not checkpoint_path_text:
            raise RunnerError(f"Completed final checkpoint has no path: {run_dir}")
        checkpoint_path = Path(checkpoint_path_text)
        if not checkpoint_path.is_absolute():
            checkpoint_path = repository / checkpoint_path
        if not checkpoint_path.is_dir():
            raise RunnerError(f"Completed final checkpoint is missing: {checkpoint_path}")
        model_files = (
            checkpoint_path / "model.safetensors",
            checkpoint_path / "model.safetensors.index.json",
        )
        if not (checkpoint_path / "config.json").is_file() or not any(
            path.is_file() for path in model_files
        ):
            raise RunnerError(f"Completed final checkpoint is incomplete: {checkpoint_path}")
        if checkpoint.get("save_optimizer", False):
            if checkpoint_manifest.get("optimizer_saved") is not True:
                raise RunnerError(
                    f"Completed optimizer checkpoint is not marked saved: {run_dir}"
                )
            if not (checkpoint_path / "optimizer.pt").is_file():
                raise RunnerError(
                    f"Completed optimizer checkpoint is missing: {checkpoint_path}"
                )
    return git_commit


def _initial_state(
    prepared: Sequence[_PreparedConfig],
    *,
    repository: Path,
    state_path: Path,
    logs_dir: Path,
    command: str | None,
    git_commit: str,
) -> dict[str, Any]:
    created_at = _utc_now()
    runner_id = uuid4().hex
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "runner_id": runner_id,
        "runner_pid": os.getpid(),
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
        "repository": str(repository),
        "state_path": str(state_path),
        "logs_dir": str(logs_dir),
        "command": command,
        "git_commit": git_commit,
        "items": [
            {
                "index": index,
                "config_path": selected.display_path,
                "config_id": selected.config_id,
                "experiment_dir": str(selected.experiment_dir),
                "existing_run_names": sorted(selected.existing_run_names),
                "planned_steps": selected.planned_steps,
                "max_wall_seconds": selected.max_wall_seconds,
                "status": "pending",
            }
            for index, selected in enumerate(prepared)
        ],
    }


def _refresh_summary(state: dict[str, Any]) -> None:
    items = state.get("items") if isinstance(state.get("items"), list) else []
    now = _utc_datetime()
    active = next((item for item in items if item.get("status") == "running"), None)
    finished = sum(item.get("status") in _FINISHED_ITEM_STATUSES for item in items)
    seconds_per_step = _measured_seconds_per_step(items)
    remaining_seconds = _estimated_remaining_seconds(
        items,
        seconds_per_step=seconds_per_step,
        now=now,
    )
    completion = (
        (now + timedelta(seconds=remaining_seconds)).isoformat()
        if remaining_seconds is not None
        else None
    )
    active_remaining = (
        _estimated_item_remaining_seconds(
            active,
            fallback_seconds_per_step=seconds_per_step,
            now=now,
        )
        if active is not None
        else None
    )
    active_completion = (
        (now + timedelta(seconds=active_remaining)).isoformat()
        if active_remaining is not None
        else None
    )
    completed_configs = [
        str(item.get("config_path"))
        for item in items
        if item.get("status") in _FINISHED_ITEM_STATUSES
    ]
    failed_configs = [
        str(item.get("config_path"))
        for item in items
        if item.get("status") == "failed"
    ]
    remaining_configs = [
        str(item.get("config_path"))
        for item in items
        if item.get("status") in {"pending", "running"}
    ]
    rate_basis = None
    if active is not None and isinstance(active.get("seconds_per_step"), int | float):
        rate_basis = "the active run's elapsed-time average"
    elif seconds_per_step is not None:
        rate_basis = "the mean of available completed/observed run rates"
    state["updated_at"] = now.isoformat()
    state["progress"] = {
        "finished_configs": finished,
        "total_configs": len(items),
        "active_config": active.get("config_path") if active else None,
        "active_pid": active.get("pid") if active else None,
        "active_run_dir": active.get("run_dir") if active else None,
        "current_step": active.get("current_step") if active else None,
        "planned_steps": active.get("planned_steps") if active else None,
        "tokens_seen": active.get("tokens_seen") if active else None,
        "latest_event_at": active.get("latest_event_at") if active else None,
        "latest_task_loss": active.get("latest_task_loss") if active else None,
        "latest_pressure_loss": active.get("latest_pressure_loss") if active else None,
        "latest_augmented_loss": active.get("latest_augmented_loss") if active else None,
        "recent_tokens_per_second": (
            active.get("recent_tokens_per_second") if active else None
        ),
        "active_estimated_remaining_seconds": active_remaining,
        "active_estimated_completion_at": active_completion,
        "estimated_remaining_seconds": remaining_seconds,
        "estimated_completion_at": completion,
        "seconds_per_step": seconds_per_step,
        "rate_basis": rate_basis,
        "completed_configs": completed_configs,
        "failed_configs": failed_configs,
        "remaining_configs": remaining_configs,
    }


def _measured_seconds_per_step(items: Sequence[dict[str, Any]]) -> float | None:
    rates = [
        float(item["seconds_per_step"])
        for item in items
        if isinstance(item.get("seconds_per_step"), int | float)
        and math.isfinite(float(item["seconds_per_step"]))
        and float(item["seconds_per_step"]) > 0.0
    ]
    return sum(rates) / len(rates) if rates else None


def _estimated_remaining_seconds(
    items: Sequence[dict[str, Any]],
    *,
    seconds_per_step: float | None,
    now: datetime,
) -> float | None:
    total = 0.0
    for item in items:
        status = item.get("status")
        if status in _FINISHED_ITEM_STATUSES:
            continue
        if status == "failed":
            return None
        item_remaining = _estimated_item_remaining_seconds(
            item,
            fallback_seconds_per_step=seconds_per_step,
            now=now,
        )
        if item_remaining is None:
            return None
        total += item_remaining
    return total


def _estimated_item_remaining_seconds(
    item: Mapping[str, Any],
    *,
    fallback_seconds_per_step: float | None,
    now: datetime,
) -> float | None:
    status = item.get("status")
    planned_steps = item.get("planned_steps")
    if not isinstance(planned_steps, int) or planned_steps <= 0:
        return None
    current_step = item.get("current_step") if status == "running" else 0
    if not isinstance(current_step, int) or current_step < 0:
        current_step = 0
    item_rate = item.get("seconds_per_step")
    rate = (
        float(item_rate)
        if isinstance(item_rate, int | float)
        and math.isfinite(float(item_rate))
        and float(item_rate) > 0.0
        else fallback_seconds_per_step
    )
    max_wall = item.get("max_wall_seconds")
    wall_remaining: float | None = None
    if isinstance(max_wall, int | float) and math.isfinite(float(max_wall)):
        elapsed = 0.0
        started = _parse_timestamp(item.get("started_at")) if status == "running" else None
        if started is not None:
            elapsed = max(0.0, (now - started).total_seconds())
        wall_remaining = max(0.0, float(max_wall) - elapsed)
    step_remaining = rate * max(0, planned_steps - current_step) if rate is not None else None
    if wall_remaining is not None and step_remaining is not None:
        return min(wall_remaining, step_remaining)
    return wall_remaining if wall_remaining is not None else step_remaining


def _latest_train_event(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (FileNotFoundError, PermissionError, UnicodeError, OSError):
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("event") == "train":
            return row
    return None


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise RunnerError(f"Cannot read completed JSONL artifact: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise RunnerError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise RunnerError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _start_child(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> _ChildProcess:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_log.open("x", encoding="utf-8", newline="\n")
    try:
        stderr_handle = stderr_log.open("x", encoding="utf-8", newline="\n")
    except BaseException:
        stdout_handle.close()
        raise
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return _ChildProcess(process, stdout_handle, stderr_handle)


def _attempt_directories(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.is_dir():
        return []
    return sorted(path for path in experiment_dir.iterdir() if path.is_dir())


def _resolve_config_path(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve()
    config_root = (repository / "configs").resolve()
    try:
        relative = resolved.relative_to(config_root)
    except ValueError as error:
        raise RunnerError(
            f"Queued configs must be committed files under {config_root}: {resolved}"
        ) from error
    if relative.parent != Path("."):
        raise RunnerError(
            f"Queued configs must be direct children of {config_root}: {resolved}"
        )
    if not resolved.is_file():
        raise RunnerError(f"Queued config does not exist: {resolved}")
    return resolved


def resolve_launch_config(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve one tracked top-level config and its owning repository."""

    repository_path = (
        Path(repository).resolve()
        if repository is not None
        else _git_root(Path.cwd())
    )
    resolved = _resolve_config_path(path, repository_path)
    _require_tracked_config(repository_path, resolved)
    return repository_path, resolved


def resolve_launch_run_dir(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve one exact run directory beneath repository/results."""

    repository_path = (
        Path(repository).resolve()
        if repository is not None
        else _git_root(Path.cwd())
    )
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_path / candidate).resolve()
    )
    results_root = (repository_path / "results").resolve()
    try:
        relative = resolved.relative_to(results_root)
    except ValueError as error:
        raise RunnerError(f"Source run must be inside {results_root}: {resolved}") from error
    if len(relative.parts) != 2 or not resolved.is_dir():
        raise RunnerError(
            f"Source run must be an exact results/<config-id>/<run-id> directory: {resolved}"
        )
    return repository_path, resolved


def require_results_output(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    source: str | Path,
) -> Path:
    """Require the release artifact root promised by the repository contract."""

    repository_path = Path(repository).resolve()
    output = config.get("output")
    output_dir = output.get("dir") if isinstance(output, Mapping) else None
    candidate = Path(str(output_dir))
    output_root = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_path / candidate).resolve()
    )
    required = (repository_path / "results").resolve()
    if output_root != required:
        raise RunnerError(
            f"Config output.dir must resolve to {required}: {source}"
        )
    return output_root


def require_token_cache_output(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    source: str | Path,
) -> Path:
    """Require the portable token-cache root used by release configs."""

    repository_path = Path(repository).resolve()
    preprocessing = config.get("preprocessing")
    output_dir = (
        preprocessing.get("output_dir")
        if isinstance(preprocessing, Mapping)
        else None
    )
    candidate = Path(str(output_dir))
    output_root = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_path / candidate).resolve()
    )
    required = (repository_path / "data" / "tokenized").resolve()
    if output_root != required:
        raise RunnerError(
            f"Config preprocessing.output_dir must resolve to {required}: {source}"
        )
    return output_root


def preflight_token_caches(
    config: dict[str, Any],
    *,
    repository: Path,
    source: Path,
) -> None:
    cache_root = require_token_cache_output(
        config,
        repository=repository,
        source=source,
    )
    cache_dir = cache_root / str(config["preprocessing"]["cache_id"])
    train_metadata = _read_cache_metadata(
        cache_dir / "metadata.json",
        context=f"Training cache for {source}",
    )
    if not metadata_matches_config(
        train_metadata,
        config,
        split=config["data"]["split"],
        max_documents=config["data"]["max_documents"],
    ):
        raise RunnerError(f"Training cache metadata does not match queued config: {source}")
    _verify_cache_file(
        train_metadata,
        repository=repository,
        context=f"Training cache for {source}",
    )

    validation = config["validation"]
    if not validation["enabled"]:
        return
    validation_dir = cache_dir / "validation"
    if validation["partition"] in {"selection", "confirmation"}:
        validation_dir /= str(validation["partition"])
    validation_metadata = _read_cache_metadata(
        validation_dir / "metadata.json",
        context=f"Validation cache for {source}",
    )
    if not metadata_matches_config(
        validation_metadata,
        config,
        split=validation["split"],
        max_documents=validation["max_documents"],
        partition=validation["partition"],
        partition_seed=validation["partition_seed"],
    ):
        raise RunnerError(f"Validation cache metadata does not match queued config: {source}")
    expected_hash = validation["partition_hash"]
    if (
        expected_hash is not None
        and validation_metadata.get("source_document_indices_sha256") != expected_hash
    ):
        raise RunnerError(f"Validation partition hash does not match queued config: {source}")
    _verify_cache_file(
        validation_metadata,
        repository=repository,
        context=f"Validation cache for {source}",
    )


def _read_cache_metadata(path: Path, *, context: str) -> dict[str, Any]:
    try:
        metadata = read_json(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise RunnerError(f"{context} metadata is missing or unreadable: {path}") from error
    if not isinstance(metadata, dict):
        raise RunnerError(f"{context} metadata is not an object: {path}")
    return metadata


def _verify_cache_file(
    metadata: dict[str, Any],
    *,
    repository: Path,
    context: str,
) -> Path:
    normalized = dict(metadata)
    recorded = normalized.get("tokens_path")
    if isinstance(recorded, str) and recorded.strip():
        path = Path(recorded)
        if not path.is_absolute():
            normalized["tokens_path"] = str((repository / path).resolve())
    try:
        return verify_token_cache(normalized, context=context)
    except (OSError, ValueError) as error:
        raise RunnerError(str(error)) from error


def _require_tracked_config(repository: Path, config_path: Path) -> None:
    try:
        relative = config_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise RunnerError(f"Config is outside the repository: {config_path}") from error
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerError(
            f"Scientific configs must be tracked by Git before launch: {config_path}"
        ) from error


def _resolve_runtime_path(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repository / candidate).resolve()


def _display_path(path: Path, repository: Path) -> str:
    try:
        return path.relative_to(repository).as_posix()
    except ValueError:
        return str(path)


def _positive_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunnerError(f"{field} must be a positive integer in {source}.")
    return value


def _optional_positive_number(value: Any, field: str, source: Path) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunnerError(f"{field} must be a positive finite number in {source}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RunnerError(f"{field} must be a positive finite number in {source}.")
    return result


def _git_root(cwd: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerError("run-configs must execute inside a Git repository.") from error
    return Path(result.stdout.strip()).resolve()


def _require_clean_git_tree(repository: Path) -> None:
    dirty = collect_git_dirty(repository)
    if dirty is None:
        raise RunnerError("Cannot determine Git dirty state before launch.")
    if dirty:
        raise RunnerError(
            "Refusing to launch from a dirty Git worktree; commit or stash tracked changes first."
        )


def _require_git_commit(repository: Path) -> str:
    commit = collect_git_commit(repository)
    if commit is None or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise RunnerError("Cannot determine a valid launch Git commit before launch.")
    return commit


@contextmanager
def direct_launch_guard(
    *,
    runner_token: str | None = None,
    repository: str | Path | None = None,
) -> Iterator[None]:
    """Hold the experiment lock, or verify an owning runner's child token."""

    repository_path = (
        Path(repository).resolve()
        if repository is not None
        else _git_root(Path.cwd())
    )
    lock_path = repository_path / "tmp" / "experiment-runner.lock"
    if runner_token is not None:
        recorded_token = _lock_token(lock_path)
        if recorded_token is None or not secrets.compare_digest(
            recorded_token,
            runner_token,
        ):
            raise RunnerError(
                "Pretraining child token does not match the active sequential runner."
            )
        owner = _lock_owner(lock_path)
        if owner is None or _process_exists(owner) is not True:
            raise RunnerError("The owning sequential runner is not alive.")
        _require_clean_git_tree(repository_path)
        _require_reviewed_plan(repository_path)
        yield
        return

    with _exclusive_lock(lock_path):
        _require_clean_git_tree(repository_path)
        _require_reviewed_plan(repository_path)
        yield


def _require_reviewed_plan(repository: Path) -> None:
    plan_path = repository / "docs" / "experiment_plan.md"
    try:
        plan_text = plan_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise RunnerError(f"Cannot read the definitive experiment plan: {plan_path}") from error
    if re.search(r"(?im)^Plan status:\s*(?:reviewed|`reviewed`)\s*$", plan_text) is None:
        raise RunnerError(
            "Scientific launches are blocked until docs/experiment_plan.md declares "
            "`Plan status: reviewed`."
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = _lock_owner(path)
        if owner is None:
            raise RunnerError(
                f"Cannot verify the owner of existing runner lock {path}; inspect it manually."
            )
        owner_alive = _process_exists(owner)
        if owner_alive is not False:
            raise RunnerError(f"Another experiment runner owns {path} (PID {owner}).")
        try:
            path.unlink()
        except OSError as error:
            raise RunnerError(f"Cannot remove stale runner lock: {path}") from error
    try:
        token = uuid4().hex
        try:
            with path.open("x", encoding="ascii", errors="strict", newline="\n") as handle:
                handle.write(f"{os.getpid()} {token}\n")
        except FileExistsError as error:
            raise RunnerError(f"Another experiment runner acquired {path}.") from error
        yield token
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _lock_owner(path: Path) -> int | None:
    try:
        fields = path.read_text(encoding="ascii").split()
        return int(fields[0]) if fields else None
    except (OSError, UnicodeError, ValueError):
        return None


def _lock_token(path: Path) -> str | None:
    try:
        fields = path.read_text(encoding="ascii").split()
    except (OSError, UnicodeError):
        return None
    if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{32}", fields[1]) is None:
        return None
    return fields[1]


def _process_exists(pid: Any) -> bool | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if _IS_WINDOWS:
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_exists(pid: int) -> bool | None:
    """Probe a Windows PID without sending it a signal.

    ``os.kill(pid, 0)`` is not a liveness probe on Windows: values other than
    the console-control signals are passed to ``TerminateProcess``. Querying a
    process handle preserves the monitoring contract that status inspection
    cannot mutate or terminate a process.
    """

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    error_access_denied = 5
    error_invalid_parameter = 87
    still_active = 259

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return False
        if error == error_access_denied:
            return True
        return None

    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now() -> str:
    return _utc_datetime().isoformat()


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
