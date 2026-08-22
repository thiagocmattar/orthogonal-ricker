from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from paper_exp.config import ConfigError
import paper_exp.runner as runner


class _FakeChild:
    def __init__(self, finish: Callable[[], None], *, exit_code: int = 0) -> None:
        self.pid = 43210
        self._finish = finish
        self._exit_code = exit_code
        self._polls = 0
        self.closed = False

    def poll(self) -> int | None:
        self._polls += 1
        if self._polls == 1:
            return None
        self._finish()
        return self._exit_code

    def close(self) -> None:
        self.closed = True


def test_runner_executes_children_in_order_and_persists_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    calls: list[Path] = []
    children: list[_FakeChild] = []
    _disable_git_check(monkeypatch)

    def start_child(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> _FakeChild:
        assert "--runner-token" in command
        command_token = command[command.index("--runner-token") + 1]
        assert runner._lock_token(tmp_path / "tmp" / "experiment-runner.lock") == command_token
        config_path = _config_from_command(command, cwd)
        calls.append(config_path)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("started\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        run_dir = _write_running_attempt(config_path)
        child = _FakeChild(lambda: _complete_attempt(run_dir, config_path))
        children.append(child)
        return child

    monkeypatch.setattr(runner, "_start_child", start_child)
    state_path = tmp_path / "run-logs" / "runner.json"
    state = runner.run_configs(
        configs,
        state_path=state_path,
        logs_dir=tmp_path / "run-logs",
        repository=tmp_path,
        poll_seconds=0.001,
    )

    assert calls == configs
    assert all(child.closed for child in children)
    assert state["status"] == "completed"
    assert [item["status"] for item in state["items"]] == ["completed", "completed"]
    assert state["progress"]["finished_configs"] == 2
    assert state["progress"]["estimated_remaining_seconds"] == 0.0
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert not list(state_path.parent.glob(".runner.json.*.tmp"))
    assert not (tmp_path / "tmp" / "experiment-runner.lock").exists()


def test_runner_preflights_every_config_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_config(tmp_path, 1)
    invalid = tmp_path / "configs" / "02-invalid.yaml"
    invalid.write_text("experiment_name: incomplete\n", encoding="utf-8")
    _disable_git_check(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_start_child",
        lambda *args, **kwargs: pytest.fail("no child may launch before full preflight"),
    )

    with pytest.raises(ConfigError):
        runner.run_configs(
            [first, invalid],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )

    assert not (tmp_path / "state.json").exists()


def test_runner_preflights_cache_content_before_any_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    (tmp_path / "data" / "tokenized" / "test-cache" / "tokens.int32.bin").write_bytes(
        b"corrupt"
    )
    _disable_git_check(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_start_child",
        lambda *args, **kwargs: pytest.fail("no child may launch with a corrupt cache"),
    )

    with pytest.raises(runner.RunnerError, match="size does not match"):
        runner.run_configs(
            [config],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )

    assert not (tmp_path / "state.json").exists()


def test_runner_rejects_duplicate_config_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    _disable_git_check(monkeypatch)

    with pytest.raises(runner.RunnerError, match="duplicate"):
        runner.run_configs(
            [config, config],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )


def test_runner_rejects_a_config_outside_the_repository_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    external = tmp_path / "outside" / config.name
    external.parent.mkdir()
    external.write_bytes(config.read_bytes())
    _disable_git_check(monkeypatch)

    with pytest.raises(runner.RunnerError, match="under"):
        runner.run_configs(
            [external],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )


def test_runner_rejects_output_outside_repository_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, 1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output"]["dir"] = str(tmp_path / "elsewhere")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _disable_git_check(monkeypatch)

    with pytest.raises(runner.RunnerError, match="output.dir"):
        runner.run_configs(
            [config_path],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )


def test_untracked_config_is_not_launchable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, 1)

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise runner.subprocess.CalledProcessError(1, "git ls-files")

    monkeypatch.setattr(runner.subprocess, "run", reject)
    with pytest.raises(runner.RunnerError, match="tracked by Git"):
        runner._require_tracked_config(tmp_path, config_path)


def test_runner_never_overwrites_an_existing_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    state_path = tmp_path / "state.json"
    original = b'{"status":"failed","runner_id":"preserve-me"}\n'
    state_path.write_bytes(original)
    _disable_git_check(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_start_child",
        lambda *args, **kwargs: pytest.fail("an existing state must block launch"),
    )

    with pytest.raises(runner.RunnerError, match="state already exists"):
        runner.run_configs(
            [config],
            state_path=state_path,
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )

    assert state_path.read_bytes() == original


def test_runner_skips_only_an_exact_verified_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    run_dir = _write_running_attempt(config)
    _complete_attempt(run_dir, config)
    _disable_git_check(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_start_child",
        lambda *args, **kwargs: pytest.fail("verified completion must be skipped"),
    )

    state = runner.run_configs(
        [config],
        state_path=tmp_path / "state.json",
        logs_dir=tmp_path / "logs",
        repository=tmp_path,
    )

    assert state["items"][0]["status"] == "skipped"
    assert Path(state["items"][0]["run_dir"]) == run_dir


def test_runner_refuses_an_existing_running_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, 1)
    _write_running_attempt(config)
    _disable_git_check(monkeypatch)

    with pytest.raises(runner.RunnerError, match="still marked running"):
        runner.run_configs(
            [config],
            state_path=tmp_path / "state.json",
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
        )

    assert not (tmp_path / "state.json").exists()


def test_runner_fails_stop_without_launching_later_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    calls: list[Path] = []
    _disable_git_check(monkeypatch)

    def start_child(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> _FakeChild:
        config_path = _config_from_command(command, cwd)
        calls.append(config_path)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("failed\n", encoding="utf-8")
        run_dir = _write_running_attempt(config_path)
        return _FakeChild(lambda: _fail_attempt(run_dir), exit_code=7)

    monkeypatch.setattr(runner, "_start_child", start_child)
    state_path = tmp_path / "state.json"
    with pytest.raises(runner.RunnerError, match="exit code 7"):
        runner.run_configs(
            configs,
            state_path=state_path,
            logs_dir=tmp_path / "logs",
            repository=tmp_path,
            poll_seconds=0.001,
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert calls == [configs[0]]
    assert state["status"] == "failed"
    assert [item["status"] for item in state["items"]] == ["failed", "pending"]


def test_read_status_refreshes_progress_and_etc_without_writing(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, 1, max_steps=10)
    run_dir = _write_running_attempt(config, step=4)
    started = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
    state_path = tmp_path / "runner.json"
    state = {
        "schema_version": runner.RUNNER_SCHEMA_VERSION,
        "runner_id": "test-runner",
        "runner_pid": 999_999_999,
        "status": "running",
        "created_at": started,
        "updated_at": started,
        "items": [
            {
                "index": 0,
                "config_path": config.relative_to(tmp_path).as_posix(),
                "config_id": config.stem,
                "planned_steps": 10,
                "max_wall_seconds": None,
                "status": "running",
                "started_at": started,
                "pid": 24680,
                "run_dir": str(run_dir),
            }
        ],
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    before = state_path.read_bytes()

    refreshed = runner.read_runner_status(state_path, repository=tmp_path)

    assert refreshed["progress"]["current_step"] == 4
    assert refreshed["progress"]["planned_steps"] == 10
    assert refreshed["progress"]["estimated_remaining_seconds"] == pytest.approx(60.0, abs=2.0)
    assert refreshed["progress"]["estimated_completion_at"] is not None
    assert refreshed["runner_process_alive"] is False
    assert state_path.read_bytes() == before
    rendered = runner.format_runner_status(refreshed)
    assert "PID 24680" in rendered
    assert f"run {run_dir}" in rendered
    assert "step 4/10 (40.0%)" in rendered
    assert "ETC:" in rendered
    assert "Remaining:" in rendered
    assert "task loss 2.5" in rendered
    assert "recent throughput 64.0 tokens/s" in rendered


def test_status_reports_unknown_etc_without_measurement() -> None:
    state = {
        "status": "pending",
        "items": [{"status": "pending", "planned_steps": 10}],
        "progress": {
            "finished_configs": 0,
            "total_configs": 1,
            "estimated_remaining_seconds": None,
            "estimated_completion_at": None,
        },
    }

    assert "unknown until measurable progress exists" in runner.format_runner_status(state)


def test_windows_liveness_probe_never_calls_os_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    monkeypatch.setattr(runner, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        runner,
        "_windows_process_exists",
        lambda pid: observed.append(pid) or True,
    )
    monkeypatch.setattr(
        runner.os,
        "kill",
        lambda *_args: pytest.fail("Windows liveness checks must not call os.kill"),
    )

    assert runner._process_exists(12345) is True
    assert observed == [12345]


@pytest.mark.skipif(not runner._IS_WINDOWS, reason="Windows process API only")
def test_windows_liveness_probe_recognizes_current_process() -> None:
    assert runner._windows_process_exists(os.getpid()) is True


def test_runner_refuses_a_lock_with_unverifiable_ownership(tmp_path: Path) -> None:
    lock_path = tmp_path / "tmp" / "experiment-runner.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("not-a-pid\n", encoding="ascii")

    with pytest.raises(runner.RunnerError, match="Cannot verify the owner"):
        with runner._exclusive_lock(lock_path):
            pytest.fail("an unverifiable existing lock must not be replaced")

    assert lock_path.read_text(encoding="ascii") == "not-a-pid\n"


def test_direct_launch_guard_accepts_only_the_active_runner_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_clean_git_tree", lambda _repository: None)
    monkeypatch.setattr(runner, "_require_reviewed_plan", lambda _repository: None)
    lock_path = tmp_path / "tmp" / "experiment-runner.lock"

    with runner._exclusive_lock(lock_path) as token:
        with runner.direct_launch_guard(runner_token=token, repository=tmp_path):
            assert lock_path.is_file()
        with pytest.raises(runner.RunnerError, match="does not match"):
            with runner.direct_launch_guard(
                runner_token="0" * 32,
                repository=tmp_path,
            ):
                pytest.fail("a mismatched child token must not launch")

    assert not lock_path.exists()


def test_reviewed_plan_gate_is_machine_readable(tmp_path: Path) -> None:
    plan_path = tmp_path / "docs" / "experiment_plan.md"
    plan_path.parent.mkdir()
    plan_path.write_text("# Plan\n\nPlan status: `placeholder`\n", encoding="utf-8")

    with pytest.raises(runner.RunnerError, match="launches are blocked"):
        runner._require_reviewed_plan(tmp_path)

    plan_path.write_text("# Plan\n\nPlan status: `reviewed`\n", encoding="utf-8")
    runner._require_reviewed_plan(tmp_path)
    plan_path.write_text("# Plan\n\nPlan status: reviewed\n", encoding="utf-8")
    runner._require_reviewed_plan(tmp_path)


def _disable_git_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_require_clean_git_tree", lambda _repository: None)
    monkeypatch.setattr(runner, "_require_reviewed_plan", lambda _repository: None)
    monkeypatch.setattr(runner, "collect_git_commit", lambda _repository: "a" * 40)
    monkeypatch.setattr(runner, "_require_tracked_config", lambda *_args: None)


def _write_config(tmp_path: Path, index: int, *, max_steps: int = 4) -> Path:
    config_path = tmp_path / "configs" / f"{index:02d}-experiment-{index}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": f"experiment_{index}",
        "model": {
            "provider": "huggingface",
            "name": "pythia-test-random",
            "architecture": "test/pythia",
            "revision": "a" * 40,
            "initialization": "random",
        },
        "data": {
            "name": "test/data",
            "revision": "b" * 40,
            "split": "train",
            "text_column": "text",
            "max_documents": 8,
        },
        "tokenizer": {"name": "test/tokenizer", "revision": "c" * 40},
        "preprocessing": {
            "output_dir": "data/tokenized",
            "cache_id": "test-cache",
            "block_size": 16,
            "append_eos": True,
            "overwrite": False,
        },
        "evaluation": {"metric": "loss"},
        "run": {
            "seed": 0,
            "max_examples": 8,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
            "model_initialization_seed": 0,
            "data_order_seed": 1,
            "training_schedule_hash": None,
        },
        "training": {
            "device": "cpu",
            "precision": "float32",
            "max_steps": max_steps,
            "max_wall_seconds": None,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            "log_every": 1,
            "optimizer": "adamw",
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1.0e-8,
            "weight_decay": 0.01,
            "threshold_learning_rate_multiplier": None,
        },
        "validation": {
            "enabled": False,
        },
        "checkpoint": {"save_final": False, "save_optimizer": False},
        "activation_pressure": {
            "enabled": True,
            "method": "none",
            "sites": ["mlp_hiddens"],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001],
        },
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cache_dir = tmp_path / "data" / "tokenized" / "test-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.int32.bin"
    tokens_path.write_bytes(b"\x00" * (64 * 4))
    _write_json(
        cache_dir / "metadata.json",
        {
            "tokens_path": str(tokens_path),
            "dtype": "int32",
            "tokens": 64,
            "tokens_bytes": tokens_path.stat().st_size,
            "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
            "dataset_name": "test/data",
            "dataset_revision": "b" * 40,
            "split": "train",
            "text_column": "text",
            "max_documents": 8,
            "tokenizer_name": "test/tokenizer",
            "tokenizer_revision": "c" * 40,
            "block_size": 16,
            "append_eos": True,
        },
    )
    return config_path.resolve()


def _config_from_command(command: list[str], cwd: Path) -> Path:
    selected = Path(command[command.index("--config") + 1])
    return (cwd / selected).resolve() if not selected.is_absolute() else selected.resolve()


def _write_running_attempt(config_path: Path, *, step: int = 1) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["output"]["dir"]) / config_path.stem / "001-test-run"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        run_dir / "manifest.json",
        {
            "config_id": config_path.stem,
            "run_id": run_dir.name,
            "status": "running",
            "started_at": now,
        },
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "train",
                "step": step,
                "tokens_seen": 128 * step,
                "task_loss": 2.5,
                "pressure_loss": 0.1,
                "step_wall_seconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _complete_attempt(run_dir: Path, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    max_steps = int(config["training"]["max_steps"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat()})
    manifest.update(
        {
            "mode": "pretrain",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "training": {"completed_steps": max_steps},
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "metrics.json",
        {
            "training/optimizer_steps": max_steps,
            "training/planned_optimizer_steps": max_steps,
            "training/wall_seconds_total": float(max_steps * 2),
        },
    )
    event = {"event": "train", "step": max_steps, "tokens_seen": 128 * max_steps}
    serialized_event = json.dumps(event) + "\n"
    (run_dir / "predictions.jsonl").write_text(serialized_event, encoding="utf-8")
    (run_dir / "events.jsonl").write_text(serialized_event, encoding="utf-8")


def _fail_attempt(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "failure": {"type": "RuntimeError", "message": "test failure"},
        }
    )
    _write_json(run_dir / "manifest.json", manifest)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
