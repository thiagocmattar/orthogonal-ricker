from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

import paper_exp.pretrain_queue as queue
from paper_exp.cli import build_parser


def test_queue_runs_children_in_order_and_records_verified_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    calls: list[Path] = []
    clean_checks: list[Path] = []
    _stabilize_repository(tmp_path, monkeypatch)

    def record_clean(repository: Path) -> None:
        clean_checks.append(repository)

    def complete_child(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> int:
        config_path = _config_from_command(command, cwd)
        assert not calls or calls[-1] != config_path
        calls.append(config_path)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text(f"stdout {config_path.stem}\n", encoding="utf-8")
        stderr_log.write_text(f"stderr {config_path.stem}\n", encoding="utf-8")
        _write_completed_attempt(config_path, cwd)
        return 0

    monkeypatch.setattr(queue, "_require_clean_git_tree", record_clean)
    monkeypatch.setattr(queue, "_run_child", complete_child)

    state_path = tmp_path / "run-logs" / "queue.json"
    state = queue.run_pretrain_queue(
        configs,
        state_path=state_path,
        logs_dir=tmp_path / "run-logs",
    )

    assert calls == configs
    assert clean_checks == [tmp_path, tmp_path]
    assert state["status"] == "completed"
    assert [item["status"] for item in state["items"]] == ["completed", "completed"]
    assert all(Path(item["stdout_log"]).read_text(encoding="utf-8").startswith("stdout") for item in state["items"])
    assert all(Path(item["stderr_log"]).read_text(encoding="utf-8").startswith("stderr") for item in state["items"])
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert not list(state_path.parent.glob(".queue.json.*.tmp"))
    assert not (tmp_path / "tmp" / "pretrain-queue.lock").exists()


def test_queue_skips_only_a_verified_completed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, 1)
    completed = _write_completed_attempt(config_path, tmp_path)
    _stabilize_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        queue,
        "_run_child",
        lambda *args, **kwargs: pytest.fail("verified completion must be skipped"),
    )

    state = queue.run_pretrain_queue(
        [config_path],
        state_path=tmp_path / "run-logs" / "skip.json",
        logs_dir=tmp_path / "run-logs",
    )

    assert state["status"] == "completed"
    assert state["items"][0]["status"] == "skipped"
    assert Path(state["items"][0]["run_dir"]) == completed


def test_queue_rejects_an_existing_running_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, 1)
    _write_running_attempt(config_path, tmp_path)
    _stabilize_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        queue,
        "_run_child",
        lambda *args, **kwargs: pytest.fail("running attempt must block relaunch"),
    )
    state_path = tmp_path / "run-logs" / "running.json"

    with pytest.raises(queue.PretrainQueueError, match="still marked running"):
        queue.run_pretrain_queue(
            [config_path],
            state_path=state_path,
            logs_dir=tmp_path / "run-logs",
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert not (tmp_path / "tmp" / "pretrain-queue.lock").exists()


def test_queue_rejects_inconsistent_completed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, 1)
    run_dir = _write_completed_attempt(config_path, tmp_path)
    (run_dir / "metrics.json").unlink()
    _stabilize_repository(tmp_path, monkeypatch)

    with pytest.raises(queue.PretrainQueueError, match="missing required artifacts"):
        queue.run_pretrain_queue(
            [config_path],
            state_path=tmp_path / "run-logs" / "inconsistent.json",
            logs_dir=tmp_path / "run-logs",
        )


def test_queue_fails_stop_without_launching_later_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    calls: list[Path] = []
    _stabilize_repository(tmp_path, monkeypatch)

    def fail_child(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> int:
        config_path = _config_from_command(command, cwd)
        calls.append(config_path)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("failed stdout\n", encoding="utf-8")
        stderr_log.write_text("failed stderr\n", encoding="utf-8")
        _write_failed_attempt(config_path, cwd)
        return 7

    monkeypatch.setattr(queue, "_run_child", fail_child)
    state_path = tmp_path / "run-logs" / "fail-stop.json"

    with pytest.raises(queue.PretrainQueueError, match="exit code 7"):
        queue.run_pretrain_queue(
            configs,
            state_path=state_path,
            logs_dir=tmp_path / "run-logs",
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert calls == [configs[0]]
    assert state["status"] == "failed"
    assert [item["status"] for item in state["items"]] == ["failed", "pending"]


def test_queue_checks_clean_git_tree_before_each_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    calls: list[Path] = []
    dirty_states = iter((False, True))
    monkeypatch.setattr(queue, "_git_root", lambda cwd: tmp_path)
    monkeypatch.setattr(queue, "collect_git_dirty", lambda repository: next(dirty_states))

    def complete_first(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> int:
        config_path = _config_from_command(command, cwd)
        calls.append(config_path)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("ok\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        _write_completed_attempt(config_path, cwd)
        return 0

    monkeypatch.setattr(queue, "_run_child", complete_first)

    with pytest.raises(queue.PretrainQueueError, match="working tree is dirty"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "dirty.json",
            logs_dir=tmp_path / "run-logs",
        )

    assert calls == [configs[0]]


def test_queue_uses_one_repository_wide_exclusive_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, 1)
    _stabilize_repository(tmp_path, monkeypatch)
    lock_path = tmp_path / "tmp" / "pretrain-queue.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("owned\n", encoding="utf-8")

    with pytest.raises(queue.PretrainQueueError, match="owns the lock"):
        queue.run_pretrain_queue(
            [config_path],
            state_path=tmp_path / "run-logs" / "locked.json",
            logs_dir=tmp_path / "run-logs",
        )

    assert lock_path.read_text(encoding="utf-8") == "owned\n"


def test_cli_accepts_repeated_queue_configs() -> None:
    args = build_parser().parse_args(
        [
            "run-pretrain-queue",
            "--config",
            "configs/01-a.yaml",
            "--config",
            "configs/02-b.yaml",
            "--state-path",
            "run-logs/batch.json",
            "--logs-dir",
            "run-logs/batch",
        ]
    )

    assert args.config == ["configs/01-a.yaml", "configs/02-b.yaml"]
    assert args.state_path == "run-logs/batch.json"
    assert args.logs_dir == "run-logs/batch"


def test_child_runner_keeps_stdout_and_stderr_separate(tmp_path: Path) -> None:
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"

    returncode = queue._run_child(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        cwd=tmp_path,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

    assert returncode == 0
    assert stdout_log.read_text(encoding="utf-8").strip() == "out"
    assert stderr_log.read_text(encoding="utf-8").strip() == "err"


def _stabilize_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue, "_git_root", lambda cwd: tmp_path)
    monkeypatch.setattr(queue, "collect_git_dirty", lambda repository: False)


def _config_from_command(command: list[str], cwd: Path) -> Path:
    config_path = Path(command[-1])
    return config_path if config_path.is_absolute() else (cwd / config_path).resolve()


def _write_config(tmp_path: Path, index: int) -> Path:
    config: dict[str, Any] = {
        "experiment_name": f"queue_test_{index}",
        "model": {
            "provider": "huggingface",
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "random",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "training": {"max_steps": 2, "max_wall_seconds": None},
        "validation": {"enabled": True},
        "checkpoint": {"save_final": True},
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / f"{index:02d}-queue-test-{index}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path.resolve()


def _write_completed_attempt(config_path: Path, repository: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = _new_run_dir(config_path, config)
    checkpoint_dir = run_dir / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True)
    _write_yaml(run_dir / "config.yaml", config)
    _write_json(
        run_dir / "manifest.json",
        {
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "config_id": config_path.stem,
            "run_id": run_dir.name,
            "mode": "pretrain",
            "git_commit": "clean-commit",
            "git_dirty": False,
            "training": {"completed_steps": 2},
            "checkpoint": {"saved": True, "path": str(checkpoint_dir)},
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "calibration/optimizer_steps": 2,
            "calibration/planned_optimizer_steps": 2,
            "calibration/validation_loss_final": 1.25,
        },
    )
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    return run_dir


def _write_running_attempt(config_path: Path, repository: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = _new_run_dir(config_path, config)
    _write_yaml(run_dir / "config.yaml", config)
    _write_json(
        run_dir / "manifest.json",
        {
            "status": "running",
            "started_at": "2026-01-01T00:00:00+00:00",
            "config_id": config_path.stem,
            "run_id": run_dir.name,
            "mode": "pretrain",
            "git_commit": "clean-commit",
            "git_dirty": False,
        },
    )
    return run_dir


def _write_failed_attempt(config_path: Path, repository: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = _new_run_dir(config_path, config)
    _write_yaml(run_dir / "config.yaml", config)
    _write_json(
        run_dir / "manifest.json",
        {
            "status": "failed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "config_id": config_path.stem,
            "run_id": run_dir.name,
            "mode": "pretrain",
            "git_commit": "clean-commit",
            "git_dirty": False,
            "failure": {"type": "RuntimeError", "message": "test failure"},
        },
    )
    return run_dir


def _new_run_dir(config_path: Path, config: dict[str, Any]) -> Path:
    experiment_dir = Path(config["output"]["dir"]) / config_path.stem
    experiment_dir.mkdir(parents=True, exist_ok=True)
    sequence = len([path for path in experiment_dir.iterdir() if path.is_dir()]) + 1
    run_dir = experiment_dir / f"{sequence:03d}-test"
    run_dir.mkdir()
    return run_dir


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
