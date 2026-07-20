from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
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
            "--recovery-of-state-path",
            "run-logs/failed.json",
            "--reviewed-retry-run-id",
            "001-reviewed-partial",
            "--reviewed-retry-terminated-pid",
            "35252",
            "--confirm-reviewed-retry-process-exited",
        ]
    )

    assert args.config == ["configs/01-a.yaml", "configs/02-b.yaml"]
    assert args.state_path == "run-logs/batch.json"
    assert args.logs_dir == "run-logs/batch"
    assert args.recovery_of_state_path == "run-logs/failed.json"
    assert args.reviewed_retry_run_id == "001-reviewed-partial"
    assert args.reviewed_retry_terminated_pid == 35252
    assert args.confirm_reviewed_retry_process_exited is True


def test_reviewed_recovery_launches_exact_attempts_and_records_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    calls: list[Path] = []

    def complete_child(
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

    monkeypatch.setattr(queue, "_run_child", complete_child)
    state_path = tmp_path / "run-logs" / "recovery.json"
    logs_dir = tmp_path / "run-logs" / "recovery"
    state = queue.run_pretrain_queue(
        configs,
        state_path=state_path,
        logs_dir=logs_dir,
        recovery_of_state_path=predecessor_path,
        reviewed_retry_run_id=run_dir.name,
        confirm_reviewed_retry_process_exited=True,
        reviewed_retry_terminated_pid=35252,
    )

    assert calls == configs
    assert state["status"] == "completed"
    assert [item["status"] for item in state["items"]] == ["completed", "completed"]
    assert Path(state["items"][0]["run_dir"]).name.startswith("002-")
    assert Path(state["items"][1]["run_dir"]).name.startswith("001-")
    assert state["recovery"]["predecessor_failed_run_id"] == run_dir.name
    assert state["recovery"]["expected_new_attempts"] == {
        configs[0].stem: 2,
        configs[1].stem: 1,
    }
    assert state["queue_id"] != state["recovery"]["predecessor_queue_id"]


def test_reviewed_recovery_requires_exact_committed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    registry_path = tmp_path / "docs" / "experimental-design" / "run-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["records"][0]["failure_type"] = "infrastructure_native_crash"
    _write_yaml(registry_path, registry)

    with pytest.raises(queue.PretrainQueueError, match="exact authorization contract"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_rejects_partial_artifact_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    (run_dir / "extra.txt").write_text("not reviewed\n", encoding="utf-8")

    with pytest.raises(queue.PretrainQueueError, match="must retain exactly"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_requires_exact_failed_suffix_and_fresh_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )

    with pytest.raises(queue.PretrainQueueError, match="exact failed-and-pending suffix"):
        queue.run_pretrain_queue(
            configs[1:],
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )

    (tmp_path / "run-logs" / "recovery").mkdir()
    with pytest.raises(queue.PretrainQueueError, match="fresh nonexistent logs"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_rejects_a_live_terminated_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    monkeypatch.setattr(queue, "_process_exists", lambda pid: True)

    with pytest.raises(queue.PretrainQueueError, match="still alive"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_revalidates_retained_evidence_after_each_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )

    def mutate_evidence_during_child(
        command: list[str],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> int:
        config_path = _config_from_command(command, cwd)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("ok\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        _write_completed_attempt(config_path, cwd)
        (run_dir / "events.jsonl").write_text("mutated\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(queue, "_run_child", mutate_evidence_during_child)
    state_path = tmp_path / "run-logs" / "recovery.json"
    with pytest.raises(queue.PretrainQueueError, match="changed after authorization"):
        queue.run_pretrain_queue(
            configs,
            state_path=state_path,
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_reviewed_recovery_binds_suffix_configs_to_launch_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    launch_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in configs
    }
    config = yaml.safe_load(configs[1].read_text(encoding="utf-8"))
    config["training"]["max_steps"] = 3
    _write_yaml(configs[1], config)
    monkeypatch.setattr(
        queue,
        "_git_file_bytes",
        lambda repository, commit, relative_path: launch_bytes[relative_path],
    )

    with pytest.raises(queue.PretrainQueueError, match="changed scientifically from launch commit"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_rejects_overlapping_or_link_like_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    same_path = tmp_path / "run-logs" / "same"
    with pytest.raises(queue.PretrainQueueError, match="must not overlap"):
        queue.run_pretrain_queue(
            configs,
            state_path=same_path,
            logs_dir=same_path,
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )

    link_parent = tmp_path / "run-logs" / "link-recovery"
    original_is_link_like = queue._is_link_like
    monkeypatch.setattr(
        queue,
        "_is_link_like",
        lambda path: queue._absolute_path(Path(path)) == link_parent
        or original_is_link_like(path),
    )
    with pytest.raises(queue.PretrainQueueError, match="link-like path component"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=link_parent,
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_rejects_preexisting_suffix_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    _write_failed_attempt(configs[1], tmp_path)

    with pytest.raises(queue.PretrainQueueError, match="unexpected attempt inventory"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_requires_singleton_marker_and_ready_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    registry_path = tmp_path / "docs" / "experimental-design" / "run-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    original_registry = registry_path.read_text(encoding="utf-8")
    registry["records"][0]["notes"] += " " + registry["records"][0]["notes"]
    _write_yaml(registry_path, registry)
    with pytest.raises(queue.PretrainQueueError, match="singleton authorization marker"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )

    registry_path.write_text(original_registry, encoding="utf-8")
    config_registry_path = tmp_path / "docs" / "experimental-design" / "config-registry.yaml"
    config_registry = yaml.safe_load(config_registry_path.read_text(encoding="utf-8"))
    config_registry["records"][1]["config_status"] = "draft"
    _write_yaml(config_registry_path, config_registry)
    with pytest.raises(queue.PretrainQueueError, match="ready and noncanonical"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_refuses_nested_recovery_without_new_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [_write_config(tmp_path, index) for index in (1, 2)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        tmp_path, configs, monkeypatch
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor["recovery"] = {"predecessor_queue_id": "0" * 32}
    predecessor_path.write_text(json.dumps(predecessor, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(queue.PretrainQueueError, match="Recovery-of-recovery is intentionally unsupported"):
        queue.run_pretrain_queue(
            configs,
            state_path=tmp_path / "run-logs" / "recovery.json",
            logs_dir=tmp_path / "run-logs" / "recovery",
            recovery_of_state_path=predecessor_path,
            reviewed_retry_run_id=run_dir.name,
            confirm_reviewed_retry_process_exited=True,
            reviewed_retry_terminated_pid=35252,
        )


def test_reviewed_recovery_models_external_runner_junction_and_main_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "runner"
    main = tmp_path / "main"
    runner.mkdir()
    main.mkdir()
    configs = [_write_config(runner, index) for index in range(1, 7)]
    run_dir, predecessor_path = _write_reviewed_running_recovery(
        runner, configs, monkeypatch
    )
    shutil.copytree(runner / "results", main / "results")
    (runner / "run-logs").rename(main / "run-logs")
    predecessor_path = main / "run-logs" / predecessor_path.name
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    old_log_root = str(runner / "run-logs")
    new_log_root = str(main / "run-logs")
    for item in predecessor["items"]:
        for field in ("stdout_log", "stderr_log"):
            if item.get(field):
                item[field] = str(item[field]).replace(old_log_root, new_log_root)
    predecessor_path.write_text(
        json.dumps(predecessor, indent=2) + "\n", encoding="utf-8"
    )
    predecessor_sha = queue._sha256_file(predecessor_path)
    registry_path = runner / "docs" / "experimental-design" / "run-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["records"][0]["notes"] = re.sub(
        r"predecessor_queue_sha256=[0-9a-f]{64}",
        f"predecessor_queue_sha256={predecessor_sha}",
        registry["records"][0]["notes"],
    )
    _write_yaml(registry_path, registry)

    original_same_file = queue._same_file

    def authoritative(path: Path) -> Path:
        absolute = queue._absolute_path(path)
        try:
            relative = absolute.relative_to(runner / "results")
        except ValueError:
            return absolute
        return main / "results" / relative

    monkeypatch.setattr(
        queue,
        "_same_file",
        lambda left, right: original_same_file(authoritative(left), authoritative(right)),
    )
    runner_experiments = {
        queue._absolute_path(runner / "results" / config.stem) for config in configs
    }
    original_is_link_like = queue._is_link_like
    monkeypatch.setattr(
        queue,
        "_is_link_like",
        lambda path: queue._absolute_path(Path(path)) in runner_experiments
        or original_is_link_like(path),
    )

    recovery = queue._reviewed_recovery_contract(
        repository=runner,
        config_paths=[str(path) for path in configs],
        state_path=main / "run-logs" / "recovery.json",
        logs_dir=main / "run-logs" / "recovery",
        predecessor_state_path=predecessor_path,
        reviewed_retry_run_id=run_dir.name,
        confirm_process_exited=True,
        terminated_pid=35252,
    )
    assert recovery is not None
    assert all(
        guard["external_runner"]
        for guard in recovery["state"]["experiment_guards"]
    )
    assert Path(recovery["state"]["predecessor_state_path"]).parent == main / "run-logs"

    monkeypatch.setattr(queue, "_same_file", original_same_file)
    with pytest.raises(queue.PretrainQueueError, match="junction target changed"):
        queue._revalidate_reviewed_recovery(recovery)


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
    monkeypatch.setattr(queue, "_git_head", lambda repository: "b" * 40)
    monkeypatch.setattr(queue, "_git_is_ancestor", lambda repository, ancestor, descendant: True)
    monkeypatch.setattr(
        queue,
        "_git_blob_oid",
        lambda repository, commit, relative_path: f"blob-{relative_path}",
    )
    monkeypatch.setattr(
        queue,
        "_git_file_bytes",
        lambda repository, commit, relative_path: (repository / relative_path).read_bytes(),
    )


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
            "run_sequence": int(run_dir.name.split("-", 1)[0]),
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


def _write_running_attempt(
    config_path: Path,
    repository: Path,
    *,
    git_commit: str = "clean-commit",
) -> Path:
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
            "run_sequence": int(run_dir.name.split("-", 1)[0]),
            "mode": "pretrain",
            "git_commit": git_commit,
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
    run_dir = experiment_dir / f"{sequence:03d}-20260101-000000-{sequence:08x}"
    run_dir.mkdir()
    return run_dir


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_reviewed_running_recovery(
    repository: Path,
    configs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    _stabilize_repository(repository, monkeypatch)
    monkeypatch.setattr(queue, "_process_exists", lambda pid: False)
    git_commit = "a" * 40
    run_dir = _write_running_attempt(configs[0], repository, git_commit=git_commit)
    for config_path in configs[1:]:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        (Path(config["output"]["dir"]) / config_path.stem).mkdir(
            parents=True, exist_ok=True
        )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event": "train", "step": 1, "tokens_seen": 1}) + "\n",
        encoding="utf-8",
    )
    old_logs = repository / "run-logs" / "original"
    old_logs.mkdir(parents=True)
    stdout_log = old_logs / "failed.log"
    stderr_log = old_logs / "failed.err.log"
    stdout_log.write_text("", encoding="utf-8")
    stderr_log.write_text("warning\n", encoding="utf-8")
    failure_message = (
        f"Pretraining child failed for {configs[0]} with exit code 4294967295; "
        "manifest status is 'running'."
    )
    predecessor = {
        "schema_version": 1,
        "queue_id": "1" * 32,
        "status": "failed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:02:00+00:00",
        "finished_at": "2026-01-01T00:02:00+00:00",
        "current_index": 0,
        "failure": {"type": "PretrainQueueError", "message": failure_message},
        "items": [
            {
                "config_path": str(configs[0]),
                "status": "failed",
                "started_at": "2026-01-01T00:00:01+00:00",
                "finished_at": "2026-01-01T00:02:00+00:00",
                "run_dir": str(run_dir),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "returncode": 4294967295,
                "message": failure_message,
            },
            *[
                {
                    "config_path": str(config_path),
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "run_dir": None,
                    "stdout_log": None,
                    "stderr_log": None,
                    "returncode": None,
                    "message": None,
                }
                for config_path in configs[1:]
            ],
        ],
    }
    predecessor_path = repository / "run-logs" / "original.json"
    predecessor_path.write_text(json.dumps(predecessor, indent=2) + "\n", encoding="utf-8")
    predecessor_sha = queue._sha256_file(predecessor_path)
    inventory_sha = queue._attempt_inventory_sha256(run_dir)
    marker = (
        f"{queue.REVIEWED_RETRY_AUTHORIZATION_MARKER} failed_attempt=001 "
        f"authorized_next_attempt=002 inventory_sha256={inventory_sha} "
        f"predecessor_queue_id={predecessor['queue_id']} "
        f"predecessor_queue_sha256={predecessor_sha} "
        f"stdout_sha256={queue._sha256_file(stdout_log)} "
        f"stderr_sha256={queue._sha256_file(stderr_log)} terminated_pid=35252"
    )
    config_registry = {
        "records": [
            {
                "config_id": config_path.stem,
                "config_path": config_path.relative_to(repository).as_posix(),
                "config_status": "ready",
                "canonical_run_id": None,
            }
            for config_path in configs
        ]
    }
    run_registry = {
        "records": [
            {
                "config_id": configs[0].stem,
                "run_id": run_dir.name,
                "attempt": 1,
                "mode": "pretrain",
                "result_path": run_dir.relative_to(repository).as_posix(),
                "lifecycle_status": "partial",
                "evidence_status": "invalid",
                "canonical": False,
                "finished_at": None,
                "git_commit": git_commit,
                "git_dirty": False,
                "failure_type": "infrastructure_operator_process_misidentification",
                "failure_message": "Operator killed the live training PID after misidentifying it.",
                "artifact_manifest_uri": None,
                "artifact_manifest_sha256": None,
                "reviewed_at": "2026-01-01T00:03:00+00:00",
                "notes": marker,
            }
        ]
    }
    _write_yaml(
        repository / "docs" / "experimental-design" / "config-registry.yaml",
        config_registry,
    )
    _write_yaml(
        repository / "docs" / "experimental-design" / "run-registry.yaml",
        run_registry,
    )
    return run_dir, predecessor_path
