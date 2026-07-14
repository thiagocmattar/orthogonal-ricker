from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import paper_exp.run as run_module
import paper_exp.utils as utils_module
from paper_exp.run import complete_run, run_lifecycle, start_run


def test_start_run_writes_launch_envelope_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)

    run = start_run(
        config,
        config_path=config_path,
        command="pytest lifecycle",
        mode="pretrain",
        run_id="launch-test",
    )

    assert run.config_id == "01-lifecycle"
    assert run.run_id == "001-launch-test"
    assert run.run_dir.is_dir()
    assert yaml.safe_load((run.run_dir / "config.yaml").read_text(encoding="utf-8")) == config
    manifest = _read_manifest(run.run_dir)
    assert manifest["status"] == "running"
    assert manifest["started_at"] == manifest["timestamp"]
    assert manifest["git_commit"] == "launch-commit"
    assert manifest["git_dirty"] is False
    assert manifest["command"] == "pytest lifecycle"
    assert manifest["mode"] == "pretrain"
    assert not (run.run_dir / "metrics.json").exists()
    assert not (run.run_dir / "predictions.jsonl").exists()
    assert not list(run.run_dir.glob(".*.tmp"))


def test_complete_run_writes_completed_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run = start_run(
        config,
        config_path=config_path,
        command="pytest lifecycle",
        mode="pretrain",
        run_id="complete-test",
    )
    launch_manifest = _read_manifest(run.run_dir)

    writes: list[str] = []
    original_json = run_module._atomic_write_json
    original_jsonl = run_module._atomic_write_jsonl

    def record_json(path: str | Path, data: dict[str, object]) -> None:
        writes.append(Path(path).name)
        original_json(path, data)

    def record_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
        writes.append(Path(path).name)
        original_jsonl(path, rows)

    monkeypatch.setattr(run_module, "_atomic_write_json", record_json)
    monkeypatch.setattr(run_module, "_atomic_write_jsonl", record_jsonl)

    complete_run(
        run,
        metrics={"loss": 1.25},
        predictions=[{"id": 1, "prediction": "ok"}],
        manifest_updates={
            "git_commit": "terminal-commit-must-not-win",
            "started_at": "terminal-start-must-not-win",
            "training": {"completed_steps": 3},
        },
    )

    assert writes == ["metrics.json", "predictions.jsonl", "manifest.json"]
    manifest = _read_manifest(run.run_dir)
    assert manifest["status"] == "completed"
    assert manifest["finished_at"]
    assert manifest["git_commit"] == launch_manifest["git_commit"]
    assert manifest["started_at"] == launch_manifest["started_at"]
    assert manifest["training"] == {"completed_steps": 3}
    assert json.loads((run.run_dir / "metrics.json").read_text(encoding="utf-8")) == {
        "loss": 1.25
    }
    assert _read_jsonl(run.run_dir / "predictions.jsonl") == [
        {"id": 1, "prediction": "ok"}
    ]


def test_launch_manifest_snapshot_cannot_be_mutated_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run = start_run(
        config,
        config_path=config_path,
        command="pytest lifecycle",
        mode="pretrain",
        run_id="provenance-test",
    )
    detached_manifest = run.launch_manifest
    detached_manifest["git_commit"] = "mutated-copy"
    monkeypatch.setattr(
        run_module,
        "build_manifest",
        lambda **_: pytest.fail("completion must not rebuild launch provenance"),
    )

    complete_run(run, metrics={}, predictions=[])

    assert _read_manifest(run.run_dir)["git_commit"] == "launch-commit"


def test_run_lifecycle_records_runtime_error_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run_dir: Path | None = None

    with pytest.raises(RuntimeError, match="training failed"):
        with run_lifecycle(
            config,
            config_path=config_path,
            command="pytest lifecycle",
            mode="pretrain",
            run_id="runtime-error",
        ) as run:
            run_dir = run.run_dir
            raise RuntimeError("training failed")

    assert run_dir is not None
    manifest = _read_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "training failed",
    }
    assert manifest["finished_at"]
    assert manifest["git_commit"] == "launch-commit"
    assert manifest["started_at"] == manifest["timestamp"]


def test_run_lifecycle_records_keyboard_interrupt_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run_dir: Path | None = None

    with pytest.raises(KeyboardInterrupt):
        with run_lifecycle(
            config,
            config_path=config_path,
            command="pytest lifecycle",
            mode="pretrain",
            run_id="keyboard-interrupt",
        ) as run:
            run_dir = run.run_dir
            raise KeyboardInterrupt

    assert run_dir is not None
    manifest = _read_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {"type": "KeyboardInterrupt", "message": ""}
    assert manifest["finished_at"]


def test_terminal_transitions_do_not_rewrite_or_alias_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    original = deepcopy(config)
    run = start_run(
        config,
        config_path=config_path,
        command="pytest lifecycle",
        mode="pretrain",
        run_id="config-snapshot",
    )
    saved_before = (run.run_dir / "config.yaml").read_bytes()

    config["model"]["name"] = "mutated-after-launch"
    config["run"]["seed"] = 999
    assert run.config == original
    complete_run(run, metrics={}, predictions=[])

    assert (run.run_dir / "config.yaml").read_bytes() == saved_before
    assert yaml.safe_load(saved_before) == original
    manifest = _read_manifest(run.run_dir)
    assert manifest["model_name"] == original["model"]["name"]
    assert manifest["seed"] == original["run"]["seed"]


def test_failure_recording_error_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)

    def fail_to_record(*_: object, **__: object) -> Path:
        raise OSError("manifest unavailable")

    monkeypatch.setattr(run_module, "fail_run", fail_to_record)

    with pytest.raises(RuntimeError, match="primary failure"):
        with run_lifecycle(
            config,
            config_path=config_path,
            command="pytest lifecycle",
            mode="pretrain",
            run_id="recording-error",
        ):
            raise RuntimeError("primary failure")


def test_terminal_state_cannot_be_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run = start_run(
        config,
        config_path=config_path,
        command="pytest lifecycle",
        mode="pretrain",
        run_id="terminal-guard",
    )

    complete_run(run, metrics={"first": True}, predictions=[])
    completed_manifest = (run.run_dir / "manifest.json").read_bytes()
    completed_metrics = (run.run_dir / "metrics.json").read_bytes()

    with pytest.raises(RuntimeError, match="already terminal"):
        complete_run(run, metrics={"second": True}, predictions=[])
    with pytest.raises(RuntimeError, match="already terminal"):
        run_module.fail_run(run, RuntimeError("too late"))

    assert (run.run_dir / "manifest.json").read_bytes() == completed_manifest
    assert (run.run_dir / "metrics.json").read_bytes() == completed_metrics


def test_exception_after_completion_does_not_flip_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run_dir: Path | None = None

    with pytest.raises(RuntimeError, match="after publication"):
        with run_lifecycle(
            config,
            config_path=config_path,
            command="pytest lifecycle",
            mode="pretrain",
            run_id="post-completion-error",
        ) as run:
            run_dir = complete_run(run, metrics={}, predictions=[])
            raise RuntimeError("after publication")

    assert run_dir is not None
    assert _read_manifest(run_dir)["status"] == "completed"


def test_normal_exit_without_completion_is_failed_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stabilize_provenance(monkeypatch)
    config, config_path = _write_config(tmp_path)
    run_dir: Path | None = None

    with pytest.raises(RuntimeError, match="exited without completion"):
        with run_lifecycle(
            config,
            config_path=config_path,
            command="pytest lifecycle",
            mode="pretrain",
            run_id="unterminated",
        ) as run:
            run_dir = run.run_dir

    assert run_dir is not None
    manifest = _read_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def _stabilize_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils_module, "collect_git_commit", lambda _root: "launch-commit")
    monkeypatch.setattr(utils_module, "collect_git_dirty", lambda _root: False)
    monkeypatch.setattr(utils_module, "collect_gpu_info", lambda: [])
    monkeypatch.setattr(utils_module, "collect_package_versions", lambda: {"paper-exp": "test"})


def _write_config(tmp_path: Path) -> tuple[dict[str, object], Path]:
    config: dict[str, object] = {
        "experiment_name": "lifecycle_test",
        "model": {
            "provider": "huggingface",
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "random",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "01-lifecycle.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def _read_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
