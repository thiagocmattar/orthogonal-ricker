from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator

import pytest

import paper_exp.integrity as integrity
import paper_exp.launch as launch
import paper_exp.runner as runner


def test_parent_runner_executes_one_config_at_a_time_in_numeric_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    active = False
    calls: list[str] = []

    @contextmanager
    def guard(**_kwargs: object) -> Iterator[None]:
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        assert active
        assert command.endswith("experiments/01-first-set/run/runner.py")
        calls.append(config_path.name)
        return (
            tmp_path
            / "experiments"
            / "01-first-set"
            / "raw"
            / config_path.stem
            / "001-test"
        )

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "direct_launch_guard", guard)
    monkeypatch.setattr(runner, "_run_one", run_one)

    completed = runner.run_launch(runner_path, configs, repository=tmp_path)

    assert calls == ["001-case.yaml", "002-case.yaml", "003-case.yaml"]
    assert len(completed) == 3
    assert active is False


@pytest.mark.parametrize("prefixes", [(2, 1), (1, 1)])
def test_parent_runner_rejects_non_increasing_config_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefixes: tuple[int, int],
) -> None:
    runner_path, configs = _layout(tmp_path, prefixes)
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="strictly increasing"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def test_parent_runner_validates_the_whole_set_before_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    _stub_preflight(monkeypatch)
    validated: list[str] = []

    def validate(config: dict[str, object]) -> None:
        validated.append(str(config["name"]))
        if config["name"] == "002-case":
            raise ValueError("invalid second config")

    monkeypatch.setattr(runner, "validate_training_config", validate)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("no run may start after failed preflight"),
    )

    with pytest.raises(ValueError, match="invalid second config"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert validated == ["001-case", "002-case"]


def test_parent_runner_stops_after_first_failed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _stub_preflight(monkeypatch)
    calls: list[str] = []

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        del command
        calls.append(config_path.name)
        if config_path.name.startswith("002-"):
            raise RuntimeError("experiment failed")
        return tmp_path / "result"

    monkeypatch.setattr(runner, "_run_one", run_one)

    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert calls == ["001-case.yaml", "002-case.yaml"]


def test_parent_runner_skips_one_completion_and_retries_only_failed_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2, 3))
    _write_attempt(configs[0], sequence=1, status="failed")
    existing = _write_attempt(configs[0], sequence=2, status="complete")
    _write_attempt(configs[1], sequence=1, status="failed")
    _write_attempt(configs[1], sequence=2, status="failed")
    calls: list[str] = []

    def run_one(
        _config: dict[str, object],
        *,
        config_path: Path,
        command: str,
    ) -> Path:
        assert command.endswith(
            "experiments/01-first-set/run/runner.py --retry-failed"
        )
        calls.append(config_path.name)
        return config_path.parent.parent / "raw" / config_path.stem / "new-attempt"

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "_run_one", run_one)

    completed = runner.run_launch(
        runner_path,
        configs,
        repository=tmp_path,
        retry_failed=True,
    )

    assert calls == ["002-case.yaml", "003-case.yaml"]
    assert completed == [
        existing,
        configs[1].parent.parent / "raw" / configs[1].stem / "new-attempt",
        configs[2].parent.parent / "raw" / configs[2].stem / "new-attempt",
    ]


def test_parent_runner_requires_explicit_failed_retry_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    failed = _write_attempt(configs[0], sequence=1, status="failed")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: pytest.fail("unauthorized retry must fail before locking"),
    )

    with pytest.raises(runner.RunnerError, match="--retry-failed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)

    assert sorted(path.name for path in failed.parent.iterdir()) == [failed.name]


@pytest.mark.parametrize("unsafe_status", ("running", "inconsistent"))
def test_parent_runner_rejects_unsafe_attempt_state_before_launch_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_status: str,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    first = _write_attempt(configs[0], sequence=1, status="failed")
    _write_attempt(configs[1], sequence=1, status=unsafe_status)
    guard_entered = False

    @contextmanager
    def guard(**_kwargs: object) -> Iterator[None]:
        nonlocal guard_entered
        guard_entered = True
        yield

    _stub_preflight(monkeypatch)
    monkeypatch.setattr(runner, "direct_launch_guard", guard)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("unsafe state must abort the tranche"),
    )

    with pytest.raises(runner.RunnerError, match="running or inconsistent"):
        runner.run_launch(
            runner_path,
            configs,
            repository=tmp_path,
            retry_failed=True,
        )

    assert guard_entered is False
    assert sorted(path.name for path in first.parent.iterdir()) == [first.name]


def test_parent_runner_rejects_multiple_coherent_completions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _write_attempt(configs[0], sequence=1, status="complete")
    _write_attempt(configs[0], sequence=2, status="complete")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: pytest.fail("ambiguous state must fail before locking"),
    )

    with pytest.raises(runner.RunnerError, match="multiple coherent completed"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def test_parent_runner_rejects_attempt_from_mutated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    _write_attempt(
        configs[0],
        sequence=1,
        status="failed",
        config={"name": "changed-after-attempt"},
    )
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="inconsistent"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


@pytest.mark.parametrize("invalid_name", ("launch.py", "01-first-set.py"))
def test_case_runner_requires_exact_scaffold_location_and_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_name: str,
) -> None:
    scaffold = _scaffold(tmp_path, "01-first-set")
    invalid = scaffold / "run" / invalid_name
    invalid.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)

    with pytest.raises(runner.RunnerError, match="run/runner.py"):
        runner._resolve_runner(invalid, tmp_path)

    misplaced = scaffold / "runner.py"
    misplaced.write_text("", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="run/runner.py"):
        runner._resolve_runner(misplaced, tmp_path)


def test_smoke_scaffold_cannot_be_a_scientific_case_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path = (
        _scaffold(tmp_path, "00-infrastructure-smoke") / "run" / "runner.py"
    )
    runner_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)

    with pytest.raises(runner.RunnerError, match="nonzero scaffold"):
        runner._resolve_runner(runner_path, tmp_path)


@pytest.mark.parametrize(
    "invalid_name",
    ("000-case.yaml", "01-case.yaml", "001-case.yml"),
)
def test_parent_runner_requires_exact_scientific_config_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_name: str,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    invalid = configs[0].with_name(invalid_name)
    configs[0].rename(invalid)
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="CCC-<case>.yaml"):
        runner.run_launch(runner_path, [invalid], repository=tmp_path)


def test_case_runner_owns_one_matching_config_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1,))
    wrong_root = _scaffold(tmp_path, "02-other-set") / "run"
    wrong_config = wrong_root / configs[0].name
    wrong_config.write_text("", encoding="utf-8")
    _stub_preflight(monkeypatch)

    with pytest.raises(runner.RunnerError, match="may only use configs"):
        runner.run_launch(runner_path, [wrong_config], repository=tmp_path)


def test_case_runner_requires_every_yaml_in_its_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_path, configs = _layout(tmp_path, (1, 2))
    omitted = configs[0].parent / "003-omitted.yaml"
    omitted.write_text("", encoding="utf-8")
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *_args, **_kwargs: pytest.fail("an incomplete tranche must not start"),
    )

    with pytest.raises(runner.RunnerError, match="exactly all YAML files"):
        runner.run_launch(runner_path, configs, repository=tmp_path)


def _layout(tmp_path: Path, prefixes: tuple[int, ...]) -> tuple[Path, list[Path]]:
    scaffold = _scaffold(tmp_path, "01-first-set")
    runner_path = scaffold / "run" / "runner.py"
    runner_path.write_text("", encoding="utf-8")
    config_root = scaffold / "run"
    configs: list[Path] = []
    for index, prefix in enumerate(prefixes):
        suffix = "case" if prefixes.count(prefix) == 1 else f"case-{index}"
        path = config_root / f"{prefix:03d}-{suffix}.yaml"
        path.write_text("", encoding="utf-8")
        configs.append(path)
    return runner_path, configs


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)
    monkeypatch.setattr(launch, "require_tracked_file", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path, **_kwargs: {"name": Path(path).stem},
    )
    monkeypatch.setattr(runner, "validate_training_config", lambda _config: None)
    monkeypatch.setattr(integrity, "validate_training_config", lambda _config: None)
    monkeypatch.setattr(runner, "require_raw_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "require_token_cache_output",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "direct_launch_guard",
        lambda **_kwargs: _null_guard(),
    )


@contextmanager
def _null_guard() -> Iterator[None]:
    yield


def _write_attempt(
    config_path: Path,
    *,
    sequence: int,
    status: str,
    config: dict[str, object] | None = None,
) -> Path:
    run_dir = (
        config_path.parent.parent
        / "raw"
        / config_path.stem
        / f"{sequence:03d}-{status}"
    )
    run_dir.mkdir(parents=True)
    snapshot = config or {"name": config_path.stem}
    (run_dir / "config.yaml").write_text(
        json.dumps(snapshot) + "\n", encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "config_id": config_path.stem,
        "run_id": run_dir.name,
        "tranche_id": config_path.parent.parent.name,
        "mode": "pretrain",
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    if status == "complete":
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            '{"event": "train"}\n', encoding="utf-8"
        )
    if status in {"complete", "running", "failed"}:
        lifecycle_status = "completed" if status == "complete" else status
        manifest.update(
            {
                "status": lifecycle_status,
                "started_at": "2026-01-01T00:00:00Z",
            }
        )
        if status in {"complete", "failed"}:
            manifest["finished_at"] = "2026-01-01T00:01:00Z"
        if lifecycle_status == "failed":
            manifest.update(
                {
                    "failure": {"type": "RuntimeError", "message": "test"},
                }
            )
    elif status != "inconsistent":
        raise ValueError(f"Unsupported test attempt status: {status}")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return run_dir


def _scaffold(tmp_path: Path, scaffold_id: str) -> Path:
    scaffold = tmp_path / "experiments" / scaffold_id
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    return scaffold
