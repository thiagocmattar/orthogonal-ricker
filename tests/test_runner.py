from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

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
        assert command.endswith("runners/01-first-set.py")
        calls.append(config_path.name)
        return tmp_path / "results" / config_path.stem / "001-test"

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


@pytest.mark.parametrize(
    "invalid_name",
    (
        "launch.py",
        "1-pilot-screen.py",
        "001-pilot-screen.py",
        "00-pilot-screen.py",
        "01-screen.py",
    ),
)
def test_case_runner_requires_exact_top_level_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_name: str,
) -> None:
    invalid = tmp_path / "runners" / invalid_name
    invalid.parent.mkdir()
    invalid.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "require_tracked_file", lambda *_args: None)

    with pytest.raises(runner.RunnerError, match="NN-<phase>-<tranche>"):
        runner._resolve_runner(invalid, tmp_path)


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
    wrong_root = tmp_path / "configs" / "02-other-set"
    wrong_root.mkdir()
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
    runner_path = tmp_path / "runners" / "01-first-set.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("", encoding="utf-8")
    config_root = tmp_path / "configs" / runner_path.stem
    config_root.mkdir(parents=True)
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
    monkeypatch.setattr(runner, "require_results_output", lambda *_args, **_kwargs: None)
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
