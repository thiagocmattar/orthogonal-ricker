from __future__ import annotations

from pathlib import Path

import pytest

import paper_exp.launch as launch
import paper_exp.design as design


def test_config_resolution_accepts_only_exact_scaffold_run_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _scaffold(tmp_path, "00-infrastructure-smoke")
    config = smoke / "run" / "00-smoke.yaml"
    config.write_text("experiment_name: test\n", encoding="utf-8")
    observed: list[Path] = []
    monkeypatch.setattr(
        launch,
        "require_tracked_file",
        lambda _root, path: observed.append(path),
    )

    root, resolved = launch.resolve_launch_config(config, repository=tmp_path)

    assert root == tmp_path.resolve()
    assert resolved == config.resolve()
    assert observed == [config.resolve()]

    misplaced_scientific = config.with_name("001-case.yaml")
    misplaced_scientific.write_text("", encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="CCC-<case>"):
        launch.resolve_launch_config(misplaced_scientific, repository=tmp_path)

    scientific = _scaffold(tmp_path, "01-a1-grid") / "run" / "001-case.yaml"
    scientific.write_text("", encoding="utf-8")
    _, scientific_resolved = launch.resolve_launch_config(
        scientific, repository=tmp_path
    )
    assert scientific_resolved == scientific.resolve()

    too_deep = scientific.parent / "extra" / "002-case.yaml"
    too_deep.parent.mkdir()
    too_deep.write_text("", encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="directly under"):
        launch.resolve_launch_config(too_deep, repository=tmp_path)


def test_run_resolution_requires_exact_config_and_run_levels(tmp_path: Path) -> None:
    scaffold = _scaffold(tmp_path, "01-a1-grid")
    run_dir = scaffold / "raw" / "001-case" / "001-test"
    run_dir.mkdir(parents=True)

    _, resolved = launch.resolve_launch_run_dir(run_dir, repository=tmp_path)

    assert resolved == run_dir.resolve()
    with pytest.raises(launch.LaunchError, match="exact experiments"):
        launch.resolve_launch_run_dir(run_dir.parent, repository=tmp_path)


def test_release_outputs_are_owned_by_the_config_scaffold(tmp_path: Path) -> None:
    scaffold = _scaffold(tmp_path, "01-a1-grid")
    config_path = scaffold / "run" / "001-case.yaml"
    config_path.write_text("", encoding="utf-8")
    config = {
        "output": {"dir": "experiments/01-a1-grid/raw"},
        "preprocessing": {"output_dir": "data/tokenized"},
    }

    assert launch.require_raw_output(
        config, repository=tmp_path, config_path=config_path
    ) == (scaffold / "raw").resolve()
    assert launch.require_token_cache_output(
        config, repository=tmp_path, source="config"
    ) == (tmp_path / "data" / "tokenized").resolve()

    config["output"]["dir"] = "elsewhere"
    with pytest.raises(launch.LaunchError, match="output.dir"):
        launch.require_raw_output(
            config, repository=tmp_path, config_path=config_path
        )


def test_scaffold_requires_all_owned_directories(tmp_path: Path) -> None:
    scaffold = tmp_path / "experiments" / "01-a1-grid"
    (scaffold / "run").mkdir(parents=True)
    (scaffold / "raw").mkdir()

    with pytest.raises(launch.LaunchError, match="missing: figs"):
        launch.resolve_experiment_scaffold("01-a1-grid", repository=tmp_path)

    invalid = tmp_path / "experiments" / "00-a1-grid"
    for name in ("run", "raw", "figs"):
        (invalid / name).mkdir(parents=True, exist_ok=True)
    with pytest.raises(launch.LaunchError, match="prefix 00"):
        launch.resolve_experiment_scaffold("00-a1-grid", repository=tmp_path)


def test_launch_guard_requires_reviewed_plan_and_cleans_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "docs" / "experiment_plan.md"
    plan.parent.mkdir()
    plan.write_text(
        "Plan status: placeholder\n"
        "Reviewed design commit: none\n"
        "Reviewed case groups: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launch, "collect_git_dirty", lambda _root: False)

    with pytest.raises(launch.LaunchError, match="Plan status: reviewed"):
        with launch.direct_launch_guard(repository=tmp_path):
            pytest.fail("placeholder plan must block launch")

    plan.write_text(
        "Plan status: reviewed\n"
        f"Reviewed design commit: {'a' * 40}\n"
        "Reviewed case groups: [A1-lr-screen]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(design, "validate_reviewed_design", lambda _root: None)
    lock = tmp_path / "tmp" / "experiment.lock"
    with launch.direct_launch_guard(repository=tmp_path):
        assert lock.is_file()
        with pytest.raises(launch.LaunchError, match="already exists"):
            with launch.direct_launch_guard(repository=tmp_path):
                pytest.fail("a second launch must not acquire the lock")
    assert not lock.exists()


def test_launch_guard_rejects_dirty_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "docs" / "experiment_plan.md"
    plan.parent.mkdir()
    plan.write_text(
        "Plan status: reviewed\n"
        f"Reviewed design commit: {'a' * 40}\n"
        "Reviewed case groups: [A1-lr-screen]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(design, "validate_reviewed_design", lambda _root: None)
    monkeypatch.setattr(launch, "collect_git_dirty", lambda _root: True)

    with pytest.raises(launch.LaunchError, match="Commit or stash"):
        with launch.direct_launch_guard(repository=tmp_path):
            pytest.fail("dirty checkout must block launch")


def _scaffold(tmp_path: Path, scaffold_id: str) -> Path:
    scaffold = tmp_path / "experiments" / scaffold_id
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    return scaffold
