from __future__ import annotations

from pathlib import Path

import pytest

import paper_exp.launch as launch


def test_config_resolution_accepts_only_smoke_or_one_launch_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "configs" / "00-smoke.yaml"
    config.parent.mkdir()
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

    root_scientific = config.parent / "001-case.yaml"
    root_scientific.write_text("", encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="00-smoke"):
        launch.resolve_launch_config(root_scientific, repository=tmp_path)

    nested = config.parent / "01-a1-grid" / "001-case.yaml"
    nested.parent.mkdir()
    nested.write_text("", encoding="utf-8")
    _, nested_resolved = launch.resolve_launch_config(nested, repository=tmp_path)
    assert nested_resolved == nested.resolve()

    too_deep = nested.parent / "extra" / "002-case.yaml"
    too_deep.parent.mkdir()
    too_deep.write_text("", encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="configs/<launch-id>"):
        launch.resolve_launch_config(too_deep, repository=tmp_path)


def test_run_resolution_requires_exact_config_and_run_levels(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "01-case" / "001-test"
    run_dir.mkdir(parents=True)

    _, resolved = launch.resolve_launch_run_dir(run_dir, repository=tmp_path)

    assert resolved == run_dir.resolve()
    with pytest.raises(launch.LaunchError, match="exact results"):
        launch.resolve_launch_run_dir(run_dir.parent, repository=tmp_path)


def test_release_outputs_are_repository_local(tmp_path: Path) -> None:
    config = {
        "output": {"dir": "results"},
        "preprocessing": {"output_dir": "data/tokenized"},
    }

    assert launch.require_results_output(
        config, repository=tmp_path, source="config"
    ) == (tmp_path / "results").resolve()
    assert launch.require_token_cache_output(
        config, repository=tmp_path, source="config"
    ) == (tmp_path / "data" / "tokenized").resolve()

    config["output"]["dir"] = "elsewhere"
    with pytest.raises(launch.LaunchError, match="output.dir"):
        launch.require_results_output(config, repository=tmp_path, source="config")


def test_launch_guard_requires_reviewed_plan_and_cleans_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "docs" / "experiment_plan.md"
    plan.parent.mkdir()
    plan.write_text("Plan status: placeholder\n", encoding="utf-8")
    monkeypatch.setattr(launch, "collect_git_dirty", lambda _root: False)

    with pytest.raises(launch.LaunchError, match="Plan status: reviewed"):
        with launch.direct_launch_guard(repository=tmp_path):
            pytest.fail("placeholder plan must block launch")

    plan.write_text("Plan status: reviewed\n", encoding="utf-8")
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
    plan.write_text("Plan status: reviewed\n", encoding="utf-8")
    monkeypatch.setattr(launch, "collect_git_dirty", lambda _root: True)

    with pytest.raises(launch.LaunchError, match="Commit or stash"):
        with launch.direct_launch_guard(repository=tmp_path):
            pytest.fail("dirty checkout must block launch")
