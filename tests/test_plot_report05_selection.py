from __future__ import annotations

import json
from pathlib import Path

from paper_exp import plots
from paper_exp.plot_catalog import REPORT05_FIGURES
from paper_exp.plot_report05 import REPORT05_PINNED_RUN_IDS, REPORT05_TRAINING_RUNS


def test_report05_public_wrappers_match_the_catalog() -> None:
    for entry in REPORT05_FIGURES:
        assert callable(getattr(plots, entry.public_wrapper))


def test_pinned_training_selection_never_substitutes_a_newer_run(tmp_path: Path) -> None:
    label, experiment_id = REPORT05_TRAINING_RUNS[0]
    pinned_id = REPORT05_PINNED_RUN_IDS[experiment_id]
    pinned = tmp_path / experiment_id / pinned_id
    newer = tmp_path / experiment_id / "999-20990101-000000-newer"
    _write_terminal_run(pinned, experiment_id)
    _write_terminal_run(newer, experiment_id)
    (pinned / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (newer / "events.jsonl").write_text("{}\n", encoding="utf-8")

    selected = plots._pinned_report05_training_runs(tmp_path, "events.jsonl")

    assert selected == [(label, pinned)]


def test_report05_clipping_selectors_use_declared_suffix_contracts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[tuple[tuple[str, str], ...], str]] = []

    def record(
        _results_path: Path,
        experiments: list[tuple[str, str]],
        artifact_name: str,
        **_kwargs: object,
    ) -> list[tuple[str, Path]]:
        calls.append((tuple(experiments), artifact_name))
        return []

    monkeypatch.setattr(plots, "_latest_labeled_runs", record)

    plots._report05_site_clipping_runs(tmp_path)
    plots._report05_joint_clipping_runs(tmp_path)

    experiment_groups = [experiments for experiments, _artifact in calls]
    assert any(
        experiment_id.endswith("-clipping-sweep-report05-exact-joint")
        and experiment_id.startswith("77-")
        for experiments in experiment_groups
        for _method, experiment_id in experiments
    )
    assert any(
        experiment_id.endswith("-clipping-sweep-report04-attention-inputs")
        and experiment_id.startswith("98-")
        for experiments in experiment_groups
        for _method, experiment_id in experiments
    )
    assert any(
        experiment_id.endswith("-clipping-sweep-r05s-q")
        and experiment_id.startswith("107-")
        for experiments in experiment_groups
        for _method, experiment_id in experiments
    )
    assert all(artifact == "clipping_frontier.jsonl" for _experiments, artifact in calls)


def test_strict_report05_suite_promotes_staged_outputs_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    destination = figures / REPORT05_FIGURES[0].filename
    destination.write_text("old", encoding="utf-8")

    def render(
        _results_dir: str | Path,
        staging_dir: str | Path,
        **_kwargs: object,
    ) -> list[Path]:
        staged = Path(staging_dir) / destination.name
        staged.write_text("new", encoding="utf-8")
        return [staged]

    monkeypatch.setattr(plots, "_generate_report05_figures_in_place", render)

    outputs = plots.generate_report05_figures(tmp_path / "results", figures)

    assert outputs == [destination]
    assert destination.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".report05-stage-*"))


def _write_terminal_run(run_dir: Path, config_id: str) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config_id": config_id,
                "run_id": run_dir.name,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
