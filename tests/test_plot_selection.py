from __future__ import annotations

import json
from pathlib import Path

import pytest

import paper_exp.plots as plots


def test_strict_selection_does_not_let_an_incomplete_run_shadow_a_complete_run(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "results" / "01-example"
    complete_run = _write_complete_run(experiment_dir, "001-complete")
    (complete_run / "events.jsonl").write_text("{}\n", encoding="utf-8")

    incomplete_run = experiment_dir / "002-in-progress"
    incomplete_run.mkdir()
    (incomplete_run / "events.jsonl").write_text("{}\n", encoding="utf-8")

    assert plots._latest_run_with(experiment_dir, "events.jsonl") == incomplete_run
    assert (
        plots._latest_run_with(
            experiment_dir,
            "events.jsonl",
            require_complete_run=True,
        )
        == complete_run
    )


@pytest.mark.parametrize("status", ["running", "failed"])
def test_strict_selection_rejects_nonterminal_manifest_status(
    tmp_path: Path,
    status: str,
) -> None:
    experiment_dir = tmp_path / "results" / "01-example"
    run_dir = _write_complete_run(experiment_dir, "001-run", status=status)
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")

    assert (
        plots._latest_run_with(
            experiment_dir,
            "events.jsonl",
            require_complete_run=True,
        )
        is None
    )


def test_strict_selection_accepts_legacy_terminal_manifest_without_status(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "results" / "01-example"
    run_dir = _write_complete_run(experiment_dir, "001-run")
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")

    assert (
        plots._latest_run_with(
            experiment_dir,
            "events.jsonl",
            require_complete_run=True,
        )
        == run_dir
    )


@pytest.mark.parametrize("manifest", ["not json", {"config_id": "wrong", "run_id": "001-run"}])
def test_strict_selection_rejects_malformed_or_mismatched_manifest(
    tmp_path: Path,
    manifest: str | dict[str, str],
) -> None:
    experiment_dir = tmp_path / "results" / "01-example"
    run_dir = _write_complete_run(experiment_dir, "001-run")
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    if isinstance(manifest, str):
        (run_dir / "manifest.json").write_text(manifest, encoding="utf-8")
    else:
        _write_json(run_dir / "manifest.json", manifest)

    assert (
        plots._latest_run_with(
            experiment_dir,
            "events.jsonl",
            require_complete_run=True,
        )
        is None
    )


def test_architecture_only_report04_figures_do_not_require_full_training_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / plots.REPORT04_INPUT_HISTOGRAM_EXPERIMENT
    run_dir = _write_complete_run(experiment_dir, "001-complete")
    _write_json(run_dir / "activation_histograms.json", {})

    generated: list[Path] = []

    def record_output(*, output: str | Path, save_png: bool) -> list[Path]:
        assert save_png is False
        path = Path(output)
        generated.append(path)
        return [path]

    monkeypatch.setattr(plots, "generate_report04_three_relu_architecture", record_output)
    monkeypatch.setattr(plots, "generate_report04_pythia_family_compute_ceiling", record_output)

    outputs = plots._generate_known_paper_figures(
        results_dir,
        tmp_path / "figures",
        save_png=False,
    )

    assert [path.name for path in generated] == [
        "87-pythia-14m-minipile-three-relu-architecture-compute-map.pdf",
        "90-pythia-family-three-relu-model-compute-ceilings.pdf",
    ]
    assert outputs == generated


def test_incomplete_report04_artifact_does_not_activate_architecture_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    run_dir = (
        results_dir
        / plots.REPORT04_INPUT_HISTOGRAM_EXPERIMENT
        / "001-in-progress"
    )
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "activation_histograms.json", {})

    def fail_if_called(**_kwargs) -> list[Path]:
        raise AssertionError("architecture renderer should not run")

    monkeypatch.setattr(
        plots,
        "generate_report04_three_relu_architecture",
        fail_if_called,
    )
    monkeypatch.setattr(
        plots,
        "generate_report04_pythia_family_compute_ceiling",
        fail_if_called,
    )

    assert (
        plots._generate_known_paper_figures(
            results_dir,
            tmp_path / "figures",
            save_png=False,
        )
        == []
    )


def _write_complete_run(
    experiment_dir: Path,
    run_id: str,
    *,
    status: str | None = None,
) -> Path:
    run_dir = experiment_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("experiment_name: test\n", encoding="utf-8")
    _write_json(run_dir / "metrics.json", {})
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    manifest: dict[str, str] = {
        "config_id": experiment_dir.name,
        "run_id": run_id,
    }
    if status is not None:
        manifest["status"] = status
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
