from __future__ import annotations

import json
from pathlib import Path

import yaml

from paper_exp.config import load_config
from paper_exp.plots import generate_clipping_frontier, generate_plots
from paper_exp.run import run_smoke


def test_smoke_run_creates_expected_output_files(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    run_dir = run_smoke(config, config_path=config_path, command="pytest smoke", run_id="test-run")

    assert run_dir.parent.name == "01-smoke-test"
    assert run_dir.name == "001-test-run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions.jsonl").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smoke/num_examples"] == 3
    assert metrics["smoke/passed"] is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "smoke_test"
    assert manifest["config_id"] == "01-smoke-test"
    assert manifest["run_id"] == "001-test-run"
    assert manifest["run_sequence"] == 1
    assert manifest["seed"] == 0


def test_plots_command_can_generate_pdf_from_smoke_results(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)
    run_smoke(config, config_path=config_path, command="pytest smoke", run_id="test-run")

    outputs = generate_plots(
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        save_png=True,
    )

    assert tmp_path / "figures" / "01-results-summary.pdf" in outputs
    assert (tmp_path / "figures" / "01-results-summary.pdf").exists()
    assert (tmp_path / "figures" / "01-results-summary.png").exists()


def test_clipping_frontier_plot_generates_pdf_and_png(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "01-clipping" / "001-test-run"
    run_dir.mkdir(parents=True)
    rows = [
        {"threshold": 0.0, "achieved_sparsity": 0.0, "validation_loss": 7.6, "validation_tokens": 8192},
        {"threshold": 0.01, "achieved_sparsity": 0.05, "validation_loss": 7.61, "validation_tokens": 8192},
    ]
    with (run_dir / "clipping_frontier.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    outputs = generate_clipping_frontier(
        run_dir=run_dir,
        output=tmp_path / "figures" / "02-clipping-frontier.pdf",
        save_png=True,
    )

    assert tmp_path / "figures" / "02-clipping-frontier.pdf" in outputs
    assert (tmp_path / "figures" / "02-clipping-frontier.pdf").exists()
    assert (tmp_path / "figures" / "02-clipping-frontier.png").exists()


def _write_temp_config(tmp_path: Path) -> Path:
    config = {
        "experiment_name": "smoke_test",
        "model": {
            "provider": "huggingface",
            "name": "TODO_MODEL_NAME",
            "architecture": "TODO_MODEL_ARCHITECTURE",
            "initialization": "random",
        },
        "data": {"name": "TODO_DATASET_NAME", "split": "TODO_SPLIT"},
        "evaluation": {"metric": "TODO_METRIC"},
        "run": {"seed": 0, "max_examples": 100},
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "01-smoke-test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
