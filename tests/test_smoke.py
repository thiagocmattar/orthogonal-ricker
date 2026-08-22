from __future__ import annotations

import json
from pathlib import Path

import yaml

from paper_exp.config import load_config
from paper_exp.run import run_smoke


def test_smoke_run_creates_a_completed_artifact_envelope(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    run_dir = run_smoke(
        config,
        config_path=config_path,
        command="pytest smoke",
        run_id="test-run",
    )

    assert run_dir.parent.name == "01-smoke-test"
    assert run_dir.name == "001-test-run"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smoke/num_examples"] == 3
    assert metrics["smoke/passed"] is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "smoke_test"
    assert manifest["config_id"] == "01-smoke-test"
    assert manifest["run_id"] == "001-test-run"
    assert manifest["run_sequence"] == 1
    assert manifest["seed"] == 0
    assert manifest["status"] == "completed"
    assert manifest["started_at"] == manifest["timestamp"]
    assert manifest["finished_at"] >= manifest["started_at"]


def _write_temp_config(tmp_path: Path) -> Path:
    config = {
        "experiment_name": "smoke_test",
        "model": {
            "provider": "TODO: provider",
            "name": "TODO: model name",
            "architecture": "TODO: model architecture",
            "initialization": "random",
        },
        "data": {"name": "TODO: dataset", "split": "TODO: split"},
        "evaluation": {"metric": "smoke_pass"},
        "run": {"seed": 0, "max_examples": 3},
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "01-smoke-test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
