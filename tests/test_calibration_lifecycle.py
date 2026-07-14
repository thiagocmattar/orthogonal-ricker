from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import paper_exp.calibration as calibration
from paper_exp.config import load_config


def test_calibration_dependency_failure_preserves_launch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        "configs/01-pythia-14m-minipile-smoke.yaml",
        allow_todos=False,
    )
    config["output"]["dir"] = str(tmp_path / "results")
    config_path = tmp_path / "01-calibration-failure.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    def fail_dependencies() -> None:
        raise RuntimeError("dependency load failed")

    monkeypatch.setattr(
        calibration,
        "_load_training_dependencies",
        fail_dependencies,
    )

    with pytest.raises(RuntimeError, match="dependency load failed"):
        calibration.run_calibration(
            config,
            config_path=config_path,
            command="pytest calibration lifecycle",
            mode="pretrain",
            run_id="failure",
        )

    run_dir = tmp_path / "results" / "01-calibration-failure" / "001-failure"
    assert (run_dir / "config.yaml").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["mode"] == "pretrain"
    assert manifest["command"] == "pytest calibration lifecycle"
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "dependency load failed",
    }
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()
