from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import paper_exp.training as training


def test_training_dependency_failure_preserves_launch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "experiment_name": "calibration_lifecycle_test",
        "model": {
            "provider": "huggingface",
            "name": "pythia-test-random",
            "architecture": "test/pythia",
            "revision": "a" * 40,
            "initialization": "random",
        },
        "data": {
            "name": "test/data",
            "revision": "b" * 40,
            "split": "train",
            "text_column": "text",
            "max_documents": 8,
        },
        "tokenizer": {"name": "test/tokenizer", "revision": "c" * 40},
        "preprocessing": {
            "output_dir": str(tmp_path / "data" / "tokenized"),
            "cache_id": "test-cache",
            "block_size": 16,
            "append_eos": True,
            "overwrite": False,
        },
        "evaluation": {"metric": "loss"},
        "run": {
            "seed": 0,
            "max_examples": 8,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
            "model_initialization_seed": 0,
            "data_order_seed": 1,
            "training_schedule_hash": None,
        },
        "training": {
            "device": "cpu",
            "precision": "float32",
            "max_steps": 1,
            "max_wall_seconds": None,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            "log_every": 1,
            "optimizer": "adamw",
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "validation": {"enabled": False},
        "checkpoint": {"save_final": False, "save_optimizer": False},
        "activation_pressure": {
            "enabled": True,
            "method": "none",
            "sites": ["mlp_hiddens"],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001],
        },
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "01-calibration-failure.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    def fail_dependencies() -> None:
        raise RuntimeError("dependency load failed")

    monkeypatch.setattr(
        training,
        "_load_training_dependencies",
        fail_dependencies,
    )

    with pytest.raises(RuntimeError, match="dependency load failed"):
        training.run_training(
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
