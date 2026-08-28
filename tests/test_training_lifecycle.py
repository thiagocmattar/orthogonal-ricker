from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import paper_exp.training as training


def test_training_wall_limit_is_fixed_and_calibration_only() -> None:
    assert training.CALIBRATION_TRAINING_WALL_SECONDS == 600.0
    assert training._training_wall_limit_seconds("calibrate") == 600.0
    assert training._training_wall_limit_seconds("pretrain") is None

    assert training._reached_training_wall_limit(
        600.0,
        training_elapsed=600.0,
        completed_steps=1,
        max_steps=2,
    )
    assert not training._reached_training_wall_limit(
        None,
        training_elapsed=10_000.0,
        completed_steps=1,
        max_steps=2,
    )
    assert not training._reached_training_wall_limit(
        600.0,
        training_elapsed=600.0,
        completed_steps=2,
        max_steps=2,
    )


def test_training_phase_timing_metrics_are_recorded_separately() -> None:
    assert training._phase_timing_metrics(
        setup=1.0,
        training=2.0,
        validation=3.0,
        diagnostic=4.0,
        checkpoint=5.0,
        training_sample=9.0,
        total=15.0,
    ) == {
        "wall_seconds_setup": 1.0,
        "wall_seconds_train": 2.0,
        "wall_seconds_validation": 3.0,
        "wall_seconds_diagnostic": 4.0,
        "wall_seconds_checkpoint": 5.0,
        "wall_seconds_training_sample": 9.0,
        "wall_seconds_total": 15.0,
    }


def test_ol1_terminal_counters_cover_every_optimizer_boundary() -> None:
    counters = training._empty_ol1_boundary_counters()
    step_results = (
        {
            "pressure/pressure_conflict": True,
            "pressure/pressure_update_projected": False,
            "pressure/pressure_update_applied_scale": 1.0,
            "pressure/eligible_parameters": 7,
            "pressure/skipped_parameters": 2,
        },
        {
            "pressure/pressure_conflict": False,
            "pressure/pressure_update_projected": True,
            "pressure/pressure_update_applied_scale": 0.25,
            "pressure/eligible_parameters": 8,
            "pressure/skipped_parameters": 1,
        },
        {
            "pressure/pressure_conflict": True,
            "pressure/pressure_update_projected": True,
            "pressure/pressure_update_applied_scale": 0.75,
            "pressure/eligible_parameters": 9,
            "pressure/skipped_parameters": 0,
        },
    )
    for step_result in step_results:
        training._accumulate_ol1_boundary_counters(counters, step_result)

    terminal = training._final_ol1_boundary_counters(
        counters,
        completed_steps=len(step_results),
    )
    assert terminal == {
        "ol1/optimizer_boundary_count": 3,
        "ol1/raw_gradient_conflict_boundary_count": 2,
        "ol1/preconditioned_projection_boundary_count": 2,
        "ol1/trust_budget_limited_boundary_count": 2,
        "ol1/eligible_parameter_tensor_count_sum": 24,
        "ol1/skipped_parameter_tensor_count_sum": 3,
    }
    assert json.loads(json.dumps(terminal)) == terminal


def test_ol1_terminal_counters_require_complete_boundary_coverage() -> None:
    counters = training._empty_ol1_boundary_counters()

    with pytest.raises(RuntimeError, match="coverage does not match completed steps"):
        training._final_ol1_boundary_counters(counters, completed_steps=1)


@pytest.mark.parametrize(
    ("method", "scope"),
    [
        ("none", "task_only"),
        ("l1_naive", "task_plus_weighted_pressure"),
        ("orthogonal_l1", "task_only"),
    ],
)
def test_gradient_clipping_manifest_records_fixed_method_scope(
    method: str,
    scope: str,
) -> None:
    assert training._gradient_clipping_manifest(method) == {
        "type": "global_l2_norm",
        "max_norm": 1.0,
        "gradient_scope": scope,
        "applied_immediately_before": "adamw_step",
        "error_if_nonfinite": True,
        "orthogonal_pressure_direction_included": False,
    }


def test_checkpoint_manifest_path_is_run_relative(tmp_path: Path) -> None:
    class Model:
        def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization is True
            (path / "config.json").write_text("{}\n", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"model")

    config = {"checkpoint": {"save_final": True, "save_optimizer": False}}
    metadata = training._save_final_checkpoint(
        config,
        tmp_path / "run",
        Model(),
        object(),
        object(),
    )

    assert metadata["saved"] is True
    assert metadata["path"] == "checkpoints/final"


def test_saved_checkpoint_confirmation_records_identity_coverage_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "source" / "checkpoints" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    tokens = np.arange(23, dtype=np.int32)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.inputs: list[np.ndarray] = []

        def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> object:
            del labels
            self.inputs.append(input_ids.detach().cpu().numpy().copy())
            loss = self.weight * 0.0 + math.log(4.0)
            return type("Output", (), {"loss": loss})()

    model = Model()

    def load_model(auto_model: object, checkpoint_path: Path, *, torch: object) -> Model:
        del auto_model, torch
        assert checkpoint_path == checkpoint.resolve()
        return model

    monkeypatch.setattr(training, "load_checkpoint_model", load_model)
    metadata = _confirmation_metadata(tmp_path, token_count=len(tokens), block_size=4)
    result = training.evaluate_saved_checkpoint_confirmation(
        checkpoint_path=checkpoint,
        source_identity={
            "tranche_id": "01-a1-lr-screen",
            "config_id": "001-a1-lr-5e-4",
            "run_id": "001-test",
        },
        provenance={"git_commit": "a" * 40, "config_sha256": "b" * 64},
        validation_metadata=metadata,
        tokens=tokens,
        batch_size=4,
        torch=torch,
        np=np,
        auto_model=object(),
        device=torch.device("cpu"),
        dtype=None,
    )

    assert result["kind"] == "saved_checkpoint_confirmation_validation"
    assert result["source"] == {
        "tranche_id": "01-a1-lr-screen",
        "config_id": "001-a1-lr-5e-4",
        "run_id": "001-test",
    }
    assert result["checkpoint"]["path"] == str(checkpoint.resolve())
    assert len(result["checkpoint"]["content_sha256"]) == 64
    assert len(result["checkpoint"]["parameter_sha256"]) == 64
    assert result["validation_cache"] == {
        field: metadata[field] for field in training.VALIDATION_CACHE_IDENTITY_FIELDS
    }
    assert result["coverage"] == {
        "fixed_order": True,
        "expected_complete_sequences": 5,
        "evaluated_sequences": 5,
        "sequence_length": 4,
        "evaluated_tokens": 20,
        "evaluated_batches": 2,
        "excluded_tail_tokens": 3,
        "complete": True,
    }
    assert result["metrics"]["validation_loss"] == pytest.approx(math.log(4.0))
    assert result["metrics"]["perplexity"] == pytest.approx(4.0)
    assert all(result["completeness"].values())
    assert result["provenance"] == {
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
    }
    assert np.array_equal(
        np.concatenate(model.inputs),
        np.stack([tokens[start : start + 4] for start in (0, 4, 8, 12, 16)]),
    )


def test_saved_checkpoint_confirmation_requires_exact_source_and_confirmation_cache(
    tmp_path: Path,
) -> None:
    metadata = _confirmation_metadata(tmp_path, token_count=8, block_size=4)
    with pytest.raises(ValueError, match="exact source identity"):
        training._exact_source_identity(
            {"tranche_id": "01-a1-lr-screen", "config_id": "001-a1-lr-5e-4"}
        )

    metadata["partition"] = "selection"
    with pytest.raises(ValueError, match="requires partition confirmation"):
        training._confirmation_cache_identity(metadata, token_count=8)


def test_saved_checkpoint_confirmation_rejects_cache_token_count_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _confirmation_metadata(tmp_path, token_count=8, block_size=4)
    with pytest.raises(ValueError, match="token count does not match"):
        training._confirmation_cache_identity(metadata, token_count=9)


def test_training_cache_hash_must_match_explicit_config_pin() -> None:
    digest = "a" * 64
    training._require_expected_cache_hash(
        {"tokens_sha256": digest},
        digest,
        context="Training token cache",
    )

    with pytest.raises(ValueError, match="hash does not match config"):
        training._require_expected_cache_hash(
            {"tokens_sha256": "b" * 64},
            digest,
            context="Training token cache",
        )


def _confirmation_metadata(
    tmp_path: Path,
    *,
    token_count: int,
    block_size: int,
) -> dict[str, object]:
    return {
        "tokens_path": str(tmp_path / "confirmation.bin"),
        "dtype": "int32",
        "block_size": block_size,
        "tokens": token_count,
        "tokens_bytes": token_count * 4,
        "tokens_sha256": "c" * 64,
        "partition": "confirmation",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "source_documents": 500,
        "source_document_indices_sha256": "d" * 64,
    }


def test_training_dependency_failure_preserves_launch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "experiment_name": "calibration_lifecycle_test",
        "identity": {
            "group_id": "A1-lr-screen",
            "condition_fingerprint": "d" * 64,
            "training_implementation_id": "a1_pretraining_v1",
        },
        "model": {
            "provider": "huggingface",
            "name": "pythia-test-random",
            "architecture": "test/pythia",
            "revision": "a" * 40,
            "initialization": "random",
            "topology_id": "A0",
            "site_gate": None,
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
            "training_schedule_scheme": training.TRAINING_SCHEDULE_SCHEME,
            "model_initialization_seed": 0,
            "data_order_seed": 1,
            "training_schedule_hash": "e" * 64,
        },
        "training": {
            "device": "cpu",
            "precision": "float32",
            "max_steps": 1,
            "learning_rate": 0.001,
            "warmup_steps": 1,
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
            "sites": ["h"],
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
