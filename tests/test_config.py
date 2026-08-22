from __future__ import annotations

from typing import Any

import pytest

from paper_exp.config import (
    ConfigError,
    load_config,
    validate_config,
    validate_config_filename,
    validate_training_config,
)


TOPOLOGY_IDS = (
    "A0",
    "A1-H",
    "A2",
    "A3",
    "A4-Q",
    "A4-K",
    "A4-V",
    "A5-QK-PRE",
    "A5-QK-POST",
    "A6-PRE",
    "A6-POST",
)


def test_load_smoke_config() -> None:
    config = load_config("configs/00-smoke.yaml", allow_todos=True)

    assert config["experiment_name"] == "harness_smoke"
    assert config["model"]["initialization"] == "random"
    assert isinstance(config["model"]["provider"], str)
    assert config["model"]["provider"].strip()


def test_required_config_fields_are_checked() -> None:
    config = {
        "experiment_name": "missing_fields",
        "model": {"provider": "huggingface", "name": "TODO_MODEL_NAME"},
    }

    with pytest.raises(ConfigError, match="Missing required config field"):
        validate_config(config, allow_todos=True)


def test_todo_placeholders_can_be_rejected() -> None:
    config = _base_config()
    config["model"]["name"] = "TODO_MODEL"

    with pytest.raises(ConfigError, match="TODO placeholders"):
        validate_config(config, allow_todos=False)


def test_config_filenames_must_be_numbered() -> None:
    validate_config_filename("100-diagnostic.yaml")

    with pytest.raises(ConfigError, match="at least two digits"):
        validate_config_filename("baseline.yaml")
    with pytest.raises(ConfigError, match=r"\.yaml"):
        validate_config_filename("100-diagnostic.yml")


def test_pretraining_configs_must_use_random_initialization() -> None:
    config = _base_config()
    config["model"]["initialization"] = "pretrained"

    with pytest.raises(ConfigError, match="initialization"):
        validate_config(config, allow_todos=False)


@pytest.mark.parametrize("topology_id", TOPOLOGY_IDS)
def test_all_eleven_topology_ids_are_accepted(topology_id: str) -> None:
    config = _base_config()
    config["model"].update(
        {
            "topology_id": topology_id,
            "site_gate": None if topology_id == "A0" else {"operator": "relu"},
        }
    )

    validate_config(config, allow_todos=False)


@pytest.mark.parametrize(
    "site_gate",
    [
        {"operator": "relu"},
        {"operator": "one_sided_threshold", "kappa": 0.0},
        {"operator": "symmetric_threshold", "kappa": 0.1},
    ],
)
def test_active_topology_accepts_only_canonical_site_gate_operators(
    site_gate: dict[str, object],
) -> None:
    config = _base_config()
    config["model"].update({"topology_id": "A2", "site_gate": site_gate})

    validate_config(config, allow_todos=False)


@pytest.mark.parametrize(
    ("topology_id", "site_gate", "message"),
    [
        (None, None, "topology_id"),
        ("a2", {"operator": "relu"}, "topology_id"),
        ("A6", {"operator": "relu"}, "topology_id"),
        ("A0", {"operator": "relu"}, "must be null"),
        ("A2", None, "explicit mapping"),
        ("A2", True, "explicit mapping"),
        ("A2", {}, "operator"),
        ("A2", {"operator": []}, "operator"),
        ("A2", {"operator": "unknown"}, "operator"),
        ("A2", {"operator": "relu", "kappa": 0.1}, "must be omitted"),
        ("A2", {"operator": "one_sided_threshold"}, "kappa is required"),
        ("A2", {"operator": "symmetric_threshold"}, "kappa is required"),
        (
            "A2",
            {"operator": "one_sided_threshold", "kappa": -0.1},
            "finite non-negative",
        ),
        (
            "A2",
            {"operator": "symmetric_threshold", "kappa": float("inf")},
            "finite non-negative",
        ),
        (
            "A2",
            {"operator": "symmetric_threshold", "kappa": True},
            "finite non-negative",
        ),
        (
            "A2",
            {"operator": "relu", "enabled": True},
            "unsupported fields",
        ),
    ],
)
def test_topology_and_site_gate_validation_is_strict(
    topology_id: object,
    site_gate: object,
    message: str,
) -> None:
    config = _base_config()
    config["model"].update(
        {"topology_id": topology_id, "site_gate": site_gate}
    )

    with pytest.raises(ConfigError, match=message):
        validate_config(config, allow_todos=False)


def test_partial_topology_configuration_is_rejected() -> None:
    only_topology = _base_config()
    only_topology["model"]["topology_id"] = "A2"
    with pytest.raises(ConfigError, match="site_gate"):
        validate_config(only_topology, allow_todos=False)

    only_gate = _base_config()
    only_gate["model"]["site_gate"] = {"operator": "relu"}
    with pytest.raises(ConfigError, match="topology_id"):
        validate_config(only_gate, allow_todos=False)


def test_a2_relu_configuration_is_explicit_and_valid() -> None:
    config = _base_config()
    config["model"].update(
        {"topology_id": "A2", "site_gate": {"operator": "relu"}}
    )

    validate_config(config, allow_todos=False)


def test_seed_schedule_and_validation_partition_fields_are_validated() -> None:
    config = _base_config(topology_id="A0", site_gate=None)
    config["run"].update(
        {
            "model_initialization_seed": 0,
            "data_order_seed": 11,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
            "training_schedule_hash": "a" * 64,
        }
    )
    config["validation"] = {
        "enabled": True,
        "split": "validation",
        "max_documents": 500,
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "partition_hash": "b" * 64,
    }

    validate_config(config, allow_todos=False)

    config["run"]["training_schedule_hash"] = "not-a-hash"
    with pytest.raises(ConfigError, match="training_schedule_hash"):
        validate_config(config, allow_todos=False)


def test_reproducibility_fields_require_explicit_supported_schedule() -> None:
    config = _base_config(topology_id="A0", site_gate=None)
    config["run"].update(
        {
            "model_initialization_seed": 0,
            "data_order_seed": 11,
            "training_schedule_hash": "a" * 64,
        }
    )

    with pytest.raises(ConfigError, match="require run.training_schedule_scheme"):
        validate_config(config, allow_todos=False)

    config["run"]["training_schedule_scheme"] = "future_schedule_v2"
    with pytest.raises(ConfigError, match="training_schedule_scheme"):
        validate_config(config, allow_todos=False)


def test_model_initialization_seed_must_match_run_seed() -> None:
    config = _base_config(topology_id="A0", site_gate=None)
    config["run"].update(
        {
            "model_initialization_seed": 3,
            "data_order_seed": 0,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
        }
    )

    with pytest.raises(ConfigError, match="must equal"):
        validate_config(config, allow_todos=False)


def test_definitive_training_config_requires_explicit_pinned_inputs() -> None:
    config = _definitive_training_config()

    validate_training_config(config)

    config["model"]["revision"] = "main"
    with pytest.raises(ConfigError, match="immutable"):
        validate_training_config(config)


@pytest.mark.parametrize("field", ["topology_id", "site_gate"])
def test_definitive_training_config_requires_explicit_topology_fields(field: str) -> None:
    config = _definitive_training_config()
    del config["model"][field]

    with pytest.raises(ConfigError, match=field):
        validate_training_config(config)


def test_definitive_training_config_rejects_missing_scientific_field() -> None:
    config = _definitive_training_config()
    del config["training"]["learning_rate"]

    with pytest.raises(ConfigError, match="training.learning_rate"):
        validate_training_config(config)


def test_definitive_training_config_requires_canonical_pressure_sites() -> None:
    config = _definitive_training_config()
    config["activation_pressure"]["sites"] = ["unknown"]

    with pytest.raises(ConfigError, match="Unsupported transformer site alias"):
        validate_training_config(config)


def _base_config(
    *,
    topology_id: str | None = None,
    site_gate: dict[str, object] | None = None,
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "provider": "huggingface",
        "name": "test-random-model",
        "architecture": "test/architecture",
        "initialization": "random",
    }
    if topology_id is not None:
        model["topology_id"] = topology_id
        model["site_gate"] = site_gate
    return {
        "experiment_name": "topology_validation_test",
        "model": model,
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }


def _definitive_training_config() -> dict[str, Any]:
    return {
        "experiment_name": "definitive_validation_test",
        "model": {
            "provider": "huggingface",
            "name": "random-model",
            "architecture": "test/architecture",
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
            "max_documents": None,
        },
        "tokenizer": {"name": "test/tokenizer", "revision": "c" * 40},
        "preprocessing": {
            "output_dir": "data/tokenized",
            "cache_id": "definitive-test",
            "block_size": 128,
            "append_eos": True,
            "overwrite": False,
        },
        "evaluation": {"metric": "validation_loss"},
        "run": {
            "seed": 11,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
            "model_initialization_seed": 11,
            "data_order_seed": 29,
            "training_schedule_hash": None,
        },
        "training": {
            "device": "cuda",
            "precision": "bfloat16",
            "max_steps": 100,
            "max_wall_seconds": 3600,
            "learning_rate": 0.001,
            "warmup_steps": 10,
            "gradient_accumulation_steps": 2,
            "micro_batch_size": 4,
            "log_every": 5,
            "optimizer": "adamw",
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1.0e-8,
            "weight_decay": 0.01,
        },
        "validation": {"enabled": False},
        "checkpoint": {"save_final": True, "save_optimizer": False},
        "activation_pressure": {
            "enabled": True,
            "method": "none",
            "sites": ["h"],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001, 0.01],
        },
        "output": {"dir": "results"},
    }
