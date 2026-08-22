from __future__ import annotations

import pytest

from paper_exp.config import (
    ConfigError,
    load_config,
    validate_config,
    validate_config_filename,
    validate_training_config,
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
    config = {
        "experiment_name": "todo_config",
        "model": {
            "provider": "huggingface",
            "name": "TODO_MODEL",
            "architecture": "TODO_MODEL_ARCHITECTURE",
            "initialization": "random",
        },
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="TODO placeholders"):
        validate_config(config, allow_todos=False)


def test_config_filenames_must_be_numbered() -> None:
    validate_config_filename("100-diagnostic.yaml")

    with pytest.raises(ConfigError, match="at least two digits"):
        validate_config_filename("baseline.yaml")
    with pytest.raises(ConfigError, match=r"\.yaml"):
        validate_config_filename("100-diagnostic.yml")


def test_pretraining_configs_must_use_random_initialization() -> None:
    config = {
        "experiment_name": "bad_init",
        "model": {
            "provider": "huggingface",
            "name": "test-random-model",
            "architecture": "test/architecture",
            "initialization": "pretrained",
        },
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="initialization"):
        validate_config(config, allow_todos=False)


def test_optional_hidden_act_must_be_non_empty_string() -> None:
    config = {
        "experiment_name": "bad_hidden_act",
        "model": {
            "provider": "huggingface",
            "name": "test-random-model",
            "architecture": "test/architecture",
            "initialization": "random",
            "hidden_act": "",
        },
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="hidden_act"):
        validate_config(config, allow_todos=False)


def test_optional_post_layernorm_relu_must_be_boolean() -> None:
    config = {
        "experiment_name": "bad_post_layernorm_relu",
        "model": {
            "provider": "huggingface",
            "name": "test-random-model",
            "architecture": "test/architecture",
            "initialization": "random",
            "post_layernorm_relu": "yes",
        },
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="post_layernorm_relu"):
        validate_config(config, allow_todos=False)


def test_seed_schedule_and_validation_partition_fields_are_validated() -> None:
    config = _post_qkv_config(None)
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
    config = _post_qkv_config(None)
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
    config = _post_qkv_config(None)
    config["run"].update(
        {
            "model_initialization_seed": 3,
            "data_order_seed": 0,
            "training_schedule_scheme": "random_contiguous_blocks_with_replacement_v1",
        }
    )

    with pytest.raises(ConfigError, match="must equal"):
        validate_config(config, allow_todos=False)


def test_fixed_one_sided_branch_gates_are_accepted() -> None:
    config = _post_qkv_config(None)
    config["model"].update(
        {
            "hidden_act": "relu",
            "post_layernorm_relu": True,
            "post_layernorm_gate": {
                "gate_type": "one_sided_threshold",
                "kappa": 0.1,
            },
            "mlp_hidden_gate": {
                "gate_type": "one_sided_threshold",
                "kappa": 0.1,
            },
        }
    )

    validate_config(config, allow_todos=False)


@pytest.mark.parametrize("field", ["post_layernorm_gate", "mlp_hidden_gate"])
@pytest.mark.parametrize(
    ("gate", "message"),
    [
        (True, "mapping"),
        ({"gate_type": "relu", "kappa": 0.1}, "one_sided_threshold"),
        ({"gate_type": "one_sided_threshold"}, "kappa"),
        ({"gate_type": "one_sided_threshold", "kappa": -0.1}, "non-negative"),
        ({"gate_type": "one_sided_threshold", "kappa": float("inf")}, "finite"),
        ({"gate_type": "one_sided_threshold", "kappa": True}, "finite"),
        (
            {"gate_type": "one_sided_threshold", "kappa": 0.1, "enabled": True},
            "unsupported",
        ),
    ],
)
def test_fixed_one_sided_branch_gates_reject_invalid_mappings(
    field: str,
    gate: object,
    message: str,
) -> None:
    config = _post_qkv_config(None)
    config["model"].update(
        {
            "hidden_act": "relu",
            "post_layernorm_relu": True,
            field: gate,
        }
    )

    with pytest.raises(ConfigError, match=message):
        validate_config(config, allow_todos=False)


def test_fixed_one_sided_branch_gates_require_active_base_relu_sites() -> None:
    post_layernorm = _post_qkv_config(None)
    post_layernorm["model"]["post_layernorm_gate"] = {
        "gate_type": "one_sided_threshold",
        "kappa": 0.1,
    }
    with pytest.raises(ConfigError, match="post_layernorm_relu"):
        validate_config(post_layernorm, allow_todos=False)

    mlp_hidden = _post_qkv_config(None)
    mlp_hidden["model"]["mlp_hidden_gate"] = {
        "gate_type": "one_sided_threshold",
        "kappa": 0.1,
    }
    with pytest.raises(ConfigError, match="hidden_act"):
        validate_config(mlp_hidden, allow_todos=False)


@pytest.mark.parametrize("placement", ["pre_rope", "post_rope"])
def test_post_qkv_relu_accepts_both_qk_placements(placement: str) -> None:
    config = _post_qkv_config(
        {
            "enabled": True,
            "query": True,
            "key": True,
            "value": True,
            "qk_placement": placement,
        }
    )

    validate_config(config, allow_todos=False)


def test_post_qkv_relu_accepts_fixed_symmetric_threshold() -> None:
    config = _post_qkv_config(
        {
            "enabled": True,
            "query": True,
            "key": True,
            "value": True,
            "qk_placement": "post_rope",
            "gate_type": "symmetric_threshold",
            "kappa": 0.1,
        }
    )

    validate_config(config, allow_todos=False)


@pytest.mark.parametrize("placement", ["pre_rope", "post_rope"])
def test_post_qkv_relu_accepts_fixed_one_sided_threshold(placement: str) -> None:
    config = _post_qkv_config(
        {
            "enabled": True,
            "query": True,
            "key": True,
            "value": False,
            "qk_placement": placement,
            "gate_type": "one_sided_threshold",
            "kappa": 0.1,
        }
    )

    validate_config(config, allow_todos=False)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"gate_type": "unknown"}, "gate_type"),
        ({"gate_type": "symmetric_threshold"}, "kappa"),
        ({"gate_type": "symmetric_threshold", "kappa": -0.1}, "non-negative"),
        ({"gate_type": "symmetric_threshold", "kappa": float("inf")}, "finite"),
        ({"gate_type": "symmetric_threshold", "kappa": True}, "finite"),
        ({"gate_type": "one_sided_threshold"}, "kappa"),
        ({"gate_type": "one_sided_threshold", "kappa": -0.1}, "non-negative"),
        ({"gate_type": "relu", "kappa": 0.1}, "must be omitted"),
    ],
)
def test_post_qkv_relu_rejects_invalid_gate_configuration(
    extra: dict[str, object],
    message: str,
) -> None:
    post_qkv_relu: dict[str, object] = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
    }
    post_qkv_relu.update(extra)

    with pytest.raises(ConfigError, match=message):
        validate_config(_post_qkv_config(post_qkv_relu), allow_todos=False)


@pytest.mark.parametrize(
    ("post_qkv_relu", "message"),
    [
        (True, "must be a mapping"),
        (
            {"enabled": True, "query": True, "key": True, "value": True},
            "qk_placement",
        ),
        (
            {
                "enabled": True,
                "query": True,
                "key": True,
                "value": True,
                "qk_placement": "between_rope",
            },
            "qk_placement",
        ),
        (
            {
                "enabled": True,
                "query": "yes",
                "key": True,
                "value": True,
                "qk_placement": "pre_rope",
            },
            "query",
        ),
    ],
)
def test_post_qkv_relu_rejects_invalid_mappings(
    post_qkv_relu: object,
    message: str,
) -> None:
    config = _post_qkv_config(post_qkv_relu)

    with pytest.raises(ConfigError, match=message):
        validate_config(config, allow_todos=False)


def test_disabled_post_qkv_relu_rejects_a_qk_placement() -> None:
    config = _post_qkv_config(
        {
            "enabled": False,
            "query": False,
            "key": False,
            "value": False,
            "qk_placement": "pre_rope",
        }
    )

    with pytest.raises(ConfigError, match="must be omitted"):
        validate_config(config, allow_todos=False)


def _post_qkv_config(post_qkv_relu: object) -> dict[str, object]:
    return {
        "experiment_name": "post_qkv_relu_test",
        "model": {
            "provider": "huggingface",
            "name": "test-random-model",
            "architecture": "test/architecture",
            "initialization": "random",
            "post_qkv_relu": post_qkv_relu,
        },
        "data": {"name": "test/dataset", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }


def test_definitive_training_config_requires_explicit_pinned_inputs() -> None:
    config = _definitive_training_config()

    validate_training_config(config)

    config["model"]["revision"] = "main"
    with pytest.raises(ConfigError, match="immutable"):
        validate_training_config(config)


def test_definitive_training_config_rejects_missing_scientific_field() -> None:
    config = _definitive_training_config()
    del config["training"]["learning_rate"]

    with pytest.raises(ConfigError, match="training.learning_rate"):
        validate_training_config(config)


def _definitive_training_config() -> dict[str, object]:
    return {
        "experiment_name": "definitive_validation_test",
        "model": {
            "provider": "huggingface",
            "name": "random-model",
            "architecture": "test/architecture",
            "revision": "a" * 40,
            "initialization": "random",
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
            "sites": ["mlp_hiddens"],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001, 0.01],
        },
        "output": {"dir": "results"},
    }
