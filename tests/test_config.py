from __future__ import annotations

import pytest

from paper_exp.config import ConfigError, load_config, validate_config, validate_config_filename


def test_load_baseline_config_allows_todos() -> None:
    config = load_config("configs/01-pythia-14m-minipile-smoke.yaml", allow_todos=True)

    assert config["experiment_name"] == "pythia_14m_minipile_smoke"
    assert config["model"]["provider"] == "huggingface"
    assert config["model"]["name"] == "pythia-14m-random"
    assert config["model"]["architecture"] == "EleutherAI/pythia-14m-deduped"
    assert config["model"]["initialization"] == "random"


def test_required_config_fields_are_checked() -> None:
    config = {
        "experiment_name": "missing_fields",
        "model": {"provider": "huggingface", "name": "TODO_MODEL_NAME"},
    }

    with pytest.raises(ConfigError, match="Missing required config field"):
        validate_config(config, allow_todos=True)


def test_pythia_smoke_config_has_no_todos() -> None:
    config = load_config("configs/01-pythia-14m-minipile-smoke.yaml", allow_todos=True)

    validate_config(config, allow_todos=False)


def test_todo_placeholders_can_be_rejected() -> None:
    config = {
        "experiment_name": "todo_config",
        "model": {
            "provider": "huggingface",
            "name": "TODO_MODEL",
            "architecture": "TODO_MODEL_ARCHITECTURE",
            "initialization": "random",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="TODO placeholders"):
        validate_config(config, allow_todos=False)


def test_config_filenames_must_be_numbered() -> None:
    with pytest.raises(ConfigError, match="at least two digits"):
        validate_config_filename("baseline.yaml")


def test_config_filenames_allow_three_digit_prefixes() -> None:
    validate_config_filename("100-diagnostic.yaml")


def test_pretraining_configs_must_use_random_initialization() -> None:
    config = {
        "experiment_name": "bad_init",
        "model": {
            "provider": "huggingface",
            "name": "pythia-14m",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "pretrained",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
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
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "random",
            "hidden_act": "",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
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
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "random",
            "post_layernorm_relu": "yes",
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }

    with pytest.raises(ConfigError, match="post_layernorm_relu"):
        validate_config(config, allow_todos=False)


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
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "initialization": "random",
            "post_qkv_relu": post_qkv_relu,
        },
        "data": {"name": "JeanKaddour/minipile", "split": "train"},
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1},
        "output": {"dir": "results"},
    }
