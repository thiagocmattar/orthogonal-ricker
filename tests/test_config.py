from __future__ import annotations

import pytest

from paper_exp.config import ConfigError, load_config, validate_config, validate_config_filename


def test_load_baseline_config_allows_todos() -> None:
    config = load_config("configs/01-baseline.yaml", allow_todos=True)

    assert config["experiment_name"] == "baseline"
    assert config["model"]["provider"] == "huggingface"


def test_required_config_fields_are_checked() -> None:
    config = {
        "experiment_name": "missing_fields",
        "model": {"provider": "huggingface", "name": "TODO_MODEL_NAME"},
    }

    with pytest.raises(ConfigError, match="Missing required config field"):
        validate_config(config, allow_todos=True)


def test_todo_placeholders_can_be_rejected() -> None:
    config = load_config("configs/01-baseline.yaml", allow_todos=True)

    with pytest.raises(ConfigError, match="TODO placeholders"):
        validate_config(config, allow_todos=False)


def test_config_filenames_must_be_numbered() -> None:
    with pytest.raises(ConfigError, match="Config filenames must be numbered"):
        validate_config_filename("baseline.yaml")
