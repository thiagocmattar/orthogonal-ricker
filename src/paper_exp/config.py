from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config is missing required experiment information."""


REQUIRED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("experiment_name",),
    ("model", "provider"),
    ("model", "name"),
    ("model", "architecture"),
    ("model", "initialization"),
    ("data", "name"),
    ("data", "split"),
    ("evaluation", "metric"),
    ("run", "seed"),
    ("run", "max_examples"),
    ("output", "dir"),
)

CONFIG_FILE_RE = re.compile(r"^\d{2,}-[a-z0-9][a-z0-9-]*\.ya?ml$")


def load_config(path: str | Path, *, allow_todos: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    validate_config_filename(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")

    validate_config(data, allow_todos=allow_todos)
    return data


def validate_config_filename(path: str | Path) -> None:
    name = Path(path).name
    if CONFIG_FILE_RE.match(name) is None:
        raise ConfigError(
            "Config filenames must start with at least two digits, like 01-baseline.yaml or 100-diagnostic.yaml."
        )


def validate_config(config: Mapping[str, Any], *, allow_todos: bool = True) -> None:
    if not isinstance(config, Mapping):
        raise ConfigError("Config must be a mapping.")

    for field_path in REQUIRED_FIELDS:
        _get_required(config, field_path)

    seed = _get_required(config, ("run", "seed"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("Config field run.seed must be an integer.")

    max_examples = _get_required(config, ("run", "max_examples"))
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples <= 0:
        raise ConfigError("Config field run.max_examples must be a positive integer.")

    initialization = _get_required(config, ("model", "initialization"))
    if initialization != "random":
        raise ConfigError("Config field model.initialization must be 'random' for pretraining runs.")

    hidden_act = config.get("model", {}).get("hidden_act")
    if hidden_act is not None and (not isinstance(hidden_act, str) or not hidden_act.strip()):
        raise ConfigError("Config field model.hidden_act must be a non-empty string when provided.")

    post_layernorm_relu = config.get("model", {}).get("post_layernorm_relu")
    if post_layernorm_relu is not None and not isinstance(post_layernorm_relu, bool):
        raise ConfigError("Config field model.post_layernorm_relu must be a boolean when provided.")

    if not allow_todos:
        todos = list(find_todo_values(config))
        if todos:
            fields = ", ".join(f"{path}={value}" for path, value in todos)
            raise ConfigError(f"Config contains TODO placeholders: {fields}")


def find_todo_values(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str) and value.strip().upper().startswith("TODO"):
        found.append((prefix or "<root>", value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_todo_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            found.extend(find_todo_values(child, child_prefix))
    return found


def _get_required(config: Mapping[str, Any], field_path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in field_path:
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigError(f"Missing required config field: {'.'.join(field_path)}")
        current = current[key]
    return current
