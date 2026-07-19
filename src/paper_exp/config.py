from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

import yaml

from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME


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

    _validate_campaign_reproducibility(config, seed=seed)

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

    post_layernorm_gate = config.get("model", {}).get("post_layernorm_gate")
    _validate_one_sided_gate(post_layernorm_gate, field_path="model.post_layernorm_gate")
    if post_layernorm_gate is not None and post_layernorm_relu is not True:
        raise ConfigError(
            "Config field model.post_layernorm_gate requires model.post_layernorm_relu: true."
        )

    mlp_hidden_gate = config.get("model", {}).get("mlp_hidden_gate")
    _validate_one_sided_gate(mlp_hidden_gate, field_path="model.mlp_hidden_gate")
    if mlp_hidden_gate is not None and str(hidden_act).lower() != "relu":
        raise ConfigError("Config field model.mlp_hidden_gate requires model.hidden_act: relu.")

    _validate_post_qkv_relu(config.get("model", {}).get("post_qkv_relu"))
    _validate_learned_threshold_training_contract(config)

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


def _validate_campaign_reproducibility(config: Mapping[str, Any], *, seed: int) -> None:
    run_config = config.get("run", {})
    schedule_scheme = run_config.get("training_schedule_scheme")
    model_seed = run_config.get("model_initialization_seed")
    data_seed = run_config.get("data_order_seed")
    schedule_hash = run_config.get("training_schedule_hash")

    if schedule_scheme is None:
        campaign_fields = {
            "model_initialization_seed": model_seed,
            "data_order_seed": data_seed,
            "training_schedule_hash": schedule_hash,
        }
        provided = [field for field, value in campaign_fields.items() if value is not None]
        if provided:
            raise ConfigError(
                "Campaign run fields require run.training_schedule_scheme; provided: "
                + ", ".join(provided)
                + "."
            )
    elif schedule_scheme != TRAINING_SCHEDULE_SCHEME:
        raise ConfigError(
            f"Config field run.training_schedule_scheme must be '{TRAINING_SCHEDULE_SCHEME}'."
        )

    if (model_seed is None) != (data_seed is None):
        raise ConfigError(
            "Config fields run.model_initialization_seed and run.data_order_seed must be provided together."
        )
    if schedule_scheme is not None and model_seed is None:
        raise ConfigError(
            "Config fields run.model_initialization_seed and run.data_order_seed are required "
            "when run.training_schedule_scheme is set."
        )
    for field, value in (
        ("model_initialization_seed", model_seed),
        ("data_order_seed", data_seed),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"Config field run.{field} must be an integer.")
    if model_seed is not None and model_seed != seed:
        raise ConfigError(
            "Config field run.seed must equal run.model_initialization_seed for campaign runs."
        )

    if schedule_hash is not None and re.fullmatch(r"[0-9a-f]{64}", str(schedule_hash)) is None:
        raise ConfigError("Config field run.training_schedule_hash must be a lowercase SHA-256 hex digest.")

    validation = config.get("validation", {})
    partition = validation.get("partition")
    if partition is None:
        return
    if partition not in {"selection", "confirmation"}:
        raise ConfigError("Config field validation.partition must be 'selection' or 'confirmation'.")
    if validation.get("partition_scheme") != "shuffled_source_documents_half_v1":
        raise ConfigError(
            "Config field validation.partition_scheme must be 'shuffled_source_documents_half_v1'."
        )
    partition_seed = validation.get("partition_seed")
    if isinstance(partition_seed, bool) or not isinstance(partition_seed, int):
        raise ConfigError("Config field validation.partition_seed must be an integer.")
    max_documents = validation.get("max_documents")
    if isinstance(max_documents, bool) or not isinstance(max_documents, int) or max_documents < 2:
        raise ConfigError(
            "Document-disjoint validation partitions require validation.max_documents >= 2."
        )
    partition_hash = validation.get("partition_hash")
    if partition_hash is not None and re.fullmatch(r"[0-9a-f]{64}", str(partition_hash)) is None:
        raise ConfigError("Config field validation.partition_hash must be a lowercase SHA-256 hex digest.")


def _validate_post_qkv_relu(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigError("Config field model.post_qkv_relu must be a mapping when provided.")

    boolean_fields = ("enabled", "query", "key", "value")
    for field in boolean_fields:
        if field not in value:
            raise ConfigError(f"Missing required config field: model.post_qkv_relu.{field}")
        if not isinstance(value[field], bool):
            raise ConfigError(f"Config field model.post_qkv_relu.{field} must be a boolean.")

    enabled = value["enabled"]
    placement = value.get("qk_placement")
    threshold_fields = {
        "gate_type",
        "kappa",
        "kappa_init",
        "kappa_scope",
        "threshold_scale",
        "surrogate",
        "temperature",
        "rms_epsilon",
    }
    if not enabled:
        extra = set(value) - {"enabled", "query", "key", "value", "qk_placement"} - threshold_fields
        if extra:
            fields = ", ".join(sorted(str(field) for field in extra))
            raise ConfigError(
                f"Disabled config field model.post_qkv_relu contains unsupported fields: {fields}."
            )
        if placement is not None:
            raise ConfigError(
                "Config field model.post_qkv_relu.qk_placement must be omitted when post-QKV ReLU is disabled."
            )
        if any(value[field] for field in ("query", "key", "value")):
            raise ConfigError(
                "Config fields model.post_qkv_relu.query/key/value must be false when post-QKV ReLU is disabled."
            )
        if threshold_fields.intersection(value):
            raise ConfigError(
                "Config fields model.post_qkv_relu gate settings must be omitted when post-QKV ReLU is disabled."
            )
        return

    if placement not in {"pre_rope", "post_rope"}:
        raise ConfigError(
            "Config field model.post_qkv_relu.qk_placement must be 'pre_rope' or 'post_rope'."
        )
    if not any(value[field] for field in ("query", "key", "value")):
        raise ConfigError("Configured model.post_qkv_relu is enabled, but no Q/K/V gate is enabled.")

    gate_type = value.get("gate_type", "relu")
    if gate_type not in {
        "relu",
        "one_sided_threshold",
        "symmetric_threshold",
        "learned_one_sided_threshold",
        "learned_symmetric_threshold",
    }:
        raise ConfigError(
            "Config field model.post_qkv_relu.gate_type must be 'relu', "
            "'one_sided_threshold', 'symmetric_threshold', "
            "'learned_one_sided_threshold', or 'learned_symmetric_threshold'."
        )
    base_fields = {"enabled", "query", "key", "value", "qk_placement", "gate_type"}
    if gate_type == "relu" and threshold_fields.intersection(value) - {"gate_type"}:
        raise ConfigError(
            "Config field model.post_qkv_relu threshold settings must be omitted for ordinary ReLU gates."
        )
    if gate_type == "relu":
        allowed_fields = base_fields
    elif gate_type in {"one_sided_threshold", "symmetric_threshold"}:
        allowed_fields = base_fields | {"kappa"}
    else:
        allowed_fields = base_fields | {
            "kappa_init",
            "kappa_scope",
            "threshold_scale",
            "surrogate",
            "temperature",
            "rms_epsilon",
        }
    extra = set(value) - allowed_fields
    if extra:
        fields = ", ".join(sorted(str(field) for field in extra))
        raise ConfigError(f"Config field model.post_qkv_relu contains unsupported fields: {fields}.")
    if gate_type == "relu":
        return

    if gate_type.startswith("learned_"):
        _validate_learned_gate(value, field_path="model.post_qkv_relu")
        return

    if "kappa" not in value:
        raise ConfigError(
            "Missing required config field: model.post_qkv_relu.kappa for threshold gates."
        )
    kappa = value["kappa"]
    if (
        isinstance(kappa, bool)
        or not isinstance(kappa, (int, float))
        or not math.isfinite(float(kappa))
        or float(kappa) < 0.0
    ):
        raise ConfigError(
            "Config field model.post_qkv_relu.kappa must be a finite non-negative number."
        )


def _validate_one_sided_gate(value: Any, *, field_path: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config field {field_path} must be a mapping when provided.")
    gate_type = value.get("gate_type")
    learned_fields = {
        "gate_type",
        "kappa_init",
        "kappa_scope",
        "threshold_scale",
        "surrogate",
        "temperature",
        "rms_epsilon",
    }
    allowed = learned_fields if gate_type == "learned_one_sided_threshold" else {"gate_type", "kappa"}
    extra = set(value) - allowed
    if extra:
        fields = ", ".join(sorted(str(field) for field in extra))
        raise ConfigError(f"Config field {field_path} contains unsupported fields: {fields}.")
    if gate_type == "learned_one_sided_threshold":
        _validate_learned_gate(value, field_path=field_path)
        return
    if gate_type != "one_sided_threshold":
        raise ConfigError(
            f"Config field {field_path}.gate_type must be 'one_sided_threshold' "
            "or 'learned_one_sided_threshold'."
        )
    if "kappa" not in value:
        raise ConfigError(f"Missing required config field: {field_path}.kappa")
    kappa = value["kappa"]
    if (
        isinstance(kappa, bool)
        or not isinstance(kappa, (int, float))
        or not math.isfinite(float(kappa))
        or float(kappa) < 0.0
    ):
        raise ConfigError(f"Config field {field_path}.kappa must be a finite non-negative number.")


def _validate_learned_gate(value: Mapping[str, Any], *, field_path: str) -> None:
    for field in ("kappa_init", "kappa_scope", "threshold_scale", "temperature"):
        if field not in value:
            raise ConfigError(f"Missing required config field: {field_path}.{field}")
    for field in ("kappa_init", "temperature"):
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise ConfigError(f"Config field {field_path}.{field} must be a finite positive number.")
    if value["kappa_scope"] not in {"global", "per_site", "per_layer_site"}:
        raise ConfigError(
            f"Config field {field_path}.kappa_scope must be 'global', 'per_site', or 'per_layer_site'."
        )
    if value["threshold_scale"] not in {"absolute", "rms_relative"}:
        raise ConfigError(
            f"Config field {field_path}.threshold_scale must be 'absolute' or 'rms_relative'."
        )
    if value.get("surrogate", "hard_forward_soft_backward") != "hard_forward_soft_backward":
        raise ConfigError(
            f"Config field {field_path}.surrogate must be 'hard_forward_soft_backward'."
        )
    rms_epsilon = value.get("rms_epsilon", 1e-8)
    if (
        isinstance(rms_epsilon, bool)
        or not isinstance(rms_epsilon, (int, float))
        or not math.isfinite(float(rms_epsilon))
        or float(rms_epsilon) <= 0.0
    ):
        raise ConfigError(f"Config field {field_path}.rms_epsilon must be a finite positive number.")
    if "kappa" in value:
        raise ConfigError(f"Config field {field_path}.kappa must be omitted for learned gates.")


def _validate_learned_threshold_training_contract(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    candidates = [
        model.get("post_layernorm_gate"),
        model.get("mlp_hidden_gate"),
        model.get("post_qkv_relu"),
    ]
    learned = [
        value
        for value in candidates
        if isinstance(value, Mapping) and str(value.get("gate_type", "")).startswith("learned_")
    ]
    if not learned:
        training = config.get("training", {})
        if isinstance(training, Mapping) and "threshold_learning_rate_multiplier" in training:
            raise ConfigError(
                "Config field training.threshold_learning_rate_multiplier requires a learned threshold gate."
            )
        return

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("Learned threshold gates require a training mapping.")
    if "threshold_learning_rate_multiplier" not in training:
        raise ConfigError(
            "Learned threshold gates require training.threshold_learning_rate_multiplier."
        )
    multiplier = training["threshold_learning_rate_multiplier"]
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(float(multiplier))
        or float(multiplier) <= 0.0
    ):
        raise ConfigError(
            "Config field training.threshold_learning_rate_multiplier must be a finite positive number."
        )
    if config.get("checkpoint", {}).get("save_final") is not True:
        raise ConfigError("Learned threshold gates require checkpoint.save_final: true.")
    if config.get("checkpoint", {}).get("save_optimizer") is not True:
        raise ConfigError("Learned threshold gates require checkpoint.save_optimizer: true.")

    global_specs = [value for value in learned if value.get("kappa_scope") == "global"]
    if global_specs:
        signatures = {
            (
                value["gate_type"],
                float(value["kappa_init"]),
                value["threshold_scale"],
                float(value["temperature"]),
                float(value.get("rms_epsilon", 1e-8)),
            )
            for value in learned
        }
        if len(signatures) != 1 or len(global_specs) != len(learned):
            raise ConfigError(
                "Global learned thresholds require identical learned gate settings across all active gate groups."
            )


def _get_required(config: Mapping[str, Any], field_path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in field_path:
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigError(f"Missing required config field: {'.'.join(field_path)}")
        current = current[key]
    return current
