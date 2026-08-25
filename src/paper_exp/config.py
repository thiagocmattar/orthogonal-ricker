from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

import yaml

from paper_exp.activations import resolve_site_aliases
from paper_exp.design import DesignError, validate_training_identity_fields
from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME
from paper_exp.topology import resolve_topology_and_gate


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
    ("output", "dir"),
)

CONFIG_FILE_RE = re.compile(r"^\d{2,}-[a-z0-9][a-z0-9-]*\.yaml$")
TRANCHE_ID_RE = re.compile(r"^(?!00)\d{2}-[a-z0-9]+-[a-z0-9][a-z0-9-]*$")
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
LEGACY_TOPOLOGY_FIELDS = {
    "hidden_act",
    "post_layernorm_relu",
    "post_layernorm_gate",
    "mlp_hidden_gate",
    "post_qkv_relu",
}


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
            "Config filenames must start with at least two digits and end in .yaml, "
            "like 01-baseline.yaml or 100-diagnostic.yaml."
        )


def validate_config(config: Mapping[str, Any], *, allow_todos: bool = True) -> None:
    if not isinstance(config, Mapping):
        raise ConfigError("Config must be a mapping.")

    for field_path in REQUIRED_FIELDS:
        _get_required(config, field_path)

    for field_path in (
        ("experiment_name",),
        ("model", "provider"),
        ("model", "name"),
        ("model", "architecture"),
        ("data", "name"),
        ("data", "split"),
        ("evaluation", "metric"),
        ("output", "dir"),
    ):
        _nonempty_string(_get_required(config, field_path), ".".join(field_path))

    seed = _get_required(config, ("run", "seed"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("Config field run.seed must be an integer.")

    _validate_reproducibility_fields(config, seed=seed)

    initialization = _get_required(config, ("model", "initialization"))
    if initialization != "random":
        raise ConfigError("Config field model.initialization must be 'random' for pretraining runs.")

    model = config.get("model", {})
    legacy_fields = sorted(LEGACY_TOPOLOGY_FIELDS.intersection(model))
    if legacy_fields:
        fields = ", ".join(f"model.{field}" for field in legacy_fields)
        raise ConfigError(
            f"Legacy topology fields are not supported ({fields}); use model.topology_id "
            "and model.site_gate."
        )
    if "topology_id" in model or "site_gate" in model:
        try:
            resolve_topology_and_gate(model.get("topology_id"), model.get("site_gate"))
        except ValueError as error:
            raise ConfigError(str(error)) from error

    if not allow_todos:
        todos = list(find_todo_values(config))
        if todos:
            fields = ", ".join(f"{path}={value}" for path, value in todos)
            raise ConfigError(f"Config contains TODO placeholders: {fields}")


def validate_data_config(config: Mapping[str, Any]) -> None:
    """Validate immutable data/tokenizer inputs before creating a cache run."""

    validate_config(config, allow_todos=False)
    data = _required_mapping(config, "data")
    _require_explicit_fields(data, "data", ("revision", "text_column", "max_documents"))
    _immutable_revision(data["revision"], "data.revision")
    _nonempty_string(data["text_column"], "data.text_column")
    if data["max_documents"] is not None:
        _positive_integer(data["max_documents"], "data.max_documents")

    tokenizer = _required_mapping(config, "tokenizer")
    _require_explicit_fields(tokenizer, "tokenizer", ("name", "revision"))
    _nonempty_string(tokenizer["name"], "tokenizer.name")
    _immutable_revision(tokenizer["revision"], "tokenizer.revision")

    preprocessing = _required_mapping(config, "preprocessing")
    _require_explicit_fields(
        preprocessing,
        "preprocessing",
        ("output_dir", "cache_id", "block_size", "append_eos", "overwrite"),
    )
    _nonempty_string(preprocessing["output_dir"], "preprocessing.output_dir")
    _nonempty_string(preprocessing["cache_id"], "preprocessing.cache_id")
    _positive_integer(preprocessing["block_size"], "preprocessing.block_size")
    if not isinstance(preprocessing["append_eos"], bool):
        raise ConfigError("Config field preprocessing.append_eos must be a boolean.")
    if not isinstance(preprocessing["overwrite"], bool):
        raise ConfigError("Config field preprocessing.overwrite must be a boolean.")

    validation = _required_mapping(config, "validation")
    _require_explicit_fields(validation, "validation", ("enabled",))
    if not isinstance(validation["enabled"], bool):
        raise ConfigError("Config field validation.enabled must be a boolean.")
    if validation["enabled"]:
        _require_explicit_fields(
            validation,
            "validation",
            (
                "split",
                "max_documents",
                "partition",
                "partition_scheme",
                "partition_seed",
                "partition_hash",
            ),
        )
        _nonempty_string(validation["split"], "validation.split")
        if validation["max_documents"] is not None:
            _positive_integer(validation["max_documents"], "validation.max_documents")
        if validation["partition"] is None:
            for field in ("partition_scheme", "partition_seed", "partition_hash"):
                if validation[field] is not None:
                    raise ConfigError(
                        f"Config field validation.{field} must be null when validation.partition is null."
                    )


def validate_smoke_config(config: Mapping[str, Any]) -> None:
    """Validate the infrastructure-only smoke input."""

    validate_config(config, allow_todos=True)
    run = _required_mapping(config, "run")
    _require_explicit_fields(run, "run", ("max_examples",))
    _positive_integer(run["max_examples"], "run.max_examples")


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate the explicit scientific inputs required by training workflows."""

    validate_data_config(config)
    try:
        validate_training_identity_fields(config)
    except DesignError as error:
        raise ConfigError(str(error)) from error
    model = _required_mapping(config, "model")
    if model["provider"] != "huggingface":
        raise ConfigError("Config field model.provider must be 'huggingface'.")
    _require_explicit_fields(model, "model", ("revision", "topology_id", "site_gate"))
    _immutable_revision(model["revision"], "model.revision")

    run = _required_mapping(config, "run")
    _require_explicit_fields(
        run,
        "run",
        (
            "training_schedule_scheme",
            "model_initialization_seed",
            "data_order_seed",
            "training_schedule_hash",
        ),
    )
    if not isinstance(run["training_schedule_hash"], str) or re.fullmatch(
        r"[0-9a-f]{64}", run["training_schedule_hash"]
    ) is None:
        raise ConfigError(
            "Config field run.training_schedule_hash must contain the realized "
            "lowercase SHA-256 digest for definitive training."
        )
    training = _required_mapping(config, "training")
    required_training_fields = (
        "device",
        "precision",
        "max_steps",
        "learning_rate",
        "warmup_steps",
        "gradient_accumulation_steps",
        "micro_batch_size",
        "log_every",
        "optimizer",
        "adamw_betas",
        "adamw_eps",
        "weight_decay",
    )
    _require_explicit_fields(training, "training", required_training_fields)

    if not isinstance(training["device"], str) or not training["device"].strip():
        raise ConfigError("Config field training.device must be a non-empty string.")
    if training["precision"] not in {"auto", "float32", "float16", "bfloat16"}:
        raise ConfigError(
            "Config field training.precision must be auto, float32, float16, or bfloat16."
        )
    max_steps = _positive_integer(training["max_steps"], "training.max_steps")
    if "max_wall_seconds" in training:
        raise ConfigError(
            "Config field training.max_wall_seconds is not supported; wall-time limits "
            "are calibration-only operational settings."
        )
    _positive_number(training["learning_rate"], "training.learning_rate")
    warmup_steps = _nonnegative_integer(training["warmup_steps"], "training.warmup_steps")
    expected_warmup_steps = math.ceil(0.01 * max_steps)
    if warmup_steps != expected_warmup_steps:
        raise ConfigError(
            "Config field training.warmup_steps must equal ceil(0.01 * training.max_steps): "
            f"expected {expected_warmup_steps}."
        )
    _positive_integer(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
    )
    _positive_integer(training["micro_batch_size"], "training.micro_batch_size")
    _positive_integer(training["log_every"], "training.log_every")
    if training["optimizer"] != "adamw":
        raise ConfigError("Config field training.optimizer must be 'adamw'.")
    betas = training["adamw_betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("Config field training.adamw_betas must contain two numbers.")
    parsed_betas = [_finite_number(value, "training.adamw_betas") for value in betas]
    if any(value < 0.0 or value >= 1.0 for value in parsed_betas):
        raise ConfigError("Config field training.adamw_betas values must be in [0, 1).")
    _positive_number(training["adamw_eps"], "training.adamw_eps")
    weight_decay = _finite_number(training["weight_decay"], "training.weight_decay")
    if weight_decay < 0.0:
        raise ConfigError("Config field training.weight_decay must be non-negative.")

    validation = _required_mapping(config, "validation")
    if validation["enabled"]:
        _require_explicit_fields(
            validation,
            "validation",
            ("split", "batch_size", "eval_every_steps", "eval_batches"),
        )
        if not isinstance(validation["split"], str) or not validation["split"].strip():
            raise ConfigError("Config field validation.split must be a non-empty string.")
        validation_batch_size = _positive_integer(
            validation["batch_size"], "validation.batch_size"
        )
        if validation_batch_size != 4:
            raise ConfigError("Config field validation.batch_size must equal 4.")
        eval_every_steps = _positive_integer(
            validation["eval_every_steps"],
            "validation.eval_every_steps",
        )
        if eval_every_steps != 191:
            raise ConfigError("Config field validation.eval_every_steps must equal 191.")
        if validation["eval_batches"] is not None:
            raise ConfigError(
                "Config field validation.eval_batches must be null so every complete block is evaluated."
            )

    checkpoint = _required_mapping(config, "checkpoint")
    _require_explicit_fields(
        checkpoint,
        "checkpoint",
        ("save_final", "save_optimizer"),
    )
    if not isinstance(checkpoint["save_final"], bool) or not isinstance(
        checkpoint["save_optimizer"], bool
    ):
        raise ConfigError("Checkpoint save flags must be booleans.")
    if checkpoint["save_optimizer"] and not checkpoint["save_final"]:
        raise ConfigError("checkpoint.save_optimizer requires checkpoint.save_final: true.")

    try:
        from paper_exp.activation_pressure import activation_pressure_config

        activation_pressure_config(dict(config))
    except ValueError as error:
        raise ConfigError(str(error)) from error


def validate_diagnostic_config(config: Mapping[str, Any], kind: str) -> None:
    """Validate static inputs for a checkpoint diagnostic before launch."""

    validate_config(config, allow_todos=False)
    if kind not in {
        "activation_histograms",
        "weight_histograms",
        "activation_propagation",
    }:
        raise ConfigError(f"Unsupported diagnostic config kind: {kind}.")
    diagnostic = _required_mapping(config, kind)
    _require_explicit_fields(diagnostic, kind, ("selected_runs",))
    _validate_selected_runs(diagnostic["selected_runs"], prefix=f"{kind}.selected_runs")

    if kind in {"activation_histograms", "activation_propagation"}:
        _validate_diagnostic_validation(config)

    if kind == "activation_histograms":
        _require_explicit_fields(
            diagnostic,
            kind,
            ("bins", "range_min", "range_max", "thresholds", "sites"),
        )
        _validate_histogram_geometry(diagnostic, prefix=kind)
        thresholds = diagnostic["thresholds"]
        if not isinstance(thresholds, list) or not thresholds:
            raise ConfigError(
                "Config field activation_histograms.thresholds must be a non-empty list."
            )
        parsed_thresholds = [
            _finite_number(value, "activation_histograms.thresholds")
            for value in thresholds
        ]
        if any(value < 0.0 for value in parsed_thresholds):
            raise ConfigError(
                "Config field activation_histograms.thresholds must be non-negative."
            )
        if parsed_thresholds != sorted(set(parsed_thresholds)):
            raise ConfigError(
                "Config field activation_histograms.thresholds must be strictly increasing."
            )
        _nonempty_unique_strings(diagnostic["sites"], "activation_histograms.sites")
        try:
            resolve_site_aliases(diagnostic["sites"])
        except ValueError as error:
            raise ConfigError(str(error)) from error

    if kind == "weight_histograms":
        _require_explicit_fields(
            diagnostic,
            kind,
            ("scope", "bins", "range_min", "range_max"),
        )
        if diagnostic["scope"] not in {"mlp_weights", "attention_weights"}:
            raise ConfigError(
                "Config field weight_histograms.scope must be mlp_weights or attention_weights."
            )
        _validate_histogram_geometry(diagnostic, prefix=kind)


def _validate_selected_runs(value: Any, *, prefix: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Config field {prefix} must be a non-empty list.")
    labels: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    for index, item in enumerate(value):
        item_prefix = f"{prefix}[{index}]"
        if not isinstance(item, Mapping):
            raise ConfigError(f"Config field {item_prefix} must be a mapping.")
        _require_explicit_fields(
            item,
            item_prefix,
            ("label", "tranche_id", "config_id", "run_id"),
        )
        label = _nonempty_string(item["label"], f"{item_prefix}.label")
        tranche_id = _nonempty_string(
            item["tranche_id"], f"{item_prefix}.tranche_id"
        )
        config_id = _nonempty_string(item["config_id"], f"{item_prefix}.config_id")
        run_id = _nonempty_string(item["run_id"], f"{item_prefix}.run_id")
        if TRANCHE_ID_RE.fullmatch(tranche_id) is None:
            raise ConfigError(
                f"Config field {item_prefix}.tranche_id is not a numbered "
                "scientific tranche ID."
            )
        if re.fullmatch(r"\d{2,}-[a-z0-9][a-z0-9-]*", config_id) is None:
            raise ConfigError(f"Config field {item_prefix}.config_id is not a numbered config ID.")
        if re.fullmatch(r"\d{3}-[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
            raise ConfigError(f"Config field {item_prefix}.run_id is not a numbered run ID.")
        if label in labels:
            raise ConfigError(f"Config field {prefix} contains duplicate labels.")
        identity = (tranche_id, config_id, run_id)
        if identity in identities:
            raise ConfigError(f"Config field {prefix} contains duplicate source runs.")
        labels.add(label)
        identities.add(identity)


def _validate_diagnostic_validation(config: Mapping[str, Any]) -> None:
    validation = _required_mapping(config, "validation")
    _require_explicit_fields(
        validation,
        "validation",
        (
            "enabled",
            "split",
            "max_documents",
            "partition",
            "partition_scheme",
            "partition_seed",
            "partition_hash",
            "batch_size",
            "eval_batches",
        ),
    )
    if validation["enabled"] is not True:
        raise ConfigError("Checkpoint diagnostics require validation.enabled: true.")
    _nonempty_string(validation["split"], "validation.split")
    if validation["max_documents"] is not None:
        _positive_integer(validation["max_documents"], "validation.max_documents")
    _positive_integer(validation["batch_size"], "validation.batch_size")
    if validation["eval_batches"] is not None:
        _positive_integer(validation["eval_batches"], "validation.eval_batches")
    if validation["partition"] is None:
        for field in ("partition_scheme", "partition_seed", "partition_hash"):
            if validation[field] is not None:
                raise ConfigError(
                    f"Config field validation.{field} must be null when validation.partition is null."
                )


def _validate_histogram_geometry(value: Mapping[str, Any], *, prefix: str) -> None:
    _positive_integer(value["bins"], f"{prefix}.bins")
    range_min = _finite_number(value["range_min"], f"{prefix}.range_min")
    range_max = _finite_number(value["range_max"], f"{prefix}.range_max")
    if range_min >= range_max:
        raise ConfigError(f"Config field {prefix}.range_min must be below range_max.")


def _nonempty_unique_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Config field {field} must be a non-empty list.")
    parsed = [_nonempty_string(item, field) for item in value]
    if len(set(parsed)) != len(parsed):
        raise ConfigError(f"Config field {field} must not contain duplicates.")
    return parsed


def _required_mapping(config: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config field {field} must be an explicit mapping.")
    return value


def _require_explicit_fields(
    mapping: Mapping[str, Any],
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    missing = [f"{prefix}.{field}" for field in fields if field not in mapping]
    if missing:
        raise ConfigError("Missing explicit config fields: " + ", ".join(missing))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"Config field {field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"Config field {field} must be a finite number.")
    return result


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config field {field} must be a non-empty string.")
    return value


def _immutable_revision(value: Any, field: str) -> str:
    revision = _nonempty_string(value, field)
    if IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ConfigError(
            f"Config field {field} must be an immutable 40- to 64-character lowercase hex commit."
        )
    return revision


def _positive_number(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    if result <= 0.0:
        raise ConfigError(f"Config field {field} must be positive.")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Config field {field} must be a positive integer.")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"Config field {field} must be a non-negative integer.")
    return value


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


def _validate_reproducibility_fields(config: Mapping[str, Any], *, seed: int) -> None:
    run_config = config.get("run", {})
    schedule_scheme = run_config.get("training_schedule_scheme")
    model_seed = run_config.get("model_initialization_seed")
    data_seed = run_config.get("data_order_seed")
    schedule_hash = run_config.get("training_schedule_hash")

    if schedule_scheme is None:
        reproducibility_fields = {
            "model_initialization_seed": model_seed,
            "data_order_seed": data_seed,
            "training_schedule_hash": schedule_hash,
        }
        provided = [
            field for field, value in reproducibility_fields.items() if value is not None
        ]
        if provided:
            raise ConfigError(
                "Reproducibility fields require run.training_schedule_scheme; provided: "
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
            "Config field run.seed must equal run.model_initialization_seed when explicit "
            "reproducibility seeds are used."
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


def _get_required(config: Mapping[str, Any], field_path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in field_path:
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigError(f"Missing required config field: {'.'.join(field_path)}")
        current = current[key]
    return current
