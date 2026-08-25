"""Reviewed-design, catalog, and scientific-condition identity contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

import yaml

from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME


class DesignError(ValueError):
    """Raised when a reviewed design or condition identity is incoherent."""


TRAINING_IMPLEMENTATION_ID = "a1_pretraining_v1"
PLAN_PATH = Path("docs/experiment_plan.md")
CATALOG_PATH = Path("docs/experimental-design/cases.yaml")
PROTOCOL_PATH = Path("docs/experimental-design/protocol.md")
DECISIONS_PATH = Path("docs/experimental-design/decisions.md")
WORKBOARD_PATH = Path("docs/experimental-design/workboard.md")
NORMATIVE_DESIGN_PATHS: tuple[Path, ...] = (
    PROTOCOL_PATH,
    CATALOG_PATH,
    Path("docs/experimental-design/run-reuse.md"),
    Path("docs/experimental-design/phases/a-pressure.md"),
    Path("docs/experimental-design/phases/b-threshold.md"),
    Path("docs/experimental-design/phases/c-scale.md"),
    Path("docs/experimental-design/outputs.md"),
    DECISIONS_PATH,
)

_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GROUP_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*-[a-z0-9][a-z0-9-]*$")
_IMPLEMENTATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_PLAN_FIELDS = (
    "Plan status",
    "Reviewed design commit",
    "Reviewed case groups",
)
_GRID_FACTOR_KEYS = frozenset({"kappa", "lambda"})
_MODEL_LABELS = {
    "Pythia-14M": "pythia-14m",
    "Pythia-70M": "pythia-70m",
    "Pythia-410M": "pythia-410m",
}
_MISSING = object()


@dataclass(frozen=True)
class PlanReview:
    status: str
    design_commit: str | None
    reviewed_groups: tuple[str, ...]


@dataclass(frozen=True)
class CatalogSummary:
    schema_version: int
    groups: Mapping[str, Mapping[str, Any]]
    fingerprint_exclude_paths: tuple[str, ...]


def load_plan_review(repository: str | Path) -> PlanReview:
    """Parse the three raw launch-authority fields without guessing."""

    root = Path(repository).resolve()
    path = root / PLAN_PATH
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DesignError(f"Cannot read the experiment plan: {path}") from error

    values: dict[str, str] = {}
    unfenced_lines: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            unfenced_lines.append(line)
    for field in _PLAN_FIELDS:
        matches = [
            line for line in unfenced_lines if line.startswith(f"{field}:")
        ]
        if len(matches) != 1:
            raise DesignError(f"Experiment plan must contain exactly one raw `{field}:` line.")
        values[field] = matches[0].split(":", 1)[1].strip()

    try:
        parsed_groups = yaml.safe_load(values["Reviewed case groups"])
    except yaml.YAMLError as error:
        raise DesignError("Reviewed case groups must be a one-line YAML list.") from error
    if not isinstance(parsed_groups, list) or any(
        not isinstance(group, str) or not group for group in parsed_groups
    ):
        raise DesignError("Reviewed case groups must be a one-line list of group IDs.")
    groups = tuple(parsed_groups)
    if len(set(groups)) != len(groups):
        raise DesignError("Reviewed case groups must not contain duplicates.")

    status = values["Plan status"]
    commit_text = values["Reviewed design commit"]
    if status == "placeholder":
        if commit_text != "none" or groups:
            raise DesignError(
                "A placeholder plan must use `Reviewed design commit: none` and an empty group list."
            )
        return PlanReview(status=status, design_commit=None, reviewed_groups=groups)
    if status != "reviewed":
        raise DesignError("Plan status must be exactly `placeholder` or `reviewed`.")
    if _FULL_GIT_SHA_RE.fullmatch(commit_text) is None:
        raise DesignError("A reviewed plan requires a full 40-character lowercase Git SHA.")
    if not groups:
        raise DesignError("A reviewed plan must name at least one reviewed case group.")
    return PlanReview(status=status, design_commit=commit_text, reviewed_groups=groups)


def validate_catalog(repository: str | Path) -> CatalogSummary:
    """Validate catalog grids, counts, decisions, reuse, and functional aliases."""

    root = Path(repository).resolve()
    catalog = _load_yaml_mapping(root / CATALOG_PATH, label="case catalog")
    if catalog.get("schema_version") != 1:
        raise DesignError("Case catalog schema_version must equal 1.")
    fingerprint = _mapping(catalog.get("condition_fingerprint"), "condition_fingerprint")
    exclude_paths = _unique_strings(
        fingerprint.get("exclude_paths"), "condition_fingerprint.exclude_paths"
    )
    required_exclusions = {
        "experiment_name",
        "identity.group_id",
        "identity.condition_fingerprint",
        "output.dir",
        "preprocessing.output_dir",
        "preprocessing.overwrite",
    }
    if set(exclude_paths) != required_exclusions:
        raise DesignError(
            "condition_fingerprint.exclude_paths must match the reviewed six-path contract."
        )
    canonicalization = _mapping(
        fingerprint.get("canonicalization"),
        "condition_fingerprint.canonicalization",
    )
    expected_canonicalization = {
        "format": "JSON",
        "object_keys": "lexicographically sorted",
        "numbers": "schema-normalized before serialization",
        "paths": "repository-relative POSIX where applicable",
        "encoding": "UTF-8",
        "hash": "SHA-256 lowercase hexadecimal",
    }
    if dict(canonicalization) != expected_canonicalization:
        raise DesignError("condition_fingerprint.canonicalization is not the supported contract.")

    seed_sets_raw = _mapping(catalog.get("seed_sets"), "seed_sets")
    seed_sets: dict[str, tuple[int, ...]] = {}
    for name, values in seed_sets_raw.items():
        if not isinstance(name, str):
            raise DesignError("Seed-set names must be strings.")
        if not isinstance(values, list) or not values:
            raise DesignError(f"Seed set {name} must be a non-empty list.")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise DesignError(f"Seed set {name} must contain only integers.")
        if len(set(values)) != len(values):
            raise DesignError(f"Seed set {name} contains duplicate seeds.")
        seed_sets[name] = tuple(values)

    rules_raw = catalog.get("functional_equivalence_rules")
    if not isinstance(rules_raw, list):
        raise DesignError("functional_equivalence_rules must be a list.")
    rules: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(rules_raw):
        rule = _mapping(value, f"functional_equivalence_rules[{index}]")
        rule_id = _nonempty_string(rule.get("rule_id"), f"functional rule {index}.rule_id")
        if rule_id in rules:
            raise DesignError(f"Duplicate functional-equivalence rule: {rule_id}.")
        _unique_strings(rule.get("required_work_items"), f"functional rule {rule_id}.required_work_items")
        _unique_strings(rule.get("acceptance_tests"), f"functional rule {rule_id}.acceptance_tests")
        rules[rule_id] = rule

    groups_raw = catalog.get("case_groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise DesignError("case_groups must be a non-empty list.")
    groups: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(groups_raw):
        group = _mapping(value, f"case_groups[{index}]")
        group_id = _nonempty_string(group.get("group_id"), f"case_groups[{index}].group_id")
        if _GROUP_ID_RE.fullmatch(group_id) is None:
            raise DesignError(f"Invalid case-group ID: {group_id}.")
        if group_id in groups:
            raise DesignError(f"Duplicate case-group ID: {group_id}.")
        seed_set = group.get("seeds")
        if seed_set not in seed_sets:
            raise DesignError(f"Case group {group_id} references an unknown seed set.")
        _group_models(group, group_id)
        groups[group_id] = group

    decisions = _decision_ids(root / DECISIONS_PATH)
    lr_grids = _protocol_lr_grids(root / PROTOCOL_PATH)
    for group_id, group in groups.items():
        _validate_decision_references(group, group_id=group_id, decisions=decisions)
        expected = _expected_conceptual_count(
            group,
            group_id=group_id,
            seed_sets=seed_sets,
            lr_grids=lr_grids,
        )
        _validate_declared_conceptual_count(group, group_id=group_id, expected=expected)
        alias_count = _validate_functional_aliases(
            group,
            group_id=group_id,
            rules=rules,
            seed_sets=seed_sets,
        )
        _validate_unique_count(
            group,
            group_id=group_id,
            expected=expected,
            alias_count=alias_count,
            seed_sets=seed_sets,
        )
    _validate_reuse_graph(groups)
    for rule_id, rule in rules.items():
        scope = str(rule.get("scope", "")).removesuffix(" only")
        if scope not in groups:
            raise DesignError(f"Functional rule {rule_id} has unknown scope {scope!r}.")

    return CatalogSummary(
        schema_version=1,
        groups=groups,
        fingerprint_exclude_paths=exclude_paths,
    )


def validate_reviewed_design(
    repository: str | Path,
    *,
    require_reviewed: bool = True,
) -> PlanReview:
    """Validate plan state and pin every normative blob to the reviewed commit."""

    root = Path(repository).resolve()
    catalog = validate_catalog(root)
    review = load_plan_review(root)
    if review.status == "placeholder":
        if require_reviewed:
            raise DesignError(
                "Scientific launches are blocked until `Plan status: reviewed`."
            )
        return review

    assert review.design_commit is not None
    unknown = sorted(set(review.reviewed_groups) - set(catalog.groups))
    if unknown:
        raise DesignError("Reviewed case groups are absent from the catalog: " + ", ".join(unknown))
    _require_commit(root, review.design_commit)
    changed = [
        path.as_posix()
        for path in NORMATIVE_DESIGN_PATHS
        if _working_blob_id(root, path) != _reviewed_blob_id(
            root, review.design_commit, path
        )
    ]
    if changed:
        raise DesignError(
            "Normative design files differ from the reviewed commit: " + ", ".join(changed)
        )
    protocol_training_implementation_id(root)
    _validate_reviewed_functional_rules(root, catalog, review.reviewed_groups)
    return review


def validate_training_identity_fields(config: Mapping[str, Any]) -> None:
    """Require immutable scientific identity fields in one training config."""

    identity = _mapping(config.get("identity"), "identity")
    required = {"group_id", "condition_fingerprint", "training_implementation_id"}
    missing = sorted(required - set(identity))
    if missing:
        raise DesignError("Missing explicit config fields: " + ", ".join(f"identity.{x}" for x in missing))
    group_id = _nonempty_string(identity.get("group_id"), "identity.group_id")
    if _GROUP_ID_RE.fullmatch(group_id) is None:
        raise DesignError("Config field identity.group_id is not a canonical case-group ID.")
    fingerprint = _nonempty_string(
        identity.get("condition_fingerprint"), "identity.condition_fingerprint"
    )
    if _SHA256_RE.fullmatch(fingerprint) is None:
        raise DesignError(
            "Config field identity.condition_fingerprint must be a lowercase SHA-256 digest."
        )
    implementation_id = _nonempty_string(
        identity.get("training_implementation_id"),
        "identity.training_implementation_id",
    )
    if _IMPLEMENTATION_ID_RE.fullmatch(implementation_id) is None:
        raise DesignError("Config field identity.training_implementation_id is invalid.")


def condition_fingerprint(
    config: Mapping[str, Any],
    *,
    exclude_paths: Sequence[str],
) -> str:
    """Hash one complete normalized scientific condition after exact exclusions."""

    payload = deepcopy(dict(config))
    for dotted_path in exclude_paths:
        _remove_path(payload, dotted_path)
    canonical = {
        "schema_version": 1,
        "training_config": _normalize_json_value(payload),
    }
    return sha256(_canonical_json_bytes(canonical)).hexdigest()


def complete_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete schema-normalized config snapshot without exclusions."""

    canonical = {
        "schema_version": 1,
        "config": _normalize_json_value(dict(config)),
    }
    return sha256(_canonical_json_bytes(canonical)).hexdigest()


def validate_config_for_reviewed_design(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    config_path: str | Path,
) -> None:
    """Apply reviewed group, implementation, fingerprint, and duplicate preflight."""

    root = Path(repository).resolve()
    review = validate_reviewed_design(root)
    catalog = validate_catalog(root)
    validate_training_identity_fields(config)
    identity = _mapping(config.get("identity"), "identity")
    group_id = str(identity["group_id"])
    if group_id not in review.reviewed_groups:
        raise DesignError(f"Config case group {group_id} is not in the reviewed scope.")
    _validate_group_membership(config, group_id=group_id, repository=root)
    expected_implementation = protocol_training_implementation_id(root)
    if identity["training_implementation_id"] != expected_implementation:
        raise DesignError(
            "Config training implementation does not match the reviewed protocol pin."
        )
    expected_fingerprint = condition_fingerprint(
        config,
        exclude_paths=catalog.fingerprint_exclude_paths,
    )
    if identity["condition_fingerprint"] != expected_fingerprint:
        raise DesignError("Config condition fingerprint does not match its normalized content.")

    target = Path(config_path).resolve()
    tracked = _tracked_training_configs(root)
    if target not in tracked:
        raise DesignError("Scientific config is not a tracked training config.")
    fingerprints: dict[str, list[Path]] = {}
    for path, candidate in tracked.items():
        validate_training_identity_fields(candidate)
        candidate_identity = _mapping(candidate.get("identity"), "identity")
        computed = condition_fingerprint(
            candidate,
            exclude_paths=catalog.fingerprint_exclude_paths,
        )
        if candidate_identity["condition_fingerprint"] != computed:
            raise DesignError(
                f"Tracked config has a stale condition fingerprint: {_relative(root, path)}."
            )
        fingerprints.setdefault(computed, []).append(path)
    duplicates = [paths for paths in fingerprints.values() if len(paths) > 1]
    if duplicates:
        rendered = "; ".join(
            ", ".join(_relative(root, path) for path in paths)
            for paths in duplicates
        )
        raise DesignError("Duplicate scientific condition fingerprints: " + rendered)


def protocol_training_implementation_id(repository: str | Path) -> str:
    root = Path(repository).resolve()
    path = root / PROTOCOL_PATH
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DesignError(f"Cannot read training implementation pin: {path}") from error
    matches = []
    for line in lines:
        cells = _markdown_cells(line)
        if cells and cells[0] == "Training-implementation identity":
            matches.append(cells[1] if len(cells) > 1 else "")
    if len(matches) != 1:
        raise DesignError("Protocol must contain exactly one Training-implementation identity row.")
    value = matches[0].strip().strip("`")
    if value.upper().startswith("TODO"):
        raise DesignError("Training-implementation identity is unresolved in the protocol.")
    if _IMPLEMENTATION_ID_RE.fullmatch(value) is None:
        raise DesignError("Protocol training-implementation identity is invalid.")
    if value != TRAINING_IMPLEMENTATION_ID:
        raise DesignError("Protocol training-implementation identity does not match the code.")
    return value


def tracked_training_identities(
    repository: str | Path,
) -> list[tuple[Path, str, str]]:
    """Return tracked training config paths, groups, and verified fingerprints."""

    root = Path(repository).resolve()
    catalog = validate_catalog(root)
    rows: list[tuple[Path, str, str]] = []
    for path, config in _tracked_training_configs(root).items():
        validate_training_identity_fields(config)
        identity = _mapping(config.get("identity"), "identity")
        _validate_group_membership(
            config,
            group_id=str(identity["group_id"]),
            repository=root,
        )
        fingerprint = condition_fingerprint(
            config, exclude_paths=catalog.fingerprint_exclude_paths
        )
        if identity["condition_fingerprint"] != fingerprint:
            raise DesignError(
                f"Tracked config has a stale condition fingerprint: {_relative(root, path)}."
            )
        rows.append((path, str(identity["group_id"]), fingerprint))
    return rows


def _validate_group_membership(
    config: Mapping[str, Any],
    *,
    group_id: str,
    repository: Path,
) -> None:
    """Fail closed until each catalog group has an exact materialization contract."""

    if group_id != "A1-lr-screen":
        raise DesignError(
            f"Exact config membership validation is not implemented for {group_id}; "
            "that group cannot be materialized or launched."
        )
    lr_grid = _protocol_lr_grids(repository / PROTOCOL_PATH)["pythia-14m"]
    exact_values: tuple[tuple[str, Any], ...] = (
        ("model.provider", "huggingface"),
        ("model.name", "pythia-14m-random"),
        ("model.architecture", "EleutherAI/pythia-14m-deduped"),
        ("model.revision", "7386d9a4ae45aef494a6e704910394def3037fc5"),
        ("model.initialization", "random"),
        ("model.topology_id", "A0"),
        ("model.site_gate", None),
        ("data.name", "JeanKaddour/minipile"),
        ("data.revision", "18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0"),
        ("data.split", "train"),
        ("data.text_column", "text"),
        ("data.max_documents", None),
        ("tokenizer.name", "EleutherAI/pythia-14m-deduped"),
        ("tokenizer.revision", "7386d9a4ae45aef494a6e704910394def3037fc5"),
        ("preprocessing.cache_id", "03-pythia-14m-minipile-random-full-10min"),
        ("preprocessing.tokens_sha256", "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c"),
        ("preprocessing.block_size", 2048),
        ("preprocessing.append_eos", True),
        ("preprocessing.overwrite", False),
        ("evaluation.metric", "validation_loss"),
        ("run.seed", 0),
        ("run.training_schedule_scheme", TRAINING_SCHEDULE_SCHEME),
        ("run.model_initialization_seed", 0),
        ("run.data_order_seed", 0),
        ("run.training_schedule_hash", "35da3f6aa891a2248407344715e4c75e99cb518b17119a8e66004466a823a21c"),
        ("training.device", "cuda"),
        ("training.precision", "bfloat16"),
        ("training.max_steps", 5691),
        ("training.warmup_steps", 57),
        ("training.micro_batch_size", 16),
        ("training.gradient_accumulation_steps", 8),
        ("training.log_every", 10),
        ("training.optimizer", "adamw"),
        ("training.adamw_betas", [0.9, 0.95]),
        ("training.adamw_eps", 1.0e-8),
        ("training.weight_decay", 0.1),
        ("validation.enabled", True),
        ("validation.split", "validation"),
        ("validation.max_documents", 500),
        ("validation.partition", "selection"),
        ("validation.partition_scheme", "shuffled_source_documents_half_v1"),
        ("validation.partition_seed", 20260718),
        ("validation.partition_hash", "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"),
        ("validation.tokens_sha256", "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19"),
        ("validation.batch_size", 4),
        ("validation.eval_every_steps", 191),
        ("validation.eval_batches", None),
        ("checkpoint.save_final", True),
        ("checkpoint.save_optimizer", False),
        ("activation_pressure.enabled", False),
        ("activation_pressure.method", "none"),
        ("activation_pressure.sites", ["h"]),
        ("activation_pressure.weight", 0.0),
        ("activation_pressure.step_budget", None),
    )
    mismatches = [
        f"{path}={_value_at(config, path)!r}"
        for path, expected in exact_values
        if _value_at(config, path) != expected
    ]
    if mismatches:
        raise DesignError(
            "Config is not an A1-lr-screen physical cell: " + ", ".join(mismatches)
        )
    learning_rate = _value_at(config, "training.learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
        raise DesignError("A1 peak learning rate must be numeric.")
    if float(learning_rate) not in lr_grid:
        raise DesignError("A1 peak learning rate is outside the reviewed 14M grid.")
    micro_batch = int(_value_at(config, "training.micro_batch_size"))
    accumulation = int(_value_at(config, "training.gradient_accumulation_steps"))
    block_size = int(_value_at(config, "preprocessing.block_size"))
    if micro_batch * accumulation * block_size != 262_144:
        raise DesignError("A1 physical batch does not equal 262,144 input tokens per update.")


def _expected_conceptual_count(
    group: Mapping[str, Any],
    *,
    group_id: str,
    seed_sets: Mapping[str, tuple[int, ...]],
    lr_grids: Mapping[str, tuple[float, ...]],
) -> int | tuple[int, str]:
    seed_count = len(seed_sets[str(group["seeds"])])
    models = _group_models(group, group_id)
    model_count = len(models)
    factors = _mapping(group.get("factors", {}), f"{group_id}.factors")
    components = group.get("components")
    if components is not None:
        return seed_count * model_count * len(_unique_strings(components, f"{group_id}.components"))
    if "no_attention_topologies" in factors or "attention_topologies" in factors:
        no_attention = _unique_strings(
            factors.get("no_attention_topologies"), f"{group_id}.no_attention_topologies"
        )
        attention = _unique_strings(
            factors.get("attention_topologies"), f"{group_id}.attention_topologies"
        )
        forms = _unique_strings(
            factors.get("attention_threshold"), f"{group_id}.attention_threshold"
        )
        kappas = _numeric_list(factors.get("kappa"), f"{group_id}.kappa")
        return seed_count * model_count * len(kappas) * (
            len(no_attention) + len(attention) * len(forms)
        )
    recipes = factors.get("recipes")
    if recipes is not None:
        recipe_count = len(_mapping(recipes, f"{group_id}.recipes"))
        kappa = factors.get("kappa")
        if not isinstance(kappa, str) or not kappa.startswith("decision:"):
            raise DesignError(f"{group_id} recipe grid requires a decision-backed kappa list.")
        return seed_count * model_count * recipe_count, kappa.removeprefix("decision:")
    if "peak_lr_ref" in factors:
        if factors["peak_lr_ref"] != "protocol.md#batch-and-learning-rate-grids":
            raise DesignError(f"{group_id} has an unsupported peak-LR reference.")
        try:
            return seed_count * sum(len(lr_grids[model]) for model in models)
        except KeyError as error:
            raise DesignError(f"{group_id} model has no protocol peak-LR grid.") from error
    count = seed_count * model_count
    for key in _GRID_FACTOR_KEYS:
        if isinstance(factors.get(key), list):
            count *= len(_numeric_list(factors[key], f"{group_id}.{key}"))
    return count


def _validate_declared_conceptual_count(
    group: Mapping[str, Any],
    *,
    group_id: str,
    expected: int | tuple[int, str],
) -> None:
    declared = group.get("conceptual_cells")
    if isinstance(expected, int):
        if isinstance(declared, bool) or declared != expected:
            raise DesignError(
                f"{group_id} conceptual_cells is {declared!r}; expansion yields {expected}."
            )
        return
    multiplier, reference = expected
    expression = _mapping(declared, f"{group_id}.conceptual_cells")
    if dict(expression) != {"multiplier": multiplier, "cardinality_of": f"decision:{reference}"}:
        raise DesignError(f"{group_id} conceptual count expression does not match its grid.")


def _validate_unique_count(
    group: Mapping[str, Any],
    *,
    group_id: str,
    expected: int | tuple[int, str],
    alias_count: int,
    seed_sets: Mapping[str, tuple[int, ...]],
) -> None:
    unique = group.get("unique_cases")
    if isinstance(expected, tuple):
        if unique != {"same_as": "conceptual_cells"} or alias_count:
            raise DesignError(f"{group_id} unique count expression is invalid.")
        return
    if isinstance(unique, int) and not isinstance(unique, bool):
        if unique != expected - alias_count:
            raise DesignError(
                f"{group_id} unique_cases is {unique}; expected {expected - alias_count}."
            )
        return
    conditional = _mapping(unique, f"{group_id}.unique_cases")
    if set(conditional) != {"if_lambda_B2_is_1", "otherwise"}:
        raise DesignError(f"{group_id} has an unsupported unique-case expression.")
    unit = len(seed_sets[str(group["seeds"])]) * len(_group_models(group, group_id))
    expected_values = {
        "if_lambda_B2_is_1": expected - 2 * unit,
        "otherwise": expected - unit,
    }
    if dict(conditional) != expected_values:
        raise DesignError(f"{group_id} conditional unique-case counts do not match declared reuse.")


def _validate_functional_aliases(
    group: Mapping[str, Any],
    *,
    group_id: str,
    rules: Mapping[str, Mapping[str, Any]],
    seed_sets: Mapping[str, tuple[int, ...]],
) -> int:
    aliases = group.get("functional_equivalence_aliases", [])
    if not isinstance(aliases, list):
        raise DesignError(f"{group_id}.functional_equivalence_aliases must be a list.")
    total = 0
    seen_selectors: set[bytes] = set()
    for index, value in enumerate(aliases):
        alias = _mapping(value, f"{group_id}.functional_equivalence_aliases[{index}]")
        selector = _mapping(alias.get("selector"), f"{group_id} alias selector")
        reuse = _mapping(alias.get("reuse"), f"{group_id} alias reuse")
        count = alias.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise DesignError(f"{group_id} functional alias count must be positive.")
        expanded_count = math.prod(
            len(value) if isinstance(value, list) else 1 for value in selector.values()
        )
        if expanded_count != count:
            raise DesignError(
                f"{group_id} functional alias count is {count}; selector expands to {expanded_count}."
            )
        rule_id = alias.get("equivalence_rule")
        if rule_id not in rules or rules[str(rule_id)].get("scope") not in {
            group_id,
            f"{group_id} only",
        }:
            raise DesignError(f"{group_id} functional alias references an invalid rule.")
        selector_key = _canonical_json_bytes(_normalize_json_value(dict(selector)))
        if selector_key in seen_selectors:
            raise DesignError(f"{group_id} contains duplicate functional-alias selectors.")
        seen_selectors.add(selector_key)
        _validate_alias_cell(selector, group, seed_sets, group_id=group_id, label="selector")
        _validate_alias_cell(reuse, group, seed_sets, group_id=group_id, label="reuse")
        total += count
    return total


def _validate_alias_cell(
    cell: Mapping[str, Any],
    group: Mapping[str, Any],
    seed_sets: Mapping[str, tuple[int, ...]],
    *,
    group_id: str,
    label: str,
) -> None:
    factors = _mapping(group.get("factors", {}), f"{group_id}.factors")
    topology_domain: set[Any] = set()
    for key in ("topology", "no_attention_topologies", "attention_topologies"):
        value = factors.get(key)
        topology_domain.update(value if isinstance(value, list) else [value] if value is not None else [])
    domains: dict[str, set[Any]] = {
        "topology": topology_domain,
        "kappa": set(factors.get("kappa", [])),
        "attention_threshold": set(factors.get("attention_threshold", [])) | {"not_applicable"},
        "ffn_threshold": {factors.get("ffn_threshold")},
        "seed": set(seed_sets[str(group["seeds"])]),
    }
    for key, value in cell.items():
        if key not in domains:
            raise DesignError(f"{group_id} functional alias {label} uses unknown factor {key}.")
        values = value if isinstance(value, list) else [value]
        if any(item not in domains[key] for item in values):
            raise DesignError(f"{group_id} functional alias {label} is outside the group grid.")


def _validate_reuse_graph(groups: Mapping[str, Mapping[str, Any]]) -> None:
    graph: dict[str, set[str]] = {group_id: set() for group_id in groups}
    for group_id, group in groups.items():
        controls = group.get("controls", [])
        if not isinstance(controls, list):
            raise DesignError(f"{group_id}.controls must be a list.")
        for index, value in enumerate(controls):
            control = _mapping(value, f"{group_id}.controls[{index}]")
            target = control.get("reuse_group")
            if target not in groups:
                raise DesignError(f"{group_id} reuses unknown group {target!r}.")
            graph[group_id].add(str(target))
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(group_id: str) -> None:
        if group_id in visiting:
            raise DesignError("Case-group reuse aliases contain a cycle.")
        if group_id in visited:
            return
        visiting.add(group_id)
        for target in graph[group_id]:
            visit(target)
        visiting.remove(group_id)
        visited.add(group_id)

    for group_id in graph:
        visit(group_id)


def _validate_decision_references(
    value: Any,
    *,
    group_id: str,
    decisions: set[str],
) -> None:
    if isinstance(value, str) and value.startswith("decision:"):
        decision = value.removeprefix("decision:").split(".", 1)[0]
        if decision not in decisions:
            raise DesignError(f"{group_id} references unknown decision {decision}.")
    elif isinstance(value, Mapping):
        for child in value.values():
            _validate_decision_references(child, group_id=group_id, decisions=decisions)
    elif isinstance(value, list):
        for child in value:
            _validate_decision_references(child, group_id=group_id, decisions=decisions)
    if isinstance(value, Mapping):
        for key in ("requires_decisions", "produces_decisions"):
            if key in value:
                for decision in _unique_strings(value[key], f"{group_id}.{key}"):
                    if decision not in decisions:
                        raise DesignError(f"{group_id} names unknown decision {decision}.")
        if "produces_decision" in value:
            decision = value["produces_decision"]
            if decision not in decisions:
                raise DesignError(f"{group_id} names unknown decision {decision}.")


def _validate_reviewed_functional_rules(
    root: Path,
    catalog: CatalogSummary,
    reviewed_groups: Sequence[str],
) -> None:
    raw = _load_yaml_mapping(root / CATALOG_PATH, label="case catalog")
    rules = {
        str(rule["rule_id"]): rule
        for rule in raw.get("functional_equivalence_rules", [])
    }
    states = _workboard_states(root / WORKBOARD_PATH)
    for group_id in reviewed_groups:
        group = catalog.groups[group_id]
        for alias in group.get("functional_equivalence_aliases", []):
            rule = rules[str(alias["equivalence_rule"])]
            unresolved = [
                item
                for item in rule["required_work_items"]
                if states.get(str(item)) != "resolved"
            ]
            if unresolved:
                raise DesignError(
                    f"Reviewed functional alias {rule['rule_id']} has unresolved work: "
                    + ", ".join(unresolved)
                )
            for test_ref in rule["acceptance_tests"]:
                path_text, separator, test_name = str(test_ref).partition("::")
                path = root / path_text
                if not separator or not path.is_file():
                    raise DesignError(f"Functional-alias acceptance test is missing: {test_ref}.")
                text = path.read_text(encoding="utf-8-sig")
                if re.search(rf"^def {re.escape(test_name)}\s*\(", text, re.MULTILINE) is None:
                    raise DesignError(f"Functional-alias acceptance test is missing: {test_ref}.")


def _tracked_training_configs(root: Path) -> dict[Path, Mapping[str, Any]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", "experiments/*/run/*.yaml"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DesignError("Cannot enumerate tracked scientific configs.") from error
    configs: dict[Path, Mapping[str, Any]] = {}
    for relative in result.stdout.splitlines():
        path = (root / relative).resolve()
        if path.name == "00-smoke.yaml":
            continue
        loaded = _load_yaml_mapping(path, label=f"tracked config {relative}")
        if "training" in loaded:
            configs[path] = loaded
    return configs


def _protocol_lr_grids(path: Path) -> dict[str, tuple[float, ...]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DesignError(f"Cannot read protocol LR grids: {path}") from error
    grids: dict[str, tuple[float, ...]] = {}
    for line in lines:
        cells = _markdown_cells(line)
        if not cells or cells[0] not in _MODEL_LABELS or len(cells) < 3:
            continue
        text = cells[2].strip().strip("`")
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            values = tuple(float(item.strip()) for item in text[1:-1].split(","))
        except ValueError as error:
            raise DesignError(f"Invalid peak-LR grid for {cells[0]}.") from error
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise DesignError(f"Invalid peak-LR grid for {cells[0]}.")
        if len(set(values)) != len(values):
            raise DesignError(f"Duplicate peak LR in grid for {cells[0]}.")
        grids[_MODEL_LABELS[cells[0]]] = values
    if set(grids) != set(_MODEL_LABELS.values()):
        raise DesignError("Protocol must define one peak-LR grid for every catalog model.")
    return grids


def _decision_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DesignError(f"Cannot read decision register: {path}") from error
    decisions = set()
    for line in lines:
        cells = _markdown_cells(line)
        if not cells or cells[0] in {"Decision ID", "---"}:
            continue
        value = cells[0].strip().strip("`")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
            decisions.add(value)
    if not decisions:
        raise DesignError("Decision register contains no decision IDs.")
    return decisions


def _workboard_states(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise DesignError(f"Cannot read workboard: {path}") from error
    states: dict[str, str] = {}
    for line in lines:
        cells = _markdown_cells(line)
        if len(cells) < 2:
            continue
        item = cells[0].strip().strip("`")
        if re.fullmatch(r"[A-Z]+-[0-9]+", item):
            states[item] = cells[1]
    return states


def _group_models(group: Mapping[str, Any], group_id: str) -> tuple[str, ...]:
    has_model = "model" in group
    has_models = "models" in group
    if has_model == has_models:
        raise DesignError(f"{group_id} must define exactly one of model or models.")
    if has_model:
        return (_nonempty_string(group["model"], f"{group_id}.model"),)
    return _unique_strings(group["models"], f"{group_id}.models")


def _remove_path(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _value_at(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _normalize_json_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key in {"dir", "path", "output_dir"} or (
            key is not None and key.endswith(("_dir", "_path"))
        ):
            return PurePosixPath(value.replace("\\", "/")).as_posix()
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DesignError("Canonical scientific identity cannot contain nonfinite numbers.")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DesignError("Canonical scientific identity requires string object keys.")
        return {
            child_key: _normalize_json_value(value[child_key], key=child_key)
            for child_key in sorted(value)
        }
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item, key=key) for item in value]
    raise DesignError(f"Unsupported canonical identity value: {type(value).__name__}.")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DesignError(f"Cannot read {label}: {path}") from error
    return _mapping(value, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignError(f"{label} must be a mapping.")
    return value


def _unique_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DesignError(f"{label} must be a non-empty list.")
    if any(not isinstance(item, str) or not item for item in value):
        raise DesignError(f"{label} must contain non-empty strings.")
    if len(set(value)) != len(value):
        raise DesignError(f"{label} must not contain duplicates.")
    return tuple(value)


def _numeric_list(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise DesignError(f"{label} must be a non-empty numeric list.")
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise DesignError(f"{label} must contain finite numbers.")
        number = float(item)
        if not math.isfinite(number):
            raise DesignError(f"{label} must contain finite numbers.")
        parsed.append(number)
    if len(set(parsed)) != len(parsed):
        raise DesignError(f"{label} must not contain duplicates.")
    return tuple(parsed)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignError(f"{label} must be a non-empty string.")
    return value


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _require_commit(root: Path, commit: str) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DesignError("Reviewed design commit is not available in this repository.") from error


def _reviewed_blob_id(root: Path, commit: str, path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DesignError(
            f"Reviewed design commit does not contain {path.as_posix()}."
        ) from error
    return result.stdout.strip()


def _working_blob_id(root: Path, path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "hash-object",
                f"--path={path.as_posix()}",
                "--",
                path.as_posix(),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DesignError(f"Cannot hash normative design file: {path}") from error
    return result.stdout.strip()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
