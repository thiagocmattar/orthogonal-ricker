from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any

import pytest
import yaml

from paper_exp.design import (
    CATALOG_PATH,
    DECISIONS_PATH,
    NORMATIVE_DESIGN_PATHS,
    PLAN_PATH,
    PROTOCOL_PATH,
    TRAINING_IMPLEMENTATION_ID,
    WORKBOARD_PATH,
    DesignError,
    condition_fingerprint,
    tracked_training_identities,
    validate_catalog,
    validate_config_for_reviewed_design,
    validate_reviewed_design,
)
from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME


REPOSITORY = Path(__file__).resolve().parents[1]
A2_GROUPS = ("A2-relu-control", "A2-l1-screen")
A2_L1_WEIGHTS = (0.1, 0.5, 1.0, 2.0, 5.0)


@pytest.fixture(autouse=True)
def _make_nested_git_objects_cleanup_safe(tmp_path: Path):
    yield
    for git_dir in tmp_path.rglob(".git"):
        for path in git_dir.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_repository_catalog_expands_every_declared_group_and_alias() -> None:
    summary = validate_catalog(REPOSITORY)

    assert tuple(summary.groups) == (
        "A1-lr-screen",
        "A2-relu-control",
        "A2-l1-screen",
        "A3-ol1-screen",
        "B1-threshold-screen",
        "B2-combined-screen",
        "B2-winner-confirmation",
        "C1-lr-screens",
        "C2-dense-controls",
        "C2-relu-controls",
        "C2-l1-screens",
        "C3-ol1-controls",
        "C3-frontier-replication",
        "C3-winner-confirmation",
    )
    assert summary.groups["A1-lr-screen"]["conceptual_cells"] == 11
    assert summary.groups["A1-lr-screen"]["unique_cases"] == 11
    assert summary.groups["A2-relu-control"]["unique_cases"] == 1
    expected_pressure_grid = [0.1, 0.5, 1, 2, 5]
    assert summary.groups["A2-l1-screen"]["factors"]["lambda"] == expected_pressure_grid
    assert summary.groups["A2-l1-screen"]["unique_cases"] == 5
    assert summary.groups["A3-ol1-screen"]["factors"]["lambda"] == expected_pressure_grid
    assert summary.groups["A3-ol1-screen"]["unique_cases"] == 5
    assert summary.groups["C2-l1-screens"]["factors"]["lambda"] == expected_pressure_grid
    assert summary.groups["C2-l1-screens"]["unique_cases"] == 10
    assert summary.groups["B2-winner-confirmation"]["unique_cases"] == 12
    assert summary.groups["C3-winner-confirmation"]["unique_cases"] == 24
    assert summary.groups["B1-threshold-screen"]["conceptual_cells"] == 56
    assert summary.groups["B1-threshold-screen"]["unique_cases"] == 50


def test_catalog_rejects_a_count_that_does_not_match_expansion(tmp_path: Path) -> None:
    repository = _design_repository(tmp_path)
    catalog_path = repository / CATALOG_PATH
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    b1 = next(
        group
        for group in catalog["case_groups"]
        if group["group_id"] == "B1-threshold-screen"
    )
    b1["unique_cases"] = 49
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(DesignError, match="unique_cases is 49; expected 50"):
        validate_catalog(repository)


def test_fingerprint_is_canonical_and_excludes_only_operational_identity() -> None:
    excludes = validate_catalog(REPOSITORY).fingerprint_exclude_paths
    first = {
        "experiment_name": "first",
        "identity": {
            "group_id": "A1-lr-screen",
            "condition_fingerprint": "a" * 64,
            "training_implementation_id": TRAINING_IMPLEMENTATION_ID,
        },
        "run": {"seed": 0},
        "training": {"learning_rate": 1},
        "artifact_path": "data\\cache\\tokens.bin",
        "preprocessing": {
            "output_dir": "data/tokenized",
            "overwrite": False,
        },
        "output": {"dir": "experiments/01-a1/raw"},
    }
    second = deepcopy(first)
    second["experiment_name"] = "second"
    second["identity"]["group_id"] = "later-consumer"
    second["identity"]["condition_fingerprint"] = "b" * 64
    second["training"]["learning_rate"] = 1.0
    second["artifact_path"] = "data/cache/tokens.bin"
    second["preprocessing"]["overwrite"] = True
    second["output"]["dir"] = "experiments/99-other/raw"

    assert condition_fingerprint(first, exclude_paths=excludes) == condition_fingerprint(
        second, exclude_paths=excludes
    )

    second["identity"]["training_implementation_id"] = "a1_pretraining_v2"
    assert condition_fingerprint(first, exclude_paths=excludes) != condition_fingerprint(
        second, exclude_paths=excludes
    )


def test_review_requires_full_sha_and_unchanged_normative_blobs(tmp_path: Path) -> None:
    repository, design_sha = _reviewed_repository(tmp_path)

    review = validate_reviewed_design(repository)

    assert review.design_commit == design_sha
    assert review.reviewed_groups == ("A1-lr-screen",)

    outputs = repository / "docs/experimental-design/outputs.md"
    outputs.write_text(outputs.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    with pytest.raises(DesignError, match="outputs.md"):
        validate_reviewed_design(repository)


def test_review_rejects_short_design_sha(tmp_path: Path) -> None:
    repository = _design_repository(tmp_path)
    plan = repository / PLAN_PATH
    text = plan.read_text(encoding="utf-8")
    text = text.replace("Plan status: placeholder", "Plan status: reviewed")
    text = text.replace("Reviewed design commit: none", "Reviewed design commit: deadbeef")
    text = text.replace("Reviewed case groups: []", "Reviewed case groups: [A1-lr-screen]")
    plan.write_text(text, encoding="utf-8")

    with pytest.raises(DesignError, match="full 40-character"):
        validate_reviewed_design(repository)


def test_a1_preflight_enforces_exact_physical_cell_and_duplicate_reuse(
    tmp_path: Path,
) -> None:
    repository, _design_sha = _reviewed_repository(tmp_path)
    scaffold = repository / "experiments/01-a1-lr-screen"
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    config_path = scaffold / "run/001-a1-lr-1e-3.yaml"
    config = _a1_config()
    excludes = validate_catalog(repository).fingerprint_exclude_paths
    config["identity"]["condition_fingerprint"] = condition_fingerprint(
        config, exclude_paths=excludes
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _git(repository, "add", config_path.relative_to(repository).as_posix())

    validate_config_for_reviewed_design(
        config,
        repository=repository,
        config_path=config_path,
    )

    wrong = deepcopy(config)
    wrong["training"]["max_steps"] = 5691
    wrong["training"]["warmup_steps"] = 57
    wrong["identity"]["condition_fingerprint"] = condition_fingerprint(
        wrong, exclude_paths=excludes
    )
    with pytest.raises(DesignError, match="not an A1-lr-screen physical cell"):
        validate_config_for_reviewed_design(
            wrong,
            repository=repository,
            config_path=config_path,
        )
    config_path.write_text(yaml.safe_dump(wrong, sort_keys=False), encoding="utf-8")
    with pytest.raises(DesignError, match="not an A1-lr-screen physical cell"):
        tracked_training_identities(repository)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    duplicate_path = scaffold / "run/002-duplicate.yaml"
    duplicate = deepcopy(config)
    duplicate["experiment_name"] = "duplicate-label"
    duplicate["output"]["dir"] = "experiments/another/raw"
    duplicate["identity"]["condition_fingerprint"] = condition_fingerprint(
        duplicate, exclude_paths=excludes
    )
    assert duplicate["identity"]["condition_fingerprint"] == config["identity"][
        "condition_fingerprint"
    ]
    duplicate_path.write_text(yaml.safe_dump(duplicate, sort_keys=False), encoding="utf-8")
    _git(repository, "add", duplicate_path.relative_to(repository).as_posix())

    with pytest.raises(DesignError, match="Duplicate scientific condition fingerprints"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=config_path,
        )


@pytest.mark.parametrize(
    "learning_rate",
    (1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1, 2.56e-1, 5.12e-1),
)
def test_a1_preflight_accepts_reviewed_high_lr_cells(
    tmp_path: Path,
    learning_rate: float,
) -> None:
    repository, _design_sha = _reviewed_repository(tmp_path)
    scaffold = repository / "experiments/01-a1-lr-screen"
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    config_path = scaffold / "run/006-a1-lr-extension.yaml"
    config = _a1_config()
    config["training"]["learning_rate"] = learning_rate
    excludes = validate_catalog(repository).fingerprint_exclude_paths
    config["identity"]["condition_fingerprint"] = condition_fingerprint(
        config, exclude_paths=excludes
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _git(repository, "add", config_path.relative_to(repository).as_posix())

    validate_config_for_reviewed_design(
        config,
        repository=repository,
        config_path=config_path,
    )



def test_a1_preflight_rejects_unreviewed_1_024_extension(
    tmp_path: Path,
) -> None:
    repository, _design_sha = _reviewed_repository(tmp_path)
    scaffold = repository / "experiments/01-a1-lr-screen"
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    config_path = scaffold / "run/012-a1-lr-1p024.yaml"
    config = _a1_config()
    config["training"]["learning_rate"] = 1.024
    excludes = validate_catalog(repository).fingerprint_exclude_paths
    config["identity"]["condition_fingerprint"] = condition_fingerprint(
        config, exclude_paths=excludes
    )
    with pytest.raises(DesignError, match="outside the reviewed 14M grid"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=config_path,
        )


def test_a2_preflight_accepts_exact_six_cell_cohort(tmp_path: Path) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    run_dir = repository / "experiments/02-a2-l1-screen/run"
    cases = [
        ("012-a2-relu-control.yaml", _a2_config("A2-relu-control")),
        *[
            (
                f"{index:03d}-a2-l1-{weight:g}.yaml",
                _a2_config("A2-l1-screen", weight=weight),
            )
            for index, weight in enumerate(A2_L1_WEIGHTS, start=13)
        ],
    ]
    tracked: list[tuple[Path, dict[str, Any]]] = []
    for filename, config in cases:
        path = run_dir / filename
        _track_training_config(repository, path, config)
        tracked.append((path, config))

    identities = tracked_training_identities(repository)

    assert len(identities) == 6
    assert len({fingerprint for _path, _group, fingerprint in identities}) == 6
    assert sum(group == "A2-relu-control" for _path, group, _fingerprint in identities) == 1
    assert sum(group == "A2-l1-screen" for _path, group, _fingerprint in identities) == 5
    for path, config in tracked:
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=path,
        )


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    (
        ("model.topology_id", "A0"),
        ("model.site_gate", None),
        ("data.revision", "0" * 40),
        ("preprocessing.cache_id", "another-cache"),
        ("preprocessing.overwrite", True),
        ("run.seed", 1),
        (
            "run.training_schedule_hash",
            "5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c",
        ),
        ("training.max_steps", 1526),
        ("training.learning_rate", 3.2e-2),
        ("training.warmup_steps", 16),
        ("training.micro_batch_size", 8),
        ("validation.partition", "confirmation"),
        ("checkpoint.save_optimizer", True),
        ("activation_pressure.sites", ["a"]),
        ("activation_pressure.step_budget", 0.1),
        ("activation_pressure.eps", 1.0e-9),
        ("activation_pressure.log_thresholds", [0.0, 0.01]),
    ),
)
def test_a2_preflight_rejects_common_envelope_mutations(
    tmp_path: Path,
    dotted_path: str,
    value: Any,
) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/013-a2-l1-1.yaml"
    config = _a2_config("A2-l1-screen", weight=1.0)
    _set_config_value(config, dotted_path, value)
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="not an exact A2-l1-screen physical cell"):
        tracked_training_identities(repository)


@pytest.mark.parametrize("weight", (0.0, 0.2, 10.0))
def test_a2_l1_preflight_rejects_weights_outside_reviewed_grid(
    tmp_path: Path,
    weight: float,
) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/013-a2-l1.yaml"
    config = _a2_config("A2-l1-screen", weight=weight)
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="outside the reviewed grid"):
        tracked_training_identities(repository)


@pytest.mark.parametrize(
    ("source_group", "claimed_group", "weight"),
    (
        ("A2-relu-control", "A2-l1-screen", 0.0),
        ("A2-l1-screen", "A2-relu-control", 1.0),
    ),
)
def test_a2_preflight_rejects_cross_labeled_pressure_shapes(
    tmp_path: Path,
    source_group: str,
    claimed_group: str,
    weight: float,
) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/012-a2-cross-label.yaml"
    config = _a2_config(source_group, weight=weight)
    config["identity"]["group_id"] = claimed_group
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match=f"not an exact {claimed_group} physical cell"):
        tracked_training_identities(repository)


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    (
        ("activation_pressure.enabled", False),
        ("activation_pressure.method", "orthogonal_l1"),
    ),
)
def test_a2_l1_preflight_rejects_wrong_pressure_method_contract(
    tmp_path: Path,
    dotted_path: str,
    value: Any,
) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/013-a2-l1.yaml"
    config = _a2_config("A2-l1-screen", weight=1.0)
    _set_config_value(config, dotted_path, value)
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="not an exact A2-l1-screen physical cell"):
        tracked_training_identities(repository)


def test_a2_preflight_resolves_only_a_frozen_lr_decision(tmp_path: Path) -> None:
    repository = _design_repository(tmp_path)
    decisions = repository / DECISIONS_PATH
    original = decisions.read_text(encoding="utf-8")
    changed = original.replace(
        "| `lr_14m` | frozen | `6.4e-2` |",
        "| `lr_14m` | unresolved | `TODO:` |",
    )
    assert changed != original
    decisions.write_text(changed, encoding="utf-8")
    repository, _design_sha = _review_repository(
        repository,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/012-a2-relu.yaml"
    config = _a2_config("A2-relu-control")
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="Decision lr_14m must be frozen"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=path,
        )


def test_a2_preflight_uses_the_frozen_lr_decision_value(tmp_path: Path) -> None:
    repository = _design_repository(tmp_path)
    decisions = repository / DECISIONS_PATH
    original = decisions.read_text(encoding="utf-8")
    changed = original.replace(
        "| `lr_14m` | frozen | `6.4e-2` |",
        "| `lr_14m` | frozen | `3.2e-2` |",
    )
    assert changed != original
    decisions.write_text(changed, encoding="utf-8")
    repository, _design_sha = _review_repository(
        repository,
        reviewed_groups=A2_GROUPS,
    )
    path = repository / "experiments/02-a2-l1-screen/run/012-a2-relu.yaml"
    config = _a2_config("A2-relu-control")
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="training.learning_rate"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=path,
        )


def test_a2_preflight_rejects_stale_and_duplicate_fingerprints(tmp_path: Path) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=A2_GROUPS,
    )
    run_dir = repository / "experiments/02-a2-l1-screen/run"
    first_path = run_dir / "013-a2-l1-1.yaml"
    first = _a2_config("A2-l1-screen", weight=1.0)
    _track_training_config(
        repository,
        first_path,
        first,
        refresh_fingerprint=False,
    )
    with pytest.raises(DesignError, match="condition fingerprint does not match"):
        validate_config_for_reviewed_design(
            first,
            repository=repository,
            config_path=first_path,
        )

    _track_training_config(repository, first_path, first)
    duplicate_path = run_dir / "014-a2-l1-duplicate.yaml"
    duplicate = deepcopy(first)
    duplicate["experiment_name"] = "duplicate-a2-label"
    _track_training_config(repository, duplicate_path, duplicate)
    assert duplicate["identity"]["condition_fingerprint"] == first["identity"][
        "condition_fingerprint"
    ]
    with pytest.raises(DesignError, match="Duplicate scientific condition fingerprints"):
        validate_config_for_reviewed_design(
            first,
            repository=repository,
            config_path=first_path,
        )


def test_a2_preflight_rejects_an_unreviewed_a2_group(tmp_path: Path) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=("A2-relu-control",),
    )
    path = repository / "experiments/02-a2-l1-screen/run/013-a2-l1.yaml"
    config = _a2_config("A2-l1-screen", weight=1.0)
    _track_training_config(repository, path, config)

    with pytest.raises(DesignError, match="not in the reviewed scope"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=path,
        )


def test_non_a1_a2_groups_remain_fail_closed(tmp_path: Path) -> None:
    repository, _design_sha = _reviewed_repository(
        tmp_path,
        reviewed_groups=("A3-ol1-screen",),
    )
    path = repository / "experiments/02-a2-l1-screen/run/018-a3-ol1.yaml"
    config = _a2_config("A2-l1-screen", weight=1.0)
    config["identity"]["group_id"] = "A3-ol1-screen"
    config["activation_pressure"].update(
        {
            "method": "orthogonal_l1",
            "step_budget": 0.1,
        }
    )
    _track_training_config(repository, path, config)

    with pytest.raises(
        DesignError,
        match="Exact config membership validation is not implemented for A3-ol1-screen",
    ):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=path,
        )


def _design_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in (*NORMATIVE_DESIGN_PATHS, PLAN_PATH, WORKBOARD_PATH):
        source = REPOSITORY / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    plan = repository / PLAN_PATH
    plan_lines = plan.read_text(encoding="utf-8").splitlines()
    plan_lines = [
        "Plan status: placeholder"
        if line.startswith("Plan status:")
        else "Reviewed design commit: none"
        if line.startswith("Reviewed design commit:")
        else "Reviewed case groups: []"
        if line.startswith("Reviewed case groups:")
        else line
        for line in plan_lines
    ]
    plan.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    protocol = repository / PROTOCOL_PATH
    protocol.write_text(
        protocol.read_text(encoding="utf-8").replace(
            "`TODO:` freeze after the training and method blockers close",
            f"`{TRAINING_IMPLEMENTATION_ID}`",
        ),
        encoding="utf-8",
    )
    return repository


def _reviewed_repository(
    tmp_path: Path,
    *,
    reviewed_groups: tuple[str, ...] = ("A1-lr-screen",),
) -> tuple[Path, str]:
    repository = _design_repository(tmp_path)
    return _review_repository(repository, reviewed_groups=reviewed_groups)


def _review_repository(
    repository: Path,
    *,
    reviewed_groups: tuple[str, ...],
) -> tuple[Path, str]:
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Design Tests")
    _git(repository, "add", "docs")
    _git(repository, "commit", "-m", "Reviewed design snapshot")
    design_sha = _git(repository, "rev-parse", "HEAD").strip()
    plan = repository / PLAN_PATH
    text = plan.read_text(encoding="utf-8")
    text = text.replace("Plan status: placeholder", "Plan status: reviewed")
    text = text.replace("Reviewed design commit: none", f"Reviewed design commit: {design_sha}")
    rendered_groups = "[" + ", ".join(reviewed_groups) + "]"
    text = text.replace("Reviewed case groups: []", f"Reviewed case groups: {rendered_groups}")
    plan.write_text(text, encoding="utf-8")
    return repository, design_sha


def _a1_config() -> dict[str, object]:
    return {
        "experiment_name": "pythia_14m_a1_lr_1e_3",
        "identity": {
            "group_id": "A1-lr-screen",
            "condition_fingerprint": "0" * 64,
            "training_implementation_id": TRAINING_IMPLEMENTATION_ID,
        },
        "model": {
            "provider": "huggingface",
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "revision": "7386d9a4ae45aef494a6e704910394def3037fc5",
            "initialization": "random",
            "topology_id": "A0",
            "site_gate": None,
        },
        "data": {
            "name": "JeanKaddour/minipile",
            "revision": "18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0",
            "split": "train",
            "text_column": "text",
            "max_documents": None,
        },
        "tokenizer": {
            "name": "EleutherAI/pythia-14m-deduped",
            "revision": "7386d9a4ae45aef494a6e704910394def3037fc5",
        },
        "preprocessing": {
            "output_dir": "data/tokenized",
            "cache_id": "03-pythia-14m-minipile-random-full-10min",
            "tokens_sha256": "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c",
            "block_size": 2048,
            "append_eos": True,
            "overwrite": False,
        },
        "evaluation": {"metric": "validation_loss"},
        "run": {
            "seed": 0,
            "training_schedule_scheme": TRAINING_SCHEDULE_SCHEME,
            "model_initialization_seed": 0,
            "data_order_seed": 0,
            "training_schedule_hash": "5feffe55fe37c764e86c6709500f1b0afad85be652de127f5fc7c958a7eb481c",
        },
        "training": {
            "device": "cuda",
            "precision": "bfloat16",
            "max_steps": 1526,
            "learning_rate": 1.0e-3,
            "warmup_steps": 16,
            "gradient_accumulation_steps": 8,
            "micro_batch_size": 16,
            "log_every": 10,
            "optimizer": "adamw",
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1.0e-8,
            "weight_decay": 0.1,
        },
        "validation": {
            "enabled": True,
            "split": "validation",
            "max_documents": 500,
            "partition": "selection",
            "partition_scheme": "shuffled_source_documents_half_v1",
            "partition_seed": 20260718,
            "partition_hash": "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47",
            "tokens_sha256": "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19",
            "batch_size": 4,
            "eval_every_steps": 191,
            "eval_batches": None,
        },
        "checkpoint": {"save_final": True, "save_optimizer": False},
        "activation_pressure": {
            "enabled": False,
            "method": "none",
            "sites": ["h"],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001, 0.01],
        },
        "output": {"dir": "experiments/01-a1-lr-screen/raw"},
    }


def _a2_config(
    group_id: str,
    *,
    weight: float = 0.0,
) -> dict[str, Any]:
    if group_id not in A2_GROUPS:
        raise ValueError(f"Unsupported A2 test group: {group_id}")
    config: dict[str, Any] = deepcopy(_a1_config())
    config["experiment_name"] = f"pythia_14m_a2_{group_id}_{weight:g}"
    config["identity"]["group_id"] = group_id
    config["identity"]["condition_fingerprint"] = "0" * 64
    config["model"]["topology_id"] = "A1-H"
    config["model"]["site_gate"] = {"operator": "relu"}
    config["run"]["training_schedule_hash"] = (
        "35da3f6aa891a2248407344715e4c75e99cb518b17119a8e66004466a823a21c"
    )
    config["training"].update(
        {
            "max_steps": 5691,
            "learning_rate": 6.4e-2,
            "warmup_steps": 57,
        }
    )
    config["activation_pressure"].update(
        {
            "enabled": group_id == "A2-l1-screen",
            "method": "l1_naive" if group_id == "A2-l1-screen" else "none",
            "sites": ["h"],
            "weight": weight if group_id == "A2-l1-screen" else 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
            "log_thresholds": [0.0, 0.001, 0.01],
        }
    )
    config["output"]["dir"] = "experiments/02-a2-l1-screen/raw"
    return config


def _track_training_config(
    repository: Path,
    path: Path,
    config: dict[str, Any],
    *,
    refresh_fingerprint: bool = True,
) -> None:
    if refresh_fingerprint:
        excludes = validate_catalog(repository).fingerprint_exclude_paths
        config["identity"]["condition_fingerprint"] = condition_fingerprint(
            config,
            exclude_paths=excludes,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _git(repository, "add", path.relative_to(repository).as_posix())


def _set_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
