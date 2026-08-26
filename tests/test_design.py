from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import stat
import subprocess

import pytest
import yaml

from paper_exp.design import (
    CATALOG_PATH,
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
        "A2-relu-confirmation",
        "A2-spillover-confirmation",
        "A3-ol1-screen",
        "B1-threshold-screen",
        "B2-combined-screen",
        "B2-winner-confirmation",
        "C1-lr-screens",
        "C2-dense-controls",
        "C2-relu-controls",
        "C2-l1-screens",
        "C2-relu-confirmation",
        "C2-spillover-confirmation",
        "C3-ol1-controls",
        "C3-frontier-replication",
        "C3-winner-confirmation",
    )
    assert summary.groups["A1-lr-screen"]["conceptual_cells"] == 5
    assert summary.groups["A1-lr-screen"]["unique_cases"] == 5
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


def test_a1_preflight_accepts_8e_3_and_rejects_unreviewed_extension(
    tmp_path: Path,
) -> None:
    repository, _design_sha = _reviewed_repository(tmp_path)
    scaffold = repository / "experiments/01-a1-lr-screen"
    for name in ("run", "raw", "figs"):
        (scaffold / name).mkdir(parents=True, exist_ok=True)
    config_path = scaffold / "run/005-a1-lr-8e-3.yaml"
    config = _a1_config()
    config["training"]["learning_rate"] = 8.0e-3
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

    config["training"]["learning_rate"] = 1.6e-2
    config["identity"]["condition_fingerprint"] = condition_fingerprint(
        config, exclude_paths=excludes
    )
    with pytest.raises(DesignError, match="outside the reviewed 14M grid"):
        validate_config_for_reviewed_design(
            config,
            repository=repository,
            config_path=config_path,
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


def _reviewed_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = _design_repository(tmp_path)
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
    text = text.replace("Reviewed case groups: []", "Reviewed case groups: [A1-lr-screen]")
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


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
