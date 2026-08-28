from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import paper_exp.diagnostics.clipping_frontier as clipping
from paper_exp.diagnostics.logical_products import LOGICAL_MATMUL_STAGES


SITES = ["a", "m", "h", "q_post", "k_post", "v"]
THRESHOLDS = [0.0, 0.01, 0.03, 0.1, 0.3]
SELECTED_RUNS = [
    {
        "label": label,
        "tranche_id": "02-a2-l1-screen",
        "config_id": config_id,
        "run_id": run_id,
    }
    for label, config_id, run_id in (
        ("ReLU control", "012-a2-relu-control", "001-20260827-150809-2eb832f6"),
        ("L1 lambda=0.1", "013-a2-l1-1e-1", "001-20260827-150808-8117d1fe"),
        ("L1 lambda=0.5", "014-a2-l1-5e-1", "001-20260827-173546-360c077f"),
        ("L1 lambda=1", "015-a2-l1-1", "001-20260827-193752-3fbbd6c0"),
        ("L1 lambda=2", "016-a2-l1-2", "001-20260827-220532-79995961"),
        ("L1 lambda=5", "017-a2-l1-5", "001-20260828-000829-0959f855"),
    )
]
SITE_COUNTS = {
    "a": 239_075_328,
    "m": 239_075_328,
    "h": 956_301_312,
    "q_post": 239_075_328,
    "k_post": 239_075_328,
    "v": 239_075_328,
}
PRODUCT_COUNTS = {
    "qkv_projection": 91_804_925_952,
    "qk_scores": 244_932_673_536,
    "probability_value": 244_932_673_536,
    "attention_output_projection": 30_601_641_984,
    "mlp_w1": 122_406_567_936,
    "mlp_w2": 122_406_567_936,
}
ZERO_THRESHOLD_COUNTS = (
    (0, 5_142, 6_613_382_304, 384, 0, 115_487_154_560),
    (0, 4_322, 6_353_361_167, 640, 0, 115_638_036_992),
    (0, 2_552, 5_009_855_303, 512, 0, 119_723_018_752),
    (0, 5_191, 8_425, 256, 0, 120_462_950_272),
    (0, 5_609, 4_693_401_720, 512, 0, 121_553_470_208),
    (0, 8_028, 7_905_080_161, 256, 0, 122_150_657_024),
)
BLOCK_PRODUCTS = 857_085_050_880
LM_HEAD_PRODUCTS = 2_004_407_549_952
MODEL_PRODUCTS = 2_861_492_600_832


def test_exact_a2_clipping_rows_pass_complete_integer_validation() -> None:
    rows = _cohort_rows()

    clipping.validate_clipping_frontier_rows(
        rows,
        selected_runs=SELECTED_RUNS,
        thresholds=THRESHOLDS,
        sites=SITES,
        expected_validation_batches=38,
        expected_validation_sequences=152,
        expected_validation_tokens=311_296,
        validation_batch_size=4,
    )

    assert len(rows) == 30
    assert rows[0]["site_activation_count"] == SITE_COUNTS
    assert rows[0]["block_matmul_product_count"] == BLOCK_PRODUCTS
    assert rows[0]["lm_head_matmul_product_count"] == LM_HEAD_PRODUCTS
    assert rows[0]["model_matmul_product_count"] == MODEL_PRODUCTS
    assert [
        rows[index * len(THRESHOLDS)]["block_zero_product_count"]
        for index in range(len(SELECTED_RUNS))
    ] == [sum(values) for values in ZERO_THRESHOLD_COUNTS]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exactly one row"),
        (lambda rows: rows[1].update(threshold=0.03), "source/cutoff order"),
        (
            lambda rows: rows[0].update(
                potentially_avoidable_model_matmul_fraction=0.5
            ),
            "R_model is inconsistent",
        ),
        (lambda rows: rows[0]["site_zero_hits"].pop("v"), "per-site"),
        (
            lambda rows: rows[1]["matmul_product_count"].update(mlp_w2=1),
            "zero products|pooled block|denominators",
        ),
        (
            lambda rows: rows[1].update(
                validation_sequences=151,
                validation_tokens=309_248,
            ),
            "changes validation coverage",
        ),
        (
            lambda rows: rows[0].update(validation_loss=-1.0),
            "validation loss must be positive",
        ),
        (
            lambda rows: rows[0]["source_validation_cache"].pop("split"),
            "incomplete validation-cache identity",
        ),
    ],
)
def test_clipping_frontier_rows_fail_closed(
    mutation: Any,
    message: str,
) -> None:
    rows = _cohort_rows()
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        clipping.validate_clipping_frontier_rows(
            rows,
            selected_runs=SELECTED_RUNS,
            thresholds=THRESHOLDS,
            sites=SITES,
        )


def test_clipping_frontier_rows_validate_batch_arithmetic() -> None:
    rows = _cohort_rows()
    for row in rows:
        row["validation_batches"] = 37

    with pytest.raises(ValueError, match="batch and sequence coverage"):
        clipping.validate_clipping_frontier_rows(
            rows,
            selected_runs=SELECTED_RUNS,
            thresholds=THRESHOLDS,
            sites=SITES,
            validation_batch_size=4,
        )


def test_clipping_frontier_does_not_assert_monotonicity() -> None:
    rows = _cohort_rows()
    row = rows[1]
    row["matmul_zero_product_count"] = {
        name: max(0, value - 1)
        for name, value in row["matmul_zero_product_count"].items()
    }
    _refresh_product_summary(row)

    clipping.validate_clipping_frontier_rows(
        rows,
        selected_runs=SELECTED_RUNS,
        thresholds=THRESHOLDS,
        sites=SITES,
    )


def test_zero_threshold_rows_must_match_exact_propagation_reference() -> None:
    rows = _cohort_rows()
    reference = _propagation_reference(rows)

    clipping._validate_zero_threshold_reference(
        rows,
        selected_runs=SELECTED_RUNS,
        thresholds=THRESHOLDS,
        reference=reference,
        expected_validation_batches=38,
        expected_validation_sequences=152,
        expected_validation_tokens=311_296,
    )

    rows[0]["matmul_zero_product_count"]["qkv_projection"] += 1
    _refresh_product_summary(rows[0])
    with pytest.raises(ValueError, match="numerators do not match"):
        clipping._validate_zero_threshold_reference(
            rows,
            selected_runs=SELECTED_RUNS,
            thresholds=THRESHOLDS,
            reference=reference,
            expected_validation_batches=38,
            expected_validation_sequences=152,
            expected_validation_tokens=311_296,
        )


def test_zero_threshold_reference_pins_commit_and_artifact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    run_dir = (
        raw_dir
        / "019-a2-activation-propagation"
        / "001-20260828-110533-6ac813e6"
    )
    run_dir.mkdir(parents=True)
    artifact = _propagation_reference(_cohort_rows())["artifact"]
    artifact.update(
        schema_version=5,
        validation_partition="selection",
        validation_partition_hash="e" * 64,
        complete_named_partition=True,
        attention_implementation="eager",
    )
    artifact_path = run_dir / "activation_propagation.json"
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    for name, payload in (
        ("config.yaml", "{}\n"),
        ("metrics.json", "{}\n"),
        ("predictions.jsonl", "{}\n"),
    ):
        (run_dir / name).write_text(payload, encoding="utf-8")
    git_commit = "9" * 40
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tranche_id": "02-a2-l1-screen",
                "config_id": "019-a2-activation-propagation",
                "run_id": run_dir.name,
                "mode": "activation-propagation",
                "status": "completed",
                "git_dirty": False,
                "git_commit": git_commit,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reference = {
        "tranche_id": "02-a2-l1-screen",
        "config_id": "019-a2-activation-propagation",
        "run_id": run_dir.name,
        "git_commit": git_commit,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    validation_metadata = {
        "partition": "selection",
        "source_document_indices_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        clipping,
        "resolve_experiment_scaffold",
        lambda _tranche_id, **_kwargs: SimpleNamespace(raw_dir=raw_dir),
    )

    loaded = clipping._load_zero_threshold_reference(
        reference,
        selected_runs=SELECTED_RUNS,
        validation_metadata=validation_metadata,
    )
    assert loaded["source_artifact_sha256"] == reference["artifact_sha256"]

    wrong_commit = {**reference, "git_commit": "8" * 40}
    with pytest.raises(ValueError, match="exact completed clean"):
        clipping._load_zero_threshold_reference(
            wrong_commit,
            selected_runs=SELECTED_RUNS,
            validation_metadata=validation_metadata,
        )
    wrong_hash = {**reference, "artifact_sha256": "7" * 64}
    with pytest.raises(ValueError, match="pinned SHA-256"):
        clipping._load_zero_threshold_reference(
            wrong_hash,
            selected_runs=SELECTED_RUNS,
            validation_metadata=validation_metadata,
        )


def test_calibration_is_outcome_blind_and_creates_no_raw_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = tmp_path / "source"
    checkpoint = source_run / "checkpoints" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (source_run / "config.yaml").write_text(
        "training:\n  device: cpu\n  precision: float32\n", encoding="utf-8"
    )
    tokens_path = tmp_path / "data" / "selection.int32.bin"
    tokens_path.parent.mkdir()
    np.arange(9, dtype=np.int32).tofile(tokens_path)
    metadata = {
        "tokens_path": "data/selection.int32.bin",
        "dtype": "int32",
        "tokens": 9,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
        "block_size": 4,
        "split": "validation",
        "max_documents": 1,
        "partition": "selection",
        "partition_scheme": "test",
        "partition_seed": 0,
        "source_document_indices_sha256": "a" * 64,
    }
    (source_run / "manifest.json").write_text(
        json.dumps({"status": "completed", "tokenized_data": {"validation": metadata}}),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "experiments" / "02-a2-l1-screen" / "raw"
    config = {
        "experiment_name": "calibration",
        "model": {"provider": "huggingface", "name": "test", "initialization": "random"},
        "data": {"name": "test", "split": "train"},
        "evaluation": {"metric": "validation_loss"},
        "run": {"seed": 0},
        "validation": {"batch_size": 1, "eval_batches": None},
        "clipping_frontier": {
            "selected_runs": [SELECTED_RUNS[0]],
            "sites": ["h"],
        },
        "output": {"dir": "experiments/02-a2-l1-screen/raw"},
    }
    captured: dict[str, Any] = {}
    fake_torch = SimpleNamespace(
        float32="float32",
        __version__="test",
        version=SimpleNamespace(cuda=None),
    )
    fake_model = SimpleNamespace(to=lambda **_kwargs: None, eval=lambda: None)
    monkeypatch.setattr(clipping, "validate_diagnostic_config", lambda *_args: None)
    monkeypatch.setattr(
        clipping, "find_source_run", lambda *_args, **_kwargs: source_run
    )
    monkeypatch.setattr(
        clipping,
        "source_checkpoint_path",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        clipping._evaluation,
        "_load_clipping_dependencies",
        lambda: (fake_torch, np, object(), object()),
    )
    monkeypatch.setattr(
        clipping, "select_device", lambda *_args: SimpleNamespace(type="cpu")
    )
    monkeypatch.setattr(clipping, "select_dtype", lambda *_args: None)
    monkeypatch.setattr(clipping, "load_checkpoint_model", lambda *_args, **_kwargs: fake_model)

    def evaluate(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "validation_batches": 2,
            "validation_tokens": 8,
            "wall_seconds": 2.0,
            "validation_loss": 4.0,
            "achieved_sparsity": 0.5,
            "block_zero_product_count": 123,
        }

    monkeypatch.setattr(clipping._evaluation, "_evaluate_clipped_loss", evaluate)

    report = clipping.calibrate_clipping_frontier(config, repository=tmp_path)

    assert not raw_dir.exists()
    assert set(report) == {"calibration", "timing", "coverage", "memory", "runtime"}
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in ("validation_loss", "sparsity", "zero_product", "site_zero"):
        assert forbidden not in encoded
    assert report["coverage"]["validation_sequences"] == 2
    assert report["coverage"]["validation_tokens"] == 8
    assert captured["clipping_cfg"]["threshold"] == 0.0
    assert captured["measure_zero_products"] is True


def test_cohort_workflow_publishes_last_and_rechecks_checkpoint_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [SELECTED_RUNS[0]]
    source_run = tmp_path / "source"
    checkpoint = source_run / "checkpoints" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    tokens_path = tmp_path / "tokens.int32.bin"
    np.arange(9, dtype=np.int32).tofile(tokens_path)
    metadata = {
        "tokens_path": str(tokens_path),
        "dtype": "int32",
        "tokens": 9,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
        "block_size": 4,
        "split": "validation",
        "max_documents": 1,
        "partition": "selection",
        "partition_scheme": "test",
        "partition_seed": 0,
        "source_document_indices_sha256": "a" * 64,
    }
    (source_run / "config.yaml").write_text(
        "training:\n  device: cpu\n  precision: float32\n", encoding="utf-8"
    )
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "tokenized_data": {"validation": metadata},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "experiment_name": "clipping_frontier_checkpoint_recheck",
        "model": {
            "provider": "huggingface",
            "name": "test-random",
            "architecture": "test/model",
            "initialization": "random",
        },
        "data": {"name": "test/data", "split": "train"},
        "evaluation": {"metric": "validation_loss"},
        "run": {"seed": 0},
        "validation": {
            "batch_size": 1,
            "eval_batches": None,
        },
        "clipping_frontier": {
            "selected_runs": selected,
            "thresholds": [0.0],
            "sites": ["h"],
            "zero_threshold_reference": {},
        },
        "output": {"dir": str(tmp_path / "raw")},
    }
    stable = {
        "files": [
            {
                "path": "model.safetensors",
                "bytes": 10,
                "sha256": "b" * 64,
            }
        ]
    }
    changed = deepcopy(stable)
    changed["files"][0]["sha256"] = "c" * 64
    identity_state = {"change_call": None, "calls": 0}

    def checkpoint_identity(_path: Path) -> dict[str, Any]:
        identity_state["calls"] += 1
        if identity_state["calls"] == identity_state["change_call"]:
            return deepcopy(changed)
        return deepcopy(stable)

    monkeypatch.setattr(clipping, "validate_diagnostic_config", lambda *_args: None)
    monkeypatch.setattr(clipping, "find_source_run", lambda *_args, **_kwargs: source_run)
    monkeypatch.setattr(clipping, "validate_shared_validation_cache", lambda *_args: None)
    monkeypatch.setattr(clipping, "_validate_frontier_validation_request", lambda *_args: None)
    monkeypatch.setattr(
        clipping,
        "_load_zero_threshold_reference",
        lambda *_args, **_kwargs: {
            "artifact": {},
            "source_run": "reference/run",
            "source_artifact": "reference/activation_propagation.json",
            "source_artifact_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        clipping,
        "source_checkpoint_path",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(clipping, "verify_token_cache", lambda *_args, **_kwargs: tokens_path)
    monkeypatch.setattr(clipping, "_checkpoint_content_identity", checkpoint_identity)
    monkeypatch.setattr(
        clipping._evaluation,
        "_load_clipping_dependencies",
        lambda: (SimpleNamespace(float32="float32"), np, object(), object()),
    )
    monkeypatch.setattr(clipping, "select_device", lambda *_args: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(clipping, "select_dtype", lambda *_args: None)
    monkeypatch.setattr(clipping, "resolved_precision", lambda _dtype: "float32")
    fake_model = SimpleNamespace(to=lambda **_kwargs: None, eval=lambda: None)
    monkeypatch.setattr(clipping, "load_checkpoint_model", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(clipping._evaluation, "_evaluate_clipped_loss", lambda **_kwargs: {})
    monkeypatch.setattr(clipping, "validate_clipping_frontier_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(clipping, "_validate_zero_threshold_reference", lambda *_args, **_kwargs: None)

    original_complete_run = clipping.complete_run
    publication_checks: list[bool] = []

    def complete_after_specialized_artifact(run: Any, **kwargs: Any) -> Path:
        manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
        publication_checks.append(
            manifest["status"] == "running"
            and (run.run_dir / "clipping_frontier.jsonl").is_file()
            and not (run.run_dir / "metrics.json").exists()
            and not (run.run_dir / "predictions.jsonl").exists()
        )
        return original_complete_run(run, **kwargs)

    monkeypatch.setattr(clipping, "complete_run", complete_after_specialized_artifact)
    completed_dir = clipping.run_clipping_frontier(
        config,
        config_path="020-a2-posthoc-clipping-frontier.yaml",
        command="pytest clipping-frontier",
        run_id="completed",
    )
    completed_manifest = json.loads(
        (completed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert publication_checks == [True]
    assert completed_manifest["status"] == "completed"
    assert (completed_dir / "metrics.json").is_file()
    assert (completed_dir / "predictions.jsonl").is_file()

    identity_state.update(change_call=2, calls=0)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        clipping.run_clipping_frontier(
            config,
            config_path="020-a2-posthoc-clipping-frontier.yaml",
            command="pytest clipping-frontier",
            run_id="checkpoint-change",
        )

    run_dir = next(
        (
            tmp_path / "raw" / "020-a2-posthoc-clipping-frontier"
        ).glob("*-checkpoint-change")
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (run_dir / "clipping_frontier.jsonl").exists()

    identity_state.update(change_call=3, calls=0)
    with pytest.raises(RuntimeError, match="before publication"):
        clipping.run_clipping_frontier(
            config,
            config_path="020-a2-posthoc-clipping-frontier.yaml",
            command="pytest clipping-frontier",
            run_id="late-checkpoint-change",
        )

    late_run_dir = next(
        (
            tmp_path / "raw" / "020-a2-posthoc-clipping-frontier"
        ).glob("*-late-checkpoint-change")
    )
    late_manifest = json.loads(
        (late_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert late_manifest["status"] == "failed"
    assert not (late_run_dir / "clipping_frontier.jsonl").exists()


def test_completed_frontier_envelope_fails_closed_on_integrity_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _cohort_rows()
    metadata = _cache_identity()
    source_runs: list[Path] = []
    checkpoint_identities: list[dict[str, Any]] = []
    for index, selected in enumerate(SELECTED_RUNS):
        source_run = tmp_path / "sources" / str(index)
        checkpoint = source_run / "checkpoints" / "final"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(f"model-{index}".encode())
        (source_run / "config.yaml").write_text(
            "training:\n  device: cuda\n  precision: bfloat16\n",
            encoding="utf-8",
        )
        (source_run / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "tokenized_data": {"validation": metadata},
                }
            ),
            encoding="utf-8",
        )
        source_runs.append(source_run)
        checkpoint_identities.append(clipping._checkpoint_content_identity(checkpoint))
        for row in rows[index * len(THRESHOLDS) : (index + 1) * len(THRESHOLDS)]:
            row["source_run"] = source_run.relative_to(tmp_path).as_posix()
            row["source_checkpoint"] = checkpoint.relative_to(tmp_path).as_posix()
            row["source_checkpoint_content"] = checkpoint_identities[-1]

    token_path = tmp_path / metadata["tokens_path"]
    token_path.parent.mkdir(parents=True)
    token_path.write_bytes(b"token-cache")
    source_by_config = {
        selected["config_id"]: source_run
        for selected, source_run in zip(SELECTED_RUNS, source_runs, strict=True)
    }
    monkeypatch.setattr(
        clipping,
        "find_source_run",
        lambda selected, **_kwargs: source_by_config[selected["config_id"]],
    )
    monkeypatch.setattr(
        clipping,
        "source_checkpoint_path",
        lambda source_run, _manifest, **_kwargs: source_run / "checkpoints" / "final",
    )
    monkeypatch.setattr(
        clipping,
        "verify_token_cache",
        lambda *_args, **_kwargs: token_path,
    )
    reference = {
        **_propagation_reference(rows),
        "source_run": "experiments/02-a2-l1-screen/raw/019/reference",
        "source_artifact": (
            "experiments/02-a2-l1-screen/raw/019/reference/activation_propagation.json"
        ),
        "source_artifact_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        clipping,
        "_load_zero_threshold_reference",
        lambda *_args, **_kwargs: deepcopy(reference),
    )
    reference_config = {
        "tranche_id": "02-a2-l1-screen",
        "config_id": "019-a2-activation-propagation",
        "run_id": "001-reference",
        "git_commit": "a" * 40,
        "artifact_sha256": "f" * 64,
    }
    config = {
        "validation": {
            "split": "validation",
            "max_documents": 500,
            "partition": "selection",
            "partition_scheme": "shuffled_source_documents_half_v1",
            "partition_seed": 20_260_718,
            "partition_hash": "e" * 64,
            "tokens_sha256": "d" * 64,
            "batch_size": 4,
            "eval_batches": None,
        },
        "clipping_frontier": {
            "selected_runs": SELECTED_RUNS,
            "thresholds": THRESHOLDS,
            "sites": SITES,
            "zero_threshold_reference": reference_config,
        },
    }
    execution = {
        "requested_device": "cuda",
        "requested_precision": "bfloat16",
        "resolved_device": "cuda",
        "resolved_precision": "bfloat16",
    }
    manifest_frontier = {
        "schema_version": clipping.CLIPPING_FRONTIER_SCHEMA_VERSION,
        "selected_runs": SELECTED_RUNS,
        "mode": "threshold",
        "thresholds": THRESHOLDS,
        "sites": SITES,
        "measure_zero_products": True,
        "zero_threshold_reference": {
            **reference_config,
            "source_run": reference["source_run"],
            "source_artifact": reference["source_artifact"],
            "source_artifact_sha256": reference["source_artifact_sha256"],
            "audit_scope": clipping.ZERO_THRESHOLD_AUDIT_SCOPE,
        },
        "attention_implementation": "eager",
        "eval_batches": None,
        "batch_size": 4,
        "validation_batches": 38,
        "validation_sequences": 152,
        "validation_tokens": 311_296,
        "validation_cache_tokens": 311_739,
        "trailing_tokens_excluded": 443,
        "validation_partition": "selection",
        "validation_partition_hash": "e" * 64,
        "complete_named_partition": True,
        "execution": execution,
        "exact_zero_definition": "exact",
        "logical_opportunity_definition": "logical",
    }
    manifest = {
        "run_id": "001-frontier",
        "source_runs": [path.relative_to(tmp_path).as_posix() for path in source_runs],
        "source_checkpoints": [
            (path / "checkpoints" / "final").relative_to(tmp_path).as_posix()
            for path in source_runs
        ],
        "source_manifest_statuses": ["completed"] * len(source_runs),
        "source_checkpoint_contents": checkpoint_identities,
        "tokenized_data": {"validation": metadata},
        "clipping_frontier": manifest_frontier,
    }
    total_tokens = 311_296 * len(rows)
    metrics = {
        "clipping_frontier/sources": 6,
        "clipping_frontier/cutoffs": 5,
        "clipping_frontier/points": 30,
        "clipping_frontier/validation_batches_per_point": 38,
        "clipping_frontier/validation_sequences_per_point": 152,
        "clipping_frontier/validation_tokens_per_point": 311_296,
        "clipping_frontier/total_point_tokens": total_tokens,
        "clipping_frontier/wall_seconds": 10.0,
        "clipping_frontier/tokens_per_second": total_tokens / 10.0,
        "clipping_frontier/peak_gpu_memory_mb": 100.0,
        "clipping_frontier/peak_gpu_reserved_mb": 120.0,
    }
    run_dir = (
        tmp_path
        / "experiments/02-a2-l1-screen/raw/020-a2-posthoc-clipping-frontier/001-frontier"
    )
    run_dir.mkdir(parents=True)

    def validate(
        *,
        rows_value: list[dict[str, Any]] | None = None,
        manifest_value: dict[str, Any] | None = None,
        metrics_value: dict[str, Any] | None = None,
    ) -> None:
        clipping.validate_completed_clipping_frontier_artifacts(
            run_dir=run_dir,
            config=config,
            manifest=manifest if manifest_value is None else manifest_value,
            metrics=metrics if metrics_value is None else metrics_value,
            rows=rows if rows_value is None else rows_value,
            repository=tmp_path,
        )

    validate()
    bad_manifest = deepcopy(manifest)
    bad_manifest["source_runs"][0] = "wrong/source"
    with pytest.raises(ValueError, match="manifest source runs"):
        validate(manifest_value=bad_manifest)
    bad_metrics = deepcopy(metrics)
    bad_metrics["clipping_frontier/points"] = 29
    with pytest.raises(ValueError, match="metric .*points"):
        validate(metrics_value=bad_metrics)
    bad_rows = deepcopy(rows)
    bad_rows[0]["matmul_zero_product_count"]["qkv_projection"] += 1
    _refresh_product_summary(bad_rows[0])
    with pytest.raises(ValueError, match="numerators do not match"):
        validate(rows_value=bad_rows)


def _cohort_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, selected in enumerate(SELECTED_RUNS):
        baseline = dict(
            zip(LOGICAL_MATMUL_STAGES, ZERO_THRESHOLD_COUNTS[source_index], strict=True)
        )
        for threshold_index, threshold in enumerate(THRESHOLDS):
            zero_counts = {
                operation: min(
                    PRODUCT_COUNTS[operation],
                    baseline[operation] + threshold_index * (operation_index + 1),
                )
                for operation_index, operation in enumerate(LOGICAL_MATMUL_STAGES)
            }
            site_hits = {
                site: min(
                    count,
                    int(count * (0.10 + 0.01 * source_index + 0.005 * threshold_index)),
                )
                for site, count in SITE_COUNTS.items()
            }
            row = {
                "schema_version": clipping.CLIPPING_FRONTIER_SCHEMA_VERSION,
                "label": selected["label"],
                "source_tranche_id": selected["tranche_id"],
                "source_config_id": selected["config_id"],
                "source_run_id": selected["run_id"],
                "source_run": f"experiments/{selected['tranche_id']}/raw/{selected['config_id']}/{selected['run_id']}",
                "source_manifest_status": "completed",
                "source_checkpoint": f"checkpoints/{source_index}/final",
                "source_checkpoint_content": {
                    "files": [
                        {
                            "path": "model.safetensors",
                            "bytes": 1,
                            "sha256": f"{source_index:064x}",
                        }
                    ]
                },
                "source_validation_cache": _cache_identity(),
                "event": "clipping_sweep",
                "mode": "threshold",
                "threshold": threshold,
                "quantile": None,
                "rms_multiplier": None,
                "rms_scope": None,
                "sites": list(SITES),
                "site_achieved_sparsity": {
                    site: site_hits[site] / SITE_COUNTS[site] for site in SITES
                },
                "site_zero_hits": site_hits,
                "site_activation_count": dict(SITE_COUNTS),
                "validation_loss": 4.0 + 0.01 * source_index + 0.001 * threshold_index,
                "achieved_sparsity": sum(site_hits.values()) / sum(SITE_COUNTS.values()),
                "validation_batches": 38,
                "validation_sequences": 152,
                "validation_tokens": 311_296,
                "wall_seconds": 1.0,
                "tokens_per_second": 311_296.0,
                "matmul_zero_product_count": zero_counts,
                "matmul_product_count": dict(PRODUCT_COUNTS),
            }
            _refresh_product_summary(row)
            rows.append(row)
    return rows


def _refresh_product_summary(row: dict[str, Any]) -> None:
    zero_counts = row["matmul_zero_product_count"]
    product_counts = row["matmul_product_count"]
    block_zeros = sum(zero_counts.values())
    block_products = sum(product_counts.values())
    model_products = block_products + LM_HEAD_PRODUCTS
    row.update(
        {
            "matmul_zero_product_fraction": {
                operation: zero_counts[operation] / product_counts[operation]
                for operation in LOGICAL_MATMUL_STAGES
            },
            "block_zero_product_count": block_zeros,
            "block_matmul_product_count": block_products,
            "lm_head_matmul_product_count": LM_HEAD_PRODUCTS,
            "model_matmul_product_count": model_products,
            "potentially_avoidable_block_matmul_fraction": block_zeros
            / block_products,
            "potentially_avoidable_model_matmul_fraction": block_zeros
            / model_products,
        }
    )


def _propagation_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = []
    for source_index, selected in enumerate(SELECTED_RUNS):
        row = rows[source_index * len(THRESHOLDS)]
        methods.append(
            {
                "config_id": selected["config_id"],
                "run_id": selected["run_id"],
                "endpoint": {
                    "block_zero_product_count": row["block_zero_product_count"],
                    "block_product_count": row["block_matmul_product_count"],
                    "lm_head_product_count": row["lm_head_matmul_product_count"],
                    "model_product_count": row["model_matmul_product_count"],
                    "per_operation": {
                        operation: {
                            "zero_product_count": row["matmul_zero_product_count"][operation],
                            "product_count": row["matmul_product_count"][operation],
                        }
                        for operation in LOGICAL_MATMUL_STAGES
                    },
                    "zero_sites": {
                        f"z_{site}": {"total": SITE_COUNTS[site]} for site in SITES
                    },
                },
            }
        )
    return {
        "artifact": {
            "validation_batches": 38,
            "validation_sequences": 152,
            "validation_tokens": 311_296,
            "methods": methods,
        }
    }


def _cache_identity() -> dict[str, Any]:
    return {
        "tokens_path": "data/tokenized/selection/tokens.int32.bin",
        "dtype": "int32",
        "tokens": 311_739,
        "tokens_bytes": 1_246_956,
        "tokens_sha256": "d" * 64,
        "block_size": 2_048,
        "split": "validation",
        "max_documents": 500,
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20_260_718,
        "source_document_indices_sha256": "e" * 64,
    }
