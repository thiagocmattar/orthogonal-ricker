from __future__ import annotations

import math
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activations import resolve_site_aliases
from paper_exp.config import validate_diagnostic_config
from paper_exp.data import verify_token_cache
from paper_exp.launch import repository_path, resolve_experiment_scaffold
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import CORE_RUN_ARTIFACTS, RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_jsonl

from . import clipping_evaluation as _evaluation
from .clipping import _checkpoint_content_identity, _file_sha256
from .evaluation import (
    eval_starts,
    peak_gpu_memory_mb,
    peak_gpu_reserved_mb,
    resolved_precision,
    select_device,
    select_dtype,
)
from .logical_products import LOGICAL_MATMUL_STAGES
from .sources import find_source_run
from .sources import portable_path as _portable_path
from .sources import source_checkpoint_path
from .sources import validate_shared_validation_cache


CLIPPING_FRONTIER_SCHEMA_VERSION = 1
ZERO_THRESHOLD_AUDIT_SCOPE = (
    "Only zero-threshold per-operation integer counts and operation/block/"
    "LM-head/model denominators are compared with diagnostic 019. Validation "
    "losses and all paired deltas come solely from this diagnostic's own "
    "zero-threshold rows."
)


def run_clipping_frontier(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
    repository: str | Path | None = None,
) -> Path:
    """Evaluate one ordered clipping grid over pinned checkpoint sources."""

    root = repository_path(repository)
    validate_diagnostic_config(config, "clipping_frontier")
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode="clipping-frontier",
        run_id=run_id,
        repository=root,
    ) as run:
        return _run_clipping_frontier(run, repository=root)


def calibrate_clipping_frontier(
    config: dict[str, Any],
    *,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Time one production-shaped point without creating scientific artifacts."""

    root = repository_path(repository)
    validate_diagnostic_config(config, "clipping_frontier")
    frontier_config = config["clipping_frontier"]
    validation_config = config["validation"]
    selected = frontier_config["selected_runs"][0]
    sites = list(resolve_site_aliases(frontier_config["sites"]))
    started = time.perf_counter()

    torch, np, auto_model, modeling_gpt_neox = _evaluation._load_clipping_dependencies()
    np.random.seed(int(config["run"]["seed"]))
    source_run = find_source_run(
        selected,
        section="clipping_frontier",
        repository=root,
        require_final=True,
    )
    source_manifest = read_json(source_run / "manifest.json")
    validation_metadata = (source_manifest.get("tokenized_data") or {}).get(
        "validation"
    )
    if not isinstance(validation_metadata, dict):
        raise ValueError(
            "Selected clipping-frontier calibration source has no validation token cache."
        )
    validate_shared_validation_cache([source_manifest], validation_metadata)
    _validate_frontier_validation_request(validation_config, validation_metadata)
    checkpoint_path = source_checkpoint_path(
        source_run,
        source_manifest,
        repository=root,
        require_final=True,
    )
    checkpoint_identity = _checkpoint_content_identity(checkpoint_path)
    validation_tokens_path = verify_token_cache(
        _cache_metadata_with_resolved_path(validation_metadata, repository=root),
        context="Clipping frontier calibration validation cache",
    )
    validation_cache = _frontier_validation_cache_identity(
        validation_metadata,
        validation_tokens_path=validation_tokens_path,
        repository=root,
    )
    _validate_frontier_cache_identity(validation_cache, prefix="Calibration")
    validation_tokens = np.memmap(validation_tokens_path, dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(validation_config["batch_size"])
    eval_batches = validation_config.get("eval_batches")
    starts = eval_starts(
        validation_tokens,
        block_size,
        eval_batches=eval_batches,
        batch_size=batch_size,
        np=np,
    )
    validation_sequences = len(starts)
    validation_tokens_count = validation_sequences * block_size
    validation_batches = math.ceil(validation_sequences / batch_size)

    execution_request = _shared_execution_request([source_run])
    device = select_device(torch, execution_request["device"])
    dtype = select_dtype(torch, device, execution_request["precision"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    setup_wall_seconds = time.perf_counter() - started

    result = _evaluation._evaluate_clipped_loss(
        model=model,
        torch=torch,
        np=np,
        tokens=validation_tokens,
        block_size=block_size,
        batch_size=batch_size,
        eval_batches=eval_batches,
        starts=starts,
        device=device,
        dtype=dtype,
        clipping_cfg={
            "enabled": True,
            "mode": "threshold",
            "sites": sites,
            "threshold": 0.0,
        },
        measure_zero_products=True,
        modeling_gpt_neox=modeling_gpt_neox,
    )
    evaluation_wall_seconds = _finite_row_number(
        result.get("wall_seconds"), "Calibration evaluation wall time"
    )
    if evaluation_wall_seconds <= 0.0:
        raise RuntimeError("Clipping-frontier calibration wall time must be positive.")
    if (
        result.get("validation_batches") != validation_batches
        or result.get("validation_tokens") != validation_tokens_count
    ):
        raise RuntimeError(
            "Clipping-frontier calibration did not cover the complete selection partition."
        )
    allocated_mb = peak_gpu_memory_mb(torch, device)
    reserved_mb = peak_gpu_reserved_mb(torch, device)
    runtime = _calibration_runtime_identity(torch, device, dtype)
    del result
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if _checkpoint_content_identity(checkpoint_path) != checkpoint_identity:
        raise RuntimeError(
            "Clipping-frontier calibration source checkpoint changed during evaluation."
        )
    if (
        validation_tokens_path.stat().st_size != validation_cache["tokens_bytes"]
        or _file_sha256(validation_tokens_path) != validation_cache["tokens_sha256"]
    ):
        raise RuntimeError(
            "Clipping-frontier calibration validation cache changed during evaluation."
        )
    total_wall_seconds = time.perf_counter() - started
    return {
        "calibration": "clipping-frontier",
        "timing": {
            "setup_wall_seconds": setup_wall_seconds,
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "total_wall_seconds": total_wall_seconds,
            "evaluation_tokens_per_second": (
                validation_tokens_count / evaluation_wall_seconds
            ),
        },
        "coverage": {
            "validation_batches": validation_batches,
            "validation_sequences": validation_sequences,
            "validation_tokens": validation_tokens_count,
            "batch_size": batch_size,
            "block_size": block_size,
            "complete_named_partition": bool(validation_metadata.get("partition"))
            and eval_batches is None,
        },
        "memory": {
            "peak_gpu_allocated_mb": allocated_mb,
            "peak_gpu_reserved_mb": reserved_mb,
        },
        "runtime": runtime,
    }


def _run_clipping_frontier(run: RunHandle, *, repository: Path) -> Path:
    config = run.config
    frontier_config = config["clipping_frontier"]
    validation_config = config["validation"]
    selected_runs = frontier_config["selected_runs"]
    thresholds = [float(value) for value in frontier_config["thresholds"]]
    sites = list(resolve_site_aliases(frontier_config["sites"]))

    torch, np, auto_model, modeling_gpt_neox = _evaluation._load_clipping_dependencies()
    np.random.seed(int(config["run"]["seed"]))
    source_runs = [
        find_source_run(
            item,
            section="clipping_frontier",
            repository=repository,
            require_final=True,
        )
        for item in selected_runs
    ]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    validation_metadata = (source_manifests[0].get("tokenized_data") or {}).get(
        "validation"
    )
    if not isinstance(validation_metadata, dict):
        raise ValueError(
            "Selected clipping-frontier source runs have no validation token cache."
        )
    validate_shared_validation_cache(source_manifests, validation_metadata)
    _validate_frontier_validation_request(validation_config, validation_metadata)
    zero_threshold_reference = _load_zero_threshold_reference(
        frontier_config["zero_threshold_reference"],
        selected_runs=selected_runs,
        validation_metadata=validation_metadata,
        repository=repository,
    )
    source_checkpoints = [
        source_checkpoint_path(
            source_run,
            manifest,
            repository=repository,
            require_final=True,
        )
        for source_run, manifest in zip(source_runs, source_manifests, strict=True)
    ]
    checkpoint_identities = [
        _checkpoint_content_identity(path) for path in source_checkpoints
    ]

    validation_tokens_path = verify_token_cache(
        _cache_metadata_with_resolved_path(validation_metadata, repository=repository),
        context="Clipping frontier validation cache",
    )
    validation_cache = _frontier_validation_cache_identity(
        validation_metadata,
        validation_tokens_path=validation_tokens_path,
        repository=repository,
    )
    validation_tokens = np.memmap(validation_tokens_path, dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(validation_config["batch_size"])
    eval_batches = validation_config.get("eval_batches")
    starts = eval_starts(
        validation_tokens,
        block_size,
        eval_batches=eval_batches,
        batch_size=batch_size,
        np=np,
    )
    validation_sequences = len(starts)
    validation_token_count = validation_sequences * block_size
    validation_batches = math.ceil(validation_sequences / batch_size)

    execution_request = _shared_execution_request(source_runs)
    device = select_device(torch, execution_request["device"])
    dtype = select_dtype(torch, device, execution_request["precision"])
    execution = {
        "requested_device": execution_request["device"],
        "requested_precision": execution_request["precision"],
        "resolved_device": str(device),
        "resolved_precision": resolved_precision(dtype),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for selected, source_run, source_manifest, checkpoint_path, checkpoint_identity in zip(
        selected_runs,
        source_runs,
        source_manifests,
        source_checkpoints,
        checkpoint_identities,
        strict=True,
    ):
        print(
            f"Measuring clipping frontier for {selected['label']} from {source_run}",
            flush=True,
        )
        model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        for threshold in thresholds:
            clipping_config = {
                "enabled": True,
                "mode": "threshold",
                "sites": sites,
                "threshold": threshold,
            }
            result = _evaluation._evaluate_clipped_loss(
                model=model,
                torch=torch,
                np=np,
                tokens=validation_tokens,
                block_size=block_size,
                batch_size=batch_size,
                eval_batches=eval_batches,
                starts=starts,
                device=device,
                dtype=dtype,
                clipping_cfg=clipping_config,
                measure_zero_products=True,
                modeling_gpt_neox=modeling_gpt_neox,
            )
            rows.append(
                {
                    "schema_version": CLIPPING_FRONTIER_SCHEMA_VERSION,
                    "label": str(selected["label"]),
                    "source_tranche_id": str(selected["tranche_id"]),
                    "source_config_id": str(selected["config_id"]),
                    "source_run_id": str(selected["run_id"]),
                    "source_run": _portable_path(source_run, root=repository),
                    "source_manifest_status": source_manifest.get("status"),
                    "source_checkpoint": _portable_path(
                        checkpoint_path,
                        root=repository,
                    ),
                    "source_checkpoint_content": checkpoint_identity,
                    "source_validation_cache": validation_cache,
                    "validation_sequences": validation_sequences,
                    **result,
                }
            )
        if _checkpoint_content_identity(checkpoint_path) != checkpoint_identity:
            raise RuntimeError(
                f"Clipping-frontier source checkpoint changed during evaluation: {checkpoint_path}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_checkpoint_identities = [
        _checkpoint_content_identity(path) for path in source_checkpoints
    ]
    if final_checkpoint_identities != checkpoint_identities:
        raise RuntimeError(
            "One or more clipping-frontier source checkpoints changed before publication."
        )

    if (
        validation_tokens_path.stat().st_size != validation_cache["tokens_bytes"]
        or _file_sha256(validation_tokens_path) != validation_cache["tokens_sha256"]
    ):
        raise RuntimeError(
            "Clipping-frontier validation cache changed during evaluation."
        )

    validate_clipping_frontier_rows(
        rows,
        selected_runs=selected_runs,
        thresholds=thresholds,
        sites=sites,
        expected_validation_batches=validation_batches,
        expected_validation_sequences=validation_sequences,
        expected_validation_tokens=validation_token_count,
        validation_batch_size=batch_size,
    )
    _validate_zero_threshold_reference(
        rows,
        selected_runs=selected_runs,
        thresholds=thresholds,
        reference=zero_threshold_reference,
        expected_validation_batches=validation_batches,
        expected_validation_sequences=validation_sequences,
        expected_validation_tokens=validation_token_count,
    )
    wall_seconds = time.perf_counter() - started
    total_point_tokens = validation_token_count * len(rows)
    metrics = {
        "clipping_frontier/sources": len(selected_runs),
        "clipping_frontier/cutoffs": len(thresholds),
        "clipping_frontier/points": len(rows),
        "clipping_frontier/validation_batches_per_point": validation_batches,
        "clipping_frontier/validation_sequences_per_point": validation_sequences,
        "clipping_frontier/validation_tokens_per_point": validation_token_count,
        "clipping_frontier/total_point_tokens": total_point_tokens,
        "clipping_frontier/wall_seconds": wall_seconds,
        "clipping_frontier/tokens_per_second": (
            total_point_tokens / wall_seconds if wall_seconds > 0 else None
        ),
        "clipping_frontier/peak_gpu_memory_mb": peak_gpu_memory_mb(torch, device),
        "clipping_frontier/peak_gpu_reserved_mb": peak_gpu_reserved_mb(torch, device),
    }
    manifest_updates = {
        "source_runs": [
            _portable_path(path, root=repository) for path in source_runs
        ],
        "source_checkpoints": [
            _portable_path(path, root=repository) for path in source_checkpoints
        ],
        "source_manifest_statuses": [
            manifest.get("status") for manifest in source_manifests
        ],
        "source_checkpoint_contents": checkpoint_identities,
        "tokenized_data": {"validation": validation_metadata},
        "clipping_frontier": {
            "schema_version": CLIPPING_FRONTIER_SCHEMA_VERSION,
            "selected_runs": selected_runs,
            "mode": "threshold",
            "thresholds": thresholds,
            "sites": sites,
            "measure_zero_products": True,
            "zero_threshold_reference": {
                **frontier_config["zero_threshold_reference"],
                "source_run": zero_threshold_reference["source_run"],
                "source_artifact": zero_threshold_reference["source_artifact"],
                "source_artifact_sha256": zero_threshold_reference[
                    "source_artifact_sha256"
                ],
                "audit_scope": ZERO_THRESHOLD_AUDIT_SCOPE,
            },
            "attention_implementation": "eager",
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "validation_batches": validation_batches,
            "validation_sequences": validation_sequences,
            "validation_tokens": validation_token_count,
            "validation_cache_tokens": int(len(validation_tokens)),
            "trailing_tokens_excluded": int(len(validation_tokens))
            - validation_token_count,
            "validation_partition": validation_metadata.get("partition"),
            "validation_partition_hash": validation_metadata.get(
                "source_document_indices_sha256"
            ),
            "complete_named_partition": bool(validation_metadata.get("partition"))
            and eval_batches is None,
            "execution": execution,
            "exact_zero_definition": (
                "A tensor coordinate is zero iff its computed value compares equal "
                "to numeric 0 with no tolerance."
            ),
            "logical_opportunity_definition": (
                "R_block uses pooled exact zero-operand products in QKV, valid-causal "
                "QK, valid-causal PV, Wo, W1, and W2. R_model keeps that numerator "
                "and adds the dense LM head to the denominator. Neither is measured speedup."
            ),
        },
    }

    write_jsonl(run.run_dir / "clipping_frontier.jsonl", rows)
    return complete_run(
        run,
        metrics=metrics,
        predictions=rows,
        manifest_updates=manifest_updates,
    )


def validate_clipping_frontier_rows(
    rows: Any,
    *,
    selected_runs: list[dict[str, Any]],
    thresholds: list[float],
    sites: list[str],
    expected_validation_batches: int | None = None,
    expected_validation_sequences: int | None = None,
    expected_validation_tokens: int | None = None,
    validation_batch_size: int | None = None,
) -> None:
    """Fail closed on an incomplete or arithmetically inconsistent cohort."""

    if not isinstance(rows, list):
        raise ValueError("Clipping frontier rows must be a list.")
    expected_points = [
        (selected, float(threshold))
        for selected in selected_runs
        for threshold in thresholds
    ]
    if len(rows) != len(expected_points):
        raise ValueError(
            "Clipping frontier must contain exactly one row per source/cutoff pair."
        )
    if validation_batch_size is not None:
        _positive_row_integer(validation_batch_size, "Clipping frontier batch size")

    reference_cache: Any = None
    reference_coverage: tuple[int, int, int] | None = None
    reference_site_counts: Any = None
    reference_product_counts: Any = None
    reference_lm_head_count: int | None = None
    checkpoint_by_source: dict[tuple[str, str, str], Any] = {}
    path_by_source: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    for index, (row, (selected, threshold)) in enumerate(
        zip(rows, expected_points, strict=True)
    ):
        prefix = f"Clipping frontier row {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must be an object.")
        if row.get("schema_version") != CLIPPING_FRONTIER_SCHEMA_VERSION:
            raise ValueError(f"{prefix} has an unsupported schema version.")
        expected_identity = {
            "label": str(selected["label"]),
            "source_tranche_id": str(selected["tranche_id"]),
            "source_config_id": str(selected["config_id"]),
            "source_run_id": str(selected["run_id"]),
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise ValueError(f"{prefix} has the wrong {field}.")
        source_key = (
            expected_identity["source_tranche_id"],
            expected_identity["source_config_id"],
            expected_identity["source_run_id"],
        )
        if row.get("source_manifest_status") != "completed":
            raise ValueError(f"{prefix} does not name a completed source.")
        if not isinstance(row.get("source_run"), str) or not row["source_run"]:
            raise ValueError(f"{prefix} has no source run path.")
        if not isinstance(row.get("source_checkpoint"), str) or not row["source_checkpoint"]:
            raise ValueError(f"{prefix} has no source checkpoint path.")
        if Path(row["source_run"]).is_absolute() or Path(
            row["source_checkpoint"]
        ).is_absolute():
            raise ValueError(f"{prefix} source paths must be repository-relative.")
        source_paths = (row["source_run"], row["source_checkpoint"])
        prior_paths = path_by_source.setdefault(source_key, source_paths)
        if source_paths != prior_paths:
            raise ValueError(f"{prefix} changes source paths within one checkpoint.")
        _validate_checkpoint_identity(row.get("source_checkpoint_content"), prefix=prefix)
        prior_checkpoint = checkpoint_by_source.setdefault(
            source_key, row["source_checkpoint_content"]
        )
        if row["source_checkpoint_content"] != prior_checkpoint:
            raise ValueError(f"{prefix} changes checkpoint content within one source.")

        if row.get("mode") != "threshold":
            raise ValueError(f"{prefix} must use absolute threshold clipping.")
        row_threshold = _finite_row_number(row.get("threshold"), f"{prefix} threshold")
        if row_threshold != threshold:
            raise ValueError(f"{prefix} is out of source/cutoff order.")
        if row.get("quantile") is not None or row.get("rms_multiplier") is not None:
            raise ValueError(f"{prefix} mixes clipping modes.")
        if row.get("sites") != sites:
            raise ValueError(f"{prefix} has the wrong clipping sites or site order.")

        batches = _positive_row_integer(row.get("validation_batches"), f"{prefix} batches")
        sequences = _positive_row_integer(
            row.get("validation_sequences"), f"{prefix} sequences"
        )
        tokens = _positive_row_integer(row.get("validation_tokens"), f"{prefix} tokens")
        for actual, expected, label in (
            (batches, expected_validation_batches, "batches"),
            (sequences, expected_validation_sequences, "sequences"),
            (tokens, expected_validation_tokens, "tokens"),
        ):
            if expected is not None and actual != expected:
                raise ValueError(f"{prefix} has the wrong validation {label}.")
        coverage = (batches, sequences, tokens)
        if reference_coverage is None:
            reference_coverage = coverage
        elif coverage != reference_coverage:
            raise ValueError(f"{prefix} changes validation coverage across the cohort.")
        if (
            validation_batch_size is not None
            and batches != math.ceil(sequences / validation_batch_size)
        ):
            raise ValueError(f"{prefix} has inconsistent batch and sequence coverage.")
        if tokens % sequences != 0:
            raise ValueError(f"{prefix} has inconsistent token and sequence coverage.")
        validation_loss = _finite_row_number(
            row.get("validation_loss"), f"{prefix} validation loss"
        )
        if validation_loss <= 0.0:
            raise ValueError(f"{prefix} validation loss must be positive.")
        wall_seconds = _finite_row_number(row.get("wall_seconds"), f"{prefix} wall time")
        if wall_seconds < 0.0:
            raise ValueError(f"{prefix} wall time must be non-negative.")

        cache = row.get("source_validation_cache")
        _validate_frontier_cache_identity(cache, prefix=prefix)
        if int(cache["block_size"]) * sequences != tokens:
            raise ValueError(f"{prefix} validation cache block size is inconsistent.")
        if reference_cache is None:
            reference_cache = cache
        elif cache != reference_cache:
            raise ValueError(f"{prefix} does not share the cohort validation cache.")

        site_hits = row.get("site_zero_hits")
        site_counts = row.get("site_activation_count")
        site_fractions = row.get("site_achieved_sparsity")
        if not all(isinstance(value, dict) for value in (site_hits, site_counts, site_fractions)):
            raise ValueError(f"{prefix} has incomplete per-site exact-zero accounting.")
        expected_site_keys = set(sites)
        if any(set(value) != expected_site_keys for value in (site_hits, site_counts, site_fractions)):
            raise ValueError(f"{prefix} has incomplete per-site exact-zero accounting.")
        total_site_hits = 0
        total_site_count = 0
        for site in sites:
            hits = _nonnegative_row_integer(site_hits[site], f"{prefix} {site} zero hits")
            count = _positive_row_integer(site_counts[site], f"{prefix} {site} count")
            if hits > count:
                raise ValueError(f"{prefix} has more {site} zero hits than elements.")
            fraction = _finite_row_number(
                site_fractions[site], f"{prefix} {site} sparsity"
            )
            _require_close(fraction, hits / count, f"{prefix} {site} sparsity")
            total_site_hits += hits
            total_site_count += count
        if reference_site_counts is None:
            reference_site_counts = site_counts
        elif site_counts != reference_site_counts:
            raise ValueError(f"{prefix} changes per-site denominators across the cohort.")
        achieved = _finite_row_number(
            row.get("achieved_sparsity"), f"{prefix} achieved sparsity"
        )
        _require_close(
            achieved,
            total_site_hits / total_site_count,
            f"{prefix} achieved sparsity",
        )

        zero_counts = row.get("matmul_zero_product_count")
        product_counts = row.get("matmul_product_count")
        operation_fractions = row.get("matmul_zero_product_fraction")
        if not all(
            isinstance(value, dict)
            for value in (zero_counts, product_counts, operation_fractions)
        ):
            raise ValueError(f"{prefix} has incomplete per-operation accounting.")
        expected_operations = set(LOGICAL_MATMUL_STAGES)
        if any(
            set(value) != expected_operations
            for value in (zero_counts, product_counts, operation_fractions)
        ):
            raise ValueError(f"{prefix} has incomplete per-operation accounting.")
        block_zeros = 0
        block_products = 0
        for operation in LOGICAL_MATMUL_STAGES:
            zeros = _nonnegative_row_integer(
                zero_counts[operation], f"{prefix} {operation} zero products"
            )
            products = _positive_row_integer(
                product_counts[operation], f"{prefix} {operation} products"
            )
            if zeros > products:
                raise ValueError(f"{prefix} has too many zero products for {operation}.")
            fraction = _finite_row_number(
                operation_fractions[operation],
                f"{prefix} {operation} zero-product fraction",
            )
            _require_close(
                fraction,
                zeros / products,
                f"{prefix} {operation} zero-product fraction",
            )
            block_zeros += zeros
            block_products += products
        if reference_product_counts is None:
            reference_product_counts = product_counts
        elif product_counts != reference_product_counts:
            raise ValueError(f"{prefix} changes operation denominators across the cohort.")

        saved_block_zeros = _nonnegative_row_integer(
            row.get("block_zero_product_count"), f"{prefix} block zero products"
        )
        saved_block_products = _positive_row_integer(
            row.get("block_matmul_product_count"), f"{prefix} block products"
        )
        lm_head_products = _positive_row_integer(
            row.get("lm_head_matmul_product_count"), f"{prefix} LM-head products"
        )
        model_products = _positive_row_integer(
            row.get("model_matmul_product_count"), f"{prefix} model products"
        )
        if saved_block_zeros != block_zeros or saved_block_products != block_products:
            raise ValueError(f"{prefix} has inconsistent pooled block counts.")
        if model_products != block_products + lm_head_products:
            raise ValueError(f"{prefix} has an inconsistent model denominator.")
        if reference_lm_head_count is None:
            reference_lm_head_count = lm_head_products
        elif lm_head_products != reference_lm_head_count:
            raise ValueError(f"{prefix} changes the LM-head denominator across the cohort.")
        r_block = _finite_row_number(
            row.get("potentially_avoidable_block_matmul_fraction"),
            f"{prefix} R_block",
        )
        r_model = _finite_row_number(
            row.get("potentially_avoidable_model_matmul_fraction"),
            f"{prefix} R_model",
        )
        _require_close(r_block, block_zeros / block_products, f"{prefix} R_block")
        _require_close(r_model, block_zeros / model_products, f"{prefix} R_model")


def validate_completed_clipping_frontier_artifacts(
    *,
    run_dir: str | Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    repository: str | Path,
) -> None:
    """Validate a completed cohort against its live pinned evidence envelope."""

    root = repository_path(repository)
    run_path = Path(run_dir).resolve()
    frontier_config = config["clipping_frontier"]
    validation_config = config["validation"]
    selected_runs = frontier_config["selected_runs"]
    thresholds = [float(value) for value in frontier_config["thresholds"]]
    sites = list(resolve_site_aliases(frontier_config["sites"]))
    batch_size = int(validation_config["batch_size"])

    source_runs = [
        find_source_run(
            selected,
            section="clipping_frontier",
            repository=root,
            require_final=True,
        )
        for selected in selected_runs
    ]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    validation_metadata = (source_manifests[0].get("tokenized_data") or {}).get(
        "validation"
    )
    if not isinstance(validation_metadata, dict):
        raise ValueError("Completed clipping frontier has no source validation cache.")
    validate_shared_validation_cache(source_manifests, validation_metadata)
    _validate_frontier_validation_request(validation_config, validation_metadata)
    source_checkpoints = [
        source_checkpoint_path(
            source_run,
            source_manifest,
            repository=root,
            require_final=True,
        )
        for source_run, source_manifest in zip(
            source_runs, source_manifests, strict=True
        )
    ]
    checkpoint_identities = [
        _checkpoint_content_identity(path) for path in source_checkpoints
    ]
    validation_tokens_path = verify_token_cache(
        _cache_metadata_with_resolved_path(validation_metadata, repository=root),
        context="Completed clipping frontier validation cache",
    )
    validation_cache = _frontier_validation_cache_identity(
        validation_metadata,
        validation_tokens_path=validation_tokens_path,
        repository=root,
    )
    cache_tokens = int(validation_cache["tokens"])
    block_size = int(validation_cache["block_size"])
    validation_sequences = max(1, (cache_tokens - 1) // block_size)
    validation_tokens = validation_sequences * block_size
    validation_batches = math.ceil(validation_sequences / batch_size)

    validate_clipping_frontier_rows(
        rows,
        selected_runs=selected_runs,
        thresholds=thresholds,
        sites=sites,
        expected_validation_batches=validation_batches,
        expected_validation_sequences=validation_sequences,
        expected_validation_tokens=validation_tokens,
        validation_batch_size=batch_size,
    )

    points_per_source = len(thresholds)
    for source_index, (source_run, checkpoint_path, checkpoint_identity) in enumerate(
        zip(source_runs, source_checkpoints, checkpoint_identities, strict=True)
    ):
        expected_source_run = _portable_path(source_run, root=root)
        expected_checkpoint = _portable_path(checkpoint_path, root=root)
        for row in rows[
            source_index * points_per_source : (source_index + 1) * points_per_source
        ]:
            if row["source_run"] != expected_source_run:
                raise ValueError("Clipping frontier row source path is not pinned.")
            if row["source_checkpoint"] != expected_checkpoint:
                raise ValueError("Clipping frontier row checkpoint path is not pinned.")
            if row["source_checkpoint_content"] != checkpoint_identity:
                raise ValueError("Clipping frontier row checkpoint content changed.")
            if row["source_validation_cache"] != validation_cache:
                raise ValueError("Clipping frontier row validation cache is not pinned.")

    zero_threshold_reference = _load_zero_threshold_reference(
        frontier_config["zero_threshold_reference"],
        selected_runs=selected_runs,
        validation_metadata=validation_metadata,
        repository=root,
    )
    _validate_zero_threshold_reference(
        rows,
        selected_runs=selected_runs,
        thresholds=thresholds,
        reference=zero_threshold_reference,
        expected_validation_batches=validation_batches,
        expected_validation_sequences=validation_sequences,
        expected_validation_tokens=validation_tokens,
    )

    expected_source_runs = [_portable_path(path, root=root) for path in source_runs]
    expected_checkpoints = [
        _portable_path(path, root=root) for path in source_checkpoints
    ]
    if manifest.get("source_runs") != expected_source_runs:
        raise ValueError("Clipping frontier manifest source runs are inconsistent.")
    if manifest.get("source_checkpoints") != expected_checkpoints:
        raise ValueError("Clipping frontier manifest checkpoints are inconsistent.")
    if manifest.get("source_manifest_statuses") != [
        "completed" for _ in source_runs
    ]:
        raise ValueError("Clipping frontier manifest source statuses are inconsistent.")
    if manifest.get("source_checkpoint_contents") != checkpoint_identities:
        raise ValueError("Clipping frontier manifest checkpoint contents are inconsistent.")
    tokenized_data = manifest.get("tokenized_data")
    if not isinstance(tokenized_data, dict) or tokenized_data.get(
        "validation"
    ) != validation_metadata:
        raise ValueError("Clipping frontier manifest validation cache is inconsistent.")

    saved_frontier = manifest.get("clipping_frontier")
    if not isinstance(saved_frontier, dict):
        raise ValueError("Clipping frontier manifest has no diagnostic envelope.")
    expected_manifest_values = {
        "schema_version": CLIPPING_FRONTIER_SCHEMA_VERSION,
        "selected_runs": selected_runs,
        "mode": "threshold",
        "thresholds": thresholds,
        "sites": sites,
        "measure_zero_products": True,
        "attention_implementation": "eager",
        "eval_batches": None,
        "batch_size": batch_size,
        "validation_batches": validation_batches,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_tokens,
        "validation_cache_tokens": cache_tokens,
        "trailing_tokens_excluded": cache_tokens - validation_tokens,
        "validation_partition": validation_metadata.get("partition"),
        "validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "complete_named_partition": True,
    }
    for field, expected in expected_manifest_values.items():
        if saved_frontier.get(field) != expected:
            raise ValueError(
                f"Clipping frontier manifest field {field} is inconsistent."
            )
    expected_reference = {
        **frontier_config["zero_threshold_reference"],
        "source_run": zero_threshold_reference["source_run"],
        "source_artifact": zero_threshold_reference["source_artifact"],
        "source_artifact_sha256": zero_threshold_reference[
            "source_artifact_sha256"
        ],
        "audit_scope": ZERO_THRESHOLD_AUDIT_SCOPE,
    }
    if saved_frontier.get("zero_threshold_reference") != expected_reference:
        raise ValueError("Clipping frontier zero-threshold reference is inconsistent.")
    execution = saved_frontier.get("execution")
    execution_request = _shared_execution_request(source_runs)
    if (
        not isinstance(execution, dict)
        or execution.get("requested_device") != execution_request["device"]
        or execution.get("requested_precision") != execution_request["precision"]
        or not isinstance(execution.get("resolved_device"), str)
        or not execution["resolved_device"]
        or not isinstance(execution.get("resolved_precision"), str)
        or not execution["resolved_precision"]
    ):
        raise ValueError("Clipping frontier execution evidence is inconsistent.")
    for field in ("exact_zero_definition", "logical_opportunity_definition"):
        if not isinstance(saved_frontier.get(field), str) or not saved_frontier[
            field
        ].strip():
            raise ValueError(f"Clipping frontier manifest has no {field}.")

    expected_metric_values = {
        "clipping_frontier/sources": len(selected_runs),
        "clipping_frontier/cutoffs": len(thresholds),
        "clipping_frontier/points": len(rows),
        "clipping_frontier/validation_batches_per_point": validation_batches,
        "clipping_frontier/validation_sequences_per_point": validation_sequences,
        "clipping_frontier/validation_tokens_per_point": validation_tokens,
        "clipping_frontier/total_point_tokens": validation_tokens * len(rows),
    }
    for field, expected in expected_metric_values.items():
        if metrics.get(field) != expected:
            raise ValueError(f"Clipping frontier metric {field} is inconsistent.")
    wall_seconds = _finite_row_number(
        metrics.get("clipping_frontier/wall_seconds"),
        "Clipping frontier metric wall_seconds",
    )
    if wall_seconds <= 0.0:
        raise ValueError("Clipping frontier wall time must be positive.")
    throughput = _finite_row_number(
        metrics.get("clipping_frontier/tokens_per_second"),
        "Clipping frontier metric tokens_per_second",
    )
    _require_close(
        throughput,
        expected_metric_values["clipping_frontier/total_point_tokens"]
        / wall_seconds,
        "Clipping frontier metric tokens_per_second",
    )
    for field in (
        "clipping_frontier/peak_gpu_memory_mb",
        "clipping_frontier/peak_gpu_reserved_mb",
    ):
        value = metrics.get(field)
        if value is not None and _finite_row_number(value, field) < 0.0:
            raise ValueError(f"Clipping frontier metric {field} must be non-negative.")
    if run_path.name != manifest.get("run_id"):
        raise ValueError("Clipping frontier run identity is inconsistent.")


def _load_zero_threshold_reference(
    reference: dict[str, Any],
    *,
    selected_runs: list[dict[str, Any]],
    validation_metadata: dict[str, Any],
    repository: str | Path | None = None,
) -> dict[str, Any]:
    tranche_id = str(reference["tranche_id"])
    config_id = str(reference["config_id"])
    run_id = str(reference["run_id"])
    root = repository_path(repository)
    scaffold = resolve_experiment_scaffold(tranche_id, repository=root)
    source_run = scaffold.raw_dir / config_id / run_id
    required = (*CORE_RUN_ARTIFACTS, "activation_propagation.json")
    missing = [name for name in required if not (source_run / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Zero-threshold reference is missing required artifacts "
            f"({', '.join(missing)}): {source_run}"
        )
    manifest = read_json(source_run / "manifest.json")
    if not isinstance(manifest, dict) or any(
        manifest.get(field) != expected
        for field, expected in (
            ("tranche_id", tranche_id),
            ("config_id", config_id),
            ("run_id", run_id),
            ("mode", "activation-propagation"),
            ("status", "completed"),
            ("git_dirty", False),
            ("git_commit", str(reference["git_commit"])),
        )
    ):
        raise ValueError(
            "Zero-threshold reference must be the exact completed clean "
            "activation-propagation run."
        )
    artifact_path = source_run / "activation_propagation.json"
    artifact_sha256 = _file_sha256(artifact_path)
    if artifact_sha256 != str(reference["artifact_sha256"]):
        raise ValueError(
            "Zero-threshold reference artifact does not match its pinned SHA-256."
        )
    artifact = read_json(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 5:
        raise ValueError("Zero-threshold reference must use propagation schema version 5.")
    expected_coverage = {
        "validation_partition": validation_metadata.get("partition"),
        "validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "complete_named_partition": True,
        "attention_implementation": "eager",
    }
    for field, expected in expected_coverage.items():
        if artifact.get(field) != expected:
            raise ValueError(
                f"Zero-threshold reference has incompatible {field}."
            )
    methods = artifact.get("methods")
    if not isinstance(methods, list) or len(methods) != len(selected_runs):
        raise ValueError(
            "Zero-threshold reference does not contain the complete selected cohort."
        )
    expected_sources = [
        (str(item["config_id"]), str(item["run_id"])) for item in selected_runs
    ]
    actual_sources = [
        (str(item.get("config_id")), str(item.get("run_id")))
        if isinstance(item, dict)
        else ("", "")
        for item in methods
    ]
    if actual_sources != expected_sources:
        raise ValueError(
            "Zero-threshold reference source order or identity does not match config 020."
        )
    return {
        "source_run": _portable_path(source_run, root=root),
        "source_artifact": _portable_path(artifact_path, root=root),
        "source_artifact_sha256": artifact_sha256,
        "artifact": artifact,
    }


def _validate_zero_threshold_reference(
    rows: list[dict[str, Any]],
    *,
    selected_runs: list[dict[str, Any]],
    thresholds: list[float],
    reference: dict[str, Any],
    expected_validation_batches: int,
    expected_validation_sequences: int,
    expected_validation_tokens: int,
) -> None:
    artifact = reference["artifact"]
    for field, expected in (
        ("validation_batches", expected_validation_batches),
        ("validation_sequences", expected_validation_sequences),
        ("validation_tokens", expected_validation_tokens),
    ):
        if artifact.get(field) != expected:
            raise ValueError(
                f"Zero-threshold reference has incompatible {field}."
            )

    zero_index = thresholds.index(0.0)
    points_per_source = len(thresholds)
    for source_index, (selected, method) in enumerate(
        zip(selected_runs, artifact["methods"], strict=True)
    ):
        row = rows[source_index * points_per_source + zero_index]
        endpoint = method.get("endpoint") if isinstance(method, dict) else None
        if not isinstance(endpoint, dict):
            raise ValueError(
                f"Zero-threshold reference has no endpoint for {selected['config_id']}."
            )
        per_operation = endpoint.get("per_operation")
        if not isinstance(per_operation, dict) or set(per_operation) != set(
            LOGICAL_MATMUL_STAGES
        ):
            raise ValueError(
                f"Zero-threshold reference has incomplete operations for {selected['config_id']}."
            )
        expected_zero_counts = {
            operation: per_operation[operation].get("zero_product_count")
            for operation in LOGICAL_MATMUL_STAGES
        }
        expected_product_counts = {
            operation: per_operation[operation].get("product_count")
            for operation in LOGICAL_MATMUL_STAGES
        }
        if row["matmul_zero_product_count"] != expected_zero_counts:
            raise ValueError(
                f"Config 020 t=0 operation numerators do not match diagnostic 019 "
                f"for {selected['config_id']}."
            )
        if row["matmul_product_count"] != expected_product_counts:
            raise ValueError(
                f"Config 020 operation denominators do not match diagnostic 019 "
                f"for {selected['config_id']}."
            )
        for row_field, endpoint_field in (
            ("block_zero_product_count", "block_zero_product_count"),
            ("block_matmul_product_count", "block_product_count"),
            ("lm_head_matmul_product_count", "lm_head_product_count"),
            ("model_matmul_product_count", "model_product_count"),
        ):
            if row[row_field] != endpoint.get(endpoint_field):
                raise ValueError(
                    f"Config 020 t=0 {row_field} does not match diagnostic 019 "
                    f"for {selected['config_id']}."
                )
        zero_sites = endpoint.get("zero_sites")
        if not isinstance(zero_sites, dict):
            raise ValueError(
                f"Zero-threshold reference has no site denominators for {selected['config_id']}."
            )
        for site in row["sites"]:
            site_reference = zero_sites.get(f"z_{site}")
            if (
                not isinstance(site_reference, dict)
                or row["site_activation_count"][site] != site_reference.get("total")
            ):
                raise ValueError(
                    f"Config 020 site denominator for {site} does not match diagnostic 019 "
                    f"for {selected['config_id']}."
                )


def _validate_frontier_validation_request(
    validation_config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    for field in ("split", "max_documents", "partition", "partition_scheme", "partition_seed"):
        expected = validation_config.get(field)
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(
                f"Clipping-frontier validation {field} does not match the source cache."
            )
    for config_field, metadata_field in (
        ("partition_hash", "source_document_indices_sha256"),
        ("tokens_sha256", "tokens_sha256"),
    ):
        expected = validation_config.get(config_field)
        if expected is not None and metadata.get(metadata_field) != expected:
            raise ValueError(
                f"Clipping-frontier validation {config_field} does not match the source cache."
            )
    if validation_config.get("eval_batches") is not None:
        raise ValueError(
            "Clipping frontier must evaluate the complete named partition; "
            "set validation.eval_batches to null."
        )


def _frontier_validation_cache_identity(
    metadata: dict[str, Any],
    *,
    validation_tokens_path: Path,
    repository: str | Path,
) -> dict[str, Any]:
    fields = (
        "dtype",
        "tokens",
        "tokens_bytes",
        "tokens_sha256",
        "block_size",
        "split",
        "max_documents",
        "partition",
        "partition_scheme",
        "partition_seed",
        "source_document_indices_sha256",
    )
    return {
        "tokens_path": _portable_path(validation_tokens_path, root=repository),
        **{field: metadata.get(field) for field in fields},
    }


def _cache_metadata_with_resolved_path(
    metadata: dict[str, Any], *, repository: str | Path
) -> dict[str, Any]:
    """Resolve a saved repository-relative token path without changing evidence."""

    result = dict(metadata)
    recorded = Path(str(metadata.get("tokens_path") or ""))
    if not recorded.is_absolute():
        result["tokens_path"] = str((Path(repository).resolve() / recorded).resolve())
    return result


def _shared_execution_request(source_runs: list[Path]) -> dict[str, str]:
    requests: list[dict[str, str]] = []
    for source_run in source_runs:
        source_config = yaml.safe_load(
            (source_run / "config.yaml").read_text(encoding="utf-8")
        )
        training = source_config.get("training") if isinstance(source_config, dict) else None
        if not isinstance(training, dict):
            raise ValueError(f"Source run has no training configuration: {source_run}")
        requests.append(
            {
                "device": str(training.get("device", "auto")),
                "precision": str(training.get("precision", "auto")),
            }
        )
    reference = requests[0]
    if any(request != reference for request in requests[1:]):
        raise ValueError(
            "Selected runs do not share the same diagnostic device and precision request."
        )
    return reference


def _calibration_runtime_identity(
    torch: Any, device: Any, dtype: Any
) -> dict[str, Any]:
    runtime = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_runtime_version": str(
            getattr(getattr(torch, "version", None), "cuda", None)
        ),
        "resolved_device": str(device),
        "resolved_precision": resolved_precision(dtype),
    }
    if getattr(device, "type", None) == "cuda":
        runtime["gpu_name"] = str(torch.cuda.get_device_name(device))
        capability = torch.cuda.get_device_capability(device)
        runtime["compute_capability"] = f"{int(capability[0])}.{int(capability[1])}"
    else:
        runtime["gpu_name"] = None
        runtime["compute_capability"] = None
    return runtime


def _validate_checkpoint_identity(value: Any, *, prefix: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("files"), list) or not value["files"]:
        raise ValueError(f"{prefix} has no checkpoint-content identity.")
    paths: set[str] = set()
    for file_row in value["files"]:
        if not isinstance(file_row, dict):
            raise ValueError(f"{prefix} has an invalid checkpoint-content identity.")
        path = file_row.get("path")
        size = file_row.get("bytes")
        sha256 = file_row.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path in paths
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError(f"{prefix} has an invalid checkpoint-content identity.")
        paths.add(path)


def _validate_frontier_cache_identity(value: Any, *, prefix: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} has no validation-cache identity.")
    required_fields = (
        "tokens_path",
        "dtype",
        "tokens",
        "tokens_bytes",
        "tokens_sha256",
        "block_size",
        "split",
        "max_documents",
        "partition",
        "partition_scheme",
        "partition_seed",
        "source_document_indices_sha256",
    )
    if any(field not in value for field in required_fields):
        raise ValueError(f"{prefix} has an incomplete validation-cache identity.")
    for field in (
        "tokens_path",
        "dtype",
        "tokens_sha256",
        "split",
        "partition",
        "partition_scheme",
        "source_document_indices_sha256",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{prefix} has an incomplete validation-cache identity.")
    if Path(value["tokens_path"]).is_absolute():
        raise ValueError(f"{prefix} validation-cache path must be repository-relative.")
    if value["dtype"] != "int32":
        raise ValueError(f"{prefix} validation cache must use int32 tokens.")
    for field in ("tokens", "tokens_bytes", "block_size"):
        _positive_row_integer(value.get(field), f"{prefix} validation cache {field}")
    if value["max_documents"] is not None:
        _positive_row_integer(
            value["max_documents"], f"{prefix} validation cache max_documents"
        )
    _nonnegative_row_integer(
        value["partition_seed"], f"{prefix} validation cache partition_seed"
    )
    if value["tokens_bytes"] != value["tokens"] * 4:
        raise ValueError(f"{prefix} validation-cache byte count is inconsistent.")
    for field in ("tokens_sha256", "source_document_indices_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            raise ValueError(f"{prefix} validation-cache {field} is invalid.")


def _finite_row_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _nonnegative_row_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _positive_row_integer(value: Any, field: str) -> int:
    result = _nonnegative_row_integer(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive.")
    return result


def _require_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{field} is inconsistent with its integer counts.")
