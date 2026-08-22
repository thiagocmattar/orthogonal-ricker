from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.config import validate_diagnostic_config
from paper_exp.data import verify_token_cache
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_json

from .evaluation import (
    autocast_context,
    eval_starts,
    peak_gpu_memory_mb,
    peak_gpu_reserved_mb,
    resolved_precision,
    select_device,
    select_dtype,
)
from . import propagation_summary as _summary
from .propagation_capture import _PropagationAccumulator, _capture_model_propagation
from .sources import (
    find_source_run,
    portable_path,
    source_checkpoint_path,
    validate_shared_validation_cache,
)


def run_activation_propagation(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_diagnostic_config(config, "activation_propagation")
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode="activation-propagation",
        run_id=run_id,
    ) as run:
        return _run_activation_propagation(run)


def _run_activation_propagation(run: RunHandle) -> Path:
    config = run.config
    torch, np, auto_model, modeling_gpt_neox = _load_dependencies()
    propagation_config = config["activation_propagation"]
    validation_config = config["validation"]
    selected_runs = propagation_config["selected_runs"]

    np.random.seed(int(config["run"]["seed"]))
    source_runs = [
        find_source_run(item, section="activation_propagation")
        for item in selected_runs
    ]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    validation_metadata = next(
        (
            (manifest.get("tokenized_data") or {}).get("validation")
            for manifest in source_manifests
            if (manifest.get("tokenized_data") or {}).get("validation") is not None
        ),
        None,
    )
    if validation_metadata is None:
        raise ValueError("Selected source runs have no validation token cache in their manifests.")
    validate_shared_validation_cache(source_manifests, validation_metadata)
    _validate_requested_validation_cache(validation_config, validation_metadata)
    source_checkpoints = [
        source_checkpoint_path(source_run, manifest)
        for source_run, manifest in zip(source_runs, source_manifests, strict=True)
    ]

    validation_tokens_path = verify_token_cache(
        validation_metadata,
        context="Activation propagation validation cache",
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

    output_dir = run.run_dir
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for selected, source_run, source_manifest, source_checkpoint in zip(
        selected_runs,
        source_runs,
        source_manifests,
        source_checkpoints,
        strict=True,
    ):
        print(f"Measuring activation propagation for {selected['label']} from {source_run}", flush=True)
        results.append(
            _measure_one_run(
                label=str(selected["label"]),
                source_run=source_run,
                source_manifest=source_manifest,
                checkpoint_path=source_checkpoint,
                auto_model=auto_model,
                modeling_gpt_neox=modeling_gpt_neox,
                torch=torch,
                np=np,
                validation_tokens=validation_tokens,
                block_size=block_size,
                batch_size=batch_size,
                starts=starts,
                device=device,
                dtype=dtype,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - started
    validation_sequences = len(starts)
    validation_token_count = validation_sequences * block_size
    validation_cache_tokens = int(len(validation_tokens))
    trailing_tokens_excluded = validation_cache_tokens - validation_token_count
    metrics = {
        "activation_propagation/methods": len(results),
        "activation_propagation/layers": results[0]["num_layers"] if results else 0,
        "activation_propagation/activation_stages": len(_summary.ACTIVATION_STAGE_ORDER),
        "activation_propagation/matmul_stages": len(_summary.MATMUL_STAGE_ORDER),
        "activation_propagation/validation_batches": (
            results[0]["batches"] if results else 0
        ),
        "activation_propagation/validation_sequences": validation_sequences,
        "activation_propagation/validation_tokens": validation_token_count,
        "activation_propagation/validation_cache_tokens": validation_cache_tokens,
        "activation_propagation/trailing_tokens_excluded": trailing_tokens_excluded,
        "activation_propagation/validation_partition": validation_metadata.get(
            "partition"
        ),
        "activation_propagation/validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "activation_propagation/wall_seconds": wall_seconds,
        "activation_propagation/tokens_per_second": (
            validation_token_count * len(results) / wall_seconds if wall_seconds > 0 else None
        ),
        "activation_propagation/peak_gpu_memory_mb": peak_gpu_memory_mb(torch, device),
        "activation_propagation/peak_gpu_reserved_mb": peak_gpu_reserved_mb(torch, device),
    }
    for result in results:
        endpoint = result["endpoint"]
        prefix = f"activation_propagation/endpoint/{result['config_id']}"
        metrics.update(
            {
                f"{prefix}/R_block": endpoint["R_block"],
                f"{prefix}/R_model": endpoint["R_model"],
            }
        )
    if len(results) == 1:
        endpoint = results[0]["endpoint"]
        metrics.update(
            {
                "activation_propagation/R_block": endpoint["R_block"],
                "activation_propagation/R_model": endpoint["R_model"],
            }
        )

    exact_zero_definition = (
        "An activation element is an exact zero iff its computed tensor value compares equal to numeric 0 "
        "with no tolerance. Counts are integer sums pooled over every evaluated validation sequence, token, "
        "and feature/head element for each layer; attention-probability counts instead pool valid causal "
        "query-key entries. Percentages are not averages of batch percentages."
    )
    matmul_definition = (
        "A zero product opportunity is one scalar multiplication whose activation-side operand, or either "
        "activation operand for QK and PV, is exactly zero. Bias additions, reductions, and realized kernel "
        "speedups are excluded. QK and PV totals include only key positions at or before each query position."
    )
    compute_endpoint_definition = (
        "R_block is the pooled direct zero-product count across QKV, valid-causal QK, valid-causal PV, Wo, "
        "W1, and W2 divided by all products in those block operations. R_model keeps that numerator and "
        "adds the dense hidden-to-vocabulary LM head to the denominator. These are direct logical scalar "
        "product counters, not architecture ceilings or measured kernel speedups."
    )
    rope_survival_definition = (
        "For PRE-RoPE Q/K gates only, compare each exact-zero gate-output coordinate with the "
        "corresponding actual QK operand after RoPE. Preservation is input-zero/output-zero; "
        "repopulation is input-zero/output-nonzero; creation is input-nonzero/output-zero. "
        "Preservation and repopulation fractions condition on input zeros; creation conditions "
        "on input nonzeros. "
        "Rotary and pass-through coordinates are reported separately. All-zero rotary pairs use "
        "the paired first-half/second-half coordinates mixed by GPT-NeoX rotate_half. POST-RoPE "
        "and absent-gate checkpoints report these placement-specific metrics as unavailable."
    )
    manifest_updates = {
        "source_runs": [portable_path(path) for path in source_runs],
        "source_checkpoints": [portable_path(path) for path in source_checkpoints],
        "source_manifest_statuses": [
            source_manifest.get("status") for source_manifest in source_manifests
        ],
        "tokenized_data": {"validation": validation_metadata},
        "activation_propagation": {
            "selected_runs": selected_runs,
            "attention_implementation": "eager",
            "future_causal_positions_excluded": True,
            "exact_zero_definition": exact_zero_definition,
            "matmul_zero_product_definition": matmul_definition,
            "compute_endpoint_definition": compute_endpoint_definition,
            "rope_zero_survival_definition": rope_survival_definition,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "validation_sequences": validation_sequences,
            "validation_tokens": validation_token_count,
            "validation_cache_tokens": validation_cache_tokens,
            "trailing_tokens_excluded": trailing_tokens_excluded,
            "validation_partition": validation_metadata.get("partition"),
            "validation_partition_hash": validation_metadata.get(
                "source_document_indices_sha256"
            ),
            "complete_named_partition": bool(validation_metadata.get("partition"))
            and eval_batches is None,
            "execution": execution,
        },
    }

    payload = {
        "schema_version": 5,
        "validation_batches": results[0]["batches"] if results else 0,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_token_count,
        "validation_cache_tokens": validation_cache_tokens,
        "trailing_tokens_excluded": trailing_tokens_excluded,
        "validation_partition": validation_metadata.get("partition"),
        "validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "complete_named_partition": bool(validation_metadata.get("partition"))
        and eval_batches is None,
        "execution": execution,
        "block_size": block_size,
        "batch_size": batch_size,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "exact_zero_definition": exact_zero_definition,
        "matmul_zero_product_definition": matmul_definition,
        "compute_endpoint_definition": compute_endpoint_definition,
        "rope_zero_survival_definition": rope_survival_definition,
        "activation_stage_order": _summary.ACTIVATION_STAGE_ORDER,
        "activation_stage_labels": _summary.ACTIVATION_STAGE_LABELS,
        "matmul_stage_order": _summary.MATMUL_STAGE_ORDER,
        "matmul_stage_labels": _summary.MATMUL_STAGE_LABELS,
        "methods": results,
    }
    write_json(output_dir / "activation_propagation.json", payload)
    return complete_run(
        run,
        metrics=metrics,
        predictions=[],
        manifest_updates=manifest_updates,
    )


def _measure_one_run(
    *,
    label: str,
    source_run: Path,
    source_manifest: dict[str, Any],
    checkpoint_path: Path,
    auto_model: Any,
    modeling_gpt_neox: Any,
    torch: Any,
    np: Any,
    validation_tokens: Any,
    block_size: int,
    batch_size: int,
    starts: list[int],
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    layers = list(model.gpt_neox.layers)
    if not layers:
        raise ValueError("Activation propagation requires at least one GPT-NeoX layer.")
    if not bool(getattr(model.config, "use_parallel_residual", False)):
        raise ValueError("This diagnostic currently describes the Pythia parallel-residual block only.")

    attention_gates = _summary._attention_gate_metadata(layers)
    accumulator = _PropagationAccumulator(torch)
    batches = 0
    method_started = time.perf_counter()
    with _capture_model_propagation(
        model,
        accumulator=accumulator,
        modeling_gpt_neox=modeling_gpt_neox,
        torch=torch,
    ):
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                batch = np.stack([validation_tokens[start : start + block_size] for start in batch_starts])
                input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                with autocast_context(torch, device, dtype):
                    model.gpt_neox(input_ids=input_ids, use_cache=False)
                batches += 1

    activation_rows = accumulator.rows(
        "activations", _summary.ACTIVATION_STAGE_ORDER, num_layers=len(layers)
    )
    matmul_rows = accumulator.rows(
        "matmuls", _summary.MATMUL_STAGE_ORDER, num_layers=len(layers)
    )
    architecture = _summary._architecture_metadata(
        model,
        layers=layers,
        attention_gates=attention_gates,
        block_size=block_size,
        torch=torch,
    )
    endpoint = _summary._endpoint_summary(
        architecture=architecture,
        activation_rows=activation_rows,
        matmul_rows=matmul_rows,
        validation_tokens=len(starts) * block_size,
    )

    method_result = {
        "label": label,
        "config_id": source_manifest["config_id"],
        "run_id": source_manifest["run_id"],
        "source_run": portable_path(source_run),
        "source_checkpoint": portable_path(checkpoint_path),
        "source_manifest_status": source_manifest.get("status"),
        "num_layers": len(layers),
        "use_parallel_residual": True,
        "attention_gates": attention_gates,
        "architecture": architecture,
        "endpoint": endpoint,
        "batches": batches,
        "wall_seconds": time.perf_counter() - method_started,
        "activations": activation_rows,
        "matmuls": matmul_rows,
        "rope_zero_survival": accumulator.rope_survival_rows(num_layers=len(layers)),
        "rope_all_zero_pairs": accumulator.rope_pair_rows(num_layers=len(layers)),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return method_result


def _validate_requested_validation_cache(
    validation_config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Bind diagnostics to the exact requested validation cache."""

    requested_partition = validation_config.get("partition")
    realized_partition = metadata.get("partition")
    if requested_partition is None:
        if realized_partition not in {None, "full"}:
            raise ValueError(
                "Activation propagation validation partition does not match the source cache."
            )
    elif realized_partition != requested_partition:
        raise ValueError(
            "Activation propagation validation partition does not match the source cache."
        )
    for field in ("split", "max_documents"):
        expected = validation_config.get(field)
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(
                f"Activation propagation validation {field} does not match the source cache."
            )
    if requested_partition not in {"selection", "confirmation"}:
        return
    expected_fields = {
        "partition_scheme": validation_config.get("partition_scheme"),
        "partition_seed": validation_config.get("partition_seed"),
    }
    for field, expected in expected_fields.items():
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(
                f"Activation propagation validation {field} does not match the source cache."
            )
    requested_hash = validation_config.get("partition_hash")
    realized_hash = metadata.get("source_document_indices_sha256")
    if requested_hash is not None and realized_hash != requested_hash:
        raise ValueError(
            "Activation propagation validation partition hash does not match the source cache."
        )
    if validation_config.get("eval_batches") is not None:
        raise ValueError(
            "Named-partition activation propagation must evaluate the complete partition; "
            "set validation.eval_batches to null."
        )


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


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.gpt_neox import modeling_gpt_neox
    except ImportError as exc:
        raise RuntimeError(
            "Activation propagation analysis requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoModelForCausalLM, modeling_gpt_neox
