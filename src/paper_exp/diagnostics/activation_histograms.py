from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activations import ActivationCapture, resolve_site_aliases
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
from .sources import (
    find_source_run,
    portable_path,
    source_checkpoint_path,
    validate_shared_validation_cache,
)


def run_activation_histograms(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_diagnostic_config(config, "activation_histograms")
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode="activation-histograms",
        run_id=run_id,
    ) as run:
        return _run_activation_histograms(run)


def _run_activation_histograms(run: RunHandle) -> Path:
    config = run.config
    torch, np, auto_model = _load_dependencies()
    histogram_config = config["activation_histograms"]
    validation_config = config["validation"]
    selected_runs = histogram_config["selected_runs"]
    if not isinstance(selected_runs, list) or not selected_runs:
        raise ValueError(
            "activation_histograms.selected_runs must be an explicit non-empty list."
        )
    bins = int(histogram_config["bins"])
    range_min = float(histogram_config["range_min"])
    range_max = float(histogram_config["range_max"])
    configured_thresholds = histogram_config.get("thresholds")
    if not isinstance(configured_thresholds, list) or not configured_thresholds:
        raise ValueError("activation_histograms.thresholds must be an explicit non-empty list.")
    thresholds = tuple(float(value) for value in configured_thresholds)
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("activation_histograms.thresholds must be finite and nonnegative.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("activation_histograms.thresholds must not contain duplicates.")
    sites = histogram_config.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("activation_histograms.sites must be an explicit non-empty list.")
    sites = list(resolve_site_aliases(sites))
    if bins <= 0:
        raise ValueError("activation_histograms.bins must be positive.")
    if range_min >= range_max:
        raise ValueError("activation_histograms.range_min must be less than range_max.")

    np.random.seed(int(config["run"]["seed"]))
    source_runs = [
        find_source_run(config, item, section="activation_histograms")
        for item in selected_runs
    ]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    reference_manifest = source_manifests[0]
    validation_metadata = reference_manifest["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")
    validate_shared_validation_cache(source_manifests, validation_metadata)
    _validate_requested_validation_cache(validation_config, validation_metadata)

    validation_tokens_path = verify_token_cache(
        validation_metadata,
        context="Activation histogram validation cache",
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
    bin_edges = np.linspace(range_min, range_max, bins + 1).tolist()
    results: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for selected, source_run in zip(selected_runs, source_runs, strict=True):
        print(f"Measuring activation histograms for {selected['label']} from {source_run}", flush=True)
        result = _measure_one_run(
            label=selected["label"],
            source_run=source_run,
            auto_model=auto_model,
            torch=torch,
            np=np,
            validation_tokens=validation_tokens,
            block_size=block_size,
            batch_size=batch_size,
            starts=starts,
            device=device,
            dtype=dtype,
            sites=sites,
            bins=bins,
            range_min=range_min,
            range_max=range_max,
            thresholds=thresholds,
        )
        results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - start_time
    total_sequences = len(starts)
    total_tokens = total_sequences * block_size
    metrics = {
        "activation_histograms/methods": len(results),
        "activation_histograms/layers": len(results[0]["layers"]) if results else 0,
        "activation_histograms/bins": bins,
        "activation_histograms/range_min": range_min,
        "activation_histograms/range_max": range_max,
        "activation_histograms/validation_sequences": total_sequences,
        "activation_histograms/validation_tokens": total_tokens,
        "activation_histograms/wall_seconds": wall_seconds,
        "activation_histograms/tokens_per_second": (total_tokens * len(results)) / wall_seconds if wall_seconds > 0 else None,
        "activation_histograms/peak_gpu_memory_mb": peak_gpu_memory_mb(torch, device),
        "activation_histograms/peak_gpu_reserved_mb": peak_gpu_reserved_mb(torch, device),
    }
    manifest_updates = {
        "source_runs": [portable_path(path) for path in source_runs],
        "source_checkpoints": [
            portable_path(
                source_checkpoint_path(path, read_json(path / "manifest.json"))
            )
            for path in source_runs
        ],
        "tokenized_data": {"validation": validation_metadata},
        "activation_histograms": {
            "sites": sites,
            "bins": bins,
            "range_min": range_min,
            "range_max": range_max,
            "selected_runs": selected_runs,
            "thresholds": list(thresholds),
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "validation_sequences": total_sequences,
            "validation_tokens": total_tokens,
            "execution": execution,
        },
    }

    payload = {
        "schema_version": 3,
        "plot_title": histogram_config.get("plot_title"),
        "bin_edges": bin_edges,
        "range_min": range_min,
        "range_max": range_max,
        "bins": bins,
        "sites": sites,
        "thresholds": list(thresholds),
        "validation_sequences": total_sequences,
        "validation_tokens": total_tokens,
        "execution": execution,
        "methods": results,
    }
    write_json(output_dir / "activation_histograms.json", payload)
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
    auto_model: Any,
    torch: Any,
    np: Any,
    validation_tokens: Any,
    block_size: int,
    batch_size: int,
    starts: list[int],
    device: Any,
    dtype: Any,
    sites: list[str],
    bins: int,
    range_min: float,
    range_max: float,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    source_manifest = read_json(source_run / "manifest.json")
    checkpoint_path = source_checkpoint_path(source_run, source_manifest)
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    layer_counts: dict[str, Any] = {}
    underflow: dict[str, int] = {}
    overflow: dict[str, int] = {}
    nonfinite: dict[str, int] = {}
    totals: dict[str, int] = {}
    finite_totals: dict[str, int] = {}
    threshold_hits: dict[str, dict[str, int]] = {}
    sums: dict[str, float] = {}
    square_sums: dict[str, float] = {}
    absolute_sums: dict[str, float] = {}
    nonzero_absolute_sums: dict[str, float] = {}
    nonzero_totals: dict[str, int] = {}
    positive_totals: dict[str, int] = {}
    negative_totals: dict[str, int] = {}
    batches = 0
    method_start = time.perf_counter()

    with ActivationCapture(model, sites, torch=torch) as capture:
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                capture.clear()
                batch_starts = starts[offset : offset + batch_size]
                batch = np.stack([validation_tokens[start : start + block_size] for start in batch_starts])
                input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                with autocast_context(torch, device, dtype):
                    model(input_ids=input_ids)
                for name, value in capture.activations.items():
                    flat = value.detach().float().reshape(-1)
                    finite = torch.isfinite(flat)
                    finite_values = flat[finite]
                    if name not in layer_counts:
                        layer_counts[name] = torch.zeros(bins, dtype=torch.float64)
                        underflow[name] = 0
                        overflow[name] = 0
                        nonfinite[name] = 0
                        totals[name] = 0
                        finite_totals[name] = 0
                        threshold_hits[name] = {_threshold_key(threshold): 0 for threshold in thresholds}
                        sums[name] = 0.0
                        square_sums[name] = 0.0
                        absolute_sums[name] = 0.0
                        nonzero_absolute_sums[name] = 0.0
                        nonzero_totals[name] = 0
                        positive_totals[name] = 0
                        negative_totals[name] = 0
                    if finite_values.numel():
                        counts = torch.histc(finite_values, bins=bins, min=range_min, max=range_max).cpu().double()
                        layer_counts[name] += counts
                        underflow[name] += int((finite_values < range_min).sum().detach().cpu())
                        overflow[name] += int((finite_values > range_max).sum().detach().cpu())
                        absolute_values = finite_values.abs()
                        nonzero = finite_values != 0
                        finite_totals[name] += int(finite_values.numel())
                        sums[name] += float(finite_values.sum().detach().cpu())
                        square_sums[name] += float(finite_values.square().sum().detach().cpu())
                        absolute_sums[name] += float(absolute_values.sum().detach().cpu())
                        nonzero_absolute_sums[name] += float(absolute_values[nonzero].sum().detach().cpu())
                        nonzero_totals[name] += int(nonzero.sum().detach().cpu())
                        positive_totals[name] += int((finite_values > 0).sum().detach().cpu())
                        negative_totals[name] += int((finite_values < 0).sum().detach().cpu())
                        for threshold in thresholds:
                            threshold_hits[name][_threshold_key(threshold)] += int(
                                (absolute_values <= threshold).sum().detach().cpu()
                            )
                    nonfinite[name] += int((~finite).sum().detach().cpu())
                    totals[name] += int(flat.numel())
                batches += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    layers = []
    for name in sorted(layer_counts, key=_layer_sort_key):
        total = totals[name]
        in_range = int(layer_counts[name].sum().item())
        finite_total = finite_totals[name]
        nonzero_total = nonzero_totals[name]
        layers.append(
            {
                "name": name,
                "counts": [int(value) for value in layer_counts[name].tolist()],
                "total": total,
                "in_range": in_range,
                "underflow": underflow[name],
                "overflow": overflow[name],
                "nonfinite": nonfinite[name],
                "finite": finite_total,
                "underflow_fraction": underflow[name] / total if total else None,
                "overflow_fraction": overflow[name] / total if total else None,
                "threshold_hits": threshold_hits[name],
                "threshold_fractions": {
                    key: value / total if total else None
                    for key, value in threshold_hits[name].items()
                },
                "mean": sums[name] / finite_total if finite_total else None,
                "rms": (square_sums[name] / finite_total) ** 0.5 if finite_total else None,
                "mean_abs": absolute_sums[name] / finite_total if finite_total else None,
                "nonzero_mean_abs": (
                    nonzero_absolute_sums[name] / nonzero_total if nonzero_total else None
                ),
                "positive_fraction": positive_totals[name] / total if total else None,
                "negative_fraction": negative_totals[name] / total if total else None,
            }
        )

    return {
        "label": label,
        "config_id": source_manifest["config_id"],
        "run_id": source_manifest["run_id"],
        "source_run": portable_path(source_run),
        "source_checkpoint": portable_path(checkpoint_path),
        "batches": batches,
        "wall_seconds": time.perf_counter() - method_start,
        "layers": layers,
    }


def _validate_requested_validation_cache(
    validation_config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    requested_partition = validation_config.get("partition")
    realized_partition = metadata.get("partition")
    if requested_partition is None:
        if realized_partition not in {None, "full"}:
            raise ValueError(
                "Activation histogram validation partition does not match the source cache."
            )
    elif realized_partition != requested_partition:
        raise ValueError(
            "Activation histogram validation partition does not match the source cache."
        )

    for field in ("split", "max_documents"):
        expected = validation_config.get(field)
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(
                f"Activation histogram validation {field} does not match the source cache."
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
                f"Activation histogram validation {field} does not match the source cache."
            )
    requested_hash = validation_config.get("partition_hash")
    if (
        requested_hash is not None
        and metadata.get("source_document_indices_sha256") != requested_hash
    ):
        raise ValueError(
            "Activation histogram validation partition hash does not match the source cache."
        )
    if validation_config.get("eval_batches") is not None:
        raise ValueError(
            "Named-partition activation histograms must evaluate the complete partition; "
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
        request = {
            "device": str(training.get("device", "auto")),
            "precision": str(training.get("precision", "auto")),
        }
        requests.append(request)
    reference = requests[0]
    if any(request != reference for request in requests[1:]):
        raise ValueError(
            "Selected runs do not share the same diagnostic device and precision request."
        )
    return reference


def _layer_sort_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 9999


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Activation histogram analysis requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoModelForCausalLM
