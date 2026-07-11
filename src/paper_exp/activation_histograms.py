from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activations import ActivationCapture
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_json


def run_activation_histograms(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    torch, np, auto_model = _load_dependencies()
    histogram_config = config["activation_histograms"]
    validation_config = config["validation"]
    selected_runs = histogram_config["selected_runs"]
    bins = int(histogram_config["bins"])
    range_min = float(histogram_config["range_min"])
    range_max = float(histogram_config["range_max"])
    thresholds = tuple(float(value) for value in histogram_config.get("thresholds", [0.0, 0.01]))
    if bins <= 0:
        raise ValueError("activation_histograms.bins must be positive.")
    if range_min >= range_max:
        raise ValueError("activation_histograms.range_min must be less than range_max.")

    np.random.seed(int(config["run"]["seed"]))
    source_runs = [_find_latest_source_run(config, item["config_id"]) for item in selected_runs]
    reference_manifest = read_json(source_runs[0] / "manifest.json")
    validation_metadata = reference_manifest["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")

    validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(validation_config["batch_size"])
    eval_batches = validation_config.get("eval_batches")
    starts = _eval_starts(validation_tokens, block_size, eval_batches=eval_batches, batch_size=batch_size, np=np)

    training = yaml.safe_load((source_runs[0] / "config.yaml").read_text(encoding="utf-8"))["training"]
    device = _select_device(torch, training.get("device", "auto"))
    dtype = _select_dtype(torch, device, training.get("precision", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    experiment_id, numbered_run_id, output_dir = create_run_dir(config, config_path, run_id=run_id)
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
            sites=histogram_config.get("sites", ["mlp_hiddens"]),
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
        "activation_histograms/peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "activation_histograms/peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
    }
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="activation-histograms",
        config_id=experiment_id,
        result_path=output_dir,
    )
    manifest["source_runs"] = [str(path) for path in source_runs]
    manifest["source_checkpoints"] = [str(Path(read_json(path / "manifest.json")["checkpoint"]["path"])) for path in source_runs]
    manifest["tokenized_data"] = {"validation": validation_metadata}
    manifest["activation_histograms"] = {
        "sites": histogram_config.get("sites", ["mlp_hiddens"]),
        "bins": bins,
        "range_min": range_min,
        "range_max": range_max,
        "selected_runs": selected_runs,
        "thresholds": list(thresholds),
        "eval_batches": eval_batches,
        "batch_size": batch_size,
        "validation_sequences": total_sequences,
        "validation_tokens": total_tokens,
    }

    payload = {
        "schema_version": 2,
        "plot_title": histogram_config.get("plot_title"),
        "bin_edges": bin_edges,
        "range_min": range_min,
        "range_max": range_max,
        "bins": bins,
        "sites": histogram_config.get("sites", ["mlp_hiddens"]),
        "thresholds": list(thresholds),
        "validation_sequences": total_sequences,
        "validation_tokens": total_tokens,
        "methods": results,
    }
    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=[])
    write_json(output_dir / "activation_histograms.json", payload)
    return output_dir


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
    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
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
                with _autocast_context(torch, device, dtype):
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
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "batches": batches,
        "wall_seconds": time.perf_counter() - method_start,
        "layers": layers,
    }


def _find_latest_source_run(config: dict[str, Any], config_id: str) -> Path:
    experiment_dir = Path(config["output"]["dir"]) / config_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing result directory for selected config: {experiment_dir}")
    candidates = []
    for run_dir in sorted(experiment_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        checkpoint_path = run_dir / "checkpoints" / "final" / "model.safetensors"
        if manifest_path.exists() and checkpoint_path.exists():
            manifest = read_json(manifest_path)
            candidates.append((int(manifest.get("run_sequence", 0)), run_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpointed runs found for selected config: {experiment_dir}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _eval_starts(tokens: Any, block_size: int, *, eval_batches: int | None, batch_size: int, np: Any) -> list[int]:
    if eval_batches is None:
        total_blocks = max(1, (len(tokens) - 1) // block_size)
        return [index * block_size for index in range(total_blocks)]
    max_start = len(tokens) - block_size - 1
    return list(np.random.randint(0, max_start, size=int(eval_batches) * batch_size))


def _layer_sort_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 9999


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def _autocast_context(torch: Any, device: Any, dtype: Any) -> Any:
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


def _select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype(torch: Any, device: Any, requested: str) -> Any:
    if requested == "float32" or device.type != "cuda":
        return None
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown precision: {requested}")


def _peak_gpu_memory_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _peak_gpu_reserved_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024 * 1024)


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
