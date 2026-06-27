from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activation_pressure import activation_near_zero_metrics
from paper_exp.activations import ActivationCapture
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_jsonl


def run_clipping_sweep(
    *,
    checkpoint_run_dir: str | Path,
    command: str,
    thresholds: list[float],
    quantiles: list[float],
    eval_batches: int | None,
    run_id: str | None = None,
) -> Path:
    run_path = Path(checkpoint_run_dir)
    source_config_path = run_path / "config.yaml"
    source_manifest_path = run_path / "manifest.json"
    if not source_config_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint run must contain config.yaml and manifest.json: {run_path}")

    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    source_manifest = read_json(source_manifest_path)
    config["experiment_name"] = f"{config['experiment_name']}_clipping_sweep"

    torch, np, auto_model = _load_clipping_dependencies()
    training = config["training"]
    device = _select_device(torch, training.get("device", "auto"))
    dtype = _select_dtype(torch, device, training.get("precision", "auto"))

    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
    model = auto_model.from_pretrained(checkpoint_path)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    validation_metadata = source_manifest["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")
    validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(config["validation"]["batch_size"])
    starts = _eval_starts(validation_tokens, block_size, eval_batches=eval_batches, batch_size=batch_size, np=np)

    sweep_config_path = f"{source_manifest['config_id']}-clipping-sweep.yaml"
    experiment_id, numbered_run_id, output_dir = create_run_dir(config, sweep_config_path, run_id=run_id)
    rows: list[dict[str, Any]] = []
    for clipping_cfg in _clipping_configs(config, thresholds=thresholds, quantiles=quantiles):
        result = _evaluate_clipped_loss(
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
            clipping_cfg=clipping_cfg,
        )
        rows.append(result)

    best_loss = min(rows, key=lambda row: row["validation_loss"]) if rows else None
    metrics = {
        "clipping/num_points": len(rows),
        "clipping/best_validation_loss": best_loss["validation_loss"] if best_loss else None,
        "clipping/best_achieved_sparsity": best_loss["achieved_sparsity"] if best_loss else None,
    }
    manifest = build_manifest(
        config=config,
        config_path=sweep_config_path,
        run_id=numbered_run_id,
        command=command,
        mode="clip-sweep",
        config_id=experiment_id,
        result_path=output_dir,
    )
    manifest["source_run"] = str(run_path)
    manifest["source_checkpoint"] = str(checkpoint_path)
    manifest["thresholds"] = thresholds
    manifest["quantiles"] = quantiles
    manifest["eval_batches"] = eval_batches

    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=rows)
    write_jsonl(output_dir / "clipping_frontier.jsonl", rows)
    return output_dir


def _clipping_configs(config: dict[str, Any], *, thresholds: list[float], quantiles: list[float]) -> list[dict[str, Any]]:
    base = config.get("activation_clipping", {})
    sites = base.get("sites", ["mlp_hiddens"])
    configs = []
    for threshold in thresholds:
        configs.append({"enabled": True, "mode": "threshold", "sites": sites, "threshold": threshold})
    for quantile in quantiles:
        configs.append({"enabled": True, "mode": "quantile", "sites": sites, "quantile": quantile})
    return configs


def _evaluate_clipped_loss(
    *,
    model: Any,
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    eval_batches: int | None,
    starts: list[int],
    device: Any,
    dtype: Any,
    clipping_cfg: dict[str, Any],
) -> dict[str, Any]:
    losses: list[float] = []
    batches = 0
    total_sequences = 0
    total_tokens = 0
    zero_hits = 0.0
    zero_count = 0.0
    start_time = time.perf_counter()

    with ActivationCapture(model, clipping_cfg.get("sites", ["mlp_hiddens"]), torch=torch, clipping=clipping_cfg) as capture:
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                capture.clear()
                batch_starts = starts[offset : offset + batch_size]
                batch = np.stack([tokens[start : start + block_size] for start in batch_starts])
                input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                with _autocast_context(torch, device, dtype):
                    output = model(input_ids=input_ids, labels=input_ids)
                if not bool(torch.isfinite(output.loss.detach()).item()):
                    raise RuntimeError("Non-finite clipped validation loss.")
                losses.append(float(output.loss.detach().cpu()) * len(batch_starts))
                near_zero = activation_near_zero_metrics(capture.activations, (0.0,))
                zero_hits += near_zero.get("activation/near_zero_mass/k0", 0.0) * _activation_count(capture.activations)
                zero_count += _activation_count(capture.activations)
                total_sequences += len(batch_starts)
                total_tokens += len(batch_starts) * block_size
                batches += 1

    wall_seconds = time.perf_counter() - start_time
    return {
        "event": "clipping_sweep",
        "mode": clipping_cfg["mode"],
        "threshold": clipping_cfg.get("threshold"),
        "quantile": clipping_cfg.get("quantile"),
        "sites": clipping_cfg.get("sites", ["mlp_hiddens"]),
        "validation_loss": sum(losses) / total_sequences,
        "achieved_sparsity": zero_hits / zero_count if zero_count else None,
        "validation_batches": batches,
        "validation_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": total_tokens / wall_seconds if wall_seconds > 0 else None,
    }


def _eval_starts(tokens: Any, block_size: int, *, eval_batches: int | None, batch_size: int, np: Any) -> list[int]:
    if eval_batches is None:
        total_blocks = max(1, (len(tokens) - 1) // block_size)
        return [index * block_size for index in range(total_blocks)]
    max_start = len(tokens) - block_size - 1
    return list(np.random.randint(0, max_start, size=int(eval_batches) * batch_size))


def _activation_count(activations: dict[str, Any]) -> int:
    return sum(value.numel() for value in activations.values())


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


def _load_clipping_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Clipping sweep requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoModelForCausalLM
