from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from paper_exp.activations import ActivationCapture
from paper_exp.activations import activation_exact_zero_counts
from paper_exp.activations import activation_exact_zero_counts_by_alias
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_jsonl


def run_clipping_sweep(
    *,
    checkpoint_run_dir: str | Path,
    command: str,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float] | None = None,
    sites: list[str] | None = None,
    experiment_suffix: str | None = None,
    eval_batches: int | None,
    seed: int = 0,
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
    if sites:
        config.setdefault("activation_clipping", {})["sites"] = sites

    torch, np, auto_model = _load_clipping_dependencies()
    np.random.seed(seed)
    training = config["training"]
    device = _select_device(torch, training.get("device", "auto"))
    dtype = _select_dtype(torch, device, training.get("precision", "auto"))

    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    validation_metadata = source_manifest["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")
    validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(config["validation"]["batch_size"])
    starts = _eval_starts(validation_tokens, block_size, eval_batches=eval_batches, batch_size=batch_size, np=np)

    suffix = experiment_suffix or (_site_suffix(sites) if sites else None)
    suffix_part = f"-{suffix}" if suffix else ""
    sweep_config_path = f"{source_manifest['config_id']}-clipping-sweep{suffix_part}.yaml"
    experiment_id, numbered_run_id, output_dir = create_run_dir(config, sweep_config_path, run_id=run_id)
    rows: list[dict[str, Any]] = []
    rms_multipliers = rms_multipliers or []
    for clipping_cfg in _clipping_configs(
        config,
        thresholds=thresholds,
        quantiles=quantiles,
        rms_multipliers=rms_multipliers,
    ):
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
    manifest["rms_multipliers"] = rms_multipliers
    manifest["clipping_sites"] = _clipping_sites(config)
    if suffix:
        manifest["clipping_sweep_suffix"] = suffix
    if rms_multipliers:
        manifest["rms_threshold_semantics"] = (
            "For each captured activation tensor and forward pass, clip entries with "
            "|a| <= rms_multiplier * RMS(A), where RMS(A) is computed over that tensor."
        )
    manifest["eval_batches"] = eval_batches
    manifest["seed"] = seed
    if "sweep" in source_manifest:
        manifest["source_sweep"] = source_manifest["sweep"]

    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=rows)
    write_jsonl(output_dir / "clipping_frontier.jsonl", rows)
    return output_dir


def _clipping_configs(
    config: dict[str, Any],
    *,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float],
) -> list[dict[str, Any]]:
    base = config.get("activation_clipping", {})
    sites = base.get("sites", ["mlp_hiddens"])
    configs = []
    for threshold in thresholds:
        configs.append({"enabled": True, "mode": "threshold", "sites": sites, "threshold": threshold})
    for quantile in quantiles:
        configs.append({"enabled": True, "mode": "quantile", "sites": sites, "quantile": quantile})
    for multiplier in rms_multipliers:
        configs.append(
            {
                "enabled": True,
                "mode": "rms_threshold",
                "sites": sites,
                "rms_multiplier": multiplier,
            }
        )
    return configs


def _clipping_sites(config: dict[str, Any]) -> list[str]:
    return list(config.get("activation_clipping", {}).get("sites", ["mlp_hiddens"]))


def _site_suffix(sites: list[str] | None) -> str | None:
    if not sites:
        return None
    if set(sites) == {"mlp_hiddens", "attention_outputs", "residual_streams"}:
        return "all-sites"
    return "sites-" + "-".join(site.replace("_", "-") for site in sites)


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
    zero_hits = 0
    zero_count = 0
    site_zero_hits: dict[str, int] = {}
    site_zero_counts: dict[str, int] = {}
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
                batch_zero_hits, batch_activation_count = activation_exact_zero_counts(capture.activations)
                zero_hits += batch_zero_hits
                zero_count += batch_activation_count
                for alias, (alias_hits, alias_count) in activation_exact_zero_counts_by_alias(
                    capture.activations
                ).items():
                    site_zero_hits[alias] = site_zero_hits.get(alias, 0) + alias_hits
                    site_zero_counts[alias] = site_zero_counts.get(alias, 0) + alias_count
                total_sequences += len(batch_starts)
                total_tokens += len(batch_starts) * block_size
                batches += 1

    wall_seconds = time.perf_counter() - start_time
    return {
        "event": "clipping_sweep",
        "mode": clipping_cfg["mode"],
        "threshold": clipping_cfg.get("threshold"),
        "quantile": clipping_cfg.get("quantile"),
        "rms_multiplier": clipping_cfg.get("rms_multiplier"),
        "rms_scope": (
            "per captured activation tensor per forward pass"
            if clipping_cfg["mode"] == "rms_threshold"
            else None
        ),
        "sites": clipping_cfg.get("sites", ["mlp_hiddens"]),
        "site_achieved_sparsity": {
            alias: site_zero_hits[alias] / site_zero_counts[alias]
            for alias in sorted(site_zero_counts)
            if site_zero_counts[alias]
        },
        "site_zero_hits": {alias: site_zero_hits[alias] for alias in sorted(site_zero_hits)},
        "site_activation_count": {alias: site_zero_counts[alias] for alias in sorted(site_zero_counts)},
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
