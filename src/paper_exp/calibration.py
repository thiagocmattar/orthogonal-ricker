from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from paper_exp.config import validate_config
from paper_exp.data import tokenized_cache_dir, validation_metadata_path
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_jsonl


def run_calibration(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    validate_config(config, allow_todos=False)

    torch, np, auto_config, auto_model = _load_training_dependencies()
    _set_seed(torch, config["run"]["seed"])

    experiment_id, numbered_run_id, run_dir = create_run_dir(config, config_path, run_id=run_id)
    metadata_path = tokenized_cache_dir(config, experiment_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token cache not found. Run prepare-data first: {metadata_path}")

    train_metadata = read_json(metadata_path)
    train_tokens = np.memmap(train_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(train_metadata["block_size"])
    if len(train_tokens) <= block_size + 1:
        raise ValueError("Token cache is too small for the configured block_size.")

    validation_config = config.get("validation", {})
    validation_metadata = None
    validation_tokens = None
    if validation_config.get("enabled", False):
        val_metadata_path = validation_metadata_path(config, experiment_id)
        if not val_metadata_path.exists():
            raise FileNotFoundError(f"Validation token cache not found. Run prepare-data first: {val_metadata_path}")
        validation_metadata = read_json(val_metadata_path)
        validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
        if len(validation_tokens) <= block_size + 1:
            raise ValueError("Validation token cache is too small for the configured block_size.")

    training = config["training"]
    device = _select_device(torch, training.get("device", "auto"))
    dtype = _select_dtype(torch, device, training.get("precision", "auto"))

    model = _build_random_model(
        torch=torch,
        auto_config=auto_config,
        auto_model=auto_model,
        model_config=config["model"],
        dtype=dtype,
        device=device,
    )
    model.train()

    base_learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate)
    max_steps = int(training["max_steps"])
    max_wall_seconds = training.get("max_wall_seconds")
    max_wall_seconds = float(max_wall_seconds) if max_wall_seconds is not None else None
    warmup_steps = int(training.get("warmup_steps", 0))
    grad_accum = int(training["gradient_accumulation_steps"])
    micro_batch_size = int(training["micro_batch_size"])
    log_every = int(training.get("log_every", 1))
    tokens_per_step = grad_accum * micro_batch_size * block_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_start = time.perf_counter()
    train_start = time.perf_counter()
    events: list[dict[str, Any]] = []
    train_losses: list[float] = []
    validation_losses: list[tuple[int, float]] = []
    validation_wall_seconds: list[float] = []
    final_validation_batches = None
    final_validation_tokens = None
    final_grad_norm = None
    final_weight_norm = None
    final_learning_rate = None
    completed_steps = 0

    for step in range(1, max_steps + 1):
        step_start = time.perf_counter()
        learning_rate = _learning_rate_for_step(step, base_learning_rate, warmup_steps)
        _set_optimizer_lr(optimizer, learning_rate)

        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = _sample_batch(torch, np, train_tokens, block_size, micro_batch_size, device)
            output = model(input_ids=batch, labels=batch)
            loss = output.loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu()) * grad_accum
        step_loss /= grad_accum

        should_log = step == 1 or step % log_every == 0 or step == max_steps
        should_eval = (
            validation_tokens is not None
            and (step == 1 or step % int(validation_config["eval_every_steps"]) == 0 or step == max_steps)
        )
        grad_norm = _global_grad_norm(model) if should_log or should_eval else None
        optimizer.step()
        weight_norm = _global_weight_norm(model) if should_log or should_eval else None

        tokens_seen = step * tokens_per_step
        estimated_epoch = tokens_seen / train_metadata["tokens"]
        train_losses.append(step_loss)

        if should_log:
            final_grad_norm = grad_norm
            final_weight_norm = weight_norm
            final_learning_rate = learning_rate
            events.append(
                {
                    "event": "train",
                    "step": step,
                    "estimated_epoch": estimated_epoch,
                    "tokens_seen": tokens_seen,
                    "train_loss": step_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": grad_norm,
                    "weight_norm": weight_norm,
                    "step_wall_seconds": time.perf_counter() - step_start,
                }
            )

        if should_eval and validation_tokens is not None:
            validation_start = time.perf_counter()
            validation_result = _evaluate_loss(
                model=model,
                torch=torch,
                np=np,
                tokens=validation_tokens,
                block_size=block_size,
                batch_size=int(validation_config["batch_size"]),
                eval_batches=validation_config.get("eval_batches"),
                device=device,
            )
            validation_elapsed = time.perf_counter() - validation_start
            validation_losses.append((step, validation_result["loss"]))
            validation_wall_seconds.append(validation_elapsed)
            final_validation_batches = validation_result["batches"]
            final_validation_tokens = validation_result["tokens"]
            events.append(
                {
                    "event": "validation",
                    "step": step,
                    "estimated_epoch": estimated_epoch,
                    "tokens_seen": tokens_seen,
                    "validation_loss": validation_result["loss"],
                    "validation_batches": validation_result["batches"],
                    "validation_tokens": validation_result["tokens"],
                    "validation_wall_seconds": validation_elapsed,
                }
            )
        completed_steps = step
        if max_wall_seconds is not None and time.perf_counter() - train_start >= max_wall_seconds:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_elapsed = time.perf_counter() - train_start

    tokens_seen = completed_steps * tokens_per_step
    if validation_tokens is not None and (not validation_losses or validation_losses[-1][0] != completed_steps):
        validation_start = time.perf_counter()
        validation_result = _evaluate_loss(
            model=model,
            torch=torch,
            np=np,
            tokens=validation_tokens,
            block_size=block_size,
            batch_size=int(validation_config["batch_size"]),
            eval_batches=validation_config.get("eval_batches"),
            device=device,
        )
        validation_elapsed = time.perf_counter() - validation_start
        validation_losses.append((completed_steps, validation_result["loss"]))
        validation_wall_seconds.append(validation_elapsed)
        final_validation_batches = validation_result["batches"]
        final_validation_tokens = validation_result["tokens"]
        events.append(
            {
                "event": "validation",
                "step": completed_steps,
                "estimated_epoch": tokens_seen / train_metadata["tokens"],
                "tokens_seen": tokens_seen,
                "validation_loss": validation_result["loss"],
                "validation_batches": validation_result["batches"],
                "validation_tokens": validation_result["tokens"],
                "validation_wall_seconds": validation_elapsed,
            }
        )

    checkpoint_metadata = _save_final_checkpoint(config, run_dir, model, optimizer, torch)
    total_elapsed = time.perf_counter() - total_start

    final_validation = validation_losses[-1] if validation_losses else None
    best_validation = min(validation_losses, key=lambda item: item[1]) if validation_losses else None
    metrics = {
        "calibration/train_loss_final": train_losses[-1] if train_losses else None,
        "calibration/train_loss_mean": sum(train_losses) / len(train_losses) if train_losses else None,
        "calibration/validation_loss_final": final_validation[1] if final_validation else None,
        "calibration/validation_loss_final_step": final_validation[0] if final_validation else None,
        "calibration/validation_loss_best": best_validation[1] if best_validation else None,
        "calibration/validation_loss_best_step": best_validation[0] if best_validation else None,
        "calibration/validation_wall_seconds_total": sum(validation_wall_seconds),
        "calibration/validation_wall_seconds_final": validation_wall_seconds[-1] if validation_wall_seconds else None,
        "calibration/validation_batches_final": final_validation_batches,
        "calibration/validation_tokens_final": final_validation_tokens,
        "calibration/loss_final": train_losses[-1] if train_losses else None,
        "calibration/optimizer_steps": completed_steps,
        "calibration/planned_optimizer_steps": max_steps,
        "calibration/target_wall_seconds": max_wall_seconds,
        "calibration/tokens_seen": tokens_seen,
        "calibration/tokens_per_step": tokens_per_step,
        "calibration/estimated_epochs": tokens_seen / train_metadata["tokens"],
        "calibration/wall_seconds": train_elapsed,
        "calibration/wall_seconds_train": train_elapsed,
        "calibration/wall_seconds_total": total_elapsed,
        "calibration/tokens_per_second": tokens_seen / train_elapsed if train_elapsed > 0 else None,
        "calibration/device": str(device),
        "calibration/precision": str(dtype).replace("torch.", "") if dtype is not None else "float32",
        "calibration/peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "calibration/peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
        "calibration/learning_rate_final": final_learning_rate,
        "calibration/grad_norm_final": final_grad_norm,
        "calibration/weight_norm_final": final_weight_norm,
        "checkpoint/final_path": checkpoint_metadata["path"],
        "checkpoint/final_size_mb": checkpoint_metadata["size_mb"],
        "checkpoint/final_saved": checkpoint_metadata["saved"],
    }
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="calibrate",
        config_id=experiment_id,
        result_path=run_dir,
    )
    manifest["tokenized_data"] = {"train": train_metadata, "validation": validation_metadata}
    manifest["training"] = {
        "block_size": block_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "max_wall_seconds": max_wall_seconds,
        "tokens_per_step": tokens_per_step,
        "loss_logged_as": "mean_micro_batch_loss_over_gradient_accumulation",
        "sampling": "random_contiguous_blocks_with_replacement",
    }
    manifest["model"] = {
        "name": config["model"]["name"],
        "architecture": config["model"]["architecture"],
        "initialization": config["model"]["initialization"],
        "loaded_checkpoint_weights": False,
    }
    manifest["validation"] = validation_config
    manifest["checkpoint"] = checkpoint_metadata

    write_run_artifacts(run_dir, config=config, metrics=metrics, manifest=manifest, predictions=events)
    write_jsonl(run_dir / "events.jsonl", events)
    return run_dir


def _sample_batch(
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    device: Any,
) -> Any:
    max_start = len(tokens) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    batch = np.stack([tokens[start : start + block_size] for start in starts])
    return torch.as_tensor(batch, dtype=torch.long, device=device)


def _build_random_model(
    *,
    torch: Any,
    auto_config: Any,
    auto_model: Any,
    model_config: dict[str, Any],
    dtype: Any,
    device: Any,
) -> Any:
    if model_config["initialization"] != "random":
        raise ValueError("This pretraining harness only supports model.initialization: random.")

    architecture_kwargs = {"revision": model_config.get("revision")}
    architecture_kwargs = {key: value for key, value in architecture_kwargs.items() if value is not None}
    architecture = auto_config.from_pretrained(model_config["architecture"], **architecture_kwargs)
    model = auto_model.from_config(architecture)
    if dtype is not None and device.type == "cuda":
        return model.to(device=device, dtype=dtype)
    return model.to(device)


def _evaluate_loss(
    *,
    model: Any,
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    eval_batches: int | None,
    device: Any,
) -> dict[str, Any]:
    model.eval()
    weighted_loss = 0.0
    total_sequences = 0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        if eval_batches is None:
            total_blocks = max(1, (len(tokens) - 1) // block_size)
            starts = [index * block_size for index in range(total_blocks)]
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                batch = np.stack([tokens[start : start + block_size] for start in batch_starts])
                input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                batch_sequences = len(batch_starts)
                loss = float(model(input_ids=input_ids, labels=input_ids).loss.detach().cpu())
                weighted_loss += loss * batch_sequences
                total_sequences += batch_sequences
                total_tokens += batch_sequences * block_size
                batches += 1
        else:
            for _ in range(int(eval_batches)):
                input_ids = _sample_batch(torch, np, tokens, block_size, batch_size, device)
                loss = float(model(input_ids=input_ids, labels=input_ids).loss.detach().cpu())
                weighted_loss += loss * batch_size
                total_sequences += batch_size
                total_tokens += batch_size * block_size
                batches += 1
    model.train()
    return {
        "loss": weighted_loss / total_sequences,
        "batches": batches,
        "tokens": total_tokens,
    }


def _learning_rate_for_step(step: int, base_learning_rate: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_learning_rate
    return base_learning_rate * min(1.0, step / warmup_steps)


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _global_grad_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _global_weight_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        param_norm = parameter.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _save_final_checkpoint(config: dict[str, Any], run_dir: Path, model: Any, optimizer: Any, torch: Any) -> dict[str, Any]:
    checkpoint_config = config.get("checkpoint", {})
    if not checkpoint_config.get("save_final", False):
        return {"saved": False, "path": None, "size_mb": None}

    checkpoint_dir = run_dir / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    if checkpoint_config.get("save_optimizer", False):
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    return {
        "saved": True,
        "path": str(checkpoint_dir),
        "size_mb": _directory_size_mb(checkpoint_dir),
    }


def _directory_size_mb(path: Path) -> float:
    total_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total_bytes / (1024 * 1024)


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


def _set_seed(torch: Any, seed: int) -> None:
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak_gpu_memory_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _peak_gpu_reserved_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024 * 1024)


def _load_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Calibration requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoConfig, AutoModelForCausalLM
