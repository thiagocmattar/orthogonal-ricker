from __future__ import annotations

from contextlib import nullcontext
import time
from pathlib import Path
from typing import Any

from paper_exp.activation_pressure import accumulate_grads
from paper_exp.activation_pressure import activation_near_zero_metrics
from paper_exp.activation_pressure import activation_pressure_config
from paper_exp.activation_pressure import apply_adam_step_orthogonal_pressure
from paper_exp.activation_pressure import clone_grads
from paper_exp.activation_pressure import grad_metrics
from paper_exp.activation_pressure import pressure_loss
from paper_exp.activations import ActivationCapture
from paper_exp.config import validate_config
from paper_exp.data import metadata_matches_config, tokenized_cache_dir, validation_metadata_path
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_jsonl


def run_calibration(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
    mode: str = "calibrate",
) -> Path:
    validate_config(config, allow_todos=False)

    torch, np, auto_config, auto_model = _load_training_dependencies()
    _set_seed(torch, config["run"]["seed"])

    experiment_id, numbered_run_id, run_dir = create_run_dir(config, config_path, run_id=run_id)
    metadata_path = tokenized_cache_dir(config, experiment_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token cache not found. Run prepare-data first: {metadata_path}")

    train_metadata = read_json(metadata_path)
    if not metadata_matches_config(
        train_metadata,
        config,
        split=config["data"]["split"],
        max_documents=config["data"].get("max_documents"),
    ):
        raise ValueError(f"Token cache metadata does not match config: {metadata_path}")
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
        if not metadata_matches_config(
            validation_metadata,
            config,
            split=validation_config["split"],
            max_documents=validation_config.get("max_documents"),
        ):
            raise ValueError(f"Validation token cache metadata does not match config: {val_metadata_path}")
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
        device=device,
    )
    model.train()

    base_learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pressure_config = activation_pressure_config(config)
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
    final_mlp_weight_norm = None
    final_learning_rate = None
    final_pressure_metrics: dict[str, Any] = {}
    completed_steps = 0

    capture_sites = pressure_config.sites if pressure_config.enabled else []
    capture_context = (
        ActivationCapture(model, capture_sites, torch=torch)
        if capture_sites
        else nullcontext(None)
    )

    with capture_context as activation_capture:
        for step in range(1, max_steps + 1):
            step_start = time.perf_counter()
            learning_rate = _learning_rate_for_step(step, base_learning_rate, warmup_steps)
            _set_optimizer_lr(optimizer, learning_rate)

            step_result = _run_training_step(
                model=model,
                optimizer=optimizer,
                params=trainable_params,
                torch=torch,
                np=np,
                train_tokens=train_tokens,
                block_size=block_size,
                micro_batch_size=micro_batch_size,
                grad_accum=grad_accum,
                device=device,
                dtype=dtype,
                pressure_config=pressure_config,
                activation_capture=activation_capture,
                step=step,
            )

            should_log = step == 1 or step % log_every == 0 or step == max_steps
            should_eval = (
                validation_tokens is not None
                and (step == 1 or step % int(validation_config["eval_every_steps"]) == 0 or step == max_steps)
            )
            grad_norm = step_result["pressure/task_gradient_norm"] if should_log or should_eval else None
            weight_norm = _global_weight_norm(model) if should_log or should_eval else None
            mlp_weight_norm = _mlp_weight_norm(model) if should_log or should_eval else None

            tokens_seen = step * tokens_per_step
            estimated_epoch = tokens_seen / train_metadata["tokens"]
            step_loss = step_result["task_loss"]
            train_losses.append(step_loss)

            if should_log:
                final_grad_norm = grad_norm
                final_weight_norm = weight_norm
                final_mlp_weight_norm = mlp_weight_norm
                final_learning_rate = learning_rate
                final_pressure_metrics = {
                    key: value
                    for key, value in step_result.items()
                    if key.startswith("pressure/") or key.startswith("activation/") or key in {
                        "pressure_loss",
                        "pressure_weight",
                        "weighted_pressure_loss",
                        "augmented_loss",
                    }
                }
                event = {
                    "event": "train",
                    "step": step,
                    "estimated_epoch": estimated_epoch,
                    "tokens_seen": tokens_seen,
                    "train_loss": step_loss,
                    "task_loss": step_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": grad_norm,
                    "weight_norm": weight_norm,
                    "mlp_weight_norm": mlp_weight_norm,
                    "step_wall_seconds": time.perf_counter() - step_start,
                }
                event.update(final_pressure_metrics)
                events.append(event)

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
                    dtype=dtype,
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
            dtype=dtype,
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
        "calibration/precision": _precision_label(dtype, device),
        "calibration/peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "calibration/peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
        "calibration/learning_rate_final": final_learning_rate,
        "calibration/grad_norm_final": final_grad_norm,
        "calibration/weight_norm_final": final_weight_norm,
        "calibration/mlp_weight_norm_final": final_mlp_weight_norm,
        "checkpoint/final_path": checkpoint_metadata["path"],
        "checkpoint/final_size_mb": checkpoint_metadata["size_mb"],
        "checkpoint/final_saved": checkpoint_metadata["saved"],
    }
    metrics.update({f"final/{key}": value for key, value in final_pressure_metrics.items()})
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode=mode,
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
    manifest["activation_pressure"] = {
        "enabled": pressure_config.enabled,
        "method": pressure_config.method,
        "sites": pressure_config.sites,
        "weight": pressure_config.weight,
        "pressure_kind": pressure_config.pressure_kind,
        "ricker_c": pressure_config.ricker_c,
        "ricker_sigma": pressure_config.ricker_sigma,
        "step_budget": pressure_config.step_budget,
        "log_thresholds": list(pressure_config.log_thresholds),
    }
    manifest["model"] = {
        "name": config["model"]["name"],
        "architecture": config["model"]["architecture"],
        "initialization": config["model"]["initialization"],
        "loaded_checkpoint_weights": False,
        "parameter_dtype": _parameter_dtype(model),
    }
    manifest["validation"] = validation_config
    manifest["checkpoint"] = checkpoint_metadata
    if "sweep" in config:
        manifest["sweep"] = config["sweep"]

    write_run_artifacts(run_dir, config=config, metrics=metrics, manifest=manifest, predictions=events)
    write_jsonl(run_dir / "events.jsonl", events)
    return run_dir


def _run_training_step(
    *,
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    np: Any,
    train_tokens: Any,
    block_size: int,
    micro_batch_size: int,
    grad_accum: int,
    device: Any,
    dtype: Any,
    pressure_config: Any,
    activation_capture: Any,
    step: int,
) -> dict[str, Any]:
    if pressure_config.enabled and activation_capture is None:
        raise ValueError("Activation pressure is enabled but no activation capture is registered.")

    if pressure_config.orthogonal:
        return _run_orthogonal_pressure_step(
            model=model,
            optimizer=optimizer,
            params=params,
            torch=torch,
            np=np,
            train_tokens=train_tokens,
            block_size=block_size,
            micro_batch_size=micro_batch_size,
            grad_accum=grad_accum,
            device=device,
            dtype=dtype,
            pressure_config=pressure_config,
            activation_capture=activation_capture,
            step=step,
        )

    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    activation_metrics: dict[str, float] = {}
    task_grads_for_metrics: list[Any | None] = []
    pressure_grads_for_metrics: list[Any | None] = []
    pressure_active = pressure_config.applies_pressure

    for _ in range(grad_accum):
        if activation_capture is not None:
            activation_capture.clear()
        batch = _sample_batch(torch, np, train_tokens, block_size, micro_batch_size, device)
        with _autocast_context(torch, device, dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            current_pressure_loss = pressure_loss(torch, activation_capture.activations, pressure_config) if pressure_active else None
            augmented_loss = (
                task_loss + pressure_config.weight * current_pressure_loss
                if current_pressure_loss is not None
                else task_loss
            )
        _require_finite_loss(torch, task_loss, f"task loss at step {step}")
        _require_finite_loss(torch, augmented_loss, f"training loss at step {step}")
        if pressure_active and current_pressure_loss is not None:
            task_grads_for_metrics = accumulate_grads(
                task_grads_for_metrics,
                torch.autograd.grad(task_loss / grad_accum, params, retain_graph=True, allow_unused=True),
            )
            pressure_grads_for_metrics = accumulate_grads(
                pressure_grads_for_metrics,
                torch.autograd.grad(current_pressure_loss / grad_accum, params, retain_graph=True, allow_unused=True),
            )
        (augmented_loss / grad_accum).backward()
        task_loss_total += float(task_loss.detach().cpu())
        if current_pressure_loss is not None:
            _require_finite_loss(torch, current_pressure_loss, f"pressure loss at step {step}")
            pressure_loss_total += float(current_pressure_loss.detach().cpu())
        if activation_capture is not None:
            activation_metrics = activation_near_zero_metrics(activation_capture.activations, pressure_config.log_thresholds)

    task_grads = task_grads_for_metrics if pressure_active else clone_grads(params)
    pressure_grads = pressure_grads_for_metrics if pressure_active else [None for _ in params]
    step_metrics = grad_metrics(torch, task_grads, pressure_grads)
    optimizer.step()

    task_loss_mean = task_loss_total / grad_accum
    pressure_loss_mean = pressure_loss_total / grad_accum if pressure_active else None
    result = {
        "task_loss": task_loss_mean,
        "pressure/task_gradient_norm": step_metrics["pressure/task_gradient_norm"],
    }
    if pressure_active:
        result.update(step_metrics)
        result.update(
            {
                "pressure_loss": pressure_loss_mean,
                "pressure_weight": pressure_config.weight,
                "weighted_pressure_loss": pressure_config.weight * pressure_loss_mean,
                "augmented_loss": task_loss_mean + pressure_config.weight * pressure_loss_mean,
            }
        )
    if pressure_config.enabled:
        result.update(activation_metrics)
    return result


def _run_orthogonal_pressure_step(
    *,
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    np: Any,
    train_tokens: Any,
    block_size: int,
    micro_batch_size: int,
    grad_accum: int,
    device: Any,
    dtype: Any,
    pressure_config: Any,
    activation_capture: Any,
    step: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    pressure_grads: list[Any | None] = []
    activation_metrics: dict[str, float] = {}

    for _ in range(grad_accum):
        activation_capture.clear()
        batch = _sample_batch(torch, np, train_tokens, block_size, micro_batch_size, device)
        with _autocast_context(torch, device, dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            current_pressure_loss = pressure_loss(torch, activation_capture.activations, pressure_config)
        _require_finite_loss(torch, task_loss, f"task loss at step {step}")
        _require_finite_loss(torch, current_pressure_loss, f"pressure loss at step {step}")

        (task_loss / grad_accum).backward(retain_graph=True)
        new_pressure_grads = torch.autograd.grad(
            current_pressure_loss / grad_accum,
            params,
            allow_unused=True,
        )
        pressure_grads = accumulate_grads(pressure_grads, new_pressure_grads)
        task_loss_total += float(task_loss.detach().cpu())
        pressure_loss_total += float(current_pressure_loss.detach().cpu())
        activation_metrics = activation_near_zero_metrics(activation_capture.activations, pressure_config.log_thresholds)

    task_grads = clone_grads(params)
    result = {
        "task_loss": task_loss_total / grad_accum,
        "pressure_loss": pressure_loss_total / grad_accum,
        "pressure_weight": pressure_config.weight,
        "weighted_pressure_loss": pressure_config.weight * pressure_loss_total / grad_accum,
        "augmented_loss": task_loss_total / grad_accum + pressure_config.weight * pressure_loss_total / grad_accum,
    }
    result.update(grad_metrics(torch, task_grads, pressure_grads))

    optimizer.step()
    result.update(
        apply_adam_step_orthogonal_pressure(
            optimizer,
            params,
            task_grads,
            pressure_grads,
            pressure_weight=pressure_config.weight,
            step_budget=pressure_config.step_budget,
            eps=pressure_config.eps,
        )
    )
    result.update(activation_metrics)
    return result


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
    device: Any,
) -> Any:
    if model_config["initialization"] != "random":
        raise ValueError("This pretraining harness only supports model.initialization: random.")

    architecture_kwargs = {"revision": model_config.get("revision")}
    architecture_kwargs = {key: value for key, value in architecture_kwargs.items() if value is not None}
    architecture = auto_config.from_pretrained(model_config["architecture"], **architecture_kwargs)
    architecture.torch_dtype = torch.float32
    model = auto_model.from_config(architecture)
    return model.to(device=device, dtype=torch.float32)


def _autocast_context(torch: Any, device: Any, dtype: Any) -> Any:
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _require_finite_loss(torch: Any, loss: Any, label: str) -> None:
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(f"Non-finite {label}.")


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
    dtype: Any,
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
                with _autocast_context(torch, device, dtype):
                    output = model(input_ids=input_ids, labels=input_ids)
                if not bool(torch.isfinite(output.loss.detach()).item()):
                    raise RuntimeError("Non-finite validation loss.")
                loss = float(output.loss.detach().cpu())
                weighted_loss += loss * batch_sequences
                total_sequences += batch_sequences
                total_tokens += batch_sequences * block_size
                batches += 1
        else:
            for _ in range(int(eval_batches)):
                input_ids = _sample_batch(torch, np, tokens, block_size, batch_size, device)
                with _autocast_context(torch, device, dtype):
                    output = model(input_ids=input_ids, labels=input_ids)
                if not bool(torch.isfinite(output.loss.detach()).item()):
                    raise RuntimeError("Non-finite validation loss.")
                loss = float(output.loss.detach().cpu())
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


def _mlp_weight_norm(model: Any) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if ".mlp." not in name or not name.endswith(".weight"):
            continue
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


def _precision_label(dtype: Any, device: Any) -> str:
    if dtype is None or device.type != "cuda":
        return "float32"
    return f"{str(dtype).replace('torch.', '')}_autocast"


def _parameter_dtype(model: Any) -> str:
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        return "unknown"
    return str(first_parameter.dtype).replace("torch.", "")


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
