from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
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
from paper_exp.modeling import (
    adaptive_threshold_parameter_items,
    adaptive_threshold_parameter_snapshot,
    adaptive_threshold_training_metrics,
    apply_mlp_hidden_gate,
    apply_post_layernorm_relu,
    apply_post_qkv_relu,
    set_adaptive_threshold_stats_enabled,
)
from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME
from paper_exp.reproducibility import build_training_schedule
from paper_exp.run import RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_jsonl


def run_calibration(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
    mode: str = "calibrate",
) -> Path:
    validate_config(config, allow_todos=False)
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode=mode,
        run_id=run_id,
    ) as run:
        return _run_started_calibration(run.config, run=run)


def _run_started_calibration(
    config: dict[str, Any],
    *,
    run: RunHandle,
) -> Path:
    """Execute validated training inside an already-persisted run lifecycle."""

    torch, np, auto_config, auto_model = _load_training_dependencies()
    run_config = config["run"]
    model_initialization_seed = int(
        run_config.get("model_initialization_seed", run_config["seed"])
    )
    data_order_seed = int(run_config.get("data_order_seed", run_config["seed"]))
    _set_seed(torch, model_initialization_seed)

    experiment_id = run.config_id
    run_dir = run.run_dir
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
            partition=validation_config.get("partition"),
            partition_seed=validation_config.get("partition_seed"),
        ):
            raise ValueError(f"Validation token cache metadata does not match config: {val_metadata_path}")
        expected_partition_hash = validation_config.get("partition_hash")
        actual_partition_hash = validation_metadata.get("source_document_indices_sha256")
        if expected_partition_hash is not None and actual_partition_hash != expected_partition_hash:
            raise ValueError(
                "Validation partition hash does not match config: "
                f"expected {expected_partition_hash}, got {actual_partition_hash}."
            )
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
    initial_parameter_sha256 = _model_parameter_sha256(model)
    model.train()

    base_learning_rate = float(training["learning_rate"])
    optimizer_config = _optimizer_config(training)
    optimizer_parameters = _adamw_parameters(
        model,
        weight_decay=optimizer_config["weight_decay"],
        threshold_learning_rate_multiplier=optimizer_config["threshold_learning_rate_multiplier"],
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=base_learning_rate,
        betas=optimizer_config["betas"],
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
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
    training_schedule_scheme = run_config.get("training_schedule_scheme")
    training_schedule = None
    training_schedule_hash = None
    if training_schedule_scheme == TRAINING_SCHEDULE_SCHEME:
        training_schedule, training_schedule_hash = build_training_schedule(
            np,
            token_count=len(train_tokens),
            block_size=block_size,
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
            micro_batch_size=micro_batch_size,
            seed=data_order_seed,
        )
    expected_schedule_hash = run_config.get("training_schedule_hash")
    if expected_schedule_hash is not None and training_schedule_hash != expected_schedule_hash:
        raise ValueError(
            "Training schedule hash does not match config: "
            f"expected {expected_schedule_hash}, got {training_schedule_hash}."
        )

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

            should_log = step == 1 or step % log_every == 0 or step == max_steps
            should_eval = (
                validation_tokens is not None
                and (step == 1 or step % int(validation_config["eval_every_steps"]) == 0 or step == max_steps)
            )

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
                schedule_step=(None if training_schedule is None else training_schedule[step - 1]),
                record_threshold_metrics=should_log,
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
                    if key.startswith(("pressure/", "activation/", "atg/")) or key in {
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
                _write_live_events(run_dir, events)

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
                    deterministic_batches=training_schedule is not None,
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
                _write_live_events(run_dir, events)
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
            deterministic_batches=training_schedule is not None,
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
        _write_live_events(run_dir, events)

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
        "calibration/model_initialization_seed": model_initialization_seed,
        "calibration/data_order_seed": data_order_seed,
        "calibration/training_schedule_hash": training_schedule_hash,
        "calibration/initial_parameter_sha256": initial_parameter_sha256,
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
    if validation_metadata is not None:
        metrics["calibration/validation_partition"] = validation_config.get("partition", "full")
        metrics["calibration/validation_partition_hash"] = validation_metadata.get(
            "source_document_indices_sha256"
        )
    metrics.update({f"final/{key}": value for key, value in final_pressure_metrics.items()})
    manifest_updates: dict[str, Any] = {
        "tokenized_data": {"train": train_metadata, "validation": validation_metadata},
        "training": {
            "block_size": block_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_steps": max_steps,
            "completed_steps": completed_steps,
            "max_wall_seconds": max_wall_seconds,
            "tokens_per_step": tokens_per_step,
            "loss_logged_as": "mean_micro_batch_loss_over_gradient_accumulation",
            "sampling": "random_contiguous_blocks_with_replacement",
            "sampling_scheme": training_schedule_scheme,
            "model_initialization_seed": model_initialization_seed,
            "data_order_seed": data_order_seed,
            "training_schedule_hash": training_schedule_hash,
            "learning_rate": base_learning_rate,
            "warmup_steps": warmup_steps,
            "learning_rate_schedule": "linear_warmup_then_constant",
            "optimizer": optimizer_config["name"],
            "adamw_betas": list(optimizer_config["betas"]),
            "adamw_eps": optimizer_config["eps"],
            "weight_decay": optimizer_config["weight_decay"],
            "threshold_learning_rate_multiplier": optimizer_config["threshold_learning_rate_multiplier"],
            "gradient_clipping": None,
        },
        "activation_pressure": {
            "enabled": pressure_config.enabled,
            "method": pressure_config.method,
            "sites": pressure_config.sites,
            "weight": pressure_config.weight,
            "pressure_kind": pressure_config.pressure_kind,
            "ricker_c": pressure_config.ricker_c,
            "ricker_sigma": pressure_config.ricker_sigma,
            "step_budget": pressure_config.step_budget,
            "log_thresholds": list(pressure_config.log_thresholds),
        },
    }
    model_manifest = {
        "name": config["model"]["name"],
        "architecture": config["model"]["architecture"],
        "initialization": config["model"]["initialization"],
        "loaded_checkpoint_weights": False,
        "parameter_dtype": _parameter_dtype(model),
        "initial_parameter_sha256": initial_parameter_sha256,
    }
    hidden_act = getattr(getattr(model, "config", None), "hidden_act", None)
    if hidden_act is not None:
        model_manifest["hidden_act"] = hidden_act
    model_manifest["post_layernorm_relu"] = bool(
        getattr(getattr(model, "config", None), "post_layernorm_relu", False)
    )
    post_layernorm_gate = getattr(getattr(model, "config", None), "post_layernorm_gate", None)
    if post_layernorm_gate is not None:
        model_manifest["post_layernorm_gate"] = dict(post_layernorm_gate)
    mlp_hidden_gate = getattr(getattr(model, "config", None), "mlp_hidden_gate", None)
    if mlp_hidden_gate is not None:
        model_manifest["mlp_hidden_gate"] = dict(mlp_hidden_gate)
    post_qkv_relu = getattr(getattr(model, "config", None), "post_qkv_relu", None)
    if post_qkv_relu is not None:
        model_manifest["post_qkv_relu"] = dict(post_qkv_relu)
    threshold_items = adaptive_threshold_parameter_items(model)
    if threshold_items:
        model_manifest["adaptive_threshold_parameter_count"] = len(threshold_items)
        model_manifest["adaptive_threshold_parameters"] = {
            name: float(torch.nn.functional.softplus(parameter.detach().float()).cpu())
            for name, parameter in threshold_items
        }
    manifest_updates["model"] = model_manifest
    validation_manifest = dict(validation_config)
    if validation_metadata is not None:
        validation_manifest["realized_partition_hash"] = validation_metadata.get(
            "source_document_indices_sha256"
        )
        validation_manifest["realized_documents"] = validation_metadata.get("documents")
        validation_manifest["realized_tokens"] = validation_metadata.get("tokens")
    manifest_updates["validation"] = validation_manifest
    manifest_updates["checkpoint"] = checkpoint_metadata
    if "sweep" in config:
        manifest_updates["sweep"] = config["sweep"]

    write_jsonl(run_dir / "events.jsonl", events)
    return complete_run(
        run,
        metrics=metrics,
        predictions=events,
        manifest_updates=manifest_updates,
    )


def _write_live_events(run_dir: Path, events: list[dict[str, Any]]) -> None:
    write_jsonl(run_dir / "events.jsonl", events)


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
    schedule_step: Any = None,
    record_threshold_metrics: bool = False,
) -> dict[str, Any]:
    set_adaptive_threshold_stats_enabled(model, record_threshold_metrics)
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
            schedule_step=schedule_step,
            record_threshold_metrics=record_threshold_metrics,
        )

    threshold_before = (
        adaptive_threshold_parameter_snapshot(model)
        if record_threshold_metrics
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    activation_metrics: dict[str, float] = {}
    task_grads_for_metrics: list[Any | None] = []
    pressure_grads_for_metrics: list[Any | None] = []
    pressure_active = pressure_config.applies_pressure

    for micro_step in range(grad_accum):
        if activation_capture is not None:
            activation_capture.clear()
        batch = _sample_batch(
            torch,
            np,
            train_tokens,
            block_size,
            micro_batch_size,
            device,
            starts=(None if schedule_step is None else schedule_step[micro_step]),
        )
        with _autocast_context(torch, device, dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            current_pressure_loss = (
                pressure_loss(torch, activation_capture.activations, pressure_config)
                if pressure_active
                else None
            )
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
            activation_metrics = activation_near_zero_metrics(
                activation_capture.activations,
                pressure_config.log_thresholds,
            )

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
    if record_threshold_metrics and threshold_before is not None:
        result.update(
            adaptive_threshold_training_metrics(model, before_step=threshold_before)
        )
    set_adaptive_threshold_stats_enabled(model, False)
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
    schedule_step: Any = None,
    record_threshold_metrics: bool = False,
) -> dict[str, Any]:
    set_adaptive_threshold_stats_enabled(model, record_threshold_metrics)
    threshold_before = (
        adaptive_threshold_parameter_snapshot(model)
        if record_threshold_metrics
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    pressure_grads: list[Any | None] = []
    activation_metrics: dict[str, float] = {}

    for micro_step in range(grad_accum):
        activation_capture.clear()
        batch = _sample_batch(
            torch,
            np,
            train_tokens,
            block_size,
            micro_batch_size,
            device,
            starts=(None if schedule_step is None else schedule_step[micro_step]),
        )
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
        activation_metrics = activation_near_zero_metrics(
            activation_capture.activations,
            pressure_config.log_thresholds,
        )

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
    if record_threshold_metrics and threshold_before is not None:
        result.update(
            adaptive_threshold_training_metrics(model, before_step=threshold_before)
        )
    set_adaptive_threshold_stats_enabled(model, False)
    return result


def _sample_batch(
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    device: Any,
    *,
    starts: Any = None,
) -> Any:
    if starts is None:
        max_start = len(tokens) - block_size - 1
        starts = np.random.randint(0, max_start, size=batch_size)
    if len(starts) != batch_size:
        raise ValueError(f"Expected {batch_size} batch starts, got {len(starts)}.")
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
    _apply_model_architecture_overrides(architecture, model_config)
    architecture.torch_dtype = torch.float32
    model = auto_model.from_config(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    return model.to(device=device, dtype=torch.float32)


def _apply_model_architecture_overrides(architecture: Any, model_config: dict[str, Any]) -> None:
    hidden_act = model_config.get("hidden_act")
    if hidden_act is not None:
        if not hasattr(architecture, "hidden_act"):
            raise ValueError("Configured model.hidden_act, but the loaded architecture has no hidden_act field.")
        architecture.hidden_act = hidden_act

    post_layernorm_relu = model_config.get("post_layernorm_relu")
    if post_layernorm_relu is not None:
        architecture.post_layernorm_relu = post_layernorm_relu

    post_layernorm_gate = model_config.get("post_layernorm_gate")
    if post_layernorm_gate is not None:
        architecture.post_layernorm_gate = dict(post_layernorm_gate)

    mlp_hidden_gate = model_config.get("mlp_hidden_gate")
    if mlp_hidden_gate is not None:
        architecture.mlp_hidden_gate = dict(mlp_hidden_gate)

    post_qkv_relu = model_config.get("post_qkv_relu")
    if post_qkv_relu is not None:
        architecture.post_qkv_relu = dict(post_qkv_relu)


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
    deterministic_batches: bool = False,
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
        elif deterministic_batches:
            total_blocks = max(1, (len(tokens) - 1) // block_size)
            starts = [index * block_size for index in range(total_blocks)]
            starts = starts[: int(eval_batches) * batch_size]
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


def _optimizer_config(training: dict[str, Any]) -> dict[str, Any]:
    name = str(training.get("optimizer", "adamw"))
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    betas = training.get("adamw_betas", [0.9, 0.999])
    if not isinstance(betas, list | tuple) or len(betas) != 2:
        raise ValueError("training.adamw_betas must contain exactly two values.")
    beta1 = float(betas[0])
    beta2 = float(betas[1])
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("training.adamw_betas values must be in [0, 1).")
    eps = float(training.get("adamw_eps", 1e-8))
    if eps <= 0.0:
        raise ValueError("training.adamw_eps must be positive.")
    weight_decay = float(training.get("weight_decay", 0.01))
    if weight_decay < 0.0:
        raise ValueError("training.weight_decay must be non-negative.")
    threshold_learning_rate_multiplier = float(
        training.get("threshold_learning_rate_multiplier", 1.0)
    )
    if not math.isfinite(threshold_learning_rate_multiplier) or threshold_learning_rate_multiplier <= 0.0:
        raise ValueError("training.threshold_learning_rate_multiplier must be finite and positive.")
    return {
        "name": name,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "threshold_learning_rate_multiplier": threshold_learning_rate_multiplier,
    }


def _adamw_parameters(
    model: Any,
    *,
    weight_decay: float,
    threshold_learning_rate_multiplier: float,
) -> Any:
    threshold_items = adaptive_threshold_parameter_items(model)
    if not threshold_items:
        return model.parameters()
    threshold_parameters = [parameter for _name, parameter in threshold_items]
    threshold_ids = {id(parameter) for parameter in threshold_parameters}
    model_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in threshold_ids
    ]
    if not model_parameters:
        raise ValueError("AdamW model parameter group is empty.")
    return [
        {
            "params": model_parameters,
            "group_name": "model",
            "lr_multiplier": 1.0,
            "weight_decay": weight_decay,
        },
        {
            "params": threshold_parameters,
            "group_name": "adaptive_threshold",
            "lr_multiplier": threshold_learning_rate_multiplier,
            "weight_decay": 0.0,
        },
    ]


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))


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


def _save_final_checkpoint(
    config: dict[str, Any],
    run_dir: Path,
    model: Any,
    optimizer: Any,
    torch: Any,
) -> dict[str, Any]:
    checkpoint_config = config.get("checkpoint", {})
    if not checkpoint_config.get("save_final", False):
        return {"saved": False, "path": None, "size_mb": None}

    checkpoint_dir = run_dir / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    optimizer_saved = bool(checkpoint_config.get("save_optimizer", False))
    if optimizer_saved:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    return {
        "saved": True,
        "path": str(checkpoint_dir),
        "size_mb": _directory_size_mb(checkpoint_dir),
        "optimizer_saved": optimizer_saved,
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


def _model_parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


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
