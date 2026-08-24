from __future__ import annotations

from contextlib import nullcontext
import hashlib
import time
from pathlib import Path
from typing import Any

from paper_exp.activation_pressure import activation_pressure_config
from paper_exp.activations import ActivationCapture
from paper_exp.config import validate_training_config
from paper_exp.data import (
    metadata_matches_config,
    tokenized_cache_dir,
    validation_metadata_path,
    verify_token_cache,
)
from paper_exp.modeling import _build_random_model
from paper_exp.modeling import model_topology_metadata
from paper_exp.optimization import (
    _autocast_context,
    _build_adamw_optimizer,
    _global_weight_norm,
    _learning_rate_for_step,
    _mlp_weight_norm,
    _run_training_step,
    _sample_batch,
    _set_optimizer_lr,
)
from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME
from paper_exp.reproducibility import build_training_schedule
from paper_exp.run import RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_jsonl


CALIBRATION_TRAINING_WALL_SECONDS = 600.0


def run_training(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
    mode: str,
) -> Path:
    if mode not in {"calibrate", "pretrain"}:
        raise ValueError("Training lifecycle mode must be 'calibrate' or 'pretrain'.")
    validate_training_config(config)
    with run_lifecycle(
        config,
        config_path=config_path,
        command=command,
        mode=mode,
        run_id=run_id,
    ) as run:
        return _run_started_training(run.config, run=run)


def _run_started_training(
    config: dict[str, Any],
    *,
    run: RunHandle,
) -> Path:
    """Execute validated training inside an already-persisted run lifecycle."""

    total_start = time.perf_counter()
    lifecycle_mode = run.launch_manifest.get("mode")
    training_wall_limit_seconds = _training_wall_limit_seconds(lifecycle_mode)
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
    train_tokens_path = verify_token_cache(train_metadata, context="Training token cache")
    train_tokens = np.memmap(train_tokens_path, dtype=np.int32, mode="r")
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
        validation_tokens_path = verify_token_cache(
            validation_metadata,
            context="Validation token cache",
        )
        validation_tokens = np.memmap(validation_tokens_path, dtype=np.int32, mode="r")
        if len(validation_tokens) <= block_size + 1:
            raise ValueError("Validation token cache is too small for the configured block_size.")

    training = config["training"]
    device = _select_device(torch, training["device"])
    dtype = _select_dtype(torch, device, training["precision"])

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
    optimizer, optimizer_config = _build_adamw_optimizer(
        torch=torch,
        model=model,
        training=training,
    )
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pressure_config = activation_pressure_config(config)
    max_steps = int(training["max_steps"])
    warmup_steps = int(training["warmup_steps"])
    grad_accum = int(training["gradient_accumulation_steps"])
    micro_batch_size = int(training["micro_batch_size"])
    log_every = int(training["log_every"])
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

    setup_elapsed = time.perf_counter() - total_start
    train_start = time.perf_counter()
    training_elapsed = 0.0
    diagnostic_elapsed = 0.0
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
    stopped_by_wall_limit = False

    capture_sites = pressure_config.sites if pressure_config.enabled else []
    capture_context = (
        ActivationCapture(model, capture_sites, torch=torch)
        if capture_sites
        else nullcontext(None)
    )

    with capture_context as activation_capture:
        for step in range(1, max_steps + 1):
            step_training_start = time.perf_counter()
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
            )
            step_training_elapsed = time.perf_counter() - step_training_start
            training_elapsed += step_training_elapsed

            diagnostic_start = time.perf_counter()
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
                    "step_wall_seconds": step_training_elapsed,
                }
                event.update(final_pressure_metrics)
                events.append(event)
                _write_live_events(run_dir, events)
            diagnostic_elapsed += time.perf_counter() - diagnostic_start

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
                diagnostic_start = time.perf_counter()
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
                diagnostic_elapsed += time.perf_counter() - diagnostic_start
            completed_steps = step
            if _reached_training_wall_limit(
                training_wall_limit_seconds,
                training_elapsed=training_elapsed,
                completed_steps=completed_steps,
                max_steps=max_steps,
            ):
                stopped_by_wall_limit = True
                break

    if device.type == "cuda":
        synchronization_start = time.perf_counter()
        torch.cuda.synchronize(device)
        training_elapsed += time.perf_counter() - synchronization_start
    training_sample_elapsed = time.perf_counter() - train_start

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
        diagnostic_start = time.perf_counter()
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
        diagnostic_elapsed += time.perf_counter() - diagnostic_start

    checkpoint_start = time.perf_counter()
    checkpoint_metadata = _save_final_checkpoint(config, run_dir, model, optimizer, torch)
    checkpoint_elapsed = time.perf_counter() - checkpoint_start
    total_elapsed = time.perf_counter() - total_start

    final_validation = validation_losses[-1] if validation_losses else None
    best_validation = min(validation_losses, key=lambda item: item[1]) if validation_losses else None
    phase_timing_metrics = _phase_timing_metrics(
        setup=setup_elapsed,
        training=training_elapsed,
        validation=sum(validation_wall_seconds),
        diagnostic=diagnostic_elapsed,
        checkpoint=checkpoint_elapsed,
        training_sample=training_sample_elapsed,
        total=total_elapsed,
    )
    metric_prefix = (
        "training" if run.launch_manifest.get("mode") == "pretrain" else "calibration"
    )
    run_metrics = {
        "train_loss_final": train_losses[-1] if train_losses else None,
        "train_loss_mean": sum(train_losses) / len(train_losses) if train_losses else None,
        "validation_loss_final": final_validation[1] if final_validation else None,
        "validation_loss_final_step": final_validation[0] if final_validation else None,
        "validation_loss_best": best_validation[1] if best_validation else None,
        "validation_loss_best_step": best_validation[0] if best_validation else None,
        "validation_wall_seconds_total": sum(validation_wall_seconds),
        "validation_wall_seconds_final": (
            validation_wall_seconds[-1] if validation_wall_seconds else None
        ),
        "validation_batches_final": final_validation_batches,
        "validation_tokens_final": final_validation_tokens,
        "loss_final": train_losses[-1] if train_losses else None,
        "optimizer_steps": completed_steps,
        "planned_optimizer_steps": max_steps,
        "target_wall_seconds": training_wall_limit_seconds,
        "wall_time_limit_reached": stopped_by_wall_limit,
        "tokens_seen": tokens_seen,
        "tokens_per_step": tokens_per_step,
        "model_initialization_seed": model_initialization_seed,
        "data_order_seed": data_order_seed,
        "training_schedule_hash": training_schedule_hash,
        "initial_parameter_sha256": initial_parameter_sha256,
        "estimated_epochs": tokens_seen / train_metadata["tokens"],
        "wall_seconds": training_elapsed,
        **phase_timing_metrics,
        "tokens_per_second": tokens_seen / training_elapsed if training_elapsed > 0 else None,
        "device": str(device),
        "precision": _precision_label(dtype, device),
        "peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
        "learning_rate_final": final_learning_rate,
        "grad_norm_final": final_grad_norm,
        "weight_norm_final": final_weight_norm,
        "mlp_weight_norm_final": final_mlp_weight_norm,
    }
    metrics = {
        **{f"{metric_prefix}/{key}": value for key, value in run_metrics.items()},
        "checkpoint/final_path": checkpoint_metadata["path"],
        "checkpoint/final_size_mb": checkpoint_metadata["size_mb"],
        "checkpoint/final_saved": checkpoint_metadata["saved"],
    }
    if validation_metadata is not None:
        metrics[f"{metric_prefix}/validation_partition"] = validation_config.get(
            "partition", "full"
        )
        metrics[f"{metric_prefix}/validation_partition_hash"] = validation_metadata.get(
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
            "operational_wall_time_limit_seconds": training_wall_limit_seconds,
            "operational_wall_time_limit_scope": (
                "training" if training_wall_limit_seconds is not None else None
            ),
            "stopped_by_operational_wall_time_limit": stopped_by_wall_limit,
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
            "gradient_clipping": None,
        },
        "activation_pressure": {
            "enabled": pressure_config.enabled,
            "method": pressure_config.method,
            "sites": pressure_config.sites,
            "weight": pressure_config.weight,
            "pressure_kind": pressure_config.pressure_kind,
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
    model_manifest["activation_topology"] = model_topology_metadata(model)
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


def _phase_timing_metrics(
    *,
    setup: float,
    training: float,
    validation: float,
    diagnostic: float,
    checkpoint: float,
    training_sample: float,
    total: float,
) -> dict[str, float]:
    return {
        "wall_seconds_setup": setup,
        "wall_seconds_train": training,
        "wall_seconds_validation": validation,
        "wall_seconds_diagnostic": diagnostic,
        "wall_seconds_checkpoint": checkpoint,
        "wall_seconds_training_sample": training_sample,
        "wall_seconds_total": total,
    }


def _training_wall_limit_seconds(mode: Any) -> float | None:
    if mode == "calibrate":
        return CALIBRATION_TRAINING_WALL_SECONDS
    if mode == "pretrain":
        return None
    raise ValueError(f"Training lifecycle has unsupported mode: {mode!r}.")


def _reached_training_wall_limit(
    limit_seconds: float | None,
    *,
    training_elapsed: float,
    completed_steps: int,
    max_steps: int,
) -> bool:
    return (
        limit_seconds is not None
        and completed_steps < max_steps
        and training_elapsed >= limit_seconds
    )


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
        "path": "checkpoints/final",
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
            "Training requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoConfig, AutoModelForCausalLM
