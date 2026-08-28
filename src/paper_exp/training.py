from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
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
from paper_exp.modeling import load_checkpoint_model
from paper_exp.modeling import model_topology_metadata
from paper_exp.optimization import (
    GLOBAL_GRADIENT_CLIP_MAX_NORM,
    _autocast_context,
    _build_adamw_optimizer,
    _global_weight_norm,
    _learning_rate_for_step,
    _mlp_weight_norm,
    _run_training_step,
    _sample_batch,
    _set_optimizer_lr,
    _warmup_steps_for_budget,
)
from paper_exp.reproducibility import TRAINING_SCHEDULE_SCHEME
from paper_exp.reproducibility import VALIDATION_PARTITION_SCHEME
from paper_exp.reproducibility import build_training_schedule
from paper_exp.reproducibility import training_schedule_metadata
from paper_exp.run import RunHandle, complete_run, run_lifecycle
from paper_exp.utils import read_json, write_jsonl


CALIBRATION_TRAINING_WALL_SECONDS = 600.0
VALIDATION_INTERVAL_STEPS = 191
VALIDATION_CACHE_IDENTITY_FIELDS = (
    "tokens_path",
    "dtype",
    "block_size",
    "tokens",
    "tokens_bytes",
    "tokens_sha256",
    "partition",
    "partition_scheme",
    "partition_seed",
    "source_documents",
    "source_document_indices_sha256",
)

_OL1_BOUNDARY_COUNT = "ol1/optimizer_boundary_count"


def _empty_ol1_boundary_counters() -> dict[str, int]:
    return {
        _OL1_BOUNDARY_COUNT: 0,
        "ol1/raw_gradient_conflict_boundary_count": 0,
        "ol1/preconditioned_projection_boundary_count": 0,
        "ol1/trust_budget_limited_boundary_count": 0,
        "ol1/eligible_parameter_tensor_count_sum": 0,
        "ol1/skipped_parameter_tensor_count_sum": 0,
    }


def _accumulate_ol1_boundary_counters(
    counters: dict[str, int],
    step_result: dict[str, Any],
) -> None:
    counters[_OL1_BOUNDARY_COUNT] += 1
    counters["ol1/raw_gradient_conflict_boundary_count"] += int(
        bool(step_result["pressure/pressure_conflict"])
    )
    counters["ol1/preconditioned_projection_boundary_count"] += int(
        bool(step_result["pressure/pressure_update_projected"])
    )
    counters["ol1/trust_budget_limited_boundary_count"] += int(
        float(step_result["pressure/pressure_update_applied_scale"]) < 1.0
    )
    counters["ol1/eligible_parameter_tensor_count_sum"] += int(
        step_result["pressure/eligible_parameters"]
    )
    counters["ol1/skipped_parameter_tensor_count_sum"] += int(
        step_result["pressure/skipped_parameters"]
    )


def _final_ol1_boundary_counters(
    counters: dict[str, int],
    *,
    completed_steps: int,
) -> dict[str, int]:
    if counters[_OL1_BOUNDARY_COUNT] != completed_steps:
        raise RuntimeError(
            "OL1 optimizer-boundary counter coverage does not match completed steps."
        )
    return dict(counters)


def validation_update_steps(
    max_steps: int,
    *,
    interval: int = VALIDATION_INTERVAL_STEPS,
) -> tuple[int, ...]:
    """Return update 1, every interval boundary, and the final update."""

    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if isinstance(interval, bool) or int(interval) <= 0:
        raise ValueError("validation interval must be a positive integer.")
    max_steps = int(max_steps)
    interval = int(interval)
    steps = {1, max_steps}
    steps.update(range(interval, max_steps + 1, interval))
    return tuple(sorted(steps))


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
    _require_expected_cache_hash(
        train_metadata,
        config.get("preprocessing", {}).get("tokens_sha256"),
        context="Training token cache",
    )
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
        _require_expected_cache_hash(
            validation_metadata,
            validation_config.get("tokens_sha256"),
            context="Validation token cache",
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
    expected_warmup_steps = _warmup_steps_for_budget(max_steps)
    if warmup_steps != expected_warmup_steps:
        raise ValueError(
            "training.warmup_steps must equal ceil(0.01 * training.max_steps): "
            f"expected {expected_warmup_steps}, got {warmup_steps}."
        )
    grad_accum = int(training["gradient_accumulation_steps"])
    micro_batch_size = int(training["micro_batch_size"])
    log_every = int(training["log_every"])
    tokens_per_step = grad_accum * micro_batch_size * block_size
    training_schedule_scheme = run_config.get("training_schedule_scheme")
    training_schedule = None
    training_schedule_hash = None
    training_schedule_details = None
    if training_schedule_scheme == TRAINING_SCHEDULE_SCHEME:
        training_schedule_details = training_schedule_metadata(
            token_count=len(train_tokens),
            block_size=block_size,
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
            micro_batch_size=micro_batch_size,
            seed=data_order_seed,
        )
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
    final_validation_sequences = None
    final_validation_tokens = None
    final_validation_complete_blocks = None
    final_validation_excluded_tail_tokens = None
    final_validation_complete = None
    final_grad_norm = None
    final_weight_norm = None
    final_mlp_weight_norm = None
    final_learning_rate = None
    final_pressure_metrics: dict[str, Any] = {}
    ol1_boundary_counters = (
        _empty_ol1_boundary_counters() if pressure_config.orthogonal else None
    )
    completed_steps = 0
    stopped_by_wall_limit = False
    validation_steps: frozenset[int] = frozenset()
    if validation_tokens is not None:
        validation_interval = int(validation_config["eval_every_steps"])
        if validation_interval != VALIDATION_INTERVAL_STEPS:
            raise ValueError(
                f"validation.eval_every_steps must equal {VALIDATION_INTERVAL_STEPS}."
            )
        validation_steps = frozenset(validation_update_steps(max_steps))

    capture_sites = pressure_config.sites if pressure_config.enabled else []
    capture_context = (
        ActivationCapture(model, capture_sites, torch=torch)
        if capture_sites
        else nullcontext(None)
    )

    with capture_context as activation_capture:
        for step in range(1, max_steps + 1):
            step_training_start = time.perf_counter()
            learning_rate = _learning_rate_for_step(
                step,
                base_learning_rate,
                warmup_steps,
                max_steps,
            )
            _set_optimizer_lr(optimizer, learning_rate)

            should_log = step == 1 or step % log_every == 0 or step == max_steps
            should_eval = (
                validation_tokens is not None
                and step in validation_steps
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
            if ol1_boundary_counters is not None:
                _accumulate_ol1_boundary_counters(
                    ol1_boundary_counters,
                    step_result,
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
                    if key.startswith(
                        ("pressure/", "activation/", "atg/", "optimization/")
                    ) or key in {
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
                final_validation_sequences = validation_result["sequences"]
                final_validation_tokens = validation_result["tokens"]
                final_validation_complete_blocks = validation_result["available_complete_blocks"]
                final_validation_excluded_tail_tokens = validation_result[
                    "excluded_tail_tokens"
                ]
                final_validation_complete = validation_result["complete_block_coverage"]
                diagnostic_start = time.perf_counter()
                events.append(
                    {
                        "event": "validation",
                        "step": step,
                        "estimated_epoch": estimated_epoch,
                        "tokens_seen": tokens_seen,
                        "validation_loss": validation_result["loss"],
                        "validation_batches": validation_result["batches"],
                        "validation_sequences": validation_result["sequences"],
                        "validation_tokens": validation_result["tokens"],
                        "validation_available_complete_blocks": validation_result[
                            "available_complete_blocks"
                        ],
                        "validation_excluded_tail_tokens": validation_result[
                            "excluded_tail_tokens"
                        ],
                        "validation_complete_block_coverage": validation_result[
                            "complete_block_coverage"
                        ],
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
        final_validation_sequences = validation_result["sequences"]
        final_validation_tokens = validation_result["tokens"]
        final_validation_complete_blocks = validation_result["available_complete_blocks"]
        final_validation_excluded_tail_tokens = validation_result["excluded_tail_tokens"]
        final_validation_complete = validation_result["complete_block_coverage"]
        diagnostic_start = time.perf_counter()
        events.append(
            {
                "event": "validation",
                "step": completed_steps,
                "estimated_epoch": tokens_seen / train_metadata["tokens"],
                "tokens_seen": tokens_seen,
                "validation_loss": validation_result["loss"],
                "validation_batches": validation_result["batches"],
                "validation_sequences": validation_result["sequences"],
                "validation_tokens": validation_result["tokens"],
                "validation_available_complete_blocks": validation_result[
                    "available_complete_blocks"
                ],
                "validation_excluded_tail_tokens": validation_result[
                    "excluded_tail_tokens"
                ],
                "validation_complete_block_coverage": validation_result[
                    "complete_block_coverage"
                ],
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
        "validation_sequences_final": final_validation_sequences,
        "validation_tokens_final": final_validation_tokens,
        "validation_available_complete_blocks": final_validation_complete_blocks,
        "validation_excluded_tail_tokens": final_validation_excluded_tail_tokens,
        "validation_complete_block_coverage": final_validation_complete,
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
    if ol1_boundary_counters is not None:
        run_metrics.update(
            _final_ol1_boundary_counters(
                ol1_boundary_counters,
                completed_steps=completed_steps,
            )
        )
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
            "sampling": "seeded_complete_block_permutation_with_prefix_wrap",
            "sampling_scheme": training_schedule_scheme,
            "schedule": training_schedule_details,
            "model_initialization_seed": model_initialization_seed,
            "data_order_seed": data_order_seed,
            "training_schedule_hash": training_schedule_hash,
            "learning_rate": base_learning_rate,
            "warmup_steps": warmup_steps,
            "learning_rate_schedule": "linear_warmup_then_cosine_decay",
            "minimum_learning_rate_ratio": 0.1,
            "optimizer": optimizer_config["name"],
            "adamw_betas": list(optimizer_config["betas"]),
            "adamw_eps": optimizer_config["eps"],
            "weight_decay": optimizer_config["weight_decay"],
            "gradient_clipping": _gradient_clipping_manifest(pressure_config.method),
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


def _gradient_clipping_manifest(pressure_method: str) -> dict[str, Any]:
    return {
        "type": "global_l2_norm",
        "max_norm": GLOBAL_GRADIENT_CLIP_MAX_NORM,
        "gradient_scope": (
            "task_plus_weighted_pressure"
            if pressure_method == "l1_naive"
            else "task_only"
        ),
        "applied_immediately_before": "adamw_step",
        "error_if_nonfinite": True,
        "orthogonal_pressure_direction_included": False,
    }


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


def evaluate_saved_checkpoint_confirmation(
    *,
    checkpoint_path: str | Path,
    source_identity: dict[str, Any],
    provenance: dict[str, Any],
    validation_metadata: dict[str, Any],
    tokens: Any,
    batch_size: int,
    torch: Any,
    np: Any,
    auto_model: Any,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Evaluate one saved checkpoint over every complete confirmation block.

    Source-run resolution and durable publication remain caller responsibilities.
    This helper owns checkpoint loading, fixed-order complete-block coverage, and
    the self-contained result record needed by that outer workflow.
    """

    source = _exact_source_identity(source_identity)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Confirmation validation requires non-empty provenance.")
    checkpoint_dir = Path(checkpoint_path).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Saved checkpoint directory not found: {checkpoint_dir}")

    cache_identity = _confirmation_cache_identity(validation_metadata, token_count=len(tokens))
    block_size = int(cache_identity["block_size"])
    if batch_size != 4:
        raise ValueError("Confirmation validation batch_size must equal 4.")

    model = load_checkpoint_model(auto_model, checkpoint_dir, torch=torch)
    model = model.to(device=device, dtype=torch.float32)
    checkpoint_identity = {
        "path": str(checkpoint_dir),
        "content_sha256": _directory_content_sha256(checkpoint_dir),
        "parameter_sha256": _model_parameter_sha256(model),
    }
    result = _evaluate_loss(
        model=model,
        torch=torch,
        np=np,
        tokens=tokens,
        block_size=block_size,
        batch_size=int(batch_size),
        eval_batches=None,
        device=device,
        dtype=dtype,
        deterministic_batches=True,
    )
    expected_sequences, excluded_tail_tokens = divmod(len(tokens), block_size)
    complete = result["sequences"] == expected_sequences
    if not complete:
        raise RuntimeError(
            "Confirmation validation did not cover every complete block: "
            f"expected {expected_sequences}, evaluated {result['sequences']}."
        )
    validation_loss = float(result["loss"])
    try:
        perplexity = math.exp(validation_loss)
    except OverflowError as exc:
        raise RuntimeError("Confirmation validation perplexity is non-finite.") from exc
    if not math.isfinite(perplexity):
        raise RuntimeError("Confirmation validation perplexity is non-finite.")

    return {
        "schema_version": 1,
        "kind": "saved_checkpoint_confirmation_validation",
        "source": source,
        "checkpoint": checkpoint_identity,
        "validation_cache": cache_identity,
        "coverage": {
            "fixed_order": True,
            "expected_complete_sequences": expected_sequences,
            "evaluated_sequences": result["sequences"],
            "sequence_length": block_size,
            "evaluated_tokens": result["tokens"],
            "evaluated_batches": result["batches"],
            "excluded_tail_tokens": excluded_tail_tokens,
            "complete": complete,
        },
        "metrics": {
            "validation_loss": validation_loss,
            "perplexity": perplexity,
        },
        "completeness": {
            "partition_is_confirmation": True,
            "all_complete_blocks_evaluated": complete,
            "excluded_tail_not_evaluated": True,
            "finite_loss_and_perplexity": True,
        },
        "provenance": dict(provenance),
    }


def _exact_source_identity(source_identity: dict[str, Any]) -> dict[str, str]:
    if not isinstance(source_identity, dict):
        raise ValueError("Confirmation validation source identity must be an object.")
    fields = ("tranche_id", "config_id", "run_id")
    source = {field: str(source_identity.get(field) or "").strip() for field in fields}
    missing = [field for field, value in source.items() if not value]
    if missing:
        raise ValueError(
            "Confirmation validation requires exact source identity fields: "
            + ", ".join(fields)
            + "."
        )
    return source


def _require_expected_cache_hash(
    metadata: dict[str, Any],
    expected_sha256: Any,
    *,
    context: str,
) -> None:
    """Bind a scientific config to an expected cache digest when supplied."""

    if expected_sha256 is None:
        return
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(f"{context} expected SHA-256 is invalid.")
    actual = metadata.get("tokens_sha256")
    if actual != expected_sha256:
        raise ValueError(
            f"{context} hash does not match config: expected {expected_sha256}, got {actual}."
        )


def _confirmation_cache_identity(
    metadata: dict[str, Any],
    *,
    token_count: int,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Confirmation validation cache metadata must be an object.")
    missing = [field for field in VALIDATION_CACHE_IDENTITY_FIELDS if metadata.get(field) is None]
    if missing:
        raise ValueError(
            "Confirmation validation cache is missing identity fields: "
            + ", ".join(missing)
            + "."
        )
    if metadata["partition"] != "confirmation":
        raise ValueError("Saved-checkpoint confirmation validation requires partition confirmation.")
    if metadata["partition_scheme"] != VALIDATION_PARTITION_SCHEME:
        raise ValueError(
            "Confirmation validation partition scheme does not match the repository contract."
        )
    if int(metadata["tokens"]) != int(token_count):
        raise ValueError(
            "Confirmation validation token count does not match cache metadata: "
            f"expected {metadata['tokens']}, got {token_count}."
        )
    block_size = int(metadata["block_size"])
    if block_size <= 0 or token_count < block_size:
        raise ValueError("Confirmation validation cache has no complete block.")
    return {field: metadata[field] for field in VALIDATION_CACHE_IDENTITY_FIELDS}


def _directory_content_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"Saved checkpoint contains no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


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
    available_complete_blocks, excluded_tail_tokens = divmod(len(tokens), block_size)
    weighted_loss = 0.0
    total_sequences = 0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        if eval_batches is None:
            starts = [index * block_size for index in range(available_complete_blocks)]
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
            starts = [index * block_size for index in range(available_complete_blocks)]
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
        "sequences": total_sequences,
        "tokens": total_tokens,
        "available_complete_blocks": available_complete_blocks,
        "excluded_tail_tokens": excluded_tail_tokens,
        "complete_block_coverage": (
            (eval_batches is None or deterministic_batches)
            and total_sequences == available_complete_blocks
        ),
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
