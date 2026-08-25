"""Isolated CUDA worker for infrastructure-only physical-batch profiling.

The caller must start a fresh process and set ``CUDA_VISIBLE_DEVICES`` before
calling :func:`run_hardware_profile_repeat`.  This module intentionally avoids
importing NumPy, Torch, Transformers, or Torch-importing ``paper_exp`` modules
until that function loads its dependencies.  One process profiles one work
item; process isolation and candidate orchestration belong to the caller.

The workload exercises the repository's real random-model, activation-capture,
AdamW, and OL1 training-step path.  Its pressure values are explicit profiling
sentinels, not proposed experiment hyperparameters, and no scientific result
is returned from this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import math
import os
import platform
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from paper_exp.hardware_profile import GLOBAL_SEQUENCES
from paper_exp.hardware_profile import HardwareProfileRequest
from paper_exp.hardware_profile import HardwareProfileWorkItem
from paper_exp.hardware_profile import MicrobatchCandidate
from paper_exp.hardware_profile import ProfileRepeatResult
from paper_exp.hardware_profile import SEQUENCE_LENGTH
from paper_exp.hardware_profile import flat_synthetic_grouping_hash
from paper_exp.utils import collect_git_commit
from paper_exp.utils import collect_git_dirty
from paper_exp.utils import collect_package_versions


MEASURED_FULL_UPDATES = 3
EVALUATION_SEQUENCES = 152
EVALUATION_BATCH_SIZE = 4

PROFILE_ONLY_SEED = 0
PROFILE_ONLY_LEARNING_RATE = 1.0e-3
PROFILE_ONLY_PRESSURE_WEIGHT = 1.0
PROFILE_ONLY_STEP_BUDGET = 1.0
PROFILE_ONLY_PRESSURE_EPS = 1.0e-12
PROFILE_ONLY_LOG_THRESHOLDS = (0.0,)
PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM = 1.0

PROFILE_ONLY_TOPOLOGY_ID = "A1-H"
PROFILE_ONLY_SITE_GATE = {"operator": "relu"}
PROFILE_ONLY_PRESSURE_METHOD = "orthogonal_l1"
PROFILE_ONLY_PRESSURE_SITES = ("h",)

_OOM_ERROR = "cuda_out_of_memory"
_GPU_VENDOR_TOKENS = frozenset(
    {"NVIDIA", "GEFORCE", "TESLA", "GPU", "GENERATION"}
)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_NVIDIA_DRIVER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


@dataclass(frozen=True)
class CheckpointTiming:
    """Operational result of one disposable model-only checkpoint write."""

    synchronized_seconds: float
    sha256: str
    bytes_written: int


@dataclass(frozen=True)
class SelectedCandidateTiming:
    """Operational timings from a fresh selected-candidate worker."""

    setup_seconds: float
    validation_seconds: float
    checkpoint_seconds: float
    checkpoint_sha256: str
    checkpoint_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "setup_seconds": self.setup_seconds,
            "validation_seconds": self.validation_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
        }


@dataclass(frozen=True)
class _WorkerDependencies:
    torch: Any
    np: Any
    auto_config: Any
    auto_model: Any
    build_random_model: Callable[..., Any]
    activation_capture: Any
    build_adamw_optimizer: Callable[..., Any]
    run_training_step: Callable[..., dict[str, Any]]
    pressure_config_factory: Callable[[dict[str, Any]], Any]
    model_topology_metadata: Callable[[Any], dict[str, Any]]
    evaluate_loss: Callable[..., dict[str, Any]]
    save_safetensors_model: Callable[[Any, str], None]


@dataclass(frozen=True)
class _PrefetchedAutoConfig:
    """Network-free config source used only inside measured setup."""

    architecture: str
    revision: str
    config: Any

    def from_pretrained(self, architecture: str, *, revision: str) -> Any:
        if architecture != self.architecture or revision != self.revision:
            raise RuntimeError("Measured setup requested an unprefetched model config.")
        return deepcopy(self.config)


def profile_only_workload_metadata() -> dict[str, object]:
    """Describe the fixed non-evidence workload without any measured value."""

    return {
        "scientific_evidence": False,
        "model": {
            "initialization": "random",
            "released_checkpoint_weights_loaded": False,
            "seed": PROFILE_ONLY_SEED,
            "topology_id": PROFILE_ONLY_TOPOLOGY_ID,
            "site_gate": dict(PROFILE_ONLY_SITE_GATE),
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "minimum_position_embeddings": SEQUENCE_LENGTH,
        },
        "computation": {
            "autocast_dtype": "bfloat16",
            "parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
        },
        "optimizer": {
            "implementation": "torch.optim.AdamW",
            "betas": [0.9, 0.95],
            "epsilon": 1.0e-8,
            "weight_decay": 0.1,
            "learning_rate_sentinel": PROFILE_ONLY_LEARNING_RATE,
            "global_gradient_clip_max_norm": (
                PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM
            ),
            "nonfinite_task_gradient": "raises_before_adamw_step",
        },
        "pressure": {
            "method": PROFILE_ONLY_PRESSURE_METHOD,
            "sites": list(PROFILE_ONLY_PRESSURE_SITES),
            "weight_sentinel": PROFILE_ONLY_PRESSURE_WEIGHT,
            "step_budget_sentinel": PROFILE_ONLY_STEP_BUDGET,
            "adamw_moments": "clipped_task_gradient_only",
            "l1_pressure_gradient_globally_clipped": False,
            "ordering": [
                "accumulate_task_and_l1_pressure_gradients_separately",
                "globally_clip_task_gradient",
                "clone_clipped_task_gradient_for_ol1_geometry",
                "adamw_step_with_clipped_task_gradient",
                "post_adamw_ol1_correction_with_unclipped_l1_pressure_gradient",
            ],
        },
        "candidate_update": {
            "sequence_length": SEQUENCE_LENGTH,
            "global_sequences": GLOBAL_SEQUENCES,
            "global_input_tokens": GLOBAL_SEQUENCES * SEQUENCE_LENGTH,
            "decomposition": (
                "microbatch_sequences * gradient_accumulation_steps = 128"
            ),
            "matched_flat_synthetic_grouping": True,
            "cold_full_updates": 1,
            "measured_full_updates": MEASURED_FULL_UPDATES,
            "throughput_timer_includes": [
                "production_training_step_forward_backward_and_optimizer_update"
            ],
            "throughput_timer_excludes": [
                "model_and_optimizer_setup",
                "outer_learning_rate_schedule_update",
                "event_and_artifact_serialization",
                "validation",
                "post_training_diagnostics",
                "checkpoint_write",
            ],
            "valid_as_end_to_end_etc": False,
        },
        "evaluation": {
            "model_state": "fresh_random_initialization",
            "first_model_forward_in_process": True,
            "synthetic_sequences": EVALUATION_SEQUENCES,
            "batch_sequences": EVALUATION_BATCH_SIZE,
            "batches": EVALUATION_SEQUENCES // EVALUATION_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "input_tokens": EVALUATION_SEQUENCES * SEQUENCE_LENGTH,
            "ordering": "fixed_complete_sequence_order",
            "result_values_discarded": True,
        },
        "checkpoint": {
            "model_state": "fresh_random_initialization",
            "scope": "model_only",
            "format": "safetensors",
            "content_hash": "sha256",
            "operations_in_timing": [
                "model_only_safetensors_write",
                "sha256_full_file_read",
                "file_fsync",
                "checkpoint_file_delete",
            ],
            "temporary_directory_deleted_after_timing": True,
        },
        "setup_timing": {
            "excluded_before_timer": [
                "lazy_dependency_imports_and_loading",
                "cuda_validation_and_context_initialization",
                "pinned_model_config_cache_lookup_or_download",
            ],
            "included": [
                "random_model_construction_and_gpu_transfer",
                "adamw_construction",
                "pressure_configuration",
                "synthetic_grouping_construction",
            ],
            "not_part_of_profile_workload": [
                "released_checkpoint_weight_loading_or_download",
                "dataset_or_tokenizer_cache_loading_or_download",
            ],
        },
    }


def run_hardware_profile_repeat(
    request: HardwareProfileRequest,
    work_item: HardwareProfileWorkItem,
) -> ProfileRepeatResult:
    """Run one isolated CUDA repeat and return operational measurements only."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        raise RuntimeError(
            "The fresh worker process must set CUDA_VISIBLE_DEVICES before loading CUDA."
        )
    dependencies = _load_worker_dependencies()
    return _execute_profile_repeat(request, work_item, dependencies=dependencies)


def collect_hardware_profile_identity(
    request: HardwareProfileRequest,
) -> dict[str, object]:
    """Return stable environment and actual visible-GPU identity."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        raise RuntimeError(
            "The fresh worker process must set CUDA_VISIBLE_DEVICES before loading CUDA."
        )
    repo_root = Path.cwd()
    repo_git_commit = collect_git_commit(repo_root)
    repo_git_dirty = collect_git_dirty(repo_root)
    if (
        not isinstance(repo_git_commit, str)
        or _GIT_COMMIT_RE.fullmatch(repo_git_commit) is None
        or repo_git_dirty is not False
    ):
        raise RuntimeError(
            "Hardware profiling requires an immutable clean Git checkout."
        )

    dependencies = _load_worker_dependencies()
    torch = dependencies.torch
    device, total_vram_bytes = _validated_cuda_device(torch, request.gpu_class)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    gpu_uuid = _gpu_uuid(properties, visible_devices=visible_devices)
    nvidia_driver_version = _nvidia_driver_version(gpu_uuid=gpu_uuid)
    return {
        "repo_git_commit": repo_git_commit,
        "repo_git_dirty": repo_git_dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": collect_package_versions(),
        "cuda_runtime": str(getattr(torch.version, "cuda", None)),
        "nvidia_driver_version": nvidia_driver_version,
        "cuda_visible_devices": visible_devices,
        "gpu": {
            "uuid": gpu_uuid,
            "name": str(torch.cuda.get_device_name(device)),
            "total_vram_bytes": total_vram_bytes,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "compute_capability": [int(capability[0]), int(capability[1])],
        },
    }


def run_selected_candidate_timing(
    request: HardwareProfileRequest,
    candidate: MicrobatchCandidate,
    *,
    checkpoint_scratch: str | Path,
) -> SelectedCandidateTiming:
    """Time fresh setup, fixed validation, and a disposable checkpoint."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        raise RuntimeError(
            "The fresh worker process must set CUDA_VISIBLE_DEVICES before loading CUDA."
        )
    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    if not isinstance(candidate, MicrobatchCandidate):
        raise TypeError("candidate must be a MicrobatchCandidate.")
    if candidate.microbatch_sequences not in request.candidate_microbatches:
        raise ValueError("Selected candidate is not present in the request.")

    dependencies = _load_worker_dependencies()
    torch = dependencies.torch
    device, _total_vram_bytes = _validated_cuda_device(torch, request.gpu_class)
    prefetched_auto_config = _prefetch_auto_config(dependencies.auto_config, request)
    _seed_profile_workload(torch)
    torch.cuda.synchronize(device)
    setup_started = time.perf_counter()
    model = dependencies.build_random_model(
        torch=torch,
        auto_config=prefetched_auto_config,
        auto_model=dependencies.auto_model,
        model_config=_profile_model_config(request),
        device=device,
    )
    _assert_random_profile_model(model, request=request, torch=torch)
    _assert_profile_topology(dependencies.model_topology_metadata(model))
    optimizer, _optimizer_config = dependencies.build_adamw_optimizer(
        torch=torch,
        model=model,
        training=_profile_training_config(),
    )
    _assert_adamw_optimizer(optimizer, torch)
    pressure_config = dependencies.pressure_config_factory(
        {"activation_pressure": _profile_pressure_config()}
    )
    _assert_profile_pressure(pressure_config)
    selected_item = HardwareProfileWorkItem(
        microbatch_sequences=candidate.microbatch_sequences,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        repeat_index=1,
        synthetic_grouping_hash=flat_synthetic_grouping_hash(
            request,
            microbatch_sequences=candidate.microbatch_sequences,
        ),
    )
    _synthetic_training_grouping(
        dependencies.np,
        vocab_size=_model_vocab_size(model),
        work_item=selected_item,
    )
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    if setup_seconds <= 0.0:
        raise RuntimeError("CUDA-synchronized setup time must be positive.")

    validation_seconds = time_selected_candidate_evaluation(
        model,
        torch=torch,
        np=dependencies.np,
        device=device,
        dtype=torch.bfloat16,
        evaluate_loss=dependencies.evaluate_loss,
    )
    checkpoint = time_selected_candidate_checkpoint(
        model,
        torch=torch,
        device=device,
        scratch_directory=checkpoint_scratch,
        save_safetensors_model=dependencies.save_safetensors_model,
    )
    return SelectedCandidateTiming(
        setup_seconds=setup_seconds,
        validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint.synchronized_seconds,
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_bytes=checkpoint.bytes_written,
    )


def _execute_profile_repeat(
    request: HardwareProfileRequest,
    work_item: HardwareProfileWorkItem,
    *,
    dependencies: _WorkerDependencies,
    clock: Callable[[], float] | None = None,
) -> ProfileRepeatResult:
    """Dependency-injected implementation used by the isolated worker and tests."""

    _validate_work_item(request, work_item)
    torch = dependencies.torch
    device: Any | None = None
    total_vram_bytes: int | None = None
    try:
        device, total_vram_bytes = _validated_cuda_device(torch, request.gpu_class)
        _seed_profile_workload(torch)
        model = dependencies.build_random_model(
            torch=torch,
            auto_config=dependencies.auto_config,
            auto_model=dependencies.auto_model,
            model_config=_profile_model_config(request),
            device=device,
        )
        _assert_random_profile_model(model, request=request, torch=torch)
        topology = dependencies.model_topology_metadata(model)
        _assert_profile_topology(topology)
        model.train()

        optimizer, _optimizer_config = dependencies.build_adamw_optimizer(
            torch=torch,
            model=model,
            training=_profile_training_config(),
        )
        _assert_adamw_optimizer(optimizer, torch)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError("The profiling model has no trainable parameters.")

        pressure_config = dependencies.pressure_config_factory(
            {"activation_pressure": _profile_pressure_config()}
        )
        _assert_profile_pressure(pressure_config)
        tokens, schedule_step = _synthetic_training_grouping(
            dependencies.np,
            vocab_size=_model_vocab_size(model),
            work_item=work_item,
        )

        step_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "params": params,
            "torch": torch,
            "np": dependencies.np,
            "train_tokens": tokens,
            "block_size": SEQUENCE_LENGTH,
            "micro_batch_size": work_item.microbatch_sequences,
            "grad_accum": work_item.gradient_accumulation_steps,
            "device": device,
            "dtype": torch.bfloat16,
            "pressure_config": pressure_config,
            "schedule_step": schedule_step,
        }

        with dependencies.activation_capture(
            model,
            list(PROFILE_ONLY_PRESSURE_SITES),
            torch=torch,
        ) as capture:
            torch.cuda.synchronize(device)
            cold_step_result = dependencies.run_training_step(
                **step_kwargs,
                activation_capture=capture,
                step=1,
            )
            _assert_gradient_clipping_telemetry(cold_step_result)
            torch.cuda.synchronize(device)
            _assert_adamw_state_fp32(optimizer, torch)
            torch.cuda.reset_peak_memory_stats(device)

            timer = time.perf_counter if clock is None else clock
            synchronized_seconds = 0.0
            for step in range(2, 2 + MEASURED_FULL_UPDATES):
                torch.cuda.synchronize(device)
                started = timer()
                measured_step_result = dependencies.run_training_step(
                    **step_kwargs,
                    activation_capture=capture,
                    step=step,
                )
                _assert_gradient_clipping_telemetry(measured_step_result)
                torch.cuda.synchronize(device)
                synchronized_seconds += timer() - started

        if synchronized_seconds <= 0.0:
            raise RuntimeError("CUDA-synchronized profiling time must be positive.")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        measured_tokens = (
            MEASURED_FULL_UPDATES * GLOBAL_SEQUENCES * SEQUENCE_LENGTH
        )
        return ProfileRepeatResult(
            microbatch_sequences=work_item.microbatch_sequences,
            repeat_index=work_item.repeat_index,
            fit=True,
            error=None,
            synchronized_seconds=synchronized_seconds,
            tokens_per_second=measured_tokens / synchronized_seconds,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            total_vram_bytes=total_vram_bytes,
        )
    except torch.cuda.OutOfMemoryError:
        if device is None or total_vram_bytes is None:
            memory_values: tuple[int | None, int | None, int | None] = (
                None,
                None,
                None,
            )
        else:
            memory_values = (
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
                total_vram_bytes,
            )
        return ProfileRepeatResult(
            microbatch_sequences=work_item.microbatch_sequences,
            repeat_index=work_item.repeat_index,
            fit=False,
            error=_OOM_ERROR,
            synchronized_seconds=None,
            tokens_per_second=None,
            peak_allocated_bytes=memory_values[0],
            peak_reserved_bytes=memory_values[1],
            total_vram_bytes=memory_values[2],
        )


def time_selected_candidate_evaluation(
    model: Any,
    *,
    torch: Any,
    np: Any,
    device: Any,
    dtype: Any,
    evaluate_loss: Callable[..., dict[str, Any]],
    clock: Callable[[], float] | None = None,
) -> float:
    """Time exactly 152 synthetic sequences at batch size four.

    The forward pass computes the normal causal-LM objective to preserve the
    production evaluation shape, but the value is deliberately neither read
    nor returned.
    """

    if dtype != torch.bfloat16:
        raise RuntimeError("Selected-candidate evaluation requires BF16 autocast.")
    _assert_model_parameters_fp32(model, torch)
    token_matrix = _synthetic_token_matrix(
        np,
        sequence_count=EVALUATION_SEQUENCES,
        vocab_size=_model_vocab_size(model),
    )
    # The production evaluator requires one look-ahead token when deriving
    # its complete-block count. The appended token is never part of a batch.
    tokens = np.concatenate((token_matrix.reshape(-1), token_matrix.reshape(-1)[:1]))
    timer = time.perf_counter if clock is None else clock
    torch.cuda.synchronize(device)
    started = timer()
    # The production evaluator computes its ordinary return mapping. It is
    # intentionally discarded so no scientific value crosses this boundary.
    evaluate_loss(
        model=model,
        torch=torch,
        np=np,
        tokens=tokens,
        block_size=SEQUENCE_LENGTH,
        batch_size=EVALUATION_BATCH_SIZE,
        eval_batches=None,
        device=device,
        dtype=dtype,
        deterministic_batches=True,
    )
    torch.cuda.synchronize(device)
    elapsed = timer() - started
    if elapsed <= 0.0:
        raise RuntimeError("CUDA-synchronized evaluation time must be positive.")
    return elapsed


def time_selected_candidate_checkpoint(
    model: Any,
    *,
    torch: Any,
    device: Any,
    scratch_directory: str | Path,
    save_safetensors_model: Callable[[Any, str], None],
    clock: Callable[[], float] | None = None,
) -> CheckpointTiming:
    """Time a disposable model-only safetensors write/hash/fsync/delete cycle."""

    _assert_model_parameters_fp32(model, torch)
    scratch = Path(scratch_directory)
    if not scratch.is_dir():
        raise FileNotFoundError(f"Checkpoint scratch directory does not exist: {scratch}")
    timer = time.perf_counter if clock is None else clock
    with tempfile.TemporaryDirectory(prefix="hardware-profile-", dir=scratch) as temp_dir:
        checkpoint = Path(temp_dir) / "model.safetensors"
        torch.cuda.synchronize(device)
        started = timer()
        save_safetensors_model(model, str(checkpoint))
        digest, byte_count = _hash_and_fsync(checkpoint)
        checkpoint.unlink()
        torch.cuda.synchronize(device)
        elapsed = timer() - started
    if elapsed <= 0.0:
        raise RuntimeError("Synchronized checkpoint time must be positive.")
    return CheckpointTiming(
        synchronized_seconds=elapsed,
        sha256=digest,
        bytes_written=byte_count,
    )


def _prefetch_auto_config(
    auto_config: Any,
    request: HardwareProfileRequest,
) -> _PrefetchedAutoConfig:
    """Resolve the pinned architecture config before measured setup."""

    loader = getattr(auto_config, "from_pretrained", None)
    if not callable(loader):
        raise RuntimeError("The model config source has no from_pretrained loader.")
    config = loader(request.architecture, revision=request.revision)
    if config is None:
        raise RuntimeError("The pinned model config source returned no config.")
    return _PrefetchedAutoConfig(
        architecture=request.architecture,
        revision=request.revision,
        config=config,
    )


def _load_worker_dependencies() -> _WorkerDependencies:
    """Load CUDA-sensitive dependencies only inside the fresh worker call."""

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_model
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM

        from paper_exp.activation_pressure import activation_pressure_config
        from paper_exp.activations import ActivationCapture
        from paper_exp.modeling import _build_random_model
        from paper_exp.modeling import model_topology_metadata
        from paper_exp.optimization import GLOBAL_GRADIENT_CLIP_MAX_NORM
        from paper_exp.optimization import _build_adamw_optimizer
        from paper_exp.optimization import _run_training_step
        from paper_exp.training import _evaluate_loss
    except ImportError as error:
        raise RuntimeError(
            "Hardware profiling requires numpy, torch, transformers, and safetensors."
        ) from error
    if GLOBAL_GRADIENT_CLIP_MAX_NORM != PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM:
        raise RuntimeError(
            "The production gradient-clipping norm differs from the profiling contract."
        )
    return _WorkerDependencies(
        torch=torch,
        np=np,
        auto_config=AutoConfig,
        auto_model=AutoModelForCausalLM,
        build_random_model=_build_random_model,
        activation_capture=ActivationCapture,
        build_adamw_optimizer=_build_adamw_optimizer,
        run_training_step=_run_training_step,
        pressure_config_factory=activation_pressure_config,
        model_topology_metadata=model_topology_metadata,
        evaluate_loss=_evaluate_loss,
        save_safetensors_model=save_model,
    )


def _validate_work_item(
    request: HardwareProfileRequest,
    work_item: HardwareProfileWorkItem,
) -> None:
    if not isinstance(request, HardwareProfileRequest):
        raise TypeError("request must be a HardwareProfileRequest.")
    if not isinstance(work_item, HardwareProfileWorkItem):
        raise TypeError("work_item must be a HardwareProfileWorkItem.")
    if request.sequence_length != SEQUENCE_LENGTH:
        raise ValueError(f"Profiling sequence length must be {SEQUENCE_LENGTH}.")
    if request.global_sequences != GLOBAL_SEQUENCES:
        raise ValueError(f"Profiling global sequence count must be {GLOBAL_SEQUENCES}.")
    if work_item.microbatch_sequences not in request.candidate_microbatches:
        raise ValueError("Work-item microbatch is not a candidate in the request.")
    if work_item.repeat_index > request.repeats:
        raise ValueError("Work-item repeat index exceeds the requested repeats.")
    expected_hash = flat_synthetic_grouping_hash(
        request,
        microbatch_sequences=work_item.microbatch_sequences,
    )
    if work_item.synthetic_grouping_hash != expected_hash:
        raise ValueError("Work-item synthetic grouping hash does not match the request.")


def _validated_cuda_device(torch: Any, requested_gpu_class: str) -> tuple[Any, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Hardware profiling requires CUDA.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Hardware profiling requires exactly one visible CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible CUDA GPU must support BF16.")
    device = torch.device("cuda:0")
    actual_gpu_class = str(torch.cuda.get_device_name(device))
    total_vram = int(torch.cuda.get_device_properties(device).total_memory)
    if total_vram <= 0:
        raise RuntimeError("The visible CUDA GPU reports no VRAM.")
    if _canonical_gpu_class(actual_gpu_class) != _canonical_gpu_class(
        requested_gpu_class
    ):
        raise RuntimeError(
            "Visible GPU class does not match the request: "
            f"requested {requested_gpu_class!r}, found {actual_gpu_class!r}."
        )
    requested_vram_gb = _gpu_class_vram_gb(requested_gpu_class)
    if requested_vram_gb is not None:
        ratio = total_vram / (requested_vram_gb * 1_000_000_000)
        if not 0.9 <= ratio <= 1.15:
            raise RuntimeError(
                "Visible GPU VRAM does not match the requested class: "
                f"requested {requested_vram_gb} GB, found {total_vram} bytes."
            )
    return device, total_vram


def _canonical_gpu_class(value: str) -> tuple[str, ...]:
    without_capacity = re.sub(r"\b\d+\s*GB\b", " ", value.upper())
    tokens = re.findall(r"[A-Z0-9]+", without_capacity)
    return tuple(
        token
        for token in tokens
        if token not in _GPU_VENDOR_TOKENS
    )


def _gpu_class_vram_gb(value: str) -> int | None:
    matches = re.findall(r"(?:^|\W)(\d+)\s*GB(?:\W|$)", value.upper())
    if len(set(matches)) > 1:
        raise ValueError(f"GPU class contains ambiguous VRAM capacities: {value!r}.")
    return None if not matches else int(matches[0])


def _gpu_uuid(properties: Any, *, visible_devices: str) -> str:
    value = getattr(properties, "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not visible_devices.isdigit():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain one numeric device index.")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                visible_devices,
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Unable to resolve the visible GPU UUID.") from error
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("The visible GPU did not report exactly one UUID.")
    return lines[0]


def _nvidia_driver_version(*, gpu_uuid: str) -> str:
    """Query the local driver for the exact physical GPU UUID."""

    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise RuntimeError("A GPU UUID is required to resolve the NVIDIA driver.")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Unable to resolve the local NVIDIA driver version.") from error
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or _NVIDIA_DRIVER_RE.fullmatch(lines[0]) is None:
        raise RuntimeError("The visible GPU did not report one valid driver version.")
    return lines[0]


def _seed_profile_workload(torch: Any) -> None:
    torch.manual_seed(PROFILE_ONLY_SEED)
    torch.cuda.manual_seed_all(PROFILE_ONLY_SEED)


def _profile_model_config(request: HardwareProfileRequest) -> dict[str, object]:
    return {
        "architecture": request.architecture,
        "revision": request.revision,
        "initialization": "random",
        "topology_id": PROFILE_ONLY_TOPOLOGY_ID,
        "site_gate": dict(PROFILE_ONLY_SITE_GATE),
    }


def _profile_training_config() -> dict[str, object]:
    return {
        "optimizer": "adamw",
        "learning_rate": PROFILE_ONLY_LEARNING_RATE,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1.0e-8,
        "weight_decay": 0.1,
    }


def _profile_pressure_config() -> dict[str, object]:
    return {
        "enabled": True,
        "method": PROFILE_ONLY_PRESSURE_METHOD,
        "sites": list(PROFILE_ONLY_PRESSURE_SITES),
        "weight": PROFILE_ONLY_PRESSURE_WEIGHT,
        "step_budget": PROFILE_ONLY_STEP_BUDGET,
        "eps": PROFILE_ONLY_PRESSURE_EPS,
        "log_thresholds": list(PROFILE_ONLY_LOG_THRESHOLDS),
    }


def _assert_random_profile_model(
    model: Any,
    *,
    request: HardwareProfileRequest,
    torch: Any,
) -> None:
    if "pythia" not in request.architecture.casefold():
        raise ValueError("Hardware profiling supports pinned Pythia architectures only.")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "gpt_neox":
        raise RuntimeError("The requested Pythia architecture did not build GPT-NeoX.")
    for name in ("hidden_dropout", "attention_dropout"):
        value = getattr(config, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0.0:
            raise RuntimeError(f"Profiling requires {name}=0.")
    positions = getattr(config, "max_position_embeddings", None)
    if isinstance(positions, bool) or not isinstance(positions, int) or positions < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Profiling requires at least {SEQUENCE_LENGTH} position embeddings."
        )
    _assert_model_parameters_fp32(model, torch)


def _assert_model_parameters_fp32(model: Any, torch: Any) -> None:
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("The profiling model has no parameters.")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Profiling requires every model parameter to remain FP32.")


def _assert_profile_topology(topology: dict[str, Any]) -> None:
    expected = {
        "topology_id": PROFILE_ONLY_TOPOLOGY_ID,
        "active_sites": list(PROFILE_ONLY_PRESSURE_SITES),
        "qk_placement": None,
        "site_gate": dict(PROFILE_ONLY_SITE_GATE),
    }
    if topology != expected:
        raise RuntimeError(
            f"Profiling requires the exact A1-H/ReLU topology; found {topology!r}."
        )


def _assert_profile_pressure(pressure_config: Any) -> None:
    if (
        not pressure_config.enabled
        or not pressure_config.orthogonal
        or pressure_config.method != PROFILE_ONLY_PRESSURE_METHOD
        or tuple(pressure_config.sites) != PROFILE_ONLY_PRESSURE_SITES
        or pressure_config.weight != PROFILE_ONLY_PRESSURE_WEIGHT
        or pressure_config.step_budget != PROFILE_ONLY_STEP_BUDGET
    ):
        raise RuntimeError("The profiling worker did not realize its OL1 sentinel workload.")


def _assert_gradient_clipping_telemetry(step_result: dict[str, Any]) -> None:
    """Require every profiled update to traverse production clipping."""

    if not isinstance(step_result, dict):
        raise RuntimeError("The profiling training step did not return telemetry.")
    required = {
        "optimization/adamw_gradient_global_norm_pre_clip",
        "optimization/adamw_gradient_global_norm_post_clip",
        "optimization/adamw_gradient_clip_max_norm",
        "optimization/adamw_gradient_was_clipped",
    }
    missing = sorted(required - set(step_result))
    if missing:
        raise RuntimeError(
            "The profiling update did not prove production gradient clipping; "
            f"missing {', '.join(missing)}."
        )
    pre_clip = step_result["optimization/adamw_gradient_global_norm_pre_clip"]
    post_clip = step_result["optimization/adamw_gradient_global_norm_post_clip"]
    max_norm = step_result["optimization/adamw_gradient_clip_max_norm"]
    was_clipped = step_result["optimization/adamw_gradient_was_clipped"]
    if (
        isinstance(pre_clip, bool)
        or not isinstance(pre_clip, (int, float))
        or not math.isfinite(float(pre_clip))
        or float(pre_clip) < 0.0
        or isinstance(post_clip, bool)
        or not isinstance(post_clip, (int, float))
        or not math.isfinite(float(post_clip))
        or float(post_clip) < 0.0
    ):
        raise RuntimeError("Profiling gradient clipping norms must be finite and nonnegative.")
    if max_norm != PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM:
        raise RuntimeError("The profiling update used the wrong global gradient clip norm.")
    if not isinstance(was_clipped, bool) or was_clipped != (
        float(pre_clip) > PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM
    ):
        raise RuntimeError("Profiling gradient clipping decision telemetry is inconsistent.")
    if float(post_clip) > PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM + 1.0e-6:
        raise RuntimeError("The profiling update exceeded the global gradient clip norm.")
    expected_post_clip = min(
        float(pre_clip),
        PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM,
    )
    if not math.isclose(
        float(post_clip),
        expected_post_clip,
        rel_tol=1.0e-4,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError("Profiling gradient clipping norm telemetry is inconsistent.")


def _assert_adamw_optimizer(optimizer: Any, torch: Any) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise RuntimeError("Hardware profiling requires PyTorch AdamW.")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise RuntimeError("Hardware profiling requires one AdamW parameter group.")
    group = groups[0]
    expected = {
        "lr": PROFILE_ONLY_LEARNING_RATE,
        "betas": (0.9, 0.95),
        "eps": 1.0e-8,
        "weight_decay": 0.1,
    }
    betas = group.get("betas")
    actual = {
        "lr": group.get("lr"),
        "betas": tuple(betas) if isinstance(betas, (list, tuple)) else betas,
        "eps": group.get("eps"),
        "weight_decay": group.get("weight_decay"),
    }
    if actual != expected:
        raise RuntimeError(
            f"Hardware profiling requires the exact AdamW sentinel group; found {actual!r}."
        )


def _assert_adamw_state_fp32(optimizer: Any, torch: Any) -> None:
    if not optimizer.state:
        raise RuntimeError("AdamW state was not initialized by the cold full update.")
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if value is None or value.dtype != torch.float32:
                raise RuntimeError(f"AdamW {name} state must be FP32.")
        for name, value in state.items():
            is_floating_point = getattr(value, "is_floating_point", None)
            if (
                callable(is_floating_point)
                and is_floating_point()
                and value.dtype != torch.float32
            ):
                raise RuntimeError(f"Floating AdamW {name} state must be FP32.")


def _model_vocab_size(model: Any) -> int:
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 2:
        raise RuntimeError("The profiling model must expose a valid vocabulary size.")
    return vocab_size


def _synthetic_training_grouping(
    np: Any,
    *,
    vocab_size: int,
    work_item: HardwareProfileWorkItem,
) -> tuple[Any, Any]:
    matrix = _synthetic_token_matrix(
        np,
        sequence_count=GLOBAL_SEQUENCES,
        vocab_size=vocab_size,
    )
    starts = np.arange(GLOBAL_SEQUENCES, dtype=np.int64) * SEQUENCE_LENGTH
    schedule_step = starts.reshape(
        work_item.gradient_accumulation_steps,
        work_item.microbatch_sequences,
    )
    return matrix.reshape(-1), schedule_step


def _synthetic_token_matrix(
    np: Any,
    *,
    sequence_count: int,
    vocab_size: int,
) -> Any:
    flat = np.arange(sequence_count * SEQUENCE_LENGTH, dtype=np.int64)
    # A fixed affine map avoids a repeated ascending run at every sequence
    # boundary while remaining independent of candidate reshape boundaries.
    flat = (flat * 1_103_515_245 + 12_345) % vocab_size
    return flat.astype(np.int32, copy=False).reshape(sequence_count, SEQUENCE_LENGTH)


def _hash_and_fsync(path: Path) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("r+b", buffering=0) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        os.fsync(handle.fileno())
    if byte_count <= 0:
        raise RuntimeError("The model-only safetensors checkpoint is empty.")
    return digest.hexdigest(), byte_count
