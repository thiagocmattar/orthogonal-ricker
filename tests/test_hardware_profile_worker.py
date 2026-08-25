from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from paper_exp.hardware_profile import GLOBAL_SEQUENCES
from paper_exp.hardware_profile import HardwareProfileRequest
from paper_exp.hardware_profile import SEQUENCE_LENGTH
from paper_exp.hardware_profile import build_profile_work_items
from paper_exp.hardware_profile import reject_scientific_keys
from paper_exp.hardware_profile_worker import EVALUATION_BATCH_SIZE
from paper_exp.hardware_profile_worker import EVALUATION_SEQUENCES
from paper_exp.hardware_profile_worker import MEASURED_FULL_UPDATES
from paper_exp.hardware_profile_worker import PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM
from paper_exp.hardware_profile_worker import PROFILE_ONLY_PRESSURE_METHOD
from paper_exp.hardware_profile_worker import PROFILE_ONLY_PRESSURE_SITES
from paper_exp.hardware_profile_worker import PROFILE_ONLY_PRESSURE_WEIGHT
from paper_exp.hardware_profile_worker import PROFILE_ONLY_SITE_GATE
from paper_exp.hardware_profile_worker import PROFILE_ONLY_STEP_BUDGET
from paper_exp.hardware_profile_worker import PROFILE_ONLY_TOPOLOGY_ID
from paper_exp.hardware_profile_worker import _WorkerDependencies
from paper_exp.hardware_profile_worker import _assert_adamw_optimizer
from paper_exp.hardware_profile_worker import _assert_gradient_clipping_telemetry
from paper_exp.hardware_profile_worker import _assert_random_profile_model
from paper_exp.hardware_profile_worker import _execute_profile_repeat
from paper_exp.hardware_profile_worker import _nvidia_driver_version
from paper_exp.hardware_profile_worker import collect_hardware_profile_identity
from paper_exp.hardware_profile_worker import profile_only_workload_metadata
from paper_exp.hardware_profile_worker import run_hardware_profile_repeat
from paper_exp.hardware_profile_worker import run_selected_candidate_timing
from paper_exp.hardware_profile_worker import time_selected_candidate_checkpoint
from paper_exp.hardware_profile_worker import time_selected_candidate_evaluation
from paper_exp.optimization import GLOBAL_GRADIENT_CLIP_MAX_NORM


class FakeOutOfMemoryError(RuntimeError):
    pass


def test_worker_module_import_keeps_torch_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import paper_exp.hardware_profile_worker; "
                "print('torch' in sys.modules); "
                "print('paper_exp.optimization' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["False", "False"]


class FakeDevice:
    type = "cuda"


class FakeStateTensor:
    def __init__(self, dtype: object) -> None:
        self.dtype = dtype


class FakeParameter:
    def __init__(self, dtype: object) -> None:
        self.dtype = dtype
        self.requires_grad = True


class FakeOptimizer:
    def __init__(self) -> None:
        self.state: dict[FakeParameter, dict[str, FakeStateTensor]] = {}
        self.param_groups = [
            {
                "lr": 1.0e-3,
                "betas": (0.9, 0.95),
                "eps": 1.0e-8,
                "weight_decay": 0.1,
            }
        ]


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.synchronize_calls = 0
        self.reset_calls = 0

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def is_bf16_supported(self) -> bool:
        return True

    def get_device_name(self, _device: Any) -> str:
        return "NVIDIA RTX A6000"

    def get_device_properties(self, _device: Any) -> Any:
        return SimpleNamespace(total_memory=48_000_000_000, uuid="GPU-test-uuid")

    def get_device_capability(self, _device: Any) -> tuple[int, int]:
        return (8, 6)

    def manual_seed_all(self, seed: int) -> None:
        self.events.append(f"cuda_seed:{seed}")

    def synchronize(self, _device: Any) -> None:
        self.synchronize_calls += 1
        self.events.append("synchronize")

    def reset_peak_memory_stats(self, _device: Any) -> None:
        self.reset_calls += 1
        self.events.append("reset_peak")

    def max_memory_allocated(self, _device: Any) -> int:
        return 30_000_000_000

    def max_memory_reserved(self, _device: Any) -> int:
        return 40_000_000_000


class FakeTorch:
    float32 = object()
    bfloat16 = object()

    def __init__(self, events: list[str]) -> None:
        self.cuda = FakeCuda(events)
        self.optim = SimpleNamespace(AdamW=FakeOptimizer)
        self.version = SimpleNamespace(cuda="12.8")
        self.events = events

    def device(self, value: str) -> FakeDevice:
        assert value == "cuda:0"
        return FakeDevice()

    def manual_seed(self, seed: int) -> None:
        self.events.append(f"seed:{seed}")

class FakeModel:
    def __init__(self, torch: FakeTorch) -> None:
        self.config = SimpleNamespace(
            model_type="gpt_neox",
            vocab_size=101,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            max_position_embeddings=SEQUENCE_LENGTH,
        )
        self._parameters = [FakeParameter(torch.float32), FakeParameter(torch.float32)]
        self.training = False
        self.evaluation_calls: list[dict[str, Any]] = []

    def parameters(self) -> list[FakeParameter]:
        return self._parameters

    def train(self, mode: bool = True) -> FakeModel:
        self.training = mode
        return self

    def eval(self) -> FakeModel:
        return self.train(False)


class FakeAutoConfig:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def from_pretrained(self, architecture: str, *, revision: str) -> dict[str, str]:
        self.events.append("fetch_model_config")
        return {"architecture": architecture, "revision": revision}


class FakeCapture:
    def __init__(
        self,
        model: Any,
        sites: list[str],
        *,
        torch: Any,
    ) -> None:
        assert sites == ["h"]
        self.model = model
        self.torch = torch

    def __enter__(self) -> FakeCapture:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _request_and_work_item() -> tuple[HardwareProfileRequest, Any]:
    request = HardwareProfileRequest(
        architecture="EleutherAI/pythia-14m-deduped",
        revision="a" * 40,
        gpu_class="NVIDIA RTX A6000 48GB",
        candidate_microbatches=(2,),
        repeats=2,
    )
    return request, build_profile_work_items(request)[0]


def _fake_dependencies(
    *,
    run_error: BaseException | None = None,
) -> tuple[_WorkerDependencies, FakeTorch, FakeModel, list[dict[str, Any]], list[str]]:
    events: list[str] = []
    torch = FakeTorch(events)
    model = FakeModel(torch)
    optimizer = FakeOptimizer()
    calls: list[dict[str, Any]] = []

    def build_random_model(**kwargs: Any) -> FakeModel:
        assert kwargs["torch"] is torch
        assert kwargs["device"].type == "cuda"
        model_config = kwargs["model_config"]
        assert model_config == {
            "architecture": "EleutherAI/pythia-14m-deduped",
            "revision": "a" * 40,
            "initialization": "random",
            "topology_id": PROFILE_ONLY_TOPOLOGY_ID,
            "site_gate": PROFILE_ONLY_SITE_GATE,
        }
        resolved = kwargs["auto_config"].from_pretrained(
            model_config["architecture"],
            revision=model_config["revision"],
        )
        assert resolved == {
            "architecture": "EleutherAI/pythia-14m-deduped",
            "revision": "a" * 40,
        }
        events.append("build_model")
        return model

    def build_optimizer(**kwargs: Any) -> tuple[FakeOptimizer, dict[str, Any]]:
        events.append("build_optimizer")
        assert kwargs["model"] is model
        assert kwargs["training"] == {
            "optimizer": "adamw",
            "learning_rate": 1.0e-3,
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1.0e-8,
            "weight_decay": 0.1,
        }
        return optimizer, {}

    pressure = SimpleNamespace(
        enabled=True,
        orthogonal=True,
        method=PROFILE_ONLY_PRESSURE_METHOD,
        sites=list(PROFILE_ONLY_PRESSURE_SITES),
        weight=PROFILE_ONLY_PRESSURE_WEIGHT,
        step_budget=PROFILE_ONLY_STEP_BUDGET,
    )

    def pressure_factory(config: dict[str, Any]) -> Any:
        raw = config["activation_pressure"]
        assert raw["method"] == PROFILE_ONLY_PRESSURE_METHOD
        assert raw["sites"] == ["h"]
        assert raw["weight"] == PROFILE_ONLY_PRESSURE_WEIGHT
        assert raw["step_budget"] == PROFILE_ONLY_STEP_BUDGET
        return pressure

    def run_step(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        events.append(f"step:{kwargs['step']}")
        if run_error is not None:
            raise run_error
        if kwargs["step"] == 1:
            optimizer.state = {
                parameter: {
                    "exp_avg": FakeStateTensor(torch.float32),
                    "exp_avg_sq": FakeStateTensor(torch.float32),
                }
                for parameter in model.parameters()
            }
        # These values emulate the real path but must never cross the worker boundary.
        return {
            "task_loss": 12.0,
            "activation/near_zero_mass/k0": 0.5,
            "optimization/adamw_gradient_global_norm_pre_clip": 2.0,
            "optimization/adamw_gradient_global_norm_post_clip": 1.0,
            "optimization/adamw_gradient_clip_max_norm": 1.0,
            "optimization/adamw_gradient_was_clipped": True,
        }

    def evaluate_loss(**kwargs: Any) -> dict[str, Any]:
        model.evaluation_calls.append(kwargs)
        return {"loss": 99.0, "batches": 38, "tokens": 152 * 2_048}

    dependencies = _WorkerDependencies(
        torch=torch,
        np=np,
        auto_config=FakeAutoConfig(events),
        auto_model=object(),
        build_random_model=build_random_model,
        activation_capture=FakeCapture,
        build_adamw_optimizer=build_optimizer,
        run_training_step=run_step,
        pressure_config_factory=pressure_factory,
        model_topology_metadata=lambda _model: {
            "topology_id": "A1-H",
            "active_sites": ["h"],
            "qk_placement": None,
            "site_gate": {"operator": "relu"},
        },
        evaluate_loss=evaluate_loss,
        save_safetensors_model=lambda _model, path: Path(path).write_bytes(b"model"),
    )
    return dependencies, torch, model, calls, events


def test_repeat_uses_real_path_shape_and_synchronized_full_updates() -> None:
    request, work_item = _request_and_work_item()
    dependencies, torch, _model, calls, events = _fake_dependencies()
    times = iter((0.0, 1.0, 10.0, 12.0, 20.0, 23.0))

    result = _execute_profile_repeat(
        request,
        work_item,
        dependencies=dependencies,
        clock=lambda: next(times),
    )

    assert result.fit is True
    assert result.synchronized_seconds == 6.0
    assert result.tokens_per_second == (
        MEASURED_FULL_UPDATES * GLOBAL_SEQUENCES * SEQUENCE_LENGTH / 6.0
    )
    assert result.peak_allocated_bytes == 30_000_000_000
    assert result.peak_reserved_bytes == 40_000_000_000
    assert result.total_vram_bytes == 48_000_000_000
    assert len(calls) == 1 + MEASURED_FULL_UPDATES
    assert [call["step"] for call in calls] == [1, 2, 3, 4]
    for call in calls:
        assert call["block_size"] == SEQUENCE_LENGTH
        assert call["micro_batch_size"] == 2
        assert call["grad_accum"] == 64
        assert call["dtype"] is torch.bfloat16
        assert call["train_tokens"].shape == (GLOBAL_SEQUENCES * SEQUENCE_LENGTH,)
        assert call["schedule_step"].shape == (64, 2)
        np.testing.assert_array_equal(
            call["schedule_step"].reshape(-1),
            np.arange(GLOBAL_SEQUENCES) * SEQUENCE_LENGTH,
        )
    assert torch.cuda.synchronize_calls == 8
    assert torch.cuda.reset_calls == 1
    assert events.index("step:1") < events.index("reset_peak") < events.index("step:2")


def test_repeat_returns_only_operational_fields() -> None:
    request, work_item = _request_and_work_item()
    dependencies, _torch, _model, _calls, _events = _fake_dependencies()
    times = iter((0.0, 1.0, 1.0, 2.0, 2.0, 3.0))

    result = _execute_profile_repeat(
        request,
        work_item,
        dependencies=dependencies,
        clock=lambda: next(times),
    )

    assert set(result.as_dict()) == {
        "microbatch_sequences",
        "repeat_index",
        "fit",
        "error",
        "synchronized_seconds",
        "tokens_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "total_vram_bytes",
    }
    reject_scientific_keys(result.as_dict())
    metadata = profile_only_workload_metadata()
    assert metadata == {
        "scientific_evidence": False,
        "model": {
            "initialization": "random",
            "released_checkpoint_weights_loaded": False,
            "seed": 0,
            "topology_id": "A1-H",
            "site_gate": {"operator": "relu"},
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
            "learning_rate_sentinel": 1.0e-3,
            "global_gradient_clip_max_norm": GLOBAL_GRADIENT_CLIP_MAX_NORM,
            "nonfinite_task_gradient": "raises_before_adamw_step",
        },
        "pressure": {
            "method": "orthogonal_l1",
            "sites": ["h"],
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
            "batches": 38,
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
    assert (
        PROFILE_ONLY_GLOBAL_GRADIENT_CLIP_MAX_NORM
        == GLOBAL_GRADIENT_CLIP_MAX_NORM
    )
    reject_scientific_keys(metadata)


def test_profile_rejects_missing_production_gradient_clipping_telemetry() -> None:
    with pytest.raises(RuntimeError, match="did not prove production gradient clipping"):
        _assert_gradient_clipping_telemetry({"task_loss": 1.0})


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("hidden_dropout", 0.1, "hidden_dropout=0"),
        ("attention_dropout", 0.1, "attention_dropout=0"),
        ("max_position_embeddings", 1_024, "at least 2048"),
    ],
)
def test_profile_model_asserts_zero_dropout_and_2048_capacity(
    attribute: str,
    value: object,
    message: str,
) -> None:
    request, _work_item = _request_and_work_item()
    _dependencies, torch, model, _calls, _events = _fake_dependencies()
    setattr(model.config, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        _assert_random_profile_model(model, request=request, torch=torch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", 2.0e-3),
        ("betas", (0.9, 0.999)),
        ("eps", 1.0e-6),
        ("weight_decay", 0.0),
    ],
)
def test_profile_optimizer_asserts_exact_adamw_group(
    field: str,
    value: object,
) -> None:
    _dependencies, torch, _model, _calls, _events = _fake_dependencies()
    optimizer = FakeOptimizer()
    optimizer.param_groups[0][field] = value

    with pytest.raises(RuntimeError, match="exact AdamW sentinel group"):
        _assert_adamw_optimizer(optimizer, torch)


def test_only_cuda_oom_is_classified_as_non_fit() -> None:
    request, work_item = _request_and_work_item()
    dependencies, _torch, _model, _calls, _events = _fake_dependencies(
        run_error=FakeOutOfMemoryError("out")
    )

    result = _execute_profile_repeat(request, work_item, dependencies=dependencies)

    assert result.fit is False
    assert result.error == "cuda_out_of_memory"
    assert result.synchronized_seconds is None
    assert result.tokens_per_second is None
    assert result.peak_allocated_bytes == 30_000_000_000
    assert result.peak_reserved_bytes == 40_000_000_000
    assert result.total_vram_bytes == 48_000_000_000


def test_non_oom_worker_error_propagates() -> None:
    request, work_item = _request_and_work_item()
    dependencies, _torch, _model, _calls, _events = _fake_dependencies(
        run_error=RuntimeError("not an OOM")
    )

    with pytest.raises(RuntimeError, match="not an OOM"):
        _execute_profile_repeat(request, work_item, dependencies=dependencies)


def test_public_worker_checks_visibility_before_loading_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, work_item = _request_and_work_item()
    loaded = False

    def unexpected_load() -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("dependencies must not load")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker._load_worker_dependencies",
        unexpected_load,
    )
    with pytest.raises(RuntimeError, match="set CUDA_VISIBLE_DEVICES"):
        run_hardware_profile_repeat(request, work_item)
    assert loaded is False


def test_selected_evaluation_is_152_sequences_in_batches_of_four() -> None:
    dependencies, torch, model, _calls, _events = _fake_dependencies()
    model.train(True)
    times = iter((4.0, 9.0))

    elapsed = time_selected_candidate_evaluation(
        model,
        torch=torch,
        np=np,
        device=FakeDevice(),
        dtype=torch.bfloat16,
        evaluate_loss=dependencies.evaluate_loss,
        clock=lambda: next(times),
    )

    assert elapsed == 5.0
    assert len(model.evaluation_calls) == 1
    evaluation = model.evaluation_calls[0]
    assert evaluation["model"] is model
    assert evaluation["tokens"].shape == (EVALUATION_SEQUENCES * SEQUENCE_LENGTH + 1,)
    assert evaluation["block_size"] == SEQUENCE_LENGTH
    assert evaluation["batch_size"] == EVALUATION_BATCH_SIZE
    assert evaluation["eval_batches"] is None
    assert evaluation["deterministic_batches"] is True
    assert model.training is True
    assert torch.cuda.synchronize_calls == 2


def test_selected_checkpoint_is_model_only_hashed_fsynced_and_deleted(
    tmp_path: Path,
) -> None:
    _dependencies, torch, model, _calls, _events = _fake_dependencies()
    payload = b"synthetic safetensors payload"
    saved_paths: list[Path] = []

    def save_model(candidate: Any, path: str) -> None:
        assert candidate is model
        checkpoint = Path(path)
        assert checkpoint.name == "model.safetensors"
        saved_paths.append(checkpoint)
        checkpoint.write_bytes(payload)

    times = iter((7.0, 9.5))
    timing = time_selected_candidate_checkpoint(
        model,
        torch=torch,
        device=FakeDevice(),
        scratch_directory=tmp_path,
        save_safetensors_model=save_model,
        clock=lambda: next(times),
    )

    assert timing.synchronized_seconds == 2.5
    assert timing.sha256 == sha256(payload).hexdigest()
    assert timing.bytes_written == len(payload)
    assert len(saved_paths) == 1
    assert not saved_paths[0].exists()
    assert list(tmp_path.iterdir()) == []
    assert torch.cuda.synchronize_calls == 2


def test_selected_worker_builds_fresh_workload_and_returns_only_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _work_item = _request_and_work_item()
    dependencies, _torch, _model, _calls, events = _fake_dependencies()
    times = iter((1.0, 3.0, 4.0, 7.0, 8.0, 12.0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker._load_worker_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.time.perf_counter",
        lambda: events.append("timer") or next(times),
    )

    result = run_selected_candidate_timing(
        request,
        request.candidates[0],
        checkpoint_scratch=tmp_path,
    )

    assert result.setup_seconds == 2.0
    assert result.validation_seconds == 3.0
    assert result.checkpoint_seconds == 4.0
    assert result.checkpoint_sha256 == sha256(b"model").hexdigest()
    assert result.checkpoint_bytes == len(b"model")
    assert events.count("fetch_model_config") == 1
    assert events.index("fetch_model_config") < events.index("timer")
    assert events.index("timer") < events.index("build_model")
    assert events.index("build_model") < events.index("build_optimizer")
    reject_scientific_keys(result.as_dict())


def test_worker_identity_records_actual_gpu_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _work_item = _request_and_work_item()
    dependencies, _torch, _model, _calls, _events = _fake_dependencies()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker._load_worker_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.collect_git_commit",
        lambda _root: "c" * 40,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.collect_git_dirty",
        lambda _root: False,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.collect_package_versions",
        lambda: {"torch": "test"},
    )
    queried_gpu_uuids: list[str] = []
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker._nvidia_driver_version",
        lambda *, gpu_uuid: queried_gpu_uuids.append(gpu_uuid) or "550.54.15",
    )

    identity = collect_hardware_profile_identity(request)

    assert identity["repo_git_commit"] == "c" * 40
    assert identity["repo_git_dirty"] is False
    assert identity["cuda_visible_devices"] == "2"
    assert identity["nvidia_driver_version"] == "550.54.15"
    assert queried_gpu_uuids == ["GPU-test-uuid"]
    assert identity["gpu"] == {
        "uuid": "GPU-test-uuid",
        "name": "NVIDIA RTX A6000",
        "total_vram_bytes": 48_000_000_000,
        "bf16_supported": True,
        "compute_capability": [8, 6],
    }


def test_worker_identity_rejects_dirty_git_before_loading_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _work_item = _request_and_work_item()
    loaded = False

    def unexpected_load() -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("dependencies must not load")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.collect_git_commit",
        lambda _root: "c" * 40,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.collect_git_dirty",
        lambda _root: True,
    )
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker._load_worker_dependencies",
        unexpected_load,
    )

    with pytest.raises(RuntimeError, match="immutable clean Git checkout"):
        collect_hardware_profile_identity(request)
    assert loaded is False


def test_nvidia_driver_query_is_bound_to_exact_gpu_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> Any:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="550.54.15\n")

    monkeypatch.setattr("paper_exp.hardware_profile_worker.subprocess.run", run)

    assert _nvidia_driver_version(gpu_uuid="GPU-test-uuid") == "550.54.15"
    assert calls == [
        (
            [
                "nvidia-smi",
                "-i",
                "GPU-test-uuid",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            {
                "capture_output": True,
                "check": True,
                "text": True,
                "timeout": 10,
            },
        )
    ]


@pytest.mark.parametrize("stdout", ["", "unknown\n", "550.54\n551.00\n"])
def test_nvidia_driver_query_rejects_malformed_output(
    stdout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paper_exp.hardware_profile_worker.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout),
    )

    with pytest.raises(RuntimeError, match="one valid driver version"):
        _nvidia_driver_version(gpu_uuid="GPU-test-uuid")
