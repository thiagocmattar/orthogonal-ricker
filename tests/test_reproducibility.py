from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import paper_exp.calibration as calibration
from paper_exp.activation_pressure import ActivationPressureConfig
from paper_exp.calibration import _model_parameter_sha256
from paper_exp.reproducibility import build_training_schedule
from paper_exp.reproducibility import validation_document_indices


def test_training_schedule_is_stable_and_seed_specific() -> None:
    kwargs = {
        "token_count": 100_000,
        "block_size": 128,
        "max_steps": 4,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 3,
    }
    first, first_hash = build_training_schedule(np, seed=7, **kwargs)
    repeated, repeated_hash = build_training_schedule(np, seed=7, **kwargs)
    different, different_hash = build_training_schedule(np, seed=8, **kwargs)

    assert np.array_equal(first, repeated)
    assert first_hash == repeated_hash
    assert not np.array_equal(first, different)
    assert first_hash != different_hash


def test_validation_document_partitions_are_stable_disjoint_halves() -> None:
    selection, selection_hash = validation_document_indices(
        np, source_documents=500, partition="selection", seed=20260718
    )
    confirmation, confirmation_hash = validation_document_indices(
        np, source_documents=500, partition="confirmation", seed=20260718
    )

    assert len(selection) == len(confirmation) == 250
    assert set(selection).isdisjoint(set(confirmation))
    assert set(selection) | set(confirmation) == set(range(500))
    assert selection_hash == "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
    assert confirmation_hash == "8953a93f85c80a48d25fcacb7a0fbf44f6d9fd5b54037f92e01c5250f045ad99"
    repeated, repeated_hash = validation_document_indices(
        np, source_documents=500, partition="selection", seed=20260718
    )
    assert np.array_equal(selection, repeated)
    assert selection_hash == repeated_hash


def test_initial_parameter_hash_is_stable_and_parameter_sensitive() -> None:
    torch.manual_seed(3)
    first = torch.nn.Linear(4, 3)
    torch.manual_seed(3)
    repeated = torch.nn.Linear(4, 3)
    torch.manual_seed(4)
    different = torch.nn.Linear(4, 3)

    assert _model_parameter_sha256(first) == _model_parameter_sha256(repeated)
    assert _model_parameter_sha256(first) != _model_parameter_sha256(different)


@pytest.mark.parametrize("method", ["l1_naive", "orthogonal_l1"])
def test_explicit_schedule_reaches_naive_and_orthogonal_batch_sampling(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule_step = np.asarray([[3, 7], [11, 13]], dtype=np.int64)
    sampled_starts: list[list[int]] = []

    def record_sample(*args: object, starts: object = None, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        sampled_starts.append(np.asarray(starts).tolist())
        return torch.ones((2, 4), dtype=torch.long)

    monkeypatch.setattr(calibration, "_sample_batch", record_sample)

    parameter = torch.nn.Parameter(torch.tensor(0.75))

    class Capture:
        def __init__(self) -> None:
            self.activations: dict[str, torch.Tensor] = {}

        def clear(self) -> None:
            self.activations.clear()

    capture = Capture()

    class Model:
        def __call__(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
            del labels
            capture.activations["mlp_hiddens"] = parameter.reshape(1)
            loss = (parameter - 0.25).square() + input_ids.float().sum() * 0.0
            return SimpleNamespace(loss=loss)

    pressure_config = ActivationPressureConfig(
        enabled=True,
        method=method,
        sites=["mlp_hiddens"],
        weight=1.0,
        ricker_c=0.05,
        ricker_sigma=0.05,
        step_budget=0.5,
        eps=1e-12,
        log_thresholds=(0.0,),
    )
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    calibration._run_training_step(
        model=Model(),
        optimizer=optimizer,
        params=[parameter],
        torch=torch,
        np=np,
        train_tokens=np.arange(100, dtype=np.int32),
        block_size=4,
        micro_batch_size=2,
        grad_accum=2,
        device=torch.device("cpu"),
        dtype=None,
        pressure_config=pressure_config,
        activation_capture=capture,
        step=1,
        schedule_step=schedule_step,
    )

    assert sampled_starts == [[3, 7], [11, 13]]


def test_legacy_batch_sampling_keeps_global_numpy_rng_sequence() -> None:
    tokens = np.arange(100, dtype=np.int32)
    np.random.seed(19)
    expected_starts = np.random.randint(0, len(tokens) - 4 - 1, size=3)
    expected = np.stack([tokens[start : start + 4] for start in expected_starts])

    np.random.seed(19)
    actual = calibration._sample_batch(
        torch,
        np,
        tokens,
        block_size=4,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert np.array_equal(actual.numpy(), expected)


def test_campaign_finite_validation_is_deterministic_and_legacy_remains_random() -> None:
    tokens = np.arange(100, dtype=np.int32)

    class RecordingModel:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
            del labels
            self.inputs.append(input_ids.detach().cpu().numpy().copy())
            return SimpleNamespace(loss=torch.tensor(1.0))

    first_campaign = RecordingModel()
    np.random.seed(31)
    calibration._evaluate_loss(
        model=first_campaign,
        torch=torch,
        np=np,
        tokens=tokens,
        block_size=4,
        batch_size=2,
        eval_batches=2,
        device=torch.device("cpu"),
        dtype=None,
        deterministic_batches=True,
    )
    campaign_rng_next = np.random.randint(0, 10_000)

    repeated_campaign = RecordingModel()
    np.random.seed(31)
    expected_rng_next = np.random.randint(0, 10_000)
    np.random.seed(999)
    calibration._evaluate_loss(
        model=repeated_campaign,
        torch=torch,
        np=np,
        tokens=tokens,
        block_size=4,
        batch_size=2,
        eval_batches=2,
        device=torch.device("cpu"),
        dtype=None,
        deterministic_batches=True,
    )

    assert campaign_rng_next == expected_rng_next
    assert all(
        np.array_equal(first, repeated)
        for first, repeated in zip(first_campaign.inputs, repeated_campaign.inputs)
    )
    assert np.array_equal(
        np.concatenate(first_campaign.inputs),
        np.stack([tokens[start : start + 4] for start in (0, 4, 8, 12)]),
    )

    legacy = RecordingModel()
    np.random.seed(23)
    expected_starts = [
        np.random.randint(0, len(tokens) - 4 - 1, size=2)
        for _ in range(2)
    ]
    np.random.seed(23)
    calibration._evaluate_loss(
        model=legacy,
        torch=torch,
        np=np,
        tokens=tokens,
        block_size=4,
        batch_size=2,
        eval_batches=2,
        device=torch.device("cpu"),
        dtype=None,
    )
    expected_legacy = np.concatenate(
        [np.stack([tokens[start : start + 4] for start in starts]) for starts in expected_starts]
    )

    assert np.array_equal(np.concatenate(legacy.inputs), expected_legacy)
