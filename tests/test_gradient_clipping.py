from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from paper_exp import optimization
from paper_exp.activation_pressure import ActivationPressureConfig


class _Capture:
    def __init__(self) -> None:
        self.activations: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.activations.clear()


class _LinearGradientModel:
    def __init__(self, parameter: torch.nn.Parameter, capture: _Capture) -> None:
        self.parameter = parameter
        self.capture = capture

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del labels
        self.capture.activations["h"] = self.parameter
        task_loss = (self.parameter * self.parameter.new_tensor([3.0, 4.0])).sum()
        task_loss = task_loss + input_ids.float().sum() * 0.0
        return SimpleNamespace(loss=task_loss)


class _NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (torch.full((2,), float("inf"), device=grad_output.device),)


class _NonfiniteGradientModel(_LinearGradientModel):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del labels
        self.capture.activations["h"] = self.parameter
        task_loss = _NonfiniteBackward.apply(self.parameter)
        task_loss = task_loss + input_ids.float().sum() * 0.0
        return SimpleNamespace(loss=task_loss)


class _RecordingAdamW(torch.optim.AdamW):
    def __init__(self, params: list[torch.nn.Parameter], events: list[str]) -> None:
        super().__init__(params, lr=1.0e-2, weight_decay=0.0)
        self.events = events
        self.gradient_snapshots: list[list[torch.Tensor | None]] = []

    def step(self, closure: Any = None) -> Any:
        self.events.append("adamw_step")
        self.gradient_snapshots.append(
            [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for group in self.param_groups
                for parameter in group["params"]
            ]
        )
        return super().step(closure=closure)


def _pressure_config(method: str) -> ActivationPressureConfig:
    return ActivationPressureConfig(
        enabled=True,
        method=method,
        sites=["h"],
        weight=0.0 if method == "none" else 2.0,
        step_budget=0.5 if method == "orthogonal_l1" else None,
        eps=1.0e-12,
        log_thresholds=(0.0,),
    )


def _run_step(
    *,
    method: str,
    model_type: type[_LinearGradientModel],
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    torch.nn.Parameter,
    _RecordingAdamW,
    list[str],
    dict[str, Any],
]:
    parameter = torch.nn.Parameter(torch.tensor([2.0, 2.0]))
    capture = _Capture()
    events: list[str] = []
    optimizer = _RecordingAdamW([parameter], events)
    orthogonal_call: dict[str, Any] = {}
    if observed is not None:
        observed.update(
            {
                "parameter": parameter,
                "optimizer": optimizer,
                "events": events,
                "orthogonal_call": orthogonal_call,
            }
        )

    def record_orthogonal(
        candidate_optimizer: Any,
        params: list[Any],
        task_grads: list[Any | None],
        pressure_grads: list[Any | None],
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("ol1_correction")
        orthogonal_call.update(
            {
                "optimizer": candidate_optimizer,
                "params": params,
                "task_grads": task_grads,
                "pressure_grads": pressure_grads,
                **kwargs,
            }
        )
        return {}

    monkeypatch.setattr(
        optimization,
        "apply_adam_step_orthogonal_pressure",
        record_orthogonal,
    )
    result = optimization._run_training_step(
        model=model_type(parameter, capture),
        optimizer=optimizer,
        params=[parameter],
        torch=torch,
        np=np,
        train_tokens=np.arange(8, dtype=np.int32),
        block_size=2,
        micro_batch_size=1,
        grad_accum=1,
        device=torch.device("cpu"),
        dtype=None,
        pressure_config=_pressure_config(method),
        activation_capture=capture,
        step=1,
        schedule_step=np.asarray([[0]], dtype=np.int64),
    )
    return result, parameter, optimizer, events, orthogonal_call


@pytest.mark.parametrize(
    ("method", "expected_pre_clip", "expected_direction"),
    [
        ("none", 5.0, torch.tensor([3.0, 4.0])),
        ("l1_naive", 41.0**0.5, torch.tensor([4.0, 5.0])),
        ("orthogonal_l1", 5.0, torch.tensor([3.0, 4.0])),
    ],
)
def test_adamw_clips_the_method_specific_gradient_before_step(
    method: str,
    expected_pre_clip: float,
    expected_direction: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _parameter, optimizer, events, orthogonal_call = _run_step(
        method=method,
        model_type=_LinearGradientModel,
        monkeypatch=monkeypatch,
    )

    assert events[0] == "adamw_step"
    assert len(optimizer.gradient_snapshots) == 1
    adamw_gradient = optimizer.gradient_snapshots[0][0]
    assert adamw_gradient is not None
    assert torch.linalg.vector_norm(adamw_gradient).item() == pytest.approx(1.0, abs=2e-6)
    torch.testing.assert_close(
        adamw_gradient / torch.linalg.vector_norm(adamw_gradient),
        expected_direction / torch.linalg.vector_norm(expected_direction),
        atol=1e-6,
        rtol=0.0,
    )
    assert result["optimization/adamw_gradient_global_norm_pre_clip"] == pytest.approx(
        expected_pre_clip
    )
    assert result["optimization/adamw_gradient_global_norm_post_clip"] == pytest.approx(
        torch.linalg.vector_norm(adamw_gradient).item()
    )
    assert result["optimization/adamw_gradient_clip_max_norm"] == 1.0
    assert result["optimization/adamw_gradient_was_clipped"] is True

    if method == "orthogonal_l1":
        assert events == ["adamw_step", "ol1_correction"]
        assert orthogonal_call["optimizer"] is optimizer
        assert orthogonal_call["params"][0] is _parameter
        torch.testing.assert_close(orthogonal_call["task_grads"][0], adamw_gradient)
        torch.testing.assert_close(
            orthogonal_call["pressure_grads"][0],
            torch.tensor([0.5, 0.5]),
        )
        assert result["pressure/task_gradient_norm"] == pytest.approx(
            torch.linalg.vector_norm(adamw_gradient).item()
        )
    else:
        assert events == ["adamw_step"]
        assert orthogonal_call == {}
        # Existing pressure telemetry remains the raw task-gradient norm.
        assert result["pressure/task_gradient_norm"] == pytest.approx(5.0)


@pytest.mark.parametrize("method", ["none", "l1_naive", "orthogonal_l1"])
def test_nonfinite_adamw_gradient_fails_before_optimizer_step(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="non-finite"):
        _run_step(
            method=method,
            model_type=_NonfiniteGradientModel,
            monkeypatch=monkeypatch,
            observed=observed,
        )

    assert observed["optimizer"].gradient_snapshots == []
    assert observed["events"] == []
    assert observed["orthogonal_call"] == {}
    torch.testing.assert_close(
        observed["parameter"].detach(),
        torch.tensor([2.0, 2.0]),
    )
