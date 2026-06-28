from __future__ import annotations

from types import SimpleNamespace

import torch

from paper_exp.activation_pressure import activation_l1_pressure
from paper_exp.activation_pressure import activation_near_zero_metrics
from paper_exp.activation_pressure import activation_pressure_config
from paper_exp.activation_pressure import apply_adam_step_orthogonal_pressure
from paper_exp.activation_pressure import pressure_loss
from paper_exp.activation_pressure import ricker_pressure
from paper_exp.activations import ActivationCapture
from paper_exp.activations import clip_activation_tensor


def test_pressure_functions_are_finite() -> None:
    activations = {"mlp_hiddens.layer_0": torch.tensor([[0.0, 0.01, 0.2]])}

    ricker = ricker_pressure(torch, activations, c=0.05, sigma=0.05)
    l1 = activation_l1_pressure(torch, activations)

    assert torch.isfinite(ricker)
    assert torch.isfinite(l1)
    assert float(l1) > 0.0


def test_monitor_only_activation_pressure_has_no_pressure_loss() -> None:
    cfg = activation_pressure_config(
        {
            "activation_pressure": {
                "enabled": True,
                "method": "none",
                "sites": ["mlp_hiddens"],
                "weight": 0.0,
            }
        }
    )

    assert cfg.applies_pressure is False
    assert pressure_loss(torch, {"mlp_hiddens.layer_0": torch.ones(2)}, cfg) is None


def test_activation_capture_hooks_pythia_like_mlp_hidden() -> None:
    model = _TinyPythiaLikeModel()

    with ActivationCapture(model, ["mlp_hiddens"], torch=torch) as capture:
        output = model(torch.tensor([[[-1.0, 2.0]]]))

    assert "mlp_hiddens.layer_0" in capture.activations
    assert capture.site_metadata[0].downstream_operator.endswith("dense_4h_to_h")
    assert torch.equal(output, capture.activations["mlp_hiddens.layer_0"])


def test_clipping_produces_exact_zeros_and_near_zero_metrics() -> None:
    value = torch.tensor([-0.02, -0.001, 0.0, 0.003, 0.04])
    clipped = clip_activation_tensor(
        value,
        {"enabled": True, "mode": "threshold", "threshold": 0.003},
        torch=torch,
    )

    metrics = activation_near_zero_metrics({"mlp_hiddens.layer_0": clipped}, (0.0, 0.01))

    assert torch.equal(clipped, torch.tensor([-0.02, 0.0, 0.0, 0.0, 0.04]))
    assert metrics["activation/near_zero_mass/k0"] == 0.6
    assert metrics["activation/near_zero_mass/k1em02"] == 0.6


def test_adam_step_orthogonal_pressure_projects_conflict_and_caps_ratio() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    task_loss = parameter.square().sum()
    task_loss.backward()
    task_grads = [parameter.grad.detach().clone()]
    optimizer.step()

    pressure_grads = [-task_grads[0]]
    metrics = apply_adam_step_orthogonal_pressure(
        optimizer,
        [parameter],
        task_grads,
        pressure_grads,
        pressure_weight=1.0,
        step_budget=0.1,
    )

    assert metrics["pressure/pressure_update_projected"] is True
    assert metrics["pressure/task_pressure_update_dot_before"] < 0.0
    assert metrics["pressure/task_pressure_update_dot_after"] >= -1e-8
    assert metrics["pressure/pressure_update_ratio_final"] <= 0.1 + 1e-8


class _TinyPythiaLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = SimpleNamespace(mlp=SimpleNamespace(act=torch.nn.ReLU()))
        self.gpt_neox = SimpleNamespace(layers=[layer])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gpt_neox.layers[0].mlp.act(value)
