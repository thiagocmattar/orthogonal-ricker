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
from paper_exp.activations import activation_exact_zero_counts
from paper_exp.activations import activation_exact_zero_counts_by_alias
from paper_exp.activations import clip_activation_tensor
from paper_exp.activations import resolve_site_aliases


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


def test_activation_capture_hooks_residual_stream_and_attention_output() -> None:
    model = _TinyPythiaLikeBlockModel()
    value = torch.tensor([[[-1.0, 2.0]]])

    with ActivationCapture(model, ["residual_streams", "attention_outputs"], torch=torch) as capture:
        model(value)

    assert torch.equal(capture.activations["residual_streams.layer_0"], value)
    assert torch.equal(capture.activations["attention_outputs.layer_0"], value * 2.0)
    assert {site.role for site in capture.site_metadata} == {"residual_stream", "attention_output"}


def test_activation_capture_hooks_post_layernorm_mlp_input() -> None:
    model = _TinyPythiaLikeMlpInputModel()
    value = torch.tensor([[[-1.0, 2.0]]])

    with ActivationCapture(model, ["mlp_inputs"], torch=torch) as capture:
        output = model(value)

    assert torch.equal(capture.activations["mlp_inputs.layer_0"], output)
    assert capture.site_metadata[0].role == "mlp_input"
    assert capture.site_metadata[0].downstream_operator.endswith("dense_h_to_4h")


def test_activation_capture_hooks_post_relu_attention_and_mlp_inputs() -> None:
    model = _TinyPythiaLikePostLayernormReluModel()
    value = torch.tensor([[[-1.0, 2.0]]])

    with ActivationCapture(model, ["attention_inputs", "mlp_inputs"], torch=torch) as capture:
        output = model(value)

    expected = torch.tensor([[[0.0, 2.0]]])
    assert torch.equal(capture.activations["attention_inputs.layer_0"], expected)
    assert torch.equal(capture.activations["mlp_inputs.layer_0"], expected)
    assert torch.equal(output, expected)
    assert [site.role for site in capture.site_metadata] == ["attention_input", "mlp_input"]
    assert capture.site_metadata[0].module_path.endswith("attention_input_relu")
    assert capture.site_metadata[0].downstream_operator.endswith("attention.query_key_value")
    assert capture.site_metadata[1].module_path.endswith("mlp_input_relu")
    assert capture.site_metadata[1].downstream_operator.endswith("mlp.dense_h_to_4h")


def test_activation_capture_clips_post_relu_attention_input_before_projection() -> None:
    model = _TinyPythiaLikePostLayernormReluModel()
    value = torch.tensor([[[0.001, 2.0]]])

    with ActivationCapture(
        model,
        ["attention_inputs"],
        torch=torch,
        clipping={"enabled": True, "mode": "threshold", "threshold": 0.01, "sites": ["attention_inputs"]},
    ) as capture:
        output = model(value)

    expected = torch.tensor([[[0.0, 2.0]]])
    assert torch.equal(capture.activations["attention_inputs.layer_0"], expected)
    assert torch.equal(model.gpt_neox.layers[0].attention.query_key_value.last_input, expected)
    assert torch.equal(output, expected)


def test_activation_capture_clips_attention_output_tuple() -> None:
    model = _TinyPythiaLikeBlockModel()
    value = torch.tensor([[[0.001, 2.0]]])

    with ActivationCapture(
        model,
        ["attention_outputs"],
        torch=torch,
        clipping={"enabled": True, "mode": "threshold", "threshold": 0.01, "sites": ["attention_outputs"]},
    ) as capture:
        output = model(value)

    assert torch.equal(capture.activations["attention_outputs.layer_0"], torch.tensor([[[0.0, 4.0]]]))
    assert torch.equal(output, torch.tensor([[[0.001, 6.0]]]))


def test_all_sites_alias_preserves_mlp_only_pressure_scope() -> None:
    assert resolve_site_aliases(["all_sites"]) == {"mlp_hiddens"}


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


def test_exact_zero_counts_are_accumulated_as_integers() -> None:
    zero_count, activation_count = activation_exact_zero_counts(
        {
            "attention_inputs.layer_0": torch.tensor([0.0, 1.0, 0.0]),
            "mlp_inputs.layer_0": torch.tensor([2.0, 0.0]),
        }
    )

    assert zero_count == 3
    assert activation_count == 5


def test_exact_zero_counts_are_grouped_by_site_alias() -> None:
    counts = activation_exact_zero_counts_by_alias(
        {
            "attention_inputs.layer_0": torch.tensor([0.0, 1.0, 0.0]),
            "attention_inputs.layer_1": torch.tensor([2.0, 0.0]),
            "mlp_hiddens.layer_0": torch.tensor([0.0, 3.0]),
        }
    )

    assert counts == {
        "attention_inputs": (3, 5),
        "mlp_hiddens": (1, 2),
    }


def test_rms_threshold_clipping_uses_current_activation_scale() -> None:
    value = torch.tensor([0.1, 1.0, 2.0])
    clipped = clip_activation_tensor(
        value,
        {"enabled": True, "mode": "rms_threshold", "rms_multiplier": 1.0},
        torch=torch,
    )

    rms = value.float().square().mean().sqrt()

    assert torch.equal(clipped, value.masked_fill(value.abs() <= rms, 0.0))
    assert torch.equal(clipped, torch.tensor([0.0, 0.0, 2.0]))


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


class _TinyPythiaLikeBlockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = SimpleNamespace(layers=torch.nn.ModuleList([_TinyPythiaLikeBlock()]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.gpt_neox.layers:
            value = layer(value)
        return value


class _TinyPythiaLikeMlpInputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = SimpleNamespace(post_attention_layernorm=torch.nn.Identity())
        self.gpt_neox = SimpleNamespace(layers=[layer])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gpt_neox.layers[0].post_attention_layernorm(value)


class _TinyPythiaLikePostLayernormReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = SimpleNamespace(layers=torch.nn.ModuleList([_TinyPythiaLikePostLayernormReluBlock()]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gpt_neox.layers[0](value)


class _TinyPythiaLikePostLayernormReluBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.attention_input_relu = torch.nn.ReLU()
        self.attention = torch.nn.Module()
        self.attention.query_key_value = _RecordingIdentity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp_input_relu = torch.nn.ReLU()
        self.mlp = torch.nn.Module()
        self.mlp.dense_h_to_4h = _RecordingIdentity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.attention_input_relu(self.input_layernorm(value))
        value = self.attention.query_key_value(value)
        value = self.mlp_input_relu(self.post_attention_layernorm(value))
        return self.mlp.dense_h_to_4h(value)


class _RecordingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value


class _TinyPythiaLikeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _TupleAttention()
        self.mlp = SimpleNamespace(act=torch.nn.Identity())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(value)
        return self.mlp.act(value + attention_output)


class _TupleAttention(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        return value * 2.0, None
