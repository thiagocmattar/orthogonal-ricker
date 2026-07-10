from __future__ import annotations

from types import SimpleNamespace

import torch

from paper_exp.calibration import _apply_model_architecture_overrides
from paper_exp.modeling import apply_post_layernorm_relu, load_checkpoint_model


def test_post_layernorm_relu_is_applied_to_both_block_paths_only() -> None:
    model = _TinyPythiaLikeModel(post_layernorm_relu=True)
    value = torch.tensor([[[-2.0, 3.0]]])

    apply_post_layernorm_relu(model, torch=torch)
    attention_input, mlp_input, final_output = model(value)

    assert torch.equal(attention_input, torch.tensor([[[0.0, 3.0]]]))
    assert torch.equal(mlp_input, torch.tensor([[[0.0, 3.0]]]))
    assert torch.equal(final_output, value)
    assert isinstance(model.gpt_neox.layers[0].attention_input_relu, torch.nn.ReLU)
    assert isinstance(model.gpt_neox.layers[0].mlp_input_relu, torch.nn.ReLU)
    assert not hasattr(model.gpt_neox, "final_layer_norm_relu")


def test_post_layernorm_relu_modules_are_exact_downstream_inputs() -> None:
    model = _TinyPythiaLikeModel(post_layernorm_relu=True)
    layer = model.gpt_neox.layers[0]
    captured: dict[str, torch.Tensor] = {}
    apply_post_layernorm_relu(model, torch=torch)
    layer.attention_input_relu.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("attention", output)
    )
    layer.mlp_input_relu.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("mlp", output)
    )

    attention_input, mlp_input, _ = model(torch.tensor([[[-1.0, 2.0]]]))

    assert attention_input is captured["attention"]
    assert mlp_input is captured["mlp"]


def test_post_layernorm_relu_is_disabled_by_default() -> None:
    model = _TinyPythiaLikeModel(post_layernorm_relu=False)
    value = torch.tensor([[[-2.0, 3.0]]])

    apply_post_layernorm_relu(model, torch=torch)
    attention_input, mlp_input, final_output = model(value)

    assert torch.equal(attention_input, value)
    assert torch.equal(mlp_input, value)
    assert torch.equal(final_output, value)
    assert not hasattr(model.gpt_neox.layers[0], "attention_input_relu")


def test_model_architecture_override_persists_post_layernorm_relu_flag() -> None:
    architecture = SimpleNamespace(hidden_act="gelu")

    _apply_model_architecture_overrides(
        architecture,
        {"hidden_act": "relu", "post_layernorm_relu": True},
    )

    assert architecture.hidden_act == "relu"
    assert architecture.post_layernorm_relu is True


def test_checkpoint_loader_reapplies_configured_post_layernorm_relu() -> None:
    model = _TinyPythiaLikeModel(post_layernorm_relu=True)
    auto_model = _FakeAutoModel(model)

    loaded = load_checkpoint_model(auto_model, "checkpoints/final", torch=torch)

    assert loaded is model
    assert auto_model.loaded_path == "checkpoints/final"
    assert isinstance(loaded.gpt_neox.layers[0].attention_input_relu, torch.nn.ReLU)
    assert isinstance(loaded.gpt_neox.layers[0].mlp_input_relu, torch.nn.ReLU)


class _TinyPythiaLikeModel(torch.nn.Module):
    def __init__(self, *, post_layernorm_relu: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(post_layernorm_relu=post_layernorm_relu)
        self.gpt_neox = _TinyGPTNeoX()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.gpt_neox.layers[0]
        attention_input = layer.input_layernorm(value)
        mlp_input = layer.post_attention_layernorm(value)
        final_output = self.gpt_neox.final_layer_norm(value)
        return attention_input, mlp_input, final_output


class _TinyGPTNeoX(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyGPTNeoXLayer()])
        self.final_layer_norm = torch.nn.Identity()


class _TinyGPTNeoXLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()


class _FakeAutoModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.loaded_path: str | None = None

    def from_pretrained(self, path: str) -> torch.nn.Module:
        self.loaded_path = path
        return self.model
