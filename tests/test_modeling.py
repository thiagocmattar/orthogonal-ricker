from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper_exp.calibration import _apply_model_architecture_overrides
from paper_exp.modeling import (
    FixedSymmetricThreshold,
    apply_post_layernorm_relu,
    apply_post_qkv_relu,
    load_checkpoint_model,
)


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
    post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "pre_rope",
    }

    _apply_model_architecture_overrides(
        architecture,
        {
            "hidden_act": "relu",
            "post_layernorm_relu": True,
            "post_qkv_relu": post_qkv_relu,
        },
    )

    assert architecture.hidden_act == "relu"
    assert architecture.post_layernorm_relu is True
    assert architecture.post_qkv_relu == post_qkv_relu
    assert architecture.post_qkv_relu is not post_qkv_relu


def test_checkpoint_loader_reapplies_configured_post_layernorm_relu() -> None:
    model = _TinyPythiaLikeModel(post_layernorm_relu=True)
    auto_model = _FakeAutoModel(model)

    loaded = load_checkpoint_model(auto_model, "checkpoints/final", torch=torch)

    assert loaded is model
    assert auto_model.loaded_path == "checkpoints/final"
    assert isinstance(loaded.gpt_neox.layers[0].attention_input_relu, torch.nn.ReLU)
    assert isinstance(loaded.gpt_neox.layers[0].mlp_input_relu, torch.nn.ReLU)


@pytest.mark.parametrize("placement", ["pre_rope", "post_rope"])
def test_post_qkv_relu_places_separate_gates_around_rope(
    placement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.models.gpt_neox import modeling_gpt_neox

    model = _TinyPostQKVModel(placement=placement)
    attention = model.gpt_neox.layers[0].attention
    gate_inputs: dict[str, torch.Tensor] = {}
    gate_outputs: dict[str, torch.Tensor] = {}
    rope_inputs: dict[str, torch.Tensor] = {}
    attention_inputs: dict[str, torch.Tensor] = {}

    apply_post_qkv_relu(model, torch=torch)
    for name in ("query", "key", "value"):
        gate = getattr(attention, f"{name}_relu")

        def capture_gate_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            name: str = name,
        ) -> None:
            gate_inputs[name] = inputs[0].detach().clone()

        def capture_gate_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            name: str = name,
        ) -> None:
            gate_outputs[name] = output.detach().clone()

        gate.register_forward_pre_hook(capture_gate_input)
        gate.register_forward_hook(capture_gate_output)

    def controlled_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        _cos: torch.Tensor,
        _sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_inputs["query"] = query.detach().clone()
        rope_inputs["key"] = key.detach().clone()
        return query - 2.0, key - 2.0

    def capture_attention(
        _module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        _attention_mask: torch.Tensor | None,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_inputs["query"] = query.detach().clone()
        attention_inputs["key"] = key.detach().clone()
        attention_inputs["value"] = value.detach().clone()
        output = torch.zeros_like(query).transpose(1, 2)
        weights = torch.zeros((*query.shape[:-1], query.shape[-2]), dtype=query.dtype)
        return output, weights

    monkeypatch.setattr(modeling_gpt_neox, "apply_rotary_pos_emb", controlled_rope)
    monkeypatch.setattr(modeling_gpt_neox, "eager_attention_forward", capture_attention)

    hidden_states = torch.zeros((1, 1, 2))
    position = torch.zeros((1, 1, 2))
    attention(
        hidden_states,
        attention_mask=None,
        position_embeddings=(position, position),
    )

    raw_query = torch.tensor([[[[1.0, -2.0]]]])
    raw_key = torch.tensor([[[[-3.0, 4.0]]]])
    raw_value = torch.tensor([[[[-5.0, 6.0]]]])
    assert torch.equal(gate_inputs["value"], raw_value)
    assert torch.equal(gate_outputs["value"], torch.tensor([[[[0.0, 6.0]]]]))
    assert torch.equal(attention_inputs["value"], gate_outputs["value"])

    if placement == "pre_rope":
        assert torch.equal(gate_inputs["query"], raw_query)
        assert torch.equal(gate_inputs["key"], raw_key)
        assert torch.equal(rope_inputs["query"], gate_outputs["query"])
        assert torch.equal(rope_inputs["key"], gate_outputs["key"])
        assert torch.equal(attention_inputs["query"], torch.tensor([[[[-1.0, -2.0]]]]))
        assert torch.equal(attention_inputs["key"], torch.tensor([[[[-2.0, 2.0]]]]))
    else:
        assert torch.equal(rope_inputs["query"], raw_query)
        assert torch.equal(rope_inputs["key"], raw_key)
        assert torch.equal(gate_inputs["query"], raw_query - 2.0)
        assert torch.equal(gate_inputs["key"], raw_key - 2.0)
        assert torch.equal(attention_inputs["query"], gate_outputs["query"])
        assert torch.equal(attention_inputs["key"], gate_outputs["key"])
        assert torch.equal(attention_inputs["query"], torch.tensor([[[[0.0, 0.0]]]]))
        assert torch.equal(attention_inputs["key"], torch.tensor([[[[0.0, 2.0]]]]))


def test_post_qkv_relu_keeps_fused_projection_and_is_idempotent() -> None:
    model = _TinyPostQKVModel(placement="post_rope")
    attention = model.gpt_neox.layers[0].attention
    projection = attention.query_key_value
    weight = projection.weight.detach().clone()
    bias = projection.bias.detach().clone()

    apply_post_qkv_relu(model, torch=torch)
    first_gates = (attention.query_relu, attention.key_relu, attention.value_relu)
    first_forward = attention.forward
    apply_post_qkv_relu(model, torch=torch)

    assert attention.query_key_value is projection
    assert torch.equal(attention.query_key_value.weight, weight)
    assert torch.equal(attention.query_key_value.bias, bias)
    assert (attention.query_relu, attention.key_relu, attention.value_relu) == first_gates
    assert attention.forward == first_forward
    assert attention.qk_relu_placement == "post_rope"
    assert not hasattr(attention, "query_projection")
    assert not hasattr(attention, "key_projection")
    assert not hasattr(attention, "value_projection")


def test_fixed_symmetric_threshold_preserves_sign_boundary_and_gradients() -> None:
    gate = FixedSymmetricThreshold(0.1)
    value = torch.tensor(
        [-0.2, -0.1, -0.099, 0.0, 0.099, 0.1, 0.2],
        requires_grad=True,
    )

    output = gate(value)
    assert torch.equal(
        output,
        torch.tensor([-0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2]),
    )

    output.sum().backward()
    assert torch.equal(value.grad, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]))


def test_post_qkv_symmetric_threshold_constructs_three_signed_gates() -> None:
    model = _TinyPostQKVModel(
        placement="post_rope",
        gate_type="symmetric_threshold",
        kappa=0.1,
    )

    apply_post_qkv_relu(model, torch=torch)
    attention = model.gpt_neox.layers[0].attention

    for name in ("query", "key", "value"):
        gate = getattr(attention, f"{name}_relu")
        assert isinstance(gate, FixedSymmetricThreshold)
        assert gate.kappa == pytest.approx(0.1)
        assert torch.equal(gate(torch.tensor([-0.2, -0.01, 0.2])), torch.tensor([-0.2, 0.0, 0.2]))


def test_disabled_post_qkv_relu_leaves_stock_forward_and_gradients_unchanged() -> None:
    model = _TinyDisabledPostQKVModel()
    value = torch.tensor([[[1.0, -2.0]]], requires_grad=True)
    before = model(value)
    before.sum().backward()
    before_gradient = value.grad.detach().clone()
    original_forward = model.gpt_neox.layers[0].attention.forward

    value.grad = None
    apply_post_qkv_relu(model, torch=torch)
    after = model(value)
    after.sum().backward()

    assert torch.equal(after, before)
    assert torch.equal(value.grad, before_gradient)
    assert model.gpt_neox.layers[0].attention.forward == original_forward
    assert not hasattr(model.gpt_neox.layers[0].attention, "query_relu")


def test_checkpoint_loader_reconstructs_post_qkv_relu_placement() -> None:
    model = _TinyPostQKVModel(placement="pre_rope")
    auto_model = _FakeAutoModel(model)

    loaded = load_checkpoint_model(auto_model, "checkpoints/final", torch=torch)
    attention = loaded.gpt_neox.layers[0].attention

    assert loaded is model
    assert attention.qk_relu_placement == "pre_rope"
    assert isinstance(attention.query_relu, torch.nn.ReLU)
    assert isinstance(attention.key_relu, torch.nn.ReLU)
    assert isinstance(attention.value_relu, torch.nn.ReLU)


def test_checkpoint_loader_reconstructs_post_qkv_symmetric_threshold() -> None:
    model = _TinyPostQKVModel(
        placement="post_rope",
        gate_type="symmetric_threshold",
        kappa=0.1,
    )
    auto_model = _FakeAutoModel(model)

    loaded = load_checkpoint_model(auto_model, "checkpoints/final", torch=torch)
    attention = loaded.gpt_neox.layers[0].attention

    assert attention.qk_relu_placement == "post_rope"
    assert isinstance(attention.query_relu, FixedSymmetricThreshold)
    assert isinstance(attention.key_relu, FixedSymmetricThreshold)
    assert isinstance(attention.value_relu, FixedSymmetricThreshold)
    assert attention.query_relu.kappa == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("placement", "gate_type", "kappa"),
    [
        ("pre_rope", "relu", None),
        ("post_rope", "relu", None),
        ("post_rope", "symmetric_threshold", 0.1),
    ],
)
def test_real_gpt_neox_checkpoint_round_trip_preserves_gates_and_cache(
    tmp_path: Path,
    placement: str,
    gate_type: str,
    kappa: float | None,
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=True,
    )
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": placement,
        "gate_type": gate_type,
    }
    if kappa is not None:
        architecture.post_qkv_relu["kappa"] = kappa
    model = GPTNeoXForCausalLM(architecture)
    apply_post_qkv_relu(model, torch=torch)
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = load_checkpoint_model(GPTNeoXForCausalLM, tmp_path, torch=torch)
    assert loaded.config.post_qkv_relu == architecture.post_qkv_relu
    assert len(loaded.gpt_neox.layers) == 2
    for layer in loaded.gpt_neox.layers:
        assert layer.attention.qk_relu_placement == placement
        expected_gate = FixedSymmetricThreshold if kappa is not None else torch.nn.ReLU
        assert isinstance(layer.attention.query_relu, expected_gate)
        assert isinstance(layer.attention.key_relu, expected_gate)
        assert isinstance(layer.attention.value_relu, expected_gate)
        if kappa is not None:
            assert layer.attention.query_relu.kappa == pytest.approx(kappa)

    input_ids = torch.tensor([[1, 2, 3]])
    first = loaded(input_ids=input_ids, use_cache=True)
    assert torch.isfinite(first.logits).all()
    assert first.past_key_values is not None
    first.logits.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in loaded.parameters()
    )

    loaded.zero_grad(set_to_none=True)
    second = loaded(
        input_ids=torch.tensor([[4]]),
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    assert second.logits.shape == (1, 1, 32)
    assert torch.isfinite(second.logits).all()


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


class _TinyPostQKVModel(torch.nn.Module):
    def __init__(
        self,
        *,
        placement: str,
        gate_type: str | None = None,
        kappa: float | None = None,
    ) -> None:
        super().__init__()
        post_qkv_relu: dict[str, object] = {
            "enabled": True,
            "query": True,
            "key": True,
            "value": True,
            "qk_placement": placement,
        }
        if gate_type is not None:
            post_qkv_relu["gate_type"] = gate_type
        if kappa is not None:
            post_qkv_relu["kappa"] = kappa
        self.config = SimpleNamespace(
            post_qkv_relu=post_qkv_relu
        )
        self.gpt_neox = SimpleNamespace(
            layers=torch.nn.ModuleList([_TinyPostQKVLayer()]),
        )


class _TinyPostQKVLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _TinyPostQKVAttention()


class _TinyPostQKVAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = torch.nn.Linear(2, 6, bias=True)
        with torch.no_grad():
            self.query_key_value.weight.zero_()
            self.query_key_value.bias.copy_(torch.tensor([1.0, -2.0, -3.0, 4.0, -5.0, 6.0]))
        self.dense = torch.nn.Identity()
        self.head_size = 2
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.layer_idx = 0
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, *_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("The configured post-QKV path did not replace attention.forward.")


class _TinyDisabledPostQKVModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            post_qkv_relu={
                "enabled": False,
                "query": False,
                "key": False,
                "value": False,
            }
        )
        self.gpt_neox = SimpleNamespace(
            layers=torch.nn.ModuleList([_TinyDisabledPostQKVLayer()]),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gpt_neox.layers[0].attention(value)


class _TinyDisabledPostQKVLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = torch.nn.Linear(2, 2, bias=True)


class _FakeAutoModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.loaded_path: str | None = None

    def from_pretrained(self, path: str) -> torch.nn.Module:
        self.loaded_path = path
        return self.model
