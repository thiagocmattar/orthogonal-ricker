from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper_exp.calibration import _adamw_parameters, _apply_model_architecture_overrides, _set_optimizer_lr
from paper_exp.modeling import (
    AdaptiveThresholdController,
    FixedOneSidedThreshold,
    FixedSymmetricThreshold,
    LearnedThresholdGate,
    adaptive_threshold_parameter_items,
    adaptive_threshold_parameter_snapshot,
    adaptive_threshold_training_metrics,
    apply_mlp_hidden_gate,
    apply_post_layernorm_relu,
    apply_post_qkv_relu,
    load_checkpoint_model,
    set_adaptive_threshold_stats_enabled,
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
    post_layernorm_gate = {"gate_type": "one_sided_threshold", "kappa": 0.1}
    mlp_hidden_gate = {"gate_type": "one_sided_threshold", "kappa": 0.1}
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
            "post_layernorm_gate": post_layernorm_gate,
            "mlp_hidden_gate": mlp_hidden_gate,
            "post_qkv_relu": post_qkv_relu,
        },
    )

    assert architecture.hidden_act == "relu"
    assert architecture.post_layernorm_relu is True
    assert architecture.post_layernorm_gate == post_layernorm_gate
    assert architecture.post_layernorm_gate is not post_layernorm_gate
    assert architecture.mlp_hidden_gate == mlp_hidden_gate
    assert architecture.mlp_hidden_gate is not mlp_hidden_gate
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


def test_fixed_one_sided_threshold_preserves_boundary_and_gradients() -> None:
    gate = FixedOneSidedThreshold(0.1)
    value = torch.tensor(
        [-0.2, 0.0, 0.099, 0.1, 0.2],
        requires_grad=True,
    )

    output = gate(value)
    assert torch.equal(output, torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2]))

    output.sum().backward()
    assert torch.equal(value.grad, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("gate_family", "expected"),
    [
        ("gplus", [0.0, 0.0, 0.0, 0.1, 0.2]),
        ("gpm", [-0.2, -0.1, 0.0, 0.1, 0.2]),
    ],
)
def test_learned_threshold_is_exact_hard_forward_with_threshold_only_soft_backward(
    gate_family: str,
    expected: list[float],
) -> None:
    controller = AdaptiveThresholdController()
    rho = controller.parameter_for("layer_0__q", kappa_init=0.1)
    gate = LearnedThresholdGate(
        controller=controller,
        parameter_key="layer_0__q",
        metric_name="query_gate_outputs.layer_0",
        gate_family=gate_family,
        kappa_init=0.1,
        kappa_scope="per_layer_site",
        threshold_scale="absolute",
        temperature=0.03,
        rms_epsilon=1e-8,
    )
    value = torch.tensor([-0.2, -0.1, 0.05, 0.1, 0.2], requires_grad=True)

    output = gate(value)
    assert torch.equal(output, torch.tensor(expected))
    output.sum().backward()

    expected_input_gradient = (torch.tensor(expected) != 0).float()
    assert torch.equal(value.grad, expected_input_gradient)
    assert rho.dtype == torch.float32
    assert rho.grad is not None
    assert torch.isfinite(rho.grad)
    assert float(rho.grad.abs()) > 0.0
    assert gate.kappa().item() == pytest.approx(0.1)


def test_learned_rms_relative_gate_is_scale_invariant_and_detaches_rms_statistic() -> None:
    controller = AdaptiveThresholdController()
    controller.parameter_for("v", kappa_init=0.8)
    gate = LearnedThresholdGate(
        controller=controller,
        parameter_key="v",
        metric_name="value_gate_outputs.layer_0",
        gate_family="gpm",
        kappa_init=0.8,
        kappa_scope="per_site",
        threshold_scale="rms_relative",
        temperature=0.03,
        rms_epsilon=1e-8,
    )
    base = torch.tensor([-2.0, -0.2, 0.1, 1.0], requires_grad=True)
    scaled = (10.0 * base.detach()).requires_grad_(True)

    base_output = gate(base)
    scaled_output = gate(scaled)

    assert torch.equal(base_output == 0, scaled_output == 0)
    assert torch.allclose(scaled_output, 10.0 * base_output)
    base_output.sum().backward()
    assert torch.equal(base.grad, (base_output != 0).float())


def test_adaptive_threshold_metrics_cover_gate_distribution_and_parameter_diagnostics() -> None:
    class TinyLearnedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adaptive_threshold_controller = AdaptiveThresholdController()
            self.adaptive_threshold_controller.parameter_for("v", kappa_init=0.1)
            self.gate = LearnedThresholdGate(
                controller=self.adaptive_threshold_controller,
                parameter_key="v",
                metric_name="value_gate_outputs.layer_0",
                gate_family="gpm",
                kappa_init=0.1,
                kappa_scope="per_site",
                threshold_scale="absolute",
                temperature=0.03,
                rms_epsilon=1e-8,
            )

    model = TinyLearnedModel()
    set_adaptive_threshold_stats_enabled(model, True)
    parameter = adaptive_threshold_parameter_items(model)[0][1]
    before_step = adaptive_threshold_parameter_snapshot(model)
    value = torch.tensor([-2.0, -0.1, 0.0, 0.2, 1.0])
    model.gate(value).sum().backward()
    torch.optim.SGD([parameter], lr=0.01).step()

    metrics = adaptive_threshold_training_metrics(model, before_step=before_step)
    prefix = "atg/value_gate_outputs.layer_0"
    assert metrics[f"{prefix}/forward_kappa"] == pytest.approx(0.1)
    assert f"{prefix}/kappa" not in metrics
    assert metrics[f"{prefix}/threshold_quantile"] == pytest.approx(0.2)
    assert metrics[f"{prefix}/zero_fraction"] == pytest.approx(0.2)
    assert metrics[f"{prefix}/positive_survivor_fraction"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/negative_survivor_fraction"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/survivor_sign_balance"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/survivor_rms"] == pytest.approx((5.05 / 4.0) ** 0.5)
    assert metrics[f"{prefix}/all_zero_flag"] == 0.0
    assert metrics[f"{prefix}/all_survive_flag"] == 0.0
    assert metrics["atg/parameter/v/gradient_norm"] > 0.0
    assert metrics["atg/parameter/v/step_norm"] > 0.0
    assert metrics["atg/parameter/v/kappa_step_norm"] > 0.0
    assert metrics["atg/parameter/v/kappa_over_init"] > 0.0
    assert metrics["atg/parameter/v/zero_kappa_flag"] == 0.0
    assert metrics["atg/parameter/v/nonfinite_threshold_flag"] == 0.0
    assert metrics["atg/parameter/v/frozen_threshold_flag"] == 0.0


def test_adaptive_threshold_optimizer_group_has_no_decay_and_preserves_lr_multiplier() -> None:
    class TinyLearnedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.adaptive_threshold_controller = AdaptiveThresholdController()
            self.adaptive_threshold_controller.parameter_for("global", kappa_init=0.1)

    model = TinyLearnedModel()
    groups = _adamw_parameters(
        model,
        weight_decay=0.01,
        threshold_learning_rate_multiplier=10.0,
    )
    optimizer = torch.optim.AdamW(groups, lr=3e-5, weight_decay=0.01)
    _set_optimizer_lr(optimizer, 2e-5)

    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["model"]["lr"] == pytest.approx(2e-5)
    assert by_name["model"]["weight_decay"] == pytest.approx(0.01)
    assert by_name["adaptive_threshold"]["lr"] == pytest.approx(2e-4)
    assert by_name["adaptive_threshold"]["weight_decay"] == 0.0
    assert by_name["adaptive_threshold"]["lr_multiplier"] == 10.0
    threshold_parameter = adaptive_threshold_parameter_items(model)[0][1]
    assert by_name["adaptive_threshold"]["params"] == [threshold_parameter]


@pytest.mark.parametrize(
    ("scope", "expected_parameters"),
    [("global", 1), ("per_site", 6), ("per_layer_site", 12)],
)
def test_learned_a6_threshold_sharing_scope_has_expected_parameter_count(
    scope: str,
    expected_parameters: int,
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    gate = {
        "gate_type": "learned_one_sided_threshold",
        "kappa_init": 0.1,
        "kappa_scope": scope,
        "threshold_scale": "absolute",
        "temperature": 0.03,
    }
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
    )
    architecture.post_layernorm_relu = True
    architecture.post_layernorm_gate = dict(gate)
    architecture.mlp_hidden_gate = dict(gate)
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
        **gate,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    # Exercise Module._apply: ordinary model parameters may change dtype, but
    # rho remains FP32 and gates resolve the controller's current parameter.
    model.to(dtype=torch.bfloat16)
    assert model.gpt_neox.embed_in.weight.dtype == torch.bfloat16

    assert len(adaptive_threshold_parameter_items(model)) == expected_parameters
    for layer in model.gpt_neox.layers:
        for gate_module in (
            layer.attention_input_relu,
            layer.mlp_input_relu,
            layer.mlp.act,
            layer.attention.query_relu,
            layer.attention.key_relu,
            layer.attention.value_relu,
        ):
            assert isinstance(gate_module, LearnedThresholdGate)
            assert gate_module.rho.dtype == torch.float32
            assert gate_module.rho is dict(adaptive_threshold_parameter_items(model))[gate_module.parameter_key]

    model.to(dtype=torch.float32)
    loss = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        labels=torch.tensor([[1, 2, 3, 4]]),
        use_cache=False,
    ).loss
    loss.backward()
    for _name, parameter in adaptive_threshold_parameter_items(model):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)


def test_learned_gate_controller_relation_survives_whole_model_deepcopy() -> None:
    class TinyLearnedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adaptive_threshold_controller = AdaptiveThresholdController()
            self.adaptive_threshold_controller.parameter_for("global", kappa_init=0.1)
            self.gate = LearnedThresholdGate(
                controller=self.adaptive_threshold_controller,
                parameter_key="global",
                metric_name="mlp_hiddens.layer_0",
                gate_family="gplus",
                kappa_init=0.1,
                kappa_scope="global",
                threshold_scale="absolute",
                temperature=0.03,
                rms_epsilon=1e-8,
            )

    original = TinyLearnedModel()
    copied = copy.deepcopy(original)

    assert copied.gate.rho is copied.adaptive_threshold_controller.rhos["global"]
    assert copied.gate.rho is not original.gate.rho
    with torch.no_grad():
        copied.gate.rho.add_(1.0)
    assert not torch.equal(copied.gate.rho, original.gate.rho)


def test_fixed_one_sided_branch_gates_are_exact_downstream_inputs() -> None:
    model = _TinyAllGateModel(kappa=0.1)
    layer = model.gpt_neox.layers[0]
    value = torch.tensor([[[-0.2, 0.03, 0.1, 0.2]]])

    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    attention_input, mlp_input, mlp_hidden, final_output = model(value)

    expected = torch.tensor([[[0.0, 0.0, 0.1, 0.2]]])
    assert torch.equal(attention_input, expected)
    assert torch.equal(mlp_input, expected)
    assert torch.equal(mlp_hidden, expected)
    assert layer.attention.last_input is attention_input
    assert layer.mlp.dense_h_to_4h.last_input is mlp_input
    assert layer.mlp.dense_4h_to_h.last_input is mlp_hidden
    assert isinstance(layer.attention_input_relu, FixedOneSidedThreshold)
    assert isinstance(layer.mlp_input_relu, FixedOneSidedThreshold)
    assert isinstance(layer.mlp.act, FixedOneSidedThreshold)
    assert torch.equal(final_output, value)
    assert not hasattr(model.gpt_neox, "final_layer_norm_gate")


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


@pytest.mark.parametrize(
    ("placement", "query", "key", "value"),
    [
        ("pre_rope", True, True, False),
        ("post_rope", True, False, False),
        ("post_rope", False, True, False),
        ("post_rope", False, False, True),
        ("post_rope", True, True, True),
    ],
)
def test_post_qkv_one_sided_threshold_constructs_enabled_subsets(
    placement: str,
    query: bool,
    key: bool,
    value: bool,
) -> None:
    model = _TinyPostQKVModel(
        placement=placement,
        gate_type="one_sided_threshold",
        kappa=0.1,
        query=query,
        key=key,
        value=value,
    )

    apply_post_qkv_relu(model, torch=torch)
    attention = model.gpt_neox.layers[0].attention

    for name, enabled in (("query", query), ("key", key), ("value", value)):
        gate = getattr(attention, f"{name}_relu", None)
        if not enabled:
            assert gate is None
            continue
        assert isinstance(gate, FixedOneSidedThreshold)
        assert gate.kappa == pytest.approx(0.1)
        assert torch.equal(
            gate(torch.tensor([-0.2, 0.09, 0.1, 0.2])),
            torch.tensor([0.0, 0.0, 0.1, 0.2]),
        )
    assert attention.qk_relu_placement == placement


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
    ("placement", "gate_type", "kappa", "query", "key", "value"),
    [
        ("pre_rope", "relu", None, True, True, True),
        ("post_rope", "relu", None, True, True, True),
        ("post_rope", "symmetric_threshold", 0.1, True, True, True),
        ("pre_rope", "symmetric_threshold", 0.1, True, True, False),
        ("pre_rope", "one_sided_threshold", 0.1, True, True, False),
        ("post_rope", "one_sided_threshold", 0.1, False, False, True),
        ("post_rope", "one_sided_threshold", 0.1, True, True, True),
    ],
)
def test_real_gpt_neox_checkpoint_round_trip_preserves_gates_and_cache(
    tmp_path: Path,
    placement: str,
    gate_type: str,
    kappa: float | None,
    query: bool,
    key: bool,
    value: bool,
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
        "query": query,
        "key": key,
        "value": value,
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
        if gate_type == "symmetric_threshold":
            expected_gate = FixedSymmetricThreshold
        elif gate_type == "one_sided_threshold":
            expected_gate = FixedOneSidedThreshold
        else:
            expected_gate = torch.nn.ReLU
        for name, enabled in (("query", query), ("key", key), ("value", value)):
            gate = getattr(layer.attention, f"{name}_relu", None)
            if not enabled:
                assert gate is None
                continue
            assert isinstance(gate, expected_gate)
            if kappa is not None:
                assert gate.kappa == pytest.approx(kappa)

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


def test_real_gpt_neox_checkpoint_round_trip_preserves_all_fixed_gplus_sites(
    tmp_path: Path,
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    gate = {"gate_type": "one_sided_threshold", "kappa": 0.1}
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=True,
    )
    architecture.post_layernorm_relu = True
    architecture.post_layernorm_gate = dict(gate)
    architecture.mlp_hidden_gate = dict(gate)
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
        **gate,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = load_checkpoint_model(GPTNeoXForCausalLM, tmp_path, torch=torch)

    assert loaded.config.post_layernorm_gate == gate
    assert loaded.config.mlp_hidden_gate == gate
    for layer in loaded.gpt_neox.layers:
        for module in (
            layer.attention_input_relu,
            layer.mlp_input_relu,
            layer.mlp.act,
            layer.attention.query_relu,
            layer.attention.key_relu,
            layer.attention.value_relu,
        ):
            assert isinstance(module, FixedOneSidedThreshold)
            assert module.kappa == pytest.approx(0.1)
    assert not hasattr(loaded.gpt_neox, "final_layer_norm_gate")
    output = loaded(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    assert torch.isfinite(output.logits).all()


def test_real_gpt_neox_learned_threshold_and_optimizer_round_trip_is_exact(
    tmp_path: Path,
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    learned_gate = {
        "gate_type": "learned_symmetric_threshold",
        "kappa_init": 0.1,
        "kappa_scope": "per_layer_site",
        "threshold_scale": "absolute",
        "surrogate": "hard_forward_soft_backward",
        "temperature": 0.03,
    }
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
        **learned_gate,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_qkv_relu(model, torch=torch)
    optimizer = torch.optim.AdamW(
        _adamw_parameters(
            model,
            weight_decay=0.01,
            threshold_learning_rate_multiplier=3.0,
        ),
        lr=3e-5,
        weight_decay=0.01,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model.train()
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    before_logits, before_gate_outputs = _capture_learned_gate_outputs(model, input_ids)
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in adaptive_threshold_parameter_items(model)
    }
    before_optimizer_threshold_state = {
        name: {
            state_name: (
                state_value.detach().clone()
                if isinstance(state_value, torch.Tensor)
                else state_value
            )
            for state_name, state_value in optimizer.state[parameter].items()
        }
        for name, parameter in adaptive_threshold_parameter_items(model)
    }
    model.save_pretrained(tmp_path, safe_serialization=True)
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")

    loaded = load_checkpoint_model(GPTNeoXForCausalLM, tmp_path, torch=torch)
    loaded.eval()
    after_logits, after_gate_outputs = _capture_learned_gate_outputs(loaded, input_ids)
    after_parameters = dict(adaptive_threshold_parameter_items(loaded))

    assert set(after_parameters) == set(before_parameters)
    for name, before in before_parameters.items():
        assert torch.equal(after_parameters[name].detach(), before)
    assert torch.equal(after_logits, before_logits)
    assert len(after_gate_outputs) == len(before_gate_outputs) == 6
    for before, after in zip(before_gate_outputs, after_gate_outputs, strict=True):
        assert torch.equal(after, before)
        assert torch.equal(after == 0, before == 0)

    loaded_optimizer = torch.optim.AdamW(
        _adamw_parameters(
            loaded,
            weight_decay=0.01,
            threshold_learning_rate_multiplier=3.0,
        ),
        lr=3e-5,
        weight_decay=0.01,
    )
    loaded_optimizer.load_state_dict(
        torch.load(tmp_path / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    loaded_groups = {group["group_name"]: group for group in loaded_optimizer.param_groups}
    assert loaded_groups["adaptive_threshold"]["lr_multiplier"] == 3.0
    assert loaded_groups["adaptive_threshold"]["weight_decay"] == 0.0
    loaded_threshold_parameters = dict(adaptive_threshold_parameter_items(loaded))
    for name, parameter in loaded_threshold_parameters.items():
        state = loaded_optimizer.state[parameter]
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()
        before_state = before_optimizer_threshold_state[name]
        assert set(state) == set(before_state)
        for state_name, before_value in before_state.items():
            after_value = state[state_name]
            if isinstance(before_value, torch.Tensor):
                assert torch.equal(after_value, before_value)
            else:
                assert after_value == before_value


def _capture_learned_gate_outputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    handles = []
    for layer in model.gpt_neox.layers:
        for name in ("query_relu", "key_relu", "value_relu"):
            gate = getattr(layer.attention, name)
            handles.append(
                gate.register_forward_hook(
                    lambda _module, _inputs, output: outputs.append(output.detach().clone())
                )
            )
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return logits, outputs


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


class _TinyAllGateModel(torch.nn.Module):
    def __init__(self, *, kappa: float) -> None:
        super().__init__()
        gate = {"gate_type": "one_sided_threshold", "kappa": kappa}
        self.config = SimpleNamespace(
            hidden_act="relu",
            post_layernorm_relu=True,
            post_layernorm_gate=dict(gate),
            mlp_hidden_gate=dict(gate),
        )
        self.gpt_neox = _TinyAllGateGPTNeoX()

    def forward(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.gpt_neox.layers[0]
        attention_input = layer.input_layernorm(value)
        layer.attention(attention_input)
        mlp_input = layer.post_attention_layernorm(value)
        mlp_preactivation = layer.mlp.dense_h_to_4h(mlp_input)
        mlp_hidden = layer.mlp.act(mlp_preactivation)
        layer.mlp.dense_4h_to_h(mlp_hidden)
        final_output = self.gpt_neox.final_layer_norm(value)
        return attention_input, mlp_input, mlp_hidden, final_output


class _TinyAllGateGPTNeoX(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyAllGateLayer()])
        self.final_layer_norm = torch.nn.Identity()


class _TinyAllGateLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.attention = _RecordingIdentity()
        self.mlp = SimpleNamespace(
            dense_h_to_4h=_RecordingIdentity(),
            act=torch.nn.ReLU(),
            dense_4h_to_h=_RecordingIdentity(),
        )


class _RecordingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value


class _TinyPostQKVModel(torch.nn.Module):
    def __init__(
        self,
        *,
        placement: str,
        gate_type: str | None = None,
        kappa: float | None = None,
        query: bool = True,
        key: bool = True,
        value: bool = True,
    ) -> None:
        super().__init__()
        post_qkv_relu: dict[str, object] = {
            "enabled": True,
            "query": query,
            "key": key,
            "value": value,
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
