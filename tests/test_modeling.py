from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from paper_exp.modeling import (
    FixedOneSidedThreshold,
    FixedSymmetricThreshold,
    _apply_model_architecture_overrides,
    activation_gate_metadata,
    apply_activation_topology,
    load_checkpoint_model,
    model_topology_metadata,
)
from paper_exp.topology import (
    SITE_ALIAS_ORDER,
    SUPPORTED_TOPOLOGIES,
    TOPOLOGY_ID_ORDER,
    resolve_topology,
    topology_for_runtime_sites,
)


EXPECTED_TOPOLOGY_ROWS = (
    ("A0", ()),
    ("A1-H", ("h",)),
    ("A2", ("m", "h")),
    ("A3", ("a", "m", "h")),
    ("A4-Q", ("a", "m", "h", "q_post")),
    ("A4-K", ("a", "m", "h", "k_post")),
    ("A4-V", ("a", "m", "h", "v")),
    ("A5-QK-PRE", ("a", "m", "h", "q_pre", "k_pre")),
    ("A5-QK-POST", ("a", "m", "h", "q_post", "k_post")),
    ("A6-PRE", ("a", "m", "h", "q_pre", "k_pre", "v")),
    ("A6-POST", ("a", "m", "h", "q_post", "k_post", "v")),
)


def test_topology_catalog_has_exact_rows_order_and_site_vocabulary() -> None:
    actual_rows = tuple(
        (topology_id, SUPPORTED_TOPOLOGIES[topology_id].active_sites)
        for topology_id in TOPOLOGY_ID_ORDER
    )

    assert actual_rows == EXPECTED_TOPOLOGY_ROWS
    assert len(SUPPORTED_TOPOLOGIES) == 11
    assert SITE_ALIAS_ORDER == (
        "a",
        "m",
        "h",
        "q_pre",
        "k_pre",
        "q_post",
        "k_post",
        "v",
    )


@pytest.mark.parametrize(("topology_id", "active_sites"), EXPECTED_TOPOLOGY_ROWS)
def test_every_topology_row_resolves_bidirectionally(
    topology_id: str,
    active_sites: tuple[str, ...],
) -> None:
    topology = resolve_topology(topology_id)
    expected_placement = (
        "pre_rope"
        if {"q_pre", "k_pre"}.intersection(active_sites)
        else "post_rope"
        if {"q_post", "k_post"}.intersection(active_sites)
        else None
    )

    assert topology.topology_id == topology_id
    assert topology.active_sites == active_sites
    assert topology.qk_placement == expected_placement
    assert topology.as_dict() == {
        "topology_id": topology_id,
        "active_sites": list(active_sites),
        "qk_placement": expected_placement,
    }
    assert topology_for_runtime_sites(tuple(reversed(active_sites))) is topology


@pytest.mark.parametrize(("topology_id", "active_sites"), EXPECTED_TOPOLOGY_ROWS)
def test_apply_activation_topology_realizes_exact_catalog_sites(
    topology_id: str,
    active_sites: tuple[str, ...],
) -> None:
    gate = None if topology_id == "A0" else {"operator": "one_sided_threshold", "kappa": 0.1}
    model = _TinyTopologyModel(topology_id=topology_id, site_gate=gate)

    apply_activation_topology(model, torch=torch)

    layer = model.gpt_neox.layers[0]
    realized_modules = _site_modules(layer)
    realized_sites = {
        alias
        for alias, module in realized_modules.items()
        if activation_gate_metadata(module) is not None
    }
    assert realized_sites == set(active_sites)
    assert model._resolved_topology == resolve_topology(topology_id).as_dict()
    assert model_topology_metadata(model) == {
        **resolve_topology(topology_id).as_dict(),
        "site_gate": gate,
    }

    attention = layer.attention
    expected_placement = resolve_topology(topology_id).qk_placement
    if set(active_sites).intersection({"q_pre", "k_pre", "q_post", "k_post", "v"}):
        assert attention.qk_gate_placement == expected_placement
        for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
            assert isinstance(getattr(attention, f"{alias}_site"), torch.nn.Identity)
    else:
        assert not hasattr(attention, "qk_gate_placement")


@pytest.mark.parametrize(("topology_id", "active_sites"), EXPECTED_TOPOLOGY_ROWS)
def test_every_topology_runs_a_real_gpt_neox_forward_and_backward(
    topology_id: str,
    active_sites: tuple[str, ...],
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    architecture.topology_id = topology_id
    architecture.site_gate = None if topology_id == "A0" else {"operator": "relu"}
    model = GPTNeoXForCausalLM(architecture).eval()
    apply_activation_topology(model, torch=torch)

    logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    logits.square().mean().backward()

    assert logits.shape == (1, 3, 32)
    assert model_topology_metadata(model)["active_sites"] == list(active_sites)
    qkv_gradient = model.gpt_neox.layers[0].attention.query_key_value.weight.grad
    assert qkv_gradient is not None
    assert torch.isfinite(qkv_gradient).all()


def test_a2_relu_gates_m_and_h_but_not_a() -> None:
    model = _TinyTopologyModel(
        topology_id="A2",
        site_gate={"operator": "relu"},
    )
    value = torch.tensor([[[-2.0, 3.0]]])

    apply_activation_topology(model, torch=torch)
    layer = model.gpt_neox.layers[0]
    a_value = layer.input_layernorm(value)
    m_value = layer.post_attention_layernorm(value)
    h_value = layer.mlp.act(layer.mlp.dense_h_to_4h(m_value))
    layer.mlp.dense_4h_to_h(h_value)

    expected_gated = torch.tensor([[[0.0, 3.0]]])
    assert torch.equal(a_value, value)
    assert torch.equal(m_value, expected_gated)
    assert torch.equal(h_value, expected_gated)
    assert not hasattr(layer, "a_gate")
    assert isinstance(layer.m_gate, torch.nn.ReLU)
    assert isinstance(layer.mlp.act, torch.nn.ReLU)
    assert layer.mlp.dense_h_to_4h.last_input is m_value
    assert layer.mlp.dense_4h_to_h.last_input is h_value
    assert model_topology_metadata(model)["active_sites"] == ["m", "h"]


@pytest.mark.parametrize(
    ("topology_id", "gate_alias"),
    [
        ("A5-QK-PRE", "pre"),
        ("A5-QK-POST", "post"),
    ],
)
def test_q_sites_are_exactly_before_or_after_controlled_rope(
    topology_id: str,
    gate_alias: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.models.gpt_neox import modeling_gpt_neox

    model = _TinyTopologyModel(
        topology_id=topology_id,
        site_gate={"operator": "relu"},
    )
    apply_activation_topology(model, torch=torch)
    attention = model.gpt_neox.layers[0].attention
    site_outputs: dict[str, torch.Tensor] = {}
    rope_inputs: dict[str, torch.Tensor] = {}
    product_inputs: dict[str, torch.Tensor] = {}

    for alias in ("q_pre", "k_pre", "q_post", "k_post"):
        module = getattr(attention, f"{alias}_site")

        def capture_site(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            alias: str = alias,
        ) -> None:
            site_outputs[alias] = output.detach().clone()

        module.register_forward_hook(capture_site)

    def controlled_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        _cos: torch.Tensor,
        _sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_inputs["query"] = query.detach().clone()
        rope_inputs["key"] = key.detach().clone()
        return query - 2.0, key - 2.0

    def capture_attention_product(
        _module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        _attention_mask: torch.Tensor | None,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        product_inputs["query"] = query.detach().clone()
        product_inputs["key"] = key.detach().clone()
        product_inputs["value"] = value.detach().clone()
        output = torch.zeros_like(query).transpose(1, 2)
        weights = torch.zeros((*query.shape[:-1], query.shape[-2]), dtype=query.dtype)
        return output, weights

    monkeypatch.setattr(modeling_gpt_neox, "apply_rotary_pos_emb", controlled_rope)
    monkeypatch.setattr(modeling_gpt_neox, "eager_attention_forward", capture_attention_product)

    position = torch.zeros((1, 1, 2))
    attention(
        torch.zeros((1, 1, 2)),
        attention_mask=None,
        position_embeddings=(position, position),
    )

    raw_query = torch.tensor([[[[1.0, -2.0]]]])
    raw_key = torch.tensor([[[[-3.0, 4.0]]]])
    raw_value = torch.tensor([[[[-5.0, 6.0]]]])
    assert torch.equal(product_inputs["value"], raw_value)

    if gate_alias == "pre":
        assert torch.equal(site_outputs["q_pre"], raw_query.relu())
        assert torch.equal(site_outputs["k_pre"], raw_key.relu())
        assert torch.equal(rope_inputs["query"], site_outputs["q_pre"])
        assert torch.equal(rope_inputs["key"], site_outputs["k_pre"])
        assert torch.equal(site_outputs["q_post"], torch.tensor([[[[-1.0, -2.0]]]]))
        assert torch.equal(site_outputs["k_post"], torch.tensor([[[[-2.0, 2.0]]]]))
    else:
        assert torch.equal(site_outputs["q_pre"], raw_query)
        assert torch.equal(site_outputs["k_pre"], raw_key)
        assert torch.equal(rope_inputs["query"], site_outputs["q_pre"])
        assert torch.equal(rope_inputs["key"], site_outputs["k_pre"])
        assert torch.equal(site_outputs["q_post"], torch.tensor([[[[0.0, 0.0]]]]))
        assert torch.equal(site_outputs["k_post"], torch.tensor([[[[0.0, 2.0]]]]))

    assert torch.equal(product_inputs["query"], site_outputs["q_post"])
    assert torch.equal(product_inputs["key"], site_outputs["k_post"])


def test_apply_activation_topology_is_idempotent_and_keeps_fused_qkv() -> None:
    model = _TinyTopologyModel(
        topology_id="A6-POST",
        site_gate={"operator": "relu"},
    )
    attention = model.gpt_neox.layers[0].attention
    projection = attention.query_key_value
    projection_weight = projection.weight.detach().clone()
    projection_bias = projection.bias.detach().clone()

    apply_activation_topology(model, torch=torch)
    first_forward = attention.forward
    first_gates = tuple(
        getattr(attention, f"{alias}_gate") for alias in ("q_post", "k_post", "v")
    )
    apply_activation_topology(model, torch=torch)

    assert attention.query_key_value is projection
    assert torch.equal(projection.weight, projection_weight)
    assert torch.equal(projection.bias, projection_bias)
    assert attention.forward == first_forward
    assert tuple(
        getattr(attention, f"{alias}_gate") for alias in ("q_post", "k_post", "v")
    ) == first_gates
    assert not hasattr(attention, "query_projection")
    assert not hasattr(attention, "key_projection")
    assert not hasattr(attention, "value_projection")


def test_model_architecture_override_persists_only_canonical_topology_fields() -> None:
    architecture = SimpleNamespace()
    gate = {"operator": "one_sided_threshold", "kappa": 1}

    _apply_model_architecture_overrides(
        architecture,
        {"topology_id": "A6-PRE", "site_gate": gate},
    )

    assert architecture.topology_id == "A6-PRE"
    assert architecture.site_gate == {"operator": "one_sided_threshold", "kappa": 1.0}
    assert architecture.site_gate is not gate


def test_checkpoint_loader_reapplies_canonical_topology() -> None:
    model = _TinyTopologyModel(
        topology_id="A2",
        site_gate={"operator": "relu"},
    )
    auto_model = _FakeAutoModel(model)

    loaded = load_checkpoint_model(auto_model, "checkpoints/final", torch=torch)

    assert loaded is model
    assert auto_model.loaded_path == "checkpoints/final"
    layer = loaded.gpt_neox.layers[0]
    assert not hasattr(layer, "a_gate")
    assert isinstance(layer.m_gate, torch.nn.ReLU)
    assert isinstance(layer.mlp.act, torch.nn.ReLU)


def test_topology_metadata_rejects_runtime_gate_that_disagrees_with_config() -> None:
    model = _TinyTopologyModel(
        topology_id="A2",
        site_gate={"operator": "relu"},
    )
    apply_activation_topology(model, torch=torch)
    model.config.site_gate = {
        "operator": "one_sided_threshold",
        "kappa": 0.7,
    }

    with pytest.raises(ValueError, match="Runtime site gate.*does not match"):
        model_topology_metadata(model)


@pytest.mark.parametrize(
    ("topology_id", "site_gate"),
    [
        ("A2", {"operator": "relu"}),
        ("A6-PRE", {"operator": "one_sided_threshold", "kappa": 0.1}),
        ("A6-POST", {"operator": "symmetric_threshold", "kappa": 0.1}),
    ],
)
def test_real_gpt_neox_checkpoint_round_trip_preserves_topology_and_outputs(
    tmp_path: Path,
    topology_id: str,
    site_gate: dict[str, object],
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
        use_cache=False,
    )
    architecture.topology_id = topology_id
    architecture.site_gate = dict(site_gate)
    model = GPTNeoXForCausalLM(architecture)
    apply_activation_topology(model, torch=torch)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids, use_cache=False).logits
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = load_checkpoint_model(GPTNeoXForCausalLM, tmp_path, torch=torch)
    loaded.eval()
    with torch.no_grad():
        actual_logits = loaded(input_ids=input_ids, use_cache=False).logits

    assert loaded.config.topology_id == topology_id
    assert loaded.config.site_gate == site_gate
    assert model_topology_metadata(loaded) == {
        **resolve_topology(topology_id).as_dict(),
        "site_gate": site_gate,
    }
    torch.testing.assert_close(actual_logits, expected_logits)


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
    value = torch.tensor([-0.2, 0.0, 0.099, 0.1, 0.2], requires_grad=True)

    output = gate(value)

    assert torch.equal(output, torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2]))
    output.sum().backward()
    assert torch.equal(value.grad, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (torch.nn.ReLU(), {"gate_family": "gplus", "operator": "relu", "kappa": 0.0}),
        (
            FixedOneSidedThreshold(0.2),
            {"gate_family": "gplus", "operator": "one_sided_threshold", "kappa": 0.2},
        ),
        (
            FixedSymmetricThreshold(0.3),
            {"gate_family": "gpm", "operator": "symmetric_threshold", "kappa": 0.3},
        ),
        (torch.nn.GELU(), None),
    ],
)
def test_activation_gate_metadata_is_canonical(
    module: torch.nn.Module,
    expected: dict[str, object] | None,
) -> None:
    assert activation_gate_metadata(module) == expected


def _site_modules(layer: Any) -> dict[str, torch.nn.Module | None]:
    modules: dict[str, torch.nn.Module | None] = {
        "a": getattr(layer, "a_gate", None),
        "m": getattr(layer, "m_gate", None),
        "h": getattr(layer.mlp, "act", None),
    }
    for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
        modules[alias] = getattr(layer.attention, f"{alias}_gate", None)
    return modules


class _TinyTopologyModel(torch.nn.Module):
    def __init__(self, *, topology_id: str, site_gate: dict[str, object] | None) -> None:
        super().__init__()
        self.config = SimpleNamespace(topology_id=topology_id, site_gate=site_gate)
        self.gpt_neox = _TinyGPTNeoX()


class _TinyGPTNeoX(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyTopologyLayer()])
        self.final_layer_norm = torch.nn.Identity()


class _TinyTopologyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.attention = _TinyAttention()
        self.mlp = _TinyMLP()


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense_h_to_4h = _RecordingIdentity()
        self.act = torch.nn.GELU()
        self.dense_4h_to_h = _RecordingIdentity()


class _TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = torch.nn.Linear(2, 6, bias=True)
        with torch.no_grad():
            self.query_key_value.weight.zero_()
            self.query_key_value.bias.copy_(
                torch.tensor([1.0, -2.0, -3.0, 4.0, -5.0, 6.0])
            )
        self.dense = torch.nn.Identity()
        self.head_size = 2
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.layer_idx = 0
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, *_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("An attention topology did not install the canonical forward path.")


class _RecordingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value


class _FakeAutoModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.loaded_path: str | None = None

    def from_pretrained(self, path: str) -> torch.nn.Module:
        self.loaded_path = path
        return self.model
