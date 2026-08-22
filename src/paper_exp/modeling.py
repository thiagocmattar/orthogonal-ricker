from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch

from paper_exp.topology import SITE_ALIAS_ORDER
from paper_exp.topology import resolve_topology_and_gate


class FixedSymmetricThreshold(torch.nn.Module):
    """Keep signed values at or beyond a fixed magnitude threshold."""

    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = float(kappa)

    def forward(self, value: Any) -> Any:
        return value.masked_fill(value.detach().abs() < self.kappa, 0.0)

    def extra_repr(self) -> str:
        return f"kappa={self.kappa:g}"


class FixedOneSidedThreshold(torch.nn.Module):
    """Keep values at or above a fixed one-sided threshold."""

    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = float(kappa)

    def forward(self, value: Any) -> Any:
        return value.masked_fill(value.detach() < self.kappa, 0.0)

    def extra_repr(self) -> str:
        return f"kappa={self.kappa:g}"


def apply_activation_topology(model: Any, *, torch: Any) -> Any:
    """Install the exact gate ports named by ``model.config.topology_id``."""
    config = getattr(model, "config", None)
    topology, gate_config = resolve_topology_and_gate(
        getattr(config, "topology_id", None),
        getattr(config, "site_gate", None),
    )
    if getattr(model, "_activation_topology_applied", False):
        return model

    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError(
            f"Configured topology {topology.topology_id}, but the model has no GPT-NeoX layers."
        )

    active = frozenset(topology.active_sites)
    for layer_index, layer in enumerate(layers):
        _validate_layer_for_topology(layer, layer_index=layer_index, active_sites=active)

    for layer in layers:
        if "a" in active:
            layer.a_gate = _site_gate(gate_config, torch=torch)
            layer.input_layernorm.register_forward_hook(_gate_output_hook(layer.a_gate))
        if "m" in active:
            layer.m_gate = _site_gate(gate_config, torch=torch)
            layer.post_attention_layernorm.register_forward_hook(_gate_output_hook(layer.m_gate))
        if "h" in active:
            layer.mlp.act = _site_gate(gate_config, torch=torch)

        attention_sites = active.intersection(
            {"q_pre", "k_pre", "q_post", "k_post", "v"}
        )
        if attention_sites:
            _install_attention_ports(
                layer.attention,
                active_sites=active,
                gate_config=gate_config,
                torch=torch,
            )

    model._activation_topology_applied = True
    model._resolved_topology = topology.as_dict()
    return model


def _validate_layer_for_topology(
    layer: Any,
    *,
    layer_index: int,
    active_sites: frozenset[str],
) -> None:
    if "a" in active_sites and getattr(layer, "input_layernorm", None) is None:
        raise ValueError(f"Topology site a cannot resolve the input LayerNorm in layer {layer_index}.")
    if "m" in active_sites and getattr(layer, "post_attention_layernorm", None) is None:
        raise ValueError(
            f"Topology site m cannot resolve the post-attention LayerNorm in layer {layer_index}."
        )
    if "h" in active_sites and getattr(getattr(layer, "mlp", None), "act", None) is None:
        raise ValueError(f"Topology site h cannot resolve the MLP activation in layer {layer_index}.")

    if not active_sites.intersection({"q_pre", "k_pre", "q_post", "k_post", "v"}):
        return
    attention = getattr(layer, "attention", None)
    if attention is None:
        raise ValueError(f"Attention topology sites cannot resolve attention in layer {layer_index}.")
    for attribute in ("query_key_value", "dense", "head_size", "config"):
        if not hasattr(attention, attribute):
            raise ValueError(
                f"Attention topology sites require {attribute} in layer {layer_index}."
            )


def _install_attention_ports(
    attention: Any,
    *,
    active_sites: frozenset[str],
    gate_config: dict[str, Any] | None,
    torch: Any,
) -> None:
    for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
        setattr(attention, f"{alias}_site", torch.nn.Identity())
        if alias in active_sites:
            setattr(attention, f"{alias}_gate", _site_gate(gate_config, torch=torch))

    if active_sites.intersection({"q_pre", "k_pre"}):
        attention.qk_gate_placement = "pre_rope"
    elif active_sites.intersection({"q_post", "k_post"}):
        attention.qk_gate_placement = "post_rope"
    else:
        attention.qk_gate_placement = None
    attention.forward = MethodType(_topology_attention_forward, attention)


def expose_attention_sites(model: Any, *, torch: Any) -> Any:
    """Expose parameter-free attention site taps when capture requests them."""
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError("Attention site capture currently supports GPTNeoX/Pythia models only.")
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError(f"Could not resolve attention in layer {layer_index}.")
        if all(
            getattr(attention, f"{alias}_site", None) is not None
            for alias in ("q_pre", "k_pre", "q_post", "k_post", "v")
        ):
            continue
        for attribute in ("query_key_value", "dense", "head_size", "config"):
            if not hasattr(attention, attribute):
                raise ValueError(
                    f"Attention site capture requires {attribute} in layer {layer_index}."
                )
        _install_attention_ports(
            attention,
            active_sites=frozenset(),
            gate_config=None,
            torch=torch,
        )
    return model


def _build_random_model(
    *,
    torch: Any,
    auto_config: Any,
    auto_model: Any,
    model_config: dict[str, Any],
    device: Any,
) -> Any:
    if model_config["initialization"] != "random":
        raise ValueError("This pretraining harness only supports model.initialization: random.")

    architecture = auto_config.from_pretrained(
        model_config["architecture"],
        revision=model_config["revision"],
    )
    _apply_model_architecture_overrides(architecture, model_config)
    architecture.torch_dtype = torch.float32
    model = auto_model.from_config(architecture)
    apply_activation_topology(model, torch=torch)
    return model.to(device=device, dtype=torch.float32)


def _apply_model_architecture_overrides(
    architecture: Any,
    model_config: dict[str, Any],
) -> None:
    topology, site_gate = resolve_topology_and_gate(
        model_config.get("topology_id"),
        model_config.get("site_gate"),
    )
    architecture.topology_id = topology.topology_id
    architecture.site_gate = None if site_gate is None else dict(site_gate)


def load_checkpoint_model(auto_model: Any, checkpoint_path: str | Path, *, torch: Any) -> Any:
    model = auto_model.from_pretrained(checkpoint_path)
    apply_activation_topology(model, torch=torch)
    return model


def _topology_attention_forward(
    self: Any,
    hidden_states: Any,
    attention_mask: Any,
    layer_past: Any = None,
    position_embeddings: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """GPT-NeoX attention with stable PRE, POST, and V site modules."""
    from transformers.models.gpt_neox import modeling_gpt_neox

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    query_states = _apply_optional_gate(self, "q_pre", query_states)
    key_states = _apply_optional_gate(self, "k_pre", key_states)
    query_states = self.q_pre_site(query_states)
    key_states = self.k_pre_site(key_states)

    cos, sin = position_embeddings
    query_states, key_states = modeling_gpt_neox.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    query_states = _apply_optional_gate(self, "q_post", query_states)
    key_states = _apply_optional_gate(self, "k_post", key_states)
    value_states = _apply_optional_gate(self, "v", value_states)

    if layer_past is not None:
        key_states, value_states = layer_past.update(
            key_states,
            value_states,
            self.layer_idx,
        )

    query_states = self.q_post_site(query_states)
    key_states = self.k_post_site(key_states)
    value_states = self.v_site(value_states)

    attention_interface = modeling_gpt_neox.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        modeling_gpt_neox.eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)
    return attn_output, attn_weights


def _apply_optional_gate(module: Any, alias: str, value: Any) -> Any:
    gate = getattr(module, f"{alias}_gate", None)
    return value if gate is None else gate(value)


def _gate_output_hook(gate: Any) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return gate(output)

    return hook


def _site_gate(gate_config: dict[str, Any] | None, *, torch: Any) -> Any:
    if gate_config is None:
        raise ValueError("An active topology site requires model.site_gate.")
    operator = gate_config["operator"]
    if operator == "relu":
        return torch.nn.ReLU()
    if operator == "one_sided_threshold":
        return FixedOneSidedThreshold(gate_config["kappa"])
    if operator == "symmetric_threshold":
        return FixedSymmetricThreshold(gate_config["kappa"])
    raise ValueError(f"Unsupported site-gate operator: {operator}")


def activation_gate_metadata(module: Any) -> dict[str, Any] | None:
    """Return stable runtime metadata for supported exact-zero gate modules."""
    if isinstance(module, torch.nn.ReLU):
        return {"gate_family": "gplus", "operator": "relu", "kappa": 0.0}
    if isinstance(module, FixedOneSidedThreshold):
        return {
            "gate_family": "gplus",
            "operator": "one_sided_threshold",
            "kappa": module.kappa,
        }
    if isinstance(module, FixedSymmetricThreshold):
        return {
            "gate_family": "gpm",
            "operator": "symmetric_threshold",
            "kappa": module.kappa,
        }
    return None


def model_topology_metadata(model: Any) -> dict[str, Any]:
    """Verify and describe the realized gate topology on every GPT-NeoX layer."""
    config = getattr(model, "config", None)
    topology, site_gate = resolve_topology_and_gate(
        getattr(config, "topology_id", None),
        getattr(config, "site_gate", None),
    )
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError("Cannot inspect activation topology without GPT-NeoX layers.")

    expected = frozenset(topology.active_sites)
    gate_specs: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        modules = {
            "a": getattr(layer, "a_gate", None),
            "m": getattr(layer, "m_gate", None),
            "h": getattr(getattr(layer, "mlp", None), "act", None),
        }
        attention = getattr(layer, "attention", None)
        for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
            modules[alias] = getattr(attention, f"{alias}_gate", None)
        realized = frozenset(
            alias for alias, module in modules.items() if activation_gate_metadata(module) is not None
        )
        if realized != expected:
            raise ValueError(
                f"Layer {layer_index} realizes gate sites {sorted(realized)}, "
                f"but topology {topology.topology_id} requires {list(topology.active_sites)}."
            )
        gate_specs.extend(
            activation_gate_metadata(modules[alias])
            for alias in SITE_ALIAS_ORDER
            if alias in expected
        )

    if gate_specs and any(spec != gate_specs[0] for spec in gate_specs[1:]):
        raise ValueError("Site-gate operator and kappa must match across all active sites and layers.")
    if gate_specs:
        realized_site_gate = {"operator": gate_specs[0]["operator"]}
        if realized_site_gate["operator"] != "relu":
            realized_site_gate["kappa"] = gate_specs[0]["kappa"]
        if realized_site_gate != site_gate:
            raise ValueError(
                f"Runtime site gate {realized_site_gate} does not match "
                f"model.site_gate {site_gate}."
            )
    return {
        **topology.as_dict(),
        "site_gate": None if site_gate is None else dict(site_gate),
    }
