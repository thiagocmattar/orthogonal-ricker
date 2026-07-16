from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any


def apply_post_layernorm_relu(model: Any, *, torch: Any) -> Any:
    """Apply configured ReLUs after each GPT-NeoX block LayerNorm."""
    config = getattr(model, "config", None)
    if not bool(getattr(config, "post_layernorm_relu", False)):
        return model
    if getattr(model, "_post_layernorm_relu_applied", False):
        return model

    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        raise ValueError("Configured model.post_layernorm_relu, but the model has no GPT-NeoX layers.")

    for layer in layers:
        input_layernorm = getattr(layer, "input_layernorm", None)
        post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
        if input_layernorm is None or post_attention_layernorm is None:
            raise ValueError(
                "Configured model.post_layernorm_relu, but a GPT-NeoX layer is missing a branch LayerNorm."
            )

        layer.attention_input_relu = torch.nn.ReLU()
        layer.mlp_input_relu = torch.nn.ReLU()
        input_layernorm.register_forward_hook(_relu_output_hook(layer.attention_input_relu))
        post_attention_layernorm.register_forward_hook(_relu_output_hook(layer.mlp_input_relu))

    model._post_layernorm_relu_applied = True
    return model


def apply_post_qkv_relu(model: Any, *, torch: Any) -> Any:
    """Apply configured Q/K/V ReLUs inside each GPT-NeoX attention path."""
    config = getattr(model, "config", None)
    gate_config = _post_qkv_relu_config(getattr(config, "post_qkv_relu", None))
    if gate_config is None or not gate_config["enabled"]:
        return model
    if getattr(model, "_post_qkv_relu_applied", False):
        return model

    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        raise ValueError("Configured model.post_qkv_relu, but the model has no GPT-NeoX layers.")

    attentions = []
    for layer in layers:
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError("Configured model.post_qkv_relu, but a GPT-NeoX layer has no attention module.")
        for attribute in ("query_key_value", "dense", "head_size", "config"):
            if not hasattr(attention, attribute):
                raise ValueError(
                    "Configured model.post_qkv_relu, but a GPT-NeoX attention module "
                    f"is missing {attribute}."
                )
        attentions.append(attention)

    for attention in attentions:
        if gate_config["query"]:
            attention.query_relu = torch.nn.ReLU()
        if gate_config["key"]:
            attention.key_relu = torch.nn.ReLU()
        if gate_config["value"]:
            attention.value_relu = torch.nn.ReLU()
        attention.qk_relu_placement = gate_config["qk_placement"]
        attention.forward = MethodType(_post_qkv_relu_attention_forward, attention)

    model._post_qkv_relu_applied = True
    return model


def load_checkpoint_model(auto_model: Any, checkpoint_path: str | Path, *, torch: Any) -> Any:
    model = auto_model.from_pretrained(checkpoint_path)
    apply_post_layernorm_relu(model, torch=torch)
    return apply_post_qkv_relu(model, torch=torch)


def _post_qkv_relu_attention_forward(
    self: Any,
    hidden_states: Any,
    attention_mask: Any,
    layer_past: Any = None,
    position_embeddings: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """GPT-NeoX attention with Q/K/V ReLUs placed around the stock RoPE path."""
    from transformers.models.gpt_neox import modeling_gpt_neox

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    value_relu = getattr(self, "value_relu", None)
    if value_relu is not None:
        value_states = value_relu(value_states)

    placement = self.qk_relu_placement
    if placement == "pre_rope":
        query_relu = getattr(self, "query_relu", None)
        key_relu = getattr(self, "key_relu", None)
        if query_relu is not None:
            query_states = query_relu(query_states)
        if key_relu is not None:
            key_states = key_relu(key_states)

    cos, sin = position_embeddings
    query_states, key_states = modeling_gpt_neox.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    if placement == "post_rope":
        query_relu = getattr(self, "query_relu", None)
        key_relu = getattr(self, "key_relu", None)
        if query_relu is not None:
            query_states = query_relu(query_states)
        if key_relu is not None:
            key_states = key_relu(key_states)

    if layer_past is not None:
        key_states, value_states = layer_past.update(
            key_states,
            value_states,
            self.layer_idx,
        )

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


def _relu_output_hook(relu: Any) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return relu(output)

    return hook


def _post_qkv_relu_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Model config post_qkv_relu must be a mapping.")

    fields = ("enabled", "query", "key", "value")
    for field in fields:
        if not isinstance(value.get(field), bool):
            raise ValueError(f"Model config post_qkv_relu.{field} must be a boolean.")

    enabled = value["enabled"]
    placement = value.get("qk_placement")
    if enabled and placement not in {"pre_rope", "post_rope"}:
        raise ValueError("Model config post_qkv_relu.qk_placement must be 'pre_rope' or 'post_rope'.")
    if not enabled:
        if placement is not None:
            raise ValueError("Model config post_qkv_relu.qk_placement must be omitted when disabled.")
        if any(value[field] for field in ("query", "key", "value")):
            raise ValueError("Model config post_qkv_relu Q/K/V gates must be false when disabled.")
        return dict(value)
    if not any(value[field] for field in ("query", "key", "value")):
        raise ValueError("Model config post_qkv_relu is enabled, but no Q/K/V gate is enabled.")
    return dict(value)
