from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch


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


class AdaptiveThresholdController(torch.nn.Module):
    """Own each learned threshold parameter exactly once for safe serialization."""

    def __init__(self) -> None:
        super().__init__()
        self.rhos = torch.nn.ParameterDict()
        self._initial_kappas: dict[str, float] = {}

    def parameter_for(self, key: str, *, kappa_init: float) -> torch.nn.Parameter:
        if key in self.rhos:
            if not math.isclose(self._initial_kappas[key], kappa_init, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Shared adaptive-threshold parameter {key!r} has inconsistent kappa_init values."
                )
            return self.rhos[key]

        rho_init = _inverse_softplus(kappa_init)
        parameter = torch.nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
        self.rhos[key] = parameter
        self._initial_kappas[key] = float(kappa_init)
        return parameter

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        """Follow model device moves while keeping rho storage in FP32."""
        module = super()._apply(fn, recurse=recurse)
        with torch.no_grad():
            for parameter in self.rhos.values():
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.to(dtype=torch.float32)
                if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                    parameter.grad.data = parameter.grad.data.to(dtype=torch.float32)
        return module


class LearnedThresholdGate(torch.nn.Module):
    """Hard-forward gate with a soft surrogate only for threshold gradients."""

    def __init__(
        self,
        *,
        controller: AdaptiveThresholdController,
        parameter_key: str,
        metric_name: str,
        gate_family: str,
        kappa_init: float,
        kappa_scope: str,
        threshold_scale: str,
        temperature: float,
        rms_epsilon: float,
    ) -> None:
        super().__init__()
        if gate_family not in {"gplus", "gpm"}:
            raise ValueError(f"Unsupported learned gate family: {gate_family}")
        # The controller is the sole registered owner. Keeping this strong
        # reference unregistered avoids duplicate state-dict aliases while
        # allowing a whole-model deepcopy to preserve the internal relation.
        object.__setattr__(self, "_controller", controller)
        # Keep the lookup identity separate from the public metadata field so
        # malformed metadata can be diagnosed without making the parameter
        # itself inaccessible.
        self._parameter_key = parameter_key
        self.parameter_key = parameter_key
        self.metric_name = metric_name
        self.gate_family = gate_family
        self.kappa_init = float(kappa_init)
        self.kappa_scope = kappa_scope
        self.threshold_scale = threshold_scale
        self.temperature = float(temperature)
        self.rms_epsilon = float(rms_epsilon)
        self.record_stats = False
        self.last_stats: dict[str, Any] = {}

    @property
    def rho(self) -> torch.nn.Parameter:
        controller = object.__getattribute__(self, "_controller")
        return controller.rhos[self._parameter_key]

    def kappa(self) -> Any:
        # rho remains FP32 even when the surrounding forward is autocast.
        return torch.nn.functional.softplus(self.rho.float())

    def forward(self, value: Any) -> Any:
        value_fp32 = value.float()
        detached = value_fp32.detach()
        score = detached if self.gate_family == "gplus" else detached.abs()
        kappa = self.kappa()

        if self.threshold_scale == "rms_relative":
            input_rms = detached.square().mean().sqrt().clamp_min(self.rms_epsilon)
            normalized_score = score / input_rms
            margin = normalized_score - kappa
            effective_threshold = kappa * input_rms
        else:
            input_rms = None
            normalized_score = score
            margin = score - kappa
            effective_threshold = kappa

        hard_mask = normalized_score >= kappa
        soft_mask = torch.sigmoid(margin / self.temperature)
        # Forward value is the exact hard mask. Since normalized_score is
        # detached, the soft path changes rho gradients only; input gradients
        # retain the hard gate's zero-or-one derivative.
        mask = soft_mask + (hard_mask.to(soft_mask.dtype) - soft_mask).detach()
        output = value * mask.to(dtype=value.dtype)

        if not self.record_stats:
            return output

        if input_rms is None:
            input_rms = detached.square().mean().sqrt().clamp_min(self.rms_epsilon)
        kappa_over_rms = (
            kappa
            if self.threshold_scale == "rms_relative"
            else kappa / input_rms
        )
        output_detached = output.detach().float()
        survivor_mask = output_detached != 0.0
        survivor_count = survivor_mask.count_nonzero()
        element_count = survivor_mask.numel()
        survivor_denominator = survivor_count.clamp_min(1).to(dtype=torch.float32)
        positive_survivors = (survivor_mask & (output_detached > 0.0)).count_nonzero()
        negative_survivors = (survivor_mask & (output_detached < 0.0)).count_nonzero()
        survivor_rms = (
            output_detached.masked_fill(~survivor_mask, 0.0).square().sum()
            / survivor_denominator
        ).sqrt()
        positive_survivor_fraction = positive_survivors.float() / survivor_denominator
        negative_survivor_fraction = negative_survivors.float() / survivor_denominator
        zero_fraction = (~survivor_mask).float().mean()
        threshold_quantile = (~hard_mask).float().mean()

        self.last_stats = {
            # These distribution metrics describe the forward immediately
            # before the optimizer update; parameter/*/kappa is post-update.
            "forward_kappa": kappa.detach(),
            "forward_effective_threshold": effective_threshold.detach(),
            "input_rms": input_rms.detach(),
            "kappa_over_rms": kappa_over_rms.detach(),
            "transition_band_mass": (margin.detach().abs() <= self.temperature).float().mean(),
            # The empirical threshold quantile is the pre-gate mass rejected
            # by the hard threshold. The zero fraction separately measures
            # exact zeros in the gate output.
            "threshold_quantile": threshold_quantile,
            "margin_mean": margin.detach().mean(),
            "margin_min": margin.detach().amin(),
            "margin_max": margin.detach().amax(),
            "zero_fraction": zero_fraction,
            "positive_survivor_fraction": positive_survivor_fraction,
            "negative_survivor_fraction": negative_survivor_fraction,
            "survivor_sign_balance": (
                positive_survivor_fraction - negative_survivor_fraction
            ),
            "survivor_rms": survivor_rms,
            "all_zero_flag": (survivor_count == 0).float(),
            "all_survive_flag": (survivor_count == element_count).float(),
        }
        return output

    def extra_repr(self) -> str:
        return (
            f"family={self.gate_family}, kappa_init={self.kappa_init:g}, "
            f"scope={self.kappa_scope}, scale={self.threshold_scale}, "
            f"temperature={self.temperature:g}, key={self.parameter_key}"
        )


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

    gate_config = _one_sided_gate_config(
        getattr(config, "post_layernorm_gate", None),
        field_name="post_layernorm_gate",
    )
    for layer_index, layer in enumerate(layers):
        input_layernorm = getattr(layer, "input_layernorm", None)
        post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
        if input_layernorm is None or post_attention_layernorm is None:
            raise ValueError(
                "Configured model.post_layernorm_relu, but a GPT-NeoX layer is missing a branch LayerNorm."
            )

        layer.attention_input_relu = _branch_gate(
            model,
            gate_config,
            layer_index=layer_index,
            site="a",
            metric_name=f"attention_inputs.layer_{layer_index}",
            torch=torch,
        )
        layer.mlp_input_relu = _branch_gate(
            model,
            gate_config,
            layer_index=layer_index,
            site="m",
            metric_name=f"mlp_inputs.layer_{layer_index}",
            torch=torch,
        )
        input_layernorm.register_forward_hook(_relu_output_hook(layer.attention_input_relu))
        post_attention_layernorm.register_forward_hook(_relu_output_hook(layer.mlp_input_relu))

    model._post_layernorm_relu_applied = True
    return model


def apply_mlp_hidden_gate(model: Any, *, torch: Any) -> Any:
    """Replace configured GPT-NeoX MLP ReLUs with one-sided threshold gates."""
    config = getattr(model, "config", None)
    gate_config = _one_sided_gate_config(
        getattr(config, "mlp_hidden_gate", None),
        field_name="mlp_hidden_gate",
    )
    if gate_config is None:
        return model
    if getattr(model, "_mlp_hidden_gate_applied", False):
        return model
    if str(getattr(config, "hidden_act", "")).lower() != "relu":
        raise ValueError("Configured model.mlp_hidden_gate requires model.hidden_act='relu'.")

    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        raise ValueError("Configured model.mlp_hidden_gate, but the model has no GPT-NeoX layers.")

    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        activation = getattr(mlp, "act", None)
        if activation is None:
            raise ValueError("Configured model.mlp_hidden_gate, but a GPT-NeoX layer has no MLP activation.")
        if not isinstance(activation, torch.nn.ReLU):
            raise ValueError(
                "Configured model.mlp_hidden_gate requires every GPT-NeoX MLP activation to be ReLU."
            )
        mlp.act = _branch_gate(
            model,
            gate_config,
            layer_index=layer_index,
            site="h",
            metric_name=f"mlp_hiddens.layer_{layer_index}",
            torch=torch,
        )

    model._mlp_hidden_gate_applied = True
    return model


def apply_post_qkv_relu(model: Any, *, torch: Any) -> Any:
    """Apply configured Q/K/V gates inside each GPT-NeoX attention path."""
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

    for layer_index, attention in enumerate(attentions):
        if gate_config["query"]:
            attention.query_relu = _post_qkv_gate(
                model,
                gate_config,
                layer_index=layer_index,
                site="q",
                metric_name=f"query_gate_outputs.layer_{layer_index}",
                torch=torch,
            )
        if gate_config["key"]:
            attention.key_relu = _post_qkv_gate(
                model,
                gate_config,
                layer_index=layer_index,
                site="k",
                metric_name=f"key_gate_outputs.layer_{layer_index}",
                torch=torch,
            )
        if gate_config["value"]:
            attention.value_relu = _post_qkv_gate(
                model,
                gate_config,
                layer_index=layer_index,
                site="v",
                metric_name=f"value_gate_outputs.layer_{layer_index}",
                torch=torch,
            )
        attention.qk_relu_placement = gate_config["qk_placement"]
        attention.forward = MethodType(_post_qkv_relu_attention_forward, attention)

    model._post_qkv_relu_applied = True
    return model


def load_checkpoint_model(auto_model: Any, checkpoint_path: str | Path, *, torch: Any) -> Any:
    model = auto_model.from_pretrained(checkpoint_path)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    _restore_adaptive_threshold_state(model, checkpoint_path, torch=torch)
    return model


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
    gate_fields = {
        "gate_type",
        "kappa",
        "kappa_init",
        "kappa_scope",
        "threshold_scale",
        "surrogate",
        "temperature",
        "rms_epsilon",
    }
    if not enabled:
        extra = set(value) - {"enabled", "query", "key", "value", "qk_placement"} - gate_fields
        if extra:
            fields = ", ".join(sorted(str(field) for field in extra))
            raise ValueError(f"Disabled model config post_qkv_relu contains unsupported fields: {fields}.")
        if placement is not None:
            raise ValueError("Model config post_qkv_relu.qk_placement must be omitted when disabled.")
        if any(value[field] for field in ("query", "key", "value")):
            raise ValueError("Model config post_qkv_relu Q/K/V gates must be false when disabled.")
        if gate_fields.intersection(value):
            raise ValueError("Model config post_qkv_relu gate settings must be omitted when disabled.")
        return dict(value)
    if not any(value[field] for field in ("query", "key", "value")):
        raise ValueError("Model config post_qkv_relu is enabled, but no Q/K/V gate is enabled.")

    gate_type = value.get("gate_type", "relu")
    supported_gate_types = {
        "relu",
        "one_sided_threshold",
        "symmetric_threshold",
        "learned_one_sided_threshold",
        "learned_symmetric_threshold",
    }
    if gate_type not in supported_gate_types:
        raise ValueError(
            "Model config post_qkv_relu.gate_type must be 'relu', "
            "'one_sided_threshold', 'symmetric_threshold', "
            "'learned_one_sided_threshold', or 'learned_symmetric_threshold'."
        )
    base_fields = {"enabled", "query", "key", "value", "qk_placement", "gate_type"}
    if gate_type == "relu":
        allowed_fields = base_fields
    elif gate_type in {"one_sided_threshold", "symmetric_threshold"}:
        allowed_fields = base_fields | {"kappa"}
    else:
        allowed_fields = base_fields | {
            "kappa_init",
            "kappa_scope",
            "threshold_scale",
            "surrogate",
            "temperature",
            "rms_epsilon",
        }
    extra = set(value) - allowed_fields
    if extra:
        fields = ", ".join(sorted(str(field) for field in extra))
        raise ValueError(f"Model config post_qkv_relu contains unsupported fields: {fields}.")
    if gate_type == "relu":
        if gate_fields.intersection(value) - {"gate_type"}:
            raise ValueError("Model config post_qkv_relu threshold fields must be omitted for ordinary ReLU gates.")
    elif gate_type in {"one_sided_threshold", "symmetric_threshold"}:
        if "kappa" not in value:
            raise ValueError("Model config post_qkv_relu.kappa is required for threshold gates.")
        _require_finite_number(value["kappa"], field_name="post_qkv_relu.kappa", minimum=0.0)
        unexpected = gate_fields.intersection(value) - {"gate_type", "kappa"}
        if unexpected:
            raise ValueError("Learned threshold fields must be omitted for fixed post_qkv_relu gates.")
    else:
        _validate_learned_gate_fields(value, field_name="post_qkv_relu")

    normalized = dict(value)
    normalized["gate_type"] = gate_type
    if "kappa" in normalized:
        normalized["kappa"] = float(normalized["kappa"])
    if gate_type.startswith("learned_"):
        normalized.update(_normalized_learned_gate_fields(value))
    return normalized


def _post_qkv_gate(
    model: Any,
    gate_config: Mapping[str, Any],
    *,
    layer_index: int,
    site: str,
    metric_name: str,
    torch: Any,
) -> Any:
    if gate_config["gate_type"] == "relu":
        return torch.nn.ReLU()
    if gate_config["gate_type"] == "one_sided_threshold":
        return FixedOneSidedThreshold(gate_config["kappa"])
    if gate_config["gate_type"] == "symmetric_threshold":
        return FixedSymmetricThreshold(gate_config["kappa"])
    return _learned_gate(
        model,
        gate_config,
        layer_index=layer_index,
        site=site,
        metric_name=metric_name,
    )


def activation_gate_metadata(module: Any) -> dict[str, Any] | None:
    """Return stable runtime metadata for supported exact-zero gate modules."""
    if isinstance(module, torch.nn.ReLU):
        return {"gate_family": "gplus", "gate_type": "relu", "kappa": 0.0}
    if isinstance(module, FixedOneSidedThreshold):
        return {
            "gate_family": "gplus",
            "gate_type": "one_sided_threshold",
            "kappa": module.kappa,
        }
    if isinstance(module, FixedSymmetricThreshold):
        return {
            "gate_family": "gpm",
            "gate_type": "symmetric_threshold",
            "kappa": module.kappa,
        }
    if isinstance(module, LearnedThresholdGate):
        return {
            "gate_family": module.gate_family,
            "gate_type": (
                "learned_one_sided_threshold"
                if module.gate_family == "gplus"
                else "learned_symmetric_threshold"
            ),
            "kappa": float(module.kappa().detach().cpu()),
            "kappa_init": module.kappa_init,
            "kappa_scope": module.kappa_scope,
            "threshold_scale": module.threshold_scale,
            "surrogate": "hard_forward_soft_backward",
            "temperature": module.temperature,
            "rms_epsilon": module.rms_epsilon,
            "parameter_key": module.parameter_key,
        }
    return None


def _branch_gate(
    model: Any,
    gate_config: Mapping[str, Any] | None,
    *,
    layer_index: int,
    site: str,
    metric_name: str,
    torch: Any,
) -> Any:
    if gate_config is None:
        return torch.nn.ReLU()
    if gate_config["gate_type"] == "one_sided_threshold":
        return FixedOneSidedThreshold(gate_config["kappa"])
    return _learned_gate(
        model,
        gate_config,
        layer_index=layer_index,
        site=site,
        metric_name=metric_name,
    )


def _one_sided_gate_config(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Model config {field_name} must be a mapping.")
    fixed_fields = {"gate_type", "kappa"}
    learned_fields = {
        "gate_type",
        "kappa_init",
        "kappa_scope",
        "threshold_scale",
        "surrogate",
        "temperature",
        "rms_epsilon",
    }
    gate_type = value.get("gate_type")
    allowed = learned_fields if gate_type == "learned_one_sided_threshold" else fixed_fields
    extra = set(value) - allowed
    if extra:
        fields = ", ".join(sorted(str(field) for field in extra))
        raise ValueError(f"Model config {field_name} contains unsupported fields: {fields}.")
    if gate_type == "one_sided_threshold":
        if "kappa" not in value:
            raise ValueError(f"Model config {field_name}.kappa is required.")
        _require_finite_number(value["kappa"], field_name=f"{field_name}.kappa", minimum=0.0)
        return {"gate_type": gate_type, "kappa": float(value["kappa"])}
    if gate_type != "learned_one_sided_threshold":
        raise ValueError(
            f"Model config {field_name}.gate_type must be 'one_sided_threshold' "
            "or 'learned_one_sided_threshold'."
        )
    _validate_learned_gate_fields(value, field_name=field_name)
    return {"gate_type": gate_type, **_normalized_learned_gate_fields(value)}


def _validate_learned_gate_fields(value: Mapping[str, Any], *, field_name: str) -> None:
    required = ("kappa_init", "kappa_scope", "threshold_scale", "temperature")
    for field in required:
        if field not in value:
            raise ValueError(f"Model config {field_name}.{field} is required for learned gates.")
    _require_finite_number(value["kappa_init"], field_name=f"{field_name}.kappa_init", minimum=0.0, strict=True)
    _require_finite_number(value["temperature"], field_name=f"{field_name}.temperature", minimum=0.0, strict=True)
    if value["kappa_scope"] not in {"global", "per_site", "per_layer_site"}:
        raise ValueError(
            f"Model config {field_name}.kappa_scope must be 'global', 'per_site', or 'per_layer_site'."
        )
    if value["threshold_scale"] not in {"absolute", "rms_relative"}:
        raise ValueError(
            f"Model config {field_name}.threshold_scale must be 'absolute' or 'rms_relative'."
        )
    if value.get("surrogate", "hard_forward_soft_backward") != "hard_forward_soft_backward":
        raise ValueError(
            f"Model config {field_name}.surrogate must be 'hard_forward_soft_backward'."
        )
    _require_finite_number(
        value.get("rms_epsilon", 1e-8),
        field_name=f"{field_name}.rms_epsilon",
        minimum=0.0,
        strict=True,
    )
    if "kappa" in value:
        raise ValueError(f"Model config {field_name}.kappa must be omitted for learned gates.")


def _normalized_learned_gate_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kappa_init": float(value["kappa_init"]),
        "kappa_scope": str(value["kappa_scope"]),
        "threshold_scale": str(value["threshold_scale"]),
        "surrogate": str(value.get("surrogate", "hard_forward_soft_backward")),
        "temperature": float(value["temperature"]),
        "rms_epsilon": float(value.get("rms_epsilon", 1e-8)),
    }


def _require_finite_number(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    strict: bool = False,
) -> None:
    valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
    if strict:
        valid = valid and float(value) > minimum
        bound = "positive"
    else:
        valid = valid and float(value) >= minimum
        bound = "non-negative"
    if not valid:
        raise ValueError(f"Model config {field_name} must be a finite {bound} number.")


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Learned threshold kappa_init must be finite and positive.")
    return value + math.log(-math.expm1(-value))


def _adaptive_threshold_controller(model: Any) -> AdaptiveThresholdController:
    controller = getattr(model, "adaptive_threshold_controller", None)
    if controller is None:
        controller = AdaptiveThresholdController()
        model.adaptive_threshold_controller = controller
    if not isinstance(controller, AdaptiveThresholdController):
        raise ValueError("Model adaptive_threshold_controller has an unexpected type.")
    return controller


def _learned_gate(
    model: Any,
    gate_config: Mapping[str, Any],
    *,
    layer_index: int,
    site: str,
    metric_name: str,
) -> LearnedThresholdGate:
    scope = str(gate_config["kappa_scope"])
    if scope == "global":
        parameter_key = "global"
    elif scope == "per_site":
        parameter_key = site
    else:
        parameter_key = f"layer_{layer_index}__{site}"
    controller = _adaptive_threshold_controller(model)
    controller.parameter_for(parameter_key, kappa_init=float(gate_config["kappa_init"]))
    gate_family = (
        "gplus"
        if gate_config["gate_type"] == "learned_one_sided_threshold"
        else "gpm"
    )
    return LearnedThresholdGate(
        controller=controller,
        parameter_key=parameter_key,
        metric_name=metric_name,
        gate_family=gate_family,
        kappa_init=float(gate_config["kappa_init"]),
        kappa_scope=scope,
        threshold_scale=str(gate_config["threshold_scale"]),
        temperature=float(gate_config["temperature"]),
        rms_epsilon=float(gate_config["rms_epsilon"]),
    )


def adaptive_threshold_parameter_items(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    controller = getattr(model, "adaptive_threshold_controller", None)
    if not isinstance(controller, AdaptiveThresholdController):
        return []
    return list(controller.rhos.items())


def adaptive_threshold_parameter_snapshot(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().clone()
        for name, parameter in adaptive_threshold_parameter_items(model)
    }


def set_adaptive_threshold_stats_enabled(model: Any, enabled: bool) -> None:
    """Collect distribution reductions only on steps that will log them."""
    if not isinstance(
        getattr(model, "adaptive_threshold_controller", None),
        AdaptiveThresholdController,
    ):
        return
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return
    for module in modules():
        if not isinstance(module, LearnedThresholdGate):
            continue
        module.record_stats = enabled
        if enabled:
            module.last_stats = {}


def adaptive_threshold_training_metrics(
    model: Any,
    *,
    before_step: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    kappas = []
    grad_squares = []
    step_squares = []
    kappa_step_squares = []
    zero_kappa_flags = []
    nonfinite_threshold_flags = []
    frozen_threshold_flags = []
    controller = getattr(model, "adaptive_threshold_controller", None)
    for name, parameter in adaptive_threshold_parameter_items(model):
        kappa = torch.nn.functional.softplus(parameter.detach().float())
        kappa_init = float(controller._initial_kappas[name])
        kappas.append(kappa)
        metrics[f"atg/parameter/{name}/kappa"] = float(kappa.cpu())
        metrics[f"atg/parameter/{name}/kappa_over_init"] = float((kappa / kappa_init).cpu())
        zero_kappa_flag = (kappa == 0.0).float()
        nonfinite_threshold_flag = (~torch.isfinite(kappa)).float()
        zero_kappa_flags.append(zero_kappa_flag)
        nonfinite_threshold_flags.append(nonfinite_threshold_flag)
        metrics[f"atg/parameter/{name}/zero_kappa_flag"] = float(zero_kappa_flag.cpu())
        metrics[f"atg/parameter/{name}/nonfinite_threshold_flag"] = float(
            nonfinite_threshold_flag.cpu()
        )
        gradient = parameter.grad
        grad_norm = (
            gradient.detach().float().norm()
            if gradient is not None
            else torch.zeros((), device=parameter.device)
        )
        grad_squares.append(grad_norm.square())
        metrics[f"atg/parameter/{name}/gradient_norm"] = float(grad_norm.cpu())
        before = None if before_step is None else before_step.get(name)
        step_norm = (
            (parameter.detach() - before.to(parameter.device)).float().norm()
            if before is not None
            else torch.zeros((), device=parameter.device)
        )
        step_squares.append(step_norm.square())
        metrics[f"atg/parameter/{name}/step_norm"] = float(step_norm.cpu())
        before_kappa = (
            torch.nn.functional.softplus(before.to(parameter.device).float())
            if before is not None
            else kappa
        )
        kappa_step_norm = (kappa - before_kappa).norm()
        kappa_step_squares.append(kappa_step_norm.square())
        metrics[f"atg/parameter/{name}/kappa_step_norm"] = float(kappa_step_norm.cpu())
        frozen_threshold_flag = (step_norm == 0.0).float()
        frozen_threshold_flags.append(frozen_threshold_flag)
        metrics[f"atg/parameter/{name}/frozen_threshold_flag"] = float(
            frozen_threshold_flag.cpu()
        )

    if kappas:
        stacked = torch.stack(kappas)
        metrics["atg/kappa_min"] = float(stacked.amin().cpu())
        metrics["atg/kappa_mean"] = float(stacked.mean().cpu())
        metrics["atg/kappa_max"] = float(stacked.amax().cpu())
        metrics["atg/gradient_norm"] = float(torch.stack(grad_squares).sum().sqrt().cpu())
        metrics["atg/step_norm"] = float(torch.stack(step_squares).sum().sqrt().cpu())
        metrics["atg/kappa_step_norm"] = float(
            torch.stack(kappa_step_squares).sum().sqrt().cpu()
        )
        metrics["atg/zero_kappa_flag"] = float(torch.stack(zero_kappa_flags).amax().cpu())
        metrics["atg/nonfinite_threshold_flag"] = float(
            torch.stack(nonfinite_threshold_flags).amax().cpu()
        )
        metrics["atg/frozen_threshold_flag"] = float(
            torch.stack(frozen_threshold_flags).amax().cpu()
        )

    transition_values = []
    zero_values = []
    all_zero_flags = []
    all_survive_flags = []
    for module in model.modules():
        if not isinstance(module, LearnedThresholdGate) or not module.last_stats:
            continue
        for key, value in module.last_stats.items():
            metrics[f"atg/{module.metric_name}/{key}"] = float(value.detach().float().cpu())
        transition_values.append(module.last_stats["transition_band_mass"].detach().float())
        zero_values.append(module.last_stats["zero_fraction"].detach().float())
        all_zero_flags.append(module.last_stats["all_zero_flag"].detach().float())
        all_survive_flags.append(module.last_stats["all_survive_flag"].detach().float())
    if transition_values:
        metrics["atg/transition_band_mass_mean"] = float(torch.stack(transition_values).mean().cpu())
        metrics["atg/zero_fraction_mean"] = float(torch.stack(zero_values).mean().cpu())
        metrics["atg/all_zero_flag"] = float(torch.stack(all_zero_flags).amax().cpu())
        metrics["atg/all_survive_flag"] = float(torch.stack(all_survive_flags).amax().cpu())
    return metrics


def _restore_adaptive_threshold_state(
    model: Any,
    checkpoint_path: str | Path,
    *,
    torch: Any,
) -> None:
    controller = getattr(model, "adaptive_threshold_controller", None)
    if not isinstance(controller, AdaptiveThresholdController):
        return

    checkpoint_dir = Path(checkpoint_path)
    prefix = "adaptive_threshold_controller."
    state: dict[str, Any] = {}
    safetensors_path = checkpoint_dir / "model.safetensors"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    elif index_path.exists():
        from safetensors import safe_open

        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_names = sorted({shard for key, shard in weight_map.items() if key.startswith(prefix)})
        for shard_name in shard_names:
            with safe_open(str(checkpoint_dir / shard_name), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        state[key.removeprefix(prefix)] = handle.get_tensor(key)
    else:
        pytorch_path = checkpoint_dir / "pytorch_model.bin"
        if pytorch_path.exists():
            saved = torch.load(pytorch_path, map_location="cpu", weights_only=True)
            state = {
                key.removeprefix(prefix): value
                for key, value in saved.items()
                if key.startswith(prefix)
            }

    expected = set(controller.state_dict())
    if not state:
        raise ValueError("Learned adaptive-threshold checkpoint parameters are missing.")
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            "Learned adaptive-threshold checkpoint keys do not match the configured gates: "
            f"missing={missing}, unexpected={unexpected}."
        )
    controller.load_state_dict(state, strict=True)
