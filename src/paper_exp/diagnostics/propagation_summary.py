"""Architecture metadata and endpoint summaries for activation propagation."""

from __future__ import annotations

from typing import Any

from paper_exp.modeling import activation_gate_metadata, model_topology_metadata
from paper_exp.topology import SITE_ALIAS_ORDER

from .logical_products import LOGICAL_MATMUL_STAGES, summarize_block_model_products


ACTIVATION_STAGE_ORDER = [
    "residual_input",
    "attention_layernorm_raw",
    "attention_input_gate",
    "a",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "q_pre_gate_input",
    "q_pre_gate_output",
    "q_pre",
    "k_pre_gate_input",
    "k_pre_gate_output",
    "k_pre",
    "q_post_gate_input",
    "q_post_gate_output",
    "q_post",
    "k_post_gate_input",
    "k_post_gate_output",
    "k_post",
    "v_gate_input",
    "v_gate_output",
    "v",
    "query_qk_input",
    "key_qk_input",
    "value_pv_input",
    "attention_probabilities",
    "attention_context",
    "attention_output",
    "mlp_layernorm_raw",
    "mlp_input_gate",
    "m",
    "mlp_w1_preactivation",
    "mlp_hidden_gate",
    "h",
    "mlp_output",
    "residual_output",
]

MATMUL_STAGE_ORDER = list(LOGICAL_MATMUL_STAGES)

ACTIVATION_STAGE_LABELS = {
    "residual_input": "H_l (block input)",
    "attention_layernorm_raw": "LN_attn(H_l), before optional a gate",
    "attention_input_gate": "a gate output G_a(LN_attn(H_l))",
    "a": "a: actual fused W_QKV input",
    "query_projection_output": "Q^0 from fused QKV projection, before gate/RoPE",
    "key_projection_output": "K^0 from fused QKV projection, before gate/RoPE",
    "value_projection_output": "V^0 from fused QKV projection, before gate",
    "q_pre_gate_input": "q_pre gate input before partial RoPE",
    "q_pre_gate_output": "q_pre gate output before partial RoPE",
    "q_pre": "q_pre: query port output before partial RoPE",
    "k_pre_gate_input": "k_pre gate input before partial RoPE",
    "k_pre_gate_output": "k_pre gate output before partial RoPE",
    "k_pre": "k_pre: key port output before partial RoPE",
    "q_post_gate_input": "q_post gate input after partial RoPE",
    "q_post_gate_output": "q_post gate output after partial RoPE",
    "q_post": "q_post: query port output consumed by QK^T",
    "k_post_gate_input": "k_post gate input after partial RoPE",
    "k_post_gate_output": "k_post gate output after partial RoPE",
    "k_post": "k_post: key port output consumed by QK^T",
    "v_gate_input": "v gate input",
    "v_gate_output": "v gate output",
    "v": "v: value port output consumed by PV",
    "query_qk_input": "q_post: actual Q operand of QK^T after partial RoPE",
    "key_qk_input": "k_post: actual K operand of QK^T after partial RoPE",
    "value_pv_input": "v: actual V operand of PV",
    "attention_probabilities": "P = softmax(masked QK^T)",
    "attention_context": "C = PV",
    "attention_output": "O = C W_o + b_o",
    "mlp_layernorm_raw": "LN_mlp(H_l), before optional m gate",
    "mlp_input_gate": "m gate output G_m(LN_mlp(H_l))",
    "m": "m: actual MLP W1 input",
    "mlp_w1_preactivation": "U = X_mlp W_1 + b_1",
    "mlp_hidden_gate": "h gate output G_h(U)",
    "h": "h: actual MLP W2 input",
    "mlp_output": "M = A W_2 + b_2",
    "residual_output": "H_{l+1} = H_l + O + M",
}

MATMUL_STAGE_LABELS = {
    "qkv_projection": "QKV projection: X_attn W_qkv",
    "qk_scores": "Attention scores: Q K^T (valid causal pairs)",
    "probability_value": "Attention context: P V (valid causal pairs)",
    "attention_output_projection": "Attention output: C W_o",
    "mlp_w1": "MLP expansion: X_mlp W_1",
    "mlp_w2": "MLP contraction: A W_2",
}

ENDPOINT_ZERO_STAGES = {
    "z_a": "a",
    "z_m": "m",
    "z_h": "h",
    "z_q_pre": "q_pre",
    "z_k_pre": "k_pre",
    "z_q_post": "q_post",
    "z_k_post": "k_post",
    "z_v": "v",
    "z_a_gate": "attention_input_gate",
    "z_m_gate": "mlp_input_gate",
    "z_h_gate": "mlp_hidden_gate",
    "z_q_pre_gate": "q_pre_gate_output",
    "z_k_pre_gate": "k_pre_gate_output",
    "z_q_post_gate": "q_post_gate_output",
    "z_k_post_gate": "k_post_gate_output",
    "z_v_gate": "v_gate_output",
    "z_q_qk": "query_qk_input",
    "z_k_qk": "key_qk_input",
    "z_v_pv": "value_pv_input",
    "z_context_wo": "attention_context",
}


def _attention_gate_metadata(layers: list[Any]) -> dict[str, Any]:
    aliases = ("q_pre", "k_pre", "q_post", "k_post", "v")
    per_layer = []
    for layer_index, layer in enumerate(layers):
        attention = layer.attention
        gate_specs = {
            alias: activation_gate_metadata(getattr(attention, f"{alias}_gate", None))
            for alias in aliases
        }
        active_sites = [alias for alias in aliases if gate_specs[alias] is not None]
        row = {
            "layer": layer_index,
            "active_sites": active_sites,
            "gate_presence": {
                alias: gate_specs[alias] is not None for alias in aliases
            },
            "qk_gate_placement": getattr(attention, "qk_gate_placement", None),
            "rotary_dim": int(attention.rotary_ndims),
            "head_width": int(attention.head_size),
            "gate_specs": gate_specs,
        }
        per_layer.append(row)

    signatures = {
        (tuple(row["active_sites"]), row["qk_gate_placement"])
        for row in per_layer
    }
    if len(signatures) != 1:
        raise ValueError("Attention gate presence and placement must match across all layers.")
    active_sites_tuple, placement = next(iter(signatures))
    active_sites = list(active_sites_tuple)
    if set(active_sites).intersection({"q_pre", "k_pre"}) and placement != "pre_rope":
        raise ValueError("q_pre/k_pre gates require qk_gate_placement pre_rope.")
    if set(active_sites).intersection({"q_post", "k_post"}) and placement != "post_rope":
        raise ValueError("q_post/k_post gates require qk_gate_placement post_rope.")
    if not set(active_sites).intersection({"q_pre", "k_pre", "q_post", "k_post"}):
        placement = None
    if placement not in {None, "pre_rope", "post_rope"}:
        raise ValueError("Q/K gates require qk_placement pre_rope or post_rope.")

    entries = [
        (row["layer"], alias, row["gate_specs"][alias])
        for row in per_layer
        for alias in aliases
        if row["gate_specs"][alias] is not None
    ]
    _validate_gate_spec_entries(entries, context="Attention")
    gate_specs = {
        alias: _summarize_gate_specs(
            [
                row["gate_specs"][alias]
                for row in per_layer
                if row["gate_specs"][alias] is not None
            ]
        )
        for alias in aliases
    }
    enabled_specs = [spec for _layer, _site, spec in entries]
    common_spec = _summarize_gate_specs(enabled_specs)
    return {
        "enabled": bool(active_sites),
        "active_sites": active_sites,
        "gate_presence": {alias: alias in active_sites for alias in aliases},
        "qk_gate_placement": placement,
        "gate_family": common_spec["gate_family"] if common_spec is not None else None,
        "operator": common_spec["operator"] if common_spec is not None else None,
        "kappa": common_spec.get("kappa") if common_spec is not None else None,
        "gate_specs": gate_specs,
        "layers": per_layer,
    }


def _gate_structure(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        spec["gate_family"],
        spec["operator"],
        float(spec["kappa"]),
    )


def _validate_gate_spec_entries(
    entries: list[tuple[int, str, dict[str, Any]]],
    *,
    context: str,
) -> None:
    if not entries:
        return
    structures = {_gate_structure(spec) for _layer, _site, spec in entries}
    if len(structures) != 1:
        raise ValueError(
            f"{context} gate operator and kappa must match across enabled sites and layers."
        )


def _summarize_gate_specs(specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not specs:
        return None
    return dict(specs[0])


def _architecture_metadata(
    model: Any,
    *,
    layers: list[Any],
    attention_gates: dict[str, Any],
    block_size: int,
    torch: Any,
) -> dict[str, Any]:
    """Describe the measured GPT-NeoX topology from runtime modules and shapes."""

    if block_size <= 0:
        raise ValueError("Architecture accounting requires a positive sequence length.")

    del torch  # Kept in the call signature alongside the other diagnostic reducers.
    topology = model_topology_metadata(model)
    active_sites = list(topology["active_sites"])
    active_site_set = frozenset(active_sites)
    expected_attention_sites = [
        site
        for site in active_sites
        if site in {"q_pre", "k_pre", "q_post", "k_post", "v"}
    ]
    if attention_gates["active_sites"] != expected_attention_sites:
        raise ValueError(
            "Measured attention gate sites do not match the configured activation topology."
        )
    if attention_gates["qk_gate_placement"] != topology["qk_placement"]:
        raise ValueError(
            "Measured Q/K gate placement does not match the configured activation topology."
        )

    attention_layers = attention_gates.get("layers", [])
    if len(attention_layers) != len(layers):
        raise ValueError("Expected one attention-gate metadata row per layer.")
    gate_specs_per_layer = []
    for layer_index, layer in enumerate(layers):
        attention_specs = attention_layers[layer_index]["gate_specs"]
        modules = {
            "a": getattr(layer, "a_gate", None),
            "m": getattr(layer, "m_gate", None),
            "h": layer.mlp.act if "h" in active_site_set else None,
        }
        row = {
            "layer": layer_index,
            **{
                alias: (
                    activation_gate_metadata(modules[alias])
                    if alias in modules
                    else attention_specs[alias]
                )
                for alias in SITE_ALIAS_ORDER
            },
        }
        gate_specs_per_layer.append(row)

    gate_entries = [
        (row["layer"], alias, row[alias])
        for row in gate_specs_per_layer
        for alias in SITE_ALIAS_ORDER
        if row[alias] is not None
    ]
    _validate_gate_spec_entries(gate_entries, context="Activation topology")
    gate_specs = {
        alias: _summarize_gate_specs(
            [row[alias] for row in gate_specs_per_layer if row[alias] is not None]
        )
        for alias in SITE_ALIAS_ORDER
    }

    projection_signatures = {
        (
            int(layer.attention.query_key_value.in_features),
            int(layer.attention.query_key_value.out_features),
            int(layer.attention.dense.in_features),
            int(layer.attention.dense.out_features),
            int(layer.mlp.dense_h_to_4h.in_features),
            int(layer.mlp.dense_h_to_4h.out_features),
            int(layer.mlp.dense_4h_to_h.in_features),
            int(layer.mlp.dense_4h_to_h.out_features),
            int(layer.attention.config.num_attention_heads),
            int(layer.attention.head_size),
        )
        for layer in layers
    }
    if len(projection_signatures) != 1:
        raise ValueError("Projection dimensions must match across all layers.")
    (
        qkv_in,
        qkv_out,
        wo_in,
        wo_out,
        w1_in,
        w1_out,
        w2_in,
        w2_out,
        num_heads,
        head_width,
    ) = next(iter(projection_signatures))
    if not (
        qkv_in == wo_in == wo_out == w1_in == w2_out == num_heads * head_width
        and qkv_out == 3 * qkv_in
        and w1_out == w2_in
    ):
        raise ValueError(
            "Activation propagation compute accounting requires standard GPT-NeoX "
            "QKV, Wo, W1, and W2 projection shapes."
        )

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Architecture accounting requires an output embedding weight.")
    output_shape = tuple(int(value) for value in output_embeddings.weight.shape)
    if len(output_shape) != 2 or output_shape[1] != qkv_in:
        raise ValueError(
            "Output embedding shape is incompatible with the transformer hidden width."
        )
    vocab_size, hidden_size = output_shape
    intermediate_size = w1_out

    attention_core_products = hidden_size * block_size * (block_size + 1) // 2
    operation_products_per_sequence_per_layer = {
        "qkv_projection": block_size * qkv_in * qkv_out,
        "qk_scores": attention_core_products,
        "probability_value": attention_core_products,
        "attention_output_projection": block_size * wo_in * wo_out,
        "mlp_w1": block_size * w1_in * w1_out,
        "mlp_w2": block_size * w2_in * w2_out,
    }
    dense_products_per_sequence_per_layer = sum(
        operation_products_per_sequence_per_layer.values()
    )

    return {
        "topology_id": topology["topology_id"],
        "active_sites": active_sites,
        "gate_presence": {
            alias: alias in active_site_set for alias in SITE_ALIAS_ORDER
        },
        "site_gate": topology["site_gate"],
        "gate_specs": gate_specs,
        "gate_specs_per_layer": gate_specs_per_layer,
        "qk_gate_placement": topology["qk_placement"],
        "base_hidden_activation": str(getattr(model.config, "hidden_act", "")),
        "num_layers": len(layers),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_heads,
        "head_width": head_width,
        "sequence_length": int(block_size),
        "operation_products_per_sequence_per_layer": (
            operation_products_per_sequence_per_layer
        ),
        "dense_products_per_sequence_per_layer": dense_products_per_sequence_per_layer,
    }


def _endpoint_summary(
    *,
    architecture: dict[str, Any],
    activation_rows: list[dict[str, Any]],
    matmul_rows: list[dict[str, Any]],
    validation_tokens: int,
) -> dict[str, Any]:
    """Reduce direct integer counters to the documented block and model ratios."""

    block_size = int(architecture["sequence_length"])
    num_layers = int(architecture["num_layers"])
    if validation_tokens <= 0 or validation_tokens % block_size:
        raise ValueError(
            "Endpoint accounting requires a positive whole number of fixed-length sequences."
        )
    sequences = validation_tokens // block_size

    zero_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}
    for stage in MATMUL_STAGE_ORDER:
        selected = [row for row in matmul_rows if row.get("name") == stage]
        if len(selected) != num_layers or any(
            not bool(row.get("available", True)) for row in selected
        ):
            raise ValueError(
                f"Endpoint accounting requires one available {stage} row per layer."
            )
        zero_count = sum(int(row["zero_count"]) for row in selected)
        total = sum(int(row["total"]) for row in selected)
        expected = (
            int(architecture["operation_products_per_sequence_per_layer"][stage])
            * sequences
            * num_layers
        )
        if total != expected:
            raise ValueError(
                f"Measured {stage} denominator {total} does not match dynamic architecture "
                f"denominator {expected}."
            )
        zero_counts[stage] = zero_count
        product_counts[stage] = total

    lm_head_product_count = (
        int(validation_tokens)
        * int(architecture["hidden_size"])
        * int(architecture["vocab_size"])
    )
    product_summary = summarize_block_model_products(
        zero_counts,
        product_counts,
        lm_head_product_count=lm_head_product_count,
    )

    zero_sites = {
        alias: _pooled_activation_stage(activation_rows, stage, num_layers=num_layers)
        for alias, stage in ENDPOINT_ZERO_STAGES.items()
    }
    return {
        "validation_sequences": sequences,
        "validation_tokens": int(validation_tokens),
        **product_summary,
        "zero_sites": zero_sites,
    }


def _pooled_activation_stage(
    rows: list[dict[str, Any]], stage: str, *, num_layers: int
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("name") == stage]
    if len(selected) != num_layers:
        raise ValueError(f"Expected one activation row for {stage} per layer.")
    available = [bool(row.get("available", True)) for row in selected]
    if not any(available):
        return {
            "stage": stage,
            "available": False,
            "zero_count": None,
            "total": None,
            "exact_zero_fraction": None,
        }
    if not all(available):
        raise ValueError(f"Activation stage {stage} is available in only some layers.")
    zero_count = sum(int(row["zero_count"]) for row in selected)
    total = sum(int(row["total"]) for row in selected)
    return {
        "stage": stage,
        "available": True,
        "zero_count": zero_count,
        "total": total,
        "exact_zero_fraction": zero_count / total if total else None,
    }
