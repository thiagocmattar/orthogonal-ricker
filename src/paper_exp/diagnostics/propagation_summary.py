"""Architecture metadata and endpoint summaries for activation propagation."""

from __future__ import annotations

from typing import Any

from paper_exp.modeling import activation_gate_metadata

from .logical_products import LOGICAL_MATMUL_STAGES, summarize_block_model_products


ACTIVATION_STAGE_ORDER = [
    "residual_input",
    "attention_layernorm_raw",
    "attention_input_relu",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "query_gate_input",
    "key_gate_input",
    "value_gate_input",
    "query_gate_output",
    "key_gate_output",
    "value_gate_output",
    "query_qk_input",
    "key_qk_input",
    "value_pv_input",
    "attention_probabilities",
    "attention_context",
    "attention_output",
    "mlp_layernorm_raw",
    "mlp_input_relu",
    "mlp_w1_preactivation",
    "mlp_hidden_relu",
    "mlp_output",
    "residual_output",
]

MATMUL_STAGE_ORDER = list(LOGICAL_MATMUL_STAGES)

ACTIVATION_STAGE_LABELS = {
    "residual_input": "H_l (block input)",
    "attention_layernorm_raw": "LN_attn(H_l), before optional ReLU",
    "attention_input_relu": "ReLU(LN_attn(H_l))",
    "query_projection_output": "Q^0 from fused QKV projection, before gate/RoPE",
    "key_projection_output": "K^0 from fused QKV projection, before gate/RoPE",
    "value_projection_output": "V^0 from fused QKV projection, before gate",
    "query_gate_input": "Input to query ReLU (placement-dependent)",
    "key_gate_input": "Input to key ReLU (placement-dependent)",
    "value_gate_input": "V^0 input to value ReLU",
    "query_gate_output": "Output of query ReLU (placement-dependent)",
    "key_gate_output": "Output of key ReLU (placement-dependent)",
    "value_gate_output": "Output of value ReLU",
    "query_qk_input": "Actual Q operand of QK^T",
    "key_qk_input": "Actual K operand of QK^T",
    "value_pv_input": "Actual V operand of PV",
    "attention_probabilities": "P = softmax(masked QK^T)",
    "attention_context": "C = PV",
    "attention_output": "O = C W_o + b_o",
    "mlp_layernorm_raw": "LN_mlp(H_l), before optional ReLU",
    "mlp_input_relu": "ReLU(LN_mlp(H_l))",
    "mlp_w1_preactivation": "U = X_mlp W_1 + b_1",
    "mlp_hidden_relu": "A = ReLU(U)",
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
    "z_a": "attention_input_relu",
    "z_m": "mlp_input_relu",
    "z_h": "mlp_hidden_relu",
    "z_q_gate": "query_gate_output",
    "z_k_gate": "key_gate_output",
    "z_v_gate": "value_gate_output",
    "z_q_qk": "query_qk_input",
    "z_k_qk": "key_qk_input",
    "z_v_pv": "value_pv_input",
    "z_context_wo": "attention_context",
}


def _post_qkv_relu_metadata(layers: list[Any]) -> dict[str, Any]:
    per_layer = []
    for layer_index, layer in enumerate(layers):
        attention = layer.attention
        gate_specs = {
            name: activation_gate_metadata(getattr(attention, f"{name}_relu", None))
            for name in ("query", "key", "value")
        }
        row = {
            "layer": layer_index,
            "query": gate_specs["query"] is not None,
            "key": gate_specs["key"] is not None,
            "value": gate_specs["value"] is not None,
            "qk_placement": getattr(attention, "qk_relu_placement", None),
            "rotary_dim": int(attention.rotary_ndims),
            "head_width": int(attention.head_size),
            "gate_specs": gate_specs,
        }
        per_layer.append(row)

    signatures = {
        (row["query"], row["key"], row["value"], row["qk_placement"])
        for row in per_layer
    }
    if len(signatures) != 1:
        raise ValueError("Post-QKV gate presence and placement must match across all layers.")
    query, key, value, placement = next(iter(signatures))
    if (query or key) and placement not in {"pre_rope", "post_rope"}:
        raise ValueError("Q/K gates require qk_placement pre_rope or post_rope.")

    entries = [
        (row["layer"], {"query": "q", "key": "k", "value": "v"}[name], row["gate_specs"][name])
        for row in per_layer
        for name in ("query", "key", "value")
        if row["gate_specs"][name] is not None
    ]
    _validate_gate_spec_entries(entries, context="Post-QKV")
    gate_specs = {
        name: _summarize_gate_specs(
            [row["gate_specs"][name] for row in per_layer if row["gate_specs"][name] is not None]
        )
        for name in ("query", "key", "value")
    }
    enabled_specs = [spec for _layer, _site, spec in entries]
    common_spec = _summarize_gate_specs(enabled_specs)
    return {
        "enabled": bool(query or key or value),
        "query": query,
        "key": key,
        "value": value,
        "qk_placement": placement,
        "gate_family": common_spec["gate_family"] if common_spec is not None else None,
        "gate_type": common_spec["gate_type"] if common_spec is not None else None,
        "kappa": common_spec.get("kappa") if common_spec is not None else None,
        "gate_specs": gate_specs,
        "layers": per_layer,
    }


def _gate_structure(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        spec["gate_family"],
        spec["gate_type"],
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
        raise ValueError(f"{context} gate family and kappa must match across enabled sites and layers.")


def _summarize_gate_specs(specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not specs:
        return None
    return dict(specs[0])


def _architecture_metadata(
    model: Any,
    *,
    layers: list[Any],
    post_qkv_relu: dict[str, Any],
    block_size: int,
    torch: Any,
) -> dict[str, Any]:
    """Describe the measured GPT-NeoX topology from runtime modules and shapes."""

    if block_size <= 0:
        raise ValueError("Architecture accounting requires a positive sequence length.")

    branch_gate_specs_per_layer = [
        {
            "a": activation_gate_metadata(getattr(layer, "attention_input_relu", None)),
            "m": activation_gate_metadata(getattr(layer, "mlp_input_relu", None)),
            "h": activation_gate_metadata(layer.mlp.act),
        }
        for layer in layers
    ]
    branch_signatures = {
        (
            specs["a"] is not None,
            specs["m"] is not None,
            specs["h"] is not None,
        )
        for specs in branch_gate_specs_per_layer
    }
    if len(branch_signatures) != 1:
        raise ValueError(
            "Branch and MLP-hidden gate presence must match across all layers."
        )
    attention_input, mlp_input, mlp_hidden = next(iter(branch_signatures))
    branch_entries = [
        (layer_index, site, specs[site])
        for layer_index, specs in enumerate(branch_gate_specs_per_layer)
        for site in ("a", "m", "h")
        if specs[site] is not None
    ]
    for site in ("a", "m", "h"):
        _validate_gate_spec_entries(
            [entry for entry in branch_entries if entry[1] == site],
            context=f"Branch site {site}",
        )

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

    active = {
        "a": attention_input,
        "m": mlp_input,
        "h": mlp_hidden,
        "q": bool(post_qkv_relu["query"]),
        "k": bool(post_qkv_relu["key"]),
        "v": bool(post_qkv_relu["value"]),
    }
    active_sites = [
        site for site in ("a", "m", "h", "q", "k", "v") if active[site]
    ]
    placement = post_qkv_relu["qk_placement"] if active["q"] or active["k"] else None
    qkv_gate_specs = post_qkv_relu.get("gate_specs", {})
    gate_specs = {
        "a": _summarize_gate_specs([specs["a"] for specs in branch_gate_specs_per_layer if specs["a"] is not None]),
        "m": _summarize_gate_specs([specs["m"] for specs in branch_gate_specs_per_layer if specs["m"] is not None]),
        "h": _summarize_gate_specs([specs["h"] for specs in branch_gate_specs_per_layer if specs["h"] is not None]),
        "q": qkv_gate_specs.get("query"),
        "k": qkv_gate_specs.get("key"),
        "v": qkv_gate_specs.get("value"),
    }
    qkv_layers = post_qkv_relu.get("layers", [])
    gate_specs_per_layer = []
    for layer_index, branch_specs in enumerate(branch_gate_specs_per_layer):
        attention_specs = qkv_layers[layer_index]["gate_specs"] if qkv_layers else {}
        row = {
            "layer": layer_index,
            "a": branch_specs["a"],
            "m": branch_specs["m"],
            "h": branch_specs["h"],
            "q": attention_specs.get("query"),
            "k": attention_specs.get("key"),
            "v": attention_specs.get("value"),
        }
        gate_specs_per_layer.append(row)

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
        "active_gate_sites": active_sites,
        "gate_presence": active,
        "gate_specs": gate_specs,
        "gate_specs_per_layer": gate_specs_per_layer,
        "qk_gate_placement": placement,
        "hidden_activation": str(getattr(model.config, "hidden_act", "")),
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
