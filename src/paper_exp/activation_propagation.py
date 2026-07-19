from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import yaml

from paper_exp.modeling import activation_gate_metadata, load_checkpoint_model
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_json


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
    # Retained aliases keep the schema consumable by the Report 04 plotting code.
    "query_post_rope",
    "key_post_rope",
    "value",
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

MATMUL_STAGE_ORDER = [
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
]

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
    "query_post_rope": "Legacy alias: actual Q operand of QK^T",
    "key_post_rope": "Legacy alias: actual K operand of QK^T",
    "value": "Legacy alias: actual V operand of PV",
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

_GATE_TO_TARGETABLE_MATMULS = {
    "a": ("qkv_projection",),
    "m": ("mlp_w1",),
    "h": ("mlp_w2",),
    "q_or_k": ("qk_scores",),
    "v": ("probability_value", "attention_output_projection"),
}


def run_activation_propagation(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    torch, np, auto_model, modeling_gpt_neox = _load_dependencies()
    propagation_config = config["activation_propagation"]
    validation_config = config["validation"]
    selected_runs = propagation_config["selected_runs"]

    np.random.seed(int(config["run"]["seed"]))
    source_runs = [_find_source_run(config, item) for item in selected_runs]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    validation_metadata = next(
        (
            (manifest.get("tokenized_data") or {}).get("validation")
            for manifest in source_manifests
            if (manifest.get("tokenized_data") or {}).get("validation") is not None
        ),
        None,
    )
    if validation_metadata is None:
        raise ValueError("Selected source runs have no validation token cache in their manifests.")
    _validate_shared_validation_cache(
        source_manifests,
        validation_metadata,
        selected_runs=selected_runs,
    )
    _validate_requested_validation_cache(validation_config, validation_metadata)
    source_checkpoints = [
        _source_checkpoint(selected, manifest)
        for selected, manifest in zip(selected_runs, source_manifests, strict=True)
    ]

    validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(validation_config["batch_size"])
    eval_batches = validation_config.get("eval_batches")
    starts = _eval_starts(
        validation_tokens,
        block_size,
        eval_batches=eval_batches,
        batch_size=batch_size,
        np=np,
    )

    source_training = yaml.safe_load((source_runs[0] / "config.yaml").read_text(encoding="utf-8"))["training"]
    device = _select_device(torch, source_training.get("device", "auto"))
    dtype = _select_dtype(torch, device, source_training.get("precision", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    experiment_id, numbered_run_id, output_dir = create_run_dir(config, config_path, run_id=run_id)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for selected, source_run, source_manifest, source_checkpoint in zip(
        selected_runs,
        source_runs,
        source_manifests,
        source_checkpoints,
        strict=True,
    ):
        print(f"Measuring activation propagation for {selected['label']} from {source_run}", flush=True)
        results.append(
            _measure_one_run(
                label=str(selected["label"]),
                source_run=source_run,
                source_manifest=source_manifest,
                checkpoint_path=source_checkpoint,
                auto_model=auto_model,
                modeling_gpt_neox=modeling_gpt_neox,
                torch=torch,
                np=np,
                validation_tokens=validation_tokens,
                block_size=block_size,
                batch_size=batch_size,
                starts=starts,
                device=device,
                dtype=dtype,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - started
    validation_sequences = len(starts)
    validation_token_count = validation_sequences * block_size
    validation_cache_tokens = int(len(validation_tokens))
    trailing_tokens_excluded = validation_cache_tokens - validation_token_count
    metrics = {
        "activation_propagation/methods": len(results),
        "activation_propagation/layers": results[0]["num_layers"] if results else 0,
        "activation_propagation/activation_stages": len(ACTIVATION_STAGE_ORDER),
        "activation_propagation/matmul_stages": len(MATMUL_STAGE_ORDER),
        "activation_propagation/validation_batches": (
            results[0]["batches"] if results else 0
        ),
        "activation_propagation/validation_sequences": validation_sequences,
        "activation_propagation/validation_tokens": validation_token_count,
        "activation_propagation/validation_cache_tokens": validation_cache_tokens,
        "activation_propagation/trailing_tokens_excluded": trailing_tokens_excluded,
        "activation_propagation/validation_partition": validation_metadata.get(
            "partition"
        ),
        "activation_propagation/validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "activation_propagation/wall_seconds": wall_seconds,
        "activation_propagation/tokens_per_second": (
            validation_token_count * len(results) / wall_seconds if wall_seconds > 0 else None
        ),
        "activation_propagation/peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "activation_propagation/peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
    }
    for result in results:
        endpoint = result["endpoint"]
        prefix = f"activation_propagation/endpoint/{result['config_id']}"
        metrics.update(
            {
                f"{prefix}/R_block": endpoint["R_block"],
                f"{prefix}/R_model": endpoint["R_model"],
                f"{prefix}/R_block_max": endpoint["R_block_max"],
                f"{prefix}/R_model_max": endpoint["R_model_max"],
                f"{prefix}/U_arch": endpoint["U_arch"],
            }
        )
    if len(results) == 1:
        endpoint = results[0]["endpoint"]
        metrics.update(
            {
                "activation_propagation/R_block": endpoint["R_block"],
                "activation_propagation/R_model": endpoint["R_model"],
                "activation_propagation/R_block_max": endpoint["R_block_max"],
                "activation_propagation/R_model_max": endpoint["R_model_max"],
                "activation_propagation/U_arch": endpoint["U_arch"],
            }
        )

    exact_zero_definition = (
        "An activation element is an exact zero iff its computed tensor value compares equal to numeric 0 "
        "with no tolerance. Counts are integer sums pooled over every evaluated validation sequence, token, "
        "and feature/head element for each layer; attention-probability counts instead pool valid causal "
        "query-key entries. Percentages are not averages of batch percentages."
    )
    matmul_definition = (
        "A zero product opportunity is one scalar multiplication whose activation-side operand, or either "
        "activation operand for QK and PV, is exactly zero. Bias additions, reductions, and realized kernel "
        "speedups are excluded. QK and PV totals include only key positions at or before each query position."
    )
    compute_endpoint_definition = (
        "R_block is the pooled direct zero-product count across QKV, valid-causal QK, valid-causal PV, Wo, "
        "W1, and W2 divided by all products in those block operations. R_model keeps that numerator and "
        "adds the dense hidden-to-vocabulary LM head to the denominator. R_block_max and R_model_max replace "
        "the numerator by the union of operations reachable when every runtime-detected gate output is zero; "
        "Q/K share QK, while V reaches PV and Wo. U_arch is R_model/R_model_max and is undefined for a "
        "zero-ceiling topology. These are logical scalar products, not measured kernel speedups."
    )
    rope_survival_definition = (
        "For PRE-RoPE Q/K gates only, compare each exact-zero gate-output coordinate with the "
        "corresponding actual QK operand after RoPE. Preservation is input-zero/output-zero; "
        "repopulation is input-zero/output-nonzero; creation is input-nonzero/output-zero. "
        "Preservation and repopulation fractions condition on input zeros; creation conditions "
        "on input nonzeros. "
        "Rotary and pass-through coordinates are reported separately. All-zero rotary pairs use "
        "the paired first-half/second-half coordinates mixed by GPT-NeoX rotate_half. POST-RoPE "
        "and absent-gate checkpoints report these placement-specific metrics as unavailable."
    )
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="activation-propagation",
        config_id=experiment_id,
        result_path=output_dir,
    )
    manifest["source_runs"] = [str(path) for path in source_runs]
    manifest["source_checkpoints"] = [str(path) for path in source_checkpoints]
    manifest["source_manifest_statuses"] = [
        source_manifest.get("status") for source_manifest in source_manifests
    ]
    manifest["tokenized_data"] = {"validation": validation_metadata}
    manifest["activation_propagation"] = {
        "selected_runs": selected_runs,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "exact_zero_definition": exact_zero_definition,
        "matmul_zero_product_definition": matmul_definition,
        "compute_endpoint_definition": compute_endpoint_definition,
        "rope_zero_survival_definition": rope_survival_definition,
        "eval_batches": eval_batches,
        "batch_size": batch_size,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_token_count,
        "validation_cache_tokens": validation_cache_tokens,
        "trailing_tokens_excluded": trailing_tokens_excluded,
        "validation_partition": validation_metadata.get("partition"),
        "validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "complete_named_partition": bool(validation_metadata.get("partition"))
        and eval_batches is None,
    }

    payload = {
        "schema_version": 3,
        "validation_batches": results[0]["batches"] if results else 0,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_token_count,
        "validation_cache_tokens": validation_cache_tokens,
        "trailing_tokens_excluded": trailing_tokens_excluded,
        "validation_partition": validation_metadata.get("partition"),
        "validation_partition_hash": validation_metadata.get(
            "source_document_indices_sha256"
        ),
        "complete_named_partition": bool(validation_metadata.get("partition"))
        and eval_batches is None,
        "block_size": block_size,
        "batch_size": batch_size,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "exact_zero_definition": exact_zero_definition,
        "matmul_zero_product_definition": matmul_definition,
        "compute_endpoint_definition": compute_endpoint_definition,
        "rope_zero_survival_definition": rope_survival_definition,
        "activation_stage_order": ACTIVATION_STAGE_ORDER,
        "activation_stage_labels": ACTIVATION_STAGE_LABELS,
        "matmul_stage_order": MATMUL_STAGE_ORDER,
        "matmul_stage_labels": MATMUL_STAGE_LABELS,
        "methods": results,
    }
    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=[])
    write_json(output_dir / "activation_propagation.json", payload)
    return output_dir


def _measure_one_run(
    *,
    label: str,
    source_run: Path,
    source_manifest: dict[str, Any],
    checkpoint_path: Path,
    auto_model: Any,
    modeling_gpt_neox: Any,
    torch: Any,
    np: Any,
    validation_tokens: Any,
    block_size: int,
    batch_size: int,
    starts: list[int],
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    layers = list(model.gpt_neox.layers)
    if not layers:
        raise ValueError("Activation propagation requires at least one GPT-NeoX layer.")
    if not bool(getattr(model.config, "use_parallel_residual", False)):
        raise ValueError("This diagnostic currently describes the Pythia parallel-residual block only.")

    post_qkv_relu = _post_qkv_relu_metadata(layers)
    accumulator = _PropagationAccumulator(torch)
    batches = 0
    method_started = time.perf_counter()
    with _capture_model_propagation(
        model,
        accumulator=accumulator,
        modeling_gpt_neox=modeling_gpt_neox,
        torch=torch,
    ):
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset : offset + batch_size]
                batch = np.stack([validation_tokens[start : start + block_size] for start in batch_starts])
                input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                with _autocast_context(torch, device, dtype):
                    model.gpt_neox(input_ids=input_ids, use_cache=False)
                batches += 1

    activation_rows = accumulator.rows(
        "activations", ACTIVATION_STAGE_ORDER, num_layers=len(layers)
    )
    matmul_rows = accumulator.rows(
        "matmuls", MATMUL_STAGE_ORDER, num_layers=len(layers)
    )
    architecture = _architecture_metadata(
        model,
        layers=layers,
        post_qkv_relu=post_qkv_relu,
        block_size=block_size,
        torch=torch,
    )
    endpoint = _endpoint_summary(
        architecture=architecture,
        activation_rows=activation_rows,
        matmul_rows=matmul_rows,
        validation_tokens=len(starts) * block_size,
    )

    method_result = {
        "label": label,
        "config_id": source_manifest["config_id"],
        "run_id": source_manifest["run_id"],
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "source_manifest_status": source_manifest.get("status"),
        "num_layers": len(layers),
        "use_parallel_residual": True,
        "post_qkv_relu": post_qkv_relu,
        "architecture": architecture,
        "endpoint": endpoint,
        "batches": batches,
        "wall_seconds": time.perf_counter() - method_started,
        "activations": activation_rows,
        "matmuls": matmul_rows,
        "rope_zero_survival": accumulator.rope_survival_rows(num_layers=len(layers)),
        "rope_all_zero_pairs": accumulator.rope_pair_rows(num_layers=len(layers)),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return method_result


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

    gate_spec_signatures = {
        tuple(
            None
            if row["gate_specs"][name] is None
            else (
                row["gate_specs"][name]["gate_family"],
                row["gate_specs"][name]["gate_type"],
                float(row["gate_specs"][name]["kappa"]),
            )
            for name in ("query", "key", "value")
        )
        for row in per_layer
    }
    if len(gate_spec_signatures) != 1:
        raise ValueError("Post-QKV gate family and kappa must match across all layers.")
    gate_specs = dict(per_layer[0]["gate_specs"])
    enabled_specs = [spec for spec in gate_specs.values() if spec is not None]
    common_specs = {
        (spec["gate_family"], spec["gate_type"], float(spec["kappa"]))
        for spec in enabled_specs
    }
    if len(common_specs) > 1:
        raise ValueError("Enabled Q/K/V gates must share one gate family, type, and kappa.")
    common_spec = enabled_specs[0] if enabled_specs else None
    return {
        "enabled": bool(query or key or value),
        "query": query,
        "key": key,
        "value": value,
        "qk_placement": placement,
        "gate_family": common_spec["gate_family"] if common_spec is not None else None,
        "gate_type": common_spec["gate_type"] if common_spec is not None else None,
        "kappa": float(common_spec["kappa"]) if common_spec is not None else None,
        "gate_specs": gate_specs,
        "layers": per_layer,
    }


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
    branch_spec_signatures = {
        tuple(
            None
            if specs[site] is None
            else (
                specs[site]["gate_family"],
                specs[site]["gate_type"],
                float(specs[site]["kappa"]),
            )
            for site in ("a", "m", "h")
        )
        for specs in branch_gate_specs_per_layer
    }
    if len(branch_spec_signatures) != 1:
        raise ValueError("Branch gate family and kappa must match across all layers.")

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
        "a": branch_gate_specs_per_layer[0]["a"],
        "m": branch_gate_specs_per_layer[0]["m"],
        "h": branch_gate_specs_per_layer[0]["h"],
        "q": qkv_gate_specs.get("query"),
        "k": qkv_gate_specs.get("key"),
        "v": qkv_gate_specs.get("value"),
    }

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
        "topology_id": _topology_id(active, placement),
        "active_gate_sites": active_sites,
        "gate_presence": active,
        "gate_specs": gate_specs,
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


def _topology_id(active: dict[str, bool], placement: str | None) -> str:
    sites = frozenset(site for site, enabled in active.items() if enabled)
    fixed = {
        frozenset(): "A0",
        frozenset({"h"}): "A1-H",
        frozenset({"a", "m", "h"}): "A3",
        frozenset({"a", "m", "h", "v"}): "A4-V",
    }
    if sites in fixed:
        return fixed[sites]
    if sites == frozenset({"a", "m", "h", "q"}) and placement == "post_rope":
        return "A4-Q"
    if sites == frozenset({"a", "m", "h", "k"}) and placement == "post_rope":
        return "A4-K"
    if sites == frozenset({"a", "m", "h", "q", "k"}):
        return "A5-QK-PRE" if placement == "pre_rope" else "A5-QK-POST"
    if sites == frozenset({"a", "m", "h", "q", "k", "v"}):
        return "A6-PRE" if placement == "pre_rope" else "A6-POST"
    custom = "custom:" + ",".join(
        site for site in ("a", "m", "h", "q", "k", "v") if site in sites
    )
    return f"{custom}@{placement}" if sites & {"q", "k"} else custom


def _endpoint_summary(
    *,
    architecture: dict[str, Any],
    activation_rows: list[dict[str, Any]],
    matmul_rows: list[dict[str, Any]],
    validation_tokens: int,
) -> dict[str, Any]:
    """Reduce direct integer counters to architecture-aware campaign endpoints."""

    block_size = int(architecture["sequence_length"])
    num_layers = int(architecture["num_layers"])
    if validation_tokens <= 0 or validation_tokens % block_size:
        raise ValueError(
            "Endpoint accounting requires a positive whole number of fixed-length sequences."
        )
    sequences = validation_tokens // block_size

    per_operation: dict[str, dict[str, Any]] = {}
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
        per_operation[stage] = {
            "zero_product_count": zero_count,
            "product_count": total,
            "zero_product_fraction": zero_count / total,
        }

    block_zero_count = sum(row["zero_product_count"] for row in per_operation.values())
    block_product_count = sum(row["product_count"] for row in per_operation.values())
    lm_head_product_count = (
        int(validation_tokens)
        * int(architecture["hidden_size"])
        * int(architecture["vocab_size"])
    )
    model_product_count = block_product_count + lm_head_product_count

    gates = architecture["gate_presence"]
    targetable_stages: list[str] = []
    for site in ("a", "m", "h"):
        if gates[site]:
            targetable_stages.extend(_GATE_TO_TARGETABLE_MATMULS[site])
    if gates["q"] or gates["k"]:
        targetable_stages.extend(_GATE_TO_TARGETABLE_MATMULS["q_or_k"])
    if gates["v"]:
        targetable_stages.extend(_GATE_TO_TARGETABLE_MATMULS["v"])
    targetable_stages = [
        stage for stage in MATMUL_STAGE_ORDER if stage in set(targetable_stages)
    ]
    ceiling_count = sum(
        int(per_operation[stage]["product_count"]) for stage in targetable_stages
    )
    r_block = block_zero_count / block_product_count
    r_model = block_zero_count / model_product_count
    r_block_max = ceiling_count / block_product_count
    r_model_max = ceiling_count / model_product_count

    zero_sites = {
        alias: _pooled_activation_stage(activation_rows, stage, num_layers=num_layers)
        for alias, stage in ENDPOINT_ZERO_STAGES.items()
    }
    return {
        "architecture_id": architecture["topology_id"],
        "active_gate_sites": list(architecture["active_gate_sites"]),
        "validation_sequences": sequences,
        "validation_tokens": int(validation_tokens),
        "R_block": r_block,
        "R_model": r_model,
        "R_block_max": r_block_max,
        "R_model_max": r_model_max,
        "U_arch": r_model / r_model_max if r_model_max > 0.0 else None,
        "block_zero_product_count": block_zero_count,
        "block_product_count": block_product_count,
        "lm_head_product_count": lm_head_product_count,
        "model_product_count": model_product_count,
        "architecture_ceiling_zero_product_count": ceiling_count,
        "targetable_matmul_stages": targetable_stages,
        "per_operation": per_operation,
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


def _unavailable_rope_survival_row(
    layer: int,
    operand: str,
    region: str,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "operand": operand,
        "region": region,
        "available": False,
        "unavailable_reason": reason,
        "input_zero_count": None,
        "output_zero_count": None,
        "preserved_zero_count": None,
        "repopulated_zero_count": None,
        "created_zero_count": None,
        "total": None,
        "input_exact_zero_fraction": None,
        "output_exact_zero_fraction": None,
        "zero_preservation_fraction": None,
        "zero_repopulation_fraction": None,
        "zero_creation_fraction": None,
    }


class _PropagationAccumulator:
    def __init__(self, torch: Any):
        self.torch = torch
        self.activations: dict[tuple[int, str], list[int]] = {}
        self.matmuls: dict[tuple[int, str], list[int]] = {}
        self.unavailable: dict[str, dict[tuple[int, str], str]] = {
            "activations": {},
            "matmuls": {},
        }
        self.gate_metadata: dict[int, dict[str, Any]] = {}
        self.pending_gate_outputs: dict[tuple[int, str], Any] = {}
        self.rope_survival: dict[tuple[int, str, str], list[int]] = {}
        self.rope_pairs: dict[tuple[int, str], list[int]] = {}

    def add_activation(self, name: str, layer: int, value: Any) -> None:
        self.add_counts("activations", name, layer, *_exact_zero_counts(value, torch=self.torch))

    def add_linear_matmul(self, name: str, layer: int, value: Any, output_features: int) -> None:
        self.add_counts(
            "matmuls",
            name,
            layer,
            *_linear_zero_product_counts(value, output_features=output_features, torch=self.torch),
        )

    def add_counts(self, kind: str, name: str, layer: int, zero_count: int, total: int) -> None:
        if (int(layer), name) in self.unavailable[kind]:
            raise ValueError(f"Cannot add counts for unavailable {kind} stage {name!r} in layer {layer}.")
        counts = getattr(self, kind)
        current = counts.setdefault((int(layer), name), [0, 0])
        current[0] += int(zero_count)
        current[1] += int(total)

    def mark_unavailable(self, kind: str, name: str, layer: int, reason: str) -> None:
        key = (int(layer), name)
        if key in getattr(self, kind):
            raise ValueError(f"Cannot mark measured {kind} stage {name!r} unavailable in layer {layer}.")
        self.unavailable[kind][key] = str(reason)

    def set_gate_metadata(
        self,
        layer: int,
        *,
        qk_placement: str | None,
        query: bool,
        key: bool,
        value: bool,
        rotary_dim: int,
        head_width: int,
    ) -> None:
        self.gate_metadata[int(layer)] = {
            "qk_placement": qk_placement,
            "query": bool(query),
            "key": bool(key),
            "value": bool(value),
            "rotary_dim": int(rotary_dim),
            "head_width": int(head_width),
        }

    def remember_gate_output(self, name: str, layer: int, value: Any) -> None:
        self.pending_gate_outputs[(int(layer), name)] = value.detach()

    def add_rope_survival_from_actual_operand(
        self,
        name: str,
        layer: int,
        actual_operand: Any,
    ) -> None:
        metadata = self.gate_metadata[int(layer)]
        if metadata["qk_placement"] != "pre_rope" or not metadata[name]:
            return
        gate_output = self.pending_gate_outputs.pop((int(layer), name), None)
        if gate_output is None:
            raise RuntimeError(
                f"Missing pending {name} gate output for PRE-RoPE survival measurement in layer {layer}."
            )
        output = actual_operand.detach()
        if gate_output.shape != output.shape:
            raise ValueError(
                f"PRE-RoPE {name} tensors must have matching shapes, got "
                f"{tuple(gate_output.shape)} and {tuple(output.shape)}."
            )
        rotary_dim = metadata["rotary_dim"]
        head_width = metadata["head_width"]
        if output.shape[-1] != head_width or not 0 <= rotary_dim <= head_width:
            raise ValueError("Invalid rotary/head dimensions for PRE-RoPE survival measurement.")

        for region, start, stop in (
            ("rotary", 0, rotary_dim),
            ("passthrough", rotary_dim, head_width),
        ):
            before = gate_output[..., start:stop]
            after = output[..., start:stop]
            before_zero = before == 0
            after_zero = after == 0
            counts = self.rope_survival.setdefault(
                (int(layer), name, region), [0, 0, 0, 0, 0, 0]
            )
            batch_counts = self.torch.stack(
                (
                    self.torch.count_nonzero(before_zero),
                    self.torch.count_nonzero(after_zero),
                    self.torch.count_nonzero(before_zero & after_zero),
                    self.torch.count_nonzero(before_zero & ~after_zero),
                    self.torch.count_nonzero(~before_zero & after_zero),
                )
            ).cpu().tolist()
            for index, count in enumerate(batch_counts):
                counts[index] += int(count)
            counts[5] += int(before.numel())

        if rotary_dim:
            if rotary_dim % 2:
                raise ValueError("Rotary width must be even for rotary-pair accounting.")
            half = rotary_dim // 2
            rotary_zero = gate_output[..., :rotary_dim] == 0
            all_zero_pairs = rotary_zero[..., :half] & rotary_zero[..., half:]
            output_rotary_zero = output[..., :rotary_dim] == 0
            output_all_zero_pairs = (
                output_rotary_zero[..., :half] & output_rotary_zero[..., half:]
            )
            pair_counts = self.rope_pairs.setdefault((int(layer), name), [0, 0, 0])
            batch_pair_counts = self.torch.stack(
                (
                    self.torch.count_nonzero(all_zero_pairs),
                    self.torch.count_nonzero(output_all_zero_pairs),
                )
            ).cpu().tolist()
            pair_counts[0] += int(batch_pair_counts[0])
            pair_counts[1] += int(batch_pair_counts[1])
            pair_counts[2] += int(all_zero_pairs.numel())

    def rows(
        self,
        kind: str,
        order: list[str],
        *,
        num_layers: int | None = None,
    ) -> list[dict[str, Any]]:
        counts = getattr(self, kind)
        order_index = {name: index for index, name in enumerate(order)}
        unavailable = self.unavailable[kind]
        if num_layers is None:
            keys = set(counts) | set(unavailable)
        else:
            keys = {(layer, name) for layer in range(num_layers) for name in order}
        rows: list[dict[str, Any]] = []
        for layer, name in sorted(keys, key=lambda key: (key[0], order_index[key[1]])):
            key = (layer, name)
            if key in unavailable:
                rows.append(
                    {
                        "name": name,
                        "layer": layer,
                        "available": False,
                        "unavailable_reason": unavailable[key],
                        "zero_count": None,
                        "total": None,
                        "exact_zero_fraction": None,
                    }
                )
                continue
            if key not in counts:
                raise RuntimeError(f"Missing required {kind} stage {name!r} in layer {layer}.")
            zero_count, total = counts[key]
            rows.append(
                {
                    "name": name,
                    "layer": layer,
                    "available": True,
                    "zero_count": zero_count,
                    "total": total,
                    "exact_zero_fraction": zero_count / total if total else None,
                }
            )
        return rows

    def rope_survival_rows(self, *, num_layers: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(num_layers):
            metadata = self.gate_metadata[layer]
            for name in ("query", "key"):
                for region in ("rotary", "passthrough"):
                    key = (layer, name, region)
                    if metadata["qk_placement"] != "pre_rope" or not metadata[name]:
                        rows.append(
                            _unavailable_rope_survival_row(
                                layer,
                                name,
                                region,
                                reason=(
                                    "gate_absent"
                                    if not metadata[name]
                                    else "qk_gate_is_post_rope"
                                ),
                            )
                        )
                        continue
                    if key not in self.rope_survival:
                        raise RuntimeError(
                            f"Missing PRE-RoPE survival counts for {name} {region} in layer {layer}."
                        )
                    before_zero, after_zero, preserved, repopulated, created, total = (
                        self.rope_survival[key]
                    )
                    before_nonzero = total - before_zero
                    rows.append(
                        {
                            "layer": layer,
                            "operand": name,
                            "region": region,
                            "available": True,
                            "input_zero_count": before_zero,
                            "output_zero_count": after_zero,
                            "preserved_zero_count": preserved,
                            "repopulated_zero_count": repopulated,
                            "created_zero_count": created,
                            "total": total,
                            "input_exact_zero_fraction": before_zero / total if total else None,
                            "output_exact_zero_fraction": after_zero / total if total else None,
                            "zero_preservation_fraction": (
                                preserved / before_zero if before_zero else None
                            ),
                            "zero_repopulation_fraction": (
                                repopulated / before_zero if before_zero else None
                            ),
                            "zero_creation_fraction": (
                                created / before_nonzero if before_nonzero else None
                            ),
                        }
                    )
        return rows

    def rope_pair_rows(self, *, num_layers: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(num_layers):
            metadata = self.gate_metadata[layer]
            for name in ("query", "key"):
                key = (layer, name)
                if metadata["qk_placement"] != "pre_rope" or not metadata[name]:
                    rows.append(
                        {
                            "layer": layer,
                            "operand": name,
                            "available": False,
                            "unavailable_reason": (
                                "gate_absent"
                                if not metadata[name]
                                else "qk_gate_is_post_rope"
                            ),
                            "input_all_zero_pair_count": None,
                            "output_all_zero_pair_count": None,
                            "pair_total": None,
                            "input_all_zero_pair_fraction": None,
                            "output_all_zero_pair_fraction": None,
                        }
                    )
                    continue
                if metadata["rotary_dim"] == 0:
                    rows.append(
                        {
                            "layer": layer,
                            "operand": name,
                            "available": False,
                            "unavailable_reason": "no_rotary_coordinates",
                            "input_all_zero_pair_count": None,
                            "output_all_zero_pair_count": None,
                            "pair_total": None,
                            "input_all_zero_pair_fraction": None,
                            "output_all_zero_pair_fraction": None,
                        }
                    )
                    continue
                if key not in self.rope_pairs:
                    raise RuntimeError(
                        f"Missing PRE-RoPE all-zero-pair counts for {name} in layer {layer}."
                    )
                input_pairs, output_pairs, pair_total = self.rope_pairs[key]
                rows.append(
                    {
                        "layer": layer,
                        "operand": name,
                        "available": True,
                        "input_all_zero_pair_count": input_pairs,
                        "output_all_zero_pair_count": output_pairs,
                        "pair_total": pair_total,
                        "input_all_zero_pair_fraction": (
                            input_pairs / pair_total if pair_total else None
                        ),
                        "output_all_zero_pair_fraction": (
                            output_pairs / pair_total if pair_total else None
                        ),
                    }
                )
        return rows


@contextmanager
def _capture_model_propagation(
    model: Any,
    *,
    accumulator: _PropagationAccumulator,
    modeling_gpt_neox: Any,
    torch: Any,
) -> Iterator[None]:
    handles = []
    try:
        for layer_index, layer in enumerate(model.gpt_neox.layers):
            attention_relu = getattr(layer, "attention_input_relu", None)
            mlp_relu = getattr(layer, "mlp_input_relu", None)
            hidden_relu = activation_gate_metadata(layer.mlp.act) is not None

            attention = layer.attention
            query_relu = getattr(attention, "query_relu", None)
            key_relu = getattr(attention, "key_relu", None)
            value_relu = getattr(attention, "value_relu", None)
            qk_placement = getattr(attention, "qk_relu_placement", None)
            if (query_relu is not None or key_relu is not None) and qk_placement not in {
                "pre_rope",
                "post_rope",
            }:
                raise ValueError(
                    f"Layer {layer_index} has Q/K ReLU modules but no valid qk_relu_placement."
                )
            head_width = int(attention.head_size)
            rotary_dim = int(attention.rotary_ndims)
            accumulator.set_gate_metadata(
                layer_index,
                qk_placement=qk_placement,
                query=query_relu is not None,
                key=key_relu is not None,
                value=value_relu is not None,
                rotary_dim=rotary_dim,
                head_width=head_width,
            )

            for gate_name, gate_module in (
                ("query", query_relu),
                ("key", key_relu),
                ("value", value_relu),
            ):
                input_stage = f"{gate_name}_gate_input"
                output_stage = f"{gate_name}_gate_output"
                if gate_module is None:
                    accumulator.mark_unavailable(
                        "activations", input_stage, layer_index, "post_qkv_gate_absent"
                    )
                    accumulator.mark_unavailable(
                        "activations", output_stage, layer_index, "post_qkv_gate_absent"
                    )
                    continue
                handles.append(
                    gate_module.register_forward_pre_hook(
                        _activation_pre_hook(accumulator, input_stage, layer_index)
                    )
                )
                handles.append(
                    gate_module.register_forward_hook(
                        _gate_output_hook(
                            accumulator,
                            output_stage,
                            gate_name,
                            layer_index,
                            remember_for_rope=(
                                gate_name in {"query", "key"}
                                and qk_placement == "pre_rope"
                            ),
                        )
                    )
                )

            handles.append(
                layer.register_forward_pre_hook(
                    _activation_pre_hook(accumulator, "residual_input", layer_index)
                )
            )
            handles.append(
                layer.register_forward_hook(
                    _activation_output_hook(accumulator, "residual_output", layer_index)
                )
            )
            if attention_relu is None:
                accumulator.mark_unavailable(
                    "activations",
                    "attention_input_relu",
                    layer_index,
                    "post_layernorm_relu_absent",
                )
                handles.append(
                    layer.input_layernorm.register_forward_hook(
                        _activation_output_hook(
                            accumulator, "attention_layernorm_raw", layer_index
                        )
                    )
                )
            else:
                # The architecture invokes this explicit ReLU from a LayerNorm
                # output hook. Its pre-hook is therefore the only placement-safe
                # way to capture the raw LayerNorm tensor before rectification.
                handles.append(
                    attention_relu.register_forward_pre_hook(
                        _activation_pre_hook(
                            accumulator, "attention_layernorm_raw", layer_index
                        )
                    )
                )
                handles.append(
                    attention_relu.register_forward_hook(
                        _activation_output_hook(
                            accumulator, "attention_input_relu", layer_index
                        )
                    )
                )

            if mlp_relu is None:
                accumulator.mark_unavailable(
                    "activations",
                    "mlp_input_relu",
                    layer_index,
                    "post_layernorm_relu_absent",
                )
                handles.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        _activation_output_hook(
                            accumulator, "mlp_layernorm_raw", layer_index
                        )
                    )
                )
            else:
                handles.append(
                    mlp_relu.register_forward_pre_hook(
                        _activation_pre_hook(
                            accumulator, "mlp_layernorm_raw", layer_index
                        )
                    )
                )
                handles.append(
                    mlp_relu.register_forward_hook(
                        _activation_output_hook(
                            accumulator, "mlp_input_relu", layer_index
                        )
                    )
                )
            handles.append(
                layer.attention.register_forward_hook(
                    _attention_output_hook(accumulator, layer_index)
                )
            )
            handles.append(
                layer.mlp.dense_h_to_4h.register_forward_hook(
                    _activation_output_hook(accumulator, "mlp_w1_preactivation", layer_index)
                )
            )
            if hidden_relu:
                handles.append(
                    layer.mlp.act.register_forward_hook(
                        _activation_output_hook(
                            accumulator, "mlp_hidden_relu", layer_index
                        )
                    )
                )
            else:
                accumulator.mark_unavailable(
                    "activations",
                    "mlp_hidden_relu",
                    layer_index,
                    "mlp_hidden_relu_absent",
                )
            handles.append(
                layer.mlp.dense_4h_to_h.register_forward_hook(
                    _activation_output_hook(accumulator, "mlp_output", layer_index)
                )
            )

            handles.append(
                layer.attention.query_key_value.register_forward_pre_hook(
                    _linear_pre_hook(accumulator, "qkv_projection", layer_index)
                )
            )
            handles.append(
                layer.attention.query_key_value.register_forward_hook(
                    _qkv_projection_output_hook(
                        accumulator, layer_index, attention=layer.attention
                    )
                )
            )
            handles.append(
                layer.attention.dense.register_forward_pre_hook(
                    _linear_pre_hook(accumulator, "attention_output_projection", layer_index)
                )
            )
            handles.append(
                layer.mlp.dense_h_to_4h.register_forward_pre_hook(
                    _linear_pre_hook(accumulator, "mlp_w1", layer_index)
                )
            )
            handles.append(
                layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                    _linear_pre_hook(accumulator, "mlp_w2", layer_index)
                )
            )

        with _patched_eager_attention(
            model,
            accumulator=accumulator,
            modeling_gpt_neox=modeling_gpt_neox,
            torch=torch,
        ):
            yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _patched_eager_attention(
    model: Any,
    *,
    accumulator: _PropagationAccumulator,
    modeling_gpt_neox: Any,
    torch: Any,
) -> Iterator[None]:
    original_eager_attention = modeling_gpt_neox.eager_attention_forward
    original_implementation = model.config._attn_implementation

    def instrumented_eager_attention(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float | int = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        context, probabilities = original_eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        layer_index = int(module.layer_idx)
        for stage, legacy_stage, operand in (
            ("query_qk_input", "query_post_rope", query),
            ("key_qk_input", "key_post_rope", key),
            ("value_pv_input", "value", value),
        ):
            zero_count, total = _exact_zero_counts(operand, torch=torch)
            accumulator.add_counts(
                "activations", stage, layer_index, zero_count, total
            )
            accumulator.add_counts(
                "activations", legacy_stage, layer_index, zero_count, total
            )
        accumulator.add_rope_survival_from_actual_operand("query", layer_index, query)
        accumulator.add_rope_survival_from_actual_operand("key", layer_index, key)
        zero_count, total = _valid_causal_exact_zero_counts(probabilities, torch=torch)
        accumulator.add_counts(
            "activations", "attention_probabilities", layer_index, zero_count, total
        )
        accumulator.add_activation("attention_context", layer_index, context)
        accumulator.add_counts(
            "matmuls",
            "qk_scores",
            layer_index,
            *_qk_zero_product_counts(query, key, torch=torch),
        )
        accumulator.add_counts(
            "matmuls",
            "probability_value",
            layer_index,
            *_probability_value_zero_product_counts(probabilities, value, torch=torch),
        )
        return context, probabilities

    modeling_gpt_neox.eager_attention_forward = instrumented_eager_attention
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = original_implementation
        modeling_gpt_neox.eager_attention_forward = original_eager_attention


def _activation_pre_hook(
    accumulator: _PropagationAccumulator, name: str, layer: int
) -> Any:
    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        accumulator.add_activation(name, layer, inputs[0])

    return hook


def _activation_output_hook(
    accumulator: _PropagationAccumulator, name: str, layer: int
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        accumulator.add_activation(name, layer, output)

    return hook


def _gate_output_hook(
    accumulator: _PropagationAccumulator,
    stage_name: str,
    gate_name: str,
    layer: int,
    *,
    remember_for_rope: bool,
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        accumulator.add_activation(stage_name, layer, output)
        if remember_for_rope:
            accumulator.remember_gate_output(gate_name, layer, output)

    return hook


def _qkv_projection_output_hook(
    accumulator: _PropagationAccumulator, layer: int, *, attention: Any
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        query, key, value = _split_fused_qkv_projection(
            output,
            num_heads=int(attention.config.num_attention_heads),
            head_width=int(attention.head_size),
        )
        accumulator.add_activation("query_projection_output", layer, query)
        accumulator.add_activation("key_projection_output", layer, key)
        accumulator.add_activation("value_projection_output", layer, value)

    return hook


def _split_fused_qkv_projection(
    value: Any, *, num_heads: int, head_width: int
) -> tuple[Any, Any, Any]:
    """Reproduce GPT-NeoX's per-head fused-QKV layout without changing execution."""
    expected_width = 3 * num_heads * head_width
    if value.ndim != 3 or int(value.shape[-1]) != expected_width:
        raise ValueError(
            "Unexpected fused GPT-NeoX QKV projection shape: "
            f"expected last width {expected_width}, got {tuple(value.shape)}."
        )
    hidden_shape = (*value.shape[:-1], num_heads, 3 * head_width)
    fused_by_head = value.view(hidden_shape).transpose(1, 2)
    return fused_by_head.chunk(3, dim=-1)


def _attention_output_hook(accumulator: _PropagationAccumulator, layer: int) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
        accumulator.add_activation("attention_output", layer, output[0])

    return hook


def _linear_pre_hook(accumulator: _PropagationAccumulator, name: str, layer: int) -> Any:
    def hook(module: Any, inputs: tuple[Any, ...]) -> None:
        accumulator.add_linear_matmul(name, layer, inputs[0], int(module.out_features))

    return hook


def _exact_zero_counts(value: Any, *, torch: Any) -> tuple[int, int]:
    detached = value.detach()
    return int(torch.count_nonzero(detached == 0).cpu()), int(detached.numel())


def _linear_zero_product_counts(
    value: Any, *, output_features: int, torch: Any
) -> tuple[int, int]:
    zero_inputs, input_total = _exact_zero_counts(value, torch=torch)
    return zero_inputs * int(output_features), input_total * int(output_features)


def _valid_causal_exact_zero_counts(
    probabilities: Any, *, torch: Any, query_chunk_size: int = 128
) -> tuple[int, int]:
    batch, heads, queries, keys = probabilities.shape
    if queries != keys:
        raise ValueError("Activation propagation expects equal query and key lengths without a cache.")
    key_positions = torch.arange(keys, device=probabilities.device)
    zero_count = 0
    for start in range(0, queries, query_chunk_size):
        stop = min(start + query_chunk_size, queries)
        valid = key_positions.unsqueeze(0) <= torch.arange(
            start, stop, device=probabilities.device
        ).unsqueeze(1)
        chunk_zeros = probabilities[..., start:stop, :] == 0
        zero_count += int(torch.count_nonzero(chunk_zeros & valid).cpu())
    total = int(batch * heads * queries * (queries + 1) // 2)
    return zero_count, total


def _qk_zero_product_counts(query: Any, key: Any, *, torch: Any) -> tuple[int, int]:
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("QK zero-product counting expects matching [batch, heads, tokens, width] tensors.")
    batch, heads, tokens, width = query.shape
    query_nonzero = query.detach() != 0
    cumulative_key_nonzero = (key.detach() != 0).to(torch.int64).cumsum(dim=-2)
    nonzero_products = int(
        (query_nonzero.to(torch.int64) * cumulative_key_nonzero).sum().cpu()
    )
    total = int(batch * heads * width * tokens * (tokens + 1) // 2)
    return total - nonzero_products, total


def _probability_value_zero_product_counts(
    probabilities: Any,
    value: Any,
    *,
    torch: Any,
    query_chunk_size: int = 128,
) -> tuple[int, int]:
    if probabilities.ndim != 4 or value.ndim != 4:
        raise ValueError("PV zero-product counting expects rank-four probability and value tensors.")
    batch, heads, queries, keys = probabilities.shape
    value_batch, value_heads, value_keys, width = value.shape
    if (batch, heads, keys) != (value_batch, value_heads, value_keys) or queries != keys:
        raise ValueError("PV zero-product counting expects matching uncached causal-attention shapes.")

    key_positions = torch.arange(keys, device=probabilities.device)
    value_nonzero_dimensions = torch.count_nonzero(value.detach(), dim=-1).to(torch.int64)
    nonzero_products = 0
    for start in range(0, queries, query_chunk_size):
        stop = min(start + query_chunk_size, queries)
        valid = key_positions.unsqueeze(0) <= torch.arange(
            start, stop, device=probabilities.device
        ).unsqueeze(1)
        probability_nonzero = probabilities[..., start:stop, :].detach() != 0
        valid_probability_nonzero = probability_nonzero & valid
        nonzero_products += int(
            (
                valid_probability_nonzero.to(torch.int64)
                * value_nonzero_dimensions.unsqueeze(-2)
            )
            .sum()
            .cpu()
        )
    total = int(batch * heads * width * queries * (queries + 1) // 2)
    return total - nonzero_products, total


def _validate_shared_validation_cache(
    source_manifests: list[dict[str, Any]],
    reference: dict[str, Any],
    *,
    selected_runs: list[dict[str, Any]] | None = None,
) -> None:
    reference_key = (reference["tokens_path"], reference["block_size"], reference["tokens"])
    selected = selected_runs or [{} for _ in source_manifests]
    available_metadata = [reference]
    for manifest, selection in zip(source_manifests, selected, strict=True):
        candidate = (manifest.get("tokenized_data") or {}).get("validation")
        if candidate is None:
            if bool(selection.get("allow_incomplete_source", False)):
                continue
            raise ValueError(f"Source run {manifest['config_id']} has no validation token cache.")
        candidate_key = (candidate["tokens_path"], candidate["block_size"], candidate["tokens"])
        if candidate_key != reference_key:
            raise ValueError("Selected runs do not share the same validation token cache.")
        available_metadata.append(candidate)

    # Historical manifests do not necessarily carry partition or file hashes,
    # so missing optional identity fields remain compatible. Any identities
    # that are present must nevertheless agree across every selected source.
    identity_fields = (
        "partition",
        "partition_scheme",
        "partition_seed",
        "source_document_indices_sha256",
        "tokens_sha256",
    )
    for field in identity_fields:
        observed = {
            metadata[field]
            for metadata in available_metadata
            if metadata.get(field) is not None
        }
        if len(observed) > 1:
            raise ValueError(
                "Selected runs do not share the same validation token cache "
                f"identity field {field}."
            )


def _validate_requested_validation_cache(
    validation_config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Bind campaign diagnostics to the complete named validation partition."""

    requested_partition = validation_config.get("partition")
    if requested_partition is None:
        return
    if metadata.get("partition") != requested_partition:
        raise ValueError(
            "Activation propagation validation partition does not match the source cache."
        )
    expected_fields = {
        "partition_scheme": validation_config.get("partition_scheme"),
        "partition_seed": validation_config.get("partition_seed"),
        "max_documents": validation_config.get("max_documents"),
    }
    for field, expected in expected_fields.items():
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(
                f"Activation propagation validation {field} does not match the source cache."
            )
    requested_hash = validation_config.get("partition_hash")
    realized_hash = metadata.get("source_document_indices_sha256")
    if requested_hash is not None and realized_hash != requested_hash:
        raise ValueError(
            "Activation propagation validation partition hash does not match the source cache."
        )
    if validation_config.get("eval_batches") is not None:
        raise ValueError(
            "Named-partition activation propagation must evaluate the complete partition; "
            "set validation.eval_batches to null."
        )


def _find_source_run(config: dict[str, Any], selected: dict[str, Any]) -> Path:
    config_id = str(selected["config_id"])
    run_id = selected.get("run_id")
    if run_id is None:
        return _find_latest_source_run(config, config_id)

    run_dir = Path(config["output"]["dir"]) / config_id / str(run_id)
    if not run_dir.is_dir() or not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing explicitly selected source run: {run_dir}")
    return run_dir


def _source_checkpoint(selected: dict[str, Any], manifest: dict[str, Any]) -> Path:
    override = selected.get("checkpoint_path")
    if override is not None:
        if not bool(selected.get("allow_incomplete_source", False)):
            raise ValueError(
                "activation_propagation checkpoint_path overrides require "
                "allow_incomplete_source: true."
            )
        checkpoint_path = Path(str(override))
    else:
        checkpoint = manifest.get("checkpoint") or {}
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if not bool(checkpoint.get("saved")):
            raise ValueError(
                f"Source run {manifest.get('config_id')} has no saved checkpoint in its manifest."
            )

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
    return checkpoint_path


def _find_latest_source_run(config: dict[str, Any], config_id: str) -> Path:
    experiment_dir = Path(config["output"]["dir"]) / config_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing result directory for selected config: {experiment_dir}")
    candidates = []
    for run_dir in sorted(experiment_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        checkpoint = manifest.get("checkpoint") or {}
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if bool(checkpoint.get("saved")) and checkpoint_path.exists():
            candidates.append((int(manifest.get("run_sequence", 0)), run_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpointed runs found for selected config: {experiment_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _eval_starts(
    tokens: Any,
    block_size: int,
    *,
    eval_batches: int | None,
    batch_size: int,
    np: Any,
) -> list[int]:
    if eval_batches is None:
        total_blocks = max(1, (len(tokens) - 1) // block_size)
        return [index * block_size for index in range(total_blocks)]
    max_start = len(tokens) - block_size - 1
    return list(np.random.randint(0, max_start, size=int(eval_batches) * batch_size))


def _autocast_context(torch: Any, device: Any, dtype: Any) -> Any:
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype(torch: Any, device: Any, requested: str) -> Any:
    if requested == "float32" or device.type != "cuda":
        return None
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown precision: {requested}")


def _peak_gpu_memory_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _peak_gpu_reserved_mb(torch: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024 * 1024)


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.gpt_neox import modeling_gpt_neox
    except ImportError as exc:
        raise RuntimeError(
            "Activation propagation analysis requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoModelForCausalLM, modeling_gpt_neox
