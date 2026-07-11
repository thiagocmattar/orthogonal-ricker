from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import yaml

from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_json


ACTIVATION_STAGE_ORDER = [
    "residual_input",
    "attention_layernorm_raw",
    "attention_input_relu",
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
    "attention_layernorm_raw": "LN_attn(H_l), before ReLU",
    "attention_input_relu": "ReLU(LN_attn(H_l))",
    "query_post_rope": "Q after RoPE",
    "key_post_rope": "K after RoPE",
    "value": "V",
    "attention_probabilities": "P = softmax(masked QK^T)",
    "attention_context": "C = PV",
    "attention_output": "O = C W_o + b_o",
    "mlp_layernorm_raw": "LN_mlp(H_l), before ReLU",
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
    source_runs = [_find_latest_source_run(config, item["config_id"]) for item in selected_runs]
    source_manifests = [read_json(path / "manifest.json") for path in source_runs]
    validation_metadata = source_manifests[0]["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")
    _validate_shared_validation_cache(source_manifests, validation_metadata)

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

    for selected, source_run in zip(selected_runs, source_runs, strict=True):
        print(f"Measuring activation propagation for {selected['label']} from {source_run}", flush=True)
        results.append(
            _measure_one_run(
                label=str(selected["label"]),
                source_run=source_run,
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
        "activation_propagation/wall_seconds": wall_seconds,
        "activation_propagation/tokens_per_second": (
            validation_token_count * len(results) / wall_seconds if wall_seconds > 0 else None
        ),
        "activation_propagation/peak_gpu_memory_mb": _peak_gpu_memory_mb(torch, device),
        "activation_propagation/peak_gpu_reserved_mb": _peak_gpu_reserved_mb(torch, device),
    }

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
    manifest["source_checkpoints"] = [
        str(Path(source_manifest["checkpoint"]["path"])) for source_manifest in source_manifests
    ]
    manifest["tokenized_data"] = {"validation": validation_metadata}
    manifest["activation_propagation"] = {
        "selected_runs": selected_runs,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "exact_zero_definition": exact_zero_definition,
        "matmul_zero_product_definition": matmul_definition,
        "eval_batches": eval_batches,
        "batch_size": batch_size,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_token_count,
        "validation_cache_tokens": validation_cache_tokens,
        "trailing_tokens_excluded": trailing_tokens_excluded,
    }

    payload = {
        "schema_version": 1,
        "validation_batches": results[0]["batches"] if results else 0,
        "validation_sequences": validation_sequences,
        "validation_tokens": validation_token_count,
        "validation_cache_tokens": validation_cache_tokens,
        "trailing_tokens_excluded": trailing_tokens_excluded,
        "block_size": block_size,
        "batch_size": batch_size,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "exact_zero_definition": exact_zero_definition,
        "matmul_zero_product_definition": matmul_definition,
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
    source_manifest = read_json(source_run / "manifest.json")
    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    layers = list(model.gpt_neox.layers)
    if not layers:
        raise ValueError("Activation propagation requires at least one GPT-NeoX layer.")
    if not bool(getattr(model.config, "use_parallel_residual", False)):
        raise ValueError("This diagnostic currently describes the Pythia parallel-residual block only.")

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

    method_result = {
        "label": label,
        "config_id": source_manifest["config_id"],
        "run_id": source_manifest["run_id"],
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "num_layers": len(layers),
        "use_parallel_residual": True,
        "batches": batches,
        "wall_seconds": time.perf_counter() - method_started,
        "activations": accumulator.rows("activations", ACTIVATION_STAGE_ORDER),
        "matmuls": accumulator.rows("matmuls", MATMUL_STAGE_ORDER),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return method_result


class _PropagationAccumulator:
    def __init__(self, torch: Any):
        self.torch = torch
        self.activations: dict[tuple[int, str], list[int]] = {}
        self.matmuls: dict[tuple[int, str], list[int]] = {}

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
        counts = getattr(self, kind)
        current = counts.setdefault((int(layer), name), [0, 0])
        current[0] += int(zero_count)
        current[1] += int(total)

    def rows(self, kind: str, order: list[str]) -> list[dict[str, Any]]:
        counts = getattr(self, kind)
        order_index = {name: index for index, name in enumerate(order)}
        rows = []
        for (layer, name), (zero_count, total) in sorted(
            counts.items(), key=lambda item: (item[0][0], order_index[item[0][1]])
        ):
            rows.append(
                {
                    "name": name,
                    "layer": layer,
                    "zero_count": zero_count,
                    "total": total,
                    "exact_zero_fraction": zero_count / total if total else None,
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
            if attention_relu is None or mlp_relu is None:
                raise ValueError(
                    "Activation propagation requires checkpoints configured with model.post_layernorm_relu: true."
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
            handles.append(
                attention_relu.register_forward_pre_hook(
                    _activation_pre_hook(accumulator, "attention_layernorm_raw", layer_index)
                )
            )
            handles.append(
                attention_relu.register_forward_hook(
                    _activation_output_hook(accumulator, "attention_input_relu", layer_index)
                )
            )
            handles.append(
                mlp_relu.register_forward_pre_hook(
                    _activation_pre_hook(accumulator, "mlp_layernorm_raw", layer_index)
                )
            )
            handles.append(
                mlp_relu.register_forward_hook(
                    _activation_output_hook(accumulator, "mlp_input_relu", layer_index)
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
            handles.append(
                layer.mlp.act.register_forward_hook(
                    _activation_output_hook(accumulator, "mlp_hidden_relu", layer_index)
                )
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
        accumulator.add_activation("query_post_rope", layer_index, query)
        accumulator.add_activation("key_post_rope", layer_index, key)
        accumulator.add_activation("value", layer_index, value)
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
    source_manifests: list[dict[str, Any]], reference: dict[str, Any]
) -> None:
    reference_key = (reference["tokens_path"], reference["block_size"], reference["tokens"])
    for manifest in source_manifests[1:]:
        candidate = manifest["tokenized_data"]["validation"]
        if candidate is None:
            raise ValueError(f"Source run {manifest['config_id']} has no validation token cache.")
        candidate_key = (candidate["tokens_path"], candidate["block_size"], candidate["tokens"])
        if candidate_key != reference_key:
            raise ValueError("Selected runs do not share the same validation token cache.")


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
