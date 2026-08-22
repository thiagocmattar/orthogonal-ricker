"""Capture hooks and integer accumulation for activation propagation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from paper_exp.modeling import activation_gate_metadata

from .logical_products import (
    linear_zero_product_counts,
    probability_value_zero_product_counts,
    qk_zero_product_counts,
)


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
            *linear_zero_product_counts(value, output_features=output_features, torch=self.torch),
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
        for stage, operand in (
            ("query_qk_input", query),
            ("key_qk_input", key),
            ("value_pv_input", value),
        ):
            zero_count, total = _exact_zero_counts(operand, torch=torch)
            accumulator.add_counts(
                "activations", stage, layer_index, zero_count, total
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
            *qk_zero_product_counts(query, key, torch=torch),
        )
        accumulator.add_counts(
            "matmuls",
            "probability_value",
            layer_index,
            *probability_value_zero_product_counts(probabilities, value, torch=torch),
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
