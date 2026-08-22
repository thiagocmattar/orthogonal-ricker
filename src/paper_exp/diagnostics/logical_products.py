"""Authoritative integer accounting for causal logical product opportunities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LOGICAL_MATMUL_STAGES = (
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
)


def linear_zero_product_counts(
    value: Any, *, output_features: int, torch: Any
) -> tuple[int, int]:
    """Count zero scalar products induced by a linear layer's input zeros."""

    detached = value.detach()
    zero_inputs = int(torch.count_nonzero(detached == 0).cpu())
    input_total = int(detached.numel())
    return zero_inputs * int(output_features), input_total * int(output_features)


def qk_zero_product_counts(query: Any, key: Any, *, torch: Any) -> tuple[int, int]:
    """Count QK products with a zero operand over valid causal pairs."""

    if query.shape != key.shape or query.ndim != 4:
        raise ValueError(
            "QK zero-product counting expects matching [batch, heads, tokens, width] tensors."
        )
    batch, heads, tokens, width = query.shape
    query_nonzero = query.detach() != 0
    cumulative_key_nonzero = (key.detach() != 0).to(torch.int64).cumsum(dim=-2)
    nonzero_products = int(
        (query_nonzero.to(torch.int64) * cumulative_key_nonzero).sum().cpu()
    )
    total = int(batch * heads * width * tokens * (tokens + 1) // 2)
    return total - nonzero_products, total


def probability_value_zero_product_counts(
    probabilities: Any,
    value: Any,
    *,
    torch: Any,
    query_chunk_size: int = 128,
) -> tuple[int, int]:
    """Count PV products with a zero operand over valid causal pairs."""

    if probabilities.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "PV zero-product counting expects rank-four probability and value tensors."
        )
    batch, heads, queries, keys = probabilities.shape
    value_batch, value_heads, value_keys, width = value.shape
    if (batch, heads, keys) != (value_batch, value_heads, value_keys) or queries != keys:
        raise ValueError(
            "PV zero-product counting expects matching uncached causal-attention shapes."
        )

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


def summarize_block_model_products(
    zero_counts: Mapping[str, int],
    product_counts: Mapping[str, int],
    *,
    lm_head_product_count: int,
) -> dict[str, Any]:
    """Summarize the shared six-operation block and dense-LM-head denominators."""

    per_operation = {
        name: {
            "zero_product_count": int(zero_counts[name]),
            "product_count": int(product_counts[name]),
            "zero_product_fraction": int(zero_counts[name]) / int(product_counts[name]),
        }
        for name in LOGICAL_MATMUL_STAGES
    }
    block_zero_count = sum(
        row["zero_product_count"] for row in per_operation.values()
    )
    block_product_count = sum(row["product_count"] for row in per_operation.values())
    lm_head_count = int(lm_head_product_count)
    model_product_count = block_product_count + lm_head_count
    return {
        "R_block": block_zero_count / block_product_count,
        "R_model": block_zero_count / model_product_count,
        "block_zero_product_count": block_zero_count,
        "block_product_count": block_product_count,
        "lm_head_product_count": lm_head_count,
        "model_product_count": model_product_count,
        "per_operation": per_operation,
    }
