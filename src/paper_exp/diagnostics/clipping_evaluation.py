"""Clipped validation evaluation and GPT-NeoX logical-product capture."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from paper_exp.activations import ActivationCapture
from paper_exp.activations import activation_exact_zero_counts
from paper_exp.activations import activation_exact_zero_counts_by_alias

from .logical_products import (
    LOGICAL_MATMUL_STAGES,
    linear_zero_product_counts,
    probability_value_zero_product_counts,
    qk_zero_product_counts,
    summarize_block_model_products,
)
from .evaluation import autocast_context


def _evaluate_clipped_loss(
    *,
    model: Any,
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    eval_batches: int | None,
    starts: list[int],
    device: Any,
    dtype: Any,
    clipping_cfg: dict[str, Any],
    measure_zero_products: bool = False,
    modeling_gpt_neox: Any | None = None,
) -> dict[str, Any]:
    losses: list[float] = []
    batches = 0
    total_sequences = 0
    total_tokens = 0
    zero_hits = 0
    zero_count = 0
    site_zero_hits: dict[str, int] = {}
    site_zero_counts: dict[str, int] = {}
    start_time = time.perf_counter()
    zero_products = _LogicalZeroProductAccumulator()
    zero_product_context = (
        _capture_logical_zero_products(
            model,
            accumulator=zero_products,
            modeling_gpt_neox=modeling_gpt_neox,
            torch=torch,
        )
        if measure_zero_products
        else nullcontext()
    )

    with ActivationCapture(
        model,
        clipping_cfg["sites"],
        torch=torch,
        clipping=clipping_cfg,
    ) as capture:
        with zero_product_context:
            with torch.no_grad():
                for offset in range(0, len(starts), batch_size):
                    capture.clear()
                    batch_starts = starts[offset : offset + batch_size]
                    batch = np.stack([tokens[start : start + block_size] for start in batch_starts])
                    input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                    with autocast_context(torch, device, dtype):
                        output = model(input_ids=input_ids, labels=input_ids)
                    if not bool(torch.isfinite(output.loss.detach()).item()):
                        raise RuntimeError("Non-finite clipped validation loss.")
                    losses.append(float(output.loss.detach().cpu()) * len(batch_starts))
                    batch_zero_hits, batch_activation_count = activation_exact_zero_counts(capture.activations)
                    zero_hits += batch_zero_hits
                    zero_count += batch_activation_count
                    for alias, (alias_hits, alias_count) in activation_exact_zero_counts_by_alias(
                        capture.activations
                    ).items():
                        site_zero_hits[alias] = site_zero_hits.get(alias, 0) + alias_hits
                        site_zero_counts[alias] = site_zero_counts.get(alias, 0) + alias_count
                    total_sequences += len(batch_starts)
                    total_tokens += len(batch_starts) * block_size
                    batches += 1

    wall_seconds = time.perf_counter() - start_time
    site_achieved_sparsity = {
        alias: site_zero_hits[alias] / site_zero_counts[alias]
        for alias in sorted(site_zero_counts)
        if site_zero_counts[alias]
    }
    logical_zero_product_metrics = (
        zero_products.summary(model=model, total_tokens=total_tokens)
        if measure_zero_products
        else {}
    )
    return {
        "event": "clipping_sweep",
        "mode": clipping_cfg["mode"],
        "threshold": clipping_cfg.get("threshold"),
        "quantile": clipping_cfg.get("quantile"),
        "rms_multiplier": clipping_cfg.get("rms_multiplier"),
        "rms_scope": (
            "per captured activation tensor per forward pass"
            if clipping_cfg["mode"] == "rms_threshold"
            else None
        ),
        "sites": clipping_cfg.get("sites", ["mlp_hiddens"]),
        "site_achieved_sparsity": site_achieved_sparsity,
        "site_zero_hits": {alias: site_zero_hits[alias] for alias in sorted(site_zero_hits)},
        "site_activation_count": {alias: site_zero_counts[alias] for alias in sorted(site_zero_counts)},
        **logical_zero_product_metrics,
        "validation_loss": sum(losses) / total_sequences,
        "achieved_sparsity": zero_hits / zero_count if zero_count else None,
        "validation_batches": batches,
        "validation_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": total_tokens / wall_seconds if wall_seconds > 0 else None,
    }


class _LogicalZeroProductAccumulator:
    """Pooled integer logical-product counters for one clipping threshold."""

    def __init__(self) -> None:
        self.zero_counts = {name: 0 for name in LOGICAL_MATMUL_STAGES}
        self.totals = {name: 0 for name in LOGICAL_MATMUL_STAGES}

    def add(self, name: str, zero_count: int, total: int) -> None:
        if name not in self.zero_counts:
            raise KeyError(f"Unknown logical matmul stage: {name}")
        self.zero_counts[name] += int(zero_count)
        self.totals[name] += int(total)

    def summary(self, *, model: Any, total_tokens: int) -> dict[str, Any]:
        missing = [name for name in LOGICAL_MATMUL_STAGES if self.totals[name] <= 0]
        if missing:
            raise RuntimeError(
                "Logical zero-product measurement missed stages: " + ", ".join(missing)
            )
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise RuntimeError("Logical zero-product measurement requires an output embedding weight.")
        hidden_size = int(output_embeddings.weight.shape[1])
        vocab_size = int(output_embeddings.weight.shape[0])
        lm_head_product_count = int(total_tokens) * hidden_size * vocab_size
        product_summary = summarize_block_model_products(
            self.zero_counts,
            self.totals,
            lm_head_product_count=lm_head_product_count,
        )
        per_operation = product_summary["per_operation"]
        return {
            "matmul_zero_product_count": dict(self.zero_counts),
            "matmul_product_count": dict(self.totals),
            "matmul_zero_product_fraction": {
                name: per_operation[name]["zero_product_fraction"]
                for name in LOGICAL_MATMUL_STAGES
            },
            "block_zero_product_count": product_summary["block_zero_product_count"],
            "block_matmul_product_count": product_summary["block_product_count"],
            "lm_head_matmul_product_count": product_summary["lm_head_product_count"],
            "model_matmul_product_count": product_summary["model_product_count"],
            "potentially_avoidable_block_matmul_fraction": (
                product_summary["R_block"]
            ),
            "potentially_avoidable_model_matmul_fraction": (
                product_summary["R_model"]
            ),
        }


@contextmanager
def _capture_logical_zero_products(
    model: Any,
    *,
    accumulator: _LogicalZeroProductAccumulator,
    modeling_gpt_neox: Any,
    torch: Any,
) -> Iterator[None]:
    """Count the six Pythia block matmuls while clipping hooks are active."""

    if modeling_gpt_neox is None:
        raise ValueError("Logical zero-product measurement requires GPT-NeoX attention support.")
    handles = []
    original_eager_attention = modeling_gpt_neox.eager_attention_forward
    original_implementation = model.config._attn_implementation

    def linear_hook(name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            accumulator.add(
                name,
                *linear_zero_product_counts(
                    inputs[0], output_features=int(module.out_features), torch=torch
                ),
            )

        return hook

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
        accumulator.add(
            "qk_scores",
            *qk_zero_product_counts(query, key, torch=torch),
        )
        accumulator.add(
            "probability_value",
            *probability_value_zero_product_counts(probabilities, value, torch=torch),
        )
        return context, probabilities

    try:
        for layer in model.gpt_neox.layers:
            handles.append(
                layer.attention.query_key_value.register_forward_pre_hook(
                    linear_hook("qkv_projection")
                )
            )
            handles.append(
                layer.attention.dense.register_forward_pre_hook(
                    linear_hook("attention_output_projection")
                )
            )
            handles.append(
                layer.mlp.dense_h_to_4h.register_forward_pre_hook(linear_hook("mlp_w1"))
            )
            handles.append(
                layer.mlp.dense_4h_to_h.register_forward_pre_hook(linear_hook("mlp_w2"))
            )
        modeling_gpt_neox.eager_attention_forward = instrumented_eager_attention
        model.config._attn_implementation = "eager"
        yield
    finally:
        model.config._attn_implementation = original_implementation
        modeling_gpt_neox.eager_attention_forward = original_eager_attention
        for handle in handles:
            handle.remove()


def _load_clipping_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.gpt_neox import modeling_gpt_neox
    except ImportError as exc:
        raise RuntimeError(
            "Clipping sweep requires numpy, torch, and transformers. Run `make install` first."
        ) from exc
    return torch, np, AutoModelForCausalLM, modeling_gpt_neox
