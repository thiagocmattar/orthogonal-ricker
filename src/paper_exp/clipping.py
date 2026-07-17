from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import yaml

from paper_exp.activations import ActivationCapture
from paper_exp.activations import activation_exact_zero_counts
from paper_exp.activations import activation_exact_zero_counts_by_alias
from paper_exp.modeling import load_checkpoint_model
from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_jsonl


def run_clipping_sweep(
    *,
    checkpoint_run_dir: str | Path,
    command: str,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float] | None = None,
    sites: list[str] | None = None,
    experiment_suffix: str | None = None,
    eval_batches: int | None,
    measure_zero_products: bool = False,
    seed: int = 0,
    run_id: str | None = None,
) -> Path:
    run_path = Path(checkpoint_run_dir)
    source_config_path = run_path / "config.yaml"
    source_manifest_path = run_path / "manifest.json"
    if not source_config_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint run must contain config.yaml and manifest.json: {run_path}")

    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    source_manifest = read_json(source_manifest_path)
    config["experiment_name"] = f"{config['experiment_name']}_clipping_sweep"
    if sites:
        config.setdefault("activation_clipping", {})["sites"] = sites

    torch, np, auto_model, modeling_gpt_neox = _load_clipping_dependencies()
    np.random.seed(seed)
    training = config["training"]
    device = _select_device(torch, training.get("device", "auto"))
    dtype = _select_dtype(torch, device, training.get("precision", "auto"))

    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
    model = load_checkpoint_model(auto_model, checkpoint_path, torch=torch)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    validation_metadata = source_manifest["tokenized_data"]["validation"]
    if validation_metadata is None:
        raise ValueError("Source run has no validation token cache in manifest.")
    validation_tokens = np.memmap(validation_metadata["tokens_path"], dtype=np.int32, mode="r")
    block_size = int(validation_metadata["block_size"])
    batch_size = int(config["validation"]["batch_size"])
    starts = _eval_starts(validation_tokens, block_size, eval_batches=eval_batches, batch_size=batch_size, np=np)

    suffix = experiment_suffix or (_site_suffix(sites) if sites else None)
    suffix_part = f"-{suffix}" if suffix else ""
    sweep_config_path = f"{source_manifest['config_id']}-clipping-sweep{suffix_part}.yaml"
    experiment_id, numbered_run_id, output_dir = create_run_dir(config, sweep_config_path, run_id=run_id)
    rows: list[dict[str, Any]] = []
    rms_multipliers = rms_multipliers or []
    for clipping_cfg in _clipping_configs(
        config,
        thresholds=thresholds,
        quantiles=quantiles,
        rms_multipliers=rms_multipliers,
    ):
        result = _evaluate_clipped_loss(
            model=model,
            torch=torch,
            np=np,
            tokens=validation_tokens,
            block_size=block_size,
            batch_size=batch_size,
            eval_batches=eval_batches,
            starts=starts,
            device=device,
            dtype=dtype,
            clipping_cfg=clipping_cfg,
            measure_zero_products=measure_zero_products,
            modeling_gpt_neox=modeling_gpt_neox,
        )
        rows.append(result)

    best_loss = min(rows, key=lambda row: row["validation_loss"]) if rows else None
    metrics = {
        "clipping/num_points": len(rows),
        "clipping/best_validation_loss": best_loss["validation_loss"] if best_loss else None,
        "clipping/best_achieved_sparsity": best_loss["achieved_sparsity"] if best_loss else None,
        "clipping/max_potentially_avoidable_model_matmul_fraction": (
            max(
                (
                    float(row["potentially_avoidable_model_matmul_fraction"])
                    for row in rows
                    if row.get("potentially_avoidable_model_matmul_fraction") is not None
                ),
                default=None,
            )
        ),
    }
    manifest = build_manifest(
        config=config,
        config_path=sweep_config_path,
        run_id=numbered_run_id,
        command=command,
        mode="clip-sweep",
        config_id=experiment_id,
        result_path=output_dir,
    )
    manifest["source_run"] = str(run_path)
    manifest["source_checkpoint"] = str(checkpoint_path)
    manifest["thresholds"] = thresholds
    manifest["quantiles"] = quantiles
    manifest["rms_multipliers"] = rms_multipliers
    manifest["clipping_sites"] = _clipping_sites(config)
    if {"attention_inputs", "mlp_inputs", "mlp_hiddens"}.issubset(manifest["clipping_sites"]):
        manifest["compute_skip_proxy"] = {
            "activation_statistic": "exact-zero fraction",
            "eligible_projection_formula": "(3*z_attention_inputs + 4*z_mlp_inputs + 4*z_mlp_hiddens) / 11",
            "block_linear_formula": "(3*z_attention_inputs + 4*z_mlp_inputs + 4*z_mlp_hiddens) / 12",
            "eligible_projection_weights": {
                "attention_inputs_to_qkv": 3,
                "mlp_inputs_to_dense_h_to_4h": 4,
                "mlp_hiddens_to_dense_4h_to_h": 4,
            },
            "unaffected_block_linear_weight": {"attention_output_projection": 1},
            "targeted_multiplications_per_token_all_six_layers": 1081344,
            "interpretation": (
                "Ideal potentially skippable input-weight multiplications for Pythia-14M; "
                "not measured wall-clock speedup and not a structured sparsity metric."
            ),
        }
    if suffix:
        manifest["clipping_sweep_suffix"] = suffix
    if rms_multipliers:
        manifest["rms_threshold_semantics"] = (
            "For each captured activation tensor and forward pass, clip entries with "
            "|a| <= rms_multiplier * RMS(A), where RMS(A) is computed over that tensor."
        )
    manifest["eval_batches"] = eval_batches
    manifest["seed"] = seed
    manifest["measure_zero_products"] = bool(measure_zero_products)
    if measure_zero_products:
        manifest["logical_zero_products"] = {
            "exact_zero_definition": "A tensor coordinate is zero iff value == 0 with no tolerance.",
            "block_denominator": (
                "QKV, valid-causal QK, valid-causal PV, attention output, MLP up, and MLP down "
                "logical scalar products across every transformer block."
            ),
            "model_denominator": (
                "The block denominator plus the final hidden-to-vocabulary LM-head products."
            ),
            "interpretation": (
                "Potentially avoidable logical products, not measured dense-kernel speedup. "
                "PRE-RoPE Q/K clipping is credited only after standard RoPE, using the actual QK operands."
            ),
        }
    if "sweep" in source_manifest:
        manifest["source_sweep"] = source_manifest["sweep"]

    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=rows)
    write_jsonl(output_dir / "clipping_frontier.jsonl", rows)
    return output_dir


def _clipping_configs(
    config: dict[str, Any],
    *,
    thresholds: list[float],
    quantiles: list[float],
    rms_multipliers: list[float],
) -> list[dict[str, Any]]:
    base = config.get("activation_clipping", {})
    sites = base.get("sites", ["mlp_hiddens"])
    configs = []
    for threshold in thresholds:
        configs.append({"enabled": True, "mode": "threshold", "sites": sites, "threshold": threshold})
    for quantile in quantiles:
        configs.append({"enabled": True, "mode": "quantile", "sites": sites, "quantile": quantile})
    for multiplier in rms_multipliers:
        configs.append(
            {
                "enabled": True,
                "mode": "rms_threshold",
                "sites": sites,
                "rms_multiplier": multiplier,
            }
        )
    return configs


def _clipping_sites(config: dict[str, Any]) -> list[str]:
    return list(config.get("activation_clipping", {}).get("sites", ["mlp_hiddens"]))


def _site_suffix(sites: list[str] | None) -> str | None:
    if not sites:
        return None
    if set(sites) == {"mlp_hiddens", "attention_outputs", "residual_streams"}:
        return "all-sites"
    return "sites-" + "-".join(site.replace("_", "-") for site in sites)


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

    with ActivationCapture(model, clipping_cfg.get("sites", ["mlp_hiddens"]), torch=torch, clipping=clipping_cfg) as capture:
        with zero_product_context:
            with torch.no_grad():
                for offset in range(0, len(starts), batch_size):
                    capture.clear()
                    batch_starts = starts[offset : offset + batch_size]
                    batch = np.stack([tokens[start : start + block_size] for start in batch_starts])
                    input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
                    with _autocast_context(torch, device, dtype):
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
    skip_proxies = pythia_projection_skip_proxies(site_achieved_sparsity)
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
        **skip_proxies,
        **logical_zero_product_metrics,
        "validation_loss": sum(losses) / total_sequences,
        "achieved_sparsity": zero_hits / zero_count if zero_count else None,
        "validation_batches": batches,
        "validation_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": total_tokens / wall_seconds if wall_seconds > 0 else None,
    }


_LOGICAL_MATMUL_STAGES = (
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
)


class _LogicalZeroProductAccumulator:
    """Pooled integer logical-product counters for one clipping threshold."""

    def __init__(self) -> None:
        self.zero_counts = {name: 0 for name in _LOGICAL_MATMUL_STAGES}
        self.totals = {name: 0 for name in _LOGICAL_MATMUL_STAGES}

    def add(self, name: str, zero_count: int, total: int) -> None:
        if name not in self.zero_counts:
            raise KeyError(f"Unknown logical matmul stage: {name}")
        self.zero_counts[name] += int(zero_count)
        self.totals[name] += int(total)

    def summary(self, *, model: Any, total_tokens: int) -> dict[str, Any]:
        missing = [name for name in _LOGICAL_MATMUL_STAGES if self.totals[name] <= 0]
        if missing:
            raise RuntimeError(
                "Logical zero-product measurement missed stages: " + ", ".join(missing)
            )
        block_zero_count = sum(self.zero_counts.values())
        block_product_count = sum(self.totals.values())
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise RuntimeError("Logical zero-product measurement requires an output embedding weight.")
        hidden_size = int(output_embeddings.weight.shape[1])
        vocab_size = int(output_embeddings.weight.shape[0])
        lm_head_product_count = int(total_tokens) * hidden_size * vocab_size
        model_product_count = block_product_count + lm_head_product_count
        return {
            "matmul_zero_product_count": dict(self.zero_counts),
            "matmul_product_count": dict(self.totals),
            "matmul_zero_product_fraction": {
                name: self.zero_counts[name] / self.totals[name]
                for name in _LOGICAL_MATMUL_STAGES
            },
            "block_zero_product_count": block_zero_count,
            "block_matmul_product_count": block_product_count,
            "lm_head_matmul_product_count": lm_head_product_count,
            "model_matmul_product_count": model_product_count,
            "potentially_avoidable_block_matmul_fraction": (
                block_zero_count / block_product_count
            ),
            "potentially_avoidable_model_matmul_fraction": (
                block_zero_count / model_product_count
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
            value = inputs[0].detach()
            zero_inputs = int(torch.count_nonzero(value == 0).cpu())
            input_total = int(value.numel())
            output_features = int(module.out_features)
            accumulator.add(
                name,
                zero_inputs * output_features,
                input_total * output_features,
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
            *_qk_zero_product_counts(query, key, torch=torch),
        )
        accumulator.add(
            "probability_value",
            *_probability_value_zero_product_counts(probabilities, value, torch=torch),
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


def _qk_zero_product_counts(query: Any, key: Any, *, torch: Any) -> tuple[int, int]:
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("QK zero-product counting expects matching rank-four Q and K tensors.")
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
        raise ValueError("PV zero-product counting expects rank-four P and V tensors.")
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
        nonzero_products += int(
            (
                (probability_nonzero & valid).to(torch.int64)
                * value_nonzero_dimensions.unsqueeze(-2)
            )
            .sum()
            .cpu()
        )
    total = int(batch * heads * width * queries * (queries + 1) // 2)
    return total - nonzero_products, total


def pythia_projection_skip_proxies(site_exact_zero_fraction: dict[str, float]) -> dict[str, float]:
    required = {"attention_inputs", "mlp_inputs", "mlp_hiddens"}
    if not required.issubset(site_exact_zero_fraction):
        return {}
    weighted = (
        3.0 * site_exact_zero_fraction["attention_inputs"]
        + 4.0 * site_exact_zero_fraction["mlp_inputs"]
        + 4.0 * site_exact_zero_fraction["mlp_hiddens"]
    )
    return {
        "eligible_projection_skip_fraction": weighted / 11.0,
        "block_linear_skip_fraction": weighted / 12.0,
    }


def _eval_starts(tokens: Any, block_size: int, *, eval_batches: int | None, batch_size: int, np: Any) -> list[int]:
    if eval_batches is None:
        total_blocks = max(1, (len(tokens) - 1) // block_size)
        return [index * block_size for index in range(total_blocks)]
    max_start = len(tokens) - block_size - 1
    return list(np.random.randint(0, max_start, size=int(eval_batches) * batch_size))


def _autocast_context(torch: Any, device: Any, dtype: Any) -> Any:
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=dtype)
    from contextlib import nullcontext

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
