"""Report 04 data preparation and explicit Matplotlib renderers.

This module owns figures 79--90 and their scientific presentation choices.
The stable CLI and batch-dispatch facade remains in :mod:`paper_exp.plots`.
Use the figure-to-wrapper index in ``docs/plotting.md`` to find a specific
figure, then search this module for the matching ``_plot_*`` name.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from paper_exp.plot_common import (
    _finite,
    _histogram_density,
    _histogram_layer,
    _histogram_method,
    _trimmed_decimal_tick,
)
from paper_exp.plot_style import REPORT04_METHOD_COLORS, REPORT04_METHOD_MARKERS

REPORT04_TRAINING_RUNS = (
    ("GELU AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("MLP-ReLU AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("MLP-ReLU OL1", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
    ("Three-ReLU AdamW", "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass"),
    (
        "Three-ReLU OR",
        "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    ("Three-ReLU L1N", "104-pythia-14m-minipile-post-layernorm-relu-l1-naive-full-pass-w5"),
    ("Three-ReLU OL1", "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5"),
)
REPORT04_CLIPPING_RUNS = (
    ("MLP-ReLU AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("Three-ReLU AdamW", "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass"),
    (
        "Three-ReLU OR",
        "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05",
    ),
    ("Three-ReLU L1N", "104-pythia-14m-minipile-post-layernorm-relu-l1-naive-full-pass-w5"),
    ("Three-ReLU OL1", "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5"),
)
REPORT04_JOINT_CLIPPING_RUNS = (
    ("MLP-ReLU AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("Three-ReLU AdamW", "98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass"),
    ("Three-ReLU OR", "103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("Three-ReLU L1N", "104-pythia-14m-minipile-post-layernorm-relu-l1-naive-full-pass-w5"),
    ("Three-ReLU OL1", "99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5"),
)
REPORT04_CLIPPING_SITES = (
    ("attention_inputs", "Attention inputs"),
    ("mlp_inputs", "MLP inputs"),
    ("mlp_hiddens", "MLP hiddens"),
)
REPORT04_INPUT_HISTOGRAM_EXPERIMENT = "100-pythia-14m-minipile-post-layernorm-relu-input-histograms"
REPORT04_MLP_HISTOGRAM_EXPERIMENT = "101-pythia-14m-minipile-post-layernorm-relu-mlp-hidden-histograms"
POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT = (
    "102-pythia-14m-minipile-post-layernorm-relu-activation-propagation"
)
PROPAGATION_ACTIVATION_ROWS = (
    ("residual_input", r"Residual input $H_l$"),
    ("attention_layernorm_raw", r"Attention LN raw $U_l$"),
    ("attention_input_relu", r"Attention ReLU $U_l^+$ [QKV input]"),
    ("query_post_rope", r"Query $Q_l$ after RoPE"),
    ("key_post_rope", r"Key $K_l$ after RoPE"),
    ("value", r"Value $V_l$"),
    ("attention_probabilities", r"Attention prob. $P_l$ [valid causal]"),
    ("attention_context", r"Context $C_l=P_lV_l$ [$W_o$ input]"),
    ("attention_output", r"Attention output $O_l$"),
    ("mlp_layernorm_raw", r"MLP LN raw $R_l$"),
    ("mlp_input_relu", r"MLP-input ReLU $R_l^+$ [$W_1$ input]"),
    ("mlp_w1_preactivation", r"MLP preactivation $Z_l$"),
    ("mlp_hidden_relu", r"MLP-hidden ReLU $A_l$ [$W_2$ input]"),
    ("mlp_output", r"MLP output $M_l$"),
    ("residual_output", r"Residual output $H_{l+1}$"),
)
PROPAGATION_MATMUL_ROWS = (
    ("qkv_projection", r"QKV: $U_l^+ W_{qkv}$"),
    ("qk_scores", r"Attention scores: $Q_l K_l^T$ [valid causal]"),
    ("probability_value", r"Attention mix: $P_l V_l$ [valid causal]"),
    ("attention_output_projection", r"Attention projection: $C_l W_o$"),
    ("mlp_w1", r"MLP up: $R_l^+ W_1$"),
    ("mlp_w2", r"MLP down: $A_l W_2$"),
)

# Pythia-14M forward-pass accounting used in report 04.  The attention terms
# count valid causal pairs rather than the masked upper triangle.  The metric
# is a logical scalar-product opportunity, not a wall-clock or kernel FLOP claim.
REPORT04_HIDDEN_SIZE = 128
REPORT04_INTERMEDIATE_SIZE = 512
REPORT04_NUM_LAYERS = 6
REPORT04_BLOCK_SIZE = 2048
REPORT04_VOCAB_SIZE = 50304
REPORT04_TARGET_PRODUCTS_PER_TOKEN = REPORT04_NUM_LAYERS * (
    3 * REPORT04_HIDDEN_SIZE**2
    + REPORT04_HIDDEN_SIZE * REPORT04_INTERMEDIATE_SIZE
    + REPORT04_INTERMEDIATE_SIZE * REPORT04_HIDDEN_SIZE
)
REPORT04_BLOCK_PRODUCTS_PER_TOKEN = REPORT04_NUM_LAYERS * (
    3 * REPORT04_HIDDEN_SIZE**2
    + REPORT04_HIDDEN_SIZE**2
    + REPORT04_HIDDEN_SIZE * REPORT04_INTERMEDIATE_SIZE
    + REPORT04_INTERMEDIATE_SIZE * REPORT04_HIDDEN_SIZE
    + REPORT04_HIDDEN_SIZE * (REPORT04_BLOCK_SIZE + 1)
)
REPORT04_LM_HEAD_PRODUCTS_PER_TOKEN = REPORT04_HIDDEN_SIZE * REPORT04_VOCAB_SIZE
REPORT04_MODEL_PRODUCTS_PER_TOKEN = REPORT04_BLOCK_PRODUCTS_PER_TOKEN + REPORT04_LM_HEAD_PRODUCTS_PER_TOKEN
REPORT04_TARGET_MODEL_FRACTION = REPORT04_TARGET_PRODUCTS_PER_TOKEN / REPORT04_MODEL_PRODUCTS_PER_TOKEN

# Official Pythia architecture dimensions.  The family table provides L and d;
# the Hugging Face configs provide the padded vocabulary sizes used by each LM
# head.  All models use T=2048 and an intermediate width of 4d.
REPORT04_PYTHIA_FAMILY = (
    ("14M", 6, 128, 50304),
    ("31M", 6, 256, 50304),
    ("70M", 6, 512, 50304),
    ("160M", 12, 768, 50304),
    ("410M", 24, 1024, 50304),
    ("1B", 16, 2048, 50304),
    ("1.4B", 24, 2048, 50304),
    ("2.8B", 32, 2560, 50304),
    ("6.9B", 32, 4096, 50432),
    ("12B", 36, 5120, 50688),
)
def _load_report04_parameter_series(
    runs: list[tuple[str, str | Path]],
) -> list[dict[str, Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Report-04 parameter diagnostics require torch and safetensors.") from exc

    weight_specs = (
        (
            "qkv",
            "Attention QKV weights",
            re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
            (-0.65, 0.65),
        ),
        (
            "w1",
            "MLP W1 weights",
            re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
            (-0.22, 0.22),
        ),
        (
            "w2",
            "MLP W2 weights",
            re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_4h_to_h\.weight$"),
            (-0.22, 0.22),
        ),
    )
    layer_norm_specs = (
        ("attention", "Pre-attention LayerNorm", "input_layernorm"),
        ("mlp", "Pre-MLP LayerNorm", "post_attention_layernorm"),
    )

    bins = 180
    series: list[dict[str, Any]] = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
        weight_groups: dict[str, dict[str, Any]] = {}
        layer_norms: dict[str, dict[str, Any]] = {}
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for group_id, group_label, pattern, (range_min, range_max) in weight_specs:
                values = [
                    handle.get_tensor(key).detach().float().reshape(-1)
                    for key in keys
                    if pattern.match(key) is not None
                ]
                if not values:
                    raise ValueError(f"No {group_label} tensors found for {label}.")
                flat = torch.cat(values)
                finite = flat[torch.isfinite(flat)]
                counts = torch.histc(finite, bins=bins, min=range_min, max=range_max).cpu().double()
                width = (range_max - range_min) / bins
                total = int(finite.numel())
                weight_groups[group_id] = {
                    "label": group_label,
                    "centers": [range_min + (index + 0.5) * width for index in range(bins)],
                    "densities": [float(count) / total / width for count in counts.tolist()],
                    "range": (range_min, range_max),
                    "total": total,
                    "tensor_count": len(values),
                    "outside": int(((finite < range_min) | (finite > range_max)).sum().item()),
                }

            for branch_id, branch_label, module_name in layer_norm_specs:
                parameters: dict[str, list[dict[str, Any]]] = {"gamma": [], "beta": []}
                for layer_index in range(6):
                    for parameter_id, tensor_name in (("gamma", "weight"), ("beta", "bias")):
                        key = f"gpt_neox.layers.{layer_index}.{module_name}.{tensor_name}"
                        if key not in keys:
                            raise ValueError(f"Missing LayerNorm tensor {key!r} for {label}.")
                        tensor = handle.get_tensor(key).detach().double().reshape(-1)
                        parameters[parameter_id].append(
                            {
                                "layer": layer_index,
                                "mean": float(tensor.mean()),
                                "std": float(tensor.std(unbiased=False)),
                                "count": int(tensor.numel()),
                            }
                        )
                layer_norms[branch_id] = {"label": branch_label, "parameters": parameters}

        series.append(
            {
                "label": label,
                "run_dir": str(run_path),
                "weight_groups": weight_groups,
                "layer_norms": layer_norms,
            }
        )
    return series
def _report04_site_event_fraction(event: dict[str, Any], site: str, threshold_key: str) -> float | None:
    values = [
        event.get(f"activation/{site}.layer_{layer_index}/near_zero_mass/{threshold_key}")
        for layer_index in range(6)
    ]
    finite_values = [float(value) for value in values if _finite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else None


def _plot_report04_learning_diagnostics(
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    by_label = {str(item["label"]): item for item in series}
    required = {label for label, _experiment_id in REPORT04_TRAINING_RUNS}
    missing = sorted(required.difference(by_label))
    if missing:
        raise ValueError(f"Missing report-04 training series: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=False)
    ax_validation, ax_exact = axes[0]
    ax_near_zero, ax_update = axes[1]

    for label, _experiment_id in REPORT04_TRAINING_RUNS:
        item = by_label[label]
        events = item["validation_events"]
        if not events:
            continue
        ax_validation.plot(
            [float(event["tokens_seen"]) / 1e9 for event in events],
            [float(event["validation_loss"]) for event in events],
            color=REPORT04_METHOD_COLORS[label],
            marker=REPORT04_METHOD_MARKERS[label],
            markersize=3.0,
            linewidth=1.5,
            label=label,
        )
    ax_validation.set_title("(a) Full-validation learning curves")
    ax_validation.set_xlabel("Tokens seen (billions)")
    ax_validation.set_ylabel("Validation loss")
    ax_validation.legend(frameon=False, fontsize=7, ncol=2)

    site_styles = {
        "attention_inputs": ("Attention inputs", "-", "o"),
        "mlp_inputs": ("MLP inputs", "--", "s"),
        "mlp_hiddens": ("MLP hiddens", ":", "^"),
    }
    for ax, threshold_key, panel_title in (
        (ax_exact, "k0", "(b) Exact-zero trajectories"),
        (ax_near_zero, "k1em02", "(c) |activation| <= 0.01 trajectories"),
    ):
        for label in ("Three-ReLU AdamW", "Three-ReLU OR", "Three-ReLU L1N", "Three-ReLU OL1"):
            train_events = by_label[label]["train_events"]
            for site, (site_label, linestyle, marker) in site_styles.items():
                points = [
                    (float(event["tokens_seen"]) / 1e9, _report04_site_event_fraction(event, site, threshold_key))
                    for event in train_events
                ]
                points = [(x_value, y_value) for x_value, y_value in points if y_value is not None]
                if not points:
                    continue
                ax.plot(
                    [x_value for x_value, _y_value in points],
                    [100.0 * float(y_value) for _x_value, y_value in points],
                    color=REPORT04_METHOD_COLORS[label],
                    linestyle=linestyle,
                    marker=marker,
                    markevery=max(1, len(points) // 9),
                    markersize=2.4,
                    linewidth=1.25,
                    label=f"{label.replace('Three-ReLU ', '')} - {site_label}",
                )
        ax.set_title(panel_title)
        ax.set_xlabel("Tokens seen (billions)")
        ax.set_ylabel("Elementwise activation fraction (%)")
        ax.set_ylim(-2.0, 102.0)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)

    for label in ("MLP-ReLU OL1", "Three-ReLU OR", "Three-ReLU OL1"):
        train_events = by_label[label]["train_events"]
        for metric_key, metric_label, linestyle in (
            ("pressure/pressure_update_ratio_raw", "raw", ":"),
            ("pressure/pressure_update_ratio_final", "final", "-"),
        ):
            points = [
                (float(event["tokens_seen"]) / 1e9, float(event[metric_key]))
                for event in train_events
                if _finite(event.get(metric_key)) and float(event[metric_key]) > 0.0
            ]
            ax_update.plot(
                [x_value for x_value, _y_value in points],
                [y_value for _x_value, y_value in points],
                color=REPORT04_METHOD_COLORS[label],
                linestyle=linestyle,
                linewidth=1.35,
                label=f"{label} - {metric_label}",
            )
    ax_update.axhline(0.5, color="#4d4d4d", linestyle="--", linewidth=1.0, label="step budget = 0.5")
    ax_update.set_yscale("log")
    ax_update.set_title("(d) Orthogonal-pressure update ratios")
    ax_update.set_xlabel("Tokens seen (billions)")
    ax_update.set_ylabel("Pressure update / AdamW update")
    ax_update.legend(frameon=False, fontsize=7, ncol=2)

    validation_tokens = max(
        (
            int(event.get("validation_tokens") or 0)
            for item in series
            for event in item["validation_events"]
        ),
        default=0,
    )
    fig.suptitle("Post-LayerNorm ReLU Training Diagnostics", y=0.985)
    fig.text(
        0.5,
        0.948,
        (
            f"One seed per method; fixed 22,762-step / 1.492B-token budget; "
            f"each validation point uses {validation_tokens:,} tokens"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.018,
        "Activation trajectories are layer means from logged training minibatches; no seed uncertainty is estimated.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.09, hspace=0.34, wspace=0.25)
    fig.savefig(output_path)
    plt.close(fig)


def _report04_histogram_payload(payloads: dict[str, dict[str, Any]], site: str) -> dict[str, Any]:
    key = "mlp_hiddens" if site == "mlp_hiddens" else "inputs"
    if key not in payloads:
        raise ValueError(f"Missing report-04 histogram payload {key!r}.")
    return payloads[key]


def _report04_histogram_fraction(
    payloads: dict[str, dict[str, Any]],
    method_label: str,
    site: str,
    layer_index: int,
    threshold_key: str,
) -> float:
    payload = _report04_histogram_payload(payloads, site)
    method = _histogram_method(payload, method_label)
    if method is None:
        raise ValueError(f"Missing histogram method {method_label!r} for {site}.")
    layer = _histogram_layer(method, f"{site}.layer_{layer_index}")
    fractions = layer.get("threshold_fractions") or {}
    value = fractions.get(threshold_key)
    if not _finite(value):
        raise ValueError(
            f"Missing threshold fraction {threshold_key!r} for {method_label}, {site}, layer {layer_index}."
        )
    return float(value)


def _plot_report04_activation_heatmaps(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    method_labels = [label for label, _experiment_id in REPORT04_TRAINING_RUNS]
    threshold_specs = (("0", "Exact zeros"), ("0.01", "|activation| <= 0.01"))
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2), sharex=True, sharey=True)
    image = None

    for row_index, (site, site_label) in enumerate(REPORT04_CLIPPING_SITES):
        for col_index, (threshold_key, threshold_label) in enumerate(threshold_specs):
            ax = axes[row_index][col_index]
            matrix = [
                [
                    100.0
                    * _report04_histogram_fraction(
                        payloads,
                        method_label,
                        site,
                        layer_index,
                        threshold_key,
                    )
                    for layer_index in range(6)
                ]
                for method_label in method_labels
            ]
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
            ax.set_title(f"{site_label}: {threshold_label}", fontsize=10)
            ax.set_xticks(range(6), [f"L{layer_index}" for layer_index in range(6)])
            ax.set_yticks(range(len(method_labels)), method_labels, fontsize=7)
            if col_index == 1:
                ax.tick_params(labelleft=False)
            for method_index, row in enumerate(matrix):
                for layer_index, value in enumerate(row):
                    ax.text(
                        layer_index,
                        method_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if value >= 55.0 else "white",
                    )
    axes[-1][0].set_xlabel("Transformer layer")
    axes[-1][1].set_xlabel("Transformer layer")

    validation_tokens = int(payloads["inputs"].get("validation_tokens") or 0)
    validation_sequences = int(payloads["inputs"].get("validation_sequences") or 0)
    fig.suptitle("Post-LayerNorm ReLU Activation Fractions by Layer", y=0.988)
    fig.text(
        0.5,
        0.955,
        (
            f"Full deterministic validation cache: {validation_sequences:,} sequences / "
            f"{validation_tokens:,} tokens; stored elementwise counts"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.018,
        (
            "One seed per method and a fixed 1.492B-token training budget; "
            "values are descriptive, without seed uncertainty."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.19, right=0.88, top=0.91, bottom=0.07, hspace=0.30, wspace=0.08)
    if image is not None:
        colorbar_axis = fig.add_axes([0.905, 0.18, 0.018, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Elementwise fraction (%)", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_activation_densities(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    method_labels = [label for label, _experiment_id in REPORT04_TRAINING_RUNS]
    site_specs = (
        ("attention_inputs", "Attention inputs", (-4.0, 4.0)),
        ("mlp_inputs", "MLP inputs", (-4.0, 4.0)),
        ("mlp_hiddens", "MLP hiddens", (-0.25, 0.75)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.7), sharey=False)
    legend_handles: dict[str, Any] = {}
    max_hidden_outside = 0.0

    for ax, (site, site_label, x_limits) in zip(axes, site_specs, strict=True):
        payload = _report04_histogram_payload(payloads, site)
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        if site == "mlp_hiddens":
            upper_edge = next((edge for edge in edges if edge >= x_limits[1]), x_limits[1])
            x_limits = (x_limits[0], upper_edge)
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        densities_by_method: dict[str, list[float]] = {}
        positive_values: list[float] = []
        for method_label in method_labels:
            method = _histogram_method(payload, method_label)
            if method is None:
                raise ValueError(f"Missing histogram method {method_label!r} for {site}.")
            layer = _histogram_layer(method, f"{site}.layer_3")
            densities = _histogram_density(layer, edges)
            densities_by_method[method_label] = densities
            positive_values.extend(value for value in densities if value > 0.0)
            if site == "mlp_hiddens":
                outside_count = int(layer.get("underflow") or 0) + int(layer.get("overflow") or 0)
                outside_count += sum(
                    int(count)
                    for count, left, right in zip(layer.get("counts", []), edges[:-1], edges[1:], strict=True)
                    if right <= x_limits[0] or left >= x_limits[1]
                )
                max_hidden_outside = max(
                    max_hidden_outside,
                    outside_count / max(int(layer.get("total") or 0), 1),
                )
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        y_max = max_density * 1.5
        for method_label in method_labels:
            densities = densities_by_method[method_label]
            visible = [max(value, y_min) if value > 0.0 else y_min for value in densities]
            (line,) = ax.step(
                centers,
                visible,
                where="mid",
                color=REPORT04_METHOD_COLORS[method_label],
                linewidth=1.4,
                label=method_label,
            )
            legend_handles.setdefault(method_label, line)
        ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
        ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(*x_limits)
        ax.set_title(f"{site_label}, layer 3")
        ax.set_xlabel("Activation value")
        ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
    axes[0].set_ylabel("Probability density (log scale)")

    validation_tokens = int(payloads["inputs"].get("validation_tokens") or 0)
    fig.suptitle("Post-LayerNorm ReLU Layer-3 Activation Densities", y=0.99)
    fig.text(
        0.5,
        0.947,
        f"Full deterministic validation cache ({validation_tokens:,} tokens); one seed per method; fixed budget",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.075,
        (
            "Shaded band marks |activation| <= 0.01; exact zeros share the central histogram bin. "
            f"The MLP-hidden view omits at most {100.0 * max_hidden_outside:.3f}% of mass per method."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.20, wspace=0.24)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_site_clipping_frontiers(
    site_series: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), sharey=False)
    legend_handles: dict[str, Any] = {}
    total_points = 0
    validation_tokens = 0

    for ax, (site, site_label) in zip(axes, REPORT04_CLIPPING_SITES, strict=True):
        panel_deltas: list[float] = []
        for item in site_series.get(site, []):
            rows = sorted(item["rows"], key=lambda row: float(row.get("threshold") or 0.0))
            if not rows:
                continue
            baseline = min(rows, key=lambda row: abs(float(row.get("threshold") or 0.0)))
            baseline_loss = float(baseline["validation_loss"])
            sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
            deltas = [float(row["validation_loss"]) - baseline_loss for row in rows]
            panel_deltas.extend(deltas)
            total_points += len(rows)
            validation_tokens = max(validation_tokens, max(int(row.get("validation_tokens") or 0) for row in rows))
            label = str(item["label"])
            (line,) = ax.plot(
                sparsity,
                deltas,
                color=REPORT04_METHOD_COLORS[label],
                marker=REPORT04_METHOD_MARKERS[label],
                markersize=3.0,
                linewidth=1.4,
                label=label,
            )
            legend_handles.setdefault(label, line)
        ax.axhline(0.0, color="#4d4d4d", linestyle="--", linewidth=0.8)
        ax.set_title(site_label)
        ax.set_xlabel("Achieved exact zeros (%)")
        ax.set_ylabel("Validation loss change from threshold 0")
        if panel_deltas:
            maximum = max(panel_deltas)
            minimum = min(panel_deltas)
            span = max(maximum - minimum, maximum, 1e-3)
            ax.set_ylim(min(-0.02 * span, minimum - 0.03 * span), maximum + 0.08 * span)

    fig.suptitle("Site-Specific Post-Hoc Clipping Frontiers", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            f"{total_points} sweep points; {validation_tokens:,} validation tokens per point; "
            "each curve is referenced to its own threshold-0 loss"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.105,
        "One seed per method; fixed training budget; panel-specific y scales are shown explicitly.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.87, bottom=0.29, wspace=0.28)
    fig.savefig(output_path)
    plt.close(fig)


def _nondominated_skip_loss_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nondominated: list[dict[str, Any]] = []
    for point in points:
        skip = float(point["skip"])
        loss = float(point["loss"])
        dominated = any(
            float(other["skip"]) >= skip
            and float(other["loss"]) <= loss
            and (float(other["skip"]) > skip or float(other["loss"]) < loss)
            for other in points
        )
        if not dominated:
            nondominated.append(point)
    return sorted(nondominated, key=lambda point: float(point["skip"]))


def _plot_report04_joint_compute_frontier(
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    all_points: list[dict[str, Any]] = []
    plotted_series: list[dict[str, Any]] = []
    threshold_zero_points: list[dict[str, Any]] = []
    validation_tokens = 0

    for item in series:
        rows = [row for row in item["rows"] if _finite(row.get("eligible_projection_skip_fraction"))]
        rows = sorted(rows, key=lambda row: float(row["eligible_projection_skip_fraction"]))
        if not rows:
            continue
        label = str(item["label"])
        x_values = [
            100.0 * float(row["eligible_projection_skip_fraction"]) * REPORT04_TARGET_MODEL_FRACTION
            for row in rows
        ]
        y_values = [float(row["validation_loss"]) for row in rows]
        ax.plot(
            x_values,
            y_values,
            color=REPORT04_METHOD_COLORS[label],
            marker=REPORT04_METHOD_MARKERS[label],
            markersize=4.0,
            linewidth=1.45,
            label=label,
            zorder=3,
        )
        threshold_zero = min(rows, key=lambda row: abs(float(row.get("threshold") or 0.0)))
        threshold_zero_points.append(
            {
                "label": label,
                "skip": (
                    100.0
                    * float(threshold_zero["eligible_projection_skip_fraction"])
                    * REPORT04_TARGET_MODEL_FRACTION
                ),
                "loss": float(threshold_zero["validation_loss"]),
            }
        )
        plotted_series.append({"label": label, "x": x_values, "y": y_values})
        validation_tokens = max(validation_tokens, max(int(row.get("validation_tokens") or 0) for row in rows))
        all_points.extend(
            {
                "skip": (
                    100.0 * float(row["eligible_projection_skip_fraction"]) * REPORT04_TARGET_MODEL_FRACTION
                ),
                "loss": float(row["validation_loss"]),
                "label": label,
                "threshold": float(row.get("threshold") or 0.0),
            }
            for row in rows
        )

    pareto = _nondominated_skip_loss_points(all_points)
    if pareto:
        pareto_x = [float(point["skip"]) for point in pareto]
        pareto_y = [float(point["loss"]) for point in pareto]
        ax.plot(
            pareto_x,
            pareto_y,
            color="#4d4d4d",
            linestyle="--",
            linewidth=2.0,
            label="Nondominated envelope",
            zorder=2,
        )
        ax.scatter(
            pareto_x,
            pareto_y,
            facecolors="none",
            edgecolors="#1a1a1a",
            linewidths=0.9,
            s=42,
            zorder=4,
        )

    inset = ax.inset_axes([0.06, 0.51, 0.47, 0.36])
    for item in plotted_series:
        label = str(item["label"])
        inset.plot(
            item["x"],
            item["y"],
            color=REPORT04_METHOD_COLORS[label],
            marker=REPORT04_METHOD_MARKERS[label],
            markersize=2.8,
            linewidth=1.15,
        )
    if pareto:
        inset.plot(pareto_x, pareto_y, color="#4d4d4d", linestyle="--", linewidth=1.4)
        inset.scatter(
            pareto_x,
            pareto_y,
            facecolors="none",
            edgecolors="#1a1a1a",
            linewidths=0.65,
            s=25,
            zorder=4,
        )
    annotation_offsets = ((6, 10), (-58, 12), (-44, -15), (8, -18), (-58, -15))
    for point, offset in zip(threshold_zero_points, annotation_offsets, strict=True):
        label = str(point["label"])
        inset.annotate(
            "t = 0",
            (float(point["skip"]), float(point["loss"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
            color=REPORT04_METHOD_COLORS[label],
            arrowprops={"arrowstyle": "-", "color": REPORT04_METHOD_COLORS[label], "linewidth": 0.6},
        )
    inset.set_xlim(
        min(float(point["skip"]) for point in all_points) - 1.0,
        max(float(point["skip"]) for point in threshold_zero_points) + 0.8,
    )
    inset.set_ylim(
        min(float(point["loss"]) for point in all_points) - 0.02,
        max(float(point["loss"]) for point in threshold_zero_points) + 0.08,
    )
    inset.set_title("Low-loss zoom (full frontier retained)", fontsize=7.5)
    inset.tick_params(axis="both", labelsize=6)
    inset.grid(alpha=0.2)
    ax.indicate_inset_zoom(inset, edgecolor="#666666", alpha=0.55, linewidth=0.8)

    ax.set_xlabel("Potentially avoidable model matmul products (%)")
    ax.set_ylabel("Absolute validation loss")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.112),
        frameon=False,
        fontsize=8,
        ncol=3,
        columnspacing=1.3,
        handlelength=2.8,
    )
    fig.suptitle("Joint Three-Site Clipping: Quality vs Model-Matmul Opportunity", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            f"{validation_tokens:,} validation tokens per point; nondominance minimizes loss and maximizes skip "
            "across all plotted sweep points"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.024,
        (
            "Numerator = exact-zero input products at QKV, W1, and W2. Denominator = all six-block causal-logical "
            "matmul products plus the LM head (9,192,192 products/token); maximum targetable share is 11.76%."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.066,
        "One seed per method; fixed 1.492B-token training budget. Dense kernels execute every plotted opportunity.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.245)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_parameter_diagnostics(
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [str(item["label"]) for item in series]
    fig = plt.figure(figsize=(11.8, 9.2))
    grid = fig.add_gridspec(3, 6, height_ratios=(1.15, 1.0, 1.0), hspace=0.46, wspace=0.55)
    weight_axes = {
        "qkv": fig.add_subplot(grid[0, 0:2]),
        "w1": fig.add_subplot(grid[0, 2:4]),
        "w2": fig.add_subplot(grid[0, 4:6]),
    }
    layer_norm_axes = {
        ("attention", "gamma"): fig.add_subplot(grid[1, 0:3]),
        ("mlp", "gamma"): fig.add_subplot(grid[1, 3:6]),
        ("attention", "beta"): fig.add_subplot(grid[2, 0:3]),
        ("mlp", "beta"): fig.add_subplot(grid[2, 3:6]),
    }
    legend_handles: dict[str, Any] = {}
    maximum_outside = 0

    for group_id, ax in weight_axes.items():
        positive_values = [
            float(value)
            for item in series
            for value in item["weight_groups"][group_id]["densities"]
            if float(value) > 0.0
        ]
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        for item in series:
            label = str(item["label"])
            group = item["weight_groups"][group_id]
            y_values = [max(float(value), y_min) if float(value) > 0.0 else y_min for value in group["densities"]]
            (line,) = ax.step(
                group["centers"],
                y_values,
                where="mid",
                color=REPORT04_METHOD_COLORS[label],
                linewidth=1.35,
                label=label,
            )
            legend_handles.setdefault(label, line)
            maximum_outside = max(maximum_outside, int(group["outside"]))
        reference_group = series[0]["weight_groups"][group_id]
        ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
        ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(y_min, max_density * 1.5)
        ax.set_xlim(*reference_group["range"])
        ax.set_title(
            f"{reference_group['label']}\n"
            f"n={int(reference_group['total']):,} per method; {int(reference_group['tensor_count'])} tensors",
            fontsize=9,
        )
        ax.set_xlabel("Weight value")
        ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
    weight_axes["qkv"].set_ylabel("Probability density (log scale)")

    parameter_titles = {"gamma": "gamma", "beta": "beta"}
    for (branch_id, parameter_id), ax in layer_norm_axes.items():
        for item in series:
            label = str(item["label"])
            branch = item["layer_norms"][branch_id]
            summaries = branch["parameters"][parameter_id]
            layers = [int(summary["layer"]) for summary in summaries]
            means = [float(summary["mean"]) for summary in summaries]
            stds = [float(summary["std"]) for summary in summaries]
            color = REPORT04_METHOD_COLORS[label]
            ax.fill_between(
                layers,
                [mean - std for mean, std in zip(means, stds, strict=True)],
                [mean + std for mean, std in zip(means, stds, strict=True)],
                color=color,
                alpha=0.07,
                linewidth=0,
            )
            ax.plot(
                layers,
                means,
                color=color,
                marker=REPORT04_METHOD_MARKERS[label],
                markersize=3.0,
                linewidth=1.25,
            )
        branch_label = series[0]["layer_norms"][branch_id]["label"]
        ax.set_title(f"{branch_label} {parameter_titles[parameter_id]}: feature mean +/- SD", fontsize=9)
        ax.set_xticks(range(6), [f"L{layer_index}" for layer_index in range(6)])
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel(f"{parameter_titles[parameter_id]} value")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.suptitle("Final-Checkpoint Weight and Branch-LayerNorm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.958,
        (
            f"{len(labels)} matched one-seed checkpoints; all weight densities aggregate "
            "the six transformer layers; biases excluded"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.058,
        (
            "LayerNorm bands are within-layer feature SD (n=128), not seed uncertainty. "
            f"Shading marks |weight| <= 0.01; maximum values outside plotted ranges: {maximum_outside}."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.092,
        "One seed per method; fixed 22,762-step / 1.492B-token training budget.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.17)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_activation_weight_densities(
    payloads: dict[str, dict[str, Any]],
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [label for label, _experiment_id in REPORT04_TRAINING_RUNS]
    by_label = {str(item["label"]): item for item in series}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"Missing report-04 parameter series: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.0))
    legend_handles: dict[str, Any] = {}
    activation_specs = (
        ("attention_inputs", "Attention-input ReLU output, layer 3", (-0.15, 2.0)),
        ("mlp_inputs", "MLP-input ReLU output, layer 3", (-0.15, 2.0)),
        ("mlp_hiddens", "MLP-hidden ReLU output, layer 3", (-0.05, 0.75)),
    )

    for ax, (site, title, x_limits) in zip(axes[0], activation_specs, strict=True):
        payload = _report04_histogram_payload(payloads, site)
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        densities_by_label: dict[str, list[float]] = {}
        positive: list[float] = []
        for label in labels:
            method = _histogram_method(payload, label)
            if method is None:
                raise ValueError(f"Missing histogram method {label!r} for {site}.")
            densities = _histogram_density(_histogram_layer(method, f"{site}.layer_3"), edges)
            densities_by_label[label] = densities
            positive.extend(value for value in densities if value > 0.0)
        max_density = max(positive, default=1.0)
        y_min = max(min(positive) * 0.7, max_density * 1e-5, 1e-8) if positive else 1e-8
        for label in labels:
            values = [max(value, y_min) if value > 0.0 else y_min for value in densities_by_label[label]]
            (line,) = ax.step(
                centers,
                values,
                where="mid",
                color=REPORT04_METHOD_COLORS[label],
                linewidth=1.25,
                label=label,
            )
            legend_handles.setdefault(label, line)
        ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
        ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(y_min, max_density * 1.5)
        ax.set_xlim(*x_limits)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Activation value")
        ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
    axes[0, 0].set_ylabel("Probability density (log scale)")

    for ax, group_id in zip(axes[1], ("qkv", "w1", "w2"), strict=True):
        positive = [
            float(value)
            for item in series
            for value in item["weight_groups"][group_id]["densities"]
            if float(value) > 0.0
        ]
        max_density = max(positive, default=1.0)
        y_min = max(min(positive) * 0.7, max_density * 1e-5, 1e-8) if positive else 1e-8
        for label in labels:
            group = by_label[label]["weight_groups"][group_id]
            values = [
                max(float(value), y_min) if float(value) > 0.0 else y_min
                for value in group["densities"]
            ]
            ax.step(
                group["centers"],
                values,
                where="mid",
                color=REPORT04_METHOD_COLORS[label],
                linewidth=1.25,
            )
        reference = by_label[labels[0]]["weight_groups"][group_id]
        ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
        ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(y_min, max_density * 1.5)
        ax.set_xlim(*reference["range"])
        ax.set_title(
            f"{reference['label']}\n"
            f"{int(reference['total']):,} weights across {int(reference['tensor_count'])} tensors",
            fontsize=9.5,
        )
        ax.set_xlabel("Weight value")
        ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
    axes[1, 0].set_ylabel("Probability density (log scale)")

    validation_tokens = int(payloads["inputs"].get("validation_tokens") or 0)
    fig.suptitle("Activation and Immediately Downstream Weight Distributions", y=0.987, fontsize=14)
    fig.text(
        0.5,
        0.953,
        (
            f"Activations: {validation_tokens:,} deterministic validation tokens at layer 3; "
            "weights: final checkpoints pooled over six layers"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.066,
        (
            "Gray bands mark |value| <= 0.01. ReLU point masses at exactly zero are quantified by direct counters, "
            "not by the continuous histogram density. One seed per method."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.14, hspace=0.36, wspace=0.27)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_layernorm_parameters(
    series: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [str(item["label"]) for item in series]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.9), sharex=True)
    panel_specs = (
        ("attention", "gamma", r"Attention branch scale $\gamma$"),
        ("mlp", "gamma", r"MLP branch scale $\gamma$"),
        ("attention", "beta", r"Attention branch bias $\beta$"),
        ("mlp", "beta", r"MLP branch bias $\beta$"),
    )
    legend_handles: dict[str, Any] = {}
    for ax, (branch_id, parameter_id, title) in zip(axes.flat, panel_specs, strict=True):
        for item in series:
            label = str(item["label"])
            summaries = item["layer_norms"][branch_id]["parameters"][parameter_id]
            layers = [int(summary["layer"]) for summary in summaries]
            means = [float(summary["mean"]) for summary in summaries]
            stds = [float(summary["std"]) for summary in summaries]
            color = REPORT04_METHOD_COLORS[label]
            ax.fill_between(
                layers,
                [mean - std for mean, std in zip(means, stds, strict=True)],
                [mean + std for mean, std in zip(means, stds, strict=True)],
                color=color,
                alpha=0.06,
                linewidth=0,
            )
            (line,) = ax.plot(
                layers,
                means,
                color=color,
                marker=REPORT04_METHOD_MARKERS[label],
                markersize=3.0,
                linewidth=1.25,
                label=label,
            )
            legend_handles.setdefault(label, line)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(6), [f"L{layer}" for layer in range(6)])
        ax.set_ylabel("Feature mean +/- within-layer SD")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1, 0].set_xlabel("Transformer layer")
    axes[1, 1].set_xlabel("Transformer layer")

    fig.suptitle("Branch LayerNorm Affine Parameters at the Final Checkpoint", y=0.985, fontsize=14)
    fig.text(
        0.5,
        0.946,
        r"For normalized feature $\hat{x}$, LayerNorm returns $y=\gamma\odot\hat{x}+\beta$ before the added ReLU.",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.066,
        "Bands are feature variation within one checkpoint (n=128), not seed uncertainty; one seed per method.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.15, hspace=0.33, wspace=0.24)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_three_relu_architecture(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.4, 6.0))
    ax.set_xlim(0.0, 1.30)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    gate_color = "#CC79A7"
    projection_color = "#56B4E9"
    dense_color = "#E8E8E8"
    residual_color = "#333333"

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text_value: str,
        *,
        facecolor: str = "white",
        edgecolor: str = "#555555",
        linewidth: float = 1.1,
        linestyle: str = "-",
        fontsize: float = 8.5,
        weight: str = "normal",
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text_value, ha="center", va="center", fontsize=fontsize, weight=weight)

    def arrow(start: tuple[float, float], end: tuple[float, float], *, color: str = residual_color, width: float = 1.2) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=width,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )

    ax.text(0.025, 0.94, "(a) One Pythia-14M parallel-residual transformer block", fontsize=12, weight="bold")
    box(0.020, 0.43, 0.070, 0.13, "Residual\n" + r"$H_l$", facecolor="white", edgecolor=residual_color, weight="bold")
    box(0.110, 0.70, 0.075, 0.10, r"LN$_{attn}$", facecolor=dense_color)
    box(0.205, 0.70, 0.065, 0.10, "ReLU", facecolor="#FBE3F0", edgecolor=gate_color, linewidth=2.0, weight="bold")
    box(
        0.290,
        0.675,
        0.150,
        0.15,
        "QKV\n" + r"$U_l[T\!\times\!128]\,W_{QKV}[128\!\times\!384]$" + "\n" + r"$\to[T\!\times\!384]$",
        facecolor="#DDF1FA",
        edgecolor=projection_color,
        linewidth=1.7,
        fontsize=7.2,
    )
    box(0.460, 0.70, 0.105, 0.10, "RoPE / QK\nsoftmax / PV", facecolor=dense_color, fontsize=8)
    box(0.585, 0.70, 0.085, 0.10, r"$W_o$" + "\n" + r"$[T\!\times\!128]$", facecolor=dense_color)

    box(0.110, 0.20, 0.075, 0.10, r"LN$_{mlp}$", facecolor=dense_color)
    box(0.205, 0.20, 0.065, 0.10, "ReLU", facecolor="#FBE3F0", edgecolor=gate_color, linewidth=2.0, weight="bold")
    box(
        0.290,
        0.175,
        0.150,
        0.15,
        r"MLP up $W_1$" + "\n" + r"$M_l[T\!\times\!128]W_1[128\!\times\!512]$" + "\n" + r"$\to[T\!\times\!512]$",
        facecolor="#DDF1FA",
        edgecolor=projection_color,
        linewidth=1.7,
        fontsize=7.2,
    )
    box(0.460, 0.20, 0.065, 0.10, "ReLU", facecolor="#FBE3F0", edgecolor=gate_color, linewidth=2.0, weight="bold")
    box(
        0.545,
        0.175,
        0.150,
        0.15,
        r"MLP down $W_2$" + "\n" + r"$G_l[T\!\times\!512]W_2[512\!\times\!128]$" + "\n" + r"$\to[T\!\times\!128]$",
        facecolor="#DDF1FA",
        edgecolor=projection_color,
        linewidth=1.7,
        fontsize=7.2,
    )
    box(0.720, 0.43, 0.070, 0.13, r"$+$" + "\nparallel", facecolor="white", edgecolor=residual_color, fontsize=9, weight="bold")
    box(0.810, 0.43, 0.070, 0.13, r"$H_{l+1}$", facecolor="white", edgecolor=residual_color, weight="bold")

    arrow((0.090, 0.495), (0.110, 0.75))
    arrow((0.185, 0.75), (0.205, 0.75))
    arrow((0.270, 0.75), (0.290, 0.75), color=gate_color, width=2.0)
    arrow((0.440, 0.75), (0.460, 0.75))
    arrow((0.565, 0.75), (0.585, 0.75))
    arrow((0.670, 0.75), (0.755, 0.56))
    arrow((0.090, 0.495), (0.110, 0.25))
    arrow((0.185, 0.25), (0.205, 0.25))
    arrow((0.270, 0.25), (0.290, 0.25), color=gate_color, width=2.0)
    arrow((0.440, 0.25), (0.460, 0.25))
    arrow((0.525, 0.25), (0.545, 0.25), color=gate_color, width=2.0)
    arrow((0.695, 0.25), (0.755, 0.43))
    arrow((0.090, 0.495), (0.720, 0.495))
    arrow((0.790, 0.495), (0.810, 0.495))

    ax.text(0.2375, 0.655, "attention_inputs", ha="center", va="top", fontsize=7.5, color=gate_color, weight="bold")
    ax.text(0.2375, 0.155, "mlp_inputs", ha="center", va="top", fontsize=7.5, color=gate_color, weight="bold")
    ax.text(0.4925, 0.155, "mlp_hiddens", ha="center", va="top", fontsize=7.5, color=gate_color, weight="bold")
    ax.text(0.515, 0.62, "dense after projection or mixing", ha="center", fontsize=7.5, color="#666666")
    box(0.705, 0.08, 0.175, 0.075, "Final model LayerNorm: unchanged", facecolor="white", edgecolor="#777777", linestyle="--", fontsize=7.8)

    ax.plot([0.905, 0.905], [0.08, 0.90], color="#B0B0B0", linewidth=1.0)
    ax.text(0.930, 0.88, "(b) What each exact-zero gate can skip", fontsize=11, weight="bold")
    local_rows = (
        (0.70, "attention_inputs", r"QKV: $U_l[T\!\times\!128]W_{QKV}[128\!\times\!384]\to[T\!\times\!384]$", r"$3d^2=49{,}152$ products/token"),
        (0.50, "mlp_inputs", r"MLP up: $M_l[T\!\times\!128]W_1[128\!\times\!512]\to[T\!\times\!512]$", r"$4d^2=65{,}536$ products/token"),
        (0.30, "mlp_hiddens", r"MLP down: $G_l[T\!\times\!512]W_2[512\!\times\!128]\to[T\!\times\!128]$", r"$4d^2=65{,}536$ products/token"),
    )
    for y_value, site, operation, products in local_rows:
        box(0.930, y_value, 0.085, 0.095, site.replace("_", "\n"), facecolor="#FBE3F0", edgecolor=gate_color, fontsize=7.1)
        arrow((1.015, y_value + 0.0475), (1.035, y_value + 0.0475), color=gate_color, width=1.8)
        ax.text(1.045, y_value + 0.067, operation, ha="left", va="center", fontsize=7.7)
        ax.text(1.045, y_value + 0.025, products, ha="left", va="center", fontsize=7.7, weight="bold")
    ax.text(
        0.930,
        0.215,
        r"$C_{target}(z)=3d^2z_a+4d^2z_m+4d^2z_h$" + "\n"
        + r"$R_{block}=C_{target}/C_{block}$;  $R_{model}=LC_{target}/C_{model}$",
        ha="left",
        va="top",
        fontsize=8.0,
    )
    ax.text(
        0.930,
        0.125,
        "Architecture ceilings when " + r"$z_a=z_m=z_h=1$" + ":\n"
        + r"$R_{block}^{max}=39.27\%$;  $R_{model}^{max}=11.76\%$ (six blocks + LM head)",
        ha="left",
        va="top",
        fontsize=8.0,
        weight="bold",
    )
    fig.suptitle("Three-ReLU Pythia-14M: Exact-Zero Gates and Immediately Downstream Matmuls", y=0.995, fontsize=15)
    fig.text(
        0.5,
        0.01,
        (
            "Original Pythia-specific rendering. Magenta: exact-zero-producing ReLU; blue: immediately eligible "
            "projection; gray: downstream dense computation. Percentages are mathematical ceilings, not observed speedups."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.08)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_report04_pythia_family_compute_ceiling(output_path: Path) -> None:
    model_labels: list[str] = []
    targetable_shares: list[float] = []
    other_block_shares: list[float] = []
    lm_head_shares: list[float] = []

    for model_label, num_layers, hidden_size, vocab_size in REPORT04_PYTHIA_FAMILY:
        targetable_products = 11 * num_layers * hidden_size**2
        other_block_products = num_layers * (
            hidden_size**2 + hidden_size * (REPORT04_BLOCK_SIZE + 1)
        )
        lm_head_products = hidden_size * vocab_size
        model_products = targetable_products + other_block_products + lm_head_products
        model_labels.append(model_label)
        targetable_shares.append(100.0 * targetable_products / model_products)
        other_block_shares.append(100.0 * other_block_products / model_products)
        lm_head_shares.append(100.0 * lm_head_products / model_products)

    y_positions = list(range(len(model_labels)))
    lm_head_left = [
        targetable_share + other_block_share
        for targetable_share, other_block_share in zip(
            targetable_shares,
            other_block_shares,
            strict=True,
        )
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.barh(
        y_positions,
        targetable_shares,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.5,
        label=r"Three-ReLU targetable ($R_{model}^{max}$)",
    )
    ax.barh(
        y_positions,
        other_block_shares,
        left=targetable_shares,
        color="#999999",
        edgecolor="white",
        linewidth=0.5,
        label="Other block products",
    )
    ax.barh(
        y_positions,
        lm_head_shares,
        left=lm_head_left,
        color="#E69F00",
        edgecolor="white",
        linewidth=0.5,
        label="LM head",
    )

    for y_value, share in zip(y_positions, targetable_shares, strict=True):
        ax.text(
            share / 2.0,
            y_value,
            f"{share:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=8.0,
            weight="bold",
        )

    ax.text(
        lm_head_left[0] + lm_head_shares[0] / 2.0,
        y_positions[0],
        "LM head\n70.0%",
        ha="center",
        va="center",
        color="#222222",
        fontsize=7.8,
        weight="bold",
    )
    ax.annotate(
        "LM head 2.2%",
        xy=(100.0 - lm_head_shares[-1] / 2.0, y_positions[-1]),
        xytext=(97.0, y_positions[-1]),
        ha="right",
        va="center",
        fontsize=7.8,
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
    )

    ax.set_yticks(y_positions, model_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Share of full-model scalar products (%)")
    ax.set_title("Pythia-Family Model-Matmul Denominator at $T=2{,}048$", pad=36)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=7.8)
    fig.text(
        0.5,
        0.015,
        (
            r"Architecture ceiling: $z_a=z_m=z_h=1$. Other block products are $W_o$ plus valid-causal QK and PV; "
            "percentages are logical product shares, not speedups."
        ),
        ha="center",
        va="bottom",
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.79, bottom=0.14)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_layernorm_relu_propagation_heatmaps(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    methods = list(payload.get("methods", []))
    if len(methods) != 4:
        raise ValueError("Activation-propagation heatmaps require exactly four matched checkpoints.")

    num_layers = int(methods[0].get("num_layers") or 0)
    if num_layers <= 0 or any(int(method.get("num_layers") or 0) != num_layers for method in methods):
        raise ValueError("Activation-propagation payload has inconsistent layer counts.")

    matrices = [
        _propagation_matrix(method, "activations", PROPAGATION_ACTIVATION_ROWS, num_layers)
        for method in methods
    ]

    fig = plt.figure(figsize=(14.2, 9.4))
    outer_grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        height_ratios=(1.0, 1.0),
        hspace=0.34,
        wspace=0.11,
    )
    axes = [fig.add_subplot(outer_grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_axis = fig.add_subplot(outer_grid[:, 2])
    column_labels = [f"L{layer}" for layer in range(num_layers)] + ["All"]
    image = None

    for method_index, method in enumerate(methods):
        method_label = str(method.get("label") or method.get("config_id") or f"Method {method_index + 1}")
        _row_index, column_index = divmod(method_index, 2)
        ax = axes[method_index]
        image = ax.imshow(
            matrices[method_index],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=100.0,
        )
        ax.set_title(
            f"({chr(ord('a') + method_index)}) {method_label}: exact-zero activation outputs (%)",
            fontsize=10.5,
        )
        _format_propagation_heatmap(
            ax,
            image,
            matrices[method_index],
            row_labels=[label for _name, label in PROPAGATION_ACTIVATION_ROWS],
            column_labels=column_labels,
            show_row_labels=column_index == 0,
            separators=(0.5, 8.5, 13.5),
            emphasized_rows=(2, 7, 10, 12),
        )
        if method_index >= 2:
            ax.set_xlabel("Transformer layer; All pools integer counts over L0-L5")

    if image is None:
        raise ValueError("Activation-propagation payload has no plottable methods.")
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Exact-zero scalar fraction (%)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    validation_tokens = int(payload.get("validation_tokens") or 0)
    validation_sequences = int(payload.get("validation_sequences") or 0)
    block_size = int(payload.get("block_size") or 0)
    trailing_tokens = int(payload.get("trailing_tokens_excluded") or 0)
    fig.suptitle("Where Exact Zeros Persist Through Pythia-14M Blocks", y=0.995, fontsize=15)
    fig.text(
        0.5,
        0.958,
        (
            f"Direct integer counts over {validation_sequences:,} x {block_size:,} = "
            f"{validation_tokens:,} evaluated validation tokens; {trailing_tokens:,} incomplete-tail tokens excluded"
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.055,
        (
            "Exact zero = number of produced scalar values equal to numeric 0 divided by all produced values at that "
            f"stage/layer. Width-128 denominator: {validation_tokens:,} x 128 = "
            f"{validation_tokens * 128:,}; width-512: {validation_tokens * 512:,}."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.text(
        0.5,
        0.032,
        (
            r"Parallel residual: both branch LayerNorms consume $H_l$ and $H_{l+1}=H_l+O_l+M_l$. "
            "White rules separate residual, attention, MLP, and output stages."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.22, right=0.93, top=0.89, bottom=0.105)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_layernorm_relu_zero_product_heatmaps(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    methods = list(payload.get("methods", []))
    if len(methods) != 4:
        raise ValueError("Zero-product heatmaps require exactly four matched checkpoints.")
    num_layers = int(methods[0].get("num_layers") or 0)
    if num_layers <= 0 or any(int(method.get("num_layers") or 0) != num_layers for method in methods):
        raise ValueError("Activation-propagation payload has inconsistent layer counts.")

    matrices = [
        _propagation_matrix(method, "matmuls", PROPAGATION_MATMUL_ROWS, num_layers)
        for method in methods
    ]
    fig = plt.figure(figsize=(12.6, 7.2))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        hspace=0.43,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_axis = fig.add_subplot(grid[:, 2])
    column_labels = [f"L{layer}" for layer in range(num_layers)] + ["All"]
    image = None

    for method_index, method in enumerate(methods):
        label = str(method.get("label") or method.get("config_id") or f"Method {method_index + 1}")
        _row_index, column_index = divmod(method_index, 2)
        block_opportunity = _propagation_weighted_fraction(method, "matmuls")
        model_opportunity = block_opportunity * REPORT04_BLOCK_PRODUCTS_PER_TOKEN / REPORT04_MODEL_PRODUCTS_PER_TOKEN
        ax = axes[method_index]
        image = ax.imshow(matrices[method_index], aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
        ax.set_title(
            f"({chr(ord('a') + method_index)}) {label}\n"
            f"all block matmuls: {block_opportunity:.1f}% | model incl. LM head: {model_opportunity:.1f}%",
            fontsize=9.5,
        )
        _format_propagation_heatmap(
            ax,
            image,
            matrices[method_index],
            row_labels=[row_label for _name, row_label in PROPAGATION_MATMUL_ROWS],
            column_labels=column_labels,
            show_row_labels=column_index == 0,
            separators=(3.5,),
            emphasized_rows=(0, 4, 5),
        )
        if method_index >= 2:
            ax.set_xlabel("Transformer layer; All pools integer counts over L0-L5")

    if image is None:
        raise ValueError("Activation-propagation payload has no plottable methods.")
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Products with an exact-zero activation operand (%)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)

    validation_tokens = int(payload.get("validation_tokens") or 0)
    fig.suptitle("Local Exact-Zero Matmul Opportunities", y=0.985, fontsize=14)
    fig.text(
        0.5,
        0.944,
        (
            f"Direct integer counts over {validation_tokens:,} validation tokens; valid lower-triangular causal pairs "
            "only (future-mask zeros excluded)"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.035,
        (
            "A counted product has at least one activation operand exactly equal to numeric 0. "
            "Percentages are ideal logical opportunities; dense kernels still execute them."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.24, right=0.92, top=0.87, bottom=0.13)
    fig.savefig(output_path)
    plt.close(fig)


def _propagation_matrix(
    method: dict[str, Any],
    kind: str,
    row_specs: tuple[tuple[str, str], ...],
    num_layers: int,
) -> list[list[float]]:
    lookup = {
        (str(row.get("name")), int(row.get("layer"))): row
        for row in method.get(kind, [])
    }
    matrix: list[list[float]] = []
    for name, _label in row_specs:
        values: list[float] = []
        pooled_zero = 0
        pooled_total = 0
        for layer in range(num_layers):
            row = lookup.get((name, layer))
            if row is None:
                raise ValueError(
                    f"Missing activation-propagation row {kind}/{name}/layer_{layer} "
                    f"for {method.get('label')!r}."
                )
            zero_count = int(row.get("zero_count") or 0)
            total = int(row.get("total") or 0)
            if total <= 0:
                raise ValueError(f"Activation-propagation row {kind}/{name}/layer_{layer} has no denominator.")
            values.append(100.0 * zero_count / total)
            pooled_zero += zero_count
            pooled_total += total
        values.append(100.0 * pooled_zero / pooled_total)
        matrix.append(values)
    return matrix


def _propagation_weighted_fraction(method: dict[str, Any], kind: str) -> float:
    rows = list(method.get(kind, []))
    zero_count = sum(int(row.get("zero_count") or 0) for row in rows)
    total = sum(int(row.get("total") or 0) for row in rows)
    if total <= 0:
        raise ValueError(f"Activation-propagation payload has no {kind} denominator.")
    return 100.0 * zero_count / total


def _format_propagation_heatmap(
    ax: Any,
    image: Any,
    matrix: list[list[float]],
    *,
    row_labels: list[str],
    column_labels: list[str],
    show_row_labels: bool,
    separators: tuple[float, ...],
    emphasized_rows: tuple[int, ...],
) -> None:
    ax.grid(False)
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)))
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=8.2)
        for index in emphasized_rows:
            ax.get_yticklabels()[index].set_fontweight("bold")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(len(column_labels) - 1.5, color="white", linewidth=1.8, alpha=0.95)
    for separator in separators:
        ax.axhline(separator, color="white", linewidth=1.4, alpha=0.9)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            red, green, blue, _alpha = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                column_index,
                row_index,
                _propagation_cell_label(value),
                ha="center",
                va="center",
                fontsize=6.9,
                color="black" if luminance >= 0.56 else "white",
            )


def _propagation_cell_label(percent: float) -> str:
    if percent == 0.0:
        return "0"
    if percent < 0.001:
        return f"{percent:.1e}"
    if percent < 0.1:
        return f"{percent:.3f}"
    return f"{percent:.1f}"
