from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from paper_exp.utils import read_json


COLORBLIND_SAFE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
ADAMW_COLOR = "#000000"
ADAMW_LINEWIDTH = 2.6
ADAMW_MARKER_SCALE = 1.35
DEFAULT_SERIES_LINEWIDTH = 1.2

PRESSURE_SHORT_RUNS = [
    ("AdamW baseline", "03-pythia-14m-minipile-random-full-10min"),
    ("Ricker naive", "08-pythia-14m-minipile-ricker-naive-short"),
    ("L1 naive", "09-pythia-14m-minipile-l1-naive-short"),
    ("Orthogonal Ricker", "10-pythia-14m-minipile-orthogonal-ricker-short"),
    ("Orthogonal L1", "11-pythia-14m-minipile-orthogonal-l1-short"),
]

FIXED_STEP_SWEEP_NAME = "pressure_fixed_step_v1"
FIXED_STEP_ROLE_LABELS = {
    "adamw": "AdamW",
    "ricker_naive": "Ricker naive",
    "orthogonal_ricker": "Orthogonal Ricker",
    "l1_naive": "L1 naive",
    "orthogonal_l1": "Orthogonal L1",
}
FIXED_STEP_ROLE_MARKERS = {
    "adamw": "D",
    "ricker_naive": "o",
    "orthogonal_ricker": "s",
    "l1_naive": "^",
    "orthogonal_l1": "P",
}
FIXED_STEP_METHOD_FIGURES = (
    (8, "ricker_naive", "naive-ricker", "Naive Ricker"),
    (9, "l1_naive", "naive-l1", "Naive L1"),
    (10, "orthogonal_ricker", "orthogonal-ricker", "Orthogonal Ricker"),
    (11, "orthogonal_l1", "orthogonal-l1", "Orthogonal L1"),
)
SELECTED_CLIPPING_FRONTIER_CONFIGS = (12, 29, 33, 18, 25)
HIGH_PRESSURE_EXPANSION_CONFIGS = tuple([12, *range(35, 49)])
HIGH_PRESSURE_LEARNING_FIGURES = (
    (17, (12, *range(35, 40)), "rn", "High-pressure RN Learning Curves", "AdamW plus RN configs 35-39"),
    (18, (12, *range(40, 45)), "or", "High-pressure OR Learning Curves", "AdamW plus OR configs 40-44"),
    (19, (12, 45, 46, 47, 48), "l1", "High-pressure L1N/OL1 Learning Curves", "AdamW plus L1N/OL1 configs 45-48"),
)
HIGH_PRESSURE_OR_L1_NORM_CONFIGS = tuple([12, *range(40, 49)])
SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT = "49-pythia-14m-pressure-fixed-2048-selected-activation-histograms"
FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT = (
    "60-pythia-14m-minipile-full-pass-high-pressure-activation-histograms"
)
FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "63-pythia-14m-minipile-full-pass-high-pressure-residual-stream-histograms"
)
FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "64-pythia-14m-minipile-full-pass-high-pressure-attention-output-histograms"
)
FULL_PASS_HIGH_PRESSURE_WEIGHT_HISTOGRAM_EXPERIMENT = (
    "61-pythia-14m-minipile-full-pass-high-pressure-weight-histograms"
)
FULL_PASS_HIGH_PRESSURE_ATTENTION_WEIGHT_HISTOGRAM_EXPERIMENT = (
    "62-pythia-14m-minipile-full-pass-high-pressure-attention-weight-histograms"
)
FULL_PASS_ALL_SITE_MLP_HISTOGRAM_EXPERIMENT = (
    "67-pythia-14m-minipile-full-pass-all-site-pressure-mlp-activation-histograms"
)
FULL_PASS_ALL_SITE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "68-pythia-14m-minipile-full-pass-all-site-pressure-residual-stream-histograms"
)
FULL_PASS_ALL_SITE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "69-pythia-14m-minipile-full-pass-all-site-pressure-attention-output-histograms"
)
FULL_PASS_MLP_RESIDUAL_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "74-pythia-14m-minipile-full-pass-mlp-residual-pressure-attention-output-histograms"
)
FULL_PASS_MLP_RESIDUAL_MLP_HISTOGRAM_EXPERIMENT = (
    "75-pythia-14m-minipile-full-pass-mlp-residual-pressure-mlp-activation-histograms"
)
FULL_PASS_MLP_RESIDUAL_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "76-pythia-14m-minipile-full-pass-mlp-residual-pressure-residual-stream-histograms"
)
FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "82-pythia-14m-minipile-relu-completed-attention-output-histograms"
)
FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT = (
    "83-pythia-14m-minipile-relu-completed-mlp-activation-histograms"
)
FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "84-pythia-14m-minipile-relu-completed-residual-stream-histograms"
)
FULL_PASS_RELU_MLP_INPUT_HISTOGRAM_EXPERIMENT = (
    "85-pythia-14m-minipile-relu-completed-mlp-input-histograms"
)
RELU_SITE_SCOPE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT = (
    "94-pythia-14m-minipile-relu-site-scope-attention-output-histograms"
)
RELU_SITE_SCOPE_MLP_HISTOGRAM_EXPERIMENT = (
    "95-pythia-14m-minipile-relu-site-scope-mlp-activation-histograms"
)
RELU_SITE_SCOPE_RESIDUAL_HISTOGRAM_EXPERIMENT = (
    "96-pythia-14m-minipile-relu-site-scope-residual-stream-histograms"
)
RELU_SITE_SCOPE_MLP_INPUT_HISTOGRAM_EXPERIMENT = (
    "97-pythia-14m-minipile-relu-site-scope-mlp-input-histograms"
)
STATUS_UPDATE_COUPLING_METHODS = (
    ("AdamW", "AdamW"),
    ("L1N w5", "L1N w5"),
    ("RN w1 c0.05 s0.05", "RN w1 c0.05 s0.05"),
)
STATUS_UPDATE_RELU_COUPLING_METHODS = (
    ("AdamW ReLU", "AdamW ReLU"),
    ("RN ReLU", "RN ReLU"),
    ("OR ReLU", "OR ReLU"),
    ("L1N ReLU", "L1N ReLU"),
    ("OL1 ReLU", "OL1 ReLU"),
)
STATUS_UPDATE_COUPLING_SITE_SPECS = (
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
STATUS_UPDATE_RELU_COUPLING_SITE_SPECS = (
    ("mlp_inputs", "Post-LN MLP inputs", "mlp_inputs.layer_3", (-4.0, 4.0)),
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
STATUS_UPDATE_COUPLING_WEIGHT_RUNS = (
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("L1N w5", "58-pythia-14m-minipile-l1-naive-full-pass-w5"),
    ("RN w1 c0.05 s0.05", "57-pythia-14m-minipile-ricker-naive-full-pass-w1-c0p05-s0p05"),
)
STATUS_UPDATE_RELU_WEIGHT_RUNS = (
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
)
STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS = (
    (
        "mlp_hidden_input",
        "MLP hiddens",
        re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "residual_write",
        "Residual streams",
        re.compile(r"^gpt_neox\.layers\.\d+\.(?:mlp\.dense_4h_to_h|attention\.dense)\.weight$"),
        (-0.75, 0.75),
    ),
    (
        "attention_qkv",
        "Attention",
        re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
        (-0.75, 0.75),
    ),
)
STATUS_UPDATE_MLP_WEIGHT_LAYER_GROUPS = (
    (
        "mlp_layer_0",
        "MLP layer 0",
        re.compile(r"^gpt_neox\.layers\.0\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "mlp_layer_3",
        "MLP layer 3",
        re.compile(r"^gpt_neox\.layers\.3\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
    (
        "mlp_layer_5",
        "MLP layer 5",
        re.compile(r"^gpt_neox\.layers\.5\.mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$"),
        (-0.25, 0.25),
    ),
)
FULL_PASS_SELECTED_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("L1N w0.5", "51-pythia-14m-minipile-l1-naive-full-pass-w0p5"),
    ("OL1 w0.5", "52-pythia-14m-minipile-orthogonal-l1-full-pass-w0p5"),
    ("RN w0.1 c0.05 s0.025", "53-pythia-14m-minipile-ricker-naive-full-pass-w0p1-c0p05-s0p025"),
    ("OR w0.1 c0.05 s0.025", "54-pythia-14m-minipile-orthogonal-ricker-full-pass-w0p1-c0p05-s0p025"),
]
FULL_PASS_HIGH_PRESSURE_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("RN w1 c0.05 s0.05", "57-pythia-14m-minipile-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR w1 c0.05 s0.05", "56-pythia-14m-minipile-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N w5", "58-pythia-14m-minipile-l1-naive-full-pass-w5"),
    ("OL1 w5", "59-pythia-14m-minipile-orthogonal-l1-full-pass-w5"),
]
FULL_PASS_ALL_SITE_PRESSURE_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("OR all-site w1 c0.05 s0.05", "65-pythia-14m-minipile-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
    ("OL1 all-site w5", "66-pythia-14m-minipile-orthogonal-l1-all-site-full-pass-w5"),
]
FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS = [
    ("RN MLP+res w1 c0.05 s0.05", "70-pythia-14m-minipile-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("OR MLP+res w1 c0.05 s0.05", "71-pythia-14m-minipile-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("L1N MLP+res w5", "72-pythia-14m-minipile-l1-naive-mlp-residual-full-pass-w5"),
    ("OL1 MLP+res w5", "73-pythia-14m-minipile-orthogonal-l1-mlp-residual-full-pass-w5"),
]
FULL_PASS_RELU_COMPLETED_RUNS = [
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU w1 c0.05 s0.05", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU w1 c0.05 s0.05", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU w5", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU w5", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
]
STATUS_UPDATE_CLIPPING_SITES = (
    ("mlp_hiddens", "MLP hiddens"),
    ("residual_streams", "Residual streams"),
    ("attention_outputs", "Attention outputs"),
)
STATUS_UPDATE_ALL_SITE_FRONTIER_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("OR all-site", "65-pythia-14m-minipile-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
    ("OL1 all-site", "66-pythia-14m-minipile-orthogonal-l1-all-site-full-pass-w5"),
]
STATUS_UPDATE_MLP_RESIDUAL_FRONTIER_RUNS = [
    ("AdamW", "50-pythia-14m-minipile-adamw-full-pass"),
    ("RN MLP+res", "70-pythia-14m-minipile-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("OR MLP+res", "71-pythia-14m-minipile-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
    ("L1N MLP+res", "72-pythia-14m-minipile-l1-naive-mlp-residual-full-pass-w5"),
    ("OL1 MLP+res", "73-pythia-14m-minipile-orthogonal-l1-mlp-residual-full-pass-w5"),
]
STATUS_UPDATE_RELU_FRONTIER_RUNS = [
    ("AdamW ReLU", "77-pythia-14m-minipile-relu-adamw-full-pass"),
    ("RN ReLU", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
    ("OR ReLU", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
    ("L1N ReLU", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
    ("OL1 ReLU", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
]
RELU_SITE_SCOPE_TRAINING_GROUPS = (
    (
        "MLP-only",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "78-pythia-14m-minipile-relu-ricker-naive-full-pass-w1-c0p05-s0p05"),
            ("OR", "79-pythia-14m-minipile-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05"),
            ("L1N", "80-pythia-14m-minipile-relu-l1-naive-full-pass-w5"),
            ("OL1", "81-pythia-14m-minipile-relu-orthogonal-l1-full-pass-w5"),
        ),
    ),
    (
        "MLP+residual",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "86-pythia-14m-minipile-relu-ricker-naive-mlp-residual-full-pass-w1-c0p05-s0p05"),
            ("OR", "87-pythia-14m-minipile-relu-orthogonal-ricker-mlp-residual-full-pass-w1-c0p05-s0p05"),
            ("L1N", "88-pythia-14m-minipile-relu-l1-naive-mlp-residual-full-pass-w5"),
            ("OL1", "89-pythia-14m-minipile-relu-orthogonal-l1-mlp-residual-full-pass-w5"),
        ),
    ),
    (
        "All-site",
        (
            ("AdamW", "77-pythia-14m-minipile-relu-adamw-full-pass"),
            ("RN", "90-pythia-14m-minipile-relu-ricker-naive-all-site-full-pass-w1-c0p05-s0p05"),
            ("OR", "91-pythia-14m-minipile-relu-orthogonal-ricker-all-site-full-pass-w1-c0p05-s0p05"),
            ("L1N", "92-pythia-14m-minipile-relu-l1-naive-all-site-full-pass-w5"),
            ("OL1", "93-pythia-14m-minipile-relu-orthogonal-l1-all-site-full-pass-w5"),
        ),
    ),
)
RELU_SITE_SCOPE_HISTOGRAM_GROUPS = (
    (
        "MLP-only",
        (
            ("AdamW", "AdamW"),
            ("RN", "MLP-only RN"),
            ("OR", "MLP-only OR"),
            ("L1N", "MLP-only L1N"),
            ("OL1", "MLP-only OL1"),
        ),
    ),
    (
        "MLP+residual",
        (
            ("AdamW", "AdamW"),
            ("RN", "MLP+res RN"),
            ("OR", "MLP+res OR"),
            ("L1N", "MLP+res L1N"),
            ("OL1", "MLP+res OL1"),
        ),
    ),
    (
        "All-site",
        (
            ("AdamW", "AdamW"),
            ("RN", "All-site RN"),
            ("OR", "All-site OR"),
            ("L1N", "All-site L1N"),
            ("OL1", "All-site OL1"),
        ),
    ),
)
RELU_SITE_SCOPE_REPORT_METHODS = (
    ("AdamW", "AdamW"),
    ("MLP-only RN", "MLP-only RN"),
    ("MLP-only OR", "MLP-only OR"),
    ("MLP-only L1N", "MLP-only L1N"),
    ("MLP-only OL1", "MLP-only OL1"),
    ("MLP+res RN", "MLP+res RN"),
    ("MLP+res OR", "MLP+res OR"),
    ("MLP+res L1N", "MLP+res L1N"),
    ("MLP+res OL1", "MLP+res OL1"),
    ("All-site RN", "All-site RN"),
    ("All-site OR", "All-site OR"),
    ("All-site L1N", "All-site L1N"),
    ("All-site OL1", "All-site OL1"),
)
RELU_SITE_SCOPE_DENSITY_SITE_SPECS = (
    ("mlp_inputs", "Post-LN MLP inputs", "mlp_inputs.layer_3", (-4.0, 4.0)),
    ("mlp_hiddens", "MLP hiddens", "mlp_hiddens.layer_3", (-0.08, 0.75)),
    ("residual_streams", "Residual streams", "residual_streams.layer_3", (-0.35, 0.35)),
    ("attention_outputs", "Attention outputs", "attention_outputs.layer_3", (-0.35, 0.35)),
)
RELU_SITE_SCOPE_NEAR_ZERO_SITES = (
    ("mlp_hiddens", "MLP hiddens"),
    ("residual_streams", "Residual streams"),
    ("attention_outputs", "Attention outputs"),
)

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
REPORT04_METHOD_COLORS = {
    "GELU AdamW": "#000000",
    "MLP-ReLU AdamW": "#0072B2",
    "MLP-ReLU OL1": "#E69F00",
    "Three-ReLU AdamW": "#009E73",
    "Three-ReLU OR": "#D55E00",
    "Three-ReLU L1N": "#56B4E9",
    "Three-ReLU OL1": "#CC79A7",
}
REPORT04_METHOD_MARKERS = {
    "GELU AdamW": "D",
    "MLP-ReLU AdamW": "o",
    "MLP-ReLU OL1": "s",
    "Three-ReLU AdamW": "^",
    "Three-ReLU OR": "X",
    "Three-ReLU L1N": "v",
    "Three-ReLU OL1": "P",
}
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

PLOT_STYLE = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_COLORS),
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def generate_plots(
    *,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_dir)
    outputs = _generate_known_paper_figures(results_path, figures_path, save_png=save_png)
    if outputs:
        return outputs

    rows = collect_numeric_metrics(results_path)
    output_pdf = figures_path / "01-results-summary.pdf"
    if rows:
        _plot_metric_summary(rows, output_pdf)
    else:
        _plot_empty_placeholder(output_pdf)

    outputs = [output_pdf]
    if save_png:
        output_png = output_pdf.with_suffix(".png")
        if rows:
            _plot_metric_summary(rows, output_png)
        else:
            _plot_empty_placeholder(output_png)
        outputs.append(output_png)

    return outputs


def _generate_known_paper_figures(results_path: Path, figures_path: Path, *, save_png: bool) -> list[Path]:
    outputs: list[Path] = []

    diagnostic_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min",
        "events.jsonl",
    )
    if diagnostic_run is not None:
        output_pdf = figures_path / "01-pythia-14m-minipile-random-full-10min-diagnostics.pdf"
        outputs.extend(generate_run_diagnostics(run_dir=diagnostic_run, output=output_pdf, save_png=save_png))

    clipping_run = _latest_run_with(
        results_path / "03-pythia-14m-minipile-random-full-10min-clipping-sweep",
        "clipping_frontier.jsonl",
    )
    if clipping_run is not None:
        output_pdf = figures_path / "02-pythia-14m-minipile-clipping-frontier-smoke.pdf"
        outputs.extend(generate_clipping_frontier(run_dir=clipping_run, output=output_pdf, save_png=save_png))

    pressure_runs = _latest_labeled_runs(results_path, PRESSURE_SHORT_RUNS, "events.jsonl")
    if len(pressure_runs) >= 2:
        output_pdf = figures_path / "03-pythia-14m-pressure-short-learning-curves.pdf"
        outputs.extend(generate_pressure_comparison(runs=pressure_runs, output=output_pdf, save_png=save_png))

    clipping_runs = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in PRESSURE_SHORT_RUNS[1:]],
        "clipping_frontier.jsonl",
    )
    if len(clipping_runs) >= 2:
        output_pdf = figures_path / "04-pythia-14m-pressure-short-clipping-frontiers.pdf"
        outputs.extend(generate_clipping_comparison(runs=clipping_runs, output=output_pdf, save_png=save_png))

    fixed_step_rows = _load_fixed_step_sweep_rows(results_path)
    if len(fixed_step_rows) >= 2:
        output_pdf = figures_path / "05-pythia-14m-pressure-fixed-2048-summary.pdf"
        outputs.extend(
            generate_fixed_step_sweep_summary(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

        output_pdf = figures_path / "06-pythia-14m-pressure-fixed-2048-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_learning_curves(rows=fixed_step_rows, output=output_pdf, save_png=save_png)
        )

    fixed_step_clipping = _load_fixed_step_clipping_series(results_path, fixed_step_rows)
    if len(fixed_step_clipping) >= 2:
        output_pdf = figures_path / "07-pythia-14m-pressure-fixed-2048-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_clipping_frontiers(
                series=fixed_step_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_rows = _select_fixed_step_rows_for_role(fixed_step_rows, role)
        if len(role_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_role_learning_curves(
                rows=role_rows,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for offset, role, slug, role_label in FIXED_STEP_METHOD_FIGURES:
        role_series = _select_fixed_step_clipping_for_role(fixed_step_clipping, role)
        if len(role_series) < 2:
            continue
        output_pdf = figures_path / f"{offset + 4:02d}-pythia-14m-pressure-fixed-2048-{slug}-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_role_clipping_frontiers(
                series=role_series,
                role_label=role_label,
                output=output_pdf,
                save_png=save_png,
            )
        )

    selected_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        SELECTED_CLIPPING_FRONTIER_CONFIGS,
    )
    if len(selected_clipping) == len(SELECTED_CLIPPING_FRONTIER_CONFIGS):
        output_pdf = figures_path / "16-pythia-14m-pressure-fixed-2048-selected-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_selected_clipping_frontiers(
                series=selected_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    for index, config_indices, slug, title, scope_note in HIGH_PRESSURE_LEARNING_FIGURES:
        high_pressure_rows = _select_fixed_step_rows_by_config_indices(
            fixed_step_rows,
            config_indices,
        )
        if len(high_pressure_rows) < 2:
            continue
        output_pdf = figures_path / f"{index:02d}-pythia-14m-pressure-fixed-2048-high-pressure-{slug}-learning-curves.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_learning_curves(
                rows=high_pressure_rows,
                output=output_pdf,
                title=title,
                scope_note=scope_note,
                save_png=save_png,
            )
        )

    high_pressure_clipping = _select_fixed_step_clipping_by_config_indices(
        fixed_step_clipping,
        HIGH_PRESSURE_EXPANSION_CONFIGS,
    )
    if len(high_pressure_clipping) >= 2:
        output_pdf = figures_path / "20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_clipping_frontiers(
                series=high_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    high_pressure_or_l1_rows = _select_fixed_step_rows_by_config_indices(
        fixed_step_rows,
        HIGH_PRESSURE_OR_L1_NORM_CONFIGS,
    )
    if len(high_pressure_or_l1_rows) >= 2:
        output_pdf = figures_path / "21-pythia-14m-pressure-fixed-2048-or-l1-weight-norms.pdf"
        outputs.extend(
            generate_fixed_step_high_pressure_weight_norms(
                rows=high_pressure_or_l1_rows,
                output=output_pdf,
                save_png=save_png,
            )
        )

    activation_histogram_run = _latest_run_with(
        results_path / SELECTED_ACTIVATION_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if activation_histogram_run is not None:
        output_pdf = figures_path / "22-pythia-14m-pressure-fixed-2048-selected-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=activation_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_runs = _latest_labeled_runs(results_path, FULL_PASS_SELECTED_RUNS, "events.jsonl")
    if len(full_pass_runs) >= 2:
        output_pdf = figures_path / "28-pythia-14m-minipile-full-pass-selected-gradient-diagnostics.pdf"
        outputs.extend(
            generate_full_pass_gradient_diagnostics(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )
        output_pdf = figures_path / "30-pythia-14m-minipile-full-pass-pressure-dominance.pdf"
        outputs.extend(
            generate_full_pass_pressure_dominance(
                runs=full_pass_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_rms_clipping = _latest_labeled_runs(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in FULL_PASS_SELECTED_RUNS],
        "clipping_frontier.jsonl",
    )
    full_pass_rms_clipping = [
        (label, run)
        for label, run in full_pass_rms_clipping
        if _is_rms_clipping_run(run)
    ]
    if len(full_pass_rms_clipping) >= 2:
        output_pdf = figures_path / "29-pythia-14m-minipile-full-pass-selected-rms-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_rms_clipping_frontiers(
                runs=full_pass_rms_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_runs = _latest_labeled_runs(results_path, FULL_PASS_HIGH_PRESSURE_RUNS, "events.jsonl")
    if len(full_pass_high_pressure_runs) >= 2:
        output_pdf = figures_path / "31-pythia-14m-minipile-full-pass-high-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "32-pythia-14m-minipile-full-pass-high-pressure-weight-norms.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_weight_norms(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "34-pythia-14m-minipile-full-pass-high-pressure-gradient-diagnostics.pdf"
        outputs.extend(
            generate_full_pass_gradient_diagnostics(
                runs=full_pass_high_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass High-pressure Gradient Diagnostics",
                subtitle=(
                    "n={n} runs; {events} train log events; "
                    "pressure panels omit AdamW where no pressure is applied"
                ),
            )
        )

    full_pass_high_pressure_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep") for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=lambda run: not _is_rms_clipping_run(run),
    )
    if len(full_pass_high_pressure_clipping) >= 2:
        output_pdf = figures_path / "33-pythia-14m-minipile-full-pass-high-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_high_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_all_site_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep-all-sites") for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=_is_all_site_clipping_run,
    )
    if len(full_pass_high_pressure_all_site_clipping) >= 2:
        output_pdf = figures_path / "40-pythia-14m-minipile-full-pass-high-pressure-all-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_high_pressure_all_site_clipping,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass High-pressure All-site Post-hoc Clipping Frontiers",
                subtitle=(
                    "absolute thresholds applied to MLP hiddens, attention outputs, and residual streams; "
                    "validation-loss axis zoomed"
                ),
            )
        )

    full_pass_high_pressure_site_frontiers = (
        (
            49,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass MLP-pressure High-pressure MLP-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to MLP hiddens",
        ),
        (
            50,
            "residual_streams",
            "Residual Streams",
            "Full-pass MLP-pressure High-pressure Residual-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to residual streams",
        ),
        (
            51,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass MLP-pressure High-pressure Attention-only Post-hoc Clipping Frontiers",
            "pressure trained only on MLP hiddens; clipping thresholds applied only to attention outputs",
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_high_pressure_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_HIGH_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-high-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=f"{subtitle}; validation-loss axis zoomed",
            )
        )

    full_pass_all_site_pressure_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_ALL_SITE_PRESSURE_RUNS,
        "events.jsonl",
    )
    if len(full_pass_all_site_pressure_runs) >= 2:
        output_pdf = figures_path / "41-pythia-14m-minipile-full-pass-all-site-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_all_site_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass All-site Pressure Learning Curves",
                subtitle="AdamW baseline plus all-site OR and OL1; one MiniPile token-cache pass per pressure run",
            )
        )

    full_pass_all_site_pressure_clipping = _latest_labeled_runs_filtered(
        results_path,
        [(label, f"{experiment_id}-clipping-sweep-all-sites") for label, experiment_id in FULL_PASS_ALL_SITE_PRESSURE_RUNS],
        "clipping_frontier.jsonl",
        predicate=_is_all_site_clipping_run,
    )
    if len(full_pass_all_site_pressure_clipping) >= 2:
        output_pdf = figures_path / "42-pythia-14m-minipile-full-pass-all-site-pressure-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=full_pass_all_site_pressure_clipping,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass All-site Pressure Post-hoc Clipping Frontiers",
                subtitle=(
                    "absolute thresholds applied to MLP hiddens, attention outputs, and residual streams; "
                    "validation-loss axis zoomed"
                ),
            )
        )

    full_pass_all_site_pressure_site_frontiers = (
        (
            46,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass All-site Pressure MLP-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to MLP hiddens; validation-loss axis zoomed",
        ),
        (
            47,
            "residual_streams",
            "Residual Streams",
            "Full-pass All-site Pressure Residual-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to residual streams; validation-loss axis zoomed",
        ),
        (
            48,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass All-site Pressure Attention-only Post-hoc Clipping Frontiers",
            "absolute thresholds applied only to attention outputs; validation-loss axis zoomed",
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_all_site_pressure_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_ALL_SITE_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-all-site-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_mlp_residual_pressure_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS,
        "events.jsonl",
    )
    if len(full_pass_mlp_residual_pressure_runs) >= 2:
        output_pdf = figures_path / "52-pythia-14m-minipile-full-pass-mlp-residual-pressure-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_mlp_residual_pressure_runs,
                output=output_pdf,
                save_png=save_png,
                title="Full-pass MLP+Residual Pressure Learning Curves",
                subtitle=(
                    "pressure applied to MLP hiddens and residual streams; "
                    "one MiniPile token-cache pass per run"
                ),
            )
        )

    full_pass_mlp_residual_site_frontiers = (
        (
            53,
            "attention_outputs",
            "Attention Outputs",
            "Full-pass MLP+Residual Pressure Attention-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to attention outputs; validation-loss axis zoomed"
            ),
        ),
        (
            54,
            "residual_streams",
            "Residual Streams",
            "Full-pass MLP+Residual Pressure Residual-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to residual streams; validation-loss axis zoomed"
            ),
        ),
        (
            55,
            "mlp_hiddens",
            "MLP Hiddens",
            "Full-pass MLP+Residual Pressure MLP-only Post-hoc Clipping Frontiers",
            (
                "pressure trained on MLP hiddens and residual streams; "
                "clipping thresholds applied only to MLP hiddens; validation-loss axis zoomed"
            ),
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_mlp_residual_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_MLP_RESIDUAL_PRESSURE_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = (
            figures_path
            / f"{figure_index:02d}-pythia-14m-minipile-full-pass-mlp-residual-pressure-{slug}-clipping-frontiers.pdf"
        )
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_high_pressure_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_histogram_run is not None:
        output_pdf = figures_path / "35-pythia-14m-minipile-full-pass-high-pressure-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_weight_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_WEIGHT_HISTOGRAM_EXPERIMENT,
        "weight_histograms.json",
    )
    if full_pass_high_pressure_weight_histogram_run is not None:
        output_pdf = figures_path / "36-pythia-14m-minipile-full-pass-high-pressure-weight-histograms.pdf"
        outputs.extend(
            generate_weight_histogram_grid(
                run_dir=full_pass_high_pressure_weight_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_attention_weight_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_WEIGHT_HISTOGRAM_EXPERIMENT,
        "weight_histograms.json",
    )
    if full_pass_high_pressure_attention_weight_histogram_run is not None:
        output_pdf = figures_path / "37-pythia-14m-minipile-full-pass-high-pressure-attention-weight-histograms.pdf"
        outputs.extend(
            generate_weight_histogram_grid(
                run_dir=full_pass_high_pressure_attention_weight_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_residual_histogram_run is not None:
        output_pdf = figures_path / "38-pythia-14m-minipile-full-pass-high-pressure-residual-stream-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_high_pressure_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_high_pressure_attention_output_histogram_run is not None:
        output_pdf = figures_path / "39-pythia-14m-minipile-full-pass-high-pressure-attention-output-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_high_pressure_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_mlp_histogram_run is not None:
        output_pdf = figures_path / "43-pythia-14m-minipile-full-pass-all-site-pressure-mlp-activation-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_residual_histogram_run is not None:
        output_pdf = figures_path / "44-pythia-14m-minipile-full-pass-all-site-pressure-residual-stream-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_all_site_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_ALL_SITE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_all_site_attention_output_histogram_run is not None:
        output_pdf = figures_path / "45-pythia-14m-minipile-full-pass-all-site-pressure-attention-output-histograms.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_all_site_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_attention_output_histogram_run is not None:
        output_pdf = (
            figures_path
            / "56-pythia-14m-minipile-full-pass-mlp-residual-pressure-attention-output-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_mlp_histogram_run is not None:
        output_pdf = (
            figures_path
            / "57-pythia-14m-minipile-full-pass-mlp-residual-pressure-mlp-activation-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_mlp_residual_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_MLP_RESIDUAL_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_mlp_residual_residual_histogram_run is not None:
        output_pdf = (
            figures_path
            / "58-pythia-14m-minipile-full-pass-mlp-residual-pressure-residual-stream-histograms.pdf"
        )
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_mlp_residual_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_completed_runs = _latest_labeled_runs(
        results_path,
        FULL_PASS_RELU_COMPLETED_RUNS,
        "events.jsonl",
    )
    if len(full_pass_relu_completed_runs) >= 2:
        output_pdf = figures_path / "59-pythia-14m-minipile-relu-completed-learning-curves.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_learning_curves(
                runs=full_pass_relu_completed_runs,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Method Learning Curves",
                subtitle=(
                    "completed ReLU runs: AdamW, RN, OR, L1N, and OL1; "
                    "one MiniPile token-cache pass per run"
                ),
            )
        )

    full_pass_relu_site_frontiers = (
        (
            60,
            "attention_outputs",
            "Attention Outputs",
            "ReLU Completed Methods Attention-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to attention outputs; "
                "validation-loss axis zoomed"
            ),
        ),
        (
            61,
            "residual_streams",
            "Residual Streams",
            "ReLU Completed Methods Residual-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to residual streams; "
                "validation-loss axis zoomed"
            ),
        ),
        (
            62,
            "mlp_hiddens",
            "MLP Hiddens",
            "ReLU Completed Methods MLP-only Post-hoc Clipping Frontiers",
            (
                "completed ReLU runs; clipping thresholds applied only to MLP hiddens; "
                "validation-loss axis zoomed"
            ),
        ),
    )
    for figure_index, site, slug_label, title, subtitle in full_pass_relu_site_frontiers:
        site_suffix = site.replace("_", "-")
        site_clipping = _latest_labeled_runs_filtered(
            results_path,
            [
                (label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}")
                for label, experiment_id in FULL_PASS_RELU_COMPLETED_RUNS
            ],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
        if len(site_clipping) < 2:
            continue
        slug = slug_label.lower().replace(" ", "-")
        output_pdf = figures_path / f"{figure_index:02d}-pythia-14m-minipile-relu-completed-{slug}-clipping-frontiers.pdf"
        outputs.extend(
            generate_full_pass_high_pressure_clipping_frontiers(
                runs=site_clipping,
                output=output_pdf,
                save_png=save_png,
                title=title,
                subtitle=subtitle,
            )
        )

    full_pass_relu_attention_output_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_attention_output_histogram_run is not None:
        output_pdf = figures_path / "63-pythia-14m-minipile-relu-completed-attention-output-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_attention_output_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_mlp_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_mlp_histogram_run is not None:
        output_pdf = figures_path / "64-pythia-14m-minipile-relu-completed-mlp-activation-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_mlp_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    full_pass_relu_residual_histogram_run = _latest_run_with(
        results_path / FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT,
        "activation_histograms.json",
    )
    if full_pass_relu_residual_histogram_run is not None:
        output_pdf = figures_path / "65-pythia-14m-minipile-relu-completed-residual-stream-densities.pdf"
        outputs.extend(
            generate_activation_histogram_grid(
                run_dir=full_pass_relu_residual_histogram_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_coupling_histogram_runs = {
        "mlp_hiddens": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_ACTIVATION_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / FULL_PASS_HIGH_PRESSURE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(report_coupling_histogram_runs.values()):
        output_pdf = figures_path / "66-status-update-gelu-mlp-only-coupling-density-comparison.pdf"
        outputs.extend(
            generate_status_update_coupling_density_comparison(
                histogram_runs=report_coupling_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_relu_coupling_histogram_runs = {
        "mlp_inputs": _latest_run_with(
            results_path / FULL_PASS_RELU_MLP_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / FULL_PASS_RELU_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / FULL_PASS_RELU_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / FULL_PASS_RELU_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(report_relu_coupling_histogram_runs.values()):
        output_pdf = figures_path / "73-status-update-relu-coupling-density-comparison.pdf"
        outputs.extend(
            generate_status_update_coupling_density_comparison(
                histogram_runs=report_relu_coupling_histogram_runs,
                output=output_pdf,
                save_png=save_png,
                methods=STATUS_UPDATE_RELU_COUPLING_METHODS,
                site_specs=STATUS_UPDATE_RELU_COUPLING_SITE_SPECS,
                title="ReLU MLP-only Pressure Coupling Diagnostic",
                subtitle=(
                    "rows are completed ReLU AdamW, RN, OR, L1N, and OL1"
                ),
            )
        )

    report_pressure_weight_runs = _latest_labeled_runs(
        results_path,
        list(STATUS_UPDATE_COUPLING_WEIGHT_RUNS),
        "checkpoints/final/model.safetensors",
    )
    if report_pressure_weight_runs:
        output_pdf = figures_path / "70-status-update-gelu-mlp-only-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_pressure_weight_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report_relu_weight_runs = _latest_labeled_runs(
        results_path,
        list(STATUS_UPDATE_RELU_WEIGHT_RUNS),
        "checkpoints/final/model.safetensors",
    )
    if report_relu_weight_runs:
        output_pdf = figures_path / "71-status-update-relu-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_relu_weight_runs,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Pressure Weight Diagnostic",
                subtitle=(
                    "rows are completed AdamW, RN, OR, L1N, and OL1 ReLU runs; "
                    "columns aggregate final-checkpoint weights across all six layers"
                ),
                footnote=(
                    "Shaded band marks |weight| <= 0.01. "
                    "Residual-stream column uses MLP and attention output-projection weights."
                ),
            )
        )

        output_pdf = figures_path / "72-status-update-relu-mlp-weight-density-comparison.pdf"
        outputs.extend(
            generate_status_update_pressure_weight_diagnostic(
                runs=report_relu_weight_runs,
                output=output_pdf,
                save_png=save_png,
                group_specs=STATUS_UPDATE_MLP_WEIGHT_LAYER_GROUPS,
                title="ReLU Completed MLP Dense Weight Diagnostic",
                subtitle=(
                    "rows are completed AdamW, RN, OR, L1N, and OL1 ReLU runs; "
                    "columns show representative MLP dense layers"
                ),
                footnote="Shaded band marks |weight| <= 0.01. Biases excluded.",
            )
        )

    report_all_site_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_ALL_SITE_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_all_site_clipping):
        output_pdf = figures_path / "67-status-update-all-site-pressure-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_all_site_clipping,
                output=output_pdf,
                save_png=save_png,
                title="All-site Pressure: Site-specific Post-hoc Clipping",
                subtitle="one panel per clipped activation family; AdamW baseline included",
            )
        )

    report_mlp_residual_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_MLP_RESIDUAL_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_mlp_residual_clipping):
        output_pdf = figures_path / "68-status-update-mlp-residual-pressure-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_mlp_residual_clipping,
                output=output_pdf,
                save_png=save_png,
                title="MLP+Residual Pressure: Site-specific Post-hoc Clipping",
                subtitle="pressure trained on MLP hiddens and residual streams; AdamW baseline included",
            )
        )

    report_relu_clipping = _latest_status_update_site_clipping_runs(
        results_path,
        STATUS_UPDATE_RELU_FRONTIER_RUNS,
    )
    if _has_status_update_site_clipping_runs(report_relu_clipping):
        output_pdf = figures_path / "69-status-update-relu-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_status_update_site_clipping_frontiers(
                site_runs=report_relu_clipping,
                output=output_pdf,
                save_png=save_png,
                title="ReLU Completed Runs: Site-specific Post-hoc Clipping",
                subtitle="completed AdamW/RN/OR/L1N/OL1 ReLU runs",
            )
        )

    relu_site_scope_training_groups = _latest_grouped_labeled_runs(
        results_path,
        RELU_SITE_SCOPE_TRAINING_GROUPS,
        "events.jsonl",
    )
    if relu_site_scope_training_groups and all(len(runs) >= 2 for _scope, runs in relu_site_scope_training_groups):
        output_pdf = figures_path / "74-pythia-14m-minipile-relu-site-scope-learning-curves.pdf"
        outputs.extend(
            generate_relu_site_scope_learning_curves(
                grouped_runs=relu_site_scope_training_groups,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_clipping = _latest_relu_site_scope_site_clipping_groups(results_path)
    if _has_relu_site_scope_site_clipping_runs(relu_site_scope_clipping):
        output_pdf = figures_path / "75-status-update-relu-site-scope-clipping-frontiers.pdf"
        outputs.extend(
            generate_relu_site_scope_clipping_frontiers(
                grouped_site_runs=relu_site_scope_clipping,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_histogram_runs = {
        "mlp_inputs": _latest_run_with(
            results_path / RELU_SITE_SCOPE_MLP_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / RELU_SITE_SCOPE_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "residual_streams": _latest_run_with(
            results_path / RELU_SITE_SCOPE_RESIDUAL_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "attention_outputs": _latest_run_with(
            results_path / RELU_SITE_SCOPE_ATTENTION_OUTPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(relu_site_scope_histogram_runs.values()):
        output_pdf = figures_path / "76-status-update-relu-site-scope-activation-density-comparison.pdf"
        outputs.extend(
            generate_relu_site_scope_activation_density_comparison(
                histogram_runs=relu_site_scope_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "78-status-update-relu-site-scope-near-zero-heatmaps.pdf"
        outputs.extend(
            generate_relu_site_scope_near_zero_heatmaps(
                histogram_runs=relu_site_scope_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    relu_site_scope_weight_groups = _latest_grouped_labeled_runs(
        results_path,
        RELU_SITE_SCOPE_TRAINING_GROUPS,
        "checkpoints/final/model.safetensors",
    )
    if relu_site_scope_weight_groups and all(len(runs) >= 2 for _scope, runs in relu_site_scope_weight_groups):
        output_pdf = figures_path / "77-status-update-relu-site-scope-pressure-weight-diagnostic.pdf"
        outputs.extend(
            generate_relu_site_scope_pressure_weight_diagnostic(
                grouped_runs=relu_site_scope_weight_groups,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report04_training_runs = _latest_labeled_runs(
        results_path,
        list(REPORT04_TRAINING_RUNS),
        "events.jsonl",
    )
    if len(report04_training_runs) == len(REPORT04_TRAINING_RUNS):
        output_pdf = figures_path / "79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf"
        outputs.extend(
            generate_report04_learning_diagnostics(
                runs=report04_training_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "87-pythia-14m-minipile-three-relu-architecture-compute-map.pdf"
        outputs.extend(generate_report04_three_relu_architecture(output=output_pdf, save_png=save_png))

        output_pdf = figures_path / "90-pythia-family-three-relu-model-compute-ceilings.pdf"
        outputs.extend(generate_report04_pythia_family_compute_ceiling(output=output_pdf, save_png=save_png))

    report04_histogram_runs = {
        "inputs": _latest_run_with(
            results_path / REPORT04_INPUT_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
        "mlp_hiddens": _latest_run_with(
            results_path / REPORT04_MLP_HISTOGRAM_EXPERIMENT,
            "activation_histograms.json",
        ),
    }
    if all(report04_histogram_runs.values()):
        output_pdf = figures_path / "80-pythia-14m-minipile-post-layernorm-relu-activation-heatmaps.pdf"
        outputs.extend(
            generate_report04_activation_heatmaps(
                histogram_runs=report04_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "81-pythia-14m-minipile-post-layernorm-relu-activation-densities.pdf"
        outputs.extend(
            generate_report04_activation_densities(
                histogram_runs=report04_histogram_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report04_site_clipping_runs: dict[str, list[tuple[str, Path]]] = {}
    for site, _site_label in REPORT04_CLIPPING_SITES:
        suffix = site.replace("_", "-")
        experiments = [
            (label, f"{experiment_id}-clipping-sweep-report04-{suffix}")
            for label, experiment_id in REPORT04_CLIPPING_RUNS
        ]
        report04_site_clipping_runs[site] = _latest_labeled_runs(
            results_path,
            experiments,
            "clipping_frontier.jsonl",
        )
    if all(len(runs) == len(REPORT04_CLIPPING_RUNS) for runs in report04_site_clipping_runs.values()):
        output_pdf = figures_path / "82-pythia-14m-minipile-post-layernorm-relu-site-clipping-frontiers.pdf"
        outputs.extend(
            generate_report04_site_clipping_frontiers(
                site_runs=report04_site_clipping_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report04_joint_clipping_runs = _latest_labeled_runs(
        results_path,
        [
            (label, f"{experiment_id}-clipping-sweep-report04-joint")
            for label, experiment_id in REPORT04_JOINT_CLIPPING_RUNS
        ],
        "clipping_frontier.jsonl",
    )
    if len(report04_joint_clipping_runs) == len(REPORT04_JOINT_CLIPPING_RUNS):
        output_pdf = figures_path / "83-pythia-14m-minipile-post-layernorm-relu-joint-compute-frontier.pdf"
        outputs.extend(
            generate_report04_joint_compute_frontier(
                runs=report04_joint_clipping_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

    report04_parameter_runs = _latest_labeled_runs(
        results_path,
        list(REPORT04_TRAINING_RUNS),
        "checkpoints/final/model.safetensors",
    )
    if len(report04_parameter_runs) == len(REPORT04_TRAINING_RUNS):
        output_pdf = figures_path / "84-pythia-14m-minipile-post-layernorm-relu-parameter-diagnostics.pdf"
        outputs.extend(
            generate_report04_parameter_diagnostics(
                runs=report04_parameter_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = figures_path / "89-pythia-14m-minipile-post-layernorm-relu-layernorm-parameters.pdf"
        outputs.extend(
            generate_report04_layernorm_parameters(
                runs=report04_parameter_runs,
                output=output_pdf,
                save_png=save_png,
            )
        )

        if all(report04_histogram_runs.values()):
            output_pdf = figures_path / "88-pythia-14m-minipile-post-layernorm-relu-activation-weight-densities.pdf"
            outputs.extend(
                generate_report04_activation_weight_densities(
                    histogram_runs=report04_histogram_runs,
                    runs=report04_parameter_runs,
                    output=output_pdf,
                    save_png=save_png,
                )
            )

    propagation_run = _latest_run_with(
        results_path / POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT,
        "activation_propagation.json",
    )
    if propagation_run is not None:
        output_pdf = figures_path / "85-pythia-14m-minipile-post-layernorm-relu-zero-propagation-heatmaps.pdf"
        outputs.extend(
            generate_post_layernorm_relu_propagation_heatmaps(
                run_dir=propagation_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

        output_pdf = (
            figures_path
            / "86-pythia-14m-minipile-post-layernorm-relu-zero-product-propagation-heatmaps.pdf"
        )
        outputs.extend(
            generate_post_layernorm_relu_zero_product_heatmaps(
                run_dir=propagation_run,
                output=output_pdf,
                save_png=save_png,
            )
        )

    return outputs


def _latest_run_with(experiment_dir: Path, artifact_name: str) -> Path | None:
    if not experiment_dir.exists():
        return None
    candidates = [path for path in sorted(experiment_dir.iterdir()) if (path / artifact_name).exists()]
    return candidates[-1] if candidates else None


def generate_run_diagnostics(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = _read_jsonl(run_path / "events.jsonl")
    metrics = read_json(run_path / "metrics.json")
    train_events = [event for event in events if event.get("event") == "train"]
    validation_events = [event for event in events if event.get("event") == "validation"]
    if not train_events:
        raise ValueError(f"No train events found in {run_path / 'events.jsonl'}")

    _plot_run_diagnostics(train_events, validation_events, metrics, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_run_diagnostics(train_events, validation_events, metrics, png_path)
        outputs.append(png_path)
    return outputs


def generate_clipping_frontier(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frontier_path = run_path / "clipping_frontier.jsonl"
    rows = _read_jsonl(frontier_path)
    if not rows:
        raise ValueError(f"No clipping frontier rows found in {frontier_path}")

    _plot_clipping_frontier(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_frontier(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_pressure_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_pressure_series(runs)
    if not series:
        raise ValueError("No pressure comparison runs with train events were found.")

    _plot_pressure_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_pressure_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_clipping_comparison(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No clipping comparison runs were found.")

    _plot_clipping_comparison(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_clipping_comparison(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_gradient_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass Selected Methods Gradient Diagnostics",
    subtitle: str | None = None,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_gradient_diagnostics(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_gradient_diagnostics(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_full_pass_rms_clipping_frontiers(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No RMS clipping comparison runs were found.")

    _plot_full_pass_rms_clipping_frontiers(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_rms_clipping_frontiers(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_pressure_dominance(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No full-pass train events were found.")

    _plot_full_pass_pressure_dominance(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_pressure_dominance(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_learning_curves(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass High-pressure Learning Curves",
    subtitle: str = "one MiniPile token-cache pass per run",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass train events were found.")

    _plot_full_pass_high_pressure_learning_curves(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_learning_curves(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_weight_norms(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass weight-norm events were found.")

    _plot_full_pass_high_pressure_weight_norms(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_weight_norms(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_full_pass_high_pressure_clipping_frontiers(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    title: str = "Full-pass High-pressure Post-hoc Clipping Frontiers",
    subtitle: str = "absolute thresholds, no RMS normalization; validation-loss axis zoomed",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No high-pressure full-pass clipping frontier runs were found.")

    _plot_full_pass_high_pressure_clipping_frontiers(series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_full_pass_high_pressure_clipping_frontiers(series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_status_update_coupling_density_comparison(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
    methods: tuple[tuple[str, str], ...] = STATUS_UPDATE_COUPLING_METHODS,
    site_specs: tuple[tuple[str, str, str, tuple[float, float]], ...] = STATUS_UPDATE_COUPLING_SITE_SPECS,
    title: str = "GELU MLP-only Pressure Coupling Diagnostic",
    subtitle: str = "rows are AdamW, L1N w5, and RN w1 c0.05 s0.05",
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_status_update_coupling_density_comparison(
        payloads,
        output_path,
        methods=methods,
        site_specs=site_specs,
        title=title,
        subtitle=subtitle,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_coupling_density_comparison(
            payloads,
            png_path,
            methods=methods,
            site_specs=site_specs,
            title=title,
            subtitle=subtitle,
        )
        outputs.append(png_path)
    return outputs


def generate_status_update_pressure_weight_diagnostic(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
    group_specs: tuple[tuple[str, str, Any, tuple[float, float]], ...] = STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS,
    title: str = "GELU MLP-only Pressure Weight Diagnostic",
    subtitle: str = (
        "rows are AdamW, L1N w5, and RN w1 c0.05 s0.05; "
        "columns aggregate final-checkpoint weights across all six layers"
    ),
    footnote: str = (
        "Shaded band marks |weight| <= 0.01. "
        "Residual-stream column uses MLP and attention output-projection weights."
    ),
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_status_update_pressure_weight_groups(runs, group_specs=group_specs)
    _plot_status_update_pressure_weight_diagnostic(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        footnote=footnote,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_pressure_weight_diagnostic(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            footnote=footnote,
        )
        outputs.append(png_path)
    return outputs


def generate_status_update_site_clipping_frontiers(
    *,
    site_runs: dict[str, list[tuple[str, str | Path]]],
    output: str | Path,
    title: str,
    subtitle: str,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    site_series = {site: _load_clipping_series(runs) for site, runs in site_runs.items()}
    _plot_status_update_site_clipping_frontiers(site_series, output_path, title=title, subtitle=subtitle)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_status_update_site_clipping_frontiers(site_series, png_path, title=title, subtitle=subtitle)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_learning_curves(
    *,
    grouped_runs: list[tuple[str, list[tuple[str, str | Path]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_series = [(scope, _load_event_series(runs)) for scope, runs in grouped_runs]
    grouped_series = [(scope, series) for scope, series in grouped_series if series]
    if not grouped_series:
        raise ValueError("No ReLU site-scope learning-curve runs were found.")

    _plot_relu_site_scope_learning_curves(grouped_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_learning_curves(grouped_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_clipping_frontiers(
    *,
    grouped_site_runs: list[tuple[str, dict[str, list[tuple[str, str | Path]]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_site_series = [
        (scope, {site: _load_clipping_series(runs) for site, runs in site_runs.items()})
        for scope, site_runs in grouped_site_runs
    ]
    grouped_site_series = [(scope, site_series) for scope, site_series in grouped_site_series if site_series]
    if not grouped_site_series:
        raise ValueError("No ReLU site-scope clipping frontier runs were found.")

    _plot_relu_site_scope_clipping_frontiers(grouped_site_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_clipping_frontiers(grouped_site_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_activation_density_comparison(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_relu_site_scope_activation_density_comparison(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_activation_density_comparison(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_pressure_weight_diagnostic(
    *,
    grouped_runs: list[tuple[str, list[tuple[str, str | Path]]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_series = [
        (scope, _load_status_update_pressure_weight_groups(runs, group_specs=STATUS_UPDATE_PRESSURE_WEIGHT_GROUPS))
        for scope, runs in grouped_runs
    ]
    grouped_series = [(scope, series) for scope, series in grouped_series if series]
    if not grouped_series:
        raise ValueError("No ReLU site-scope pressure-weight runs were found.")

    _plot_relu_site_scope_pressure_weight_diagnostic(grouped_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_pressure_weight_diagnostic(grouped_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_relu_site_scope_near_zero_heatmaps(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        site: read_json(Path(run_dir) / "activation_histograms.json")
        for site, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_relu_site_scope_near_zero_heatmaps(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_relu_site_scope_near_zero_heatmaps(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_learning_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_event_series(runs)
    if not series:
        raise ValueError("No report-04 learning-diagnostic runs were found.")

    _plot_report04_learning_diagnostics(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_learning_diagnostics(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_activation_heatmaps(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_report04_activation_heatmaps(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_activation_heatmaps(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_activation_densities(
    *,
    histogram_runs: dict[str, str | Path | None],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    _plot_report04_activation_densities(payloads, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_activation_densities(payloads, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_site_clipping_frontiers(
    *,
    site_runs: dict[str, list[tuple[str, str | Path]]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    site_series = {site: _load_clipping_series(runs) for site, runs in site_runs.items()}
    _plot_report04_site_clipping_frontiers(site_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_site_clipping_frontiers(site_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_joint_compute_frontier(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_clipping_series(runs)
    if not series:
        raise ValueError("No report-04 joint clipping runs were found.")
    _plot_report04_joint_compute_frontier(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_joint_compute_frontier(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_parameter_diagnostics(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    _plot_report04_parameter_diagnostics(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_parameter_diagnostics(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_activation_weight_densities(
    *,
    histogram_runs: dict[str, str | Path | None],
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        key: read_json(Path(run_dir) / "activation_histograms.json")
        for key, run_dir in histogram_runs.items()
        if run_dir is not None
    }
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    _plot_report04_activation_weight_densities(payloads, series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_activation_weight_densities(payloads, series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_layernorm_parameters(
    *,
    runs: list[tuple[str, str | Path]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _load_report04_parameter_series(runs)
    if not series:
        raise ValueError("No report-04 parameter checkpoints were found.")
    _plot_report04_layernorm_parameters(series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_layernorm_parameters(series, png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_three_relu_architecture(
    *,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_report04_three_relu_architecture(output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_three_relu_architecture(png_path)
        outputs.append(png_path)
    return outputs


def generate_report04_pythia_family_compute_ceiling(
    *,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_report04_pythia_family_compute_ceiling(output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_report04_pythia_family_compute_ceiling(png_path)
        outputs.append(png_path)
    return outputs


def generate_post_layernorm_relu_propagation_heatmaps(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "activation_propagation.json")
    _plot_post_layernorm_relu_propagation_heatmaps(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_post_layernorm_relu_propagation_heatmaps(payload, png_path)
        outputs.append(png_path)
    return outputs


def generate_post_layernorm_relu_zero_product_heatmaps(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "activation_propagation.json")
    _plot_post_layernorm_relu_zero_product_heatmaps(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_post_layernorm_relu_zero_product_heatmaps(payload, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_sweep_summary(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_sweep_summary(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_sweep_summary(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = _select_representative_fixed_step_rows(rows)
    _plot_fixed_step_learning_curves(selected_rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(selected_rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_series = _select_representative_clipping_series(series)
    _plot_fixed_step_clipping_frontiers(selected_series, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(selected_series, png_path)
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_learning_curves(
    *,
    rows: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Fixed-step Learning Curves"
    subtitle = f"n={len(rows)} runs: AdamW plus all {role_label} settings"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_role_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    role_label: str,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{role_label} Post-hoc Clipping Frontiers"
    total_points = sum(len(item.get("rows", [])) for item in series)
    subtitle = (
        f"n={len(series)} runs, {total_points} sweep points: "
        f"AdamW plus all {role_label} settings; validation-loss axis zoomed"
    )
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_selected_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "Selected Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points; validation-loss axis zoomed"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_learning_curves(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    title: str,
    scope_note: str,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = sorted({int(row["tokens_seen"]) for row in rows if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    subtitle = f"n={len(rows)} runs: {scope_note}; {token_note}"
    _plot_fixed_step_learning_curves(
        rows,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=2,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_learning_curves(
            rows,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=2,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_clipping_frontiers(
    *,
    series: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(len(item.get("rows", [])) for item in series)
    title = "High-pressure Post-hoc Clipping Frontiers"
    subtitle = f"n={len(series)} runs, {total_points} sweep points: AdamW plus configs 35-48"
    _plot_fixed_step_clipping_frontiers(
        series,
        output_path,
        title=title,
        subtitle=subtitle,
        use_short_labels=True,
        legend_ncol=3,
    )
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_clipping_frontiers(
            series,
            png_path,
            title=title,
            subtitle=subtitle,
            use_short_labels=True,
            legend_ncol=3,
        )
        outputs.append(png_path)
    return outputs


def generate_fixed_step_high_pressure_weight_norms(
    *,
    rows: list[dict[str, Any]],
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_fixed_step_high_pressure_weight_norms(rows, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_fixed_step_high_pressure_weight_norms(rows, png_path)
        outputs.append(png_path)
    return outputs


def generate_activation_histogram_grid(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "activation_histograms.json")
    _plot_activation_histogram_grid(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_activation_histogram_grid(payload, png_path)
        outputs.append(png_path)
    return outputs


def generate_weight_histogram_grid(
    *,
    run_dir: str | Path,
    output: str | Path,
    save_png: bool = False,
) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)

    run_path = Path(run_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_json(run_path / "weight_histograms.json")
    _plot_weight_histogram_grid(payload, output_path)
    outputs = [output_path]
    if save_png:
        png_path = output_path.with_suffix(".png")
        _plot_weight_histogram_grid(payload, png_path)
        outputs.append(png_path)
    return outputs


def collect_numeric_metrics(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*/*/metrics.json")):
        metrics = read_json(metrics_path)
        manifest_path = metrics_path.parent / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}

        for metric_name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            rows.append(
                {
                    "experiment_name": manifest.get("experiment_name", metrics_path.parent.parent.name),
                    "config_id": manifest.get("config_id", metrics_path.parent.parent.name),
                    "run_id": manifest.get("run_id", metrics_path.parent.name),
                    "metric_name": metric_name,
                    "value": numeric_value,
                }
            )
    return rows


def _latest_labeled_runs(
    results_path: Path,
    experiments: list[tuple[str, str]],
    artifact_name: str,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for label, experiment_id in experiments:
        run = _latest_run_with(results_path / experiment_id, artifact_name)
        if run is not None:
            runs.append((label, run))
    return runs


def _latest_grouped_labeled_runs(
    results_path: Path,
    grouped_experiments: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    artifact_name: str,
) -> list[tuple[str, list[tuple[str, Path]]]]:
    groups: list[tuple[str, list[tuple[str, Path]]]] = []
    for scope, experiments in grouped_experiments:
        groups.append((scope, _latest_labeled_runs(results_path, list(experiments), artifact_name)))
    return groups


def _latest_labeled_runs_filtered(
    results_path: Path,
    experiments: list[tuple[str, str]],
    artifact_name: str,
    *,
    predicate: Any,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for label, experiment_id in experiments:
        experiment_dir = results_path / experiment_id
        if not experiment_dir.exists():
            continue
        candidates = [
            path
            for path in sorted(experiment_dir.iterdir())
            if (path / artifact_name).exists() and predicate(path)
        ]
        if candidates:
            runs.append((label, candidates[-1]))
    return runs


def _latest_relu_site_scope_site_clipping_groups(
    results_path: Path,
) -> list[tuple[str, dict[str, list[tuple[str, Path]]]]]:
    return [
        (scope, _latest_status_update_site_clipping_runs(results_path, list(experiments)))
        for scope, experiments in RELU_SITE_SCOPE_TRAINING_GROUPS
    ]


def _latest_status_update_site_clipping_runs(
    results_path: Path,
    experiments: list[tuple[str, str]],
) -> dict[str, list[tuple[str, Path]]]:
    site_runs: dict[str, list[tuple[str, Path]]] = {}
    for site, _site_label in STATUS_UPDATE_CLIPPING_SITES:
        site_suffix = site.replace("_", "-")
        site_runs[site] = _latest_labeled_runs_filtered(
            results_path,
            [(label, f"{experiment_id}-clipping-sweep-sites-{site_suffix}") for label, experiment_id in experiments],
            "clipping_frontier.jsonl",
            predicate=lambda path, selected_site=site: _is_single_site_clipping_run(path, selected_site),
        )
    return site_runs


def _has_status_update_site_clipping_runs(site_runs: dict[str, list[tuple[str, Path]]]) -> bool:
    return all(len(site_runs.get(site, [])) >= 2 for site, _site_label in STATUS_UPDATE_CLIPPING_SITES)


def _has_relu_site_scope_site_clipping_runs(
    grouped_site_runs: list[tuple[str, dict[str, list[tuple[str, Path]]]]],
) -> bool:
    return bool(grouped_site_runs) and all(
        _has_status_update_site_clipping_runs(site_runs) for _scope, site_runs in grouped_site_runs
    )


def _load_pressure_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        metrics_path = run_path / "metrics.json"
        if not events_path.exists() or not metrics_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if not train_events:
            continue
        series.append(
            {
                "label": label,
                "run_dir": run_path,
                "train_events": train_events,
                "validation_events": validation_events,
                "metrics": read_json(metrics_path),
            }
        )
    return series


def _load_clipping_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        frontier_path = run_path / "clipping_frontier.jsonl"
        if not frontier_path.exists():
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if rows:
            series.append({"label": label, "run_dir": run_path, "rows": rows})
    return series


def _load_event_series(runs: list[tuple[str, str | Path]]) -> list[dict[str, Any]]:
    series = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        events_path = run_path / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append(
                {
                    "label": label,
                    "run_dir": run_path,
                    "train_events": train_events,
                    "validation_events": validation_events,
                }
            )
    return series


def _is_rms_clipping_run(run_dir: str | Path) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    return bool(manifest.get("rms_multipliers"))


def _is_all_site_clipping_run(run_dir: str | Path) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    expected_sites = {"mlp_hiddens", "attention_outputs", "residual_streams"}
    return manifest.get("clipping_sweep_suffix") == "all-sites" and set(manifest.get("clipping_sites", [])) == expected_sites


def _is_single_site_clipping_run(run_dir: str | Path, site: str) -> bool:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    expected_suffix = "sites-" + site.replace("_", "-")
    return manifest.get("clipping_sweep_suffix") == expected_suffix and manifest.get("clipping_sites") == [site]


def _load_fixed_step_sweep_rows(results_path: Path) -> list[dict[str, Any]]:
    latest_by_config: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(results_path.glob("*/*/metrics.json")):
        run_path = metrics_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        sweep = manifest.get("sweep") or {}
        if sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        metrics = read_json(metrics_path)
        if int(metrics.get("calibration/optimizer_steps", -1)) != int(sweep.get("fixed_steps", -1)):
            continue

        config_id = str(manifest.get("config_id", run_path.parent.name))
        pressure = manifest.get("activation_pressure") or {}
        row = {
            "config_index": _config_index(config_id),
            "config_id": config_id,
            "run_id": manifest.get("run_id", run_path.name),
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "role": sweep.get("role", "unknown"),
            "label": _fixed_step_label(sweep.get("role", "unknown"), pressure),
            "short_label": _fixed_step_short_label(sweep.get("role", "unknown"), pressure),
            "pressure": pressure,
            "train_loss": metrics.get("calibration/train_loss_final"),
            "train_loss_mean": metrics.get("calibration/train_loss_mean"),
            "validation_loss": metrics.get("calibration/validation_loss_final"),
            "validation_loss_best": metrics.get("calibration/validation_loss_best"),
            "tokens_seen": metrics.get("calibration/tokens_seen"),
            "tokens_per_second": metrics.get("calibration/tokens_per_second"),
            "wall_seconds": metrics.get("calibration/wall_seconds_total"),
            "peak_gpu_memory_mb": metrics.get("calibration/peak_gpu_memory_mb"),
            "final_model_size_mb": metrics.get("checkpoint/final_size_mb"),
            "mlp_weight_norm_final": metrics.get("calibration/mlp_weight_norm_final"),
            "near_zero_k01": metrics.get("final/activation/near_zero_mass/k1em02"),
            "near_zero_k03": metrics.get("final/activation/near_zero_mass/k3em02"),
            "pressure_loss": metrics.get("final/pressure_loss"),
        }
        previous = latest_by_config.get(config_id)
        if previous is None or row["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[config_id] = row
    return sorted(latest_by_config.values(), key=lambda row: row["config_index"])


def _load_fixed_step_clipping_series(results_path: Path, training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_by_config = {row["config_id"]: row for row in training_rows}
    latest_by_config: dict[str, dict[str, Any]] = {}
    for frontier_path in sorted(results_path.glob("*-clipping-sweep/*/clipping_frontier.jsonl")):
        run_path = frontier_path.parent
        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        source_sweep = manifest.get("source_sweep") or {}
        if source_sweep.get("name") != FIXED_STEP_SWEEP_NAME:
            continue
        source_run = manifest.get("source_run")
        if not source_run:
            continue
        source_config_id = Path(source_run).parent.name
        training_row = training_by_config.get(source_config_id)
        if training_row is None:
            continue
        rows = [
            row
            for row in _read_jsonl(frontier_path)
            if row.get("achieved_sparsity") is not None
            and row.get("validation_loss") is not None
            and math.isfinite(float(row["achieved_sparsity"]))
            and math.isfinite(float(row["validation_loss"]))
        ]
        if not rows:
            continue
        series = {
            "config_id": source_config_id,
            "run_sequence": int(manifest.get("run_sequence", 0)),
            "run_dir": run_path,
            "training": training_row,
            "label": training_row["label"],
            "short_label": training_row["short_label"],
            "role": training_row["role"],
            "rows": rows,
        }
        previous = latest_by_config.get(source_config_id)
        if previous is None or series["run_sequence"] >= previous["run_sequence"]:
            latest_by_config[source_config_id] = series
    return sorted(latest_by_config.values(), key=lambda item: item["training"]["config_index"])


def _plot_run_diagnostics(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 8.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.42, wspace=0.48)

    ax_loss = fig.add_subplot(grid[0, :])
    ax_grad = fig.add_subplot(grid[1, 0])
    ax_weight = fig.add_subplot(grid[1, 1])
    ax_stats = fig.add_subplot(grid[2, :])

    train_tokens = [event["tokens_seen"] for event in train_events]
    train_loss = [event["train_loss"] for event in train_events]
    ax_loss.plot(train_tokens, train_loss, marker="o", markersize=2.5, linewidth=1.3, label="train")
    if validation_events:
        val_tokens = [event["tokens_seen"] for event in validation_events]
        val_loss = [event["validation_loss"] for event in validation_events]
        ax_loss.plot(val_tokens, val_loss, marker="s", markersize=3.0, linewidth=1.3, label="validation")
    ax_loss.set_title("Loss vs Tokens")
    ax_loss.set_xlabel("Tokens seen")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)

    grad_events = [event for event in train_events if event.get("grad_norm") is not None]
    ax_grad.plot(
        [event["tokens_seen"] for event in grad_events],
        [event["grad_norm"] for event in grad_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_grad.set_title("Gradient Norm")
    ax_grad.set_xlabel("Tokens seen")
    ax_grad.set_ylabel("L2 norm")

    weight_events = [event for event in train_events if event.get("weight_norm") is not None]
    ax_weight.plot(
        [event["tokens_seen"] for event in weight_events],
        [event["weight_norm"] for event in weight_events],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    ax_weight.set_title("Weight Norm")
    ax_weight.set_xlabel("Tokens seen")
    ax_weight.set_ylabel("L2 norm")
    ax_weight.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_weight.tick_params(axis="y", labelsize=7)

    tokens_per_step = metrics.get("calibration/tokens_per_step")
    logged_step_rates = [
        tokens_per_step / event["step_wall_seconds"]
        for event in train_events
        if tokens_per_step and event.get("step_wall_seconds")
    ]
    peak_logged_tokens_per_second = max(logged_step_rates) if logged_step_rates else None
    avg_tokens_per_second = metrics.get("calibration/tokens_per_second")
    if peak_logged_tokens_per_second is not None and avg_tokens_per_second is not None:
        peak_tokens_per_second = max(peak_logged_tokens_per_second, avg_tokens_per_second)
    else:
        peak_tokens_per_second = peak_logged_tokens_per_second or avg_tokens_per_second
    stats = [
        ("Peak tokens/s", peak_tokens_per_second, "tok/s"),
        ("Average tokens/s", avg_tokens_per_second, "tok/s"),
        ("Peak GPU allocated", metrics.get("calibration/peak_gpu_memory_mb"), "MB"),
        ("Peak GPU reserved", metrics.get("calibration/peak_gpu_reserved_mb"), "MB"),
        ("Train wall time", metrics.get("calibration/wall_seconds_train"), "sec"),
        ("Total wall time", metrics.get("calibration/wall_seconds_total"), "sec"),
        ("Final model size", metrics.get("checkpoint/final_size_mb"), "MB"),
    ]
    ax_stats.axis("off")
    ax_stats.set_title("Run Statistics")
    _draw_stats_panel(ax_stats, stats)

    fig.suptitle("Pythia-14M MiniPile Run Diagnostics", y=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _final_mlp_weight_norm_from_checkpoint(run_path: Path) -> float | None:
    checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
    if not checkpoint_path.exists():
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0.0
    tensor_count = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".mlp." not in key or not key.endswith(".weight"):
                continue
            tensor = handle.get_tensor(key).float()
            param_norm = tensor.norm(2).item()
            total += param_norm * param_norm
            tensor_count += 1
    if tensor_count == 0:
        return None
    return total**0.5


def _load_status_update_pressure_weight_groups(
    runs: list[tuple[str, str | Path]],
    *,
    group_specs: tuple[tuple[str, str, Any, tuple[float, float]], ...],
) -> list[dict[str, Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Pressure weight diagnostics require torch and safetensors.") from exc

    bins = 180
    series: list[dict[str, Any]] = []
    for label, run_dir in runs:
        run_path = Path(run_dir)
        checkpoint_path = run_path / "checkpoints" / "final" / "model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing final checkpoint for {label}: {checkpoint_path}")

        groups: list[dict[str, Any]] = []
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for group_id, group_label, pattern, (range_min, range_max) in group_specs:
                values = []
                tensor_names = []
                for key in keys:
                    if pattern.match(key) is None:
                        continue
                    values.append(handle.get_tensor(key).detach().float().reshape(-1))
                    tensor_names.append(key)
                if not values:
                    raise ValueError(f"No tensors matched pressure weight group {group_id!r} for {label}.")

                flat = torch.cat(values)
                finite_values = flat[torch.isfinite(flat)]
                counts = torch.histc(finite_values, bins=bins, min=float(range_min), max=float(range_max)).cpu().double()
                total = int(flat.numel())
                width = (float(range_max) - float(range_min)) / bins
                densities = [float(count) / total / width if total and width > 0.0 else 0.0 for count in counts.tolist()]
                centers = [
                    float(range_min) + (index + 0.5) * width
                    for index in range(bins)
                ]
                underflow = int((finite_values < float(range_min)).sum().detach().cpu())
                overflow = int((finite_values > float(range_max)).sum().detach().cpu())
                groups.append(
                    {
                        "id": group_id,
                        "label": group_label,
                        "centers": centers,
                        "densities": densities,
                        "range": (float(range_min), float(range_max)),
                        "total": total,
                        "tensor_count": len(tensor_names),
                        "underflow": underflow,
                        "overflow": overflow,
                    }
                )
        series.append({"label": label, "run_dir": str(run_path), "groups": groups})

    return series


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


def _format_stat_value(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_stats_panel(ax: Any, stats: list[tuple[str, Any, str]]) -> None:
    visible_stats = [(label, value, unit) for label, value, unit in stats if value is not None]
    columns = 3
    rows = (len(visible_stats) + columns - 1) // columns
    for index, (label, value, unit) in enumerate(visible_stats):
        row = index // columns
        column = index % columns
        x = (column + 0.5) / columns
        y = 0.74 - row * (0.58 / max(rows - 1, 1))
        ax.text(x, y + 0.08, label, transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.text(
            x,
            y - 0.07,
            f"{_format_stat_value(float(value))} {unit}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


def _plot_pressure_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(7.4, 8.4))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    token_limit = _short_run_token_limit(series)
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        label = item["label"]
        train_events = _events_up_to(item["train_events"], token_limit)
        validation_events = _events_up_to(item["validation_events"], token_limit)

        ax_train.plot(
            [event["tokens_seen"] for event in train_events],
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.2),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

        if validation_events:
            ax_val.plot(
                [event["tokens_seen"] for event in validation_events],
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.5),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                [event["tokens_seen"] for event in near_zero_events],
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                [event["tokens_seen"] for event in pressure_events],
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen")
    ax_train.set_ylabel("Task loss")

    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen")
    ax_val.set_ylabel("Loss")

    ax_near_zero.set_title("MLP Hidden Near-zero Mass")
    ax_near_zero.set_xlabel("Tokens seen")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    if not ax_near_zero.lines:
        ax_near_zero.text(0.5, 0.5, "No activation-pressure metrics found.", ha="center", va="center")

    ax_pressure.set_title("Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure loss metrics found.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
        )
    fig.suptitle("Pythia-14M Short Pressure Pretraining Checks", y=0.995)
    fig.text(
        0.5,
        0.945,
        f"n={len(series)} runs; AdamW baseline is shown only through the shared short-run token horizon",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_clipping_comparison(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    all_losses: list[float] = []
    total_points = 0
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        loss = [float(row["validation_loss"]) for row in rows]
        total_points += len(rows)
        all_losses.extend(loss)
        ax.plot(
            sparsity,
            loss,
            marker="o",
            markersize=_marker_size(item["label"], 3.5),
            linewidth=_line_width(item["label"]),
            color=colors[item["label"]],
            label=item["label"],
        )

    ax.set_title("Post-hoc MLP Activation Clipping Frontiers")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, fontsize=8)

    if all_losses and min(all_losses) > 0.0:
        loss_span = max(all_losses) - min(all_losses)
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(all_losses) - margin, max(all_losses) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    ax.text(
        0.99,
        0.02,
        f"n={len(series)} runs, {total_points} sweep points; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_gradient_diagnostics(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Full-pass Selected Methods Gradient Diagnostics",
    subtitle: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.4, 7.5))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_task = fig.add_subplot(grid[0, 0])
    ax_pressure = fig.add_subplot(grid[0, 1])
    ax_weighted = fig.add_subplot(grid[1, 0])
    ax_ratio = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    total_events = 0

    for item in series:
        label = item["label"]
        train_events = item["train_events"]
        total_events += len(train_events)
        task_norms = [
            float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        task_tokens = [
            float(event["tokens_seen"]) / 1_000_000.0
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        if task_norms:
            ax_task.plot(
                task_tokens,
                task_norms,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )

        pressure_events = [
            event
            for event in train_events
            if event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure/task_gradient_norm") is not None
        ]
        if not pressure_events:
            continue
        pressure_tokens = [float(event["tokens_seen"]) / 1_000_000.0 for event in pressure_events]
        pressure_norms = [float(event["pressure/pressure_gradient_norm"]) for event in pressure_events]
        weighted_pressure_norms = [
            float(event.get("pressure_weight", 1.0)) * float(event["pressure/pressure_gradient_norm"])
            for event in pressure_events
        ]
        weighted_ratios = [
            (
                float(event.get("pressure_weight", 1.0))
                * float(event["pressure/pressure_gradient_norm"])
                / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            )
            for event in pressure_events
        ]

        ax_pressure.plot(
            pressure_tokens,
            pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_weighted.plot(
            pressure_tokens,
            weighted_pressure_norms,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )
        ax_ratio.plot(
            pressure_tokens,
            weighted_ratios,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_pressure.set_title("Pressure Gradient Norm")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Raw pressure L2 norm")
    ax_pressure.set_ylim(bottom=0.0)
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_weighted.set_title("Weighted Pressure Gradient Norm")
    ax_weighted.set_xlabel("Tokens seen (millions)")
    ax_weighted.set_ylabel("weight * pressure grad norm")
    ax_weighted.set_ylim(bottom=0.0)
    if not ax_weighted.lines:
        ax_weighted.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_task.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.992)
    if subtitle is None:
        subtitle = (
            "n={n} runs; {events} train log events; "
            "pressure panels omit AdamW where no pressure is applied"
        )
    fig.text(
        0.5,
        0.948,
        subtitle.format(n=len(series), events=total_events),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_rms_clipping_frontiers(series: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    total_points = sum(len(item.get("rows", [])) for item in series)
    all_losses: list[float] = []
    colors = _series_colors([item["label"] for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        label = str(item["label"])
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(label, 3.4),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    _zoom_loss_axis(ax, all_losses)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass RMS-normalized Post-hoc Clipping Frontiers", y=0.975)
    fig.text(
        0.5,
        0.935,
        (
            f"n={len(series)} runs, {total_points} sweep points; "
            "threshold = multiplier * RMS(A) per captured tensor/pass; validation-loss axis zoomed"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.11, right=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_pressure_dominance(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_task = fig.add_subplot(grid[0, 1])
    ax_ratio = fig.add_subplot(grid[1, 0])
    ax_phase = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    pressure_method_count = 0

    for item in series:
        label = str(item["label"])
        color = colors[label]
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        task_events = [
            event
            for event in train_events
            if event.get("pressure/task_gradient_norm", event.get("grad_norm")) is not None
        ]
        pressure_events = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None
            and event.get("pressure/task_gradient_norm") is not None
            and event.get("pressure/pressure_gradient_norm") is not None
            and event.get("pressure_weight") is not None
        ]

        if near_zero_events:
            near_zero_tokens = _tokens_millions(near_zero_events)
            near_zero_values = [
                100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events
            ]
            ax_near_zero.plot(
                near_zero_tokens,
                near_zero_values,
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
            peak_index = max(range(len(near_zero_events)), key=lambda index: near_zero_values[index])
            ax_near_zero.scatter(
                [near_zero_tokens[peak_index]],
                [near_zero_values[peak_index]],
                marker="*",
                s=_scatter_size(label, 70.0),
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )

        if task_events:
            ax_task.plot(
                _tokens_millions(task_events),
                [
                    float(event.get("pressure/task_gradient_norm", event.get("grad_norm")))
                    for event in task_events
                ],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        if not pressure_events:
            continue
        pressure_method_count += 1
        ratio_tokens = _tokens_millions(pressure_events)
        ratio_values = [
            100.0
            * float(event["pressure_weight"])
            * float(event["pressure/pressure_gradient_norm"])
            / (float(event["pressure/task_gradient_norm"]) + 1e-12)
            for event in pressure_events
        ]
        near_zero_values = [
            100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in pressure_events
        ]
        ax_ratio.plot(
            ratio_tokens,
            ratio_values,
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        ax_phase.plot(
            ratio_values,
            near_zero_values,
            marker="o",
            markersize=_marker_size(label, 1.7),
            linewidth=_line_width(label),
            color=color,
            alpha=0.92,
            label=label,
        )
        peak_index = max(range(len(near_zero_values)), key=lambda index: near_zero_values[index])
        ax_phase.scatter(
            [ratio_values[peak_index]],
            [near_zero_values[peak_index]],
            marker="*",
            s=_scatter_size(label, 78.0),
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        ax_phase.scatter(
            [ratio_values[-1]],
            [near_zero_values[-1]],
            marker="X",
            s=_scatter_size(label, 48.0),
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_task.set_title("Task Gradient Norm")
    ax_task.set_xlabel("Tokens seen (millions)")
    ax_task.set_ylabel("L2 norm")
    ax_task.set_ylim(bottom=0.0)

    ax_ratio.set_title("Weighted Pressure / Task Gradient")
    ax_ratio.set_xlabel("Tokens seen (millions)")
    ax_ratio.set_ylabel("ratio (%)")
    ax_ratio.set_ylim(bottom=0.0)
    if not ax_ratio.lines:
        ax_ratio.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    ax_phase.set_title("Near-zero Mass vs Pressure Ratio")
    ax_phase.set_xlabel("weighted pressure/task gradient (%)")
    ax_phase.set_ylabel("|activation| <= 0.01 (%)")
    ax_phase.set_ylim(bottom=0.0)
    ax_phase.set_xlim(left=0.0)
    if not ax_phase.lines:
        ax_phase.text(0.5, 0.5, "No pressure-gradient metrics found.", ha="center", va="center")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Full-pass Pressure Dominance Timing Diagnostic", y=0.992)
    fig.text(
        0.5,
        0.948,
        (
            f"n={len(series)} runs; {pressure_method_count} pressure methods; "
            "stars mark peak near-zero mass, X marks final pressure log event"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.89, bottom=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_learning_curves(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    total_events = 0

    for item in series:
        label = str(item["label"])
        color = colors[label]
        train_events = sorted(item["train_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        validation_events = sorted(item["validation_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        total_events += len(train_events)

        ax_train.plot(
            _tokens_millions(train_events),
            [float(event["train_loss"]) for event in train_events],
            marker="o",
            markersize=_marker_size(label, 1.8),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        if validation_events:
            ax_val.plot(
                _tokens_millions(validation_events),
                [float(event["validation_loss"]) for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.2),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                _tokens_millions(pressure_events),
                [float(event["pressure_loss"]) for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen (millions)")
    ax_train.set_ylabel("Task loss")

    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen (millions)")
    ax_val.set_ylabel("Loss")

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_pressure.set_title("Auxiliary Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-loss series found.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.99)
    fig.text(
        0.5,
        0.935,
        (
            f"n={len(series)} completed runs; {total_events} train log events; "
            f"{subtitle}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_weight_norms(series: list[dict[str, Any]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.4, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_global_tokens = fig.add_subplot(grid[0, 1])
    ax_global_near_zero = fig.add_subplot(grid[1, 0])
    ax_mlp_near_zero = fig.add_subplot(grid[1, 1])
    colors = _series_colors([item["label"] for item in series])
    global_fit_x: list[float] = []
    global_fit_y: list[float] = []
    mlp_fit_x: list[float] = []
    mlp_fit_y: list[float] = []

    for item in series:
        label = str(item["label"])
        color = colors[label]
        marker = _method_marker(label)
        train_events = sorted(item["train_events"], key=lambda event: int(event.get("tokens_seen", 0)))
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        weight_events = [event for event in train_events if event.get("weight_norm") is not None]
        paired_global = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("weight_norm") is not None
        ]
        paired_mlp = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("mlp_weight_norm") is not None
        ]

        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if weight_events:
            ax_global_tokens.plot(
                _tokens_millions(weight_events),
                [float(event["weight_norm"]) for event in weight_events],
                marker="o",
                markersize=_marker_size(label, 1.8),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if paired_global:
            final_event = paired_global[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["weight_norm"])
            global_fit_x.append(final_x)
            global_fit_y.append(final_y)
            ax_global_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 38.0),
                color=color,
                alpha=0.92,
                linewidths=0.0,
                label=label,
                zorder=2,
            )
        if paired_mlp:
            final_event = paired_mlp[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["mlp_weight_norm"])
            mlp_fit_x.append(final_x)
            mlp_fit_y.append(final_y)
            ax_mlp_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 38.0),
                color=color,
                alpha=0.92,
                linewidths=0.0,
                label=label,
                zorder=2,
            )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_near_zero.set_ylim(bottom=0.0)

    ax_global_tokens.set_title("Global Weight Norm")
    ax_global_tokens.set_xlabel("Tokens seen (millions)")
    ax_global_tokens.set_ylabel("L2 norm")
    ax_global_tokens.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax_global_near_zero.set_title("Final Global Norm vs Near-zero Mass")
    ax_global_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_global_near_zero.set_ylabel("Global L2 norm")
    ax_global_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_global_near_zero.collections:
        ax_global_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_global_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_global_near_zero, global_fit_x, global_fit_y, loc="lower right")

    ax_mlp_near_zero.set_title("Final MLP Norm vs Near-zero Mass")
    ax_mlp_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_mlp_near_zero.set_ylabel("MLP weight L2 norm")
    ax_mlp_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_mlp_near_zero.collections:
        ax_mlp_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_mlp_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    else:
        ax_mlp_near_zero.text(0.5, 0.5, "No MLP weight norms found.", ha="center", va="center")
    _add_linear_fit_annotation(ax_mlp_near_zero, mlp_fit_x, mlp_fit_y, loc="upper right")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

    tokens = sorted(
        {
            int(event["tokens_seen"])
            for item in series
            for event in item["train_events"]
            if event.get("tokens_seen")
        }
    )
    token_note = f"up to {tokens[-1]:,} tokens/run" if tokens else "one MiniPile token-cache pass per run"
    fig.suptitle("Full-pass High-pressure Weight Norm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(series)} completed runs; {token_note}; lower panels use final train log event",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_full_pass_high_pressure_clipping_frontiers(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    total_points = sum(len(item.get("rows", [])) for item in series)
    all_losses: list[float] = []
    colors = _series_colors([str(item["label"]) for item in series])

    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        label = str(item["label"])
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(label, 3.2),
            linewidth=_line_width(label),
            color=colors[label],
            label=label,
        )

    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    _zoom_loss_axis(ax, all_losses)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.975)
    fig.text(
        0.5,
        0.93,
        (
            f"n={len(series)} runs, {total_points} sweep points; "
            f"{subtitle}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.83, bottom=0.27, left=0.11, right=0.99)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_coupling_density_comparison(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    methods: tuple[tuple[str, str], ...],
    site_specs: tuple[tuple[str, str, str, tuple[float, float]], ...],
    title: str,
    subtitle: str,
) -> None:
    missing = [site for site, _title, _layer, _xlim in site_specs if site not in payloads]
    if missing:
        raise ValueError(f"Missing histogram payloads for report density comparison: {missing}")

    labels = [label for _prefix, label in methods]
    colors = _series_colors(labels)
    site_density: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    site_y_limits: dict[str, tuple[float, float]] = {}

    for site, _site_title, layer_name, _xlim in site_specs:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        per_method: dict[str, tuple[list[float], list[float]]] = {}
        positive_densities: list[float] = []
        max_density = 0.0
        for method_prefix, method_label in methods:
            method = _histogram_method(payload, method_prefix)
            if method is None:
                raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
            layer = _histogram_layer(method, layer_name)
            densities = _histogram_density(layer, edges)
            per_method[method_label] = (centers, densities)
            positive_densities.extend(value for value in densities if value > 0.0)
            max_density = max(max_density, max(densities, default=0.0))
        site_density[site] = per_method
        y_min = max(min(positive_densities) * 0.7, max_density * 1e-4, 1e-6) if positive_densities else 1e-6
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        site_y_limits[site] = (y_min, y_max)

    row_count = len(methods)
    col_count = len(site_specs)
    fig_width = max(10.8, 3.25 * col_count)
    fig_height = max(7.2, 1.65 * row_count + 1.5)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    for row_index, (_method_prefix, method_label) in enumerate(methods):
        for col_index, (site, site_title, _layer_name, xlim) in enumerate(site_specs):
            ax = axes[row_index][col_index]
            centers, densities = site_density[site][method_label]
            color = colors[method_label]
            y_min, y_max = site_y_limits[site]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
            ax.fill_between(centers, y_min, y_values, step="mid", color=color, alpha=0.24, linewidth=0)
            ax.step(centers, y_values, where="mid", color=color, linewidth=_line_width(method_label))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_xlim(*xlim)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            if row_index == 0:
                ax.set_title(site_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(method_label, rotation=0, ha="right", va="center", labelpad=18, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.95,
        f"{subtitle}; layer 3 shown for each activation family; {eval_tokens:,} validation tokens",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Activation value", y=0.052, fontsize=9)
    fig.supylabel("Probability density (log scale)", x=0.008, fontsize=9)
    fig.text(
        0.5,
        0.018,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.91, bottom=0.095, hspace=0.34, wspace=0.14)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_pressure_weight_diagnostic(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
) -> None:
    if not series:
        raise ValueError("No pressure weight diagnostic series were found.")

    labels = [str(item["label"]) for item in series]
    colors = _series_colors(labels)
    group_labels = [group["label"] for group in series[0]["groups"]]
    group_y_limits: dict[str, tuple[float, float]] = {}
    for group_label in group_labels:
        positive_values = [
            float(value)
            for item in series
            for group in item["groups"]
            if group["label"] == group_label
            for value in group["densities"]
            if value > 0.0
        ]
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        group_y_limits[group_label] = (y_min, y_max)

    row_count = len(series)
    col_count = len(group_labels)
    fig_width = max(10.8, 3.25 * col_count)
    fig_height = max(7.0, 1.6 * row_count + 1.5)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    for row_index, item in enumerate(series):
        method_label = str(item["label"])
        groups_by_label = {group["label"]: group for group in item["groups"]}
        for col_index, group_label in enumerate(group_labels):
            ax = axes[row_index][col_index]
            group = groups_by_label[group_label]
            centers = [float(value) for value in group["centers"]]
            densities = [float(value) for value in group["densities"]]
            y_min, y_max = group_y_limits[group_label]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
            color = colors[method_label]
            ax.fill_between(centers, y_min, y_values, step="mid", color=color, alpha=0.24, linewidth=0)
            ax.step(centers, y_values, where="mid", color=color, linewidth=_line_width(method_label))
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(*group["range"])
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            if row_index == 0:
                ax.set_title(group_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(method_label, rotation=0, ha="right", va="center", labelpad=18, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.947,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Weight value", y=0.052, fontsize=9)
    fig.supylabel("Probability density (log scale)", x=0.008, fontsize=9)
    fig.text(
        0.5,
        0.018,
        footnote,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.905, bottom=0.095, hspace=0.32, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_status_update_site_clipping_frontiers(
    site_series: dict[str, list[dict[str, Any]]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=False)
    labels = [
        str(item["label"])
        for site, _site_label in STATUS_UPDATE_CLIPPING_SITES
        for item in site_series.get(site, [])
    ]
    if not labels:
        raise ValueError("No site-specific clipping series were found for the status update figure.")
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    total_points = 0

    for ax, (site, site_label) in zip(axes, STATUS_UPDATE_CLIPPING_SITES, strict=True):
        series = site_series.get(site, [])
        all_losses: list[float] = []
        for item in series:
            rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
            sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
            losses = [float(row["validation_loss"]) for row in rows]
            all_losses.extend(losses)
            total_points += len(rows)
            label = str(item["label"])
            (line,) = ax.plot(
                sparsity,
                losses,
                marker=_method_marker(label),
                markersize=_marker_size(label, 3.0),
                linewidth=_line_width(label),
                color=colors[label],
                label=label,
            )
            legend_handles.setdefault(label, line)

        ax.set_title(site_label, fontsize=10)
        ax.set_xlabel("Exact zeros after clipping (%)")
        ax.grid(True, alpha=0.25)
        if all_losses:
            _zoom_loss_axis(ax, all_losses)
        ax.tick_params(axis="both", labelsize=8)
    axes[0].set_ylabel("Validation loss")

    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.925,
        f"{subtitle}; {total_points} sweep points across the plotted panels; validation-loss axes zoomed",
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.81, bottom=0.28, wspace=0.26)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_learning_curves(
    grouped_series: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    row_count = len(grouped_series)
    fig, axes = plt.subplots(row_count, 2, figsize=(11.0, max(6.8, 2.1 * row_count + 1.4)), sharex="col")
    if row_count == 1:
        axes = [axes]

    labels = [
        str(item["label"])
        for _scope, series in grouped_series
        for item in series
    ]
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    token_max = 0

    for row_index, (scope, series) in enumerate(grouped_series):
        ax_loss = axes[row_index][0]
        ax_near_zero = axes[row_index][1]
        loss_values: list[float] = []

        for item in series:
            label = str(item["label"])
            color = colors[label]
            validation_events = sorted(
                item["validation_events"],
                key=lambda event: int(event.get("tokens_seen", 0)),
            )
            near_zero_events = sorted(
                [
                    event
                    for event in item["train_events"]
                    if event.get("activation/near_zero_mass/k1em02") is not None
                ],
                key=lambda event: int(event.get("tokens_seen", 0)),
            )
            if validation_events:
                token_max = max(token_max, max(int(event["tokens_seen"]) for event in validation_events))
                loss_values.extend(float(event["validation_loss"]) for event in validation_events)
                (line,) = ax_loss.plot(
                    _tokens_millions(validation_events),
                    [float(event["validation_loss"]) for event in validation_events],
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.8),
                    linewidth=_line_width(label),
                    color=color,
                    label=label,
                )
                legend_handles.setdefault(label, line)
            if near_zero_events:
                token_max = max(token_max, max(int(event["tokens_seen"]) for event in near_zero_events))
                ax_near_zero.plot(
                    _tokens_millions(near_zero_events),
                    [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.4),
                    linewidth=_line_width(label),
                    color=color,
                    label=label,
                )

        if row_index == 0:
            ax_loss.set_title("Validation loss", fontsize=10)
            ax_near_zero.set_title("Logged near-zero mass", fontsize=10)
        ax_loss.set_ylabel(f"{scope}\nloss", fontsize=8)
        ax_near_zero.set_ylabel("|a| <= 0.01 (%)", fontsize=8)
        ax_near_zero.set_ylim(0.0, 100.0)
        if loss_values:
            _zoom_loss_axis(ax_loss, loss_values)
        if row_index == row_count - 1:
            ax_loss.set_xlabel("Tokens seen (millions)")
            ax_near_zero.set_xlabel("Tokens seen (millions)")
        ax_loss.tick_params(axis="both", labelsize=8)
        ax_near_zero.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Learning Curves", y=0.985)
    token_note = f"up to {token_max:,} tokens/run" if token_max else "one MiniPile token-cache pass per run"
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; AdamW baseline repeated in each row; "
            f"{token_note}; near-zero mass is the run's logged pressure-site aggregate"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.88, bottom=0.16, hspace=0.3, wspace=0.24)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_clipping_frontiers(
    grouped_site_series: list[tuple[str, dict[str, list[dict[str, Any]]]]],
    output_path: Path,
) -> None:
    row_count = len(grouped_site_series)
    col_count = len(STATUS_UPDATE_CLIPPING_SITES)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(11.4, max(7.1, 2.15 * row_count + 1.35)),
        sharex=True,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    labels = [
        str(item["label"])
        for _scope, site_series in grouped_site_series
        for site, _site_label in STATUS_UPDATE_CLIPPING_SITES
        for item in site_series.get(site, [])
    ]
    if not labels:
        raise ValueError("No ReLU site-scope clipping series were found.")
    colors = _series_colors(labels)
    legend_handles: dict[str, Any] = {}
    total_points = 0

    for row_index, (scope, site_series) in enumerate(grouped_site_series):
        for col_index, (site, site_label) in enumerate(STATUS_UPDATE_CLIPPING_SITES):
            ax = axes[row_index][col_index]
            all_losses: list[float] = []
            for item in site_series.get(site, []):
                rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
                sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
                losses = [float(row["validation_loss"]) for row in rows]
                all_losses.extend(losses)
                total_points += len(rows)
                label = str(item["label"])
                (line,) = ax.plot(
                    sparsity,
                    losses,
                    marker=_method_marker(label),
                    markersize=_marker_size(label, 2.6),
                    linewidth=_line_width(label),
                    color=colors[label],
                    label=label,
                )
                legend_handles.setdefault(label, line)
            if row_index == 0:
                ax.set_title(site_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\nvalidation loss", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Exact zeros after clipping (%)")
            if all_losses:
                _zoom_loss_axis(ax, all_losses)
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Post-hoc Clipping Frontiers", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns clip one activation family at a time; "
            f"{total_points} sweep points; validation-loss axes zoomed"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_activation_density_comparison(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    missing = [site for site, _title, _layer, _xlim in RELU_SITE_SCOPE_DENSITY_SITE_SPECS if site not in payloads]
    if missing:
        raise ValueError(f"Missing ReLU site-scope histogram payloads: {missing}")

    method_labels = [label for _scope, methods in RELU_SITE_SCOPE_HISTOGRAM_GROUPS for label, _prefix in methods]
    colors = _series_colors(method_labels)
    site_density: dict[tuple[str, str], dict[str, tuple[list[float], list[float]]]] = {}
    site_y_limits: dict[str, tuple[float, float]] = {}

    for site, _site_title, layer_name, _xlim in RELU_SITE_SCOPE_DENSITY_SITE_SPECS:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
        positive_densities: list[float] = []
        max_density = 0.0
        for scope, methods in RELU_SITE_SCOPE_HISTOGRAM_GROUPS:
            per_method: dict[str, tuple[list[float], list[float]]] = {}
            for method_label, method_prefix in methods:
                method = _histogram_method(payload, method_prefix)
                if method is None:
                    raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
                layer = _histogram_layer(method, layer_name)
                densities = _histogram_density(layer, edges)
                per_method[method_label] = (centers, densities)
                positive_densities.extend(value for value in densities if value > 0.0)
                max_density = max(max_density, max(densities, default=0.0))
            site_density[(scope, site)] = per_method
        y_min = max(min(positive_densities) * 0.7, max_density * 1e-4, 1e-6) if positive_densities else 1e-6
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        site_y_limits[site] = (y_min, y_max)

    row_count = len(RELU_SITE_SCOPE_HISTOGRAM_GROUPS)
    col_count = len(RELU_SITE_SCOPE_DENSITY_SITE_SPECS)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(12.4, max(7.4, 2.1 * row_count + 1.35)),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    legend_handles: dict[str, Any] = {}
    for row_index, (scope, methods) in enumerate(RELU_SITE_SCOPE_HISTOGRAM_GROUPS):
        for col_index, (site, site_title, _layer_name, xlim) in enumerate(RELU_SITE_SCOPE_DENSITY_SITE_SPECS):
            ax = axes[row_index][col_index]
            y_min, y_max = site_y_limits[site]
            for method_label, _method_prefix in methods:
                centers, densities = site_density[(scope, site)][method_label]
                y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
                (line,) = ax.step(
                    centers,
                    y_values,
                    where="mid",
                    color=colors[method_label],
                    linewidth=_line_width(method_label),
                    label=method_label,
                )
                legend_handles.setdefault(method_label, line)
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(*xlim)
            if row_index == 0:
                ax.set_title(site_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\ndensity", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Activation value")
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.suptitle("ReLU Site-scope Activation Density Comparison", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns show layer 3 for each activation family; "
            f"{eval_tokens:,} validation tokens; probability density is log-scaled"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.055,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_pressure_weight_diagnostic(
    grouped_series: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    group_labels = [group["label"] for group in grouped_series[0][1][0]["groups"]]
    labels = [str(item["label"]) for _scope, series in grouped_series for item in series]
    colors = _series_colors(labels)
    group_y_limits: dict[str, tuple[float, float]] = {}
    for group_label in group_labels:
        positive_values = [
            float(value)
            for _scope, series in grouped_series
            for item in series
            for group in item["groups"]
            if group["label"] == group_label
            for value in group["densities"]
            if value > 0.0
        ]
        max_density = max(positive_values, default=1.0)
        y_min = max(min(positive_values) * 0.7, max_density * 1e-5, 1e-8) if positive_values else 1e-8
        y_max = max_density * 1.7 if max_density > 0.0 else 1.0
        group_y_limits[group_label] = (y_min, y_max)

    row_count = len(grouped_series)
    col_count = len(group_labels)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(11.4, max(7.1, 2.1 * row_count + 1.35)),
        sharex=False,
        sharey=False,
    )
    if row_count == 1:
        axes = [axes]

    legend_handles: dict[str, Any] = {}
    for row_index, (scope, series) in enumerate(grouped_series):
        for col_index, group_label in enumerate(group_labels):
            ax = axes[row_index][col_index]
            y_min, y_max = group_y_limits[group_label]
            for item in series:
                method_label = str(item["label"])
                groups_by_label = {group["label"]: group for group in item["groups"]}
                group = groups_by_label[group_label]
                centers = [float(value) for value in group["centers"]]
                densities = [float(value) for value in group["densities"]]
                y_values = [max(value, y_min) if value > 0.0 else y_min for value in densities]
                (line,) = ax.step(
                    centers,
                    y_values,
                    where="mid",
                    color=colors[method_label],
                    linewidth=_line_width(method_label),
                    label=method_label,
                )
                legend_handles.setdefault(method_label, line)
                ax.set_xlim(*group["range"])
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.08, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.7, alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            if row_index == 0:
                ax.set_title(group_label, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\ndensity", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_index == row_count - 1:
                ax.set_xlabel("Weight value")
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("ReLU Site-scope Pressure Weight Diagnostic", y=0.985)
    fig.text(
        0.5,
        0.945,
        (
            "Rows compare pressure scope; columns aggregate final-checkpoint weights across all six layers; "
            "probability density is log-scaled"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.055,
        (
            "Shaded band marks |weight| <= 0.01. "
            "Residual-stream column uses MLP and attention output-projection weights."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.17, hspace=0.28, wspace=0.12)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_relu_site_scope_near_zero_heatmaps(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    missing = [site for site, _site_label in RELU_SITE_SCOPE_NEAR_ZERO_SITES if site not in payloads]
    if missing:
        raise ValueError(f"Missing ReLU near-zero histogram payloads: {missing}")

    row_labels = [label for label, _prefix in RELU_SITE_SCOPE_REPORT_METHODS]
    matrices: dict[str, list[list[float]]] = {}
    for site, _site_label in RELU_SITE_SCOPE_NEAR_ZERO_SITES:
        payload = payloads[site]
        edges = [float(value) for value in payload.get("bin_edges", [])]
        if len(edges) < 2:
            raise ValueError(f"Histogram payload for {site} has no bin edges.")
        site_matrix: list[list[float]] = []
        for _method_label, method_prefix in RELU_SITE_SCOPE_REPORT_METHODS:
            method = _histogram_method(payload, method_prefix)
            if method is None:
                raise ValueError(f"Missing method {method_prefix!r} in histogram payload for {site}.")
            site_matrix.append(
                [
                    100.0
                    * _histogram_center_window_mass(
                        _histogram_layer(method, f"{site}.layer_{layer_index}"),
                        edges,
                        threshold=0.01,
                    )
                    for layer_index in range(6)
                ]
            )
        matrices[site] = site_matrix

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.2), sharex=True)
    image = None
    for ax, (site, site_label) in zip(axes, RELU_SITE_SCOPE_NEAR_ZERO_SITES, strict=True):
        matrix = matrices[site]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
        ax.set_title(site_label, fontsize=10)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=6)
        ax.set_xticks(range(6), [f"L{index}" for index in range(6)], fontsize=8)
        for boundary in (0.5, 4.5, 8.5):
            ax.axhline(boundary, color="white", linewidth=0.9, alpha=0.8)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if value >= 55.0 else "black",
                )
    axes[-1].set_xlabel("Transformer layer")

    fig.suptitle("ReLU Site-scope Near-zero Mass by Layer Type", y=0.985)
    eval_tokens = int(next(iter(payloads.values())).get("validation_tokens") or 0)
    fig.text(
        0.5,
        0.952,
        (
            "Entries are histogram-estimated |activation| <= 0.01 percentages by layer; "
            f"{eval_tokens:,} validation tokens"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.18, right=0.86, top=0.91, bottom=0.08, hspace=0.33)
    if image is not None:
        colorbar_axis = fig.add_axes([0.885, 0.18, 0.018, 0.64])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Near-zero mass (%)", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path)
    plt.close(fig)


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


def _plot_fixed_step_sweep_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    plottable = [
        row
        for row in rows
        if _finite(row.get("validation_loss"))
        and _finite(row.get("near_zero_k01"))
        and _finite(row.get("near_zero_k03"))
    ]
    if not plottable:
        raise ValueError("No fixed-step sweep rows with validation loss and activation metrics were found.")

    fig = plt.figure(figsize=(7.4, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
    ax_k01 = fig.add_subplot(grid[0, 0])
    ax_k03 = fig.add_subplot(grid[1, 0])

    roles = _ordered_roles(plottable)
    colors = _series_colors([FIXED_STEP_ROLE_LABELS.get(role, role) for role in roles])
    baseline = next((row for row in plottable if row["role"] == "adamw"), None)

    for role in roles:
        role_rows = [row for row in plottable if row["role"] == role]
        label = FIXED_STEP_ROLE_LABELS.get(role, role)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(role, "o")
        ax_k01.scatter(
            [100.0 * float(row["near_zero_k01"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )
        ax_k03.scatter(
            [100.0 * float(row["near_zero_k03"]) for row in role_rows],
            [float(row["validation_loss"]) for row in role_rows],
            marker=marker,
            s=_scatter_size(label, 42),
            color=color,
            label=label,
            alpha=0.9,
        )

    for row in _fixed_step_annotation_rows(plottable):
        ax_k01.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k01"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
        ax_k03.annotate(
            row["short_label"],
            (100.0 * float(row["near_zero_k03"]), float(row["validation_loss"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    for ax, threshold in [(ax_k01, 0.01), (ax_k03, 0.03)]:
        ax.set_xlabel(f"MLP hidden mass with |activation| <= {threshold:g} (%)")
        ax.set_ylabel("Final validation loss")
        if baseline is not None:
            ax.axhline(
                float(baseline["validation_loss"]),
                color=ADAMW_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
            )
        _zoom_loss_axis(ax, [float(row["validation_loss"]) for row in plottable])

    ax_k01.set_title("Fixed-step Tradeoff at Threshold 0.01")
    ax_k03.set_title("Fixed-step Tradeoff at Threshold 0.03")
    handles, labels = ax_k01.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    tokens = sorted({int(row["tokens_seen"]) for row in plottable if row.get("tokens_seen")})
    token_note = f"{tokens[0]:,} tokens/run" if len(tokens) == 1 else "fixed token budget"
    fig.suptitle("Pythia-14M MiniPile Fixed-step Pressure Screen", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(plottable)} runs; {token_note}; validation-loss axes zoomed",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.17)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_learning_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Representative Fixed-step Learning Curves",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int = 2,
) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        if train_events:
            series.append({**row, "train_events": train_events, "validation_events": validation_events})
    if not series:
        raise ValueError("No fixed-step sweep learning curves were found.")

    fig = plt.figure(figsize=(7.6, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_val = fig.add_subplot(grid[0, 1])
    ax_near_zero = fig.add_subplot(grid[1, 0])
    ax_pressure = fig.add_subplot(grid[1, 1])
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)

    for item in series:
        label = _plot_label(item, use_short_labels=use_short_labels)
        color = colors[label]
        train_events = item["train_events"]
        validation_events = item["validation_events"]
        ax_train.plot(
            _tokens_millions(train_events),
            [event["train_loss"] for event in train_events],
            marker="o",
            markersize=_marker_size(label, 2.0),
            linewidth=_line_width(label),
            color=color,
            label=label,
        )
        if validation_events:
            ax_val.plot(
                _tokens_millions(validation_events),
                [event["validation_loss"] for event in validation_events],
                marker="s",
                markersize=_marker_size(label, 2.4),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        pressure_events = [event for event in train_events if event.get("pressure_loss") is not None]
        if pressure_events:
            ax_pressure.plot(
                _tokens_millions(pressure_events),
                [event["pressure_loss"] for event in pressure_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )

    ax_train.set_title("Train Loss")
    ax_train.set_xlabel("Tokens seen (millions)")
    ax_train.set_ylabel("Task loss")
    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Tokens seen (millions)")
    ax_val.set_ylabel("Loss")
    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")
    ax_pressure.set_title("Auxiliary Pressure Loss")
    ax_pressure.set_xlabel("Tokens seen (millions)")
    ax_pressure.set_ylabel("Unweighted pressure loss")
    if not ax_pressure.lines:
        ax_pressure.text(0.5, 0.5, "No pressure-loss series selected.", ha="center", va="center")

    handles, labels = ax_train.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_ncol,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=0.99)
    subtitle = subtitle or f"n={len(series)} selected runs: AdamW plus best validation-loss run per pressure family"
    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fixed_step_high_pressure_weight_norms(rows: list[dict[str, Any]], output_path: Path) -> None:
    series = []
    for row in rows:
        events_path = Path(row["run_dir"]) / "events.jsonl"
        if not events_path.exists():
            continue
        events = _read_jsonl(events_path)
        train_events = [event for event in events if event.get("event") == "train"]
        if not train_events:
            continue
        final_mlp_weight_norm = row.get("mlp_weight_norm_final")
        if final_mlp_weight_norm is None:
            final_mlp_weight_norm = _final_mlp_weight_norm_from_checkpoint(Path(row["run_dir"]))
        series.append({**row, "train_events": train_events, "final_mlp_weight_norm": final_mlp_weight_norm})
    if not series:
        raise ValueError("No high-pressure OR/L1 weight-norm series were found.")

    fig = plt.figure(figsize=(8.2, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_near_zero = fig.add_subplot(grid[0, 0])
    ax_global_tokens = fig.add_subplot(grid[0, 1])
    ax_global_near_zero = fig.add_subplot(grid[1, 0])
    ax_mlp_near_zero = fig.add_subplot(grid[1, 1])

    labels_for_colors = [_plot_label(item, use_short_labels=True) for item in series]
    colors = _series_colors(labels_for_colors)
    mlp_time_series_count = 0
    mlp_final_count = 0
    global_fit_x: list[float] = []
    global_fit_y: list[float] = []
    mlp_fit_x: list[float] = []
    mlp_fit_y: list[float] = []

    for item in series:
        label = _plot_label(item, use_short_labels=True)
        color = colors[label]
        marker = FIXED_STEP_ROLE_MARKERS.get(str(item.get("role")), "o")
        train_events = item["train_events"]
        near_zero_events = [
            event for event in train_events if event.get("activation/near_zero_mass/k1em02") is not None
        ]
        weight_events = [event for event in train_events if event.get("weight_norm") is not None]
        paired_global = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("weight_norm") is not None
        ]
        paired_mlp = [
            event
            for event in train_events
            if event.get("activation/near_zero_mass/k1em02") is not None and event.get("mlp_weight_norm") is not None
        ]

        if near_zero_events:
            ax_near_zero.plot(
                _tokens_millions(near_zero_events),
                [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in near_zero_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if weight_events:
            ax_global_tokens.plot(
                _tokens_millions(weight_events),
                [float(event["weight_norm"]) for event in weight_events],
                marker="o",
                markersize=_marker_size(label, 2.0),
                linewidth=_line_width(label),
                color=color,
                label=label,
            )
        if paired_global:
            final_event = paired_global[-1]
            final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
            final_y = float(final_event["weight_norm"])
            global_fit_x.append(final_x)
            global_fit_y.append(final_y)
            ax_global_near_zero.scatter(
                [final_x],
                [final_y],
                marker=marker,
                s=_scatter_size(label, 28.0),
                color=color,
                alpha=0.9,
                linewidths=0.0,
                zorder=2,
            )
        if paired_mlp:
            mlp_time_series_count += 1
            xs = [100.0 * float(event["activation/near_zero_mass/k1em02"]) for event in paired_mlp]
            ys = [float(event["mlp_weight_norm"]) for event in paired_mlp]
            mlp_fit_x.extend(xs)
            mlp_fit_y.extend(ys)
            ax_mlp_near_zero.scatter(
                xs,
                ys,
                marker=marker,
                s=_scatter_size(label, 12.0),
                color=color,
                alpha=0.72,
                linewidths=0.0,
                zorder=2,
            )
        elif item.get("final_mlp_weight_norm") is not None:
            final_event = paired_global[-1] if paired_global else None
            if final_event is not None:
                final_x = 100.0 * float(final_event["activation/near_zero_mass/k1em02"])
                final_y = float(item["final_mlp_weight_norm"])
                mlp_fit_x.append(final_x)
                mlp_fit_y.append(final_y)
                mlp_final_count += 1
                ax_mlp_near_zero.scatter(
                    [final_x],
                    [final_y],
                    marker=marker,
                    s=_scatter_size(label, 28.0),
                    color=color,
                    alpha=0.9,
                    linewidths=0.0,
                    zorder=2,
                )

    ax_near_zero.set_title("Near-zero Activation Mass")
    ax_near_zero.set_xlabel("Tokens seen (millions)")
    ax_near_zero.set_ylabel("|activation| <= 0.01 (%)")

    ax_global_tokens.set_title("Global Weight Norm")
    ax_global_tokens.set_xlabel("Tokens seen (millions)")
    ax_global_tokens.set_ylabel("L2 norm")
    ax_global_tokens.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax_global_near_zero.set_title("Final Global Norm vs Near-zero Mass")
    ax_global_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_global_near_zero.set_ylabel("Global L2 norm")
    ax_global_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if ax_global_near_zero.collections:
        ax_global_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.",
            transform=ax_global_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_global_near_zero, global_fit_x, global_fit_y, loc="lower right")

    mlp_title_prefix = "MLP Weight Norm"
    if mlp_time_series_count == 0 and mlp_final_count > 0:
        mlp_title_prefix = "Final MLP Weight Norm"
    if mlp_title_prefix == "Final MLP Weight Norm":
        ax_mlp_near_zero.set_title("Final MLP Norm vs Near-zero Mass")
    else:
        ax_mlp_near_zero.set_title(f"{mlp_title_prefix} vs Near-zero Mass")
    ax_mlp_near_zero.set_xlabel("|activation| <= 0.01 (%)")
    ax_mlp_near_zero.set_ylabel("MLP weight L2 norm")
    ax_mlp_near_zero.ticklabel_format(axis="y", style="plain", useOffset=False)
    if not ax_mlp_near_zero.lines and not ax_mlp_near_zero.collections:
        ax_mlp_near_zero.text(0.5, 0.5, "No MLP weight norms found.", ha="center", va="center")
    elif mlp_time_series_count == 0 and mlp_final_count > 0:
        ax_mlp_near_zero.text(
            0.02,
            0.02,
            "Final checkpoint only.\nFuture runs log MLP norm per step.",
            transform=ax_mlp_near_zero.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    _add_linear_fit_annotation(ax_mlp_near_zero, mlp_fit_x, mlp_fit_y, loc="upper right")

    handles, labels = ax_near_zero.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            fontsize=7,
        )

    tokens = sorted(
        {
            int(event["tokens_seen"])
            for item in series
            for event in item["train_events"]
            if event.get("tokens_seen")
        }
    )
    token_note = f"up to {tokens[-1]:,} tokens/run" if tokens else "fixed token budget"
    fig.suptitle("High-pressure OR and L1 Weight Norm Diagnostics", y=0.99)
    fig.text(
        0.5,
        0.935,
        f"n={len(series)} runs: AdamW, OR configs 40-44, and L1N/OL1 configs 45-48; {token_note}",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_activation_histogram_grid(payload: dict[str, Any], output_path: Path) -> None:
    methods = payload.get("methods", [])
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if not methods or len(edges) < 2:
        raise ValueError("Activation histogram payload has no plottable methods or bin edges.")

    layer_names = [layer["name"] for layer in methods[0].get("layers", [])]
    if not layer_names:
        raise ValueError("Activation histogram payload has no layer histograms.")
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    labels = [str(method["label"]) for method in methods]
    colors = _series_colors(labels)

    positive_values: list[float] = []
    probability_by_method_layer: list[list[list[float]]] = []
    overflow_notes: list[str] = []
    for method in methods:
        method_layers: list[list[float]] = []
        method_overflow = 0.0
        method_total = 0.0
        layers_by_name = {layer["name"]: layer for layer in method.get("layers", [])}
        for layer_name in layer_names:
            layer = layers_by_name[layer_name]
            total = float(layer.get("total") or sum(layer.get("counts", [])) or 1.0)
            counts = [float(value) for value in layer["counts"]]
            probabilities = [count / total for count in counts]
            positive_values.extend(value for value in probabilities if value > 0.0)
            method_layers.append(probabilities)
            method_overflow += float(layer.get("underflow", 0)) + float(layer.get("overflow", 0))
            method_total += total
        probability_by_method_layer.append(method_layers)
        overflow_fraction = method_overflow / method_total if method_total else 0.0
        if overflow_fraction > 0.001:
            overflow_notes.append(f"{method['label']}: {100.0 * overflow_fraction:.2f}% outside range")

    rows = len(methods)
    cols = len(layer_names)
    fig_width = max(12.0, 2.05 * cols)
    if rows <= 3:
        fig_height = max(9.0, 2.05 * rows + 2.6)
        title_y = 0.99
        subtitle_y = 0.958
        top_margin = 0.895
        bottom_margin = 0.13
        xlabel_y = 0.055
        footnote_y = 0.018
        hspace = 0.22
    else:
        fig_height = max(14.0, 1.34 * rows + 2.6)
        title_y = 0.997
        subtitle_y = 0.982
        top_margin = 0.955
        bottom_margin = 0.065
        xlabel_y = 0.035
        footnote_y = 0.02
        hspace = 0.18
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    y_min = max(min(positive_values) * 0.6, 1e-9) if positive_values else 1e-9
    y_max = max(positive_values) * 2.0 if positive_values else 1.0
    site_scope = _activation_histogram_scope(payload, layer_names)
    x_min = min(edges)
    x_max = _activation_histogram_x_max(site_scope, edges)
    for row_index, method in enumerate(methods):
        label = str(method["label"])
        color = colors[label]
        for col_index, layer_name in enumerate(layer_names):
            ax = axes[row_index][col_index]
            probabilities = probability_by_method_layer[row_index][col_index]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in probabilities]
            ax.fill_between(
                centers,
                y_min,
                y_values,
                step="mid",
                color=color,
                alpha=0.26,
                linewidth=0,
            )
            ax.step(centers, y_values, where="mid", color=color, linewidth=0.75, alpha=0.95)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.07, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.45, alpha=0.5)
            if row_index == 0:
                ax.set_title(_activation_histogram_layer_title(layer_name), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{label}\nprobability", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    eval_tokens = int(payload.get("validation_tokens") or 0)
    eval_sequences = int(payload.get("validation_sequences") or 0)
    overflow_note = ""
    if overflow_notes:
        overflow_note = "; outside plotted range: " + ", ".join(overflow_notes[:3])
        if len(overflow_notes) > 3:
            overflow_note += ", ..."
    x_note = ""
    if x_max < max(edges):
        x_note = f"; x-axis shown to {x_max:g}"
    title = str(payload.get("plot_title") or _activation_histogram_default_title(site_scope))
    fig.suptitle(title, y=title_y, fontsize=14)
    fig.text(
        0.5,
        subtitle_y,
        (
            f"n={len(methods)} checkpoints; {eval_sequences:,} validation blocks; "
            f"{eval_tokens:,} validation tokens; y-axis is log probability per bin"
            f"{x_note}{overflow_note}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Activation value", y=xlabel_y, fontsize=10)
    fig.supylabel("Probability per bin (log scale)", x=0.006, fontsize=10)
    fig.text(
        0.5,
        footnote_y,
        "Shaded band marks |activation| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=top_margin,
        bottom=bottom_margin,
        hspace=hspace,
        wspace=0.08,
    )
    fig.savefig(output_path)
    plt.close(fig)


def _activation_histogram_scope(payload: dict[str, Any], layer_names: list[str]) -> str:
    sites = payload.get("sites") or []
    if sites:
        return str(sites[0])
    first_layer = layer_names[0] if layer_names else ""
    if first_layer.startswith("residual_streams."):
        return "residual_streams"
    if first_layer.startswith("attention_outputs."):
        return "attention_outputs"
    return "mlp_hiddens"


def _activation_histogram_x_max(site_scope: str, edges: list[float]) -> float:
    if site_scope == "mlp_hiddens":
        return min(max(edges), 1.0)
    return max(edges)


def _activation_histogram_layer_title(layer_name: str) -> str:
    return (
        layer_name.replace("mlp_hiddens.layer_", "MLP ")
        .replace("residual_streams.layer_", "H ")
        .replace("attention_outputs.layer_", "Attn out ")
    )


def _activation_histogram_default_title(site_scope: str) -> str:
    if site_scope == "residual_streams":
        return "Selected Method Residual Stream Distributions"
    if site_scope == "attention_outputs":
        return "Selected Method Attention Output Distributions"
    return "Selected Method MLP Activation Distributions"


def _histogram_method(payload: dict[str, Any], label_prefix: str) -> dict[str, Any] | None:
    for method in payload.get("methods", []):
        if str(method.get("label", "")).startswith(label_prefix):
            return method
    return None


def _histogram_layer(method: dict[str, Any], layer_name: str) -> dict[str, Any]:
    for layer in method.get("layers", []):
        if layer.get("name") == layer_name:
            return layer
    raise ValueError(f"Missing histogram layer {layer_name!r} for {method.get('label')!r}.")


def _histogram_density(layer: dict[str, Any], edges: list[float]) -> list[float]:
    counts = [float(value) for value in layer.get("counts", [])]
    total = float(layer.get("total") or sum(counts) or 1.0)
    widths = [right - left for left, right in zip(edges[:-1], edges[1:], strict=True)]
    return [count / total / width if width > 0.0 else 0.0 for count, width in zip(counts, widths, strict=True)]


def _histogram_center_window_mass(layer: dict[str, Any], edges: list[float], *, threshold: float) -> float:
    counts = [float(value) for value in layer.get("counts", [])]
    total = float(layer.get("total") or sum(counts) or 1.0)
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    selected_counts = [count for count, center in zip(counts, centers, strict=True) if abs(center) <= threshold]
    if selected_counts:
        window_count = sum(selected_counts)
    else:
        window_count = sum(
            count * max(0.0, min(right, threshold) - max(left, -threshold)) / (right - left)
            for count, left, right in zip(counts, edges[:-1], edges[1:], strict=True)
            if right > left
        )
    return window_count / total if total > 0.0 else 0.0


def _plot_weight_histogram_grid(payload: dict[str, Any], output_path: Path) -> None:
    methods = payload.get("methods", [])
    edges = [float(value) for value in payload.get("bin_edges", [])]
    if not methods or len(edges) < 2:
        raise ValueError("Weight histogram payload has no plottable methods or bin edges.")

    layer_names = [layer["name"] for layer in methods[0].get("layers", [])]
    if not layer_names:
        raise ValueError("Weight histogram payload has no layer histograms.")
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]
    labels = [str(method["label"]) for method in methods]
    colors = _series_colors(labels)

    positive_values: list[float] = []
    probability_by_method_layer: list[list[list[float]]] = []
    overflow_notes: list[str] = []
    for method in methods:
        method_layers: list[list[float]] = []
        method_overflow = 0.0
        method_total = 0.0
        layers_by_name = {layer["name"]: layer for layer in method.get("layers", [])}
        for layer_name in layer_names:
            layer = layers_by_name[layer_name]
            total = float(layer.get("total") or sum(layer.get("counts", [])) or 1.0)
            counts = [float(value) for value in layer["counts"]]
            probabilities = [count / total for count in counts]
            positive_values.extend(value for value in probabilities if value > 0.0)
            method_layers.append(probabilities)
            method_overflow += float(layer.get("underflow", 0)) + float(layer.get("overflow", 0))
            method_total += total
        probability_by_method_layer.append(method_layers)
        overflow_fraction = method_overflow / method_total if method_total else 0.0
        if overflow_fraction > 0.001:
            overflow_notes.append(f"{method['label']}: {100.0 * overflow_fraction:.2f}% outside range")

    rows = len(methods)
    cols = len(layer_names)
    fig_width = max(12.0, 2.05 * cols)
    fig_height = max(14.0, 1.34 * rows + 2.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    y_min = max(min(positive_values) * 0.6, 1e-9) if positive_values else 1e-9
    y_max = max(positive_values) * 2.0 if positive_values else 1.0
    x_min = min(edges)
    x_max = max(edges)
    for row_index, method in enumerate(methods):
        label = str(method["label"])
        color = colors[label]
        for col_index, layer_name in enumerate(layer_names):
            ax = axes[row_index][col_index]
            probabilities = probability_by_method_layer[row_index][col_index]
            y_values = [max(value, y_min) if value > 0.0 else y_min for value in probabilities]
            ax.fill_between(
                centers,
                y_min,
                y_values,
                step="mid",
                color=color,
                alpha=0.26,
                linewidth=0,
            )
            ax.step(centers, y_values, where="mid", color=color, linewidth=0.75, alpha=0.95)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_tick))
            ax.axvspan(-0.01, 0.01, color="#000000", alpha=0.07, linewidth=0)
            ax.axvline(0.0, color="#4d4d4d", linewidth=0.45, alpha=0.5)
            if row_index == 0:
                ax.set_title(_weight_histogram_layer_title(layer_name), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{label}\nprobability", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    overflow_note = ""
    if overflow_notes:
        overflow_note = "; outside plotted range: " + ", ".join(overflow_notes[:3])
        if len(overflow_notes) > 3:
            overflow_note += ", ..."
    scope = str(payload.get("scope") or "mlp_weights")
    if scope == "attention_weights":
        scope_note = "attention QKV and output dense weights only"
        default_title = "Selected Method Attention Weight Distributions"
    else:
        scope_note = "MLP dense weights only"
        default_title = "Selected Method MLP Weight Distributions"
    title = str(payload.get("plot_title") or default_title)
    fig.suptitle(title, y=0.997, fontsize=14)
    fig.text(
        0.5,
        0.982,
        (
            f"n={len(methods)} checkpoints; {scope_note}; "
            f"biases excluded; y-axis is log probability per bin{overflow_note}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.supxlabel("Weight value", y=0.035, fontsize=10)
    fig.supylabel("Probability per bin (log scale)", x=0.006, fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Shaded band marks |weight| <= 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.955, bottom=0.065, hspace=0.18, wspace=0.08)
    fig.savefig(output_path)
    plt.close(fig)


def _weight_histogram_layer_title(layer_name: str) -> str:
    return layer_name.replace("mlp_weights.layer_", "MLP ").replace("attention_weights.layer_", "Attn ")


def _trimmed_decimal_tick(value: float, _position: int) -> str:
    label = f"{value:.2f}".rstrip("0")
    if label.endswith("."):
        label += "0"
    return label


def _plot_fixed_step_clipping_frontiers(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Post-hoc Clipping Frontiers After Fixed-step Pretraining",
    subtitle: str | None = None,
    use_short_labels: bool = False,
    legend_ncol: int | None = None,
) -> None:
    if not series:
        raise ValueError("No fixed-step clipping frontier series were found.")

    figsize = (7.0, 5.4) if legend_ncol is not None else (7.0, 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    labels_for_colors = [_plot_label(item, use_short_labels=use_short_labels) for item in series]
    colors = _series_colors(labels_for_colors)
    all_losses: list[float] = []
    total_points = 0
    for item in series:
        rows = sorted(item["rows"], key=lambda row: float(row["achieved_sparsity"]))
        sparsity = [100.0 * float(row["achieved_sparsity"]) for row in rows]
        losses = [float(row["validation_loss"]) for row in rows]
        all_losses.extend(losses)
        total_points += len(rows)
        ax.plot(
            sparsity,
            losses,
            marker="o",
            markersize=_marker_size(_plot_label(item, use_short_labels=use_short_labels), 3.2),
            linewidth=_line_width(_plot_label(item, use_short_labels=use_short_labels)),
            color=colors[_plot_label(item, use_short_labels=use_short_labels)],
            label=_plot_label(item, use_short_labels=use_short_labels),
        )

    ax.set_title(title)
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")
    if legend_ncol is None:
        ax.legend(frameon=False, fontsize=8)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=legend_ncol,
                frameon=False,
                fontsize=7,
            )
    _zoom_loss_axis(ax, all_losses)
    ax.text(
        0.99,
        0.02,
        subtitle or f"n={len(series)} selected runs, {total_points} sweep points; validation-loss axis zoomed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    if legend_ncol is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(bottom=0.34)
    fig.savefig(output_path)
    plt.close(fig)


def _short_run_token_limit(series: list[dict[str, Any]]) -> int | None:
    pressure_maxima = [
        max((int(event["tokens_seen"]) for event in item["train_events"]), default=0)
        for item in series
        if item["label"] != "AdamW baseline"
    ]
    pressure_maxima = [value for value in pressure_maxima if value > 0]
    if pressure_maxima:
        return max(pressure_maxima)
    all_maxima = [max((int(event["tokens_seen"]) for event in item["train_events"]), default=0) for item in series]
    all_maxima = [value for value in all_maxima if value > 0]
    return max(all_maxima) if all_maxima else None


def _events_up_to(events: list[dict[str, Any]], token_limit: int | None) -> list[dict[str, Any]]:
    if token_limit is None:
        return events
    return [event for event in events if int(event.get("tokens_seen", 0)) <= token_limit]


def _series_colors(labels: list[str]) -> dict[str, str]:
    unique_labels = list(dict.fromkeys(labels))
    colors: dict[str, str] = {}
    color_index = 0
    for label in unique_labels:
        if _is_adamw_label(label):
            colors[label] = ADAMW_COLOR
            continue
        colors[label] = COLORBLIND_SAFE_COLORS[color_index % len(COLORBLIND_SAFE_COLORS)]
        color_index += 1
    return colors


def _is_adamw_label(label: str) -> bool:
    return label.lower().startswith("adamw")


def _line_width(label: str) -> float:
    return ADAMW_LINEWIDTH if _is_adamw_label(label) else DEFAULT_SERIES_LINEWIDTH


def _marker_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _scatter_size(label: str, default: float) -> float:
    return default * ADAMW_MARKER_SCALE if _is_adamw_label(label) else default


def _method_marker(label: str) -> str:
    normalized = label.lower()
    if normalized.startswith("adamw"):
        return "D"
    if normalized.startswith("rn"):
        return "o"
    if normalized.startswith("or"):
        return "s"
    if normalized.startswith("l1n"):
        return "^"
    if normalized.startswith("ol1"):
        return "P"
    return "o"


def _ordered_roles(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["adamw", "ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"]
    present = {str(row["role"]) for row in rows}
    ordered = [role for role in preferred if role in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _fixed_step_label(role: str, pressure: dict[str, Any]) -> str:
    role_label = FIXED_STEP_ROLE_LABELS.get(role, role)
    if role == "adamw":
        return role_label
    weight = pressure.get("weight")
    if "ricker" in role:
        return (
            f"{role_label} "
            f"w={_compact_number(weight)}, c={_compact_number(pressure.get('ricker_c'))}, "
            f"s={_compact_number(pressure.get('ricker_sigma'))}"
        )
    if "l1" in role:
        return f"{role_label} w={_compact_number(weight)}"
    return role_label


def _fixed_step_short_label(role: str, pressure: dict[str, Any]) -> str:
    if role == "adamw":
        return "AdamW"
    prefix = {
        "ricker_naive": "RN",
        "orthogonal_ricker": "OR",
        "l1_naive": "L1N",
        "orthogonal_l1": "OL1",
    }.get(role, role)
    weight = _compact_number(pressure.get("weight"))
    if "ricker" in role:
        return f"{prefix} w{weight} c{_compact_number(pressure.get('ricker_c'))} s{_compact_number(pressure.get('ricker_sigma'))}"
    return f"{prefix} w{weight}"


def _compact_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):g}"


def _config_index(config_id: str) -> int:
    first = config_id.split("-", 1)[0]
    try:
        return int(first)
    except ValueError:
        return 9999


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


def _fixed_step_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected[baseline["config_id"]] = baseline
    for role in _ordered_roles(rows):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role == "adamw" or not role_rows:
            continue
        sparse = max(role_rows, key=lambda row: float(row.get("near_zero_k03") or 0.0))
        selected[sparse["config_id"]] = sparse
    return sorted(selected.values(), key=lambda row: row["config_index"])


def _select_representative_fixed_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    for role in ("ricker_naive", "orthogonal_ricker", "l1_naive", "orthogonal_l1"):
        role_rows = [row for row in rows if row["role"] == role and _finite(row.get("validation_loss"))]
        if role_rows:
            selected.append(min(role_rows, key=lambda row: float(row["validation_loss"])))
    return selected


def _select_fixed_step_rows_for_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((row for row in rows if row["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((row for row in rows if row["role"] == role), key=lambda row: row["config_index"]))
    return selected


def _select_fixed_step_rows_by_config_indices(
    rows: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(row["config_index"]): row for row in rows}
    return [by_index[index] for index in config_indices if index in by_index]


def _select_representative_clipping_series(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_rows = [item["training"] for item in series]
    selected_configs = {row["config_id"] for row in _select_representative_fixed_step_rows(training_rows)}
    return [item for item in series if item["config_id"] in selected_configs]


def _select_fixed_step_clipping_for_role(series: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next((item for item in series if item["role"] == "adamw"), None)
    if baseline is not None:
        selected.append(baseline)
    selected.extend(sorted((item for item in series if item["role"] == role), key=lambda item: item["training"]["config_index"]))
    return selected


def _select_fixed_step_clipping_by_config_indices(
    series: list[dict[str, Any]],
    config_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_index = {int(item["training"]["config_index"]): item for item in series}
    return [by_index[index] for index in config_indices if index in by_index]


def _tokens_millions(events: list[dict[str, Any]]) -> list[float]:
    return [float(event["tokens_seen"]) / 1_000_000.0 for event in events]


def _plot_label(item: dict[str, Any], *, use_short_labels: bool) -> str:
    return str(item.get("short_label") if use_short_labels else item.get("label"))


def _add_linear_fit_annotation(ax: Any, x_values: list[float], y_values: list[float], *, loc: str) -> None:
    points = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(points) < 2:
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    if x_var <= 0.0:
        return
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / x_var
    intercept = y_mean - slope * x_mean
    residual_sum = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total_sum = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    x_line = [min(xs), max(xs)]
    y_line = [slope * x + intercept for x in x_line]
    ax.plot(
        x_line,
        y_line,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="_nolegend_",
        zorder=1,
    )

    positions = {
        "upper right": (0.98, 0.97, "right", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
    }
    x_pos, y_pos, ha, va = positions.get(loc, positions["upper right"])
    ax.text(
        x_pos,
        y_pos,
        f"slope={slope:.3g}/pp\nR2={r2:.2f}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def _zoom_loss_axis(ax: Any, values: list[float]) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return
    low = min(finite_values)
    high = max(finite_values)
    span = high - low
    margin = max(span * 0.15, 1e-4)
    ax.set_ylim(low - margin, high + margin)


def _plot_clipping_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    finite_rows = [
        row
        for row in rows
        if row.get("achieved_sparsity") is not None
        and row.get("validation_loss") is not None
        and math.isfinite(float(row["achieved_sparsity"]))
        and math.isfinite(float(row["validation_loss"]))
    ]
    if not finite_rows:
        raise ValueError("No finite clipping frontier points to plot.")

    finite_rows = sorted(finite_rows, key=lambda row: float(row["achieved_sparsity"]))
    sparsity = [100.0 * float(row["achieved_sparsity"]) for row in finite_rows]
    loss = [float(row["validation_loss"]) for row in finite_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(sparsity, loss, marker="o", linewidth=1.5, markersize=4.5)
    ax.set_title("Post-hoc Activation Clipping Frontier")
    ax.set_xlabel("Achieved exact-zero activation sparsity (%)")
    ax.set_ylabel("Validation loss")

    for index, (point_x, point_y, row) in enumerate(zip(sparsity, loss, finite_rows, strict=True)):
        label = _clipping_label(row)
        ax.annotate(
            label,
            (point_x, point_y),
            textcoords="offset points",
            xytext=_clipping_label_offset(index, len(finite_rows)),
            fontsize=7,
        )

    loss_span = max(loss) - min(loss)
    if min(loss) > 0.0:
        margin = max(loss_span * 0.15, 1e-4)
        ax.set_ylim(min(loss) - margin, max(loss) + margin)
        axis_note = "validation-loss axis zoomed"
    else:
        axis_note = "validation-loss axis starts at zero"

    eval_tokens = sorted({int(row["validation_tokens"]) for row in finite_rows if row.get("validation_tokens")})
    token_note = f"; {eval_tokens[0]:,} validation tokens/point" if len(eval_tokens) == 1 else ""
    ax.text(
        0.99,
        0.02,
        f"n={len(finite_rows)} sweep points{token_note}; {axis_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _clipping_label(row: dict[str, Any]) -> str:
    if row.get("threshold") is not None:
        return f"t={float(row['threshold']):g}"
    if row.get("quantile") is not None:
        return f"q={float(row['quantile']):g}"
    if row.get("rms_multiplier") is not None:
        return f"r={float(row['rms_multiplier']):g}"
    return str(row.get("mode", "clip"))


def _clipping_label_offset(index: int, total: int) -> tuple[int, int]:
    if index == 0:
        return (6, -16)
    if index == total - 1:
        return (-36, 6)
    return (6, 8 + 5 * (index % 2))


def _plot_metric_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    metric_name = sorted({row["metric_name"] for row in rows})[0]
    selected = [row for row in rows if row["metric_name"] == metric_name]
    labels = [f"{row['config_id']}\n{row['run_id'][:8]}" for row in selected]
    values = [row["value"] for row in selected]

    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_title("Result Summary")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Run")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if values and min(values) >= 0:
        ax.set_ylim(bottom=0)
    ax.text(0.99, 0.98, f"n={len(selected)} runs", transform=ax.transAxes, ha="right", va="top")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_empty_placeholder(output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "No numeric results found.\nRun make smoke or make baseline first.",
        ha="center",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
