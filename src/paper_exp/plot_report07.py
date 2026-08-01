"""Reproducible figures and endpoint tables for the S1 ablation report.

The S1 report is registry-driven.  Only canonical, valid scientific runs enter
the plots or tables; engineering and pooled-diagnostic rows remain provenance
evidence and are not treated as scientific cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from paper_exp.plot_api import (
    DOUBLE_COLUMN_WIDTH_INCHES,
    PublicationProfile,
    export_figure,
)
from paper_exp.plot_catalog import REPORT07_FIGURES
from paper_exp.plot_style import REPORT04_PLOT_STYLE


EXPECTED_S1_COUNTS = {
    "S1-B0": 20,
    "S1-B1": 36,
    "S1-B2": 26,
    "S1-B3": 40,
    "S1-B4": 10,
}
EXPECTED_SELECTION_HASH = (
    "ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47"
)
B0_LR_ARCHITECTURES = ("A0", "A1-H", "A3", "A6-PRE", "A6-POST")
B0_LR_VALUES = (1e-5, 3e-5, 1e-4)
B0_TOPOLOGY_ORDER = (
    "A0",
    "A1-H",
    "A3",
    "A4-Q",
    "A4-K",
    "A4-V",
    "A5-QK-PRE",
    "A5-QK-POST",
    "A6-PRE",
    "A6-POST",
)
B0_MATCHED_EFFECT_FILENAME = (
    "111-pythia-14m-s1-b0-matched-architecture-effects.pdf"
)
B0_MATCHED_CONTRAST_SPECS = (
    ("MLP-hidden ReLU", "A1-H", "A0", "relu_path"),
    ("+ branch ReLUs", "A3", "A1-H", "relu_path"),
    ("+ Q (POST)", "A4-Q", "A3", "attention_sites"),
    ("+ K (POST)", "A4-K", "A3", "attention_sites"),
    ("+ V", "A4-V", "A3", "attention_sites"),
    ("+ Q/K (PRE)", "A5-QK-PRE", "A3", "attention_sites"),
    ("+ Q/K (POST)", "A5-QK-POST", "A3", "attention_sites"),
    ("+ V (PRE Q/K)", "A6-PRE", "A5-QK-PRE", "add_v"),
    ("+ V (POST Q/K)", "A6-POST", "A5-QK-POST", "add_v"),
    ("Q/K PRE -> POST", "A6-POST", "A6-PRE", "placement"),
)
B1_FACTORIAL_ARCHITECTURES = (
    ("A5-QK-PRE", "pre_rope", "qk_only", "A5-QK-PRE"),
    ("A5-QK-POST", "post_rope", "qk_only", "A5-QK-POST"),
    ("A6-PRE-QKV", "pre_rope", "qkv", "A6-PRE"),
    ("A6-POST-QKV", "post_rope", "qkv", "A6-POST"),
)
B1_ARCHITECTURE_COLORS = {
    "A5-QK-PRE": "#0072B2",
    "A5-QK-POST": "#E69F00",
    "A6-PRE-QKV": "#6F4C9B",
    "A6-POST-QKV": "#009E73",
}
B1_GATE_STYLES = {
    "gplus": {"linestyle": "-", "marker": "o", "label": r"$G^+$"},
    "gpm": {"linestyle": "--", "marker": "s", "label": r"$G^\pm$"},
}
B1_TABLE_GROUPS = (
    ("A1-H", "gplus", ((0.00, 124), (0.03, 204), (0.10, 205))),
    ("A3", "gplus", ((0.00, 125), (0.03, 206), (0.10, 207))),
    ("A4-Q", "gplus", ((0.00, 129), (0.10, 197))),
    ("A4-Q", "gpm", ((0.00, 125), (0.10, 200))),
    ("A4-K", "gplus", ((0.00, 130), (0.10, 198))),
    ("A4-K", "gpm", ((0.00, 125), (0.10, 201))),
    ("A4-V", "gplus", ((0.00, 131), (0.10, 199))),
    ("A4-V", "gpm", ((0.00, 125), (0.10, 202))),
    (
        "A5-QK-PRE",
        "gplus",
        ((0.00, 132), (0.03, 149), (0.10, 151), (0.30, 153)),
    ),
    (
        "A5-QK-PRE",
        "gpm",
        ((0.00, 125), (0.03, 173), (0.10, 175), (0.30, 177)),
    ),
    (
        "A5-QK-POST",
        "gplus",
        ((0.00, 133), (0.03, 161), (0.10, 163), (0.30, 165)),
    ),
    (
        "A5-QK-POST",
        "gpm",
        ((0.00, 125), (0.03, 185), (0.10, 187), (0.30, 189)),
    ),
    (
        "A6-PRE-QKV",
        "gplus",
        ((0.00, 126), (0.03, 155), (0.10, 157), (0.30, 159)),
    ),
    (
        "A6-PRE-QKV",
        "gpm",
        ((0.00, 125), (0.03, 179), (0.10, 181), (0.30, 183)),
    ),
    (
        "A6-POST-QKV",
        "gplus",
        ((0.00, 127), (0.03, 167), (0.10, 169), (0.30, 171)),
    ),
    (
        "A6-POST-QKV",
        "gpm",
        ((0.00, 125), (0.03, 191), (0.10, 193), (0.30, 195)),
    ),
    ("A6-POST-ALL", "gplus", ((0.00, 127), (0.03, 208), (0.10, 209))),
)
B2_ARCHITECTURE_ORDER = (
    "A1-H",
    "A3",
    "A5-QK-PRE",
    "A5-QK-POST",
    "A6-PRE-QKV",
    "A6-POST-QKV",
    "A6-POST-ALL",
)
B2_ARCHITECTURE_COLORS = {
    "A1-H": "#0072B2",
    "A3": "#009E73",
    "A5-QK-PRE": "#56B4E9",
    "A5-QK-POST": "#E69F00",
    "A6-PRE-QKV": "#6F4C9B",
    "A6-POST-QKV": "#CC79A7",
    "A6-POST-ALL": "#D55E00",
}
REPORT07_PROFILE = PublicationProfile(
    width_inches=DOUBLE_COLUMN_WIDTH_INCHES,
    max_height_inches=8.8,
    min_text_points=8.0,
)

ARCHITECTURE_COLORS = {
    "A0": "#666666",
    "A1-H": "#0072B2",
    "A3": "#009E73",
    "A4-Q": "#E69F00",
    "A4-K": "#D55E00",
    "A4-V": "#CC79A7",
    "A5-QK-PRE": "#56B4E9",
    "A5-QK-POST": "#B79F00",
    "A6-PRE": "#6F4C9B",
    "A6-POST": "#A23B72",
}
ARCHITECTURE_MARKERS = {
    "A0": "o",
    "A1-H": "s",
    "A3": "^",
    "A6-PRE": "D",
    "A6-POST": "P",
}
GATE_COLORS = {"gplus": "#D55E00", "gpm": "#0072B2"}
GATE_MARKERS = {"gplus": "o", "gpm": "s"}
PLACEMENT_COLORS = {"pre_rope": "#6F4C9B", "post_rope": "#E69F00"}
PLACEMENT_MARKERS = {"pre_rope": "o", "post_rope": "s"}
PRESSURE_COLORS = {
    "l1_naive": "#D55E00",
    "orthogonal_l1": "#0072B2",
    "ricker_naive": "#D55E00",
    "orthogonal_ricker": "#0072B2",
}
PRESSURE_MARKERS = {
    "l1_naive": "o",
    "orthogonal_l1": "s",
    "ricker_naive": "o",
    "orthogonal_ricker": "s",
}
METHOD_LABELS = {
    "none": "AdamW",
    "l1_naive": "L1N",
    "orthogonal_l1": "OL1",
    "ricker_naive": "RN",
    "orthogonal_ricker": "OR",
}


@dataclass(frozen=True)
class S1Row:
    """One canonical scientific S1 config joined to its canonical run."""

    config: Mapping[str, Any]
    run: Mapping[str, Any]

    @property
    def number(self) -> int:
        return int(str(self.config["config_id"]).split("-", 1)[0])

    @property
    def block(self) -> str:
        return str(self.config["block"])

    @property
    def architecture(self) -> str:
        return str(self.config["architecture_id"])

    @property
    def loss(self) -> float:
        return float(self.run["validation_loss"])

    @property
    def r_block_pct(self) -> float:
        return 100.0 * float(self.run["r_block"])

    @property
    def r_model_pct(self) -> float:
        return 100.0 * float(self.run["r_model"])

    @property
    def u_arch_pct(self) -> float | None:
        value = self.run.get("u_arch")
        return None if value is None else 100.0 * float(value)


@dataclass(frozen=True)
class B0MatchedEffect:
    """One registered central-LR architecture contrast."""

    label: str
    child: str
    parent: str
    group: str
    delta_loss: float
    delta_r_model_pct: float


@dataclass(frozen=True)
class B1Endpoint:
    """One displayed B1 architecture/gate/threshold endpoint."""

    architecture: str
    gate_family: str
    kappa: float
    row: S1Row


@dataclass(frozen=True)
class B2Endpoint:
    """One displayed B2 architecture/gate/scale/parameter endpoint."""

    architecture: str
    gate_family: str
    threshold_scale: str
    parameterization: str
    row: S1Row


@dataclass(frozen=True)
class B3WeightEndpoint:
    """One B3 pressure-weight setting with its matched methods and control."""

    architecture: str
    pressure: str
    weight: float
    adamw: S1Row
    naive: S1Row
    orthogonal: S1Row


@dataclass(frozen=True)
class B3GeometryEndpoint:
    """One B3 Ricker-geometry setting with its matched methods and control."""

    architecture: str
    sweep: str
    value: float
    adamw: S1Row
    naive: S1Row
    orthogonal: S1Row


def load_s1_rows(
    config_registry: str | Path,
    run_registry: str | Path,
) -> tuple[S1Row, ...]:
    """Load and validate the complete canonical S1 scientific cohort."""

    config_path = Path(config_registry)
    run_path = Path(run_registry)
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_payload = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    configs = config_payload["records"]
    runs_by_id = {str(record["run_id"]): record for record in run_payload["records"]}
    repo_root = config_path.resolve().parents[2]
    propagation_cache: dict[Path, tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]] = {}

    rows: list[S1Row] = []
    for config in configs:
        if config.get("phase") != "S1" or config.get("kind") != "scientific":
            continue
        block = str(config.get("block"))
        if block not in EXPECTED_S1_COUNTS:
            continue
        run_id = config.get("canonical_run_id")
        if not run_id or str(run_id) not in runs_by_id:
            raise ValueError(f"Missing canonical run for {config['config_id']}.")
        run = runs_by_id[str(run_id)]
        issues = []
        if config.get("config_status") != "closed":
            issues.append(f"config_status={config.get('config_status')!r}")
        if not run.get("canonical"):
            issues.append("run is not canonical")
        if run.get("evidence_status") != "valid":
            issues.append(f"evidence_status={run.get('evidence_status')!r}")
        if run.get("lifecycle_status") != "completed":
            issues.append(f"lifecycle_status={run.get('lifecycle_status')!r}")
        for field in ("validation_loss", "r_block", "r_model"):
            if run.get(field) is None or not math.isfinite(float(run[field])):
                issues.append(f"{field} is missing or nonfinite")
        if int(run.get("validation_tokens") or 0) != 311_296:
            issues.append(f"validation_tokens={run.get('validation_tokens')!r}")
        for record_name, record in (("config", config), ("run", run)):
            if record.get("validation_partition") != "selection":
                issues.append(
                    f"{record_name}.validation_partition="
                    f"{record.get('validation_partition')!r}"
                )
            if record.get("validation_partition_hash") != EXPECTED_SELECTION_HASH:
                issues.append(
                    f"{record_name}.validation_partition_hash="
                    f"{record.get('validation_partition_hash')!r}"
                )

        propagation_value = run.get("propagation_result_path")
        if not propagation_value:
            issues.append("propagation_result_path is missing")
        else:
            propagation_path = Path(str(propagation_value))
            if not propagation_path.is_absolute():
                propagation_path = repo_root / propagation_path
            if not propagation_path.is_file():
                issues.append(f"propagation artifact is missing: {propagation_path}")
            else:
                if propagation_path not in propagation_cache:
                    payload = json.loads(propagation_path.read_text(encoding="utf-8"))
                    methods = {
                        str(method["run_id"]): method
                        for method in payload.get("methods", ())
                    }
                    propagation_cache[propagation_path] = (payload, methods)
                payload, methods = propagation_cache[propagation_path]
                if payload.get("validation_partition_hash") != EXPECTED_SELECTION_HASH:
                    issues.append("propagation artifact has the wrong selection hash")
                if int(payload.get("validation_tokens") or 0) != 311_296:
                    issues.append("propagation artifact has the wrong token count")
                method = methods.get(str(run["run_id"]))
                if method is None:
                    issues.append("canonical run is absent from its propagation artifact")
                else:
                    endpoint = method.get("endpoint") or {}
                    raw_values = {
                        "r_block": endpoint.get("R_block"),
                        "r_model": endpoint.get("R_model"),
                        "r_block_max": endpoint.get("R_block_max"),
                        "r_model_max": endpoint.get("R_model_max"),
                        "u_arch": endpoint.get("U_arch"),
                    }
                    zero_sites = endpoint.get("zero_sites") or {}
                    for field in (
                        "z_a",
                        "z_m",
                        "z_h",
                        "z_q_gate",
                        "z_k_gate",
                        "z_v_gate",
                        "z_q_qk",
                        "z_k_qk",
                        "z_v_pv",
                        "z_context_wo",
                    ):
                        raw_values[field] = (zero_sites.get(field) or {}).get(
                            "exact_zero_fraction"
                        )
                    for field, raw_value in raw_values.items():
                        registry_value = run.get(field)
                        if raw_value is None and registry_value is None:
                            continue
                        if (
                            raw_value is None
                            or registry_value is None
                            or not math.isclose(
                                float(raw_value),
                                float(registry_value),
                                rel_tol=0.0,
                                abs_tol=1e-14,
                            )
                        ):
                            issues.append(
                                f"{field} disagrees with the propagation artifact"
                            )
        if issues:
            detail = "; ".join(issues)
            raise ValueError(f"Invalid scientific row {config['config_id']}: {detail}.")
        rows.append(S1Row(config=config, run=run))

    rows.sort(key=lambda item: item.number)
    counts = {
        block: sum(row.block == block for row in rows)
        for block in EXPECTED_S1_COUNTS
    }
    if counts != EXPECTED_S1_COUNTS:
        raise ValueError(f"Unexpected S1 scientific census: {counts!r}.")
    if len({row.number for row in rows}) != len(rows):
        raise ValueError("S1 scientific config prefixes are not unique.")
    return tuple(rows)


def _rows(
    cohort: Sequence[S1Row],
    *,
    block: str | None = None,
    comparison: str | None = None,
) -> list[S1Row]:
    selected = list(cohort)
    if block is not None:
        selected = [row for row in selected if row.block == block]
    if comparison is not None:
        selected = [
            row
            for row in selected
            if row.config.get("comparison_id") == comparison
        ]
    return selected


def _config_index(cohort: Sequence[S1Row]) -> dict[str, S1Row]:
    return {str(row.config["design_id"]): row for row in cohort}


def _number_index(cohort: Sequence[S1Row]) -> dict[int, S1Row]:
    return {row.number: row for row in cohort}


def _panel_title(axis: Any, letter: str, text: str) -> None:
    axis.set_title(f"({letter}) {text}", loc="left", fontweight="bold")


def _finish_axis(axis: Any) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _add_identity(axis: Any, values: Iterable[float]) -> None:
    finite = np.asarray([float(value) for value in values], dtype=float)
    lower = float(finite.min())
    upper = float(finite.max())
    span = max(upper - lower, 1e-6)
    lower -= 0.08 * span
    upper += 0.08 * span
    axis.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


def _method_architecture_legend(
    *,
    method_names: tuple[str, str],
    architecture_names: tuple[str, str] = ("A3", "A6-POST"),
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=PRESSURE_COLORS[method_id],
            marker=PRESSURE_MARKERS[method_id],
            label=label,
        )
        for method_id, label in method_names
    ] + [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=linestyle,
            label=architecture,
        )
        for architecture, linestyle in zip(architecture_names, ("-", "--"), strict=True)
    ]


def build_architecture_learning_rate_figure(cohort: Sequence[S1Row]) -> Figure:
    """B0 learning-rate and ordinary-ReLU topology ablations."""

    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 2.72))
    b0 = _rows(cohort, block="S1-B0")

    for architecture in B0_LR_ARCHITECTURES:
        items = sorted(
            [row for row in b0 if row.architecture == architecture],
            key=lambda row: float(row.config["model_learning_rate"]),
        )
        x = [float(row.config["model_learning_rate"]) for row in items]
        color = ARCHITECTURE_COLORS[architecture]
        marker = ARCHITECTURE_MARKERS[architecture]
        axes[0].plot(
            x,
            [row.loss for row in items],
            marker=marker,
            color=color,
            label=architecture,
        )
        axes[1].plot(
            x,
            [row.r_model_pct for row in items],
            marker=marker,
            color=color,
        )

    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_xticks(
            [1e-5, 3e-5, 1e-4],
            [r"$10^{-5}$", r"$3{\times}10^{-5}$", r"$10^{-4}$"],
        )
        axis.set_xlabel("Model learning rate")
        _finish_axis(axis)
    axes[0].set_ylabel("Validation loss")
    axes[1].set_ylabel(r"$R_{\mathrm{model}}$ (%)")
    _panel_title(axes[0], "a", "Short-budget optimization")
    _panel_title(axes[1], "b", "Compute opportunity")

    central = _rows(cohort, comparison="S1-B0-ARCH")
    topology_order = (
        "A0",
        "A1-H",
        "A3",
        "A4-Q",
        "A4-K",
        "A4-V",
        "A5-QK-PRE",
        "A5-QK-POST",
        "A6-PRE",
        "A6-POST",
    )
    central_by_arch = {row.architecture: row for row in central}
    ordered = [central_by_arch[architecture] for architecture in topology_order]
    y = np.arange(len(ordered))
    axes[2].barh(
        y,
        [row.r_model_pct for row in ordered],
        color=[ARCHITECTURE_COLORS.get(row.architecture, "#555555") for row in ordered],
        height=0.62,
    )
    for position, row in zip(y, ordered, strict=True):
        axes[2].text(
            row.r_model_pct + 0.22,
            position,
            f"{row.loss:.3f}",
            va="center",
            fontsize=8,
        )
    axes[2].set_yticks(y, topology_order)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 16.6)
    axes[2].set_xlabel(r"$R_{\mathrm{model}}$ (%)  [bar-end: loss]")
    _panel_title(axes[2], "c", "Center-LR topology")
    _finish_axis(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.76, wspace=0.38)
    return figure


def _b0_lr_triplets(
    cohort: Sequence[S1Row],
) -> tuple[tuple[str, tuple[S1Row, ...]], ...]:
    """Return the five registered B0 architecture LR triplets."""

    b0 = _rows(cohort, block="S1-B0")
    triplets: list[tuple[str, tuple[S1Row, ...]]] = []
    for architecture in B0_LR_ARCHITECTURES:
        items = tuple(
            sorted(
                (row for row in b0 if row.architecture == architecture),
                key=lambda row: float(row.config["model_learning_rate"]),
            )
        )
        learning_rates = tuple(
            float(row.config["model_learning_rate"]) for row in items
        )
        if len(items) != len(B0_LR_VALUES) or not np.allclose(
            learning_rates,
            B0_LR_VALUES,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                f"B0 architecture {architecture} must have the registered "
                f"LR triplet {B0_LR_VALUES!r}; found {learning_rates!r}."
            )
        triplets.append((architecture, items))
    return tuple(triplets)


def _b0_central_rows(cohort: Sequence[S1Row]) -> tuple[S1Row, ...]:
    """Return all ten B0 architectures at the controlled central LR."""

    central = _rows(cohort, comparison="S1-B0-ARCH")
    by_architecture = {row.architecture: row for row in central}
    missing = set(B0_TOPOLOGY_ORDER).difference(by_architecture)
    extra = set(by_architecture).difference(B0_TOPOLOGY_ORDER)
    if missing or extra or len(central) != len(B0_TOPOLOGY_ORDER):
        raise ValueError(
            "Unexpected B0 central-LR architecture cohort: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}, "
            f"rows={len(central)}."
        )
    ordered = tuple(by_architecture[architecture] for architecture in B0_TOPOLOGY_ORDER)
    unexpected_lrs = [
        (row.architecture, row.config["model_learning_rate"])
        for row in ordered
        if not math.isclose(
            float(row.config["model_learning_rate"]),
            3e-5,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ]
    if unexpected_lrs:
        raise ValueError(f"B0 central rows have unexpected LRs: {unexpected_lrs!r}.")
    return ordered


def write_b0_endpoint_table(
    cohort: Sequence[S1Row],
    output: str | Path,
) -> Path:
    """Write B0 endpoints with one loss/opportunity pair per learning rate."""

    b0_rows = _rows(cohort, block="S1-B0")
    by_key = {
        (row.architecture, float(row.config["model_learning_rate"])): row
        for row in b0_rows
    }
    if len(by_key) != len(b0_rows):
        raise ValueError("B0 endpoint table has duplicate architecture/LR cells.")

    def endpoint_cells(row: S1Row | None) -> str:
        if row is None:
            return r"\multicolumn{2}{c}{--}"
        return f"{row.loss:.5f} & {row.r_model_pct:.2f}"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        (
            r"\caption{Complete B0 endpoints with learning rates in parallel "
            r"column groups. Each group reports validation loss and $R_m$ "
            r"(\%). All 20 cells use seed-0 AdamW for 2,048 steps without "
            r"activation pressure. A0 uses stock GELU; gated architectures use "
            r"ordinary $G^+_0$. A dash means that the architecture was not "
            r"executed at that learning rate. Config IDs remain in the complete "
            r"appendix.}"
        ),
        r"\label{tab:b0}",
        r"\begin{tabular}{l rr rr rr}",
        r"\toprule",
        (
            r"\multirow{2}{*}{Architecture} & "
            r"\multicolumn{2}{c}{$10^{-5}$} & "
            r"\multicolumn{2}{c}{$3\times10^{-5}$} & "
            r"\multicolumn{2}{c}{$10^{-4}$} \\"
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"& $L$ & $R_m$ (\%) & $L$ & $R_m$ (\%) & $L$ & $R_m$ (\%) \\",
        r"\midrule",
    ]
    for architecture in B0_TOPOLOGY_ORDER:
        cells = [
            endpoint_cells(by_key.get((architecture, learning_rate)))
            for learning_rate in B0_LR_VALUES
        ]
        lines.append(f"{architecture} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _b0_matched_effects(
    cohort: Sequence[S1Row],
) -> tuple[B0MatchedEffect, ...]:
    """Reduce the ten registered central-LR architecture contrasts."""

    central = {row.architecture: row for row in _b0_central_rows(cohort)}
    effects = []
    for label, child_name, parent_name, group in B0_MATCHED_CONTRAST_SPECS:
        child = central[child_name]
        parent = central[parent_name]
        effects.append(
            B0MatchedEffect(
                label=label,
                child=child_name,
                parent=parent_name,
                group=group,
                delta_loss=child.loss - parent.loss,
                delta_r_model_pct=child.r_model_pct - parent.r_model_pct,
            )
        )
    return tuple(effects)


def build_b0_matched_architecture_effect_figure(
    cohort: Sequence[S1Row],
) -> Figure:
    """Plot logical-opportunity gain against matched validation-loss cost."""

    effects = _b0_matched_effects(cohort)
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.35),
    )
    group_styles = {
        "relu_path": ("ReLU path", "#0072B2", "o"),
        "attention_sites": ("Attention gates", "#E69F00", "s"),
        "add_v": ("Add V to Q/K", "#009E73", "^"),
        "placement": ("PRE/POST placement", "#6F4C9B", "D"),
    }
    label_offsets = {
        "MLP-hidden ReLU": (8, 10),
        "+ branch ReLUs": (-12, 4),
        "+ Q (POST)": (-8, -20),
        "+ K (POST)": (8, 13),
        "+ V": (6, 16),
        "+ Q/K (PRE)": (-12, -25),
        "+ Q/K (POST)": (-5, 12),
        "+ V (PRE Q/K)": (-12, -28),
        "+ V (POST Q/K)": (8, -12),
        "Q/K PRE -> POST": (-8, 13),
    }

    for effect in effects:
        _group_label, color, marker = group_styles[effect.group]
        if effect.group == "add_v" and effect.child == "A6-POST":
            marker = "v"
        axis.scatter(
            [effect.delta_r_model_pct],
            [effect.delta_loss],
            s=58,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        x_offset, y_offset = label_offsets[effect.label]
        axis.annotate(
            effect.label,
            (effect.delta_r_model_pct, effect.delta_loss),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset >= 0 else "right",
            va="bottom" if y_offset >= 0 else "top",
            color=color,
            fontsize=8.0,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.65,
                "alpha": 0.75,
                "shrinkA": 1.0,
                "shrinkB": 4.0,
            },
            annotation_clip=False,
            zorder=4,
        )

    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, zorder=0)
    axis.set_xlim(0.0, 4.45)
    axis.set_ylim(-0.069, 0.031)
    axis.set_xticks((0, 1, 2, 3, 4))
    y_ticks = (-0.06, -0.04, -0.02, 0.0, 0.02)
    axis.set_yticks(
        y_ticks,
        ["-0.06", "-0.04", "-0.02", "0.00", "+0.02"],
    )
    axis.set_xlabel(
        r"$\Delta R_{\mathrm{model}}$ vs registered parent (percentage points)"
    )
    axis.set_ylabel(r"$\Delta$ validation loss vs registered parent")
    _finish_axis(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=6.0,
            markerfacecolor=color,
            markeredgecolor="white",
            label=label,
        )
        for label, color, marker in group_styles.values()
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    figure.suptitle(
        "B0 matched architecture effects",
        y=0.98,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.915,
        (
            r"Each point is child minus registered parent at LR "
            r"$3{\times}10^{-5}$; right and down is favorable"
        ),
        ha="center",
        va="center",
        fontsize=8.2,
    )
    figure.text(
        0.105,
        0.060,
        (
            "Seed 0; 2,048 steps; 311,296 selection tokens; no uncertainty "
            "estimate.\n"
            r"$R_{\mathrm{model}}$ is logical opportunity, not measured speedup."
        ),
        ha="left",
        va="center",
        fontsize=8.0,
        linespacing=1.25,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.215,
        top=0.76,
    )
    return figure


def generate_b0_matched_architecture_effect_figure(
    *,
    config_registry: str | Path,
    run_registry: str | Path,
    output: str | Path = Path("figures") / B0_MATCHED_EFFECT_FILENAME,
    save_png: bool = True,
) -> tuple[Path, ...]:
    """Generate the standalone B0 matched-effect figure."""

    cohort = load_s1_rows(config_registry, run_registry)
    return tuple(
        export_figure(
            lambda: build_b0_matched_architecture_effect_figure(cohort),
            output,
            save_png=save_png,
            style=REPORT04_PLOT_STYLE,
            profile=REPORT07_PROFILE,
        )
    )


def build_fixed_threshold_factorial_figure(cohort: Sequence[S1Row]) -> Figure:
    """B1 threshold dose and signed-versus-one-sided gate effects."""

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 3.75))
    factorial = _rows(cohort, comparison="S1-B1-FIX-ATTENTION")
    central = {
        row.architecture: row
        for row in _rows(cohort, comparison="S1-B0-ARCH")
    }
    strata = (
        ("pre_rope", "qk_only", "A5-QK-PRE"),
        ("post_rope", "qk_only", "A5-QK-POST"),
        ("pre_rope", "qkv", "A6-PRE"),
        ("post_rope", "qkv", "A6-POST"),
    )
    linestyle_by_scope = {"qk_only": "-", "qkv": "--"}
    x_positive = (0.03, 0.10, 0.30)

    def family_series(
        family: str,
        placement: str,
        scope: str,
        relu_parent: str,
    ) -> tuple[list[float], list[S1Row]]:
        items = sorted(
            [
                row
                for row in factorial
                if row.config["gate_family"] == family
                and row.config["qk_placement"] == placement
                and row.config["kappa_scope"] == scope
            ],
            key=lambda row: float(row.config["kappa"]),
        )
        if len(items) != 3:
            raise ValueError(
                "Every B1 attention stratum must contain three positive "
                f"thresholds; found {len(items)} for {family}/{placement}/{scope}."
            )
        control = central[relu_parent] if family == "gplus" else central["A3"]
        return [0.0, *x_positive], [control, *items]

    for placement, scope, relu_parent in strata:
        color = PLACEMENT_COLORS[placement]
        linestyle = linestyle_by_scope[scope]
        series: dict[str, list[S1Row]] = {}
        for family in ("gplus", "gpm"):
            x_values, rows = family_series(
                family,
                placement,
                scope,
                relu_parent,
            )
            series[family] = rows
            marker_face = color if family == "gplus" else "white"
            for axis, outcome in (
                (axes[0, 0], "loss"),
                (axes[1, 0], "r_model_pct"),
            ):
                axis.plot(
                    x_values,
                    [getattr(row, outcome) for row in rows],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    markersize=4.6,
                    linewidth=1.15,
                    zorder=3,
                )

        for axis, outcome in (
            (axes[0, 1], "loss"),
            (axes[1, 1], "r_model_pct"),
        ):
            axis.plot(
                x_positive,
                [
                    getattr(gpm, outcome) - getattr(gplus, outcome)
                    for gpm, gplus in zip(
                        series["gpm"][1:],
                        series["gplus"][1:],
                        strict=True,
                    )
                ],
                color=color,
                linestyle=linestyle,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.6,
                linewidth=1.15,
                zorder=3,
            )

    titles = (
        (0, 0, "a", r"Threshold dose: loss"),
        (0, 1, "b", r"Gate-family effect: loss"),
        (1, 0, "c", r"Threshold dose: opportunity"),
        (1, 1, "d", r"Gate-family effect: opportunity"),
    )
    for row, column, letter, title in titles:
        axis = axes[row, column]
        _panel_title(axis, letter, title)
        axis.set_xlim(-0.012, 0.315)
        axis.set_xticks((0.0, *x_positive), ("0", "0.03", "0.10", "0.30"))
        axis.set_xlabel(r"Fixed threshold $\kappa$")
        _finish_axis(axis)
    for axis in axes[:, 1]:
        axis.axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 0].set_ylabel("Validation loss")
    axes[1, 0].set_ylabel(r"$R_{\mathrm{model}}$ (%)")
    axes[0, 1].set_ylabel(r"$\Delta L$ ($G^\pm-G^+$)")
    axes[1, 1].set_ylabel(r"$\Delta R_{\mathrm{model}}$ (pp, $G^\pm-G^+$)")
    axes[0, 0].set_ylim(7.004, 7.044)
    axes[1, 0].set_ylim(4.8, 22.8)

    legend_handles = [
        Line2D([0], [0], color=PLACEMENT_COLORS["pre_rope"], label="PRE"),
        Line2D([0], [0], color=PLACEMENT_COLORS["post_rope"], label="POST"),
        Line2D([0], [0], color="#444444", linestyle="-", label="QK"),
        Line2D([0], [0], color="#444444", linestyle="--", label="QKV"),
        Line2D(
            [0], [0], color="#555555", marker="o", markerfacecolor="#555555",
            linestyle="none", label=r"$G^+$",
        ),
        Line2D(
            [0], [0], color="#555555", marker="o", markerfacecolor="white",
            linestyle="none", label=r"$G^\pm$",
        ),
    ]
    figure.legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="upper center",
        ncol=6,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.14,
        top=0.82,
        hspace=0.54,
        wspace=0.33,
    )
    return figure


def build_fixed_threshold_site_scope_figure(cohort: Sequence[S1Row]) -> Figure:
    """B1 POST site isolation and all-active branch scope."""

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.52))
    site_rows = _rows(cohort, comparison="S1-B1-POST-SITE-ISOLATION")
    branch_rows = _rows(cohort, comparison="S1-B1-BRANCH-SCOPE")
    by_number = _number_index(cohort)
    site_order = ("q_only", "k_only", "v_only")
    site_labels = ("Q", "K", "V")
    arch_order = ("A1-H", "A3", "A6-POST")

    for family, family_label in (("gplus", r"$G^+$"), ("gpm", r"$G^\pm$")):
        items = [
            next(
                row
                for row in site_rows
                if row.config["gate_family"] == family
                and row.config["kappa_scope"] == scope
            )
            for scope in site_order
        ]
        controls = (
            [by_number[number] for number in (129, 130, 131)]
            if family == "gplus"
            else [by_number[125]] * 3
        )
        color = GATE_COLORS[family]
        marker = GATE_MARKERS[family]
        axes[0, 0].plot(
            site_labels,
            [row.loss - control.loss for row, control in zip(items, controls, strict=True)],
            marker=marker,
            linestyle="none",
            color=color,
            label=family_label,
        )
        axes[1, 0].plot(
            site_labels,
            [
                row.r_model_pct - control.r_model_pct
                for row, control in zip(items, controls, strict=True)
            ],
            marker=marker,
            linestyle="none",
            color=color,
        )

    branch_colors = {0.03: "#6F4C9B", 0.10: "#009E73"}
    branch_markers = {0.03: "o", 0.10: "s"}
    branch_controls = {
        "A1-H": by_number[124],
        "A3": by_number[125],
        "A6-POST": by_number[127],
    }
    for kappa in (0.03, 0.10):
        items = [
            next(
                row
                for row in branch_rows
                if row.architecture == architecture
                and math.isclose(float(row.config["kappa"]), kappa)
            )
            for architecture in arch_order
        ]
        label = rf"$\kappa={kappa:.2f}$"
        color = branch_colors[kappa]
        axes[0, 1].plot(
            arch_order,
            [row.loss - branch_controls[row.architecture].loss for row in items],
            marker=branch_markers[kappa],
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            arch_order,
            [
                row.r_model_pct - branch_controls[row.architecture].r_model_pct
                for row in items
            ],
            marker=branch_markers[kappa],
            color=color,
        )

    titles = (
        (0, 0, "a", r"POST site effect ($\kappa=0.10$)"),
        (1, 0, "c", "POST site compute effect"),
        (0, 1, "b", "All-active branch effect"),
        (1, 1, "d", "All-active compute effect"),
    )
    for row, column, letter, title in titles:
        _panel_title(axes[row, column], letter, title)
        axes[row, column].axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
        _finish_axis(axes[row, column])
    axes[0, 0].set_ylabel(r"$\Delta$ validation loss")
    axes[1, 0].set_ylabel(r"$\Delta R_{\mathrm{model}}$ (pp)")
    axes[1, 0].set_xlabel("Thresholded attention site")
    axes[1, 1].set_xlabel("Gate topology")
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    axes[1, 1].tick_params(axis="x", rotation=12)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    figure.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.14,
        top=0.91,
        hspace=0.42,
        wspace=0.27,
    )
    return figure


def _b1_factorial_series(
    cohort: Sequence[S1Row],
) -> dict[tuple[str, str], tuple[tuple[float, S1Row], ...]]:
    """Return the four complete B1 architecture ladders with kappa-zero controls."""

    factorial = _rows(cohort, comparison="S1-B1-FIX-ATTENTION")
    central = {
        row.architecture: row
        for row in _rows(cohort, comparison="S1-B0-ARCH")
    }
    series: dict[tuple[str, str], tuple[tuple[float, S1Row], ...]] = {}
    for label, placement, scope, relu_parent in B1_FACTORIAL_ARCHITECTURES:
        for family in ("gplus", "gpm"):
            positive = sorted(
                (
                    row
                    for row in factorial
                    if row.config["gate_family"] == family
                    and row.config["qk_placement"] == placement
                    and row.config["kappa_scope"] == scope
                ),
                key=lambda row: float(row.config["kappa"]),
            )
            kappas = tuple(float(row.config["kappa"]) for row in positive)
            if kappas != (0.03, 0.10, 0.30):
                raise ValueError(
                    "Every B1 factorial architecture/gate ladder must contain "
                    f"kappa=(0.03, 0.10, 0.30); found {kappas!r} for "
                    f"{label}/{family}."
                )
            control = central[relu_parent] if family == "gplus" else central["A3"]
            series[(label, family)] = (
                (0.0, control),
                *((kappa, row) for kappa, row in zip(kappas, positive, strict=True)),
            )
    if len(series) != 8:
        raise ValueError(f"Expected eight B1 factorial ladders, found {len(series)}.")
    return series


def _b1_endpoint_rows(cohort: Sequence[S1Row]) -> tuple[B1Endpoint, ...]:
    """Return every B1 cell once plus the registered zero-threshold controls."""

    by_number = _number_index(cohort)
    displayed: list[B1Endpoint] = []
    for architecture, family, endpoints in B1_TABLE_GROUPS:
        for kappa, number in endpoints:
            row = by_number[number]
            if kappa > 0.0:
                if row.block != "S1-B1":
                    raise ValueError(f"Config {number} is not a B1 scientific cell.")
                if row.config.get("gate_family") != family:
                    raise ValueError(
                        f"Config {number} has the wrong gate family for {architecture}."
                    )
                if not math.isclose(float(row.config["kappa"]), kappa):
                    raise ValueError(
                        f"Config {number} has the wrong kappa for {architecture}."
                    )
            displayed.append(
                B1Endpoint(
                    architecture=architecture,
                    gate_family=family,
                    kappa=kappa,
                    row=row,
                )
            )

    positive_numbers = {
        endpoint.row.number for endpoint in displayed if endpoint.kappa > 0.0
    }
    canonical_b1_numbers = {row.number for row in _rows(cohort, block="S1-B1")}
    if positive_numbers != canonical_b1_numbers:
        missing = sorted(canonical_b1_numbers - positive_numbers)
        extra = sorted(positive_numbers - canonical_b1_numbers)
        raise ValueError(
            f"B1 endpoint table does not close: missing={missing!r}, extra={extra!r}."
        )
    keys = {
        (endpoint.architecture, endpoint.gate_family, endpoint.kappa)
        for endpoint in displayed
    }
    if len(keys) != len(displayed):
        raise ValueError("B1 endpoint table has duplicate architecture/gate/kappa keys.")
    if len(displayed) != 53:
        raise ValueError(f"Expected 53 displayed B1 rows, found {len(displayed)}.")
    return tuple(displayed)


def write_b1_endpoint_table(
    cohort: Sequence[S1Row],
    output: str | Path,
) -> Path:
    """Write the complete B1 endpoint table with gate families side by side."""

    endpoints = _b1_endpoint_rows(cohort)
    architecture_order = tuple(
        dict.fromkeys(endpoint.architecture for endpoint in endpoints)
    )
    compact_rows = []
    for architecture in architecture_order:
        kappas = sorted(
            {
                endpoint.kappa
                for endpoint in endpoints
                if endpoint.architecture == architecture
            }
        )
        for kappa in kappas:
            by_gate = {
                endpoint.gate_family: endpoint
                for endpoint in endpoints
                if endpoint.architecture == architecture
                and endpoint.kappa == kappa
            }
            compact_rows.append((architecture, kappa, by_gate))

    def endpoint_cells(endpoint: B1Endpoint | None) -> str:
        if endpoint is None:
            return r"\multicolumn{3}{c}{--}"
        return (
            f"{endpoint.row.number} & {endpoint.row.loss:.5f} & "
            f"{endpoint.row.r_model_pct:.2f}"
        )

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{longtable}{l r r r r r r r}",
        (
            r"\caption{Complete B1 endpoints with the two gate families in "
            r"parallel column groups. Each group reports config, validation "
            r"loss, and $R_m$ (\%). The 36 positive-threshold cells are shown "
            r"with their registered $\kappa=0$ controls; repeated config IDs "
            r"are shared controls. A6-POST-QKV thresholds Q/K/V, whereas "
            r"A6-POST-ALL thresholds a/m/h/Q/K/V. $G^+_0$ is ordinary ReLU; "
            r"$G^\pm_0$ is identity and uses A3 (config 125). A dash means that "
            r"the gate family was not executed for that row. All cells use "
            r"seed-0 AdamW at $3\times10^{-5}$ without pressure.}"
            r"\label{tab:b1}\\"
        ),
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{$\kappa$} & "
        r"\multicolumn{3}{c}{$G^+$} & \multicolumn{3}{c}{$G^\pm$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"& & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{$\kappa$} & "
        r"\multicolumn{3}{c}{$G^+$} & \multicolumn{3}{c}{$G^\pm$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"& & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, (architecture, kappa, by_gate) in enumerate(compact_rows):
        previous = compact_rows[index - 1] if index else None
        next_row = compact_rows[index + 1] if index + 1 < len(compact_rows) else None
        first_architecture = previous is None or previous[0] != architecture
        architecture_span = sum(
            item_architecture == architecture
            for item_architecture, _kappa, _by_gate in compact_rows
        )
        architecture_cell = (
            rf"\multirow{{{architecture_span}}}{{*}}{{{architecture}}}"
            if first_architecture
            else ""
        )
        kappa_label = "0" if kappa == 0.0 else f"{kappa:.2f}"
        same_architecture_next = (
            next_row is not None and next_row[0] == architecture
        )
        terminator = r" \\*" if same_architecture_next else r" \\"
        lines.append(
            f"{architecture_cell} & {kappa_label} & "
            f"{endpoint_cells(by_gate.get('gplus'))} & "
            f"{endpoint_cells(by_gate.get('gpm'))}{terminator}"
        )
        if not same_architecture_next and next_row is not None:
            lines.append(r"\midrule")
    lines.extend((r"\end{longtable}", r"\endgroup", ""))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _b1_metric_limits(
    series: Mapping[tuple[str, str], Sequence[tuple[float, S1Row]]],
    outcome: str,
) -> tuple[float, float]:
    values = [
        float(getattr(row, outcome))
        for items in series.values()
        for _kappa, row in items
    ]
    lower = min(values)
    upper = max(values)
    padding = max(0.06 * (upper - lower), 0.002 if outcome == "loss" else 0.5)
    return lower - padding, upper + padding


def _style_b1_metric_axes(
    axes: Sequence[Any],
    series: Mapping[tuple[str, str], Sequence[tuple[float, S1Row]]],
) -> None:
    for axis, letter, title, outcome in (
        (axes[0], "a", "Validation loss", "loss"),
        (axes[1], "b", r"$R_{\mathrm{model}}$", "r_model_pct"),
    ):
        _panel_title(axis, letter, title)
        axis.set_ylim(*_b1_metric_limits(series, outcome))
        _finish_axis(axis)
    axes[0].set_ylabel("Validation loss")
    axes[1].set_ylabel(r"$R_{\mathrm{model}}$ (%)")


def build_fixed_threshold_effect_figure(cohort: Sequence[S1Row]) -> Figure:
    """Show absolute B1 endpoints across the complete threshold ladders."""

    series = _b1_factorial_series(cohort)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 2.85),
    )
    for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES:
        color = B1_ARCHITECTURE_COLORS[architecture]
        for family in ("gplus", "gpm"):
            items = series[(architecture, family)]
            style = B1_GATE_STYLES[family]
            for axis, outcome in zip(axes, ("loss", "r_model_pct"), strict=True):
                axis.plot(
                    [kappa for kappa, _row in items],
                    [getattr(row, outcome) for _kappa, row in items],
                    color=color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor=color if family == "gplus" else "white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    markersize=4.5,
                    linewidth=1.15,
                    zorder=3,
                )

    _style_b1_metric_axes(axes, series)
    for axis in axes:
        axis.set_xlim(-0.012, 0.315)
        axis.set_xticks((0.0, 0.03, 0.10, 0.30), ("0", "0.03", "0.10", "0.30"))
        axis.set_xlabel(r"Fixed threshold $\kappa$")
    architecture_handles = [
        Line2D(
            [0],
            [0],
            color=B1_ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES
    ]
    gate_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=B1_GATE_STYLES[family]["linestyle"],
            marker=B1_GATE_STYLES[family]["marker"],
            markerfacecolor="#444444" if family == "gplus" else "white",
            label=B1_GATE_STYLES[family]["label"],
        )
        for family in ("gplus", "gpm")
    ]
    figure.legend(
        handles=architecture_handles + gate_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.22,
        top=0.76,
        wspace=0.28,
    )
    return figure


def build_gate_type_effect_figure(cohort: Sequence[S1Row]) -> Figure:
    """Show absolute matched G-plus and G-plus/minus endpoints."""

    series = _b1_factorial_series(cohort)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 2.85),
    )
    marker_by_kappa = {0.03: "o", 0.10: "s", 0.30: "^"}
    for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES:
        color = B1_ARCHITECTURE_COLORS[architecture]
        gplus = dict(series[(architecture, "gplus")])
        gpm = dict(series[(architecture, "gpm")])
        for kappa in (0.03, 0.10, 0.30):
            for axis, outcome in zip(axes, ("loss", "r_model_pct"), strict=True):
                axis.plot(
                    (0.0, 1.0),
                    (
                        getattr(gplus[kappa], outcome),
                        getattr(gpm[kappa], outcome),
                    ),
                    color=color,
                    marker=marker_by_kappa[kappa],
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markersize=4.8,
                    linewidth=1.05,
                    alpha=0.88,
                    zorder=3,
                )

    _style_b1_metric_axes(axes, series)
    for axis in axes:
        axis.set_xlim(-0.22, 1.22)
        axis.set_xticks((0.0, 1.0), (r"$G^+$", r"$G^\pm$"))
        axis.set_xlabel("Gate type")
    architecture_handles = [
        Line2D(
            [0],
            [0],
            color=B1_ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture, _placement, _scope, _parent in B1_FACTORIAL_ARCHITECTURES
    ]
    kappa_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=marker,
            linestyle="none",
            label=rf"$\kappa={kappa:.2f}$",
        )
        for kappa, marker in marker_by_kappa.items()
    ]
    figure.legend(
        handles=architecture_handles + kappa_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        columnspacing=1.05,
        handletextpad=0.38,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.22,
        top=0.76,
        wspace=0.28,
    )
    return figure


def _b2_architecture_label(row: S1Row) -> str:
    """Use explicit learned-gate scope in the displayed B2 architecture name."""

    comparison = str(row.config["comparison_id"])
    if row.architecture == "A6-PRE":
        return "A6-PRE-QKV"
    if row.architecture == "A6-POST":
        return (
            "A6-POST-ALL"
            if comparison == "S1-B2-LEARNED-ATG-BRANCH"
            else "A6-POST-QKV"
        )
    return row.architecture


def _b2_endpoint_rows(cohort: Sequence[S1Row]) -> tuple[B2Endpoint, ...]:
    """Return the complete B2 endpoint grid with explicit display labels."""

    parameterization_labels = {
        "global": "Model-wide",
        "per_site": "Per site",
        "per_layer_site": "Per layer/site",
    }
    scale_labels = {
        "absolute": "Absolute",
        "rms_relative": "RMS-relative",
    }
    endpoints = []
    for row in _rows(cohort, block="S1-B2"):
        scope = str(row.config["kappa_scope"])
        scale = str(row.config["threshold_scale"])
        endpoints.append(
            B2Endpoint(
                architecture=_b2_architecture_label(row),
                gate_family=str(row.config["gate_family"]),
                threshold_scale=scale_labels[scale],
                parameterization=parameterization_labels[scope],
                row=row,
            )
        )
    architecture_order = {
        architecture: index
        for index, architecture in enumerate(B2_ARCHITECTURE_ORDER)
    }
    gate_order = {"gplus": 0, "gpm": 1}
    scale_order = {"Absolute": 0, "RMS-relative": 1}
    parameter_order = {"Model-wide": 0, "Per site": 1, "Per layer/site": 2}
    endpoints.sort(
        key=lambda endpoint: (
            architecture_order[endpoint.architecture],
            gate_order[endpoint.gate_family],
            scale_order[endpoint.threshold_scale],
            parameter_order[endpoint.parameterization],
            endpoint.row.number,
        )
    )
    if len(endpoints) != 26 or len({item.row.number for item in endpoints}) != 26:
        raise ValueError("The B2 endpoint table must contain all 26 cells exactly once.")
    return tuple(endpoints)


def write_b2_endpoint_table(
    cohort: Sequence[S1Row],
    output: str | Path,
) -> Path:
    """Write the complete B2 endpoint table with gate families side by side."""

    endpoints = _b2_endpoint_rows(cohort)
    scale_order = {"Absolute": 0, "RMS-relative": 1}
    parameter_order = {"Model-wide": 0, "Per site": 1, "Per layer/site": 2}
    keys = {
        (
            endpoint.architecture,
            endpoint.threshold_scale,
            endpoint.parameterization,
        )
        for endpoint in endpoints
    }
    compact_rows = []
    for architecture in B2_ARCHITECTURE_ORDER:
        architecture_keys = sorted(
            (key for key in keys if key[0] == architecture),
            key=lambda key: (scale_order[key[1]], parameter_order[key[2]]),
        )
        for key in architecture_keys:
            by_gate = {
                endpoint.gate_family: endpoint
                for endpoint in endpoints
                if (
                    endpoint.architecture,
                    endpoint.threshold_scale,
                    endpoint.parameterization,
                )
                == key
            }
            compact_rows.append((*key, by_gate))

    def endpoint_cells(endpoint: B2Endpoint | None) -> str:
        if endpoint is None:
            return r"\multicolumn{3}{c}{--}"
        return (
            f"{endpoint.row.number} & {endpoint.row.loss:.5f} & "
            f"{endpoint.row.r_model_pct:.2f}"
        )

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{longtable}{l l l r r r r r r}",
        (
            r"\caption{Complete B2 endpoints with the two gate families in "
            r"parallel column groups. Each group reports config, validation "
            r"loss, and $R_m$ (\%). Rows specify architecture, score scale, and "
            r"threshold parameterization. A6-POST-QKV learns thresholds "
            r"only at Q/K/V; A6-POST-ALL learns them at a/m/h/Q/K/V. A dash "
            r"means that the gate family was not executed for that row. All 26 "
            r"cells use seed-0 AdamW at $3\times10^{-5}$ for 2,048 steps without "
            r"activation pressure.}"
            r"\label{tab:b2}\\"
        ),
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Score scale} & "
        r"\multirow{2}{*}{$\kappa$ parameters} & \multicolumn{3}{c}{$G^+$} & "
        r"\multicolumn{3}{c}{$G^\pm$} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Score scale} & "
        r"\multirow{2}{*}{$\kappa$ parameters} & \multicolumn{3}{c}{$G^+$} & "
        r"\multicolumn{3}{c}{$G^\pm$} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, (architecture, scale, parameterization, by_gate) in enumerate(
        compact_rows
    ):
        previous = compact_rows[index - 1] if index else None
        next_row = compact_rows[index + 1] if index + 1 < len(compact_rows) else None
        first_architecture = previous is None or previous[0] != architecture
        architecture_span = sum(
            item_architecture == architecture
            for item_architecture, _scale, _parameterization, _by_gate in compact_rows
        )
        architecture_cell = (
            rf"\multirow{{{architecture_span}}}{{*}}{{{architecture}}}"
            if first_architecture
            else ""
        )
        same_architecture_next = (
            next_row is not None and next_row[0] == architecture
        )
        terminator = r" \\*" if same_architecture_next else r" \\"
        lines.append(
            f"{architecture_cell} & {scale} & {parameterization} & "
            f"{endpoint_cells(by_gate.get('gplus'))} & "
            f"{endpoint_cells(by_gate.get('gpm'))}{terminator}"
        )
        if not same_architecture_next and next_row is not None:
            lines.append(r"\midrule")
    lines.extend((r"\end{longtable}", r"\endgroup", ""))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _b2_kappa_stratum(row: S1Row) -> tuple[int, str]:
    """Return the prespecified display stratum for one learned-threshold row."""

    comparison = str(row.config["comparison_id"])
    scale = str(row.config["threshold_scale"])
    scope = str(row.config["kappa_scope"])
    if comparison == "S1-B2-LEARNED-ATG-ATTENTION":
        label = (
            r"Q/K(/V), absolute, per layer/site"
            if scale == "absolute"
            else r"Q/K(/V), RMS-relative, per layer/site"
        )
        return (0 if scale == "absolute" else 1, label)
    if comparison == "S1-B2-LEARNED-ATG-BRANCH":
        label = (
            r"a/m/h, absolute, per layer/site"
            if scale == "absolute"
            else r"a/m/h, RMS-relative, per layer/site"
        )
        return (2 if scale == "absolute" else 3, label)
    if comparison == "S1-B2-LEARNED-ATG-GRANULARITY":
        if scale != "absolute" or scope not in {"global", "per_site"}:
            raise ValueError(
                f"Unexpected B2 granularity row {row.number}: scale={scale}, scope={scope}."
            )
        label = (
            r"A6-POST Q/K/V, absolute, model-wide"
            if scope == "global"
            else r"A6-POST Q/K/V, absolute, per site"
        )
        return (4 if scope == "global" else 5, label)
    raise ValueError(f"Unknown B2 learned-threshold stratum for config {row.number}.")


def _b2_final_kappa_values(row: S1Row, repo_root: Path) -> tuple[float, ...]:
    """Load exact final learned thresholds from one canonical metrics artifact."""

    result_path = Path(str(row.run["result_path"]))
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    metrics_path = result_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prefix = "final/atg/parameter/"
    suffix = "/kappa"
    values = tuple(
        float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and key.endswith(suffix)
    )
    if not values:
        raise ValueError(f"No final learned-kappa values in {metrics_path}.")
    for key, observed in (
        ("final/atg/kappa_min", min(values)),
        ("final/atg/kappa_mean", sum(values) / len(values)),
        ("final/atg/kappa_max", max(values)),
    ):
        if key not in metrics or not math.isclose(
            float(metrics[key]), observed, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError(f"Learned-kappa summary mismatch for config {row.number}: {key}.")
    return values


def b2_kappa_distribution_rows(
    cohort: Sequence[S1Row],
    repo_root: str | Path,
) -> tuple[tuple[str, int, int, float, float, float], ...]:
    """Aggregate final learned thresholds by the six prespecified B2 strata."""

    grouped: dict[int, dict[str, Any]] = {}
    root = Path(repo_root)
    for row in _rows(cohort, block="S1-B2"):
        order, label = _b2_kappa_stratum(row)
        group = grouped.setdefault(order, {"label": label, "runs": 0, "values": []})
        if group["label"] != label:
            raise ValueError(f"Conflicting B2 kappa labels for stratum {order}.")
        group["runs"] += 1
        group["values"].extend(_b2_final_kappa_values(row, root))
    if set(grouped) != set(range(6)):
        raise ValueError(f"B2 learned-kappa strata are incomplete: {sorted(grouped)!r}.")

    result = []
    for order in range(6):
        group = grouped[order]
        values = tuple(float(value) for value in group["values"])
        result.append(
            (
                str(group["label"]),
                int(group["runs"]),
                len(values),
                min(values),
                sum(values) / len(values),
                max(values),
            )
        )
    return tuple(result)


def write_b2_kappa_distribution_table(
    cohort: Sequence[S1Row],
    output: str | Path,
    *,
    repo_root: str | Path,
) -> Path:
    """Write a compact final learned-threshold distribution summary."""

    rows = b2_kappa_distribution_rows(cohort, repo_root)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        (
            r"\caption{Distribution of the final learned thresholds. All cells "
            r"start from $\kappa=0.10$; values are read from the exact final "
            r"checkpoint summaries. The mean pools threshold parameters: a "
            r"run contributes one model-wide value, three site-shared values, "
            r"or one value for every learned layer/site. Absolute and "
            r"RMS-relative values live on different "
            r"score scales and should not be compared as activation units.}"
        ),
        r"\label{tab:b2-kappa}",
        r"\begin{tabularx}{\textwidth}{X r r r r r}",
        r"\toprule",
        r"Threshold stratum & Runs & Parameters & Min. & Mean & Max. \\",
        r"\midrule",
    ]
    for label, runs, parameters, minimum, mean, maximum in rows:
        lines.append(
            f"{label} & {runs} & {parameters} & {minimum:.4f} & "
            f"{mean:.4f} & {maximum:.4f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _b2_abs_vs_fixed(
    cohort: Sequence[S1Row],
) -> list[tuple[S1Row, S1Row, str]]:
    by_design = _config_index(cohort)
    pairs = []
    for learned in _rows(cohort, block="S1-B2"):
        if learned.config.get("threshold_scale") != "absolute":
            continue
        control_id = learned.config.get("matched_control_id")
        if not control_id or str(control_id) not in by_design:
            raise ValueError(f"Missing fixed control for learned config {learned.number}.")
        pairs.append(
            (learned, by_design[str(control_id)], _b2_architecture_label(learned))
        )
    if len(pairs) != 15:
        raise ValueError(f"Expected 15 learned-absolute/fixed pairs, found {len(pairs)}.")
    return pairs


def _b2_layer_site_abs_vs_fixed(
    cohort: Sequence[S1Row],
) -> list[tuple[S1Row, S1Row, str]]:
    pairs = [
        pair
        for pair in _b2_abs_vs_fixed(cohort)
        if pair[0].config.get("kappa_scope") == "per_layer_site"
    ]
    if len(pairs) != 11:
        raise ValueError(
            f"Expected 11 per-layer/site learned/fixed pairs, found {len(pairs)}."
        )
    return pairs


def _b2_rms_vs_abs(cohort: Sequence[S1Row]) -> list[tuple[S1Row, S1Row, str]]:
    b2 = _rows(cohort, block="S1-B2")
    pairs = []
    for rms in [row for row in b2 if row.config.get("threshold_scale") == "rms_relative"]:
        candidates = [
            row
            for row in b2
            if row.config.get("threshold_scale") == "absolute"
            and row.config.get("comparison_id") == rms.config.get("comparison_id")
            and row.architecture == rms.architecture
            and row.config.get("gate_family") == rms.config.get("gate_family")
            and row.config.get("qk_placement") == rms.config.get("qk_placement")
            and row.config.get("kappa_scope") == rms.config.get("kappa_scope")
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one absolute match for RMS config {rms.number}, found {len(candidates)}."
            )
        pairs.append((rms, candidates[0], _b2_architecture_label(rms)))
    if len(pairs) != 11:
        raise ValueError(f"Expected 11 RMS/absolute pairs, found {len(pairs)}.")
    return pairs


def _b2_metric_limits(cohort: Sequence[S1Row], outcome: str) -> tuple[float, float]:
    rows = {
        row.number: row
        for learned, fixed, _architecture in _b2_layer_site_abs_vs_fixed(cohort)
        for row in (learned, fixed)
    }
    for rms, absolute, _architecture in _b2_rms_vs_abs(cohort):
        rows[rms.number] = rms
        rows[absolute.number] = absolute
    values = [float(getattr(row, outcome)) for row in rows.values()]
    lower = min(values)
    upper = max(values)
    padding = max(0.06 * (upper - lower), 0.003 if outcome == "loss" else 0.5)
    return lower - padding, upper + padding


def _b2_legend_handles() -> list[Line2D]:
    architecture_handles = [
        Line2D(
            [0],
            [0],
            color=B2_ARCHITECTURE_COLORS[architecture],
            label=architecture,
        )
        for architecture in B2_ARCHITECTURE_ORDER
    ]
    gate_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=B1_GATE_STYLES[family]["linestyle"],
            marker=B1_GATE_STYLES[family]["marker"],
            markerfacecolor="#444444" if family == "gplus" else "white",
            label=B1_GATE_STYLES[family]["label"],
        )
        for family in ("gplus", "gpm")
    ]
    return gate_handles + architecture_handles


def _b2_threshold_triplets(
    cohort: Sequence[S1Row],
) -> list[tuple[S1Row, S1Row, S1Row, str]]:
    """Return fixed, learned-absolute, and learned-RMS endpoints."""

    rms_by_absolute = {
        absolute.number: (rms, architecture)
        for rms, absolute, architecture in _b2_rms_vs_abs(cohort)
    }
    triplets = []
    for absolute, fixed, architecture in _b2_layer_site_abs_vs_fixed(cohort):
        rms, rms_architecture = rms_by_absolute[absolute.number]
        if rms_architecture != architecture:
            raise ValueError(
                f"Architecture mismatch for learned config {absolute.number}."
            )
        triplets.append((fixed, absolute, rms, architecture))
    if len(triplets) != 11:
        raise ValueError(f"Expected 11 B2 threshold triplets, found {len(triplets)}.")
    return triplets


def build_learned_threshold_figure(cohort: Sequence[S1Row]) -> Figure:
    """Show fixed, learned-absolute, and learned-RMS endpoints."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 3.05),
    )
    for fixed, absolute, rms, architecture in _b2_threshold_triplets(cohort):
        family = str(absolute.config["gate_family"])
        style = B1_GATE_STYLES[family]
        color = B2_ARCHITECTURE_COLORS[architecture]
        for axis, outcome in zip(axes, ("loss", "r_model_pct"), strict=True):
            axis.plot(
                (0.0, 1.0, 2.0),
                tuple(getattr(row, outcome) for row in (fixed, absolute, rms)),
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=color if family == "gplus" else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.8,
                linewidth=1.1,
                alpha=0.90,
                zorder=3,
            )
    for axis, letter, title, outcome in (
        (axes[0], "a", "Validation loss", "loss"),
        (axes[1], "b", r"$R_{\mathrm{model}}$", "r_model_pct"),
    ):
        _panel_title(axis, letter, title)
        axis.set_xlim(-0.18, 2.42)
        axis.set_xticks(
            (0.0, 1.0, 2.0),
            ("Fixed\n" + r"$\kappa=0.10$", "Learned\nabsolute", "Learned\nRMS-relative"),
        )
        axis.set_xlabel("Threshold setting")
        axis.set_ylim(*_b2_metric_limits(cohort, outcome))
        _finish_axis(axis)
    axes[0].set_ylabel("Validation loss")
    axes[1].set_ylabel(r"$R_{\mathrm{model}}$ (%)")
    figure.legend(
        handles=_b2_legend_handles(),
        loc="upper center",
        ncol=5,
        frameon=False,
        columnspacing=1.05,
        handletextpad=0.38,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.24,
        top=0.67,
        wspace=0.28,
    )
    return figure


def _b3_adamw_controls(cohort: Sequence[S1Row]) -> dict[str, S1Row]:
    controls = {}
    central = _rows(cohort, comparison="S1-B0-ARCH")
    for architecture in ("A3", "A6-POST"):
        matches = [row for row in central if row.architecture == architecture]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one B3 AdamW control for {architecture}, found {len(matches)}."
            )
        controls[architecture] = matches[0]
    return controls


def _b3_single(
    rows: Sequence[S1Row],
    *,
    architecture: str,
    method: str,
    weight: float,
    ricker_c: float | None,
    ricker_sigma: float | None,
) -> S1Row:
    def same_optional_float(value: Any, expected: float | None) -> bool:
        if expected is None:
            return value is None
        return value is not None and math.isclose(float(value), expected)

    matches = [
        row
        for row in rows
        if row.architecture == architecture
        and row.config["pressure_method"] == method
        and math.isclose(float(row.config["pressure_weight"]), weight)
        and same_optional_float(row.config.get("ricker_c"), ricker_c)
        and same_optional_float(row.config.get("ricker_sigma"), ricker_sigma)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one B3 endpoint for "
            f"{architecture}/{method}/w={weight}/c={ricker_c}/sigma={ricker_sigma}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _b3_weight_endpoints(
    cohort: Sequence[S1Row],
) -> tuple[B3WeightEndpoint, ...]:
    b3 = _rows(cohort, block="S1-B3")
    controls = _b3_adamw_controls(cohort)
    specs = (
        ("L1", (0.15, 1.0, 5.0), "l1_naive", "orthogonal_l1", None, None),
        (
            "Ricker",
            (0.10, 0.30, 1.0),
            "ricker_naive",
            "orthogonal_ricker",
            0.10,
            0.10,
        ),
    )
    endpoints = []
    for architecture in ("A3", "A6-POST"):
        for pressure, weights, naive_method, orthogonal_method, c, sigma in specs:
            for weight in weights:
                endpoints.append(
                    B3WeightEndpoint(
                        architecture=architecture,
                        pressure=pressure,
                        weight=weight,
                        adamw=controls[architecture],
                        naive=_b3_single(
                            b3,
                            architecture=architecture,
                            method=naive_method,
                            weight=weight,
                            ricker_c=c,
                            ricker_sigma=sigma,
                        ),
                        orthogonal=_b3_single(
                            b3,
                            architecture=architecture,
                            method=orthogonal_method,
                            weight=weight,
                            ricker_c=c,
                            ricker_sigma=sigma,
                        ),
                    )
                )
    used = {
        row.number
        for endpoint in endpoints
        for row in (endpoint.naive, endpoint.orthogonal)
    }
    expected = {
        row.number
        for row in b3
        if row.config["comparison_id"]
        in {"S1-B3-T1-CENTRAL", "S1-B3-T2-L1-FLANKS", "S1-B3-T3-RK-WEIGHT"}
    }
    if len(endpoints) != 12 or used != expected:
        raise ValueError("B3 pressure-weight endpoints do not close over 24 cells.")
    return tuple(endpoints)


def _b3_geometry_endpoints(
    cohort: Sequence[S1Row],
) -> tuple[B3GeometryEndpoint, ...]:
    b3 = _rows(cohort, block="S1-B3")
    controls = _b3_adamw_controls(cohort)
    specs = (
        (r"$c=\sigma$", (0.05, 0.10, 0.50), lambda value: (value, value)),
        (
            r"$\sigma$ at $c=0.10$",
            (0.05, 0.10, 0.20),
            lambda value: (0.10, value),
        ),
    )
    endpoints = []
    for architecture in ("A3", "A6-POST"):
        for sweep, values, parameters in specs:
            for value in values:
                c, sigma = parameters(value)
                endpoints.append(
                    B3GeometryEndpoint(
                        architecture=architecture,
                        sweep=sweep,
                        value=value,
                        adamw=controls[architecture],
                        naive=_b3_single(
                            b3,
                            architecture=architecture,
                            method="ricker_naive",
                            weight=0.30,
                            ricker_c=c,
                            ricker_sigma=sigma,
                        ),
                        orthogonal=_b3_single(
                            b3,
                            architecture=architecture,
                            method="orthogonal_ricker",
                            weight=0.30,
                            ricker_c=c,
                            ricker_sigma=sigma,
                        ),
                    )
                )
    used = {
        row.number
        for endpoint in endpoints
        for row in (endpoint.naive, endpoint.orthogonal)
    }
    expected = {
        row.number
        for row in b3
        if row.config["comparison_id"]
        in {"S1-B3-T4-RK-BASIN", "S1-B3-T5-RK-SHAPE"}
        or (
            row.config["comparison_id"] == "S1-B3-T1-CENTRAL"
            and row.config["pressure_method"]
            in {"ricker_naive", "orthogonal_ricker"}
        )
    }
    if len(endpoints) != 12 or used != expected:
        raise ValueError("B3 Ricker-geometry endpoints do not close over 20 cells.")
    return tuple(endpoints)


def _b3_table_cells(row: S1Row) -> str:
    return f"{row.number} & {row.loss:.5f} & {row.r_model_pct:.2f}"


def write_b3_weight_table(cohort: Sequence[S1Row], output: str | Path) -> Path:
    """Write all B3 pressure-weight endpoints with absolute metrics."""

    endpoints = _b3_weight_endpoints(cohort)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{longtable}{l l r r r r r r r r r r}",
        (
            r"\caption{Complete B3 pressure-weight endpoints. Each method group "
            r"reports config, validation loss, and $R_m$ (\%). L1 rows compare "
            r"L1N with OL1; Ricker rows compare RN with OR at $c=\sigma=0.10$. "
            r"AdamW is the matched same-architecture control. All cells use "
            r"seed 0 and 2,048 steps.}"
            r"\label{tab:b3-weight}\\"
        ),
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Pressure} & "
        r"\multirow{2}{*}{$w$} & \multicolumn{3}{c}{AdamW} & "
        r"\multicolumn{3}{c}{Naive} & \multicolumn{3}{c}{Orthogonal} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) & "
        r"Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{12}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Pressure} & "
        r"\multirow{2}{*}{$w$} & \multicolumn{3}{c}{AdamW} & "
        r"\multicolumn{3}{c}{Naive} & \multicolumn{3}{c}{Orthogonal} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) & "
        r"Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, endpoint in enumerate(endpoints):
        previous = endpoints[index - 1] if index else None
        next_endpoint = endpoints[index + 1] if index + 1 < len(endpoints) else None
        first_architecture = previous is None or previous.architecture != endpoint.architecture
        first_pressure = first_architecture or previous.pressure != endpoint.pressure
        architecture_cell = (
            rf"\multirow{{6}}{{*}}{{{endpoint.architecture}}}"
            if first_architecture
            else ""
        )
        pressure_cell = (
            rf"\multirow{{3}}{{*}}{{{endpoint.pressure}}}"
            if first_pressure
            else ""
        )
        same_architecture_next = (
            next_endpoint is not None
            and next_endpoint.architecture == endpoint.architecture
        )
        terminator = r" \\*" if same_architecture_next else r" \\"
        lines.append(
            f"{architecture_cell} & {pressure_cell} & {_compact_number(endpoint.weight)} & "
            f"{_b3_table_cells(endpoint.adamw)} & {_b3_table_cells(endpoint.naive)} & "
            f"{_b3_table_cells(endpoint.orthogonal)}{terminator}"
        )
        if next_endpoint is not None and next_endpoint.architecture != endpoint.architecture:
            lines.append(r"\midrule")
    lines.extend((r"\end{longtable}", r"\endgroup", ""))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def write_b3_geometry_table(cohort: Sequence[S1Row], output: str | Path) -> Path:
    """Write all B3 Ricker-geometry endpoints with absolute metrics."""

    endpoints = _b3_geometry_endpoints(cohort)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{longtable}{l l r r r r r r r r r r}",
        (
            r"\caption{Complete B3 Ricker-geometry endpoints at $w=0.30$. "
            r"Each method group reports config, validation loss, and $R_m$ "
            r"(\%). The basin sweep couples $c=\sigma$; the shape sweep fixes "
            r"$c=0.10$ and varies $\sigma$. AdamW is the matched "
            r"same-architecture control.}"
            r"\label{tab:b3-geometry}\\"
        ),
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Sweep} & "
        r"\multirow{2}{*}{Value} & \multicolumn{3}{c}{AdamW} & "
        r"\multicolumn{3}{c}{RN} & \multicolumn{3}{c}{OR} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) & "
        r"Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{12}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"\multirow{2}{*}{Architecture} & \multirow{2}{*}{Sweep} & "
        r"\multirow{2}{*}{Value} & \multicolumn{3}{c}{AdamW} & "
        r"\multicolumn{3}{c}{RN} & \multicolumn{3}{c}{OR} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"& & & Cfg & $L$ & $R_m$ (\%) & Cfg & $L$ & $R_m$ (\%) & "
        r"Cfg & $L$ & $R_m$ (\%) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, endpoint in enumerate(endpoints):
        previous = endpoints[index - 1] if index else None
        next_endpoint = endpoints[index + 1] if index + 1 < len(endpoints) else None
        first_architecture = previous is None or previous.architecture != endpoint.architecture
        first_sweep = first_architecture or previous.sweep != endpoint.sweep
        architecture_cell = (
            rf"\multirow{{6}}{{*}}{{{endpoint.architecture}}}"
            if first_architecture
            else ""
        )
        sweep_cell = (
            rf"\multirow{{3}}{{*}}{{{endpoint.sweep}}}"
            if first_sweep
            else ""
        )
        same_architecture_next = (
            next_endpoint is not None
            and next_endpoint.architecture == endpoint.architecture
        )
        terminator = r" \\*" if same_architecture_next else r" \\"
        lines.append(
            f"{architecture_cell} & {sweep_cell} & {_compact_number(endpoint.value)} & "
            f"{_b3_table_cells(endpoint.adamw)} & {_b3_table_cells(endpoint.naive)} & "
            f"{_b3_table_cells(endpoint.orthogonal)}{terminator}"
        )
        if next_endpoint is not None and next_endpoint.architecture != endpoint.architecture:
            lines.append(r"\midrule")
    lines.extend((r"\end{longtable}", r"\endgroup", ""))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _b3_metric_limits(endpoints: Sequence[Any], outcome: str) -> tuple[float, float]:
    values = [
        float(getattr(row, outcome))
        for endpoint in endpoints
        for row in (endpoint.adamw, endpoint.naive, endpoint.orthogonal)
    ]
    lower = min(values)
    upper = max(values)
    padding = max(0.06 * (upper - lower), 0.004 if outcome == "loss" else 0.6)
    return lower - padding, upper + padding


def _plot_b3_endpoint_sequences(
    axes: Sequence[Any],
    endpoints: Sequence[Any],
    *,
    method_ids: tuple[str, str],
    tick_labels: Sequence[str],
) -> None:
    x = tuple(range(len(tick_labels)))
    for architecture, linestyle in (("A3", "-"), ("A6-POST", "--")):
        architecture_rows = [
            endpoint for endpoint in endpoints if endpoint.architecture == architecture
        ]
        for field, method_id in zip(("naive", "orthogonal"), method_ids, strict=True):
            sequence = [architecture_rows[0].adamw] + [
                getattr(endpoint, field) for endpoint in architecture_rows
            ]
            for axis, outcome in zip(axes, ("loss", "r_model_pct"), strict=True):
                axis.plot(
                    x,
                    [getattr(row, outcome) for row in sequence],
                    color=PRESSURE_COLORS[method_id],
                    marker=PRESSURE_MARKERS[method_id],
                    linestyle=linestyle,
                    linewidth=1.15,
                    markersize=4.8,
                )
    for axis in axes:
        axis.set_xlim(-0.15, len(tick_labels) - 0.85)
        axis.set_xticks(x, tick_labels)
        _finish_axis(axis)


def build_pressure_weight_figure(cohort: Sequence[S1Row]) -> Figure:
    """Show absolute B3 L1 and Ricker pressure-weight endpoints."""

    endpoints = _b3_weight_endpoints(cohort)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.10),
        sharex="col",
        sharey="row",
    )
    specs = (
        ("L1", ("AdamW", "0.15", "1", "5"), ("l1_naive", "orthogonal_l1")),
        (
            "Ricker",
            ("AdamW", "0.10", "0.30", "1"),
            ("ricker_naive", "orthogonal_ricker"),
        ),
    )
    for column, (pressure, tick_labels, method_ids) in enumerate(specs):
        selected = [endpoint for endpoint in endpoints if endpoint.pressure == pressure]
        _plot_b3_endpoint_sequences(
            axes[:, column],
            selected,
            method_ids=method_ids,
            tick_labels=tick_labels,
        )
        _panel_title(axes[0, column], chr(ord("a") + column), f"{pressure}: validation loss")
        _panel_title(
            axes[1, column],
            chr(ord("c") + column),
            rf"{pressure}: $R_{{\mathrm{{model}}}}$",
        )
        axes[1, column].set_xlabel(rf"{pressure} pressure weight $w$")
    for axis in axes[0, :]:
        axis.set_ylim(*_b3_metric_limits(endpoints, "loss"))
    for axis in axes[1, :]:
        axis.set_ylim(*_b3_metric_limits(endpoints, "r_model_pct"))
    axes[0, 0].set_ylabel("Validation loss")
    axes[1, 0].set_ylabel(r"$R_{\mathrm{model}}$ (\%)")
    handles = _method_architecture_legend(
        method_names=(
            ("l1_naive", "Naive: L1N / RN"),
            ("orthogonal_l1", "Orthogonal: OL1 / OR"),
        )
    )
    figure.legend(
        handles,
        [item.get_label() for item in handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13,
        top=0.84,
        hspace=0.46,
        wspace=0.22,
    )
    return figure


def build_ricker_shape_figure(cohort: Sequence[S1Row]) -> Figure:
    """Show absolute B3 coupled-basin and fixed-c shape endpoints."""

    endpoints = _b3_geometry_endpoints(cohort)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.10),
        sharex="col",
        sharey="row",
    )
    specs = (
        (r"$c=\sigma$", ("AdamW", "0.05", "0.10", "0.50"), "Coupled basin"),
        (
            r"$\sigma$ at $c=0.10$",
            ("AdamW", "0.05", "0.10", "0.20"),
            "Shape",
        ),
    )
    for column, (sweep, tick_labels, title) in enumerate(specs):
        selected = [endpoint for endpoint in endpoints if endpoint.sweep == sweep]
        _plot_b3_endpoint_sequences(
            axes[:, column],
            selected,
            method_ids=("ricker_naive", "orthogonal_ricker"),
            tick_labels=tick_labels,
        )
        _panel_title(axes[0, column], chr(ord("a") + column), f"{title}: validation loss")
        _panel_title(
            axes[1, column],
            chr(ord("c") + column),
            rf"{title}: $R_{{\mathrm{{model}}}}$",
        )
        axes[1, column].set_xlabel(sweep)
    for axis in axes[0, :]:
        axis.set_ylim(*_b3_metric_limits(endpoints, "loss"))
    for axis in axes[1, :]:
        axis.set_ylim(*_b3_metric_limits(endpoints, "r_model_pct"))
    axes[0, 0].set_ylabel("Validation loss")
    axes[1, 0].set_ylabel(r"$R_{\mathrm{model}}$ (\%)")
    handles = _method_architecture_legend(
        method_names=(("ricker_naive", "RN"), ("orthogonal_ricker", "OR"))
    )
    figure.legend(
        handles,
        [item.get_label() for item in handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13,
        top=0.84,
        hspace=0.46,
        wspace=0.22,
    )
    return figure


SEED_SENTINEL_LABELS = {
    293: "A0",
    294: "A1-H",
    295: "A3",
    296: "A6-PRE",
    297: "A6-POST",
    298: r"Fixed $G^+$",
    299: r"Fixed $G^\pm$",
    300: r"Learned $G^\pm$",
    301: "L1N",
    302: "OR",
}
SEED_CONTRASTS = (
    ("A1-H - A0", 124, 123),
    ("A3 - A1-H", 125, 124),
    ("A6-PRE - A3", 126, 125),
    ("A6-POST - A3", 127, 125),
    ("POST - PRE", 127, 126),
    (r"$G^+$ - A6-POST", 169, 127),
    (r"$G^\pm$ - A3", 193, 125),
    ("ATG - fixed", 228, 193),
    ("L1N - AdamW", 250, 127),
    ("OR - AdamW", 255, 127),
)


def build_seed_sensitivity_figure(cohort: Sequence[S1Row]) -> Figure:
    """B4 seed endpoints and within-seed contrast agreement."""

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_INCHES, 4.72))
    by_design = _config_index(cohort)
    by_number = _number_index(cohort)
    sentinels = _rows(cohort, block="S1-B4")
    source_to_seed1: dict[int, S1Row] = {}
    endpoint_pairs: list[tuple[S1Row, S1Row]] = []
    for seed1 in sentinels:
        source_id = str(seed1.config["matched_control_id"])
        seed0 = by_design[source_id]
        source_to_seed1[seed0.number] = seed1
        endpoint_pairs.append((seed0, seed1))

    categories = {
        "architecture": "#0072B2",
        "threshold": "#009E73",
        "pressure": "#D55E00",
    }
    category_markers = {
        "architecture": "o",
        "threshold": "s",
        "pressure": "^",
    }

    def annotate(
        axis: Any,
        label: str,
        xy: tuple[float, float],
        offset: tuple[int, int],
        *,
        ha: str = "left",
    ) -> None:
        axis.annotate(
            label,
            xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0,
            ha=ha,
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.45},
            annotation_clip=False,
        )

    def category(number: int) -> str:
        if number <= 297:
            return "architecture"
        if number <= 300:
            return "threshold"
        return "pressure"

    for seed0, seed1 in endpoint_pairs:
        point_category = category(seed1.number)
        color = categories[point_category]
        marker = category_markers[point_category]
        axes[0, 0].scatter(
            seed0.loss,
            seed1.loss,
            color=color,
            marker=marker,
            s=32,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0, 1].scatter(
            seed0.r_model_pct,
            seed1.r_model_pct,
            color=color,
            marker=marker,
            s=32,
            edgecolor="white",
            linewidth=0.4,
        )

    _add_identity(axes[0, 0], [value for pair in endpoint_pairs for row in pair for value in (row.loss,)])
    _add_identity(
        axes[0, 1],
        [value for pair in endpoint_pairs for row in pair for value in (row.r_model_pct,)],
    )

    loss_contrasts = []
    compute_contrasts = []
    for label, child_number, parent_number in SEED_CONTRASTS:
        child0 = by_number[child_number]
        parent0 = by_number[parent_number]
        child1 = source_to_seed1[child_number]
        parent1 = source_to_seed1[parent_number]
        loss0 = child0.loss - parent0.loss
        loss1 = child1.loss - parent1.loss
        compute0 = child0.r_model_pct - parent0.r_model_pct
        compute1 = child1.r_model_pct - parent1.r_model_pct
        loss_contrasts.extend((loss0, loss1))
        compute_contrasts.extend((compute0, compute1))
        point_category = (
            "pressure"
            if child_number in {250, 255}
            else "threshold"
            if child_number in {169, 193, 228}
            else "architecture"
        )
        color = categories[point_category]
        marker = category_markers[point_category]
        axes[1, 0].scatter(
            loss0,
            loss1,
            color=color,
            marker=marker,
            s=32,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1, 1].scatter(
            compute0,
            compute1,
            color=color,
            marker=marker,
            s=32,
            edgecolor="white",
            linewidth=0.4,
        )
        if label == r"$G^\pm$ - A3":
            annotate(
                axes[1, 0],
                r"only sign reversal: $G^\pm-A3$",
                (loss0, loss1),
                (7, -15),
            )

    _add_identity(axes[1, 0], loss_contrasts)
    _add_identity(axes[1, 1], compute_contrasts)
    axes[1, 0].axhline(0.0, color="#999999", linewidth=0.6)
    axes[1, 0].axvline(0.0, color="#999999", linewidth=0.6)
    axes[1, 1].axhline(0.0, color="#999999", linewidth=0.6)
    axes[1, 1].axvline(0.0, color="#999999", linewidth=0.6)

    titles = (
        (0, 0, "a", "Endpoint validation loss"),
        (0, 1, "b", r"Endpoint $R_{\mathrm{model}}$"),
        (1, 0, "c", "Loss: 9/10 signs agree"),
        (1, 1, "d", "Compute: 10/10 signs agree"),
    )
    for row, column, letter, title in titles:
        _panel_title(axes[row, column], letter, title)
        _finish_axis(axes[row, column])
    axes[0, 0].set_xlabel("Seed 0")
    axes[0, 0].set_ylabel("Seed 1")
    axes[0, 1].set_xlabel("Seed 0 (%)")
    axes[0, 1].set_ylabel("Seed 1 (%)")
    axes[1, 0].set_xlabel("Seed-0 effect")
    axes[1, 0].set_ylabel("Seed-1 effect")
    axes[1, 1].set_xlabel("Seed-0 effect (pp)")
    axes[1, 1].set_ylabel("Seed-1 effect (pp)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=category_markers[label],
            linestyle="none",
            color=color,
            label=label.title(),
        )
        for label, color in categories.items()
    ]
    figure.legend(legend_handles, [item.get_label() for item in legend_handles], loc="upper center", ncol=3, frameon=False)
    figure.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.88, hspace=0.40, wspace=0.31)
    return figure


def _latex_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def _latex_percent(value: Any) -> str:
    if value is None:
        return "--"
    percentage = 100.0 * float(value)
    if 0.0 <= percentage < 0.0001:
        return r"$<0.0001$"
    return f"{percentage:.3f}"


def _gate_label(config: Mapping[str, Any]) -> str:
    family = {"none": "none", "gplus": r"G$^+$", "gpm": r"G$^\pm$"}[
        str(config.get("gate_family"))
    ]
    architecture = str(config["architecture_id"]).replace("_", r"\_")
    if family == "none":
        return architecture

    active_sites = tuple(str(site) for site in config.get("gate_sites") or ())
    scope = str(config.get("kappa_scope"))
    fixed_scopes = {
        "q_only": {"q"},
        "k_only": {"k"},
        "v_only": {"v"},
        "qk_only": {"q", "k"},
        "qkv": {"q", "k", "v"},
        "all_active_gates": set(active_sites),
    }
    if config.get("kappa_mode") == "fixed":
        threshold_sites = fixed_scopes.get(scope, set(active_sites))
    else:
        design_id = str(config.get("design_id") or "")
        if "-QKV-" in design_id:
            threshold_sites = {"q", "k", "v"}
        elif "-QK-" in design_id:
            threshold_sites = {"q", "k"}
        elif "-Q-" in design_id:
            threshold_sites = {"q"}
        elif "-K-" in design_id:
            threshold_sites = {"k"}
        elif "-V-" in design_id:
            threshold_sites = {"v"}
        else:
            threshold_sites = set(active_sites)
    threshold_sites &= set(active_sites)
    ordinary_sites = set(active_sites) - threshold_sites

    order = ("a", "m", "h", "q", "k", "v")
    labels = {"a": "a", "m": "m", "h": "h", "q": "Q", "k": "K", "v": "V"}

    def site_text(sites: set[str]) -> str:
        return "/".join(labels[site] for site in order if site in sites)

    parts = [architecture]
    if ordinary_sites:
        parts.append(f"ReLU {site_text(ordinary_sites)}")
    prefix = "ATG " if config.get("kappa_mode") == "learned" else ""
    parts.append(f"{prefix}{family} {site_text(threshold_sites)}")
    return "; ".join(parts)


def _compact_number(value: Any) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if numeric != 0.0 and abs(numeric) < 1e-3:
        mantissa, exponent = f"{numeric:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    text = f"{numeric:.3g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return text


def _setting_label(config: Mapping[str, Any]) -> str:
    method = METHOD_LABELS[str(config.get("pressure_method"))]
    parts = [method]
    if method == "AdamW":
        parts.append(f"lr={_compact_number(config.get('model_learning_rate'))}")
    if config.get("kappa_mode") == "fixed" and config.get("kappa") is not None:
        parts.append(r"$\kappa$=" + _compact_number(config.get("kappa")))
    if config.get("kappa_mode") == "learned":
        parts.append(r"learned $\kappa_0$=" + _compact_number(config.get("kappa_init")))
        scale = "RMS" if config.get("threshold_scale") == "rms_relative" else "ABS"
        scope = {
            "global": "global",
            "per_site": "site",
            "per_layer_site": "layer/site",
        }.get(str(config.get("kappa_scope")), str(config.get("kappa_scope")))
        parts.extend((scale, scope))
    if method != "AdamW":
        parts.append("w=" + _compact_number(config.get("pressure_weight")))
    if method in {"RN", "OR"}:
        parts.append("c=" + _compact_number(config.get("ricker_c")))
        parts.append(r"$\sigma$=" + _compact_number(config.get("ricker_sigma")))
    if method in {"OL1", "OR"}:
        parts.append("budget=" + _compact_number(config.get("step_budget")))
    if config.get("qk_placement"):
        parts.append("PRE" if config["qk_placement"] == "pre_rope" else "POST")
    scope = config.get("kappa_scope")
    if config.get("kappa_mode") == "fixed" and scope not in {None, "all_active_gates"}:
        parts.append(str(scope).replace("_only", "").replace("_", r"\_"))
    return "; ".join(parts)


def write_endpoint_tables(cohort: Sequence[S1Row], output: str | Path) -> Path:
    """Write longtable appendices containing every canonical S1 endpoint."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block_titles = {
        "S1-B0": "B0: architecture and learning-rate endpoints",
        "S1-B1": "B1: fixed-threshold endpoints",
        "S1-B2": "B2: learned-threshold endpoints",
        "S1-B3": "B3: pressure endpoints",
        "S1-B4": "B4: seed-1 sentinel endpoints",
    }
    lines = [
        "% Generated by python -m paper_exp.plot_report07. Do not edit by hand.",
        r"\renewcommand{\arraystretch}{1.00}",
    ]
    for block, title in block_titles.items():
        rows = _rows(cohort, block=block)
        label = block.lower().replace("-", ":")
        if block == "S1-B4":
            lines.append(r"\clearpage")
        lines.extend(
            [
                rf"\subsection{{{title}}}",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{2.0pt}",
                (
                    r"\begin{longtable}{r "
                    r">{\raggedright\arraybackslash}p{1.65in} "
                    r">{\raggedright\arraybackslash}p{2.25in} r r r "
                    r"r r r r r r}"
                ),
                (
                    rf"\caption{{Canonical {block} core scientific endpoints. "
                    r"$R_m$ and all $z$ columns are percentages; "
                    r"$z_Q^g,z_K^g,z_V^g$ are gate-output zeros.}"
                    rf"\label{{tab:{label}-all}}\\"
                ),
                r"\toprule",
                (
                    r"Cfg & Architecture / gate & Setting & Init/data & Loss & $R_m$ (\%) "
                    r"& $z_a$ & $z_m$ & $z_h$ & $z_Q^g$ & $z_K^g$ & $z_V^g$ \\"
                ),
                r"\midrule",
                r"\endfirsthead",
                r"\multicolumn{12}{c}{\tablename\ \thetable\ (continued)}\\",
                r"\toprule",
                (
                    r"Cfg & Architecture / gate & Setting & Init/data & Loss & $R_m$ (\%) "
                    r"& $z_a$ & $z_m$ & $z_h$ & $z_Q^g$ & $z_K^g$ & $z_V^g$ \\"
                ),
                r"\midrule",
                r"\endhead",
                r"\midrule",
                r"\multicolumn{12}{r}{Continued on next page}\\",
                r"\endfoot",
                r"\bottomrule",
                r"\endlastfoot",
            ]
        )
        for row in rows:
            values = (
                str(row.number),
                _gate_label(row.config),
                _setting_label(row.config),
                (
                    f"{row.config['model_initialization_seed']}/"
                    f"{row.config['data_order_seed']}"
                ),
                _latex_float(row.run["validation_loss"], digits=5),
                _latex_percent(row.run["r_model"]),
                _latex_percent(row.run.get("z_a")),
                _latex_percent(row.run.get("z_m")),
                _latex_percent(row.run.get("z_h")),
                _latex_percent(row.run.get("z_q_gate")),
                _latex_percent(row.run.get("z_k_gate")),
                _latex_percent(row.run.get("z_v_gate")),
            )
            lines.append(" & ".join(values) + r" \\")
        lines.extend((r"\end{longtable}", ""))
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_key(path: Path, repo_root: Path) -> str:
    """Return a portable repository-relative provenance key."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_provenance(
    cohort: Sequence[S1Row],
    *,
    config_registry: Path,
    run_registry: Path,
    claim_sources: Sequence[Path],
    propagation_artifacts: Sequence[Path],
    generated_files: Sequence[Path],
    output: Path,
) -> Path:
    """Write deterministic source and output hashes for Report 07."""

    repo_root = config_registry.resolve().parents[2]
    payload = {
        "schema_version": 1,
        "report": "07-2026-07-27-s1-ablation-study",
        "scientific_census": EXPECTED_S1_COUNTS,
        "validation_tokens_per_endpoint": 311_296,
        "sources": {
            _provenance_key(config_registry, repo_root): _sha256(config_registry),
            _provenance_key(run_registry, repo_root): _sha256(run_registry),
            **{
                _provenance_key(path, repo_root): _sha256(path)
                for path in sorted(claim_sources, key=lambda item: str(item))
            },
        },
        "raw_propagation_artifacts": {
            _provenance_key(path, repo_root): _sha256(path)
            for path in sorted(propagation_artifacts, key=lambda item: str(item))
        },
        "canonical_rows": [
            {
                "config_id": row.config["config_id"],
                "run_id": row.run["run_id"],
                "block": row.block,
            }
            for row in cohort
        ],
        "generated_files": {
            _provenance_key(path, repo_root): _sha256(path)
            for path in sorted(generated_files, key=lambda item: str(item))
        },
        "interpretation": (
            "R_model is a model-wide logical exact-zero product opportunity, "
            "not a measured sparse-kernel speedup."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return output


def generate_report07_artifacts(
    *,
    config_registry: str | Path,
    run_registry: str | Path,
    figures_dir: str | Path,
    report_dir: str | Path,
    save_png: bool = False,
) -> tuple[Path, ...]:
    """Generate all Report 07 figures, appendix tables, and provenance."""

    config_path = Path(config_registry)
    run_path = Path(run_registry)
    figure_path = Path(figures_dir)
    report_path = Path(report_dir)
    cohort = load_s1_rows(config_path, run_path)
    builders = {
        builder.__name__: builder
        for builder in (
            build_architecture_learning_rate_figure,
            build_fixed_threshold_effect_figure,
            build_gate_type_effect_figure,
            build_learned_threshold_figure,
            build_pressure_weight_figure,
            build_ricker_shape_figure,
            build_seed_sensitivity_figure,
        )
    }
    catalog_wrappers = {entry.public_wrapper for entry in REPORT07_FIGURES}
    if set(builders) != catalog_wrappers:
        raise RuntimeError(
            "Report 07 catalog/renderers differ: "
            f"catalog={sorted(catalog_wrappers)!r}, renderers={sorted(builders)!r}."
        )

    generated: list[Path] = []
    for entry in REPORT07_FIGURES:
        builder = builders[entry.public_wrapper]
        output = figure_path / entry.filename
        generated.extend(
            export_figure(
                lambda builder=builder: builder(cohort),
                output,
                save_png=save_png,
                style=REPORT04_PLOT_STYLE,
                profile=REPORT07_PROFILE,
            )
        )

    tables = write_endpoint_tables(
        cohort,
        report_path / "07-2026-07-27-s1-ablation-study-tables.tex",
    )
    generated.append(tables)
    b0_table = write_b0_endpoint_table(
        cohort,
        report_path / "07-2026-07-27-s1-b0-endpoints.tex",
    )
    generated.append(b0_table)
    b1_table = write_b1_endpoint_table(
        cohort,
        report_path / "07-2026-07-27-s1-b1-endpoints.tex",
    )
    generated.append(b1_table)
    b2_table = write_b2_endpoint_table(
        cohort,
        report_path / "07-2026-07-27-s1-b2-endpoints.tex",
    )
    generated.append(b2_table)
    repo_root = config_path.resolve().parents[2]
    b2_kappa_table = write_b2_kappa_distribution_table(
        cohort,
        report_path / "07-2026-07-27-s1-b2-kappa-distribution.tex",
        repo_root=repo_root,
    )
    generated.append(b2_kappa_table)
    b3_weight_table = write_b3_weight_table(
        cohort,
        report_path / "07-2026-07-27-s1-b3-weight-endpoints.tex",
    )
    generated.append(b3_weight_table)
    b3_geometry_table = write_b3_geometry_table(
        cohort,
        report_path / "07-2026-07-27-s1-b3-geometry-endpoints.tex",
    )
    generated.append(b3_geometry_table)
    claim_sources = [
        Path(__file__).resolve(),
        repo_root / "src" / "paper_exp" / "plot_topology_atlas.py",
        repo_root / "src" / "paper_exp" / "plot_b0_learning_rate_effect.py",
        repo_root / "src" / "paper_exp" / "plot_s1_quality_compute_landscape.py",
        repo_root
        / "src"
        / "paper_exp"
        / "plot_s1_fixed_threshold_architecture_tradeoffs.py",
        repo_root / "src" / "paper_exp" / "plot_s1_pressure_frontiers.py",
        report_path / "07-2026-07-27-s1-ablation-study.tex",
        repo_root / "docs" / "methods.md",
        repo_root / "docs" / "experimental-design" / "01-screening-matrix.md",
        *sorted(
            (repo_root / "docs" / "experimental-design").glob(
                "[01][0-9]-s1-*.md"
            )
        ),
        repo_root / "docs" / "experimental-design" / "20-s1-executable-core-closure.md",
        *[
            (
                Path(str(row.run["result_path"]))
                if Path(str(row.run["result_path"])).is_absolute()
                else repo_root / str(row.run["result_path"])
            )
            / "metrics.json"
            for row in _rows(cohort, block="S1-B2")
        ],
    ]
    claim_sources = [path for path in claim_sources if path.is_file()]
    propagation_artifacts = sorted(
        {
            (
                Path(str(row.run["propagation_result_path"]))
                if Path(str(row.run["propagation_result_path"])).is_absolute()
                else repo_root / str(row.run["propagation_result_path"])
            )
            for row in cohort
        },
        key=lambda item: str(item),
    )
    supplemental_generated = [
        report_path / "110-pythia-14m-s1-topology-atlas.pdf",
        report_path / "110-pythia-14m-s1-topology-atlas.png",
        figure_path / "112-pythia-14m-s1-learning-rate-effect.pdf",
        figure_path / "112-pythia-14m-s1-learning-rate-effect.png",
        figure_path / "113-pythia-14m-s1-quality-compute-endpoint-landscape.pdf",
        figure_path / "113-pythia-14m-s1-quality-compute-endpoint-landscape.png",
        report_path / "07-2026-07-27-s1-frontier-table.tex",
        figure_path / "114-pythia-14m-s1-fixed-threshold-architecture-tradeoffs.pdf",
        figure_path / "114-pythia-14m-s1-fixed-threshold-architecture-tradeoffs.png",
        figure_path
        / "116-pythia-14m-s1-fixed-threshold-quality-opportunity-frontiers.pdf",
        figure_path
        / "116-pythia-14m-s1-fixed-threshold-quality-opportunity-frontiers.png",
        figure_path / "118-pythia-14m-s1-pressure-quality-opportunity-frontiers.pdf",
        figure_path / "118-pythia-14m-s1-pressure-quality-opportunity-frontiers.png",
    ]
    generated.extend(path for path in supplemental_generated if path.is_file())
    provenance = write_provenance(
        cohort,
        config_registry=config_path,
        run_registry=run_path,
        claim_sources=claim_sources,
        propagation_artifacts=propagation_artifacts,
        generated_files=generated,
        output=report_path / "07-2026-07-27-s1-ablation-study-provenance.json",
    )
    generated.append(provenance)
    return tuple(generated)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-registry",
        default="docs/experimental-design/config-registry.yaml",
    )
    parser.add_argument(
        "--run-registry",
        default="docs/experimental-design/run-registry.yaml",
    )
    parser.add_argument("--figures", default="figures")
    parser.add_argument(
        "--report-dir",
        default="report/07-2026-07-27-s1-ablation-study",
    )
    parser.add_argument("--png", action="store_true", help="Also export PNG previews.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = generate_report07_artifacts(
        config_registry=args.config_registry,
        run_registry=args.run_registry,
        figures_dir=args.figures,
        report_dir=args.report_dir,
        save_png=args.png,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
