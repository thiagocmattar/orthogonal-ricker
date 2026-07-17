"""Searchable metadata for the Report 04 and Report 05 paper-figure suites.

The catalog describes figure ownership and saved-input requirements only.  It
does not select runs, load artifacts, or render figures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlotCatalogEntry:
    """One stable paper-figure entry."""

    number: int
    filename: str
    plot_type: str
    required_artifact_kinds: tuple[str, ...]
    public_wrapper: str
    embedded_in_report: bool


REPORT04_FIGURES = (
    PlotCatalogEntry(
        79,
        "79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf",
        "learning_diagnostics",
        ("events.jsonl",),
        "generate_report04_learning_diagnostics",
        True,
    ),
    PlotCatalogEntry(
        80,
        "80-pythia-14m-minipile-post-layernorm-relu-activation-heatmaps.pdf",
        "activation_heatmap",
        ("activation_histograms.json",),
        "generate_report04_activation_heatmaps",
        True,
    ),
    PlotCatalogEntry(
        81,
        "81-pythia-14m-minipile-post-layernorm-relu-activation-densities.pdf",
        "activation_density",
        ("activation_histograms.json",),
        "generate_report04_activation_densities",
        False,
    ),
    PlotCatalogEntry(
        82,
        "82-pythia-14m-minipile-post-layernorm-relu-site-clipping-frontiers.pdf",
        "site_clipping_frontier",
        ("clipping_frontier.jsonl",),
        "generate_report04_site_clipping_frontiers",
        True,
    ),
    PlotCatalogEntry(
        83,
        "83-pythia-14m-minipile-post-layernorm-relu-joint-compute-frontier.pdf",
        "joint_compute_frontier",
        ("clipping_frontier.jsonl",),
        "generate_report04_joint_compute_frontier",
        True,
    ),
    PlotCatalogEntry(
        84,
        "84-pythia-14m-minipile-post-layernorm-relu-parameter-diagnostics.pdf",
        "parameter_diagnostic",
        ("checkpoints/final/model.safetensors",),
        "generate_report04_parameter_diagnostics",
        False,
    ),
    PlotCatalogEntry(
        85,
        "85-pythia-14m-minipile-post-layernorm-relu-zero-propagation-heatmaps.pdf",
        "activation_propagation_heatmap",
        ("activation_propagation.json",),
        "generate_post_layernorm_relu_propagation_heatmaps",
        True,
    ),
    PlotCatalogEntry(
        86,
        "86-pythia-14m-minipile-post-layernorm-relu-zero-product-propagation-heatmaps.pdf",
        "zero_product_heatmap",
        ("activation_propagation.json",),
        "generate_post_layernorm_relu_zero_product_heatmaps",
        True,
    ),
    PlotCatalogEntry(
        87,
        "87-pythia-14m-minipile-three-relu-architecture-compute-map.pdf",
        "architecture_diagram",
        (),
        "generate_report04_three_relu_architecture",
        True,
    ),
    PlotCatalogEntry(
        88,
        "88-pythia-14m-minipile-post-layernorm-relu-activation-weight-densities.pdf",
        "activation_weight_density",
        ("activation_histograms.json", "checkpoints/final/model.safetensors"),
        "generate_report04_activation_weight_densities",
        True,
    ),
    PlotCatalogEntry(
        89,
        "89-pythia-14m-minipile-post-layernorm-relu-layernorm-parameters.pdf",
        "layernorm_parameter_diagnostic",
        ("checkpoints/final/model.safetensors",),
        "generate_report04_layernorm_parameters",
        True,
    ),
    PlotCatalogEntry(
        90,
        "90-pythia-family-three-relu-model-compute-ceilings.pdf",
        "compute_ceiling",
        (),
        "generate_report04_pythia_family_compute_ceiling",
        True,
    ),
)


REPORT05_FIGURES = (
    PlotCatalogEntry(
        91,
        "91-pythia-14m-minipile-relu-architecture-ladder.pdf",
        "architecture_diagram",
        (),
        "generate_report05_architecture_schematic",
        True,
    ),
    PlotCatalogEntry(
        92,
        "92-pythia-14m-minipile-relu-architecture-learning-curves.pdf",
        "learning_curves",
        ("events.jsonl",),
        "generate_report05_learning_curves",
        True,
    ),
    *(
        PlotCatalogEntry(
            number,
            f"{number}-pythia-14m-minipile-{slug}-zero-propagation.pdf",
            f"{architecture_id}_activation_propagation_heatmap",
            ("activation_propagation.json",),
            f"generate_report05_{architecture_id}_propagation",
            True,
        )
        for number, architecture_id, slug in (
            (93, "one_relu", "one-relu"),
            (94, "three_relu", "three-relu"),
            (95, "six_relu_pre", "six-relu-pre"),
            (96, "six_relu_post", "six-relu-post"),
        )
    ),
    *(
        PlotCatalogEntry(
            number,
            f"{number}-pythia-14m-minipile-{slug}-activation-weight-densities.pdf",
            f"{architecture_id}_activation_weight_density",
            ("activation_histograms.json", "checkpoints/final/model.safetensors"),
            f"generate_report05_{architecture_id}_distributions",
            True,
        )
        for number, architecture_id, slug in (
            (97, "one_relu", "one-relu"),
            (98, "three_relu", "three-relu"),
            (99, "six_relu_pre", "six-relu-pre"),
            (100, "six_relu_post", "six-relu-post"),
        )
    ),
    PlotCatalogEntry(
        101,
        "101-pythia-14m-minipile-relu-architecture-site-clipping-frontiers.pdf",
        "site_clipping_frontier",
        ("clipping_frontier.jsonl",),
        "generate_report05_site_clipping_frontiers",
        True,
    ),
    PlotCatalogEntry(
        102,
        "102-pythia-14m-minipile-relu-architecture-model-compute-frontiers.pdf",
        "logical_compute_frontier",
        ("clipping_frontier.jsonl",),
        "generate_report05_model_compute_frontiers",
        True,
    ),
)


def list_report04_figures(*, embedded_only: bool = False) -> tuple[PlotCatalogEntry, ...]:
    """Return Report 04 entries in stable figure-number order."""

    if not embedded_only:
        return REPORT04_FIGURES
    return tuple(entry for entry in REPORT04_FIGURES if entry.embedded_in_report)


def get_report04_figure(identifier: int | str) -> PlotCatalogEntry:
    """Look up a figure by number, filename, plot type, or public wrapper."""

    for entry in REPORT04_FIGURES:
        if identifier in {
            entry.number,
            entry.filename,
            entry.plot_type,
            entry.public_wrapper,
        }:
            return entry
    raise KeyError(f"Unknown Report 04 figure: {identifier!r}")


def report04_catalog_rows(*, embedded_only: bool = False) -> tuple[str, ...]:
    """Return deterministic, human-readable rows for navigation or documentation."""

    rows = []
    for entry in list_report04_figures(embedded_only=embedded_only):
        artifacts = ", ".join(entry.required_artifact_kinds) or "none"
        report_status = "embedded" if entry.embedded_in_report else "generated only"
        rows.append(
            f"{entry.number} | {entry.plot_type} | {entry.filename} | "
            f"artifacts: {artifacts} | wrapper: {entry.public_wrapper} | {report_status}"
        )
    return tuple(rows)


def list_report05_figures(*, embedded_only: bool = False) -> tuple[PlotCatalogEntry, ...]:
    """Return Report 05 entries in stable figure-number order."""

    if not embedded_only:
        return REPORT05_FIGURES
    return tuple(entry for entry in REPORT05_FIGURES if entry.embedded_in_report)


def get_report05_figure(identifier: int | str) -> PlotCatalogEntry:
    """Look up a Report 05 figure by number, filename, type, or wrapper."""

    for entry in REPORT05_FIGURES:
        if identifier in {
            entry.number,
            entry.filename,
            entry.plot_type,
            entry.public_wrapper,
        }:
            return entry
    raise KeyError(f"Unknown Report 05 figure: {identifier!r}")


def report05_catalog_rows(*, embedded_only: bool = False) -> tuple[str, ...]:
    """Return deterministic, human-readable Report 05 catalog rows."""

    rows = []
    for entry in list_report05_figures(embedded_only=embedded_only):
        artifacts = ", ".join(entry.required_artifact_kinds) or "none"
        report_status = "embedded" if entry.embedded_in_report else "generated only"
        rows.append(
            f"{entry.number} | {entry.plot_type} | {entry.filename} | "
            f"artifacts: {artifacts} | wrapper: {entry.public_wrapper} | {report_status}"
        )
    return tuple(rows)
