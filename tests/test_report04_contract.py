from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

import paper_exp.plots as plots
from paper_exp import plot_report04
from paper_exp.plot_api import REPORT04_PUBLICATION_PROFILE, publication_figure_issues
from paper_exp.plot_style import REPORT04_PLOT_STYLE
from paper_exp.plot_catalog import REPORT04_FIGURES


REPORT04_RENDERERS = (
    "generate_report04_learning_diagnostics",
    "generate_report04_activation_heatmaps",
    "generate_report04_activation_densities",
    "generate_report04_site_clipping_frontiers",
    "generate_report04_joint_compute_frontier",
    "generate_report04_parameter_diagnostics",
    "generate_report04_layernorm_parameters",
    "generate_report04_activation_weight_densities",
    "generate_post_layernorm_relu_propagation_heatmaps",
    "generate_post_layernorm_relu_zero_product_heatmaps",
    "generate_report04_three_relu_architecture",
    "generate_report04_pythia_family_compute_ceiling",
)

REPORT04_OUTPUT_NAMES = (
    "79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf",
    "80-pythia-14m-minipile-post-layernorm-relu-activation-heatmaps.pdf",
    "81-pythia-14m-minipile-post-layernorm-relu-activation-densities.pdf",
    "82-pythia-14m-minipile-post-layernorm-relu-site-clipping-frontiers.pdf",
    "83-pythia-14m-minipile-post-layernorm-relu-joint-compute-frontier.pdf",
    "84-pythia-14m-minipile-post-layernorm-relu-parameter-diagnostics.pdf",
    "89-pythia-14m-minipile-post-layernorm-relu-layernorm-parameters.pdf",
    "88-pythia-14m-minipile-post-layernorm-relu-activation-weight-densities.pdf",
    "85-pythia-14m-minipile-post-layernorm-relu-zero-propagation-heatmaps.pdf",
    "86-pythia-14m-minipile-post-layernorm-relu-zero-product-propagation-heatmaps.pdf",
    "87-pythia-14m-minipile-three-relu-architecture-compute-map.pdf",
    "90-pythia-family-three-relu-model-compute-ceilings.pdf",
)


@pytest.mark.parametrize(
    ("generator", "stem"),
    [
        (plots.generate_report04_three_relu_architecture, "architecture"),
        (plots.generate_report04_pythia_family_compute_ceiling, "compute-ceiling"),
    ],
)
def test_report04_architecture_renderers_write_pdf_and_png(
    tmp_path: Path,
    generator: Any,
    stem: str,
) -> None:
    pdf_path = tmp_path / f"{stem}.pdf"

    outputs = generator(output=pdf_path, save_png=True)

    assert outputs == [pdf_path, pdf_path.with_suffix(".png")]
    assert all(path.stat().st_size > 1_000 for path in outputs)


def test_report04_wrapper_builds_once_for_pdf_and_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0

    def build_figure() -> Any:
        nonlocal build_calls
        build_calls += 1
        figure, axis = plt.subplots(figsize=(7.16, 3.0))
        axis.set_axis_off()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        return figure

    monkeypatch.setattr(plots, "_plot_report04_three_relu_architecture", build_figure)
    pdf_path = tmp_path / "architecture.pdf"

    outputs = plots.generate_report04_three_relu_architecture(
        output=pdf_path,
        save_png=True,
    )

    assert outputs == [pdf_path, pdf_path.with_suffix(".png")]
    assert build_calls == 1


def test_report04_multirow_legends_preserve_visual_method_order() -> None:
    handles = list("ABCDEFG")
    labels = [f"Method {handle}" for handle in handles]

    ordered_handles, ordered_labels = (
        plot_report04._legend_items_in_row_major_order(handles, labels, 4)
    )

    assert ordered_handles == list("AEBFCGD")
    assert ordered_labels == [
        "Method A",
        "Method E",
        "Method B",
        "Method F",
        "Method C",
        "Method G",
        "Method D",
    ]


def test_report04_compute_ceiling_uses_true_percent_axis_and_contained_labels() -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report04._plot_report04_pythia_family_compute_ceiling()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[0]
        axis_box = axis.get_window_extent(renderer)

        assert axis.get_xlim() == pytest.approx((0.0, 100.0))
        assert all(
            text.get_window_extent(renderer).x0 >= axis_box.x0 - 1.0
            and text.get_window_extent(renderer).x1 <= axis_box.x1 + 1.0
            for text in axis.texts
            if text.get_visible()
        )
        assert any("LM-head share declines" in text.get_text() for text in figure.texts)
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def test_report04_joint_frontier_legends_stay_inside_the_figure() -> None:
    series = [
        {
            "label": label,
            "rows": [
                {
                    "threshold": 0.0,
                    "eligible_projection_skip_fraction": 0.45 + 0.05 * index,
                    "validation_loss": 5.0 + 0.1 * index,
                    "validation_tokens": 20,
                },
                {
                    "threshold": 0.01,
                    "eligible_projection_skip_fraction": 0.50 + 0.05 * index,
                    "validation_loss": 5.2 + 0.1 * index,
                    "validation_tokens": 20,
                },
            ],
        }
        for index, (label, _experiment_id) in enumerate(
            plot_report04.REPORT04_JOINT_CLIPPING_RUNS
        )
    ]
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report04._plot_report04_joint_compute_frontier(series)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        figure_box = figure.bbox

        assert len(figure.legends) == 2
        for legend in figure.legends:
            legend_box = legend.get_window_extent(renderer)
            assert legend_box.x0 >= figure_box.x0 - 1.0
            assert legend_box.x1 <= figure_box.x1 + 1.0
            assert legend_box.y0 >= figure_box.y0 - 1.0
            assert legend_box.y1 <= figure_box.y1 + 1.0
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def test_report04_activation_heatmap_keeps_tiny_positive_cells_distinct() -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = plot_report04._plot_report04_activation_heatmaps(
            _synthetic_activation_heatmap_payloads()
        )
    try:
        panel_axes = figure.axes[:6]
        cell_texts = [text for axis in panel_axes for text in axis.texts]

        assert "0" in {text.get_text() for text in cell_texts}
        assert "<.1" in {text.get_text() for text in cell_texts}
        assert all(text.get_fontsize() >= 8.5 for text in cell_texts)
        assert all(
            label.get_fontsize() >= 8.5
            for axis in panel_axes[::2]
            for label in axis.get_yticklabels()
        )
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("renderer", "message"),
    [
        (
            plot_report04._plot_post_layernorm_relu_propagation_heatmaps,
            "Activation-propagation heatmaps require at least one matched checkpoint.",
        ),
        (
            plot_report04._plot_post_layernorm_relu_zero_product_heatmaps,
            "Zero-product heatmaps require at least one matched checkpoint.",
        ),
    ],
)
def test_report04_propagation_layout_rejects_empty_method_payloads(
    renderer: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        renderer({"methods": []})


@pytest.mark.parametrize("method_count", [1, 2, 4, 5, 7])
@pytest.mark.parametrize(
    "renderer",
    [
        plot_report04._plot_post_layernorm_relu_propagation_heatmaps,
        plot_report04._plot_post_layernorm_relu_zero_product_heatmaps,
    ],
)
def test_report04_propagation_layout_tracks_method_cardinality(
    renderer: Any,
    method_count: int,
) -> None:
    with plt.rc_context(REPORT04_PLOT_STYLE):
        figure = renderer(_synthetic_propagation_payload(method_count))
    try:
        column_count = min(2, method_count)
        grid_cell_count = math.ceil(method_count / column_count) * column_count
        panel_axes = figure.axes[:-1]
        colorbar_axis = figure.axes[-1]

        assert len(panel_axes) == grid_cell_count
        assert sum(axis.get_visible() for axis in panel_axes) == method_count
        assert sum(not axis.get_visible() for axis in panel_axes) == grid_cell_count - method_count
        assert colorbar_axis.get_visible()
        assert colorbar_axis.get_ylabel() in {
            "Exact-zero scalar fraction (%)",
            "Products with an exact-zero activation operand (%)",
        }
        assert all(not axis.get_xlabel() for axis in panel_axes)
        assert figure._supxlabel is not None
        assert figure._supxlabel.get_fontsize() >= 8.0
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.8
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        (0.0, "0"),
        (1e-12, "<.1"),
        (0.099, "<.1"),
        (0.1, "0.1"),
        (47.94, "47.9"),
    ],
)
def test_report04_propagation_cell_labels_distinguish_zero_from_display_bound(
    percent: float,
    expected: str,
) -> None:
    assert plot_report04._propagation_cell_label(percent) == expected


def test_strict_report04_suite_dispatches_every_output_in_current_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    _write_report04_cohort(results_dir)
    calls = _patch_plot_renderers(monkeypatch)

    outputs = plots.generate_report04_figures(
        results_dir,
        tmp_path / "figures",
        save_png=False,
    )

    assert [path.name for path in outputs] == list(REPORT04_OUTPUT_NAMES)
    assert [name for name, _kwargs in calls] == list(REPORT04_RENDERERS)

    call_by_renderer = {name: kwargs for name, kwargs in calls}
    assert (
        call_by_renderer["generate_report04_learning_diagnostics"]["training_specs"]
        == plots.REPORT04_TRAINING_RUNS
    )
    assert [label for label, _run in call_by_renderer["generate_report04_learning_diagnostics"]["runs"]] == [
        label for label, _experiment_id in plots.REPORT04_TRAINING_RUNS
    ]
    assert [
        label
        for label, _run in call_by_renderer["generate_report04_joint_compute_frontier"]["runs"]
    ] == [label for label, _experiment_id in plots.REPORT04_JOINT_CLIPPING_RUNS]
    assert list(call_by_renderer["generate_report04_site_clipping_frontiers"]["site_runs"]) == [
        site for site, _site_label in plots.REPORT04_CLIPPING_SITES
    ]


def test_rn_report04_cohort_is_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    figures_dir = tmp_path / "figures"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir, include_rn=True)
    rn_index = [
        label for label, _experiment_id in plots.REPORT04_RN_TRAINING_RUNS
    ].index("Three-ReLU RN")
    rn_manifest_path = artifacts["checkpoint"][rn_index].parents[2] / "manifest.json"
    rn_manifest = json.loads(rn_manifest_path.read_text(encoding="utf-8"))
    rn_manifest["git_dirty"] = True
    _write_json(rn_manifest_path, rn_manifest)
    calls = _patch_plot_renderers(monkeypatch)

    plots.generate_report04_figures(
        results_dir,
        figures_dir,
        include_rn=True,
        write_provenance=True,
    )

    call_by_renderer = {name: kwargs for name, kwargs in calls}
    learning_call = call_by_renderer["generate_report04_learning_diagnostics"]
    assert learning_call["training_specs"] == plots.REPORT04_RN_TRAINING_RUNS
    assert [label for label, _run in learning_call["runs"]] == [
        label for label, _experiment_id in plots.REPORT04_RN_TRAINING_RUNS
    ]
    propagation_call = call_by_renderer[
        "generate_post_layernorm_relu_propagation_heatmaps"
    ]
    assert Path(propagation_call["run_dir"]).parent.name == (
        plots.REPORT04_RN_PROPAGATION_EXPERIMENT
    )

    payload = json.loads((figures_dir / "report04-provenance.json").read_text())
    assert payload["cohort"] == "rn-comparison"
    figures = {figure["number"]: figure for figure in payload["figures"]}
    rn_input = next(
        item for item in figures[84]["inputs"] if item["label"] == "Three-ReLU RN"
    )
    assert rn_input["run_manifest"]["git_dirty"] is True
    assert "Three-ReLU RN" not in {item["label"] for item in figures[88]["inputs"]}


def test_figure88_uses_histogram_source_checkpoint_not_latest_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    label, experiment_id = plots.REPORT04_TRAINING_RUNS[0]
    source_run = artifacts["checkpoint"][0].parents[2]
    latest_run = _write_complete_run(
        results_dir,
        experiment_id,
        "events.jsonl",
        "checkpoints/final/model.safetensors",
        run_id="002",
    )
    calls = _patch_plot_renderers(monkeypatch)
    figures_dir = tmp_path / "figures"

    plots.generate_report04_figures(
        results_dir,
        figures_dir,
        write_provenance=True,
    )

    call_by_renderer = {name: kwargs for name, kwargs in calls}
    parameter_runs = dict(
        call_by_renderer["generate_report04_parameter_diagnostics"]["runs"]
    )
    histogram_checkpoint_runs = dict(
        call_by_renderer["generate_report04_activation_weight_densities"]["runs"]
    )
    assert parameter_runs[label] == latest_run
    assert histogram_checkpoint_runs[label] == source_run

    payload = json.loads((figures_dir / "report04-provenance.json").read_text())
    figures = {figure["number"]: figure for figure in payload["figures"]}
    figure84_input = next(
        item for item in figures[84]["inputs"] if item["label"] == label
    )
    figure88_input = next(
        item for item in figures[88]["inputs"] if item["label"] == label
    )
    assert figure84_input["run_id"] == latest_run.name
    assert figure88_input["run_id"] == source_run.name
    assert figure88_input["role"] == "histogram_source_checkpoint"


def test_strict_report04_rejects_disagreeing_histogram_source_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    hidden_histogram_path = artifacts["histogram"][1]
    hidden_histogram = json.loads(hidden_histogram_path.read_text(encoding="utf-8"))
    hidden_histogram["methods"][0]["run_id"] = "002"
    _write_json(hidden_histogram_path, hidden_histogram)
    calls = _patch_plot_renderers(monkeypatch)

    with pytest.raises(plots.Report04InputError) as error:
        plots.generate_report04_figures(results_dir, tmp_path / "figures")

    message = str(error.value)
    assert "histogram_matched_checkpoints (figure 88)" in message
    assert "histogram sources disagree for GELU AdamW" in message
    assert calls == []


def test_newer_checkpoint_does_not_mask_missing_figure88_source_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    label, experiment_id = plots.REPORT04_TRAINING_RUNS[0]
    source_checkpoint = artifacts["checkpoint"][0]
    _write_complete_run(
        results_dir,
        experiment_id,
        "events.jsonl",
        "checkpoints/final/model.safetensors",
        run_id="002",
    )
    source_checkpoint.unlink()
    calls = _patch_plot_renderers(monkeypatch)

    with pytest.raises(plots.Report04InputError) as error:
        plots.generate_report04_figures(results_dir, tmp_path / "figures")

    message = str(error.value)
    assert "histogram_matched_checkpoints (figure 88)" in message
    assert f"missing coherent source checkpoint for {label}" in message
    assert calls == []


def test_strict_report04_provenance_is_complete_relative_and_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    figures_dir = tmp_path / "figures"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    _patch_plot_renderers(monkeypatch)

    first_outputs = plots.generate_report04_figures(
        results_dir,
        figures_dir,
        write_provenance=True,
    )
    provenance_path = figures_dir / "report04-provenance.json"
    first_bytes = provenance_path.read_bytes()
    second_outputs = plots.generate_report04_figures(
        results_dir,
        figures_dir,
        write_provenance=True,
    )

    assert first_outputs[-1] == provenance_path
    assert second_outputs[-1] == provenance_path
    assert provenance_path.read_bytes() == first_bytes
    assert first_bytes.endswith(b"\n") and b"\r\n" not in first_bytes

    payload = json.loads(first_bytes)
    assert payload["schema_version"] == 3
    assert payload["suite"] == "report04"
    assert payload["cohort"] == "published-pre-rn"
    assert [figure["number"] for figure in payload["figures"]] == list(range(79, 91))
    assert [
        (
            figure["number"],
            figure["filename"],
            figure["plot_type"],
            tuple(figure["required_artifact_kinds"]),
            figure["public_wrapper"],
            figure["embedded_in_report"],
        )
        for figure in payload["figures"]
    ] == [
        (
            entry.number,
            entry.filename,
            entry.plot_type,
            entry.required_artifact_kinds,
            entry.public_wrapper,
            entry.embedded_in_report,
        )
        for entry in REPORT04_FIGURES
    ]
    figures = {figure["number"]: figure for figure in payload["figures"]}
    for entry in REPORT04_FIGURES:
        output_path = figures_dir / entry.filename
        assert figures[entry.number]["outputs"] == [
            {
                "filename": entry.filename,
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "size_bytes": output_path.stat().st_size,
            }
        ]
    assert figures[87]["inputs"] == []
    assert figures[90]["inputs"] == []
    assert [item["artifact"] for item in figures[88]["inputs"]] == [
        "activation_histograms.json",
        "activation_histograms.json",
        *(
            ["checkpoints/final/model.safetensors"]
            * len(plot_report04.REPORT04_HISTOGRAM_METHOD_LABELS)
        ),
    ]
    assert [item["label"] for item in figures[88]["inputs"][2:]] == list(
        plot_report04.REPORT04_HISTOGRAM_METHOD_LABELS
    )

    first_event = artifacts["training_event"][0]
    first_input = figures[79]["inputs"][0]
    assert first_input["run_dir"] == first_event.parent.relative_to(results_dir).as_posix()
    assert first_input["artifact_path"] == first_event.relative_to(results_dir).as_posix()
    assert first_input["sha256"] == hashlib.sha256(first_event.read_bytes()).hexdigest()
    config_path = first_event.parent / "config.yaml"
    manifest_path = first_event.parent / "manifest.json"
    assert first_input["run_config"] == {
        "artifact_path": config_path.relative_to(results_dir).as_posix(),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    assert first_input["run_manifest"] == {
        "artifact_path": manifest_path.relative_to(results_dir).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": False,
    }
    provenance_text = first_bytes.decode("utf-8")
    assert str(results_dir.resolve()) not in provenance_text
    assert '"timestamp"' not in provenance_text


def test_strict_report04_preflight_aggregates_missing_inputs_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    figures_dir = tmp_path / "figures"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    for artifact_group in artifacts.values():
        artifact_group[-1].unlink()
    calls = _patch_plot_renderers(monkeypatch)

    with pytest.raises(plots.Report04InputError) as first_error:
        plots.generate_report04_figures(results_dir, figures_dir, write_provenance=True)
    with pytest.raises(plots.Report04InputError) as second_error:
        plots.generate_report04_figures(results_dir, figures_dir, write_provenance=True)

    message = str(first_error.value)
    assert str(second_error.value) == message
    issue_names = [
        "training_events (figure 79)",
        "activation_histograms (figures 80, 81, 88)",
        "site_clipping_frontiers (figure 82)",
        "joint_clipping_frontiers (figure 83)",
        "final_checkpoints (figures 84, 89)",
        "histogram_matched_checkpoints (figure 88)",
        "activation_propagation (figures 85, 86)",
    ]
    assert [message.index(issue_name) for issue_name in issue_names] == sorted(
        message.index(issue_name) for issue_name in issue_names
    )
    assert calls == []
    assert not figures_dir.exists()
    assert not (figures_dir / "report04-provenance.json").exists()


def test_strict_report04_renderer_failure_leaves_existing_suite_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    _shorten_clipping_experiment_ids(monkeypatch)
    _write_report04_cohort(results_dir)
    _patch_plot_renderers(monkeypatch)
    first_output = figures_dir / REPORT04_OUTPUT_NAMES[0]
    second_output = figures_dir / REPORT04_OUTPUT_NAMES[1]
    first_output.write_bytes(b"old first")
    second_output.write_bytes(b"old second")

    def fail_renderer(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise OSError("renderer failed")

    monkeypatch.setattr(plots, "generate_report04_activation_heatmaps", fail_renderer)

    with pytest.raises(OSError, match="renderer failed"):
        plots.generate_report04_figures(results_dir, figures_dir)

    assert first_output.read_bytes() == b"old first"
    assert second_output.read_bytes() == b"old second"
    assert not list(tmp_path.glob(".report04-stage-*"))
    assert not (figures_dir / "report04-provenance.json").exists()


def test_rn_checkpoint_is_not_claimed_as_a_figure88_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir, include_rn=True)
    training_labels = [label for label, _experiment_id in plots.REPORT04_RN_TRAINING_RUNS]
    rn_index = training_labels.index("Three-ReLU RN")
    artifacts["checkpoint"][rn_index].unlink()
    _patch_plot_renderers(monkeypatch)

    with pytest.raises(plots.Report04InputError) as error:
        plots.generate_report04_figures(
            results_dir,
            tmp_path / "figures",
            include_rn=True,
        )

    message = str(error.value)
    assert "final_checkpoints (figures 84, 89)" in message
    assert "Three-ReLU RN" in message
    assert "histogram_matched_checkpoints (figure 88)" not in message


@pytest.mark.parametrize(
    ("missing_cohort_member", "suppressed_outputs"),
    [
        ("training_event", {REPORT04_OUTPUT_NAMES[0]}),
        (
            "histogram",
            {
                REPORT04_OUTPUT_NAMES[1],
                REPORT04_OUTPUT_NAMES[2],
                REPORT04_OUTPUT_NAMES[7],
            },
        ),
        ("site_clipping", {REPORT04_OUTPUT_NAMES[3]}),
        ("joint_clipping", {REPORT04_OUTPUT_NAMES[4]}),
        (
            "checkpoint",
            {
                REPORT04_OUTPUT_NAMES[5],
                REPORT04_OUTPUT_NAMES[6],
                REPORT04_OUTPUT_NAMES[7],
            },
        ),
        (
            "propagation",
            {
                REPORT04_OUTPUT_NAMES[8],
                REPORT04_OUTPUT_NAMES[9],
            },
        ),
    ],
)
def test_report04_figure_families_require_their_complete_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_cohort_member: str,
    suppressed_outputs: set[str],
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir)
    artifacts[missing_cohort_member][-1].unlink()
    _patch_plot_renderers(monkeypatch)

    outputs = plots._generate_known_paper_figures(
        results_dir,
        tmp_path / "figures",
        save_png=False,
    )

    expected = [name for name in REPORT04_OUTPUT_NAMES if name not in suppressed_outputs]
    assert [path.name for path in outputs] == expected


def test_parameter_cohort_alone_does_not_activate_architecture_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    for _label, experiment_id in plots.REPORT04_TRAINING_RUNS:
        _write_complete_run(
            results_dir,
            experiment_id,
            "checkpoints/final/model.safetensors",
        )
    _patch_plot_renderers(monkeypatch)

    outputs = plots._generate_known_paper_figures(
        results_dir,
        tmp_path / "figures",
        save_png=False,
    )

    assert [path.name for path in outputs] == [
        REPORT04_OUTPUT_NAMES[5],
        REPORT04_OUTPUT_NAMES[6],
    ]


def test_missing_rn_checkpoint_does_not_suppress_histogram_matched_figure88(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    artifacts = _write_report04_cohort(results_dir, include_rn=True)
    rn_index = next(
        index
        for index, (label, _experiment_id) in enumerate(plots.REPORT04_RN_TRAINING_RUNS)
        if label == "Three-ReLU RN"
    )
    artifacts["checkpoint"][rn_index].unlink()
    _patch_plot_renderers(monkeypatch)

    outputs = plots.generate_report04_figures(
        results_dir,
        tmp_path / "figures",
        strict=False,
        include_rn=True,
    )
    names = {path.name for path in outputs}

    assert REPORT04_OUTPUT_NAMES[5] not in names
    assert REPORT04_OUTPUT_NAMES[6] not in names
    assert REPORT04_OUTPUT_NAMES[7] in names


def test_report04_architecture_outputs_require_report_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_plot_renderers(monkeypatch)

    outputs = plots._generate_known_paper_figures(
        tmp_path / "empty-results",
        tmp_path / "figures",
        save_png=False,
    )

    assert outputs == []
    assert calls == []


def _synthetic_activation_heatmap_payloads() -> dict[str, dict[str, Any]]:
    def payload_for_sites(sites: tuple[str, ...]) -> dict[str, Any]:
        methods = []
        for method_index, label in enumerate(
            plot_report04.REPORT04_HISTOGRAM_METHOD_LABELS
        ):
            layers = []
            for site in sites:
                for layer_index in range(6):
                    exact_fraction = 0.0 if method_index == 0 else 0.5
                    near_zero_fraction = (
                        0.0004 if method_index == 0 else 0.6
                    )
                    layers.append(
                        {
                            "name": f"{site}.layer_{layer_index}",
                            "threshold_fractions": {
                                "0": exact_fraction,
                                "0.01": near_zero_fraction,
                            },
                        }
                    )
            methods.append({"label": label, "layers": layers})
        return {
            "methods": methods,
            "validation_sequences": 2,
            "validation_tokens": 20,
        }

    return {
        "inputs": payload_for_sites(("attention_inputs", "mlp_inputs")),
        "mlp_hiddens": payload_for_sites(("mlp_hiddens",)),
    }


def _synthetic_propagation_payload(method_count: int) -> dict[str, Any]:
    num_layers = 6
    methods = []
    for method_index in range(method_count):
        activations = [
            {
                "name": name,
                "layer": layer,
                "zero_count": (method_index + layer) % 5,
                "total": 10,
            }
            for name, _label in plot_report04.PROPAGATION_ACTIVATION_ROWS
            for layer in range(num_layers)
        ]
        matmuls = [
            {
                "name": name,
                "layer": layer,
                "zero_count": (method_index + layer) % 5,
                "total": 10,
            }
            for name, _label in plot_report04.PROPAGATION_MATMUL_ROWS
            for layer in range(num_layers)
        ]
        methods.append(
            {
                "label": f"Method {method_index + 1}",
                "num_layers": num_layers,
                "activations": activations,
                "matmuls": matmuls,
            }
        )
    return {
        "methods": methods,
        "validation_tokens": 20,
        "validation_sequences": 2,
        "block_size": 10,
        "trailing_tokens_excluded": 0,
    }


def _write_report04_cohort(
    results_dir: Path,
    *,
    include_rn: bool = False,
) -> dict[str, list[Path]]:
    artifacts: dict[str, list[Path]] = {
        "training_event": [],
        "checkpoint": [],
        "histogram": [],
        "site_clipping": [],
        "joint_clipping": [],
        "propagation": [],
    }

    training_specs = (
        plots.REPORT04_RN_TRAINING_RUNS
        if include_rn
        else plots.REPORT04_TRAINING_RUNS
    )
    training_run_by_label: dict[str, Path] = {}
    for label, experiment_id in training_specs:
        run_dir = _write_complete_run(
            results_dir,
            experiment_id,
            "events.jsonl",
            "checkpoints/final/model.safetensors",
        )
        training_run_by_label[label] = run_dir
        artifacts["training_event"].append(run_dir / "events.jsonl")
        artifacts["checkpoint"].append(run_dir / "checkpoints/final/model.safetensors")

    histogram_methods = [
        {
            "label": label,
            "config_id": training_run_by_label[label].parent.name,
            "run_id": training_run_by_label[label].name,
        }
        for label in plots.REPORT04_HISTOGRAM_METHOD_LABELS
    ]
    for experiment_id in (
        plots.REPORT04_INPUT_HISTOGRAM_EXPERIMENT,
        plots.REPORT04_MLP_HISTOGRAM_EXPERIMENT,
    ):
        run_dir = _write_complete_run(results_dir, experiment_id, "activation_histograms.json")
        histogram_path = run_dir / "activation_histograms.json"
        _write_json(histogram_path, {"methods": histogram_methods})
        artifacts["histogram"].append(histogram_path)

    for site, _site_label in plots.REPORT04_CLIPPING_SITES:
        suffix = site.replace("_", "-")
        for _label, experiment_id in plots.REPORT04_CLIPPING_RUNS:
            run_dir = _write_complete_run(
                results_dir,
                f"{experiment_id}-clipping-sweep-report04-{suffix}",
                "clipping_frontier.jsonl",
            )
            artifacts["site_clipping"].append(run_dir / "clipping_frontier.jsonl")

    for _label, experiment_id in plots.REPORT04_JOINT_CLIPPING_RUNS:
        run_dir = _write_complete_run(
            results_dir,
            f"{experiment_id}-clipping-sweep-report04-joint",
            "clipping_frontier.jsonl",
        )
        artifacts["joint_clipping"].append(run_dir / "clipping_frontier.jsonl")

    run_dir = _write_complete_run(
        results_dir,
        (
            plots.REPORT04_RN_PROPAGATION_EXPERIMENT
            if include_rn
            else plots.POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT
        ),
        "activation_propagation.json",
    )
    artifacts["propagation"].append(run_dir / "activation_propagation.json")
    return artifacts


def _shorten_clipping_experiment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid legacy Windows MAX_PATH failures while preserving labels and gates."""
    monkeypatch.setattr(
        plots,
        "REPORT04_CLIPPING_RUNS",
        tuple(
            (label, f"report04-site-{index}")
            for index, (label, _experiment_id) in enumerate(plots.REPORT04_CLIPPING_RUNS)
        ),
    )
    monkeypatch.setattr(
        plots,
        "REPORT04_JOINT_CLIPPING_RUNS",
        tuple(
            (label, f"report04-joint-{index}")
            for index, (label, _experiment_id) in enumerate(plots.REPORT04_JOINT_CLIPPING_RUNS)
        ),
    )


def _write_complete_run(
    results_dir: Path,
    experiment_id: str,
    *specialized_artifacts: str,
    run_id: str = "001",
) -> Path:
    # Keep the synthetic path below the legacy Windows MAX_PATH boundary even
    # for the longest report experiment identifiers.
    run_dir = results_dir / experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text("experiment_name: contract-test\n", encoding="utf-8")
    _write_json(run_dir / "metrics.json", {})
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "manifest.json",
        {
            "config_id": experiment_id,
            "run_id": run_id,
            "status": "completed",
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "started_at": "2026-07-14T12:00:00+00:00",
            "finished_at": "2026-07-14T12:01:00+00:00",
        },
    )

    for artifact_name in specialized_artifacts:
        artifact_path = run_dir / artifact_name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.suffix == ".safetensors":
            artifact_path.write_bytes(b"synthetic checkpoint sentinel")
        elif artifact_path.suffix == ".jsonl":
            artifact_path.write_text("{}\n", encoding="utf-8")
        else:
            _write_json(artifact_path, {})
    return run_dir


def _patch_plot_renderers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def ignore_renderer(*_args: Any, **_kwargs: Any) -> list[Path]:
        return []

    for name, value in vars(plots).copy().items():
        if (
            name.startswith("generate_")
            and name != "generate_report04_figures"
            and callable(value)
        ):
            monkeypatch.setattr(plots, name, ignore_renderer)

    for renderer_name in REPORT04_RENDERERS:
        def record_renderer(
            *_args: Any,
            _renderer_name: str = renderer_name,
            **kwargs: Any,
        ) -> list[Path]:
            assert kwargs["save_png"] is False
            output = Path(kwargs["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"{_renderer_name}\n".encode("ascii"))
            calls.append((_renderer_name, kwargs))
            return [output]

        monkeypatch.setattr(plots, renderer_name, record_renderer)
    return calls


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
