from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import paper_exp.plots as plots
from paper_exp import plot_report04


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


@pytest.mark.parametrize(
    ("renderer", "message"),
    [
        (
            plot_report04._plot_post_layernorm_relu_propagation_heatmaps,
            "Activation-propagation heatmaps require exactly four matched checkpoints.",
        ),
        (
            plot_report04._plot_post_layernorm_relu_zero_product_heatmaps,
            "Zero-product heatmaps require exactly four matched checkpoints.",
        ),
    ],
)
def test_report04_propagation_layout_rejects_non_four_method_payloads(
    tmp_path: Path,
    renderer: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        renderer({"methods": []}, tmp_path / "unused.pdf")


def test_full_report04_cohort_dispatches_every_output_in_current_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "r"
    _shorten_clipping_experiment_ids(monkeypatch)
    _write_report04_cohort(results_dir)
    calls = _patch_plot_renderers(monkeypatch)

    outputs = plots._generate_known_paper_figures(
        results_dir,
        tmp_path / "figures",
        save_png=False,
    )

    assert [path.name for path in outputs] == list(REPORT04_OUTPUT_NAMES)
    assert [name for name, _kwargs in calls] == list(REPORT04_RENDERERS)

    call_by_renderer = {name: kwargs for name, kwargs in calls}
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


def _write_report04_cohort(results_dir: Path) -> dict[str, list[Path]]:
    artifacts: dict[str, list[Path]] = {
        "training_event": [],
        "checkpoint": [],
        "histogram": [],
        "site_clipping": [],
        "joint_clipping": [],
        "propagation": [],
    }

    for _label, experiment_id in plots.REPORT04_TRAINING_RUNS:
        run_dir = _write_complete_run(
            results_dir,
            experiment_id,
            "events.jsonl",
            "checkpoints/final/model.safetensors",
        )
        artifacts["training_event"].append(run_dir / "events.jsonl")
        artifacts["checkpoint"].append(run_dir / "checkpoints/final/model.safetensors")

    for experiment_id in (
        plots.REPORT04_INPUT_HISTOGRAM_EXPERIMENT,
        plots.REPORT04_MLP_HISTOGRAM_EXPERIMENT,
    ):
        run_dir = _write_complete_run(results_dir, experiment_id, "activation_histograms.json")
        artifacts["histogram"].append(run_dir / "activation_histograms.json")

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
        plots.POST_LAYERNORM_RELU_PROPAGATION_EXPERIMENT,
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
) -> Path:
    # Keep the synthetic path below the legacy Windows MAX_PATH boundary even
    # for the longest report experiment identifiers.
    run_id = "001"
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
        if name.startswith("generate_") and callable(value):
            monkeypatch.setattr(plots, name, ignore_renderer)

    for renderer_name in REPORT04_RENDERERS:
        def record_renderer(
            *_args: Any,
            _renderer_name: str = renderer_name,
            **kwargs: Any,
        ) -> list[Path]:
            assert kwargs["save_png"] is False
            output = Path(kwargs["output"])
            calls.append((_renderer_name, kwargs))
            return [output]

        monkeypatch.setattr(plots, renderer_name, record_renderer)
    return calls


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
