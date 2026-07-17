from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pytest

from paper_exp import plot_report05_diagnostics as diagnostics
from paper_exp.plot_api import REPORT04_PUBLICATION_PROFILE, publication_figure_issues


def test_propagation_matrix_pools_integer_counts_before_dividing() -> None:
    method = {
        "config_id": "example",
        "activations": [
            {"name": "a", "layer": 0, "available": True, "zero_count": 1, "total": 2},
            {"name": "a", "layer": 1, "available": True, "zero_count": 1, "total": 8},
            {"name": "b", "layer": 0, "available": True, "zero_count": 8, "total": 10},
            {"name": "b", "layer": 1, "available": True, "zero_count": 0, "total": 10},
        ],
    }
    stages = (
        diagnostics._StageSpec("a", "A"),
        diagnostics._StageSpec("b", "B"),
    )

    matrix = diagnostics._propagation_matrix(method, stages, num_layers=2)

    assert matrix[0] == pytest.approx([50.0, 12.5, 20.0])
    assert matrix[1] == pytest.approx([80.0, 0.0, 40.0])


def test_propagation_matrix_rejects_an_unavailable_selected_stage() -> None:
    method = {
        "config_id": "example",
        "activations": [
            {
                "name": "a",
                "layer": 0,
                "available": False,
                "unavailable_reason": "gate_absent",
                "zero_count": None,
                "total": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="gate_absent"):
        diagnostics._propagation_matrix(
            method,
            (diagnostics._StageSpec("a", "A"),),
            num_layers=1,
        )


def test_architecture_selection_uses_config_ids_not_display_labels() -> None:
    spec = diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES["three_relu"]
    payload = {
        "methods": [
            {"config_id": config_id, "label": f"misleading-{index}"}
            for index, (_method, _label, config_id) in enumerate(reversed(spec.runs))
        ]
    }

    selected = diagnostics._select_architecture_methods(payload, "three_relu")

    assert [item["config_id"] for item in selected] == [run[2] for run in spec.runs]


@pytest.mark.parametrize(
    ("percent", "expected"),
    [(0.0, "0"), (0.01, "<.1"), (83.07, "83.1"), (99.96, "100"), (100.0, "100")],
)
def test_compact_percent_labels_fit_narrow_heatmap_cells(percent: float, expected: str) -> None:
    assert diagnostics._compact_percent_label(percent) == expected
    assert len(expected) <= 4


@pytest.mark.parametrize("architecture_id", diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES)
def test_report05_heatmaps_are_compact_one_by_three_figures(architecture_id: str) -> None:
    payload = _synthetic_propagation_payload(architecture_id)

    figure = diagnostics._plot_report05_propagation_heatmaps(payload, architecture_id)
    try:
        assert len(figure.axes) == 4
        assert len(figure.axes[:3]) == 3
        assert figure.axes[-1].get_ylabel() == "Exact zeros (%)"
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert figure.get_figheight() <= 8.25
        assert [axis.get_title().split()[-1] for axis in figure.axes[:3]] == ["AdamW", "OR", "OL1"]
        first_panel_labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert first_panel_labels
        assert all(not tick.get_text() for tick in figure.axes[1].get_yticklabels())
        cell_labels = [text.get_text() for axis in figure.axes[:3] for text in axis.texts]
        assert cell_labels
        assert max(map(len, cell_labels)) <= 4
        assert publication_figure_issues(
            figure,
            diagnostics.REPORT05_HEATMAP_PROFILE,
        ) == ()
    finally:
        plt.close(figure)


def test_pooled_activation_distribution_sums_layers_before_normalizing() -> None:
    payloads = _synthetic_histogram_payloads()
    config_id = diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES["one_relu"].runs[0][2]

    distribution = diagnostics._pooled_activation_distribution(
        payloads,
        site="attention_inputs",
        config_id=config_id,
    )

    assert distribution["total"] == 60
    assert distribution["densities"] == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert distribution["validation_tokens"] == 24


@pytest.mark.parametrize(
    ("architecture_id", "expected_axes"),
    [("one_relu", 6), ("three_relu", 6), ("six_relu_pre", 9), ("six_relu_post", 9)],
)
def test_report05_distribution_figures_omit_exact_zero_atom_panels(
    architecture_id: str,
    expected_axes: int,
) -> None:
    figure = diagnostics._plot_report05_architecture_distributions(
        _synthetic_histogram_payloads(),
        _synthetic_weight_series(),
        architecture_id,
    )
    try:
        assert len(figure.axes) == expected_axes
        visible_text = " ".join(
            text.get_text()
            for text in figure.findobj()
            if hasattr(text, "get_text") and callable(text.get_text)
        )
        assert "P(x=0)" not in visible_text
        assert "exact-zero atoms are intentionally omitted" in visible_text
        assert figure.get_figwidth() == pytest.approx(7.16)
        assert publication_figure_issues(
            figure,
            REPORT04_PUBLICATION_PROFILE,
        ) == ()
        if architecture_id.startswith("six_relu"):
            assert "Q gate" in visible_text
            assert "no learned weight tensor" in visible_text
    finally:
        plt.close(figure)


def test_one_relu_weight_series_accepts_existing_mlp_relu_label() -> None:
    _method, display_label, config_id = diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES["one_relu"].runs[0]
    item = {"label": "MLP-ReLU AdamW", "weight_groups": {}}

    selected = diagnostics._weight_series_for_run(
        [item],
        architecture_id="one_relu",
        method="AdamW",
        display_label=display_label,
        config_id=config_id,
    )

    assert selected is item


def _synthetic_propagation_payload(architecture_id: str) -> dict[str, Any]:
    spec = diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES[architecture_id]
    methods = []
    for method_index, (_method, display_label, config_id) in enumerate(spec.runs):
        rows = []
        for stage_index, stage in enumerate(spec.stages):
            for layer in range(6):
                rows.append(
                    {
                        "name": stage.name,
                        "layer": layer,
                        "available": True,
                        "zero_count": (method_index + stage_index + layer) % 10,
                        "total": 10,
                    }
                )
        methods.append(
            {
                "label": display_label,
                "config_id": config_id,
                "num_layers": 6,
                "activations": rows,
            }
        )
    return {"validation_tokens": 24, "methods": methods}


def _synthetic_histogram_payloads() -> dict[str, dict[str, Any]]:
    all_runs = [
        run
        for spec in diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES.values()
        for run in spec.runs
    ]

    def payload(sites: tuple[str, ...], edges: list[float]) -> dict[str, Any]:
        methods = []
        for _method, display_label, config_id in all_runs:
            layers = []
            for site in sites:
                positive_range = edges[0] == 0.0
                for layer in range(6):
                    layers.append(
                        {
                            "name": f"{site}.layer_{layer}",
                            "counts": [8, 2] if positive_range else [2, 8],
                            "total": 10,
                            "underflow": 0,
                            "overflow": 0,
                            "threshold_hits": {"0": 4},
                        }
                    )
            methods.append(
                {
                    "label": display_label,
                    "config_id": config_id,
                    "layers": layers,
                }
            )
        return {
            "sites": list(sites),
            "bin_edges": edges,
            "validation_tokens": 24,
            "methods": methods,
        }

    return {
        "inputs": payload(("attention_inputs", "mlp_inputs"), [-1.0, 0.0, 1.0]),
        "hidden": payload(("mlp_hiddens",), [0.0, 1.0, 2.0]),
        "gates": payload(
            ("query_gate_outputs", "key_gate_outputs", "value_gate_outputs"),
            [0.0, 1.0, 2.0],
        ),
    }


def _synthetic_weight_series() -> list[dict[str, Any]]:
    series = []
    for spec in diagnostics.REPORT05_DIAGNOSTIC_ARCHITECTURES.values():
        for _method, display_label, config_id in spec.runs:
            series.append(
                {
                    "label": display_label,
                    "config_id": config_id,
                    "weight_groups": {
                        group_id: {
                            "centers": [-0.5, 0.5],
                            "densities": [0.5, 1.0],
                            "range": (-1.0, 1.0),
                        }
                        for group_id in ("qkv", "w1", "w2")
                    },
                }
            )
    return series
