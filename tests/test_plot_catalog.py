from __future__ import annotations

import pytest

from paper_exp.plot_catalog import (
    REPORT04_FIGURES,
    get_report04_figure,
    list_report04_figures,
    report04_catalog_rows,
)


def test_report04_catalog_covers_the_numbered_suite_in_order() -> None:
    assert tuple(entry.number for entry in REPORT04_FIGURES) == tuple(range(79, 91))
    assert len({entry.filename for entry in REPORT04_FIGURES}) == 12
    assert len({entry.public_wrapper for entry in REPORT04_FIGURES}) == 12


def test_report04_catalog_matches_report_embedding_contract() -> None:
    embedded = list_report04_figures(embedded_only=True)

    assert tuple(entry.number for entry in embedded) == (79, 80, 82, 83, 85, 86, 87, 88, 89, 90)
    assert all(entry.embedded_in_report for entry in embedded)
    assert not get_report04_figure(81).embedded_in_report
    assert not get_report04_figure(84).embedded_in_report


def test_report04_catalog_records_multi_input_and_architecture_only_figures() -> None:
    assert get_report04_figure(88).required_artifact_kinds == (
        "activation_histograms.json",
        "checkpoints/final/model.safetensors",
    )
    assert get_report04_figure(87).required_artifact_kinds == ()
    assert get_report04_figure(90).required_artifact_kinds == ()


def test_report04_catalog_looks_up_exact_filename_and_rejects_unknown_entries() -> None:
    entry = get_report04_figure(
        "85-pythia-14m-minipile-post-layernorm-relu-zero-propagation-heatmaps.pdf"
    )

    assert entry.number == 85
    assert entry.public_wrapper == "generate_post_layernorm_relu_propagation_heatmaps"
    assert get_report04_figure("activation_propagation_heatmap") is entry
    assert get_report04_figure(entry.public_wrapper) is entry
    with pytest.raises(KeyError, match="Unknown Report 04 figure"):
        get_report04_figure(91)


def test_report04_catalog_rows_are_stable_and_human_readable() -> None:
    rows = report04_catalog_rows()

    assert len(rows) == 12
    assert rows[0] == (
        "79 | learning_diagnostics | "
        "79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf | "
        "artifacts: events.jsonl | wrapper: generate_report04_learning_diagnostics | embedded"
    )
    assert rows[-1].endswith(
        "artifacts: none | wrapper: generate_report04_pythia_family_compute_ceiling | embedded"
    )
    assert len(report04_catalog_rows(embedded_only=True)) == 10
