from __future__ import annotations

import json

import pytest

from paper_exp.plot_fixed_step_l1_coupling import (
    BASELINE_CONFIG_ID,
    COHORT,
    generate_figure,
    load_coupling_summary,
    pearson_correlation,
)


def _method(config_id: str, label: str, *, index: int) -> dict[str, object]:
    mlp_hits = 60 + 5 * index
    attention_hits = 40 - index
    return {
        "config_id": config_id,
        "label": label,
        "layers": [
            {
                "name": "attention_outputs.layer_0",
                "total": 50,
                "threshold_hits": {"0": 0, "0.01": 5},
            },
            {
                "name": "attention_outputs.layer_1",
                "total": 150,
                "threshold_hits": {"0": 0, "0.01": attention_hits - 5},
            },
            {
                "name": "mlp_hiddens.layer_0",
                "total": 100,
                "threshold_hits": {"0": 0, "0.01": 5},
            },
            {
                "name": "mlp_hiddens.layer_1",
                "total": 300,
                "threshold_hits": {"0": 0, "0.01": mlp_hits - 5},
            },
        ],
    }


def _payload() -> dict[str, object]:
    methods = [_method(BASELINE_CONFIG_ID, "AdamW", index=0)]
    methods.extend(
        _method(config_id, f"method-{index}", index=index)
        for index, (_method_id, _weight, config_id) in enumerate(COHORT, start=1)
    )
    return {
        "schema_version": 2,
        "thresholds": [0.0, 0.01],
        "validation_tokens": 1024,
        "validation_sequences": 8,
        "methods": methods,
    }


def test_load_coupling_summary_pools_integer_counts_and_subtracts_adamw(tmp_path) -> None:
    path = tmp_path / "activation_histograms.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    summary = load_coupling_summary(path)

    assert summary.baseline_mlp_fraction == pytest.approx(60 / 400)
    assert summary.baseline_attention_fraction == pytest.approx(40 / 200)
    assert len(summary.points) == 12
    assert summary.points[0].delta_mlp_pp == pytest.approx(1.25)
    assert summary.points[0].delta_attention_pp == pytest.approx(-0.5)
    assert pearson_correlation(summary.points[:6]) == pytest.approx(-1.0)
    assert pearson_correlation(summary.points[6:]) == pytest.approx(-1.0)


def test_load_coupling_summary_rejects_an_incomplete_cohort(tmp_path) -> None:
    payload = _payload()
    payload["methods"] = payload["methods"][:-1]
    path = tmp_path / "activation_histograms.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected FFN-only L1 coupling cohort"):
        load_coupling_summary(path)


def test_generate_figure_exports_one_build_pdf_and_png(tmp_path) -> None:
    histograms = tmp_path / "activation_histograms.json"
    histograms.write_text(json.dumps(_payload()), encoding="utf-8")
    output = tmp_path / "coupling.pdf"

    paths = generate_figure(histograms, output, save_png=True)

    assert paths == [output, output.with_suffix(".png")]
    assert all(path.stat().st_size > 0 for path in paths)
