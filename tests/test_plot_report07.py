from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import paper_exp.plot_report07 as report07
from paper_exp.plot_catalog import (
    REPORT07_FIGURES,
    get_report07_figure,
    list_report07_figures,
    report07_catalog_rows,
)
from paper_exp.plot_report07 import (
    EXPECTED_S1_COUNTS,
    _b1_endpoint_rows,
    _b2_abs_vs_fixed,
    _b2_endpoint_rows,
    _b2_rms_vs_abs,
    _b2_threshold_triplets,
    _b3_geometry_endpoints,
    _b3_weight_endpoints,
    b2_kappa_distribution_rows,
    _gate_label,
    build_fixed_threshold_effect_figure,
    build_gate_type_effect_figure,
    build_learned_threshold_figure,
    build_pressure_weight_figure,
    build_ricker_shape_figure,
    build_seed_sensitivity_figure,
    load_s1_rows,
    write_b0_endpoint_table,
    write_b1_endpoint_table,
    write_b2_endpoint_table,
    write_b2_kappa_distribution_table,
    write_b3_geometry_table,
    write_b3_weight_table,
    write_endpoint_tables,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cohort():
    return load_s1_rows(
        ROOT / "docs/experimental-design/config-registry.yaml",
        ROOT / "docs/experimental-design/run-registry.yaml",
    )


def test_report07_catalog_is_sequential_and_unique() -> None:
    assert [entry.number for entry in REPORT07_FIGURES] == [*range(103, 110)]
    assert len({entry.filename for entry in REPORT07_FIGURES}) == 7
    assert get_report07_figure(106).plot_type == "s1_learned_threshold_ablation"
    assert get_report07_figure(113).embedded_in_report is True
    assert get_report07_figure(114).embedded_in_report is False
    assert get_report07_figure(116).embedded_in_report is True
    assert get_report07_figure(117).embedded_in_report is False
    assert get_report07_figure(118).embedded_in_report is True
    assert len(report07_catalog_rows()) == 14
    assert len(list_report07_figures(embedded_only=True)) == 11
    assert {entry.number for entry in list_report07_figures(embedded_only=True)} == {
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        112,
        113,
        116,
        118,
    }
    assert all(
        callable(getattr(report07, entry.public_wrapper))
        for entry in REPORT07_FIGURES
    )


def test_report07_registry_census_is_complete(cohort) -> None:
    counts = {
        block: sum(row.block == block for row in cohort)
        for block in EXPECTED_S1_COUNTS
    }
    assert counts == EXPECTED_S1_COUNTS
    assert len(cohort) == 132


def test_report07_b0_tradeoff_uses_registered_lr_triplets(cohort) -> None:
    triplets = report07._b0_lr_triplets(cohort)
    assert tuple(architecture for architecture, _items in triplets) == (
        report07.B0_LR_ARCHITECTURES
    )
    assert all(len(items) == 3 for _architecture, items in triplets)
    assert all(
        tuple(float(row.config["model_learning_rate"]) for row in items)
        == report07.B0_LR_VALUES
        for _architecture, items in triplets
    )

    central = report07._b0_central_rows(cohort)
    assert tuple(row.architecture for row in central) == report07.B0_TOPOLOGY_ORDER
    assert all(
        float(row.config["model_learning_rate"]) == pytest.approx(3e-5)
        for row in central
    )


def test_report07_b0_matched_effects_use_registered_parent_contrasts(cohort) -> None:
    effects = report07._b0_matched_effects(cohort)
    assert len(effects) == 10
    assert [(effect.child, effect.parent) for effect in effects] == [
        (child, parent)
        for _label, child, parent, _group in report07.B0_MATCHED_CONTRAST_SPECS
    ]
    by_label = {effect.label: effect for effect in effects}
    assert by_label["MLP-hidden ReLU"].delta_loss == pytest.approx(
        -0.060386,
        abs=0.000001,
    )
    assert by_label["MLP-hidden ReLU"].delta_r_model_pct == pytest.approx(
        2.0765,
        abs=0.0001,
    )
    assert by_label["+ V"].delta_loss == pytest.approx(0.003050, abs=0.000001)
    assert by_label["+ V"].delta_r_model_pct == pytest.approx(
        3.7030,
        abs=0.0001,
    )


def test_report07_b0_table_groups_loss_and_model_opportunity_by_lr(
    cohort,
    tmp_path,
) -> None:
    output = write_b0_endpoint_table(cohort, tmp_path / "b0.tex")
    text = output.read_text(encoding="utf-8")
    assert r"\multicolumn{2}{c}{$10^{-5}$}" in text
    assert r"\multicolumn{2}{c}{$3\times10^{-5}$}" in text
    assert r"\multicolumn{2}{c}{$10^{-4}$}" in text
    assert text.count(r"$R_m$ (\%)") == 4
    assert "$R_b$" not in text
    assert "$U$" not in text
    assert "A4-Q & \\multicolumn{2}{c}{--} & 7.02126 & 7.71" in text
    assert "A6-POST & 8.38743 & 14.83 & 7.03248 & 13.64 & 6.06320 & 13.64" in text


def test_report07_b1_table_lists_complete_absolute_endpoints(
    cohort,
    tmp_path,
) -> None:
    endpoints = _b1_endpoint_rows(cohort)
    assert len(endpoints) == 53
    assert sum(endpoint.kappa > 0.0 for endpoint in endpoints) == 36
    assert sum(endpoint.kappa == 0.0 for endpoint in endpoints) == 17
    assert len(
        {
            (endpoint.architecture, endpoint.gate_family, endpoint.kappa)
            for endpoint in endpoints
        }
    ) == 53
    assert {endpoint.row.number for endpoint in endpoints if endpoint.kappa > 0.0} == {
        row.number for row in cohort if row.block == "S1-B1"
    }

    output = write_b1_endpoint_table(cohort, tmp_path / "b1.tex")
    text = output.read_text(encoding="utf-8")
    assert "\\multicolumn{3}{c}{$G^+$}" in text
    assert "\\multicolumn{3}{c}{$G^\\pm$}" in text
    assert "& & Cfg & $L$ & $R_m$ (\\%)" in text
    data_lines = [
        line
        for line in text.splitlines()
        if any(f"{endpoint.row.number} &" in line for endpoint in endpoints)
    ]
    assert len(data_lines) == 31
    assert "A6-POST-QKV" in text
    assert "A6-POST-ALL" in text
    assert "attention" not in text.lower()
    assert "branch" not in text.lower()
    for endpoint in endpoints:
        assert (
            f"{endpoint.row.number} & {endpoint.row.loss:.5f} & "
            f"{endpoint.row.r_model_pct:.2f}"
        ) in text


def test_report07_b0_matched_effect_figure_is_decision_oriented(cohort) -> None:
    figure = report07.build_b0_matched_architecture_effect_figure(cohort)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xlabel() == (
            r"$\Delta R_{\mathrm{model}}$ vs registered parent "
            "(percentage points)"
        )
        assert axis.get_ylabel() == (
            r"$\Delta$ validation loss vs registered parent"
        )
        assert axis.get_xlim()[0] == pytest.approx(0.0)
        assert len(axis.collections) == 10
        point_labels = {text.get_text() for text in axis.texts}
        assert {effect.label for effect in report07._b0_matched_effects(cohort)} == (
            point_labels
        )
        figure_text = " ".join(text.get_text() for text in figure.texts)
        assert "not measured speedup" in figure_text
        assert "right and down is favorable" in figure_text
    finally:
        plt.close(figure)


def test_report07_b2_matched_effect_contract(cohort) -> None:
    learned_pairs = _b2_abs_vs_fixed(cohort)
    learned_compute = [
        learned.r_model_pct - fixed.r_model_pct
        for learned, fixed, _category in learned_pairs
    ]
    assert len(learned_pairs) == 15
    assert min(learned_compute) == pytest.approx(0.2095, abs=0.0001)
    assert max(learned_compute) == pytest.approx(1.9074, abs=0.0001)

    rms_pairs = _b2_rms_vs_abs(cohort)
    assert len(rms_pairs) == 11
    assert all(rms.loss < absolute.loss for rms, absolute, _category in rms_pairs)
    assert all(
        rms.r_model_pct < absolute.r_model_pct
        for rms, absolute, _category in rms_pairs
    )


def test_report07_b2_table_lists_all_absolute_endpoints(cohort, tmp_path) -> None:
    endpoints = _b2_endpoint_rows(cohort)
    assert len(endpoints) == 26
    assert {endpoint.row.number for endpoint in endpoints} == {
        row.number for row in cohort if row.block == "S1-B2"
    }
    assert {endpoint.architecture for endpoint in endpoints} == {
        "A1-H",
        "A3",
        "A5-QK-PRE",
        "A5-QK-POST",
        "A6-PRE-QKV",
        "A6-POST-QKV",
        "A6-POST-ALL",
    }
    assert {endpoint.parameterization for endpoint in endpoints} == {
        "Model-wide",
        "Per site",
        "Per layer/site",
    }

    output = write_b2_endpoint_table(cohort, tmp_path / "b2.tex")
    text = output.read_text(encoding="utf-8")
    assert "\\multicolumn{3}{c}{$G^+$}" in text
    assert "\\multicolumn{3}{c}{$G^\\pm$}" in text
    assert "& & & Cfg & $L$ & $R_m$ (\\%)" in text
    data_lines = [
        line
        for line in text.splitlines()
        if any(f"{endpoint.row.number} &" in line for endpoint in endpoints)
    ]
    assert len(data_lines) == 16
    assert "A6-POST-QKV" in text
    assert "A6-POST-ALL" in text
    assert "attention" not in text.lower()
    assert "branch" not in text.lower()
    assert "PLS" not in text
    for endpoint in endpoints:
        assert (
            f"{endpoint.row.number} & {endpoint.row.loss:.5f} & "
            f"{endpoint.row.r_model_pct:.2f}"
        ) in text


def test_report07_b2_kappa_distribution_uses_final_parameter_metrics(
    cohort,
    tmp_path,
) -> None:
    rows = b2_kappa_distribution_rows(cohort, ROOT)
    assert [(runs, parameters) for _label, runs, parameters, *_values in rows] == [
        (8, 120),
        (8, 120),
        (3, 60),
        (3, 60),
        (2, 2),
        (2, 6),
    ]
    assert rows[0][3:] == pytest.approx(
        (0.1001107171, 0.144379, 0.2019860148),
        abs=1e-6,
    )
    assert min(row[3] for row in rows) == pytest.approx(0.0839349926, abs=1e-7)
    assert max(row[5] for row in rows) == pytest.approx(0.2178208679, abs=1e-7)

    output = write_b2_kappa_distribution_table(
        cohort,
        tmp_path / "b2-kappa.tex",
        repo_root=ROOT,
    )
    text = output.read_text(encoding="utf-8")
    assert "Q/K(/V), absolute, per layer/site & 8 & 120" in text
    assert "A6-POST Q/K/V, absolute, per site & 2 & 6" in text
    assert "0.0839" in text
    assert "0.2178" in text


def test_report07_b2_learned_figure_uses_absolute_endpoints(cohort) -> None:
    assert len(_b2_threshold_triplets(cohort)) == 11
    figure = build_learned_threshold_figure(cohort)
    try:
        figure.canvas.draw()
        assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 3.05))
        assert len(figure.axes) == 2
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "(a) Validation loss",
            r"(b) $R_{\mathrm{model}}$",
        ]
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "Validation loss",
            r"$R_{\mathrm{model}}$ (%)",
        ]
        assert all(len(axis.lines) == 11 for axis in figure.axes)
        assert all(
            [tick.get_text() for tick in axis.get_xticklabels()]
            == [
                "Fixed\n" + r"$\kappa=0.10$",
                "Learned\nabsolute",
                "Learned\nRMS-relative",
            ]
            for axis in figure.axes
        )
        assert all(
            len(line.get_xdata()) == 3
            for axis in figure.axes
            for line in axis.lines
        )
        assert len(figure.legends) == 1
        assert all("Delta" not in axis.get_ylabel() for axis in figure.axes)
    finally:
        plt.close(figure)


def test_report07_b3_endpoint_tables_cover_all_cells(cohort, tmp_path) -> None:
    weight_endpoints = _b3_weight_endpoints(cohort)
    geometry_endpoints = _b3_geometry_endpoints(cohort)
    assert len(weight_endpoints) == 12
    assert len(geometry_endpoints) == 12

    displayed_numbers = {
        row.number
        for endpoint in (*weight_endpoints, *geometry_endpoints)
        for row in (endpoint.naive, endpoint.orthogonal)
    }
    assert displayed_numbers == {
        row.number for row in cohort if row.block == "S1-B3"
    }
    assert {endpoint.value for endpoint in geometry_endpoints} == {
        0.05,
        0.10,
        0.20,
        0.50,
    }

    weight_output = write_b3_weight_table(cohort, tmp_path / "b3-weight.tex")
    geometry_output = write_b3_geometry_table(
        cohort, tmp_path / "b3-geometry.tex"
    )
    weight_text = weight_output.read_text(encoding="utf-8")
    geometry_text = geometry_output.read_text(encoding="utf-8")
    assert "\\multicolumn{3}{c}{AdamW}" in weight_text
    assert "\\multicolumn{3}{c}{Naive}" in weight_text
    assert "\\multicolumn{3}{c}{Orthogonal}" in weight_text
    assert "\\multicolumn{3}{c}{RN}" in geometry_text
    assert "\\multicolumn{3}{c}{OR}" in geometry_text
    assert "Delta" not in weight_text
    assert "Delta" not in geometry_text
    for endpoint in weight_endpoints:
        for row in (endpoint.adamw, endpoint.naive, endpoint.orthogonal):
            assert (
                f"{row.number} & {row.loss:.5f} & {row.r_model_pct:.2f}"
                in weight_text
            )
    for endpoint in geometry_endpoints:
        for row in (endpoint.adamw, endpoint.naive, endpoint.orthogonal):
            assert (
                f"{row.number} & {row.loss:.5f} & {row.r_model_pct:.2f}"
                in geometry_text
            )


def test_report07_gpm_latex_uses_one_command_backslash(cohort) -> None:
    gpm = next(row for row in cohort if row.config["gate_family"] == "gpm")
    label = _gate_label(gpm.config)
    assert r"G$^\pm$" in label
    assert r"G$^\\pm$" not in label


def test_report07_gate_labels_distinguish_relu_and_threshold_sites(cohort) -> None:
    by_number = {row.number: row for row in cohort}
    assert _gate_label(by_number[193].config) == (
        r"A6-POST; ReLU a/m/h; G$^\pm$ Q/K/V"
    )
    assert _gate_label(by_number[209].config) == (
        r"A6-POST; G$^+$ a/m/h/Q/K/V"
    )
    assert _gate_label(by_number[228].config) == (
        r"A6-POST; ReLU a/m/h; ATG G$^\pm$ Q/K/V"
    )
    assert _gate_label(by_number[242].config) == (
        r"A6-POST; ATG G$^+$ a/m/h/Q/K/V"
    )


def test_report07_appendix_discloses_both_coupled_seeds(cohort, tmp_path) -> None:
    output = write_endpoint_tables(cohort, tmp_path / "tables.tex")
    text = output.read_text(encoding="utf-8")
    assert "Init/data" in text
    assert "1/1" in text
    assert r"ReLU a/m/h; G$^\pm$ Q/K/V" in text
    assert r"\tablename\ \thetable\ (continued)" in text
    assert r"\clearpage" in text
    assert "$R_b$" not in text
    assert "$U$" not in text
    assert "$R_m$" in text


def test_report07_provenance_keys_are_repository_relative() -> None:
    source = ROOT / "src/paper_exp/plot_report07.py"
    assert report07._provenance_key(source, ROOT) == (
        "src/paper_exp/plot_report07.py"
    )


def test_report07_threshold_figure_has_two_absolute_metric_panels(cohort) -> None:
    figure = build_fixed_threshold_effect_figure(cohort)
    try:
        figure.canvas.draw()
        assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 2.85))
        assert len(figure.axes) == 2
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "(a) Validation loss",
            r"(b) $R_{\mathrm{model}}$",
        ]
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "Validation loss",
            r"$R_{\mathrm{model}}$ (%)",
        ]
        assert all(
            [tick.get_text() for tick in axis.get_xticklabels()]
            == ["0", "0.03", "0.10", "0.30"]
            for axis in figure.axes
        )
        assert all(len(axis.lines) == 8 for axis in figure.axes)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "A5-QK-PRE",
            "A5-QK-POST",
            "A6-PRE-QKV",
            "A6-POST-QKV",
            r"$G^+$",
            r"$G^\pm$",
        ]
    finally:
        plt.close(figure)


def test_report07_gate_type_figure_has_two_absolute_metric_panels(cohort) -> None:
    figure = build_gate_type_effect_figure(cohort)
    try:
        figure.canvas.draw()
        assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 2.85))
        assert len(figure.axes) == 2
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "(a) Validation loss",
            r"(b) $R_{\mathrm{model}}$",
        ]
        assert all(
            [tick.get_text() for tick in axis.get_xticklabels()]
            == [r"$G^+$", r"$G^\pm$"]
            for axis in figure.axes
        )
        assert all(len(axis.lines) == 12 for axis in figure.axes)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "A5-QK-PRE",
            "A5-QK-POST",
            "A6-PRE-QKV",
            "A6-POST-QKV",
            r"$\kappa=0.03$",
            r"$\kappa=0.10$",
            r"$\kappa=0.30$",
        ]
    finally:
        plt.close(figure)


def test_report07_b3_figures_use_absolute_endpoints(cohort) -> None:
    figures = (
        (
            build_pressure_weight_figure(cohort),
            [
                "(a) L1: validation loss",
                "(b) Ricker: validation loss",
                r"(c) L1: $R_{\mathrm{model}}$",
                r"(d) Ricker: $R_{\mathrm{model}}$",
            ],
        ),
        (
            build_ricker_shape_figure(cohort),
            [
                "(a) Coupled basin: validation loss",
                "(b) Shape: validation loss",
                r"(c) Coupled basin: $R_{\mathrm{model}}$",
                r"(d) Shape: $R_{\mathrm{model}}$",
            ],
        ),
    )
    try:
        for figure, titles in figures:
            figure.canvas.draw()
            assert tuple(figure.get_size_inches()) == pytest.approx((7.16, 4.10))
            assert len(figure.axes) == 4
            assert [axis.get_title(loc="left") for axis in figure.axes] == titles
            assert figure.axes[0].get_ylabel() == "Validation loss"
            assert figure.axes[2].get_ylabel() == r"$R_{\mathrm{model}}$ (\%)"
            assert all(len(axis.lines) == 4 for axis in figure.axes)
            assert all(
                len(line.get_xdata()) == 4
                for axis in figure.axes
                for line in axis.lines
            )
            assert all("Delta" not in axis.get_ylabel() for axis in figure.axes)
            assert all(
                axis.get_xticklabels()[0].get_text() == "AdamW"
                for axis in figure.axes[2:]
            )
    finally:
        for figure, _titles in figures:
            plt.close(figure)


def test_report07_seed_figure_states_sign_agreement(cohort) -> None:
    figure = build_seed_sensitivity_figure(cohort)
    try:
        assert figure.axes[2].get_title(loc="left") == "(c) Loss: 9/10 signs agree"
        assert figure.axes[3].get_title(loc="left") == (
            "(d) Compute: 10/10 signs agree"
        )
        annotations = [text.get_text() for text in figure.axes[2].texts]
        assert annotations == [r"only sign reversal: $G^\pm-A3$"]
        assert len(figure.axes[0].texts) == 0
        assert len(figure.axes[1].texts) == 0
        assert len(figure.axes[3].texts) == 0
    finally:
        plt.close(figure)
