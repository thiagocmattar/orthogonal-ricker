from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from paper_exp.plot_api import (
    DOUBLE_COLUMN_LAYOUT,
    DOUBLE_COLUMN_WIDTH_INCHES,
    REPORT04_PUBLICATION_PROFILE,
    SINGLE_COLUMN_LAYOUT,
    SINGLE_COLUMN_WIDTH_INCHES,
    GridLayout,
    PlotQualityError,
    PublicationProfile,
    export_figure,
    make_panel_grid,
    publish_staged_outputs,
    publication_figure_issues,
    validate_publication_figure,
)
from paper_exp.plot_style import REPORT04_PLOT_STYLE


def test_report04_style_is_independent_of_dark_global_style() -> None:
    dark_style = {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "savefig.facecolor": "black",
    }
    with plt.rc_context(dark_style), plt.rc_context(REPORT04_PLOT_STYLE):
        figure, axis = plt.subplots()
        try:
            assert figure.get_facecolor() == to_rgba("white")
            assert axis.get_facecolor() == to_rgba("white")
            assert axis.spines["left"].get_edgecolor() == to_rgba("black")
            assert plt.rcParams["text.color"] == "black"
            assert plt.rcParams["savefig.facecolor"] == "white"
        finally:
            plt.close(figure)


@pytest.mark.parametrize(
    ("panel_count", "columns", "expected_shape", "hidden_count"),
    [
        (1, 1, (1, 1), 0),
        (3, 3, (1, 3), 0),
        (3, 1, (3, 1), 0),
        (5, 2, (3, 2), 1),
    ],
)
def test_make_panel_grid_has_stable_axes_for_edge_shapes(
    panel_count: int,
    columns: int,
    expected_shape: tuple[int, int],
    hidden_count: int,
) -> None:
    panel_grid = make_panel_grid(panel_count, GridLayout(columns=columns))
    try:
        assert panel_grid.axes.shape == expected_shape
        assert len(panel_grid.flat_axes) == panel_count
        assert len(panel_grid.hidden_axes) == hidden_count
        assert all(axis.get_visible() for axis in panel_grid.flat_axes)
        assert all(not axis.get_visible() for axis in panel_grid.hidden_axes)
        assert panel_grid.colorbar_axis is None
    finally:
        plt.close(panel_grid.figure)


def test_make_panel_grid_caps_columns_at_panel_count() -> None:
    panel_grid = make_panel_grid(1, GridLayout(columns=4))
    try:
        assert panel_grid.axes.shape == (1, 1)
    finally:
        plt.close(panel_grid.figure)


def test_make_panel_grid_can_reserve_one_shared_colorbar_axis() -> None:
    panel_grid = make_panel_grid(
        3,
        GridLayout(columns=2, hspace=0.2, wspace=0.1),
        shared_colorbar=True,
    )
    try:
        assert panel_grid.axes.shape == (2, 2)
        assert len(panel_grid.flat_axes) == 3
        assert len(panel_grid.hidden_axes) == 1
        assert panel_grid.colorbar_axis is not None
        subplotspec = panel_grid.colorbar_axis.get_subplotspec()
        assert (subplotspec.rowspan.start, subplotspec.rowspan.stop) == (0, 2)
        assert panel_grid.colorbar_axis not in panel_grid.axes.flat
    finally:
        plt.close(panel_grid.figure)


def test_make_panel_grid_applies_requested_axis_sharing() -> None:
    panel_grid = make_panel_grid(
        4,
        GridLayout(columns=2, sharex="col", sharey="row"),
    )
    try:
        axes = panel_grid.axes
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 0])
        assert not axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[0, 1])
        assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[0, 1])
        assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])
    finally:
        plt.close(panel_grid.figure)


def test_make_panel_grid_rejects_share_modes_in_subplot_kwargs() -> None:
    existing_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="through GridLayout"):
        make_panel_grid(
            1,
            GridLayout(columns=1),
            subplot_kwargs={"sharex": True},
        )
    assert set(plt.get_fignums()) == existing_figures


def test_publication_width_presets_use_final_column_widths() -> None:
    assert SINGLE_COLUMN_WIDTH_INCHES == 3.5
    assert DOUBLE_COLUMN_WIDTH_INCHES == 7.16
    assert SINGLE_COLUMN_LAYOUT.width_inches == SINGLE_COLUMN_WIDTH_INCHES
    assert DOUBLE_COLUMN_LAYOUT.width_inches == DOUBLE_COLUMN_WIDTH_INCHES

    single = make_panel_grid(1, SINGLE_COLUMN_LAYOUT)
    double = make_panel_grid(2, DOUBLE_COLUMN_LAYOUT)
    try:
        assert single.figure.get_figwidth() == pytest.approx(3.5)
        assert double.figure.get_figwidth() == pytest.approx(7.16)
    finally:
        plt.close(single.figure)
        plt.close(double.figure)


def test_grid_layout_is_immutable_and_validated() -> None:
    layout = GridLayout(columns=2)
    with pytest.raises(FrozenInstanceError):
        layout.columns = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="columns"):
        GridLayout(columns=0)
    with pytest.raises(ValueError, match="positive integer"):
        GridLayout(columns=True)
    with pytest.raises(ValueError, match="positive integer"):
        GridLayout(columns=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="panel_aspect"):
        GridLayout(columns=1, panel_aspect=0.0)
    with pytest.raises(ValueError, match="sharex"):
        GridLayout(columns=1, sharex="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("panel_count", [0, -1, True, 1.5])
def test_make_panel_grid_rejects_invalid_panel_counts(panel_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_panel_grid(panel_count, GridLayout(columns=1))  # type: ignore[arg-type]


def test_make_panel_grid_closes_figure_when_subplot_construction_fails() -> None:
    existing_figures = set(plt.get_fignums())

    with pytest.raises(AttributeError, match="unexpected keyword"):
        make_panel_grid(
            1,
            GridLayout(columns=1),
            subplot_kwargs={"definitely_not_an_axes_property": True},
        )

    assert set(plt.get_fignums()) == existing_figures


def test_export_figure_builds_once_and_writes_pdf_and_png(tmp_path: Path) -> None:
    build_calls = 0
    built_figure_number: int | None = None

    def build() -> Figure:
        nonlocal build_calls, built_figure_number
        build_calls += 1
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [1.0, 0.0])
        built_figure_number = figure.number
        return figure

    pdf_path = tmp_path / "nested" / "figure.pdf"
    outputs = export_figure(build, pdf_path, save_png=True)

    assert outputs == [pdf_path, pdf_path.with_suffix(".png")]
    assert build_calls == 1
    assert all(path.stat().st_size > 100 for path in outputs)
    assert built_figure_number is not None
    assert not plt.fignum_exists(built_figure_number)


def test_export_figure_profile_disables_tight_output_bounding_box(tmp_path: Path) -> None:
    observed_save_bbox: list[object] = []
    output = tmp_path / "figure.pdf"

    def build() -> Figure:
        figure, axis = plt.subplots(figsize=(7.16, 3.0))
        axis.set_axis_off()
        figure.text(0.5, 0.5, "Readable note", ha="center", fontsize=8)
        original_savefig = figure.savefig

        def record_savefig(*args: object, **kwargs: object) -> None:
            observed_save_bbox.append(plt.rcParams["savefig.bbox"])
            original_savefig(*args, **kwargs)

        figure.savefig = record_savefig  # type: ignore[method-assign]
        return figure

    export_figure(
        build,
        output,
        style={"savefig.bbox": "tight"},
        profile=REPORT04_PUBLICATION_PROFILE,
    )

    assert observed_save_bbox == [None]
    media_box = re.search(
        rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        output.read_bytes(),
    )
    assert media_box is not None
    left, _bottom, right, _top = (float(value) for value in media_box.groups())
    assert (right - left) / 72.0 == pytest.approx(7.16)


def test_export_figure_scopes_rc_parameters(tmp_path: Path) -> None:
    previous_font_size = float(plt.rcParams["font.size"])
    observed_font_sizes: list[float] = []

    def build() -> Figure:
        observed_font_sizes.append(float(plt.rcParams["font.size"]))
        figure, _axis = plt.subplots()
        return figure

    export_figure(
        build,
        tmp_path / "figure.pdf",
        style={"font.size": 23.0},
    )

    assert observed_font_sizes == [23.0]
    assert float(plt.rcParams["font.size"]) == previous_font_size


def test_exported_pdf_is_byte_stable(tmp_path: Path) -> None:
    def build() -> Figure:
        figure, axis = plt.subplots(figsize=(7.16, 3.0))
        axis.plot([0.0, 1.0], [0.0, 1.0])
        axis.set_title("Deterministic paper figure", fontsize=8)
        return figure

    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    export_figure(
        build,
        first,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )
    export_figure(
        build,
        second,
        style=REPORT04_PLOT_STYLE,
        profile=REPORT04_PUBLICATION_PROFILE,
    )

    assert first.read_bytes() == second.read_bytes()


def test_export_figure_closes_partial_figures_when_builder_raises(tmp_path: Path) -> None:
    existing_figures = set(plt.get_fignums())

    def fail_after_creating_figure() -> Figure:
        plt.subplots()
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        export_figure(fail_after_creating_figure, tmp_path / "figure.pdf")

    assert set(plt.get_fignums()) == existing_figures


def test_export_figure_closes_figure_when_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_figure: Figure | None = None

    def build() -> Figure:
        nonlocal built_figure
        built_figure, _axis = plt.subplots()
        return built_figure

    def fail_savefig(self: Figure, *_args: object, **_kwargs: object) -> None:
        raise OSError("save failed")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    with pytest.raises(OSError, match="save failed"):
        export_figure(build, tmp_path / "figure.pdf")

    assert built_figure is not None
    assert not plt.fignum_exists(built_figure.number)


def test_export_figure_leaves_no_partial_output_when_png_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "figure.pdf"
    png_output = output.with_suffix(".png")
    original_savefig = Figure.savefig

    def fail_png(self: Figure, path: str | Path, *args: object, **kwargs: object) -> None:
        if Path(path).suffix == ".png":
            raise OSError("png failed")
        original_savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_png)

    with pytest.raises(OSError, match="png failed"):
        export_figure(lambda: plt.figure(), output, save_png=True)

    assert not output.exists()
    assert not png_output.exists()
    assert not list(tmp_path.glob(".*.stage.*"))


def test_export_figure_rolls_back_existing_outputs_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "figure.pdf"
    png_output = output.with_suffix(".png")
    output.write_bytes(b"old pdf")
    png_output.write_bytes(b"old png")
    original_replace = Path.replace

    def fail_png_publish(self: Path, target: str | Path) -> Path:
        if ".stage.png" in self.name:
            raise OSError("publish failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_png_publish)

    with pytest.raises(OSError, match="publish failed"):
        export_figure(lambda: plt.figure(), output, save_png=True)

    assert output.read_bytes() == b"old pdf"
    assert png_output.read_bytes() == b"old png"
    assert not list(tmp_path.glob(".*.stage.*"))
    assert not list(tmp_path.glob(".*.backup.*"))


def test_publish_staged_outputs_uses_final_to_staged_mapping(tmp_path: Path) -> None:
    first_final = tmp_path / "first.pdf"
    second_final = tmp_path / "second.png"
    first_stage = tmp_path / ".first.stage.pdf"
    second_stage = tmp_path / ".second.stage.png"
    first_final.write_bytes(b"old first")
    first_stage.write_bytes(b"new first")
    second_stage.write_bytes(b"new second")

    publish_staged_outputs(
        {
            first_final: first_stage,
            second_final: second_stage,
        }
    )

    assert first_final.read_bytes() == b"new first"
    assert second_final.read_bytes() == b"new second"
    assert not first_stage.exists()
    assert not second_stage.exists()


def test_export_figure_requires_pdf_primary_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.pdf suffix"):
        export_figure(lambda: plt.figure(), tmp_path / "figure.png")


def test_export_figure_rejects_non_figure_builder_result(tmp_path: Path) -> None:
    existing_figures = set(plt.get_fignums())
    with pytest.raises(TypeError, match="must return"):
        export_figure(lambda: object(), tmp_path / "figure.pdf")  # type: ignore[arg-type]
    assert set(plt.get_fignums()) == existing_figures


def test_publication_profile_accepts_final_width_text_inside_canvas() -> None:
    figure, axis = plt.subplots(figsize=(7.16, 3.0))
    try:
        axis.set_title("Readable title", fontsize=8)
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
        validate_publication_figure(figure, REPORT04_PUBLICATION_PROFILE)
    finally:
        plt.close(figure)


def test_publication_profile_ignores_tick_labels_outside_the_active_view() -> None:
    figure, axis = plt.subplots(figsize=(7.16, 3.0))
    try:
        axis.plot([0.0, 1.0], [0.0, 1.0])
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()

        figure.text(-0.1, 0.5, "outside figure note", fontsize=8)
        assert any(
            "outside figure note" in issue and "outside the figure canvas" in issue
            for issue in publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE)
        )
    finally:
        plt.close(figure)


def test_publication_profile_ignores_ticks_when_axes_are_disabled() -> None:
    figure, axis = plt.subplots(figsize=(7.16, 3.0))
    try:
        axis.set_axis_off()
        assert publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE) == ()
    finally:
        plt.close(figure)


def test_publication_profile_aggregates_size_font_and_containment_issues() -> None:
    figure = plt.figure(figsize=(8.0, 9.0))
    figure.text(-0.1, 0.5, "outside tiny note", fontsize=6)
    try:
        issues = publication_figure_issues(figure, REPORT04_PUBLICATION_PROFILE)
        assert any("width" in issue for issue in issues)
        assert any("height" in issue for issue in issues)
        assert any("6 pt" in issue for issue in issues)
        assert any("outside the figure canvas" in issue for issue in issues)
        with pytest.raises(PlotQualityError, match="Figure failed publication checks"):
            validate_publication_figure(figure, REPORT04_PUBLICATION_PROFILE)
    finally:
        plt.close(figure)


def test_publication_profile_validates_positive_constraints() -> None:
    with pytest.raises(ValueError, match="width_inches"):
        PublicationProfile(width_inches=0.0, max_height_inches=8.8)


def test_export_figure_checks_profile_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "invalid.pdf"

    with pytest.raises(PlotQualityError, match="figure width"):
        export_figure(
            lambda: plt.figure(figsize=(6.0, 3.0)),
            output,
            profile=REPORT04_PUBLICATION_PROFILE,
        )

    assert not output.exists()
