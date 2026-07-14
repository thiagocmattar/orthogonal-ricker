from __future__ import annotations

import pytest

import paper_exp.plots as plots
from paper_exp import plot_common, plot_report04, plot_style


def test_report04_method_styles_cover_the_training_cohort_and_facade() -> None:
    labels = {label for label, _experiment_id in plots.REPORT04_TRAINING_RUNS}

    assert set(plots.REPORT04_METHOD_COLORS) == labels
    assert set(plots.REPORT04_METHOD_MARKERS) == labels
    assert len(set(plots.REPORT04_METHOD_COLORS.values())) == len(labels)
    assert plots.REPORT04_METHOD_COLORS is plot_style.REPORT04_METHOD_COLORS
    assert plots.REPORT04_METHOD_MARKERS is plot_style.REPORT04_METHOD_MARKERS


def test_pythia_14m_three_relu_compute_ceiling_identities() -> None:
    d = plots.REPORT04_HIDDEN_SIZE
    layers = plots.REPORT04_NUM_LAYERS
    sequence_length = plots.REPORT04_BLOCK_SIZE
    vocab_size = plots.REPORT04_VOCAB_SIZE

    target_products = layers * 11 * d**2
    block_products = layers * (12 * d**2 + d * (sequence_length + 1))
    lm_head_products = d * vocab_size

    assert plots.REPORT04_TARGET_PRODUCTS_PER_TOKEN == target_products
    assert plots.REPORT04_BLOCK_PRODUCTS_PER_TOKEN == block_products
    assert plots.REPORT04_LM_HEAD_PRODUCTS_PER_TOKEN == lm_head_products
    assert plots.REPORT04_MODEL_PRODUCTS_PER_TOKEN == block_products + lm_head_products
    assert 100.0 * target_products / block_products == pytest.approx(39.27, abs=0.005)
    assert 100.0 * plots.REPORT04_TARGET_MODEL_FRACTION == pytest.approx(11.76, abs=0.005)


def test_pythia_family_dimensions_and_compute_ceiling_shares() -> None:
    expected_family = (
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
    expected_targetable_shares = (
        11.7637,
        20.8515,
        33.9748,
        54.6548,
        68.5450,
        75.6760,
        78.4374,
        82.0011,
        85.3705,
        86.7837,
    )

    assert plot_report04.REPORT04_PYTHIA_FAMILY == expected_family

    targetable_shares = []
    for _label, layers, hidden_size, vocab_size in expected_family:
        targetable_products = 11 * layers * hidden_size**2
        model_products = (
            layers
            * (
                12 * hidden_size**2
                + hidden_size * (plot_report04.REPORT04_BLOCK_SIZE + 1)
            )
            + hidden_size * vocab_size
        )
        targetable_shares.append(100.0 * targetable_products / model_products)

    assert targetable_shares == pytest.approx(expected_targetable_shares, abs=0.00005)
    assert targetable_shares == sorted(targetable_shares)


def test_nondominated_skip_loss_points_maximize_skip_and_minimize_loss() -> None:
    points = [
        {"name": "low-loss", "skip": 0.10, "loss": 1.00},
        {"name": "tradeoff", "skip": 0.30, "loss": 1.20},
        {"name": "dominated-on-both", "skip": 0.20, "loss": 1.30},
        {"name": "same-skip-worse-loss", "skip": 0.30, "loss": 1.40},
        {"name": "same-loss-less-skip", "skip": 0.20, "loss": 1.20},
    ]

    frontier = plot_report04._nondominated_skip_loss_points(points)

    assert [point["name"] for point in frontier] == ["low-loss", "tradeoff"]


def test_propagation_helpers_pool_integer_counts_before_dividing() -> None:
    method = {
        "label": "Example",
        "matmuls": [
            {"name": "qkv", "layer": 0, "zero_count": 1, "total": 2},
            {"name": "qkv", "layer": 1, "zero_count": 1, "total": 8},
            {"name": "w1", "layer": 0, "zero_count": 8, "total": 10},
            {"name": "w1", "layer": 1, "zero_count": 0, "total": 10},
        ],
    }
    row_specs = (("qkv", "QKV"), ("w1", "W1"))

    matrix = plot_report04._propagation_matrix(method, "matmuls", row_specs, num_layers=2)

    assert matrix[0] == pytest.approx([50.0, 12.5, 20.0])
    assert matrix[1] == pytest.approx([80.0, 0.0, 40.0])
    assert plot_report04._propagation_weighted_fraction(method, "matmuls") == pytest.approx(
        100.0 / 3.0
    )


def test_histogram_density_uses_stored_total_and_bin_widths() -> None:
    layer = {"counts": [2, 6], "total": 20}
    edges = [0.0, 1.0, 3.0]

    density = plot_common._histogram_density(layer, edges)

    assert density == pytest.approx([0.10, 0.15])
    integrated_fraction = density[0] * 1.0 + density[1] * 2.0
    assert integrated_fraction == pytest.approx(sum(layer["counts"]) / layer["total"])


def test_histogram_center_window_mass_uses_stored_counts() -> None:
    layer = {"counts": [2, 6, 2], "total": 20}
    edges = [-1.0, -0.1, 0.1, 1.0]

    fraction = plot_common._histogram_center_window_mass(layer, edges, threshold=0.1)

    assert fraction == pytest.approx(6 / 20)
