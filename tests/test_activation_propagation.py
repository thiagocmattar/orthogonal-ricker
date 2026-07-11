from __future__ import annotations

import torch

from paper_exp.activation_propagation import (
    _PropagationAccumulator,
    _linear_zero_product_counts,
    _probability_value_zero_product_counts,
    _qk_zero_product_counts,
    _valid_causal_exact_zero_counts,
)


def test_linear_zero_product_counts_scale_input_zeros_by_output_width() -> None:
    value = torch.tensor([[0.0, 1.0, 0.0, -2.0]])

    assert _linear_zero_product_counts(value, output_features=3, torch=torch) == (6, 12)


def test_qk_zero_product_counts_use_actual_valid_causal_pairs() -> None:
    query = torch.tensor([[[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])

    assert _qk_zero_product_counts(query, key, torch=torch) == (10, 12)


def test_probability_value_counts_exclude_future_causal_positions() -> None:
    probabilities = torch.tensor(
        [[[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]]]
    )
    value = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]])

    assert _valid_causal_exact_zero_counts(probabilities, torch=torch) == (2, 6)
    assert _probability_value_zero_product_counts(probabilities, value, torch=torch) == (8, 12)


def test_accumulator_pools_integer_counts_before_forming_fraction() -> None:
    accumulator = _PropagationAccumulator(torch)
    accumulator.add_counts("activations", "value", 0, 1, 2)
    accumulator.add_counts("activations", "value", 0, 2, 8)

    assert accumulator.rows("activations", ["value"]) == [
        {
            "name": "value",
            "layer": 0,
            "zero_count": 3,
            "total": 10,
            "exact_zero_fraction": 0.3,
        }
    ]
