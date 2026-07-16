from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from paper_exp.activation_propagation import (
    ACTIVATION_STAGE_ORDER,
    MATMUL_STAGE_ORDER,
    _PropagationAccumulator,
    _capture_model_propagation,
    _linear_zero_product_counts,
    _patched_eager_attention,
    _probability_value_zero_product_counts,
    _qk_zero_product_counts,
    _split_fused_qkv_projection,
    _valid_causal_exact_zero_counts,
)
from paper_exp.modeling import apply_post_layernorm_relu, apply_post_qkv_relu


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
            "available": True,
            "zero_count": 3,
            "total": 10,
            "exact_zero_fraction": 0.3,
        }
    ]


def test_accumulator_emits_explicit_na_for_an_absent_gate() -> None:
    accumulator = _PropagationAccumulator(torch)
    accumulator.mark_unavailable(
        "activations", "query_gate_input", 0, "post_qkv_gate_absent"
    )

    assert accumulator.rows(
        "activations", ["query_gate_input"], num_layers=1
    ) == [
        {
            "name": "query_gate_input",
            "layer": 0,
            "available": False,
            "unavailable_reason": "post_qkv_gate_absent",
            "zero_count": None,
            "total": None,
            "exact_zero_fraction": None,
        }
    ]


def test_split_fused_qkv_projection_preserves_gpt_neox_per_head_layout() -> None:
    # Each token is [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1], not [all Q, all K, all V].
    fused = torch.tensor(
        [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            ]
        ],
        dtype=torch.float32,
    )

    query, key, value = _split_fused_qkv_projection(
        fused, num_heads=2, head_width=2
    )

    assert query.tolist() == [[[[1, 2], [13, 14]], [[7, 8], [19, 20]]]]
    assert key.tolist() == [[[[3, 4], [15, 16]], [[9, 10], [21, 22]]]]
    assert value.tolist() == [[[[5, 6], [17, 18]], [[11, 12], [23, 24]]]]


def test_pre_rope_survival_distinguishes_repopulation_and_passthrough() -> None:
    accumulator = _PropagationAccumulator(torch)
    accumulator.set_gate_metadata(
        0,
        qk_placement="pre_rope",
        query=True,
        key=False,
        value=True,
        rotary_dim=2,
        head_width=4,
    )
    before_rope = torch.tensor([[[[0.0, 2.0, 0.0, 3.0]]]])
    after_rope = torch.tensor([[[[2.0, 0.0, 0.0, 3.0]]]])

    accumulator.remember_gate_output("query", 0, before_rope)
    accumulator.add_rope_survival_from_actual_operand("query", 0, after_rope)
    rows = accumulator.rope_survival_rows(num_layers=1)
    rotary = next(
        row
        for row in rows
        if row["operand"] == "query" and row["region"] == "rotary"
    )
    passthrough = next(
        row
        for row in rows
        if row["operand"] == "query" and row["region"] == "passthrough"
    )

    assert rotary["preserved_zero_count"] == 0
    assert rotary["repopulated_zero_count"] == 1
    assert rotary["created_zero_count"] == 1
    assert rotary["zero_repopulation_fraction"] == 1.0
    assert passthrough["preserved_zero_count"] == 1
    assert passthrough["repopulated_zero_count"] == 0
    assert passthrough["created_zero_count"] == 0


def test_pre_rope_all_zero_pairs_use_rotate_half_coordinate_pairs() -> None:
    accumulator = _PropagationAccumulator(torch)
    accumulator.set_gate_metadata(
        0,
        qk_placement="pre_rope",
        query=True,
        key=False,
        value=True,
        rotary_dim=4,
        head_width=4,
    )
    # rotate_half pairs dimensions (0, 2) and (1, 3); only (0, 2) is all zero.
    value = torch.tensor([[[[0.0, 1.0, 0.0, 2.0]]]])
    accumulator.remember_gate_output("query", 0, value)
    accumulator.add_rope_survival_from_actual_operand("query", 0, value)

    query_row = next(
        row
        for row in accumulator.rope_pair_rows(num_layers=1)
        if row["operand"] == "query"
    )
    assert query_row["input_all_zero_pair_count"] == 1
    assert query_row["pair_total"] == 2
    assert query_row["input_all_zero_pair_fraction"] == 0.5


def test_post_rope_survival_is_explicitly_not_applicable() -> None:
    accumulator = _PropagationAccumulator(torch)
    accumulator.set_gate_metadata(
        0,
        qk_placement="post_rope",
        query=True,
        key=True,
        value=True,
        rotary_dim=2,
        head_width=4,
    )

    rows = accumulator.rope_survival_rows(num_layers=1)

    assert len(rows) == 4
    assert all(row["available"] is False for row in rows)
    assert {row["unavailable_reason"] for row in rows} == {"qk_gate_is_post_rope"}


def test_eager_instrumentation_counts_the_actual_post_gate_qk_and_pv_operands() -> None:
    def eager_forward(
        _module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **_kwargs,
    ):
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        return context, probabilities

    original = eager_forward
    modeling_gpt_neox = SimpleNamespace(eager_attention_forward=original)
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    module = SimpleNamespace(layer_idx=0)
    accumulator = _PropagationAccumulator(torch)
    accumulator.set_gate_metadata(
        0,
        qk_placement="post_rope",
        query=True,
        key=True,
        value=True,
        rotary_dim=2,
        head_width=2,
    )
    query = torch.tensor([[[[0.0, 1.0], [2.0, 0.0]]]])
    key = torch.tensor([[[[3.0, 0.0], [0.0, 4.0]]]])
    value = torch.tensor([[[[0.0, 5.0], [6.0, 0.0]]]])

    with _patched_eager_attention(
        model,
        accumulator=accumulator,
        modeling_gpt_neox=modeling_gpt_neox,
        torch=torch,
    ):
        assert model.config._attn_implementation == "eager"
        modeling_gpt_neox.eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask=None,
            scaling=1.0,
        )

    assert modeling_gpt_neox.eager_attention_forward is original
    assert model.config._attn_implementation == "sdpa"
    assert accumulator.activations[(0, "query_qk_input")] == [2, 4]
    assert accumulator.activations[(0, "key_qk_input")] == [2, 4]
    assert accumulator.activations[(0, "value_pv_input")] == [2, 4]
    assert accumulator.matmuls[(0, "qk_scores")] == list(
        _qk_zero_product_counts(query, key, torch=torch)
    )


@pytest.mark.parametrize("placement", [None, "pre_rope", "post_rope"])
def test_real_gpt_neox_diagnostic_preserves_gate_placement_and_legacy_na(
    placement: str | None,
) -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers.models.gpt_neox import modeling_gpt_neox

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        use_parallel_residual=True,
    )
    architecture.post_layernorm_relu = True
    if placement is not None:
        architecture.post_qkv_relu = {
            "enabled": True,
            "query": True,
            "key": True,
            "value": True,
            "qk_placement": placement,
        }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    model.eval()
    accumulator = _PropagationAccumulator(torch)

    with _capture_model_propagation(
        model,
        accumulator=accumulator,
        modeling_gpt_neox=modeling_gpt_neox,
        torch=torch,
    ):
        with torch.no_grad():
            model.gpt_neox(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)

    activations = accumulator.rows(
        "activations", ACTIVATION_STAGE_ORDER, num_layers=1
    )
    accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    by_name = {row["name"]: row for row in activations}
    assert by_name["query_projection_output"]["available"] is True
    assert by_name["query_qk_input"]["available"] is True
    assert by_name["value_pv_input"]["available"] is True

    if placement is None:
        for name in (
            "query_gate_input",
            "key_gate_input",
            "value_gate_input",
            "query_gate_output",
            "key_gate_output",
            "value_gate_output",
        ):
            assert by_name[name]["available"] is False
            assert by_name[name]["unavailable_reason"] == "post_qkv_gate_absent"
        assert all(
            row["available"] is False
            for row in accumulator.rope_survival_rows(num_layers=1)
        )
        return

    assert by_name["query_gate_output"]["available"] is True
    assert by_name["key_gate_output"]["available"] is True
    assert by_name["value_gate_output"]["available"] is True
    assert (
        by_name["value_gate_output"]["zero_count"]
        == by_name["value_pv_input"]["zero_count"]
    )
    if placement == "post_rope":
        assert (
            by_name["query_gate_output"]["zero_count"]
            == by_name["query_qk_input"]["zero_count"]
        )
        assert (
            by_name["key_gate_output"]["zero_count"]
            == by_name["key_qk_input"]["zero_count"]
        )
        assert all(
            row["available"] is False
            for row in accumulator.rope_survival_rows(num_layers=1)
        )
    else:
        assert all(
            row["available"] is True
            for row in accumulator.rope_survival_rows(num_layers=1)
        )
