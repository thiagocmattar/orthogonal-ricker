from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from paper_exp.diagnostics.logical_products import (
    linear_zero_product_counts,
    probability_value_zero_product_counts,
    qk_zero_product_counts,
)
from paper_exp.diagnostics.propagation_summary import (
    ACTIVATION_STAGE_LABELS,
    ACTIVATION_STAGE_ORDER,
    ENDPOINT_ZERO_STAGES,
    MATMUL_STAGE_ORDER,
    _attention_gate_metadata,
    _architecture_metadata,
    _endpoint_summary,
)
from paper_exp.diagnostics.propagation import _validate_requested_validation_cache
from paper_exp.diagnostics.propagation_capture import (
    _PropagationAccumulator,
    _capture_model_propagation,
    _exact_zero_counts,
    _patched_eager_attention,
    _split_fused_qkv_projection,
    _valid_causal_exact_zero_counts,
)
from paper_exp.modeling import (
    FixedOneSidedThreshold,
    apply_activation_topology,
)
from paper_exp.diagnostics.sources import validate_shared_validation_cache
from paper_exp.topology import SITE_ALIAS_ORDER, resolve_topology


def test_linear_zero_product_counts_scale_input_zeros_by_output_width() -> None:
    value = torch.tensor([[0.0, 1.0, 0.0, -2.0]])

    assert linear_zero_product_counts(value, output_features=3, torch=torch) == (6, 12)


def test_qk_zero_product_counts_use_actual_valid_causal_pairs() -> None:
    query = torch.tensor([[[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])

    assert qk_zero_product_counts(query, key, torch=torch) == (10, 12)


def test_probability_value_counts_exclude_future_causal_positions() -> None:
    probabilities = torch.tensor(
        [[[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]]]
    )
    value = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]])

    assert _valid_causal_exact_zero_counts(probabilities, torch=torch) == (2, 6)
    assert probability_value_zero_product_counts(
        probabilities,
        value,
        torch=torch,
        query_chunk_size=1,
    ) == (8, 12)


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
        "activations", "q_pre_gate_input", 0, "q_pre_gate_absent"
    )

    assert accumulator.rows(
        "activations", ["q_pre_gate_input"], num_layers=1
    ) == [
        {
            "name": "q_pre_gate_input",
            "layer": 0,
            "available": False,
            "unavailable_reason": "q_pre_gate_absent",
            "zero_count": None,
            "total": None,
            "exact_zero_fraction": None,
        }
    ]


def test_stage_vocabulary_uses_generic_gates_and_exact_qk_placements() -> None:
    assert not any("_relu" in stage for stage in ACTIVATION_STAGE_ORDER)
    assert {
        "attention_input_gate",
        "mlp_input_gate",
        "mlp_hidden_gate",
        "q_pre_gate_output",
        "k_pre_gate_output",
        "q_post_gate_output",
        "k_post_gate_output",
        "v_gate_output",
    }.issubset(ACTIVATION_STAGE_ORDER)
    assert "q_post" in ACTIVATION_STAGE_LABELS["query_qk_input"]
    assert "k_post" in ACTIVATION_STAGE_LABELS["key_qk_input"]


def test_endpoint_summary_reduces_direct_operation_counters() -> None:
    topology_id = "A3"
    active_sites = set(resolve_topology(topology_id).active_sites)
    model, layers, attention_gates = _fake_architecture(topology_id=topology_id)
    architecture = _architecture_metadata(
        model,
        layers=layers,
        attention_gates=attention_gates,
        block_size=4,
        torch=torch,
    )
    sequences = 2
    matmul_rows = []
    for layer in range(architecture["num_layers"]):
        for stage in MATMUL_STAGE_ORDER:
            total = architecture["operation_products_per_sequence_per_layer"][stage] * sequences
            matmul_rows.append(
                {
                    "name": stage,
                    "layer": layer,
                    "available": True,
                    "zero_count": total // 2,
                    "total": total,
                }
            )
    activation_rows = _fake_endpoint_activation_rows(
        active_sites, num_layers=architecture["num_layers"]
    )

    endpoint = _endpoint_summary(
        architecture=architecture,
        activation_rows=activation_rows,
        matmul_rows=matmul_rows,
        validation_tokens=sequences * architecture["sequence_length"],
    )

    block_total = sum(row["total"] for row in matmul_rows)
    block_zeros = sum(row["zero_count"] for row in matmul_rows)
    model_total = (
        block_total
        + sequences
        * architecture["sequence_length"]
        * architecture["hidden_size"]
        * architecture["vocab_size"]
    )
    assert architecture["intermediate_size"] == 24  # Deliberately not hard-coded as 4d.
    assert endpoint["block_zero_product_count"] == block_zeros
    assert endpoint["block_product_count"] == block_total
    assert endpoint["model_product_count"] == model_total
    assert endpoint["R_block"] == pytest.approx(block_zeros / block_total)
    assert endpoint["R_model"] == pytest.approx(block_zeros / model_total)
    assert set(endpoint["per_operation"]) == set(MATMUL_STAGE_ORDER)


def test_dynamic_architecture_recognizes_fixed_one_sided_hidden_gate() -> None:
    model, layers, attention_gates = _fake_architecture(
        topology_id="A3",
        gate_factory=lambda: FixedOneSidedThreshold(0.1),
        site_gate={"operator": "one_sided_threshold", "kappa": 0.1},
    )

    architecture = _architecture_metadata(
        model,
        layers=layers,
        attention_gates=attention_gates,
        block_size=4,
        torch=torch,
    )

    assert architecture["topology_id"] == "A3"
    assert architecture["active_sites"] == ["a", "m", "h"]
    assert architecture["gate_specs"] == {
        "a": {"gate_family": "gplus", "operator": "one_sided_threshold", "kappa": 0.1},
        "m": {"gate_family": "gplus", "operator": "one_sided_threshold", "kappa": 0.1},
        "h": {"gate_family": "gplus", "operator": "one_sided_threshold", "kappa": 0.1},
        "q_pre": None,
        "k_pre": None,
        "q_post": None,
        "k_post": None,
        "v": None,
    }


def test_named_partition_diagnostic_requires_the_complete_matching_cache() -> None:
    validation = {
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "partition_hash": "a" * 64,
        "max_documents": 500,
        "eval_batches": None,
    }
    metadata = {
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "source_document_indices_sha256": "a" * 64,
        "max_documents": 500,
    }

    _validate_requested_validation_cache(validation, metadata)

    with pytest.raises(ValueError, match="complete partition"):
        _validate_requested_validation_cache({**validation, "eval_batches": 2}, metadata)
    with pytest.raises(ValueError, match="partition hash"):
        _validate_requested_validation_cache(
            {**validation, "partition_hash": "b" * 64}, metadata
        )


@pytest.mark.parametrize(
    "field",
    (
        "partition",
        "partition_scheme",
        "partition_seed",
        "source_document_indices_sha256",
        "dtype",
        "tokens_bytes",
        "tokens_sha256",
    ),
)
def test_shared_validation_cache_rejects_conflicting_optional_identity(
    field: str,
) -> None:
    reference = {
        "tokens_path": "selection/tokens.int32.bin",
        "dtype": "int32",
        "block_size": 2048,
        "tokens": 311_739,
        "tokens_bytes": 311_739 * 4,
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "source_document_indices_sha256": "a" * 64,
        "tokens_sha256": "b" * 64,
    }
    conflicting = {**reference, field: "different"}
    manifests = [
        {"config_id": "one", "tokenized_data": {"validation": reference}},
        {"config_id": "two", "tokenized_data": {"validation": conflicting}},
    ]

    with pytest.raises(ValueError, match=field):
        validate_shared_validation_cache(manifests, reference)


def test_shared_validation_cache_rejects_missing_file_hash() -> None:
    reference = {
        "tokens_path": "validation/tokens.int32.bin",
        "dtype": "int32",
        "block_size": 2048,
        "tokens": 692_224,
        "tokens_bytes": 692_224 * 4,
        "tokens_sha256": "a" * 64,
    }
    incomplete = {
        "tokens_path": reference["tokens_path"],
        "dtype": reference["dtype"],
        "block_size": reference["block_size"],
        "tokens": reference["tokens"],
        "tokens_bytes": reference["tokens_bytes"],
    }
    manifests = [
        {"config_id": "new", "tokenized_data": {"validation": reference}},
        {"config_id": "incomplete", "tokenized_data": {"validation": incomplete}},
    ]

    with pytest.raises(ValueError, match="tokens_sha256"):
        validate_shared_validation_cache(manifests, reference)


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
        qk_zero_product_counts(query, key, torch=torch)
    )


@pytest.mark.parametrize(
    ("topology_id", "placement"),
    (("A3", None), ("A6-PRE", "pre_rope"), ("A6-POST", "post_rope")),
)
def test_real_gpt_neox_diagnostic_preserves_gate_placement_and_unavailable_rows(
    topology_id: str,
    placement: str | None,
) -> None:
    from transformers.models.gpt_neox import modeling_gpt_neox

    model = _real_gpt_neox_model(topology_id=topology_id)
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
    matmuls = accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    attention_gates = _attention_gate_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        attention_gates=attention_gates,
        block_size=3,
        torch=torch,
    )
    endpoint = _endpoint_summary(
        architecture=architecture_metadata,
        activation_rows=activations,
        matmul_rows=matmuls,
        validation_tokens=3,
    )
    by_name = {row["name"]: row for row in activations}
    assert architecture_metadata["topology_id"] == topology_id
    assert architecture_metadata["active_sites"] == list(
        resolve_topology(topology_id).active_sites
    )
    for site in SITE_ALIAS_ORDER:
        assert by_name[site]["available"] is True
    assert by_name["q_post"]["zero_count"] == by_name["query_qk_input"]["zero_count"]
    assert by_name["k_post"]["zero_count"] == by_name["key_qk_input"]["zero_count"]
    assert by_name["v"]["zero_count"] == by_name["value_pv_input"]["zero_count"]

    if placement is None:
        for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
            for suffix in ("input", "output"):
                name = f"{alias}_gate_{suffix}"
                assert by_name[name]["available"] is False
                assert by_name[name]["unavailable_reason"] == f"{alias}_gate_absent"
        assert all(
            row["available"] is False
            for row in accumulator.rope_survival_rows(num_layers=1)
        )
        return

    q_alias = "q_pre" if placement == "pre_rope" else "q_post"
    k_alias = "k_pre" if placement == "pre_rope" else "k_post"
    assert by_name[f"{q_alias}_gate_output"]["available"] is True
    assert by_name[f"{k_alias}_gate_output"]["available"] is True
    assert by_name["v_gate_output"]["available"] is True
    assert by_name["v_gate_output"]["zero_count"] == by_name["v"]["zero_count"]
    if placement == "post_rope":
        assert (
            by_name["q_post_gate_output"]["zero_count"]
            == by_name["q_post"]["zero_count"]
        )
        assert (
            by_name["k_post_gate_output"]["zero_count"]
            == by_name["k_post"]["zero_count"]
        )
        assert all(
            row["available"] is False
            for row in accumulator.rope_survival_rows(num_layers=1)
        )
    else:
        assert by_name["q_pre_gate_output"]["zero_count"] == by_name["q_pre"]["zero_count"]
        assert by_name["k_pre_gate_output"]["zero_count"] == by_name["k_pre"]["zero_count"]
        assert all(
            row["available"] is True
            for row in accumulator.rope_survival_rows(num_layers=1)
        )


def test_real_gpt_neox_diagnostic_preserves_fixed_gplus_metadata() -> None:
    from transformers.models.gpt_neox import modeling_gpt_neox

    gate = {"operator": "one_sided_threshold", "kappa": 0.1}
    model = _real_gpt_neox_model(
        topology_id="A6-POST",
        site_gate=gate,
    )
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
    matmuls = accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    attention_gates = _attention_gate_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        attention_gates=attention_gates,
        block_size=3,
        torch=torch,
    )
    endpoint = _endpoint_summary(
        architecture=architecture_metadata,
        activation_rows=activations,
        matmul_rows=matmuls,
        validation_tokens=3,
    )

    assert attention_gates["gate_family"] == "gplus"
    assert attention_gates["operator"] == "one_sided_threshold"
    assert attention_gates["kappa"] == pytest.approx(0.1)
    expected_spec = {
        "gate_family": "gplus",
        "operator": "one_sided_threshold",
        "kappa": 0.1,
    }
    assert all(
        architecture_metadata["gate_specs"][site] == expected_spec
        for site in architecture_metadata["active_sites"]
    )
    assert architecture_metadata["topology_id"] == "A6-POST"
    assert architecture_metadata["active_sites"] == [
        "a",
        "m",
        "h",
        "q_post",
        "k_post",
        "v",
    ]
    assert set(endpoint["per_operation"]) == set(MATMUL_STAGE_ORDER)
    assert endpoint["zero_sites"]["z_h"]["available"] is True


def test_architecture_metadata_rejects_distinct_gate_specs_within_one_topology() -> None:
    model = _real_gpt_neox_model(
        topology_id="A3",
        site_gate={"operator": "one_sided_threshold", "kappa": 0.1},
        num_hidden_layers=2,
    )
    model.gpt_neox.layers[1].mlp.act = FixedOneSidedThreshold(0.2)
    attention_gates = _attention_gate_metadata(list(model.gpt_neox.layers))

    with pytest.raises(ValueError, match="operator and kappa"):
        _architecture_metadata(
            model,
            layers=list(model.gpt_neox.layers),
            attention_gates=attention_gates,
            block_size=4,
            torch=torch,
        )


def test_attention_gate_metadata_keeps_fixed_kappa_consistency_check() -> None:
    model = _real_gpt_neox_model(
        topology_id="A4-Q",
        site_gate={"operator": "one_sided_threshold", "kappa": 0.1},
        num_hidden_layers=2,
    )
    model.gpt_neox.layers[1].attention.q_post_gate = FixedOneSidedThreshold(0.2)

    with pytest.raises(ValueError, match="operator and kappa"):
        _attention_gate_metadata(list(model.gpt_neox.layers))


@pytest.mark.parametrize(
    ("topology_id", "hidden_gate_available"),
    (("A0", False), ("A1-H", True)),
)
def test_real_gpt_neox_diagnostic_measures_canonical_sites_with_or_without_gates(
    topology_id: str,
    hidden_gate_available: bool,
) -> None:
    from transformers.models.gpt_neox import modeling_gpt_neox

    model = _real_gpt_neox_model(topology_id=topology_id)
    layer = model.gpt_neox.layers[0]
    observed_inputs: dict[str, torch.Tensor] = {}

    def record_input(name: str):
        def hook(_module, inputs):
            observed_inputs[name] = inputs[0].detach().clone()

        return hook

    recorder_handles = [
        layer.attention.query_key_value.register_forward_pre_hook(
            record_input("qkv_projection")
        ),
        layer.mlp.dense_h_to_4h.register_forward_pre_hook(record_input("mlp_w1")),
        layer.mlp.dense_4h_to_h.register_forward_pre_hook(record_input("mlp_w2")),
    ]
    accumulator = _PropagationAccumulator(torch)
    try:
        with _capture_model_propagation(
            model,
            accumulator=accumulator,
            modeling_gpt_neox=modeling_gpt_neox,
            torch=torch,
        ):
            with torch.no_grad():
                model.gpt_neox(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        for handle in recorder_handles:
            handle.remove()

    activations = accumulator.rows(
        "activations", ACTIVATION_STAGE_ORDER, num_layers=1
    )
    matmuls = accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    attention_gates = _attention_gate_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        attention_gates=attention_gates,
        block_size=3,
        torch=torch,
    )
    endpoint = _endpoint_summary(
        architecture=architecture_metadata,
        activation_rows=activations,
        matmul_rows=matmuls,
        validation_tokens=3,
    )
    activations_by_name = {row["name"]: row for row in activations}
    matmuls_by_name = {row["name"]: row for row in matmuls}

    for name in ("attention_layernorm_raw", "mlp_layernorm_raw"):
        assert activations_by_name[name]["available"] is True
    for name, reason in (
        ("attention_input_gate", "a_gate_absent"),
        ("mlp_input_gate", "m_gate_absent"),
    ):
        assert activations_by_name[name] == {
            "name": name,
            "layer": 0,
            "available": False,
            "unavailable_reason": reason,
            "zero_count": None,
            "total": None,
            "exact_zero_fraction": None,
        }

    hidden_gate_row = activations_by_name["mlp_hidden_gate"]
    assert hidden_gate_row["available"] is hidden_gate_available
    if hidden_gate_available:
        assert hidden_gate_row["zero_count"] == activations_by_name["h"]["zero_count"]
    else:
        assert hidden_gate_row["unavailable_reason"] == "h_gate_absent"
        assert hidden_gate_row["zero_count"] is None

    for site, observed_name in (
        ("a", "qkv_projection"),
        ("m", "mlp_w1"),
        ("h", "mlp_w2"),
    ):
        expected_zero_count, expected_total = _exact_zero_counts(
            observed_inputs[observed_name], torch=torch
        )
        assert activations_by_name[site]["zero_count"] == expected_zero_count
        assert activations_by_name[site]["total"] == expected_total
        assert endpoint["zero_sites"][f"z_{site}"]["available"] is True

    output_widths = {
        "qkv_projection": int(layer.attention.query_key_value.out_features),
        "mlp_w1": int(layer.mlp.dense_h_to_4h.out_features),
        "mlp_w2": int(layer.mlp.dense_4h_to_h.out_features),
    }
    for name, output_width in output_widths.items():
        expected_zero_count, expected_total = linear_zero_product_counts(
            observed_inputs[name],
            output_features=output_width,
            torch=torch,
        )
        assert matmuls_by_name[name]["available"] is True
        assert matmuls_by_name[name]["zero_count"] == expected_zero_count
        assert matmuls_by_name[name]["total"] == expected_total

    for name in (*SITE_ALIAS_ORDER, "query_qk_input", "key_qk_input", "value_pv_input"):
        assert activations_by_name[name]["available"] is True
    for name in ("qk_scores", "probability_value", "attention_output_projection"):
        assert matmuls_by_name[name]["available"] is True


def _real_gpt_neox_model(
    *,
    topology_id: str,
    site_gate: dict[str, object] | None = None,
    num_hidden_layers: int = 1,
) -> object:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    if topology_id != "A0" and site_gate is None:
        site_gate = {"operator": "relu"}
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        use_parallel_residual=True,
    )
    architecture.topology_id = topology_id
    architecture.site_gate = site_gate
    model = GPTNeoXForCausalLM(architecture)
    apply_activation_topology(model, torch=torch)
    model.eval()
    return model


def _fake_architecture(
    *,
    topology_id: str,
    gate_factory=lambda: torch.nn.ReLU(),
    site_gate: dict[str, object] | None = None,
) -> tuple[object, list[object], dict[str, object]]:
    def linear(in_features: int, out_features: int) -> SimpleNamespace:
        return SimpleNamespace(in_features=in_features, out_features=out_features)

    topology = resolve_topology(topology_id)
    active_sites = frozenset(topology.active_sites)
    if topology_id != "A0" and site_gate is None:
        site_gate = {"operator": "relu"}
    layers = []
    for _ in range(2):
        attention = SimpleNamespace(
            query_key_value=linear(8, 24),
            dense=linear(8, 8),
            config=SimpleNamespace(num_attention_heads=2),
            head_size=4,
            rotary_ndims=2,
            qk_gate_placement=topology.qk_placement,
        )
        for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
            if alias in active_sites:
                setattr(attention, f"{alias}_gate", gate_factory())
        mlp = SimpleNamespace(
            dense_h_to_4h=linear(8, 24),
            dense_4h_to_h=linear(24, 8),
            act=gate_factory() if "h" in active_sites else torch.nn.GELU(),
        )
        layers.append(
            SimpleNamespace(
                attention=attention,
                mlp=mlp,
                a_gate=gate_factory() if "a" in active_sites else None,
                m_gate=gate_factory() if "m" in active_sites else None,
            )
        )

    class FakeModel:
        config = SimpleNamespace(
            hidden_act="gelu",
            topology_id=topology_id,
            site_gate=site_gate,
        )
        gpt_neox = SimpleNamespace(layers=layers)

        @staticmethod
        def get_output_embeddings() -> SimpleNamespace:
            return SimpleNamespace(weight=torch.empty(40, 8))

    return FakeModel(), layers, _attention_gate_metadata(layers)


def _fake_endpoint_activation_rows(
    active_sites: set[str], *, num_layers: int
) -> list[dict[str, object]]:
    gated_stage_sites = {
        "attention_input_gate": "a",
        "mlp_input_gate": "m",
        "mlp_hidden_gate": "h",
        "q_pre_gate_output": "q_pre",
        "k_pre_gate_output": "k_pre",
        "q_post_gate_output": "q_post",
        "k_post_gate_output": "k_post",
        "v_gate_output": "v",
    }
    rows = []
    for layer in range(num_layers):
        for stage in ENDPOINT_ZERO_STAGES.values():
            required_site = gated_stage_sites.get(stage)
            available = required_site is None or required_site in active_sites
            rows.append(
                {
                    "name": stage,
                    "layer": layer,
                    "available": available,
                    "zero_count": 1 if available else None,
                    "total": 4 if available else None,
                }
            )
    return rows
