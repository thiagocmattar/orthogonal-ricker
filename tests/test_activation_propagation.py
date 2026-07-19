from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper_exp.activation_propagation import (
    ACTIVATION_STAGE_ORDER,
    ENDPOINT_ZERO_STAGES,
    MATMUL_STAGE_ORDER,
    _PropagationAccumulator,
    _architecture_metadata,
    _capture_model_propagation,
    _endpoint_summary,
    _exact_zero_counts,
    _find_source_run,
    _linear_zero_product_counts,
    _patched_eager_attention,
    _post_qkv_relu_metadata,
    _probability_value_zero_product_counts,
    _qk_zero_product_counts,
    _source_checkpoint,
    _split_fused_qkv_projection,
    _validate_requested_validation_cache,
    _validate_shared_validation_cache,
    _valid_causal_exact_zero_counts,
)
from paper_exp.modeling import (
    FixedOneSidedThreshold,
    adaptive_threshold_parameter_items,
    apply_mlp_hidden_gate,
    apply_post_layernorm_relu,
    apply_post_qkv_relu,
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


def test_explicit_source_run_pins_the_requested_run(tmp_path: Path) -> None:
    config_id = "118-example"
    experiment_dir = tmp_path / "results" / config_id
    for run_id in ("001-first", "002-latest"):
        run_dir = experiment_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"config_id": config_id, "run_id": run_id}),
            encoding="utf-8",
        )

    selected = {"config_id": config_id, "run_id": "001-first"}

    assert _find_source_run({"output": {"dir": str(tmp_path / "results")}}, selected) == (
        experiment_dir / "001-first"
    )


def test_incomplete_source_checkpoint_override_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    selected = {"checkpoint_path": str(checkpoint)}

    with pytest.raises(ValueError, match="allow_incomplete_source"):
        _source_checkpoint(selected, {"config_id": "failed-run"})

    selected["allow_incomplete_source"] = True
    assert _source_checkpoint(selected, {"config_id": "failed-run"}) == checkpoint


@pytest.mark.parametrize(
    ("active_sites", "placement", "topology_id", "targetable_stages"),
    (
        (set(), None, "A0", set()),
        ({"h"}, None, "A1-H", {"mlp_w2"}),
        ({"a", "m", "h"}, None, "A3", {"qkv_projection", "mlp_w1", "mlp_w2"}),
        (
            {"a", "m", "h", "q", "k", "v"},
            "pre_rope",
            "A6-PRE",
            set(MATMUL_STAGE_ORDER),
        ),
        (
            {"a", "m", "h", "q", "k", "v"},
            "post_rope",
            "A6-POST",
            set(MATMUL_STAGE_ORDER),
        ),
    ),
)
def test_dynamic_endpoint_summary_covers_campaign_anchor_topologies(
    active_sites: set[str],
    placement: str | None,
    topology_id: str,
    targetable_stages: set[str],
) -> None:
    model, layers, post_qkv = _fake_architecture(
        active_sites=active_sites,
        placement=placement,
    )
    architecture = _architecture_metadata(
        model,
        layers=layers,
        post_qkv_relu=post_qkv,
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
                    "zero_count": total // 2 if stage in targetable_stages else 0,
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
    ceiling = sum(row["total"] for row in matmul_rows if row["name"] in targetable_stages)
    model_total = (
        block_total
        + sequences
        * architecture["sequence_length"]
        * architecture["hidden_size"]
        * architecture["vocab_size"]
    )
    assert architecture["topology_id"] == topology_id
    assert architecture["intermediate_size"] == 24  # Deliberately not hard-coded as 4d.
    assert set(endpoint["targetable_matmul_stages"]) == targetable_stages
    assert endpoint["architecture_ceiling_zero_product_count"] == ceiling
    assert endpoint["R_block_max"] == pytest.approx(ceiling / block_total)
    assert endpoint["R_model_max"] == pytest.approx(ceiling / model_total)
    if ceiling:
        assert endpoint["U_arch"] == pytest.approx(0.5)
    else:
        assert endpoint["U_arch"] is None
        assert endpoint["R_block"] == 0.0
        assert endpoint["R_model"] == 0.0


def test_dynamic_architecture_recognizes_fixed_one_sided_hidden_gate() -> None:
    model, layers, post_qkv = _fake_architecture(
        active_sites={"a", "m", "h"},
        placement=None,
    )
    for layer in layers:
        layer.attention_input_relu = FixedOneSidedThreshold(0.1)
        layer.mlp_input_relu = FixedOneSidedThreshold(0.1)
        layer.mlp.act = FixedOneSidedThreshold(0.1)

    architecture = _architecture_metadata(
        model,
        layers=layers,
        post_qkv_relu=post_qkv,
        block_size=4,
        torch=torch,
    )

    assert architecture["topology_id"] == "A3"
    assert architecture["active_gate_sites"] == ["a", "m", "h"]
    assert architecture["gate_specs"] == {
        "a": {"gate_family": "gplus", "gate_type": "one_sided_threshold", "kappa": 0.1},
        "m": {"gate_family": "gplus", "gate_type": "one_sided_threshold", "kappa": 0.1},
        "h": {"gate_family": "gplus", "gate_type": "one_sided_threshold", "kappa": 0.1},
        "q": None,
        "k": None,
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
        "tokens_sha256",
    ),
)
def test_shared_validation_cache_rejects_conflicting_optional_identity(
    field: str,
) -> None:
    reference = {
        "tokens_path": "selection/tokens.int32.bin",
        "block_size": 2048,
        "tokens": 311_739,
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
        _validate_shared_validation_cache(manifests, reference)


def test_shared_validation_cache_allows_missing_historical_optional_identity() -> None:
    reference = {
        "tokens_path": "validation/tokens.int32.bin",
        "block_size": 2048,
        "tokens": 692_224,
        "tokens_sha256": "a" * 64,
    }
    historical = {
        "tokens_path": reference["tokens_path"],
        "block_size": reference["block_size"],
        "tokens": reference["tokens"],
    }
    manifests = [
        {"config_id": "new", "tokenized_data": {"validation": reference}},
        {"config_id": "historical", "tokenized_data": {"validation": historical}},
    ]

    _validate_shared_validation_cache(manifests, reference)


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
    matmuls = accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
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
    assert by_name["query_projection_output"]["available"] is True
    assert by_name["query_qk_input"]["available"] is True
    assert by_name["value_pv_input"]["available"] is True

    if placement is None:
        assert architecture_metadata["topology_id"] == "A3"
        assert 0.0 < endpoint["R_block_max"] < 1.0
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

    assert architecture_metadata["topology_id"] == (
        "A6-PRE" if placement == "pre_rope" else "A6-POST"
    )
    assert endpoint["R_block_max"] == pytest.approx(1.0)
    assert endpoint["R_model_max"] < 1.0
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


def test_real_gpt_neox_diagnostic_preserves_fixed_gplus_metadata_and_ceiling() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers.models.gpt_neox import modeling_gpt_neox

    gate = {"gate_type": "one_sided_threshold", "kappa": 0.1}
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
    architecture.post_layernorm_gate = dict(gate)
    architecture.mlp_hidden_gate = dict(gate)
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
        **gate,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
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
    matmuls = accumulator.rows("matmuls", MATMUL_STAGE_ORDER, num_layers=1)
    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
        block_size=3,
        torch=torch,
    )
    endpoint = _endpoint_summary(
        architecture=architecture_metadata,
        activation_rows=activations,
        matmul_rows=matmuls,
        validation_tokens=3,
    )

    assert post_qkv["gate_family"] == "gplus"
    assert post_qkv["gate_type"] == "one_sided_threshold"
    assert post_qkv["kappa"] == pytest.approx(0.1)
    assert architecture_metadata["topology_id"] == "A6-POST"
    assert all(
        spec == {"gate_family": "gplus", "gate_type": "one_sided_threshold", "kappa": 0.1}
        for spec in architecture_metadata["gate_specs"].values()
    )
    assert endpoint["active_gate_sites"] == ["a", "m", "h", "q", "k", "v"]
    assert endpoint["targetable_matmul_stages"] == MATMUL_STAGE_ORDER
    assert endpoint["R_block_max"] == pytest.approx(1.0)
    assert endpoint["zero_sites"]["z_h"]["available"] is True


def test_dynamic_architecture_preserves_nonuniform_learned_a6_gate_metadata() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    gate = {
        "gate_type": "learned_one_sided_threshold",
        "kappa_init": 0.1,
        "kappa_scope": "per_layer_site",
        "threshold_scale": "absolute",
        "surrogate": "hard_forward_soft_backward",
        "temperature": 0.03,
    }
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        use_parallel_residual=True,
    )
    architecture.post_layernorm_relu = True
    architecture.post_layernorm_gate = dict(gate)
    architecture.mlp_hidden_gate = dict(gate)
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": True,
        "qk_placement": "post_rope",
        **gate,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    apply_post_qkv_relu(model, torch=torch)
    with torch.no_grad():
        for index, (_name, parameter) in enumerate(adaptive_threshold_parameter_items(model)):
            parameter.add_(0.05 * index)

    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
        block_size=4,
        torch=torch,
    )

    assert metadata["topology_id"] == "A6-POST"
    assert post_qkv["kappa"] is None
    assert metadata["gate_specs"]["a"]["kappa_uniform"] is False
    assert len(metadata["gate_specs_per_layer"]) == 2
    observed_keys = {
        row[site]["parameter_key"]
        for row in metadata["gate_specs_per_layer"]
        for site in ("a", "m", "h", "q", "k", "v")
    }
    assert observed_keys == {
        f"layer_{layer}__{site}"
        for layer in range(2)
        for site in ("a", "m", "h", "q", "k", "v")
    }
    observed_kappas = [
        row[site]["kappa"]
        for row in metadata["gate_specs_per_layer"]
        for site in ("a", "m", "h", "q", "k", "v")
    ]
    assert len(set(observed_kappas)) == 12


def test_dynamic_architecture_preserves_nonuniform_learned_a3_branch_metadata() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    gate = {
        "gate_type": "learned_one_sided_threshold",
        "kappa_init": 0.1,
        "kappa_scope": "per_layer_site",
        "threshold_scale": "rms_relative",
        "surrogate": "hard_forward_soft_backward",
        "temperature": 0.03,
    }
    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        use_parallel_residual=True,
    )
    architecture.post_layernorm_relu = True
    architecture.post_layernorm_gate = dict(gate)
    architecture.mlp_hidden_gate = dict(gate)
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)
    with torch.no_grad():
        for index, (_name, parameter) in enumerate(adaptive_threshold_parameter_items(model)):
            parameter.add_(0.03 * index)

    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
        block_size=4,
        torch=torch,
    )

    assert metadata["topology_id"] == "A3"
    assert metadata["gate_specs"]["a"]["threshold_scale"] == "rms_relative"
    assert metadata["gate_specs"]["a"]["rms_epsilon"] == pytest.approx(1e-8)
    assert metadata["gate_specs"]["a"]["kappa"] is None
    assert all(row["q"] is None for row in metadata["gate_specs_per_layer"])


def test_branch_metadata_allows_distinct_fixed_thresholds_at_distinct_sites() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
        hidden_act="relu",
        use_parallel_residual=True,
    )
    architecture.post_layernorm_relu = True
    architecture.post_layernorm_gate = {
        "gate_type": "one_sided_threshold",
        "kappa": 0.1,
    }
    architecture.mlp_hidden_gate = {
        "gate_type": "one_sided_threshold",
        "kappa": 0.2,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_layernorm_relu(model, torch=torch)
    apply_mlp_hidden_gate(model, torch=torch)

    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
        block_size=4,
        torch=torch,
    )

    assert metadata["gate_specs"]["a"]["kappa"] == pytest.approx(0.1)
    assert metadata["gate_specs"]["m"]["kappa"] == pytest.approx(0.1)
    assert metadata["gate_specs"]["h"]["kappa"] == pytest.approx(0.2)


def test_post_qkv_metadata_keeps_fixed_kappa_consistency_check() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
    )
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": False,
        "value": False,
        "qk_placement": "post_rope",
        "gate_type": "one_sided_threshold",
        "kappa": 0.1,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_qkv_relu(model, torch=torch)
    model.gpt_neox.layers[1].attention.query_relu = FixedOneSidedThreshold(0.2)

    with pytest.raises(ValueError, match="family and kappa"):
        _post_qkv_relu_metadata(list(model.gpt_neox.layers))


def test_post_qkv_metadata_rejects_corrupt_learned_parameter_key() -> None:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    architecture = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.5,
    )
    architecture.post_qkv_relu = {
        "enabled": True,
        "query": True,
        "key": True,
        "value": False,
        "qk_placement": "post_rope",
        "gate_type": "learned_symmetric_threshold",
        "kappa_init": 0.1,
        "kappa_scope": "per_site",
        "threshold_scale": "absolute",
        "temperature": 0.03,
    }
    model = GPTNeoXForCausalLM(architecture)
    apply_post_qkv_relu(model, torch=torch)
    model.gpt_neox.layers[1].attention.query_relu.parameter_key = "wrong"

    with pytest.raises(ValueError, match="parameter key mismatch"):
        _post_qkv_relu_metadata(list(model.gpt_neox.layers))


@pytest.mark.parametrize(
    ("hidden_act", "hidden_relu_available"),
    (("gelu", False), ("relu", True)),
)
def test_real_gpt_neox_diagnostic_supports_stock_and_mlp_relu_checkpoints(
    hidden_act: str,
    hidden_relu_available: bool,
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
        hidden_act=hidden_act,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
        use_parallel_residual=True,
    )
    model = GPTNeoXForCausalLM(architecture)
    model.eval()
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
    post_qkv = _post_qkv_relu_metadata(list(model.gpt_neox.layers))
    architecture_metadata = _architecture_metadata(
        model,
        layers=list(model.gpt_neox.layers),
        post_qkv_relu=post_qkv,
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
    for name in ("attention_input_relu", "mlp_input_relu"):
        assert activations_by_name[name] == {
            "name": name,
            "layer": 0,
            "available": False,
            "unavailable_reason": "post_layernorm_relu_absent",
            "zero_count": None,
            "total": None,
            "exact_zero_fraction": None,
        }

    hidden_row = activations_by_name["mlp_hidden_relu"]
    assert hidden_row["available"] is hidden_relu_available
    assert endpoint["zero_sites"]["z_h"]["available"] is hidden_relu_available
    if hidden_relu_available:
        assert architecture_metadata["topology_id"] == "A1-H"
        assert endpoint["targetable_matmul_stages"] == ["mlp_w2"]
        assert endpoint["R_block_max"] > 0.0
        expected_zero_count, expected_total = _exact_zero_counts(
            observed_inputs["mlp_w2"], torch=torch
        )
        assert hidden_row["zero_count"] == expected_zero_count
        assert hidden_row["total"] == expected_total
    else:
        assert architecture_metadata["topology_id"] == "A0"
        assert endpoint["targetable_matmul_stages"] == []
        assert endpoint["R_block_max"] == 0.0
        assert endpoint["R_model_max"] == 0.0
        assert endpoint["U_arch"] is None
        assert hidden_row["unavailable_reason"] == "mlp_hidden_relu_absent"
        assert hidden_row["zero_count"] is None

    output_widths = {
        "qkv_projection": int(layer.attention.query_key_value.out_features),
        "mlp_w1": int(layer.mlp.dense_h_to_4h.out_features),
        "mlp_w2": int(layer.mlp.dense_4h_to_h.out_features),
    }
    for name, output_width in output_widths.items():
        expected_zero_count, expected_total = _linear_zero_product_counts(
            observed_inputs[name],
            output_features=output_width,
            torch=torch,
        )
        assert matmuls_by_name[name]["available"] is True
        assert matmuls_by_name[name]["zero_count"] == expected_zero_count
        assert matmuls_by_name[name]["total"] == expected_total

    for name in ("query_qk_input", "key_qk_input", "value_pv_input"):
        assert activations_by_name[name]["available"] is True
    for name in ("qk_scores", "probability_value", "attention_output_projection"):
        assert matmuls_by_name[name]["available"] is True


def _fake_architecture(
    *, active_sites: set[str], placement: str | None
) -> tuple[object, list[object], dict[str, object]]:
    def linear(in_features: int, out_features: int) -> SimpleNamespace:
        return SimpleNamespace(in_features=in_features, out_features=out_features)

    layers = []
    for _ in range(2):
        attention = SimpleNamespace(
            query_key_value=linear(8, 24),
            dense=linear(8, 8),
            config=SimpleNamespace(num_attention_heads=2),
            head_size=4,
        )
        mlp = SimpleNamespace(
            dense_h_to_4h=linear(8, 24),
            dense_4h_to_h=linear(24, 8),
            act=torch.nn.ReLU() if "h" in active_sites else torch.nn.GELU(),
        )
        layers.append(
            SimpleNamespace(
                attention=attention,
                mlp=mlp,
                attention_input_relu=torch.nn.ReLU() if "a" in active_sites else None,
                mlp_input_relu=torch.nn.ReLU() if "m" in active_sites else None,
            )
        )

    class FakeModel:
        config = SimpleNamespace(hidden_act="relu" if "h" in active_sites else "gelu")

        @staticmethod
        def get_output_embeddings() -> SimpleNamespace:
            return SimpleNamespace(weight=torch.empty(40, 8))

    post_qkv = {
        "enabled": bool(active_sites & {"q", "k", "v"}),
        "query": "q" in active_sites,
        "key": "k" in active_sites,
        "value": "v" in active_sites,
        "qk_placement": placement,
        "layers": [],
    }
    return FakeModel(), layers, post_qkv


def _fake_endpoint_activation_rows(
    active_sites: set[str], *, num_layers: int
) -> list[dict[str, object]]:
    gated_stage_sites = {
        "attention_input_relu": "a",
        "mlp_input_relu": "m",
        "mlp_hidden_relu": "h",
        "query_gate_output": "q",
        "key_gate_output": "k",
        "value_gate_output": "v",
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
