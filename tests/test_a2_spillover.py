from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

import paper_exp.plots.a2_spillover as a2
from paper_exp.plots.a2_spillover import (
    A2_SOURCES,
    A2SpilloverData,
    SiteReduction,
    build_a2_layerwise_distributions_figure,
    build_a2_site_distributions_figure,
    build_a2_spillover_response_figure,
    generate_a2_spillover_suite,
    reduce_density_layers,
    reduce_site_group,
    reduce_site_layers,
)
from paper_exp.plots.export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    publication_figure_issues,
)


def test_a2_site_reduction_is_count_first_and_rms_is_second_moment_pooled() -> None:
    source = A2_SOURCES[0]
    total = a2.EXPECTED_LAYER_TOTALS["h"]
    layers = []
    for layer_index in a2.LAYERS:
        exact = 10 + layer_index
        near = 20 + 2 * layer_index
        wide = 30 + 3 * layer_index
        layers.append(
            _layer_row(
                name=f"h.layer_{layer_index}",
                total=total,
                bin_count=2,
                exact=exact,
                near=near,
                wide=wide,
                rms=float(layer_index + 1),
            )
        )

    reduction = reduce_site_layers(
        layers,
        source=source,
        site="h",
        bin_count=2,
    )

    expected_total = total * len(a2.LAYERS)
    assert reduction.total == expected_total
    assert reduction.exact_zero_hits == sum(10 + index for index in a2.LAYERS)
    assert reduction.near_zero_0p01_fraction == pytest.approx(
        sum(20 + 2 * index for index in a2.LAYERS) / expected_total
    )
    assert reduction.near_zero_0p1_fraction == pytest.approx(
        sum(30 + 3 * index for index in a2.LAYERS) / expected_total
    )
    assert reduction.pooled_rms == pytest.approx(
        math.sqrt(sum(float(index + 1) ** 2 for index in a2.LAYERS) / 6.0)
    )


def test_a2_attention_group_is_count_first_with_unequal_site_totals() -> None:
    data = _synthetic_data(bin_count=4)
    source = A2_SOURCES[0]
    replacements = {
        "a": (10, 1),
        "q_post": (90, 45),
        "k_post": (20, 2),
        "v": (80, 8),
    }
    reductions = tuple(
        replace(
            row,
            total=replacements[row.site][0],
            near_zero_0p01_hits=replacements[row.site][1],
            near_zero_0p01_fraction=(
                replacements[row.site][1] / replacements[row.site][0]
            ),
        )
        if row.config_id == source.config_id and row.site in replacements
        else row
        for row in data.reductions
    )

    pooled = reduce_site_group(
        reductions,
        source=source,
        sites=a2.ATTENTION_SITES,
    )

    assert pooled.total == 200
    assert pooled.near_zero_0p01_hits == 56
    assert pooled.near_zero_0p01_fraction == pytest.approx(0.28)
    assert pooled.near_zero_0p01_fraction != pytest.approx(
        (0.1 + 0.5 + 0.1 + 0.1) / 4.0
    )


def test_a2_density_rebin_preserves_zero_and_tail_mass() -> None:
    layer = {
        "counts": [1, 2, 5, 2],
        "total": 14,
        "underflow": 3,
        "overflow": 1,
        "threshold_hits": {"0": 5},
    }

    reduction = reduce_density_layers(
        (layer,),
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        display_window=(-2.0, 2.0),
        rebin_factor=2,
    )

    assert reduction.edges == (-2.0, 0.0, 2.0)
    assert reduction.total == 14
    assert reduction.exact_zero_hits == 5
    assert reduction.nonzero_total == 9
    assert reduction.outside_stored_hits == 4
    assert reduction.outside_display_hits == 4
    assert reduction.exact_zero_fraction == pytest.approx(5 / 14)
    assert reduction.outside_display_fraction_nonzero == pytest.approx(4 / 9)
    integral = sum(
        density * (right - left)
        for density, left, right in zip(
            reduction.density,
            reduction.edges[:-1],
            reduction.edges[1:],
            strict=True,
        )
    )
    assert integral == pytest.approx(5 / 9)

    atom_only = reduce_density_layers(
        (
            {
                "counts": [0, 0, 10, 0],
                "total": 10,
                "underflow": 0,
                "overflow": 0,
                "threshold_hits": {"0": 10},
            },
        ),
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        display_window=(-2.0, 2.0),
        rebin_factor=2,
    )
    assert atom_only.nonzero_total == 0
    assert atom_only.density == (0.0, 0.0)
    assert atom_only.outside_display_fraction_nonzero is None


def test_a2_figures_pass_publication_checks_and_keep_distribution_atoms_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a2, "DENSITY_REBIN_FACTOR", 1)
    monkeypatch.setattr(
        a2,
        "DENSITY_WINDOWS",
        {site: a2.EXPECTED_RANGE for site in a2.DENSITY_SITES},
    )
    data = _synthetic_data(bin_count=4)
    response = build_a2_spillover_response_figure(data)
    layerwise = build_a2_layerwise_distributions_figure(data)
    pooled = build_a2_site_distributions_figure(data)
    try:
        assert publication_figure_issues(
            response, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
        assert publication_figure_issues(
            layerwise, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
        assert publication_figure_issues(
            pooled, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
        assert len(response.axes) == 1
        assert len(response.legends) == 0
        assert len(response.axes[0].collections[0].get_offsets()) == 5
        assert len(response.axes[0].texts) == 5
        assert len(layerwise.axes) == 36
        assert len(layerwise.legends) == 1
        assert len(layerwise.legends[0].get_texts()) == 2
        assert len(pooled.axes) == 18
        assert len(pooled.legends) == 0
        assert all(axis.get_yscale() == "linear" for axis in layerwise.axes)
        assert all(axis.get_yscale() == "linear" for axis in pooled.axes)
    finally:
        plt.close(response)
        plt.close(layerwise)
        plt.close(pooled)


def test_a2_suite_is_atomic_and_deterministic_from_compact_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(a2, "PROPAGATION_RUN_ID", "001-20260828-120000-fixture")
    monkeypatch.setattr(a2, "PROPAGATION_GIT_COMMIT", "f" * 40)
    monkeypatch.setattr(a2, "EXPECTED_BINS", 4)
    monkeypatch.setattr(a2, "DENSITY_REBIN_FACTOR", 1)
    monkeypatch.setattr(
        a2,
        "DENSITY_WINDOWS",
        {site: a2.EXPECTED_RANGE for site in a2.DENSITY_SITES},
    )
    _write_a2_fixture(tmp_path, bin_count=4)

    outputs = generate_a2_spillover_suite(tmp_path)
    first_bytes = tuple(path.read_bytes() for path in outputs)

    assert [path.name for path in outputs] == [
        "01-a2-spillover-response.pdf",
        "01-a2-spillover-response.png",
        "01-a2-spillover-response.md",
        "01-a2-spillover-response.provenance.json",
        "02-a2-layerwise-distributions.pdf",
        "02-a2-layerwise-distributions.png",
        "02-a2-layerwise-distributions.md",
        "02-a2-layerwise-distributions.provenance.json",
        "03-a2-site-distributions.pdf",
        "03-a2-site-distributions.png",
        "03-a2-site-distributions.md",
        "03-a2-site-distributions.provenance.json",
    ]
    response_markdown = outputs[2].read_text(encoding="utf-8")
    layerwise_markdown = outputs[6].read_text(encoding="utf-8")
    pooled_markdown = outputs[10].read_text(encoding="utf-8")
    assert "single-seed directional screen" in response_markdown
    assert "R_model" in response_markdown
    assert "R_block (%)" in response_markdown
    assert "Delta R_model (pp)" in response_markdown
    assert "diagnostic `019`" in response_markdown
    assert "BF16 eager attention" in response_markdown
    assert "non-monotonic" in response_markdown
    assert (
        "Every L1 cell has higher final validation loss than control"
        in response_markdown
    )
    assert "Complete per-site activation response" in response_markdown
    assert "count-preserving" in layerwise_markdown
    assert "lambda 1" in layerwise_markdown
    assert "Counts are never pooled across sites" in pooled_markdown
    response_provenance = json.loads(outputs[3].read_text(encoding="utf-8"))
    layerwise_provenance = json.loads(outputs[7].read_text(encoding="utf-8"))
    pooled_provenance = json.loads(outputs[11].read_text(encoding="utf-8"))
    assert response_provenance["cohort"]["diagnostic_run_id"] == a2.DIAGNOSTIC_RUN_ID
    assert (
        response_provenance["cohort"]["propagation_run_id"]
        == a2.PROPAGATION_RUN_ID
    )
    assert response_provenance["reduction"]["primary_threshold"] == 0.01
    assert response_provenance["reduction"]["attention_sites"] == list(
        a2.ATTENTION_SITES
    )
    assert layerwise_provenance["reduction"]["layers"] == list(a2.LAYERS)
    assert layerwise_provenance["reduction"]["exact_zero_atom_separate"] is True
    assert pooled_provenance["reduction"]["layers"] == "pooled within site"
    assert layerwise_provenance["reduction"]["density_sources"] == [
        A2_SOURCES[index].config_id for index in a2.LAYERWISE_SOURCE_INDICES
    ]
    assert pooled_provenance["reduction"]["density_sources"] == [
        A2_SOURCES[index].config_id for index in a2.POOLED_SOURCE_INDICES
    ]
    assert len(layerwise_provenance["reduction"]["panels"]) == 72
    assert len(pooled_provenance["reduction"]["panels"]) == 18
    assert len(response_provenance["logical_opportunities"]) == len(A2_SOURCES)
    assert response_provenance["reduction"]["logical_product_metric"]["not_a_speedup"] is True
    logical_metric = response_provenance["reduction"]["logical_product_metric"]
    assert logical_metric["execution"] == a2.EXPECTED_PROPAGATION_EXECUTION
    assert logical_metric["block_size"] == 2_048
    assert logical_metric["trailing_tokens_excluded"] == 443
    assert len(response_provenance["inputs"]) == 40

    assert generate_a2_spillover_suite(tmp_path) == outputs
    assert tuple(path.read_bytes() for path in outputs) == first_bytes


def test_a2_report_fails_closed_until_propagation_identity_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a2, "PROPAGATION_RUN_ID", None)
    monkeypatch.setattr(a2, "PROPAGATION_GIT_COMMIT", None)

    with pytest.raises(RuntimeError, match="has not been pinned"):
        a2._pinned_propagation_identity()


def test_a2_report_rejects_propagation_with_a_different_source_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(a2, "PROPAGATION_RUN_ID", "001-20260828-120000-fixture")
    monkeypatch.setattr(a2, "PROPAGATION_GIT_COMMIT", "f" * 40)
    monkeypatch.setattr(a2, "EXPECTED_BINS", 4)
    _write_a2_fixture(tmp_path, bin_count=4)
    recipe_path = (
        tmp_path
        / "experiments"
        / a2.TRANCHE_ID
        / "run"
        / f"{a2.PROPAGATION_CONFIG_ID}.yaml"
    )
    saved_path = (
        tmp_path
        / "experiments"
        / a2.TRANCHE_ID
        / "raw"
        / a2.PROPAGATION_CONFIG_ID
        / a2.PROPAGATION_RUN_ID
        / "config.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe["activation_propagation"]["selected_runs"].reverse()
    changed = yaml.safe_dump(recipe, sort_keys=False)
    recipe_path.write_text(changed, encoding="utf-8")
    saved_path.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="selected runs"):
        a2.load_a2_spillover(tmp_path)


def test_a2_percentage_formatters_do_not_round_non_boundary_values_to_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert a2._format_compact_percentage(0.0) == "0"
    assert a2._format_compact_percentage(1.0) == "100"
    assert a2._format_compact_percentage(0.000015) == "<.01"
    assert a2._format_compact_percentage(0.999798) == "99.98"
    assert a2._format_compact_percentage(0.999999) == ">99.99"
    assert a2._format_probability(0.999798) == "99.98%"
    assert a2._format_probability(0.999999) == ">99.99%"
    assert a2._format_precise_percentage(0.0) == "0.0000"
    assert a2._format_precise_percentage(0.000000025) == "<0.0001"
    assert a2._format_precise_percentage(0.943476) == "94.3476"

    monkeypatch.setattr(a2, "EXPECTED_BINS", 4)
    data = _synthetic_data(bin_count=4)
    data = replace(
        data,
        reductions=tuple(
            replace(row, exact_zero_fraction=0.000000025)
            if row.config_id == A2_SOURCES[0].config_id and row.site == "q_post"
            else row
            for row in data.reductions
        ),
    )
    markdown = a2.build_a2_response_markdown(data)
    assert "| Control | `q_post` | <0.0001 |" in markdown


def _synthetic_data(*, bin_count: int) -> A2SpilloverData:
    edges = tuple(
        a2.EXPECTED_RANGE[0]
        + index * (a2.EXPECTED_RANGE[1] - a2.EXPECTED_RANGE[0]) / bin_count
        for index in range(bin_count + 1)
    )
    methods = tuple(
        _method_payload(source, source_index, bin_count)
        for source_index, source in enumerate(A2_SOURCES)
    )
    reductions = tuple(
        reduce_site_layers(
            method["layers"],
            source=source,
            site=site,
            bin_count=bin_count,
        )
        for source, method in zip(A2_SOURCES, methods, strict=True)
        for site in a2.SITES
    )
    losses = tuple(
        a2.LossPoint(
            source.config_id,
            source.run_id,
            source.label,
            source.lambda_value,
            4.0 + 0.01 * index,
        )
        for index, source in enumerate(A2_SOURCES)
    )
    opportunities = tuple(
        _opportunity_point(source, index)
        for index, source in enumerate(A2_SOURCES)
    )
    return A2SpilloverData(reductions, losses, edges, methods, opportunities, ())


def _opportunity_point(
    source: a2.A2Source, source_index: int
) -> a2.LogicalOpportunityPoint:
    block_products = 600_000
    head_products = 1_200_000
    model_products = block_products + head_products
    block_zeros = (60_000, 59_400, 64_800, 58_800, 70_200, 75_600)[source_index]
    return a2.LogicalOpportunityPoint(
        config_id=source.config_id,
        run_id=source.run_id,
        label=source.label,
        lambda_value=source.lambda_value,
        block_zero_product_count=block_zeros,
        block_product_count=block_products,
        lm_head_product_count=head_products,
        model_product_count=model_products,
        R_block=block_zeros / block_products,
        R_model=block_zeros / model_products,
    )


def _write_a2_fixture(repository: Path, *, bin_count: int) -> None:
    source_root = Path(__file__).resolve().parents[1]
    run_root = repository / "experiments" / a2.TRANCHE_ID / "run"
    raw_root = repository / "experiments" / a2.TRANCHE_ID / "raw"
    figs_root = repository / "experiments" / a2.TRANCHE_ID / "figs"
    run_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    figs_root.mkdir(parents=True)

    for index, source in enumerate(A2_SOURCES):
        tracked = run_root / f"{source.config_id}.yaml"
        shutil.copyfile(
            source_root
            / "experiments"
            / a2.TRANCHE_ID
            / "run"
            / tracked.name,
            tracked,
        )
        run_dir = raw_root / source.config_id / source.run_id
        checkpoint = run_dir / "checkpoints" / "final"
        checkpoint.mkdir(parents=True)
        shutil.copyfile(tracked, run_dir / "config.yaml")
        (checkpoint / "model.safetensors").write_bytes(f"checkpoint-{index}".encode())
        manifest = {
            "status": "completed",
            "mode": "pretrain",
            "tranche_id": a2.TRANCHE_ID,
            "config_id": source.config_id,
            "run_id": source.run_id,
            "git_commit": a2.EXPECTED_TRAINING_GIT_COMMIT,
            "git_dirty": False,
            "config_sha256": source.config_sha256,
            "condition_fingerprint": source.condition_fingerprint,
            "case_group_id": source.group_id,
            "training_implementation_id": a2.EXPECTED_IMPLEMENTATION_ID,
            "seed": 0,
            "model_initialization_seed": 0,
            "data_order_seed": 0,
            "training_schedule_hash": a2.EXPECTED_SCHEDULE_HASH,
            "validation_partition": "selection",
            "validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
            "checkpoint": {"saved": True, "path": "checkpoints/final"},
            "model": {
                "initial_parameter_sha256": a2.EXPECTED_INITIAL_PARAMETER_SHA256,
                "loaded_checkpoint_weights": False,
            },
            "training": {
                "completed_steps": a2.EXPECTED_STEPS,
                "max_steps": a2.EXPECTED_STEPS,
                "tokens_per_step": a2.EXPECTED_TOKENS_PER_STEP,
                "stopped_by_operational_wall_time_limit": False,
            },
            "tokenized_data": {
                "validation": {
                    "partition": "selection",
                    "source_document_indices_sha256": a2.EXPECTED_SELECTION_HASH,
                    "tokens_sha256": (
                        "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19"
                    ),
                }
            },
        }
        metrics = {
            "training/optimizer_steps": a2.EXPECTED_STEPS,
            "training/planned_optimizer_steps": a2.EXPECTED_STEPS,
            "training/tokens_per_step": a2.EXPECTED_TOKENS_PER_STEP,
            "training/tokens_seen": a2.EXPECTED_TRAINING_TOKENS,
            "training/validation_loss_final_step": a2.EXPECTED_STEPS,
            "training/validation_tokens_final": a2.EXPECTED_VALIDATION_TOKENS,
            "training/validation_sequences_final": a2.EXPECTED_VALIDATION_SEQUENCES,
            "training/validation_available_complete_blocks": a2.EXPECTED_VALIDATION_SEQUENCES,
            "training/validation_batches_final": a2.EXPECTED_VALIDATION_BATCHES,
            "training/validation_partition": "selection",
            "training/validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
            "training/training_schedule_hash": a2.EXPECTED_SCHEDULE_HASH,
            "training/validation_complete_block_coverage": True,
            "training/wall_time_limit_reached": False,
            "training/validation_loss_final": 4.0 + 0.01 * index,
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "metrics.json", metrics)

    diagnostic_recipe = yaml.safe_load(
        (
            source_root
            / "experiments"
            / a2.TRANCHE_ID
            / "run"
            / f"{a2.DIAGNOSTIC_CONFIG_ID}.yaml"
        ).read_text(encoding="utf-8")
    )
    diagnostic_recipe["activation_histograms"]["bins"] = bin_count
    diagnostic_recipe_path = run_root / f"{a2.DIAGNOSTIC_CONFIG_ID}.yaml"
    recipe_text = yaml.safe_dump(diagnostic_recipe, sort_keys=False)
    diagnostic_recipe_path.write_text(recipe_text, encoding="utf-8")
    diagnostic_run = raw_root / a2.DIAGNOSTIC_CONFIG_ID / a2.DIAGNOSTIC_RUN_ID
    diagnostic_run.mkdir(parents=True)
    (diagnostic_run / "config.yaml").write_text(recipe_text, encoding="utf-8")
    source_runs = [
        f"experiments/{a2.TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
        for source in A2_SOURCES
    ]
    diagnostic_manifest = {
        "status": "completed",
        "mode": "activation-histograms",
        "tranche_id": a2.TRANCHE_ID,
        "config_id": a2.DIAGNOSTIC_CONFIG_ID,
        "run_id": a2.DIAGNOSTIC_RUN_ID,
        "git_commit": a2.DIAGNOSTIC_GIT_COMMIT,
        "git_dirty": False,
        "seed": 0,
        "validation_partition": "selection",
        "validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
        "source_runs": source_runs,
        "source_checkpoints": [f"{path}/checkpoints/final" for path in source_runs],
        "activation_histograms": {
            "bins": bin_count,
            "range_min": -16.0,
            "range_max": 16.0,
            "sites": list(a2.SITES),
            "thresholds": list(a2.THRESHOLDS),
            "eval_batches": None,
            "batch_size": 4,
            "validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
        },
    }
    diagnostic_metrics = {
        "activation_histograms/methods": len(A2_SOURCES),
        "activation_histograms/layers": len(a2.SITES) * len(a2.LAYERS),
        "activation_histograms/bins": bin_count,
        "activation_histograms/range_min": -16.0,
        "activation_histograms/range_max": 16.0,
        "activation_histograms/validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
        "activation_histograms/validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
    }
    edges = [
        -16.0 + index * 32.0 / bin_count for index in range(bin_count + 1)
    ]
    payload = {
        "schema_version": 3,
        "bins": bin_count,
        "range_min": -16.0,
        "range_max": 16.0,
        "sites": list(a2.SITES),
        "thresholds": list(a2.THRESHOLDS),
        "validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
        "validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
        "bin_edges": edges,
        "methods": [
            _method_payload(source, source_index, bin_count)
            for source_index, source in enumerate(A2_SOURCES)
        ],
    }
    _write_json(diagnostic_run / "manifest.json", diagnostic_manifest)
    _write_json(diagnostic_run / "metrics.json", diagnostic_metrics)
    _write_json(diagnostic_run / "activation_histograms.json", payload)

    propagation_recipe = json.loads(json.dumps(diagnostic_recipe))
    propagation_recipe["experiment_name"] = "pythia_14m_a2_activation_propagation_seed_0"
    propagation_recipe.pop("activation_histograms")
    propagation_recipe["activation_propagation"] = {
        "selected_runs": [
            {
                "label": source.label,
                "tranche_id": a2.TRANCHE_ID,
                "config_id": source.config_id,
                "run_id": source.run_id,
            }
            for source in A2_SOURCES
        ]
    }
    propagation_recipe_path = run_root / f"{a2.PROPAGATION_CONFIG_ID}.yaml"
    propagation_recipe_text = yaml.safe_dump(propagation_recipe, sort_keys=False)
    propagation_recipe_path.write_text(propagation_recipe_text, encoding="utf-8")
    assert a2.PROPAGATION_RUN_ID is not None
    assert a2.PROPAGATION_GIT_COMMIT is not None
    propagation_run = (
        raw_root / a2.PROPAGATION_CONFIG_ID / a2.PROPAGATION_RUN_ID
    )
    propagation_run.mkdir(parents=True)
    (propagation_run / "config.yaml").write_text(
        propagation_recipe_text, encoding="utf-8"
    )
    propagation_manifest = {
        "status": "completed",
        "mode": "activation-propagation",
        "tranche_id": a2.TRANCHE_ID,
        "config_id": a2.PROPAGATION_CONFIG_ID,
        "run_id": a2.PROPAGATION_RUN_ID,
        "git_commit": a2.PROPAGATION_GIT_COMMIT,
        "git_dirty": False,
        "seed": 0,
        "validation_partition": "selection",
        "validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
        "source_runs": source_runs,
        "source_checkpoints": [f"{path}/checkpoints/final" for path in source_runs],
        "tokenized_data": {
            "validation": {
                "partition": "selection",
                "source_document_indices_sha256": a2.EXPECTED_SELECTION_HASH,
                "tokens_sha256": (
                    "22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19"
                ),
            }
        },
        "activation_propagation": {
            "selected_runs": propagation_recipe["activation_propagation"]["selected_runs"],
            "attention_implementation": "eager",
            "future_causal_positions_excluded": True,
            "eval_batches": None,
            "batch_size": 4,
            "validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
            "validation_cache_tokens": a2.EXPECTED_VALIDATION_CACHE_TOKENS,
            "trailing_tokens_excluded": a2.EXPECTED_TRAILING_VALIDATION_TOKENS,
            "validation_partition": "selection",
            "validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
            "complete_named_partition": True,
            "execution": dict(a2.EXPECTED_PROPAGATION_EXECUTION),
        },
    }
    propagation_methods = [
        _propagation_method(source, index)
        for index, source in enumerate(A2_SOURCES)
    ]
    propagation_metrics = {
        "activation_propagation/methods": len(A2_SOURCES),
        "activation_propagation/layers": len(a2.LAYERS),
        "activation_propagation/matmul_stages": len(a2.LOGICAL_MATMUL_STAGES),
        "activation_propagation/validation_batches": a2.EXPECTED_VALIDATION_BATCHES,
        "activation_propagation/validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
        "activation_propagation/validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
        "activation_propagation/validation_cache_tokens": a2.EXPECTED_VALIDATION_CACHE_TOKENS,
        "activation_propagation/trailing_tokens_excluded": a2.EXPECTED_TRAILING_VALIDATION_TOKENS,
        "activation_propagation/validation_partition": "selection",
        "activation_propagation/validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
    }
    for method in propagation_methods:
        endpoint = method["endpoint"]
        prefix = f"activation_propagation/endpoint/{method['config_id']}"
        propagation_metrics[f"{prefix}/R_block"] = endpoint["R_block"]
        propagation_metrics[f"{prefix}/R_model"] = endpoint["R_model"]
    propagation_payload = {
        "schema_version": 5,
        "validation_batches": a2.EXPECTED_VALIDATION_BATCHES,
        "validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
        "validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
        "validation_cache_tokens": a2.EXPECTED_VALIDATION_CACHE_TOKENS,
        "trailing_tokens_excluded": a2.EXPECTED_TRAILING_VALIDATION_TOKENS,
        "validation_partition": "selection",
        "validation_partition_hash": a2.EXPECTED_SELECTION_HASH,
        "complete_named_partition": True,
        "block_size": a2.EXPECTED_BLOCK_SIZE,
        "batch_size": 4,
        "attention_implementation": "eager",
        "future_causal_positions_excluded": True,
        "matmul_stage_order": list(a2.LOGICAL_MATMUL_STAGES),
        "execution": dict(a2.EXPECTED_PROPAGATION_EXECUTION),
        "methods": propagation_methods,
    }
    _write_json(propagation_run / "manifest.json", propagation_manifest)
    _write_json(propagation_run / "metrics.json", propagation_metrics)
    _write_json(
        propagation_run / "activation_propagation.json", propagation_payload
    )


def _method_payload(
    source: a2.A2Source, source_index: int, bin_count: int
) -> dict[str, object]:
    layers = []
    for layer_index in a2.LAYERS:
        for site_index, site in enumerate(a2.SITES):
            total = a2.EXPECTED_LAYER_TOTALS[site]
            base = 0.10 + 0.01 * site_index
            exact_fraction = base + (0.02 * source_index if site == "h" else 0.002 * source_index)
            exact = int(total * exact_fraction)
            near = exact + int(total * 0.02)
            wide = near + int(total * 0.08)
            layers.append(
                _layer_row(
                    name=f"{site}.layer_{layer_index}",
                    total=total,
                    bin_count=bin_count,
                    exact=exact,
                    near=near,
                    wide=wide,
                    rms=1.0 + 0.05 * site_index - 0.02 * source_index,
                )
            )
    source_run = f"experiments/{a2.TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
    return {
        "label": source.label,
        "config_id": source.config_id,
        "run_id": source.run_id,
        "source_run": source_run,
        "source_checkpoint": f"{source_run}/checkpoints/final",
        "batches": a2.EXPECTED_VALIDATION_BATCHES,
        "layers": layers,
    }


def _propagation_method(
    source: a2.A2Source, source_index: int
) -> dict[str, object]:
    point = _opportunity_point(source, source_index)
    operation_products = point.block_product_count // len(a2.LOGICAL_MATMUL_STAGES)
    operation_zeros = point.block_zero_product_count // len(a2.LOGICAL_MATMUL_STAGES)
    per_operation = {
        name: {
            "zero_product_count": operation_zeros,
            "product_count": operation_products,
            "zero_product_fraction": operation_zeros / operation_products,
        }
        for name in a2.LOGICAL_MATMUL_STAGES
    }
    source_run = f"experiments/{a2.TRANCHE_ID}/raw/{source.config_id}/{source.run_id}"
    return {
        "label": source.label,
        "config_id": source.config_id,
        "run_id": source.run_id,
        "source_run": source_run,
        "source_checkpoint": f"{source_run}/checkpoints/final",
        "source_manifest_status": "completed",
        "num_layers": len(a2.LAYERS),
        "batches": a2.EXPECTED_VALIDATION_BATCHES,
        "architecture": {
            "topology_id": "A1-H",
            "active_sites": ["h"],
            "num_layers": len(a2.LAYERS),
            "sequence_length": a2.EXPECTED_BLOCK_SIZE,
        },
        "endpoint": {
            "validation_sequences": a2.EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": a2.EXPECTED_VALIDATION_TOKENS,
            "block_zero_product_count": point.block_zero_product_count,
            "block_product_count": point.block_product_count,
            "lm_head_product_count": point.lm_head_product_count,
            "model_product_count": point.model_product_count,
            "R_block": point.R_block,
            "R_model": point.R_model,
            "per_operation": per_operation,
        },
    }


def _layer_row(
    *,
    name: str,
    total: int,
    bin_count: int,
    exact: int,
    near: int,
    wide: int,
    rms: float,
) -> dict[str, object]:
    counts = [0] * bin_count
    counts[0] = total // 8
    counts[1] = total // 8
    counts[2 if bin_count > 2 else 1] += total // 2
    counts[-1] += total - sum(counts)
    hits = {"0": exact, "0.01": near, "0.1": wide}
    return {
        "name": name,
        "counts": counts,
        "total": total,
        "finite": total,
        "nonfinite": 0,
        "in_range": total,
        "underflow": 0,
        "overflow": 0,
        "threshold_hits": hits,
        "threshold_fractions": {key: value / total for key, value in hits.items()},
        "rms": rms,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
