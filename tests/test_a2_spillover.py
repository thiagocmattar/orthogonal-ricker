from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

import paper_exp.plots.a2_spillover as a2
from paper_exp.plots.a2_spillover import (
    A2_SOURCES,
    A2SpilloverData,
    SiteReduction,
    build_a2_layer5_distributions_figure,
    build_a2_spillover_response_figure,
    generate_a2_spillover_suite,
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


def test_a2_figures_pass_publication_checks_and_keep_distribution_atoms_separate() -> None:
    data = _synthetic_data(bin_count=4)
    response = build_a2_spillover_response_figure(data)
    distributions = build_a2_layer5_distributions_figure(data)
    try:
        assert publication_figure_issues(
            response, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
        assert publication_figure_issues(
            distributions, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
        assert len(response.axes) == 2
        assert len(response.legends) == 1
        assert len(response.legends[0].get_texts()) == len(a2.SITES)
        assert len(distributions.axes) == 12
        assert all(
            axis.get_xlim() == pytest.approx(a2.EXPECTED_RANGE)
            for axis in distributions.axes
            if axis.get_yscale() == "log"
        )
        assert sum(len(axis.patches) for axis in distributions.axes) == 12
    finally:
        plt.close(response)
        plt.close(distributions)


def test_a2_suite_is_atomic_and_deterministic_from_compact_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(a2, "EXPECTED_BINS", 4)
    _write_a2_fixture(tmp_path, bin_count=4)

    outputs = generate_a2_spillover_suite(tmp_path)
    first_bytes = tuple(path.read_bytes() for path in outputs)

    assert [path.name for path in outputs] == [
        "01-a2-spillover-response.pdf",
        "01-a2-spillover-response.png",
        "01-a2-spillover-response.md",
        "01-a2-spillover-response.provenance.json",
        "02-a2-layer5-distributions.pdf",
        "02-a2-layer5-distributions.png",
        "02-a2-layer5-distributions.md",
        "02-a2-layer5-distributions.provenance.json",
    ]
    response_markdown = outputs[2].read_text(encoding="utf-8")
    distribution_markdown = outputs[6].read_text(encoding="utf-8")
    assert "single-seed directional screen" in response_markdown
    assert "sums of integer hits divided by sums of totals" in response_markdown
    assert "Underflow and overflow are not drawn" in distribution_markdown
    assert "`[-16, 16]`" in distribution_markdown
    response_provenance = json.loads(outputs[3].read_text(encoding="utf-8"))
    distribution_provenance = json.loads(outputs[7].read_text(encoding="utf-8"))
    assert response_provenance["cohort"]["diagnostic_run_id"] == a2.DIAGNOSTIC_RUN_ID
    assert response_provenance["reduction"]["primary_threshold"] == 0.01
    assert distribution_provenance["reduction"]["distribution_layer"] == 5
    assert distribution_provenance["reduction"]["exact_zero_atom_separate"] is True
    assert len(response_provenance["inputs"]) == 35

    assert generate_a2_spillover_suite(tmp_path) == outputs
    assert tuple(path.read_bytes() for path in outputs) == first_bytes


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
    return A2SpilloverData(reductions, losses, edges, methods, ())


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
    path.write_text(json.dumps(payload), encoding="utf-8")
