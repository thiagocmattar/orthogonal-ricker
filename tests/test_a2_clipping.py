from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import yaml

import paper_exp.plots.a2_clipping as clipping
from paper_exp.diagnostics.logical_products import LOGICAL_MATMUL_STAGES
from paper_exp.plots.a2_clipping import (
    A2ClippingData,
    build_a2_clipping_figure,
    build_a2_clipping_markdown,
    reduce_a2_clipping_rows,
)
from paper_exp.plots.export import (
    DOUBLE_COLUMN_PUBLICATION_PROFILE,
    publication_figure_issues,
)


def test_a2_clipping_reduction_preserves_cutoff_order_and_exact_baselines() -> None:
    points = reduce_a2_clipping_rows(_synthetic_rows())

    assert len(points) == 30
    first = points[:5]
    assert tuple(point.threshold for point in first) == clipping.THRESHOLDS
    # The fixture deliberately makes .01 exceed .03 in R_model.  The reduction
    # must retain experimental cutoff order instead of sorting by x.
    assert first[1].R_model > first[2].R_model
    assert first[2].nondominated is False
    for source_index in range(len(clipping.A2_SOURCES)):
        baseline = points[source_index * len(clipping.THRESHOLDS)]
        assert baseline.delta_validation_loss == pytest.approx(0.0, abs=0.0)
        assert baseline.delta_R_model == pytest.approx(0.0, abs=0.0)


def test_a2_clipping_reduction_rejects_missing_or_out_of_order_point() -> None:
    rows = _synthetic_rows()
    rows[0]["threshold"] = 0.001

    with pytest.raises(ValueError, match="source/cutoff order"):
        reduce_a2_clipping_rows(rows)


def test_a2_clipping_reduction_rejects_inconsistent_r_model_arithmetic() -> None:
    rows = _synthetic_rows()
    rows[7]["potentially_avoidable_model_matmul_fraction"] += 0.01

    with pytest.raises(ValueError, match="R_model is inconsistent"):
        reduce_a2_clipping_rows(rows)


def test_a2_clipping_reduction_derives_r_values_from_exact_integer_counts() -> None:
    rows = _synthetic_rows()
    rows[0]["potentially_avoidable_block_matmul_fraction"] += 5e-15
    rows[0]["potentially_avoidable_model_matmul_fraction"] += 5e-15

    point = reduce_a2_clipping_rows(rows)[0]

    assert point.R_block == point.block_zero_product_count / point.block_product_count
    assert point.R_model == point.block_zero_product_count / point.model_product_count
    assert point.R_block != rows[0]["potentially_avoidable_block_matmul_fraction"]
    assert point.R_model != rows[0]["potentially_avoidable_model_matmul_fraction"]


def test_a2_clipping_reduction_rejects_changed_source_evidence() -> None:
    rows = _synthetic_rows()
    rows[1]["source_validation_cache"] = {
        **rows[1]["source_validation_cache"],
        "tokens_sha256": "f" * 64,
    }

    with pytest.raises(ValueError, match="share the cohort validation cache"):
        reduce_a2_clipping_rows(rows)


def test_a2_clipping_figure_has_two_panels_shared_legend_and_no_jitter() -> None:
    data = _synthetic_data()
    figure = build_a2_clipping_figure(data)
    try:
        assert len(figure.axes) == 2
        assert len(figure.legends) == 1
        absolute_axis, delta_axis = figure.axes
        assert absolute_axis.get_xlabel() == (
            r"Model-wide logical opportunity, $R_{model}$ (%)"
        )
        assert delta_axis.get_xlabel() == r"Change in $R_{model}$ (pp)"
        assert len(absolute_axis.lines) == len(clipping.A2_SOURCES)
        assert len(delta_axis.lines) == len(clipping.A2_SOURCES) + 2
        delta_paths = []
        for source_index in range(len(clipping.A2_SOURCES)):
            points = data.points[
                source_index * len(clipping.THRESHOLDS) :
                (source_index + 1) * len(clipping.THRESHOLDS)
            ]
            assert absolute_axis.lines[source_index].get_xdata().tolist() == pytest.approx(
                [100.0 * point.R_model for point in points]
            )
            assert delta_axis.lines[source_index].get_xdata().tolist() == pytest.approx(
                [100.0 * point.delta_R_model for point in points]
            )
            delta_paths.append(tuple(delta_axis.lines[source_index].get_xdata()))
        assert len(set(delta_paths)) == len(clipping.A2_SOURCES)
        assert len(absolute_axis.collections) == 30
        assert len(delta_axis.collections) == 25
        assert absolute_axis.get_xlim()[0] == pytest.approx(0.0)
        assert delta_axis.get_xlim()[0] <= 0.0 <= delta_axis.get_xlim()[1]
        assert delta_axis.get_ylim()[0] <= 0.0 <= delta_axis.get_ylim()[1]
        legend_labels = [
            text.get_text() for text in figure.legends[0].get_texts()
        ]
        assert legend_labels == [
            "Control",
            r"L1 $\lambda = 0.1$",
            r"L1 $\lambda = 0.5$",
            r"L1 $\lambda = 1$",
            r"L1 $\lambda = 2$",
            r"L1 $\lambda = 5$",
            "t = 0",
            "t = 0.01",
            "t = 0.03",
            "t = 0.1",
            "t = 0.3",
        ]
        common_origins = [
            collection
            for collection in delta_axis.collections
            if collection.get_offsets().shape == (1, 2)
            and collection.get_offsets()[0].tolist() == pytest.approx([0.0, 0.0])
        ]
        assert len(common_origins) == 1
        assert publication_figure_issues(
            figure, DOUBLE_COLUMN_PUBLICATION_PROFILE
        ) == ()
    finally:
        plt.close(figure)


def test_a2_clipping_markdown_contains_caption_limits_and_all_30_rows() -> None:
    markdown = build_a2_clipping_markdown(_synthetic_data())
    data_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| Control |") or line.startswith("| L1 $")
    ]

    assert len(data_rows) == 30
    assert "zoomed validation-loss axis" in markdown
    assert "311,296 tokens" in markdown
    assert "not measured speedup" in markdown
    assert "does not causally isolate spillover" in markdown
    assert "Nondominated" in markdown


def test_a2_clipping_loader_fails_closed_before_run_and_hash_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipping, "DIAGNOSTIC_RUN_ID", None)
    monkeypatch.setattr(clipping, "DIAGNOSTIC_GIT_COMMIT", None)
    monkeypatch.setattr(clipping, "DIAGNOSTIC_ARTIFACT_SHA256", None)

    with pytest.raises(RuntimeError, match="not pinned"):
        clipping.load_a2_clipping(".")


def test_a2_clipping_generation_accepts_exact_envelope_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "001-test-accepted"
    git_commit = "b" * 40
    rows = _synthetic_rows()
    recipe = yaml.safe_load(
        Path(
            "experiments/02-a2-l1-screen/run/"
            "020-a2-posthoc-clipping-frontier.yaml"
        ).read_text(encoding="utf-8")
    )

    reference_artifact = b'{"schema_version": 5}\n'
    reference_sha256 = hashlib.sha256(reference_artifact).hexdigest()
    reference = {
        **clipping.ZERO_REFERENCE,
        "artifact_sha256": reference_sha256,
    }
    recipe["clipping_frontier"]["zero_threshold_reference"] = reference
    monkeypatch.setattr(clipping, "ZERO_REFERENCE", reference)
    monkeypatch.setattr(
        clipping, "ZERO_REFERENCE_ARTIFACT_SHA256", reference_sha256
    )
    monkeypatch.setattr(clipping, "DIAGNOSTIC_RUN_ID", run_id)
    monkeypatch.setattr(clipping, "DIAGNOSTIC_GIT_COMMIT", git_commit)
    monkeypatch.setattr(
        clipping,
        "_load_training_source",
        lambda _root, _source: (None, ()),
    )

    run_root = tmp_path / "experiments" / clipping.TRANCHE_ID
    recipe_path = run_root / "run" / f"{clipping.DIAGNOSTIC_CONFIG_ID}.yaml"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8", newline="\n"
    )
    diagnostic_dir = run_root / "raw" / clipping.DIAGNOSTIC_CONFIG_ID / run_id
    diagnostic_dir.mkdir(parents=True)
    (diagnostic_dir / "config.yaml").write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8", newline="\n"
    )

    cache = rows[0]["source_validation_cache"]
    source_runs: list[str] = []
    source_checkpoints: list[str] = []
    source_statuses: list[str] = []
    source_checkpoint_contents: list[dict[str, Any]] = []
    for source_index, source in enumerate(clipping.A2_SOURCES):
        source_dir = run_root / "raw" / source.config_id / source.run_id
        checkpoint_dir = source_dir / "checkpoints" / "final"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.safetensors").write_bytes(
            f"checkpoint-{source_index}".encode()
        )
        identity = clipping._checkpoint_content_identity(checkpoint_dir)
        for row in rows[
            source_index * len(clipping.THRESHOLDS) :
            (source_index + 1) * len(clipping.THRESHOLDS)
        ]:
            row["source_checkpoint_content"] = identity
        source_run = (
            f"experiments/{clipping.TRANCHE_ID}/raw/"
            f"{source.config_id}/{source.run_id}"
        )
        source_runs.append(source_run)
        source_checkpoints.append(f"{source_run}/checkpoints/final")
        source_statuses.append("completed")
        source_checkpoint_contents.append(identity)
        _write_json(
            source_dir / "manifest.json",
            {
                "status": "completed",
                "git_commit": clipping.EXPECTED_TRAINING_GIT_COMMIT,
                "git_dirty": False,
                "config_sha256": source.config_sha256,
                "condition_fingerprint": source.condition_fingerprint,
                "tokenized_data": {"validation": cache},
            },
        )

    reference_dir = (
        run_root
        / "raw"
        / reference["config_id"]
        / reference["run_id"]
    )
    reference_dir.mkdir(parents=True)
    (reference_dir / "activation_propagation.json").write_bytes(reference_artifact)
    _write_json(
        reference_dir / "manifest.json",
        {
            "tranche_id": reference["tranche_id"],
            "config_id": reference["config_id"],
            "run_id": reference["run_id"],
            "mode": "activation-propagation",
            "status": "completed",
            "git_commit": reference["git_commit"],
            "git_dirty": False,
        },
    )

    zero_record = {
        **reference,
        "source_run": (
            f"experiments/{clipping.TRANCHE_ID}/raw/"
            f"{reference['config_id']}/{reference['run_id']}"
        ),
        "source_artifact": (
            f"experiments/{clipping.TRANCHE_ID}/raw/"
            f"{reference['config_id']}/{reference['run_id']}/"
            "activation_propagation.json"
        ),
        "source_artifact_sha256": reference_sha256,
        "audit_scope": clipping.ZERO_REFERENCE_AUDIT_SCOPE,
    }
    manifest = {
        "tranche_id": clipping.TRANCHE_ID,
        "config_id": clipping.DIAGNOSTIC_CONFIG_ID,
        "run_id": run_id,
        "mode": "clipping-frontier",
        "status": "completed",
        "git_commit": git_commit,
        "git_dirty": False,
        "seed": 0,
        "source_runs": source_runs,
        "source_checkpoints": source_checkpoints,
        "source_manifest_statuses": source_statuses,
        "source_checkpoint_contents": source_checkpoint_contents,
        "tokenized_data": {"validation": cache},
        "clipping_frontier": {
            **recipe["clipping_frontier"],
            "zero_threshold_reference": zero_record,
            "attention_implementation": "eager",
            "eval_batches": None,
            "batch_size": 4,
            "validation_batches": clipping.EXPECTED_VALIDATION_BATCHES,
            "validation_sequences": clipping.EXPECTED_VALIDATION_SEQUENCES,
            "validation_tokens": clipping.EXPECTED_VALIDATION_TOKENS,
            "validation_cache_tokens": clipping.EXPECTED_CACHE_TOKENS,
            "trailing_tokens_excluded": (
                clipping.EXPECTED_CACHE_TOKENS
                - clipping.EXPECTED_VALIDATION_TOKENS
            ),
            "validation_partition": "selection",
            "validation_partition_hash": clipping.EXPECTED_PARTITION_HASH,
            "complete_named_partition": True,
            "execution": {
                "requested_device": "cuda",
                "requested_precision": "bfloat16",
                "resolved_device": "cuda",
                "resolved_precision": "bfloat16",
            },
        },
    }
    _write_json(diagnostic_dir / "manifest.json", manifest)
    _write_json(
        diagnostic_dir / "metrics.json",
        {
            "clipping_frontier/sources": len(clipping.A2_SOURCES),
            "clipping_frontier/cutoffs": len(clipping.THRESHOLDS),
            "clipping_frontier/points": 30,
            "clipping_frontier/validation_batches_per_point": 38,
            "clipping_frontier/validation_sequences_per_point": 152,
            "clipping_frontier/validation_tokens_per_point": 311_296,
            "clipping_frontier/total_point_tokens": 30 * 311_296,
            "clipping_frontier/wall_seconds": 10.0,
        },
    )
    _write_jsonl(diagnostic_dir / "predictions.jsonl", rows)
    artifact = diagnostic_dir / "clipping_frontier.jsonl"
    _write_jsonl(artifact, rows)
    monkeypatch.setattr(
        clipping, "DIAGNOSTIC_ARTIFACT_SHA256", _sha256(artifact)
    )

    first_outputs = clipping.generate_a2_clipping_figure(tmp_path)
    first_hashes = tuple(_sha256(path) for path in first_outputs)
    second_outputs = clipping.generate_a2_clipping_figure(tmp_path)

    assert second_outputs == first_outputs
    assert tuple(_sha256(path) for path in second_outputs) == first_hashes
    assert len(first_outputs) == 4
    assert all(path.is_file() for path in first_outputs)
    assert first_outputs[0].read_bytes().startswith(b"%PDF")
    assert first_outputs[1].read_bytes().startswith(b"\x89PNG")
    provenance = json.loads(first_outputs[3].read_text(encoding="utf-8"))
    assert provenance["diagnostic"]["artifact_sha256"] == _sha256(artifact)
    assert provenance["generator"] == {
        "path": clipping.GENERATOR_PATH,
        "sha256": _sha256(Path(clipping.__file__)),
    }
    assert len(provenance["sources"]) == len(clipping.A2_SOURCES)
    for source, record in zip(
        clipping.A2_SOURCES, provenance["sources"], strict=True
    ):
        assert record["manifest_status"] == "completed"
        assert record["git_commit"] == clipping.EXPECTED_TRAINING_GIT_COMMIT
        assert record["git_dirty"] is False
        assert record["config_sha256"] == source.config_sha256
        assert record["condition_fingerprint"] == source.condition_fingerprint
    for record in provenance["inputs"]:
        input_path = tmp_path / record["path"]
        assert input_path.is_file()
        assert record["sha256"] == _sha256(input_path)
        assert record["size_bytes"] == input_path.stat().st_size
    for record in provenance["outputs"]:
        output_path = tmp_path / record["path"]
        assert output_path.is_file()
        assert record["sha256"] == _sha256(output_path)
        assert record["size_bytes"] == output_path.stat().st_size
    assert len(provenance["reduction"]["points"]) == 30
    assert str(tmp_path) not in first_outputs[3].read_text(encoding="utf-8")

    incomplete_manifest = copy.deepcopy(manifest)
    incomplete_manifest.pop("source_runs")
    _write_json(diagnostic_dir / "manifest.json", incomplete_manifest)
    with pytest.raises(ValueError, match="source manifest mismatch.*source_runs"):
        clipping.load_a2_clipping(tmp_path)


def _synthetic_data() -> A2ClippingData:
    return A2ClippingData(
        points=reduce_a2_clipping_rows(_synthetic_rows()),
        diagnostic_run_id="001-test",
        diagnostic_git_commit="a" * 40,
        source_evidence=(),
        inputs=(),
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    operation_products = {name: 1_000 for name in LOGICAL_MATMUL_STAGES}
    block_products = sum(operation_products.values())
    lm_head_products = 4_000
    model_products = block_products + lm_head_products
    threshold_zero_increments = (0, 30, 10, 50, 80)
    cache = {
        "tokens_path": "data/tokenized/cache/validation/selection/tokens.int32.bin",
        "dtype": "int32",
        "tokens": clipping.EXPECTED_CACHE_TOKENS,
        "tokens_bytes": clipping.EXPECTED_CACHE_TOKENS * 4,
        "tokens_sha256": clipping.EXPECTED_CACHE_SHA256,
        "block_size": clipping.EXPECTED_BLOCK_SIZE,
        "split": "validation",
        "max_documents": 500,
        "partition": "selection",
        "partition_scheme": "shuffled_source_documents_half_v1",
        "partition_seed": 20260718,
        "source_document_indices_sha256": clipping.EXPECTED_PARTITION_HASH,
    }
    for source_index, source in enumerate(clipping.A2_SOURCES):
        checkpoint_identity = {
            "files": [
                {
                    "path": "model.safetensors",
                    "bytes": source_index + 1,
                    "sha256": f"{source_index + 1:064x}",
                }
            ]
        }
        source_run = (
            f"experiments/{clipping.TRANCHE_ID}/raw/"
            f"{source.config_id}/{source.run_id}"
        )
        for threshold_index, threshold in enumerate(clipping.THRESHOLDS):
            zero_total = 100 + 50 * source_index + (
                (source_index + 1) * threshold_zero_increments[threshold_index]
            )
            operation_zeros = {name: 0 for name in LOGICAL_MATMUL_STAGES}
            operation_zeros[LOGICAL_MATMUL_STAGES[-1]] = zero_total
            site_hits = {
                site: int(
                    count
                    * min(
                        0.95,
                        0.35 + 0.02 * source_index + 0.03 * threshold_index,
                    )
                )
                for site, count in clipping.EXPECTED_SITE_COUNTS.items()
            }
            site_fractions = {
                site: site_hits[site] / clipping.EXPECTED_SITE_COUNTS[site]
                for site in clipping.SITES
            }
            rows.append(
                {
                    "schema_version": 1,
                    "label": source.label,
                    "source_tranche_id": clipping.TRANCHE_ID,
                    "source_config_id": source.config_id,
                    "source_run_id": source.run_id,
                    "source_run": source_run,
                    "source_manifest_status": "completed",
                    "source_checkpoint": f"{source_run}/checkpoints/final",
                    "source_checkpoint_content": copy.deepcopy(checkpoint_identity),
                    "source_validation_cache": copy.deepcopy(cache),
                    "validation_sequences": clipping.EXPECTED_VALIDATION_SEQUENCES,
                    "validation_batches": clipping.EXPECTED_VALIDATION_BATCHES,
                    "validation_tokens": clipping.EXPECTED_VALIDATION_TOKENS,
                    "event": "clipping_sweep",
                    "mode": clipping.CLIPPING_MODE,
                    "threshold": threshold,
                    "quantile": None,
                    "rms_multiplier": None,
                    "rms_scope": None,
                    "sites": list(clipping.SITES),
                    "site_achieved_sparsity": site_fractions,
                    "site_zero_hits": site_hits,
                    "site_activation_count": dict(clipping.EXPECTED_SITE_COUNTS),
                    "matmul_zero_product_count": operation_zeros,
                    "matmul_product_count": dict(operation_products),
                    "matmul_zero_product_fraction": {
                        name: operation_zeros[name] / operation_products[name]
                        for name in LOGICAL_MATMUL_STAGES
                    },
                    "block_zero_product_count": zero_total,
                    "block_matmul_product_count": block_products,
                    "lm_head_matmul_product_count": lm_head_products,
                    "model_matmul_product_count": model_products,
                    "potentially_avoidable_block_matmul_fraction": (
                        zero_total / block_products
                    ),
                    "potentially_avoidable_model_matmul_fraction": (
                        zero_total / model_products
                    ),
                    "validation_loss": (
                        4.0
                        + 0.02 * source_index
                        + 0.005 * threshold_index * (1.0 + 0.1 * source_index)
                    ),
                    "achieved_sparsity": sum(site_hits.values())
                    / sum(clipping.EXPECTED_SITE_COUNTS.values()),
                    "wall_seconds": 1.0,
                }
            )
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
