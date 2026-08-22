from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

import paper_exp.diagnostics.activation_histograms as activation_histograms
import paper_exp.diagnostics.propagation as activation_propagation
import paper_exp.diagnostics.clipping as clipping
import paper_exp.diagnostics.clipping_evaluation as clipping_evaluation
import paper_exp.data as data
import paper_exp.run as run_module
import paper_exp.diagnostics.weight_histograms as weight_histograms
from paper_exp.diagnostics.sources import (
    find_source_run,
    portable_path,
    source_checkpoint_path,
)


@pytest.mark.parametrize(
    ("module", "entrypoint", "dependency_loader", "mode", "config_name"),
    [
        (
            activation_histograms,
            "run_activation_histograms",
            "_load_dependencies",
            "activation-histograms",
            "10-activation-histograms",
        ),
        (
            activation_propagation,
            "run_activation_propagation",
            "_load_dependencies",
            "activation-propagation",
            "11-activation-propagation",
        ),
        (
            weight_histograms,
            "run_weight_histograms",
            "_load_dependencies",
            "weight-histograms",
            "12-weight-histograms",
        ),
        (
            data,
            "prepare_tokenized_data",
            "_load_data_dependencies",
            "prepare-data",
            "13-prepare-data",
        ),
    ],
)
def test_configured_workflow_dependency_failure_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    entrypoint: str,
    dependency_loader: str,
    mode: str,
    config_name: str,
) -> None:
    config = _base_config(tmp_path)
    config_path = tmp_path / f"{config_name}.yaml"
    run_dir = Path(config["output"]["dir"]) / config_name / "001-dependency-failure"

    def fail_dependencies() -> None:
        _assert_running(run_dir, mode=mode)
        raise RuntimeError("dependency load failed")

    monkeypatch.setattr(module, dependency_loader, fail_dependencies)

    with pytest.raises(RuntimeError, match="dependency load failed"):
        getattr(module, entrypoint)(
            config,
            config_path=config_path,
            command=f"pytest {mode}",
            run_id="dependency-failure",
        )

    _assert_failed(run_dir, mode=mode)


def test_diagnostic_schema_failure_precedes_run_creation(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    del config["activation_histograms"]["bins"]
    config_path = tmp_path / "10-activation-histograms.yaml"

    with pytest.raises(ValueError, match="activation_histograms.bins"):
        activation_histograms.run_activation_histograms(
            config,
            config_path=config_path,
            command="pytest invalid diagnostic",
            run_id="must-not-exist",
        )

    assert not (
        Path(config["output"]["dir"])
        / config_path.stem
        / "001-must-not-exist"
    ).exists()


def test_clipping_dependency_failure_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = _write_completed_source(
        tmp_path,
        config_id="14-source",
        run_id="001-source",
    )
    config = _base_config(tmp_path)
    (source_run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    def fail_dependencies() -> None:
        run_dir = _find_clipping_run(Path(config["output"]["dir"]), "14-source")
        _assert_running(run_dir, mode="clip-sweep")
        raise RuntimeError("dependency load failed")

    monkeypatch.setattr(
        clipping_evaluation,
        "_load_clipping_dependencies",
        fail_dependencies,
    )

    with pytest.raises(RuntimeError, match="dependency load failed"):
        clipping.run_clipping_sweep(
            checkpoint_run_dir=source_run,
            command="pytest clip-sweep",
            thresholds=[0.0],
            quantiles=[],
            sites=["h"],
            eval_batches=1,
            run_id="dependency-failure",
        )

    run_dir = _find_clipping_run(Path(config["output"]["dir"]), "14-source")
    _assert_failed(run_dir, mode="clip-sweep")


def test_clipping_rejects_noncompleted_source_before_launch(tmp_path: Path) -> None:
    source_run = _write_completed_source(
        tmp_path,
        config_id="14-source",
        run_id="001-source",
    )
    config = _base_config(tmp_path)
    (source_run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    manifest = _read_manifest(source_run)
    manifest["status"] = "failed"
    _write_json(source_run / "manifest.json", manifest)

    with pytest.raises(ValueError, match="not completed"):
        clipping.run_clipping_sweep(
            checkpoint_run_dir=source_run,
            command="pytest clip-sweep",
            thresholds=[0.0],
            quantiles=[],
            sites=["h"],
            eval_batches=1,
        )

    assert not list(Path(config["output"]["dir"]).glob("14-source-clip-*"))


def test_source_selection_requires_one_exact_completed_checkpoint_run(
    tmp_path: Path,
) -> None:
    config_id = "15-source"
    run_id = "001-pinned"
    run_dir = _write_completed_source(tmp_path, config_id=config_id, run_id=run_id)
    config = {"output": {"dir": str(tmp_path / "results")}}

    assert find_source_run(
        config,
        {"config_id": config_id, "run_id": run_id},
        section="activation_histograms",
    ) == run_dir
    with pytest.raises(ValueError, match="exact config_id and run_id"):
        find_source_run(
            config,
            {"config_id": config_id},
            section="activation_histograms",
        )

    manifest = _read_manifest(run_dir)
    assert source_checkpoint_path(run_dir, manifest) == (
        run_dir / "checkpoints" / "final"
    ).resolve()
    with pytest.raises(ValueError, match="no saved checkpoint"):
        source_checkpoint_path(run_dir, {"config_id": config_id})

    manifest["status"] = "running"
    _write_json(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="not completed"):
        find_source_run(
            config,
            {"config_id": config_id, "run_id": run_id},
            section="activation_histograms",
        )


def test_activation_histograms_require_shared_requested_validation_identity(
    tmp_path: Path,
) -> None:
    tokens_path = tmp_path / "tokens.int32.bin"
    np.arange(8, dtype=np.int32).tofile(tokens_path)
    reference = {
        "tokens_path": str(tokens_path),
        "dtype": "int32",
        "tokens": 8,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
        "block_size": 4,
        "split": "validation",
        "max_documents": 10,
        "partition": "selection",
        "partition_scheme": "half-v1",
        "partition_seed": 7,
        "source_document_indices_sha256": "a" * 64,
    }
    activation_histograms._validate_requested_validation_cache(
        {
            "split": "validation",
            "max_documents": 10,
            "partition": "selection",
            "partition_scheme": "half-v1",
            "partition_seed": 7,
            "partition_hash": "a" * 64,
            "eval_batches": None,
        },
        reference,
    )
    assert activation_histograms.verify_token_cache(
        reference, context="Activation histogram validation cache"
    ) == (
        tokens_path.resolve()
    )

    with pytest.raises(ValueError, match="partition hash"):
        activation_histograms._validate_requested_validation_cache(
            {
                "partition": "selection",
                "partition_hash": "b" * 64,
                "eval_batches": None,
            },
            reference,
        )


def test_activation_histograms_reject_corrupt_cache_and_mixed_execution_requests(
    tmp_path: Path,
) -> None:
    tokens_path = tmp_path / "tokens.int32.bin"
    np.arange(8, dtype=np.int32).tofile(tokens_path)
    metadata = {
        "tokens_path": str(tokens_path),
        "dtype": "int32",
        "tokens": 8,
        "tokens_bytes": tokens_path.stat().st_size,
        "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
        "block_size": 4,
    }
    tokens_path.write_bytes(tokens_path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="size"):
        activation_histograms.verify_token_cache(
            metadata, context="Activation histogram validation cache"
        )

    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "config.yaml").write_text(
        "training:\n  device: cpu\n  precision: float32\n",
        encoding="utf-8",
    )
    (source_b / "config.yaml").write_text(
        "training:\n  device: cuda\n  precision: float16\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="device and precision"):
        activation_histograms._shared_execution_request([source_a, source_b])


@pytest.mark.parametrize("path_formatter", [portable_path, clipping._portable_path])
def test_source_paths_are_relative_inside_working_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_formatter: Any,
) -> None:
    source = tmp_path / "results" / "15-source" / "001-pinned"
    monkeypatch.chdir(tmp_path)

    assert path_formatter(source) == "results/15-source/001-pinned"


def test_weight_histogram_artifact_precedes_completed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config["weight_histograms"] = {
        "selected_runs": [
            {"config_id": "01-source", "run_id": "001-source", "label": "Source"}
        ],
        "scope": "mlp_weights",
        "bins": 4,
        "range_min": -1.0,
        "range_max": 1.0,
    }
    config_path = tmp_path / "20-weight-histograms.yaml"
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    _write_json(
        source_run / "manifest.json",
        {
            "checkpoint": {
                "saved": True,
                "path": str(source_run / "checkpoints" / "final"),
            }
        },
    )
    run_dir = Path(config["output"]["dir"]) / config_path.stem / "001-success"
    order: list[str] = []

    monkeypatch.setattr(
        weight_histograms,
        "_load_dependencies",
        lambda: (object(), np, object()),
    )
    monkeypatch.setattr(
        weight_histograms,
        "find_source_run",
        lambda _config, _selected, **_kwargs: source_run,
    )
    monkeypatch.setattr(
        weight_histograms,
        "_measure_one_run",
        lambda **_kwargs: {"layers": [{"total": 5}]},
    )
    real_write_json = weight_histograms.write_json

    def write_specialized(path: Path, payload: dict[str, Any]) -> None:
        _assert_running(run_dir, mode="weight-histograms")
        order.append("specialized")
        real_write_json(path, payload)

    def finish(run: Any, **kwargs: Any) -> Path:
        assert (run_dir / "weight_histograms.json").is_file()
        _assert_running(run_dir, mode="weight-histograms")
        order.append("complete")
        return run_module.complete_run(run, **kwargs)

    monkeypatch.setattr(weight_histograms, "write_json", write_specialized)
    monkeypatch.setattr(weight_histograms, "complete_run", finish)

    result = weight_histograms.run_weight_histograms(
        config,
        config_path=config_path,
        command="pytest weight-histograms",
        run_id="success",
    )

    assert result == run_dir
    assert order == ["specialized", "complete"]
    assert _read_manifest(run_dir)["status"] == "completed"


def test_data_cache_precedes_completed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config["validation"] = {"enabled": False}
    config["preprocessing"] = {
        "output_dir": str(tmp_path / "tokenized"),
        "cache_id": "21-prepare-data",
        "block_size": 4,
        "append_eos": True,
        "overwrite": False,
    }
    config_path = tmp_path / "21-prepare-data.yaml"
    cache_dir = Path(config["preprocessing"]["output_dir"]) / config_path.stem
    tokens_path = cache_dir / "tokens.bin"
    metadata_path = cache_dir / "metadata.json"

    monkeypatch.setattr(
        data,
        "_load_data_dependencies",
        lambda: (object(), object(), object()),
    )

    def write_cache(**kwargs: Any) -> dict[str, Any]:
        selected_cache = Path(kwargs["cache_dir"])
        selected_cache.mkdir(parents=True, exist_ok=True)
        tokens_path.write_bytes(b"durable tokens")
        metadata = {
            "split": kwargs["split"],
            "tokens_path": str(tokens_path),
            "documents": 2,
            "tokens": 8,
            "block_size": 4,
        }
        _write_json(metadata_path, metadata)
        return metadata

    def finish(run: Any, **kwargs: Any) -> Path:
        assert tokens_path.is_file()
        assert metadata_path.is_file()
        _assert_running(run.run_dir, mode="prepare-data")
        return run_module.complete_run(run, **kwargs)

    monkeypatch.setattr(data, "_load_or_write_cache", write_cache)
    monkeypatch.setattr(data, "complete_run", finish)

    run_dir = data.prepare_tokenized_data(
        config,
        config_path=config_path,
        command="pytest prepare-data",
        run_id="success",
    )

    assert _read_manifest(run_dir)["status"] == "completed"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()


def test_clipping_frontier_precedes_completed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = _write_completed_source(
        tmp_path,
        config_id="22-source",
        run_id="001-source",
    )
    config = _base_config(tmp_path)
    config["training"] = {"device": "cpu", "precision": "float32"}
    config["validation"] = {"batch_size": 1}
    (source_run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    tokens_path = tmp_path / "validation-tokens.bin"
    np.arange(16, dtype=np.int32).tofile(tokens_path)
    checkpoint_path = source_run / "checkpoints" / "final"
    source_manifest = _read_manifest(source_run)
    source_manifest["tokenized_data"] = {
        "validation": {
            "tokens_path": str(tokens_path),
            "dtype": "int32",
            "tokens": 16,
            "tokens_bytes": tokens_path.stat().st_size,
            "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
            "block_size": 4,
        }
    }
    _write_json(source_run / "manifest.json", source_manifest)
    order: list[str] = []

    fake_torch = SimpleNamespace(float32="float32")
    fake_device = SimpleNamespace(type="cpu")
    fake_model = SimpleNamespace(
        to=lambda **_kwargs: None,
        eval=lambda: None,
    )
    monkeypatch.setattr(
        clipping_evaluation,
        "_load_clipping_dependencies",
        lambda: (fake_torch, np, object(), object()),
    )
    monkeypatch.setattr(
        clipping,
        "select_device",
        lambda *_args, **_kwargs: fake_device,
    )
    monkeypatch.setattr(
        clipping,
        "select_dtype",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(clipping, "load_checkpoint_model", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(
        clipping_evaluation,
        "_evaluate_clipped_loss",
        lambda **_kwargs: {
            "validation_loss": 1.0,
            "achieved_sparsity": 0.0,
            "potentially_avoidable_model_matmul_fraction": None,
        },
    )
    real_write_jsonl = clipping.write_jsonl

    def write_specialized(path: Path, rows: list[dict[str, Any]]) -> None:
        run_dir = _find_clipping_run(Path(config["output"]["dir"]), "22-source")
        _assert_running(run_dir, mode="clip-sweep")
        order.append("specialized")
        real_write_jsonl(path, rows)

    def finish(run: Any, **kwargs: Any) -> Path:
        run_dir = _find_clipping_run(Path(config["output"]["dir"]), "22-source")
        assert (run_dir / "clipping_frontier.jsonl").is_file()
        _assert_running(run_dir, mode="clip-sweep")
        order.append("complete")
        return run_module.complete_run(run, **kwargs)

    monkeypatch.setattr(clipping, "write_jsonl", write_specialized)
    monkeypatch.setattr(clipping, "complete_run", finish)

    result = clipping.run_clipping_sweep(
        checkpoint_run_dir=source_run,
        command="pytest clip-sweep",
        thresholds=[0.0],
        quantiles=[],
        sites=["h"],
        eval_batches=1,
        run_id="success",
    )

    run_dir = _find_clipping_run(Path(config["output"]["dir"]), "22-source")
    assert result == run_dir
    assert order == ["specialized", "complete"]
    assert _read_manifest(run_dir)["status"] == "completed"
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    sweep = saved_config["activation_clipping"]["sweep"]
    assert sweep["source"]["config_id"] == "22-source"
    assert sweep["source"]["run_id"] == "001-source"
    assert sweep["source"]["checkpoint_content"]["files"] == [
        {
            "path": "model.safetensors",
            "bytes": len(b"checkpoint"),
            "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        }
    ]
    assert sweep["source"]["validation_cache"]["tokens_sha256"] == hashlib.sha256(
        tokens_path.read_bytes()
    ).hexdigest()
    assert sweep["thresholds"] == [0.0]
    assert sweep["quantiles"] == []
    assert sweep["rms_multipliers"] == []
    assert sweep["sites"] == ["h"]
    assert sweep["eval_batches"] == 1
    assert sweep["measure_zero_products"] is False
    assert sweep["evaluation_seed"] == 0
    assert sweep["experiment_suffix"] is None
    assert sweep["effective_suffix"] == "sites-h"


def test_clipping_sweep_id_uses_content_identity_not_checkout_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = _write_completed_source(
        tmp_path / "checkout-a",
        config_id="23-source",
        run_id="001-source",
    )
    source_b = _write_completed_source(
        tmp_path / "checkout-b",
        config_id="23-source",
        run_id="001-source",
    )
    captured: list[str] = []

    @contextmanager
    def capture_lifecycle(
        config: dict[str, Any],
        *,
        config_path: str | Path,
        **_kwargs: Any,
    ):
        captured.append(str(config_path))
        yield SimpleNamespace(config=config)

    monkeypatch.setattr(clipping, "run_lifecycle", capture_lifecycle)
    monkeypatch.setattr(
        clipping,
        "_run_clipping_sweep",
        lambda *_args, **_kwargs: Path("captured"),
    )

    common = {
        "command": "pytest clip-sweep",
        "quantiles": [],
        "sites": ["h"],
        "eval_batches": 1,
    }
    clipping.run_clipping_sweep(
        checkpoint_run_dir=source_a,
        thresholds=[0],
        **common,
    )
    clipping.run_clipping_sweep(
        checkpoint_run_dir=source_b,
        thresholds=[0.0],
        **common,
    )
    clipping.run_clipping_sweep(
        checkpoint_run_dir=source_b,
        thresholds=[0.1],
        **common,
    )

    assert captured[0] == captured[1]
    assert captured[2] != captured[1]


def _base_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "experiment_name": "lifecycle_test",
        "model": {
            "provider": "huggingface",
            "name": "pythia-test-random",
            "architecture": "test/pythia",
            "revision": "a" * 40,
            "initialization": "random",
        },
        "data": {
            "name": "test/data",
            "revision": "b" * 40,
            "split": "train",
            "text_column": "text",
            "max_documents": 8,
        },
        "tokenizer": {"name": "test/tokenizer", "revision": "c" * 40},
        "preprocessing": {
            "output_dir": str(tmp_path / "tokenized"),
            "cache_id": "lifecycle-test",
            "block_size": 4,
            "append_eos": True,
            "overwrite": False,
        },
        "validation": {
            "enabled": True,
            "split": "validation",
            "max_documents": 8,
            "partition": None,
            "partition_scheme": None,
            "partition_seed": None,
            "partition_hash": None,
            "batch_size": 1,
            "eval_batches": 1,
        },
        "activation_histograms": {
            "selected_runs": [
                {"label": "Source", "config_id": "01-source", "run_id": "001-source"}
            ],
            "bins": 4,
            "range_min": -1.0,
            "range_max": 1.0,
            "thresholds": [0.0, 0.1],
            "sites": ["h"],
        },
        "weight_histograms": {
            "selected_runs": [
                {"label": "Source", "config_id": "01-source", "run_id": "001-source"}
            ],
            "scope": "mlp_weights",
            "bins": 4,
            "range_min": -1.0,
            "range_max": 1.0,
        },
        "activation_propagation": {
            "selected_runs": [
                {"label": "Source", "config_id": "01-source", "run_id": "001-source"}
            ],
        },
        "evaluation": {"metric": "loss"},
        "run": {"seed": 0, "max_examples": 8},
        "output": {"dir": str(tmp_path / "results")},
    }


def _assert_running(run_dir: Path, *, mode: str) -> None:
    assert (run_dir / "config.yaml").is_file()
    manifest = _read_manifest(run_dir)
    assert manifest["status"] == "running"
    assert manifest["mode"] == mode
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()


def _assert_failed(run_dir: Path, *, mode: str) -> None:
    manifest = _read_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert manifest["mode"] == mode
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "dependency load failed",
    }
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_completed_source(tmp_path: Path, *, config_id: str, run_id: str) -> Path:
    run_dir = tmp_path / "results" / config_id / run_id
    checkpoint = run_dir / "checkpoints" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    validation_tokens = run_dir / "validation" / "tokens.int32.bin"
    validation_tokens.parent.mkdir()
    np.arange(16, dtype=np.int32).tofile(validation_tokens)
    (run_dir / "config.yaml").write_text("experiment_name: source\n", encoding="utf-8")
    _write_json(run_dir / "metrics.json", {})
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "manifest.json",
        {
            "config_id": config_id,
            "run_id": run_id,
            "status": "completed",
            "checkpoint": {"saved": True, "path": "checkpoints/final"},
            "tokenized_data": {
                "validation": {
                    "tokens_path": str(validation_tokens),
                    "dtype": "int32",
                    "tokens": 16,
                    "tokens_bytes": validation_tokens.stat().st_size,
                    "tokens_sha256": hashlib.sha256(
                        validation_tokens.read_bytes()
                    ).hexdigest(),
                    "block_size": 4,
                }
            },
        },
    )
    return run_dir


def _find_clipping_run(output_dir: Path, source_config_id: str) -> Path:
    candidates = list(output_dir.glob(f"{source_config_id}-clip-*/001-*"))
    assert len(candidates) == 1
    return candidates[0]
