from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from paper_exp.run import create_run_dir, write_run_artifacts
from paper_exp.utils import build_manifest, read_json, write_json


MLP_WEIGHT_RE = re.compile(
    r"^gpt_neox\.layers\.(?P<layer>\d+)\.mlp\.(?P<name>dense_h_to_4h|dense_4h_to_h)\.weight$"
)
ATTENTION_WEIGHT_RE = re.compile(
    r"^gpt_neox\.layers\.(?P<layer>\d+)\.attention\.(?P<name>query_key_value|dense)\.weight$"
)
WEIGHT_SCOPES = {
    "mlp_weights": (MLP_WEIGHT_RE, "mlp_weights"),
    "attention_weights": (ATTENTION_WEIGHT_RE, "attention_weights"),
}


def run_weight_histograms(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    command: str,
    run_id: str | None = None,
) -> Path:
    torch, np, load_file = _load_dependencies()
    histogram_config = config["weight_histograms"]
    selected_runs = histogram_config["selected_runs"]
    scope = str(histogram_config.get("scope", "mlp_weights"))
    if scope not in WEIGHT_SCOPES:
        valid_scopes = ", ".join(sorted(WEIGHT_SCOPES))
        raise ValueError(f"Unsupported weight_histograms.scope {scope!r}; expected one of: {valid_scopes}.")
    bins = int(histogram_config["bins"])
    range_min = float(histogram_config["range_min"])
    range_max = float(histogram_config["range_max"])
    if bins <= 0:
        raise ValueError("weight_histograms.bins must be positive.")
    if range_min >= range_max:
        raise ValueError("weight_histograms.range_min must be less than range_max.")

    source_runs = [_find_latest_source_run(config, item["config_id"]) for item in selected_runs]
    experiment_id, numbered_run_id, output_dir = create_run_dir(config, config_path, run_id=run_id)
    bin_edges = np.linspace(range_min, range_max, bins + 1).tolist()
    results: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for selected, source_run in zip(selected_runs, source_runs, strict=True):
        print(f"Measuring weight histograms for {selected['label']} from {source_run}", flush=True)
        results.append(
            _measure_one_run(
                label=selected["label"],
                source_run=source_run,
                torch=torch,
                load_file=load_file,
                scope=scope,
                bins=bins,
                range_min=range_min,
                range_max=range_max,
            )
        )

    wall_seconds = time.perf_counter() - start_time
    total_weights = sum(layer["total"] for result in results for layer in result["layers"])
    metrics = {
        "weight_histograms/methods": len(results),
        "weight_histograms/layers": len(results[0]["layers"]) if results else 0,
        "weight_histograms/bins": bins,
        "weight_histograms/range_min": range_min,
        "weight_histograms/range_max": range_max,
        "weight_histograms/weights": total_weights,
        "weight_histograms/wall_seconds": wall_seconds,
        "weight_histograms/weights_per_second": total_weights / wall_seconds if wall_seconds > 0 else None,
    }
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        run_id=numbered_run_id,
        command=command,
        mode="weight-histograms",
        config_id=experiment_id,
        result_path=output_dir,
    )
    manifest["source_runs"] = [str(path) for path in source_runs]
    manifest["source_checkpoints"] = [str(Path(read_json(path / "manifest.json")["checkpoint"]["path"])) for path in source_runs]
    manifest["weight_histograms"] = {
        "scope": scope,
        "biases_included": False,
        "bins": bins,
        "range_min": range_min,
        "range_max": range_max,
        "selected_runs": selected_runs,
    }

    payload = {
        "schema_version": 1,
        "plot_title": histogram_config.get("plot_title"),
        "bin_edges": bin_edges,
        "range_min": range_min,
        "range_max": range_max,
        "bins": bins,
        "scope": scope,
        "biases_included": False,
        "methods": results,
    }
    write_run_artifacts(output_dir, config=config, metrics=metrics, manifest=manifest, predictions=[])
    write_json(output_dir / "weight_histograms.json", payload)
    return output_dir


def _measure_one_run(
    *,
    label: str,
    source_run: Path,
    torch: Any,
    load_file: Any,
    scope: str,
    bins: int,
    range_min: float,
    range_max: float,
) -> dict[str, Any]:
    source_manifest = read_json(source_run / "manifest.json")
    checkpoint_path = Path(source_manifest["checkpoint"]["path"])
    method_start = time.perf_counter()
    state = load_file(str(checkpoint_path / "model.safetensors"), device="cpu")
    pattern, layer_prefix = WEIGHT_SCOPES[scope]
    layers: dict[int, list[tuple[str, Any]]] = {}
    for name, value in state.items():
        match = pattern.match(name)
        if match is None:
            continue
        layer_index = int(match.group("layer"))
        layers.setdefault(layer_index, []).append((name, value.detach().float().reshape(-1)))
    if not layers:
        raise ValueError(f"No {scope} tensors found in {checkpoint_path}")

    layer_rows = []
    for layer_index in sorted(layers):
        tensor_names = [name for name, _ in sorted(layers[layer_index], key=lambda item: item[0])]
        flat = torch.cat([value for _, value in sorted(layers[layer_index], key=lambda item: item[0])])
        finite = torch.isfinite(flat)
        finite_values = flat[finite]
        counts = torch.histc(finite_values, bins=bins, min=range_min, max=range_max).cpu().double()
        total = int(flat.numel())
        underflow = int((finite_values < range_min).sum().detach().cpu())
        overflow = int((finite_values > range_max).sum().detach().cpu())
        layer_rows.append(
            {
                "name": f"{layer_prefix}.layer_{layer_index}",
                "tensor_names": tensor_names,
                "counts": [int(value) for value in counts.tolist()],
                "total": total,
                "in_range": int(counts.sum().item()),
                "underflow": underflow,
                "overflow": overflow,
                "nonfinite": int((~finite).sum().detach().cpu()),
                "underflow_fraction": underflow / total if total else None,
                "overflow_fraction": overflow / total if total else None,
            }
        )

    return {
        "label": label,
        "config_id": source_manifest["config_id"],
        "run_id": source_manifest["run_id"],
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "layers": layer_rows,
        "wall_seconds": time.perf_counter() - method_start,
    }


def _find_latest_source_run(config: dict[str, Any], config_id: str) -> Path:
    experiment_dir = Path(config["output"]["dir"]) / config_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing result directory for selected config: {experiment_dir}")
    candidates = []
    for run_dir in sorted(experiment_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        checkpoint_path = run_dir / "checkpoints" / "final" / "model.safetensors"
        if manifest_path.exists() and checkpoint_path.exists():
            manifest = read_json(manifest_path)
            candidates.append((int(manifest.get("run_sequence", 0)), run_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpointed runs found for selected config: {experiment_dir}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Weight histogram analysis requires numpy, torch, and safetensors.") from exc
    return torch, np, load_file
