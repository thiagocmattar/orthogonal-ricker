from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paper_exp.clipping import run_clipping_sweep
from paper_exp.config import load_config
from paper_exp.run import make_experiment_id, make_experiment_dir
from paper_exp.utils import read_json


PRESSURE_SWEEP_NAME = "pressure_fixed_step_v1"
PRESSURE_SWEEP_START_INDEX = 12
PRESSURE_SWEEP_STEPS = 2048
PRESSURE_SWEEP_TOKENS = 134_217_728

RICKER_SCREEN_POINTS = (
    (0.03, 0.05, 0.05),
    (0.10, 0.05, 0.05),
    (0.30, 0.05, 0.05),
    (0.10, 0.02, 0.02),
    (0.10, 0.10, 0.10),
    (0.10, 0.05, 0.025),
    (0.10, 0.05, 0.10),
)

L1_SCREEN_WEIGHTS = (0.05, 0.15, 0.50, 1.00)

RICKER_HIGH_PRESSURE_POINTS = (
    (0.30, 0.10, 0.10),
    (0.30, 0.50, 0.50),
    (1.00, 0.05, 0.05),
    (1.00, 0.10, 0.10),
    (1.00, 0.50, 0.50),
)

L1_HIGH_PRESSURE_WEIGHTS = (2.00, 5.00)


@dataclass(frozen=True)
class SweepConfigSpec:
    index: int
    filename_slug: str
    experiment_name: str
    method: str
    pressure: dict[str, Any] | None

    @property
    def filename(self) -> str:
        return f"{self.index:02d}-{self.filename_slug}.yaml"


def pressure_fixed_step_specs() -> list[SweepConfigSpec]:
    specs = [
        SweepConfigSpec(
            index=PRESSURE_SWEEP_START_INDEX,
            filename_slug="pythia-14m-minipile-adamw-fixed-2048",
            experiment_name="pythia_14m_minipile_adamw_fixed_2048",
            method="adamw",
            pressure={
                "enabled": True,
                "method": "none",
                "sites": ["mlp_hiddens"],
                "weight": 0.0,
                "log_thresholds": [0.0, 0.001, 0.003, 0.01, 0.03],
            },
        )
    ]

    index = PRESSURE_SWEEP_START_INDEX + 1
    for method in ("ricker_naive", "orthogonal_ricker"):
        for weight, c_value, sigma in RICKER_SCREEN_POINTS:
            slug_method = method.replace("_", "-")
            specs.append(
                SweepConfigSpec(
                    index=index,
                    filename_slug=(
                        f"pythia-14m-minipile-{slug_method}-fixed-2048"
                        f"-w{_compact_float(weight)}-c{_compact_float(c_value)}-s{_compact_float(sigma)}"
                    ),
                    experiment_name=(
                        f"pythia_14m_minipile_{method}_fixed_2048"
                        f"_w{_compact_float(weight)}_c{_compact_float(c_value)}_s{_compact_float(sigma)}"
                    ),
                    method=method,
                    pressure={
                        "enabled": True,
                        "method": method,
                        "sites": ["mlp_hiddens"],
                        "weight": weight,
                        "ricker_c": c_value,
                        "ricker_sigma": sigma,
                        "log_thresholds": [0.0, 0.001, 0.003, 0.01, 0.03],
                    },
                )
            )
            index += 1

    for method in ("l1_naive", "orthogonal_l1"):
        for weight in L1_SCREEN_WEIGHTS:
            slug_method = method.replace("_", "-")
            specs.append(
                SweepConfigSpec(
                    index=index,
                    filename_slug=f"pythia-14m-minipile-{slug_method}-fixed-2048-w{_compact_float(weight)}",
                    experiment_name=f"pythia_14m_minipile_{method}_fixed_2048_w{_compact_float(weight)}",
                    method=method,
                    pressure={
                        "enabled": True,
                        "method": method,
                        "sites": ["mlp_hiddens"],
                        "weight": weight,
                        "log_thresholds": [0.0, 0.001, 0.003, 0.01, 0.03],
                    },
                )
            )
            index += 1

    for method in ("ricker_naive", "orthogonal_ricker"):
        for weight, c_value, sigma in RICKER_HIGH_PRESSURE_POINTS:
            slug_method = method.replace("_", "-")
            specs.append(
                SweepConfigSpec(
                    index=index,
                    filename_slug=(
                        f"pythia-14m-minipile-{slug_method}-fixed-2048"
                        f"-w{_compact_float(weight)}-c{_compact_float(c_value)}-s{_compact_float(sigma)}"
                    ),
                    experiment_name=(
                        f"pythia_14m_minipile_{method}_fixed_2048"
                        f"_w{_compact_float(weight)}_c{_compact_float(c_value)}_s{_compact_float(sigma)}"
                    ),
                    method=method,
                    pressure={
                        "enabled": True,
                        "method": method,
                        "sites": ["mlp_hiddens"],
                        "weight": weight,
                        "ricker_c": c_value,
                        "ricker_sigma": sigma,
                        "log_thresholds": [0.0, 0.001, 0.003, 0.01, 0.03],
                    },
                )
            )
            index += 1

    for method in ("l1_naive", "orthogonal_l1"):
        for weight in L1_HIGH_PRESSURE_WEIGHTS:
            slug_method = method.replace("_", "-")
            specs.append(
                SweepConfigSpec(
                    index=index,
                    filename_slug=f"pythia-14m-minipile-{slug_method}-fixed-2048-w{_compact_float(weight)}",
                    experiment_name=f"pythia_14m_minipile_{method}_fixed_2048_w{_compact_float(weight)}",
                    method=method,
                    pressure={
                        "enabled": True,
                        "method": method,
                        "sites": ["mlp_hiddens"],
                        "weight": weight,
                        "log_thresholds": [0.0, 0.001, 0.003, 0.01, 0.03],
                    },
                )
            )
            index += 1

    return specs


def write_pressure_fixed_step_configs(configs_dir: str | Path = "configs") -> list[Path]:
    output_dir = Path(configs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for spec in pressure_fixed_step_specs():
        config = pressure_fixed_step_config(spec)
        output_path = output_dir / spec.filename
        text = yaml.safe_dump(config, sort_keys=False)
        if output_path.exists() and output_path.read_text(encoding="utf-8") == text:
            outputs.append(output_path)
            continue
        output_path.write_text(text, encoding="utf-8")
        outputs.append(output_path)
    return outputs


def run_pressure_fixed_step_sweep(
    *,
    configs_dir: str | Path,
    command: str,
    start_at: str | None = None,
    stop_after: int | None = None,
) -> list[Path]:
    from paper_exp.calibration import run_calibration

    config_paths = _selected_config_paths(
        write_pressure_fixed_step_configs(configs_dir),
        start_at=start_at,
        stop_after=stop_after,
    )
    outputs = []
    for config_path in config_paths:
        completed = _latest_completed_training_run(config_path)
        if completed is not None:
            outputs.append(completed)
            continue
        config = load_config(config_path, allow_todos=False)
        run_dir = run_calibration(config, config_path=config_path, command=f"{command} :: {config_path}", mode="pretrain")
        _assert_completed_fixed_steps(run_dir)
        outputs.append(run_dir)
    return outputs


def run_pressure_fixed_step_clipping_sweeps(
    *,
    configs_dir: str | Path,
    command: str,
    thresholds: list[float],
    quantiles: list[float],
    eval_batches: int | None,
    seed: int,
    start_at: str | None = None,
    stop_after: int | None = None,
) -> list[Path]:
    config_paths = _selected_config_paths(
        write_pressure_fixed_step_configs(configs_dir),
        start_at=start_at,
        stop_after=stop_after,
    )
    outputs = []
    for config_path in config_paths:
        source_run = _latest_completed_training_run(config_path)
        if source_run is None:
            continue
        completed = _latest_completed_clipping_run(
            config_path,
            thresholds=thresholds,
            quantiles=quantiles,
            eval_batches=eval_batches,
            seed=seed,
        )
        if completed is not None:
            outputs.append(completed)
            continue
        outputs.append(
            run_clipping_sweep(
                checkpoint_run_dir=source_run,
                command=f"{command} :: {source_run}",
                thresholds=thresholds,
                quantiles=quantiles,
                eval_batches=eval_batches,
                seed=seed,
            )
        )
    return outputs


def pressure_fixed_step_config(spec: SweepConfigSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment_name": spec.experiment_name,
        "sweep": {
            "name": PRESSURE_SWEEP_NAME,
            "role": spec.method,
            "fixed_steps": PRESSURE_SWEEP_STEPS,
            "fixed_tokens": PRESSURE_SWEEP_TOKENS,
            "notes": "Screening run for planning the full activation-pressure ablation.",
        },
        "model": {
            "provider": "huggingface",
            "name": "pythia-14m-random",
            "architecture": "EleutherAI/pythia-14m-deduped",
            "revision": "main",
            "initialization": "random",
        },
        "data": {
            "name": "JeanKaddour/minipile",
            "revision": "main",
            "split": "train",
            "text_column": "text",
            "max_documents": None,
        },
        "evaluation": {"metric": "training_loss"},
        "run": {"seed": 0, "max_examples": 1_000_000},
        "tokenizer": {"name": "EleutherAI/pythia-14m-deduped", "revision": "main"},
        "preprocessing": {
            "output_dir": "data/tokenized",
            "cache_id": "03-pythia-14m-minipile-random-full-10min",
            "block_size": 2048,
            "append_eos": True,
            "overwrite": False,
        },
        "training": {
            "max_steps": PRESSURE_SWEEP_STEPS,
            "max_wall_seconds": None,
            "warmup_steps": 100,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.00003,
            "precision": "auto",
            "device": "auto",
            "log_every": 50,
        },
        "validation": {
            "enabled": True,
            "split": "validation",
            "max_documents": 500,
            "eval_every_steps": 250,
            "eval_batches": 8,
            "batch_size": 4,
        },
        "checkpoint": {"save_final": True, "save_optimizer": False},
        "output": {"dir": "results"},
    }
    if spec.pressure is not None:
        pressure = dict(spec.pressure)
        if spec.method.startswith("orthogonal_"):
            pressure["step_budget"] = 0.5
            pressure["eps"] = 1.0e-12
        config["activation_pressure"] = pressure
    return config


def _selected_config_paths(
    paths: list[Path],
    *,
    start_at: str | None,
    stop_after: int | None,
) -> list[Path]:
    selected = sorted(paths)
    if start_at:
        selected = [path for path in selected if path.name >= start_at or path.stem >= start_at]
    if stop_after is not None:
        selected = selected[:stop_after]
    return selected


def _latest_completed_training_run(config_path: Path) -> Path | None:
    config = load_config(config_path, allow_todos=False)
    experiment_id = make_experiment_id(config_path)
    experiment_dir = make_experiment_dir(config, experiment_id)
    runs = [path for path in sorted(experiment_dir.iterdir()) if (path / "metrics.json").exists()]
    for run_dir in reversed(runs):
        metrics = read_json(run_dir / "metrics.json")
        if int(metrics.get("calibration/optimizer_steps", -1)) == PRESSURE_SWEEP_STEPS:
            return run_dir
    return None


def _latest_completed_clipping_run(
    config_path: Path,
    *,
    thresholds: list[float],
    quantiles: list[float],
    eval_batches: int | None,
    seed: int,
) -> Path | None:
    config = load_config(config_path, allow_todos=False)
    experiment_id = f"{make_experiment_id(config_path)}-clipping-sweep"
    experiment_dir = Path(config["output"]["dir"]) / experiment_id
    if not experiment_dir.exists():
        return None
    runs = [path for path in sorted(experiment_dir.iterdir()) if (path / "clipping_frontier.jsonl").exists()]
    for run_dir in reversed(runs):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if not _same_float_list(manifest.get("thresholds", []), thresholds):
            continue
        if not _same_float_list(manifest.get("quantiles", []), quantiles):
            continue
        if manifest.get("eval_batches") != eval_batches:
            continue
        if int(manifest.get("seed", -1)) != seed:
            continue
        rows = [
            line
            for line in (run_dir / "clipping_frontier.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) == len(thresholds) + len(quantiles):
            return run_dir
    return None


def _assert_completed_fixed_steps(run_dir: Path) -> None:
    metrics = read_json(run_dir / "metrics.json")
    actual_steps = int(metrics["calibration/optimizer_steps"])
    if actual_steps != PRESSURE_SWEEP_STEPS:
        raise RuntimeError(
            f"Run ended before the fixed-step budget: {run_dir} "
            f"completed {actual_steps}, expected {PRESSURE_SWEEP_STEPS}."
        )


def _compact_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _same_float_list(left: list[Any], right: list[float], *, tolerance: float = 1e-12) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(float(left_value) - float(right_value)) <= tolerance for left_value, right_value in zip(left, right))
