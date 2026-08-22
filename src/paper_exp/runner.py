"""Parent runner for one ordered, serial experiment-set launch."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
from typing import Any, Sequence

from paper_exp.config import load_config, validate_training_config
from paper_exp.launch import (
    LaunchError,
    direct_launch_guard,
    repository_path,
    require_raw_output,
    require_token_cache_output,
    require_tracked_file,
    resolve_experiment_scaffold,
    resolve_launch_config,
)


class RunnerError(LaunchError):
    """Raised when an experiment-set runner is malformed or a run fails."""


def run_launch(
    runner_path: str | Path,
    config_paths: Sequence[str | Path],
    *,
    repository: str | Path | None = None,
) -> list[Path]:
    """Run one plan-defined config list serially under a single launch lock."""

    root = repository_path(repository)
    runner = _resolve_runner(runner_path, root)
    if not config_paths:
        raise RunnerError(f"Runner has no configs: {runner}")

    try:
        configs = [
            resolve_launch_config(path, repository=root)[1]
            for path in config_paths
        ]
    except LaunchError as error:
        raise RunnerError(str(error)) from error
    expected_config_root = runner.parent
    misplaced = [path for path in configs if path.parent != expected_config_root]
    if misplaced:
        raise RunnerError(
            f"Runner {runner} may only use configs beside it in its scaffold "
            "run directory."
        )
    if len(set(configs)) != len(configs):
        raise RunnerError(f"Runner contains a duplicate config: {runner}")
    prefixes = [int(path.name.split("-", 1)[0]) for path in configs]
    if prefixes != sorted(prefixes) or len(prefixes) != len(set(prefixes)):
        raise RunnerError(f"Runner configs must be in strictly increasing numeric order: {runner}")
    folder_configs = {
        path.resolve()
        for path in expected_config_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    }
    listed_configs = set(configs)
    if listed_configs != folder_configs:
        missing = sorted(path.name for path in folder_configs - listed_configs)
        unexpected = sorted(path.name for path in listed_configs - folder_configs)
        details: list[str] = []
        if missing:
            details.append(f"missing from CONFIGS: {', '.join(missing)}")
        if unexpected:
            details.append(f"not present in the folder: {', '.join(unexpected)}")
        raise RunnerError(
            f"Runner {runner.name} must list exactly all YAML files in "
            f"{expected_config_root.relative_to(root).as_posix()}/ "
            f"({'; '.join(details)})."
        )

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in configs:
        config = load_config(path, allow_todos=False)
        validate_training_config(config)
        require_raw_output(config, repository=root, config_path=path)
        require_token_cache_output(config, repository=root, source=path)
        loaded.append((path, config))

    command = shlex.join(
        [Path(sys.executable).name, runner.relative_to(root).as_posix()]
    )
    completed: list[Path] = []
    with direct_launch_guard(repository=root):
        for index, (path, config) in enumerate(loaded, start=1):
            display = path.relative_to(root).as_posix()
            print(f"[{index}/{len(loaded)}] pretrain {display}", flush=True)
            completed.append(
                _run_one(config, config_path=path, command=command)
            )
    return completed


def _resolve_runner(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository / candidate).resolve()
    )
    experiments_root = (repository / "experiments").resolve()
    try:
        relative = resolved.relative_to(experiments_root)
    except ValueError as error:
        raise RunnerError(f"Runner must be inside {experiments_root}: {resolved}") from error
    if (
        len(relative.parts) != 3
        or relative.parts[1:] != ("run", "runner.py")
    ):
        raise RunnerError(
            "Runner must be experiments/NN-<phase>-<tranche>/run/runner.py."
        )
    scaffold = resolve_experiment_scaffold(relative.parts[0], repository=repository)
    if (
        scaffold.is_smoke
        or resolved != scaffold.runner_path
        or not resolved.is_file()
    ):
        raise RunnerError(
            "Scientific runner must be the tracked runner.py in a nonzero scaffold."
        )
    require_tracked_file(repository, resolved)
    return resolved


def _run_one(
    config: dict[str, Any],
    *,
    config_path: Path,
    command: str,
) -> Path:
    from paper_exp.training import run_training

    return run_training(
        config,
        config_path=config_path,
        command=command,
        mode="pretrain",
    )
