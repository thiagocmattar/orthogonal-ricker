"""Parent runner for one ordered, serial experiment-set launch."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
from typing import Any, Sequence

from yaml import YAMLError, safe_load

from paper_exp.config import load_config, validate_training_config
from paper_exp.integrity import classify_run_directory
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
    prior_completed = [
        _completed_attempt_for_config(path, config)
        for path, config in loaded
    ]
    completed: list[Path] = []
    with direct_launch_guard(repository=root):
        # Close the preflight-to-lock race before creating any new attempt.
        prior_completed = [
            _completed_attempt_for_config(path, config)
            for path, config in loaded
        ]
        for index, ((path, config), existing) in enumerate(
            zip(loaded, prior_completed, strict=True), start=1
        ):
            display = path.relative_to(root).as_posix()
            if existing is not None:
                run_display = existing.relative_to(root).as_posix()
                print(
                    f"[{index}/{len(loaded)}] skip completed {display} -> {run_display}",
                    flush=True,
                )
                completed.append(existing)
                continue
            print(f"[{index}/{len(loaded)}] pretrain {display}", flush=True)
            completed.append(
                _run_one(config, config_path=path, command=command)
            )
    return completed


def _completed_attempt_for_config(
    config_path: Path,
    config: dict[str, Any],
) -> Path | None:
    """Return one reusable completion, or require a safe retry state."""

    result_group = config_path.parent.parent / "raw" / config_path.stem
    if result_group.is_symlink():
        raise RunnerError(
            f"Cannot resume {config_path.name}: result group is not a regular "
            f"directory: {result_group}"
        )
    if not result_group.exists():
        return None
    if not result_group.is_dir():
        raise RunnerError(
            f"Cannot resume {config_path.name}: result group is not a regular "
            f"directory: {result_group}"
        )

    attempts: list[tuple[Path, str]] = []
    for attempt in sorted(result_group.iterdir(), key=lambda path: path.name):
        if attempt.is_symlink() or not attempt.is_dir():
            attempts.append((attempt, "inconsistent"))
            continue
        status = classify_run_directory(attempt)
        if status != "inconsistent" and not _config_snapshot_matches(
            attempt / "config.yaml", config
        ):
            status = "inconsistent"
        attempts.append((attempt, status))

    unsafe = [
        (attempt, status)
        for attempt, status in attempts
        if status in {"running", "inconsistent"}
    ]
    if unsafe:
        details = ", ".join(
            f"{attempt.name}={status}" for attempt, status in unsafe
        )
        raise RunnerError(
            f"Cannot resume {config_path.name} safely; running or inconsistent "
            f"attempt state: {details}."
        )

    coherent_completed = [
        attempt for attempt, status in attempts if status == "complete"
    ]
    if len(coherent_completed) > 1:
        names = ", ".join(path.name for path in coherent_completed)
        raise RunnerError(
            f"Cannot resume {config_path.name}: multiple coherent completed "
            f"attempts are ambiguous: {names}."
        )
    return coherent_completed[0] if coherent_completed else None


def _config_snapshot_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            snapshot = safe_load(handle)
    except (OSError, UnicodeError, YAMLError):
        return False
    return isinstance(snapshot, dict) and snapshot == expected


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
