"""Shared preflight checks for scientific commands and launch runners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from typing import Any

from paper_exp.utils import collect_git_dirty


class LaunchError(RuntimeError):
    """Raised when a scientific command is unsafe to launch."""


def repository_path(repository: str | Path | None = None) -> Path:
    if repository is not None:
        return Path(repository).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchError("Scientific commands must run inside a Git repository.") from error
    return Path(result.stdout.strip()).resolve()


def resolve_launch_config(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the root smoke config or one tracked launch-folder config."""

    root = repository_path(repository)
    resolved = _resolve(path, root)
    config_root = (root / "configs").resolve()
    try:
        relative = resolved.relative_to(config_root)
    except ValueError as error:
        raise LaunchError(f"Config must be inside {config_root}: {resolved}") from error
    allowed_root = relative == Path("00-smoke.yaml")
    allowed_launch = len(relative.parts) == 2
    if (not allowed_root and not allowed_launch) or not resolved.is_file():
        raise LaunchError(
            "Config must be configs/00-smoke.yaml or a file under "
            f"configs/<launch-id>/: {resolved}"
        )
    require_tracked_file(root, resolved)
    return root, resolved


def resolve_launch_run_dir(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve one exact ``results/<config-id>/<run-id>`` directory."""

    root = repository_path(repository)
    resolved = _resolve(path, root)
    results_root = (root / "results").resolve()
    try:
        relative = resolved.relative_to(results_root)
    except ValueError as error:
        raise LaunchError(f"Run must be inside {results_root}: {resolved}") from error
    if len(relative.parts) != 2 or not resolved.is_dir():
        raise LaunchError(
            f"Run must be an exact results/<config-id>/<run-id> directory: {resolved}"
        )
    return root, resolved


def require_results_output(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    source: str | Path,
) -> Path:
    root = Path(repository).resolve()
    output = config.get("output")
    value = output.get("dir") if isinstance(output, Mapping) else None
    resolved = _resolve(value, root)
    expected = (root / "results").resolve()
    if resolved != expected:
        raise LaunchError(f"Config output.dir must resolve to {expected}: {source}")
    return resolved


def require_token_cache_output(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    source: str | Path,
) -> Path:
    root = Path(repository).resolve()
    preprocessing = config.get("preprocessing")
    value = (
        preprocessing.get("output_dir")
        if isinstance(preprocessing, Mapping)
        else None
    )
    resolved = _resolve(value, root)
    expected = (root / "data" / "tokenized").resolve()
    if resolved != expected:
        raise LaunchError(
            f"Config preprocessing.output_dir must resolve to {expected}: {source}"
        )
    return resolved


def require_tracked_file(repository: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise LaunchError(f"Launch file is outside the repository: {path}") from error
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchError(f"Launch file must be tracked by Git: {path}") from error


@contextmanager
def direct_launch_guard(
    *,
    repository: str | Path | None = None,
) -> Iterator[None]:
    """Require a reviewed plan, clean checkout, and one local launch owner."""

    root = repository_path(repository)
    _require_reviewed_plan(root)
    _require_clean_git_tree(root)
    with _exclusive_lock(root / "tmp" / "experiment.lock"):
        yield


def _require_reviewed_plan(repository: Path) -> None:
    plan_path = repository / "docs" / "experiment_plan.md"
    try:
        lines = plan_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise LaunchError(f"Cannot read the experiment plan: {plan_path}") from error
    status = next(
        (line.strip() for line in lines if line.strip().startswith("Plan status:")),
        None,
    )
    if status != "Plan status: reviewed":
        raise LaunchError(
            "Scientific launches are blocked until the first plan status is "
            "`Plan status: reviewed`."
        )


def _require_clean_git_tree(repository: Path) -> None:
    dirty = collect_git_dirty(repository)
    if dirty is None:
        raise LaunchError("Cannot determine Git status before launch.")
    if dirty:
        raise LaunchError("Commit or stash repository changes before a scientific launch.")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    try:
        try:
            with path.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(f"{os.getpid()}\n")
            acquired = True
        except FileExistsError as error:
            raise LaunchError(
                f"Another launch may be active because {path} already exists."
            ) from error
        yield
    finally:
        if acquired:
            path.unlink(missing_ok=True)


def _resolve(value: Any, repository: Path) -> Path:
    candidate = Path(str(value))
    return candidate.resolve() if candidate.is_absolute() else (repository / candidate).resolve()
