"""Shared preflight checks for scientific commands and launch runners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from paper_exp.utils import collect_git_dirty


class LaunchError(RuntimeError):
    """Raised when a scientific command is unsafe to launch."""


EXPERIMENTS_DIR_NAME = "experiments"
SMOKE_SCAFFOLD_ID = "00-infrastructure-smoke"
SCAFFOLD_NAME_RE = re.compile(r"^(\d{2})-[a-z0-9]+-[a-z0-9][a-z0-9-]*$")
SCIENTIFIC_CONFIG_NAME_RE = re.compile(
    r"^(?!000)\d{3}-[a-z0-9][a-z0-9-]*\.yaml$"
)


@dataclass(frozen=True)
class ExperimentScaffold:
    """Resolved ownership boundary for one chronological launch tranche."""

    repository: Path
    scaffold_id: str
    path: Path

    @property
    def run_dir(self) -> Path:
        return self.path / "run"

    @property
    def raw_dir(self) -> Path:
        return self.path / "raw"

    @property
    def figs_dir(self) -> Path:
        return self.path / "figs"

    @property
    def runner_path(self) -> Path:
        return self.run_dir / "runner.py"

    @property
    def is_smoke(self) -> bool:
        return self.scaffold_id == SMOKE_SCAFFOLD_ID


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
    """Resolve one tracked config from an exact scaffold ``run/`` directory."""

    root = repository_path(repository)
    resolved = _resolve(path, root)
    scaffold = experiment_scaffold_for_config(resolved, repository=root)
    is_smoke_config = scaffold.is_smoke and resolved.name == "00-smoke.yaml"
    is_scientific_config = (
        not scaffold.is_smoke
        and SCIENTIFIC_CONFIG_NAME_RE.fullmatch(resolved.name) is not None
    )
    if not resolved.is_file() or not (is_smoke_config or is_scientific_config):
        raise LaunchError(
            "Config must be 00-infrastructure-smoke/run/00-smoke.yaml or a "
            "scientific CCC-<case>.yaml directly under "
            f"experiments/NN-<phase>-<tranche>/run/: {resolved}"
        )
    require_tracked_file(root, resolved)
    return root, resolved


def resolve_experiment_scaffold(
    scaffold_id: str,
    *,
    repository: str | Path | None = None,
) -> ExperimentScaffold:
    """Resolve one valid scaffold and require its three owned directories."""

    root = repository_path(repository)
    match = SCAFFOLD_NAME_RE.fullmatch(scaffold_id)
    if match is None or (int(match.group(1)) == 0) != (
        scaffold_id == SMOKE_SCAFFOLD_ID
    ):
        raise LaunchError(
            "Experiment scaffold must be NN-<phase>-<tranche>; prefix 00 is "
            f"reserved for {SMOKE_SCAFFOLD_ID}: {scaffold_id}"
        )
    experiments_root = (root / EXPERIMENTS_DIR_NAME).resolve()
    scaffold_path = (experiments_root / scaffold_id).resolve()
    if scaffold_path.parent != experiments_root or not scaffold_path.is_dir():
        raise LaunchError(f"Experiment scaffold does not exist: {scaffold_path}")
    scaffold = ExperimentScaffold(root, scaffold_id, scaffold_path)
    missing = [
        name
        for name, member in (
            ("run", scaffold.run_dir),
            ("raw", scaffold.raw_dir),
            ("figs", scaffold.figs_dir),
        )
        if not member.is_dir()
    ]
    if missing:
        raise LaunchError(
            f"Experiment scaffold {scaffold_id} is missing: {', '.join(missing)}."
        )
    return scaffold


def experiment_scaffold_for_config(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> ExperimentScaffold:
    """Return the scaffold owning an exact direct ``run/*.yaml`` path."""

    root = repository_path(repository)
    resolved = _resolve(path, root)
    experiments_root = (root / EXPERIMENTS_DIR_NAME).resolve()
    try:
        relative = resolved.relative_to(experiments_root)
    except ValueError as error:
        raise LaunchError(
            f"Config must be inside a scaffold run directory: {resolved}"
        ) from error
    if len(relative.parts) != 3 or relative.parts[1] != "run":
        raise LaunchError(
            "Config must be directly under "
            f"experiments/NN-<phase>-<tranche>/run/: {resolved}"
        )
    scaffold = resolve_experiment_scaffold(relative.parts[0], repository=root)
    if resolved.parent != scaffold.run_dir:
        raise LaunchError(f"Config does not resolve inside its scaffold: {resolved}")
    return scaffold


def resolve_launch_run_dir(
    path: str | Path,
    *,
    repository: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve one exact scaffold ``raw/<config-id>/<run-id>`` directory."""

    root = repository_path(repository)
    resolved = _resolve(path, root)
    experiments_root = (root / EXPERIMENTS_DIR_NAME).resolve()
    try:
        relative = resolved.relative_to(experiments_root)
    except ValueError as error:
        raise LaunchError(
            f"Run must be inside {experiments_root}: {resolved}"
        ) from error
    if len(relative.parts) != 4 or relative.parts[1] != "raw":
        raise LaunchError(
            "Run must be an exact experiments/NN-<phase>-<tranche>/raw/"
            f"<config-id>/<run-id> directory: {resolved}"
        )
    scaffold = resolve_experiment_scaffold(relative.parts[0], repository=root)
    if resolved.parent.parent != scaffold.raw_dir or not resolved.is_dir():
        raise LaunchError(
            f"Run must resolve inside its scaffold raw directory: {resolved}"
        )
    return root, resolved


def require_raw_output(
    config: Mapping[str, Any],
    *,
    repository: str | Path,
    config_path: str | Path,
) -> Path:
    root = Path(repository).resolve()
    scaffold = experiment_scaffold_for_config(config_path, repository=root)
    output = config.get("output")
    value = output.get("dir") if isinstance(output, Mapping) else None
    resolved = _resolve(value, root)
    expected = scaffold.raw_dir
    if resolved != expected:
        raise LaunchError(
            "Config output.dir must resolve to its owning scaffold raw directory "
            f"{expected}: {config_path}"
        )
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
