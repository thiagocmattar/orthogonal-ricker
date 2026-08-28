from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from yaml import YAMLError, safe_load

from paper_exp.config import (
    ConfigError,
    load_config,
    validate_data_config,
    validate_diagnostic_config,
    validate_smoke_config,
    validate_training_config,
)
from paper_exp.diagnostics.sources import resolve_source_path
from paper_exp.design import (
    CATALOG_PATH,
    PLAN_PATH,
    DesignError,
    complete_config_sha256,
    tracked_training_identities,
    validate_catalog,
    validate_reviewed_design,
    validate_training_identity_fields,
)
from paper_exp.launch import (
    EXPERIMENTS_DIR_NAME,
    SCAFFOLD_NAME_RE,
    SMOKE_SCAFFOLD_ID,
    LaunchError,
    require_tracked_file,
)
from paper_exp.run import CORE_RUN_ARTIFACTS
from paper_exp.utils import read_json


Severity = Literal["info", "warning", "error"]
RunStatus = Literal[
    "running",
    "failed",
    "complete",
    "inconsistent",
]


@dataclass(frozen=True)
class IntegrityFinding:
    """One repository integrity observation."""

    severity: Severity
    code: str
    message: str
    path: str


_NUMBERED_PREFIX_RE = re.compile(r"^(\d+)-")
_SCIENTIFIC_CONFIG_RE = re.compile(
    r"^(\d{3})-[a-z0-9][a-z0-9-]*\.yaml$"
)
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_INLINE_CODE_RE = re.compile(r"`([^`\r\n]+)`")
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:experiments|report)/"
    r"[A-Za-z0-9][A-Za-z0-9._/{}*?\[\],-]*)"
)
_GLOB_CHARS = frozenset("*?[]{}")
_LEGACY_DIRECTORIES = ("configs", "runners", "results", "figures", "run-logs")
_SCAFFOLD_DIRECTORIES = frozenset({"run", "raw", "figs"})


def check_repository(root: str | Path = ".") -> list[IntegrityFinding]:
    """Inspect repository conventions and artifact references without writing files."""

    repository = Path(root)
    paper_map = repository / "docs" / "paper_map.md"
    experiment_log = repository / "docs" / "experiment_log.md"
    artifact_references = _document_artifact_references(paper_map, experiment_log)
    findings: list[IntegrityFinding] = _check_design(repository)
    findings.extend(_check_legacy_directories(repository))
    findings.extend(_check_experiments(repository))
    findings.extend(_check_runs(repository, references=artifact_references))

    paper_output_references: set[str] = set()
    if paper_map.is_file():
        output_findings, paper_output_references = _check_paper_map_outputs(
            repository, paper_map
        )
        findings.extend(output_findings)
    findings.extend(
        _check_numbered_figures(repository, references=paper_output_references)
    )

    findings.extend(
        _check_document_references(
            repository,
            repository / "docs" / "paper_map.md",
            skip=paper_output_references,
        )
    )
    findings.extend(
        _check_document_references(
            repository, experiment_log
        )
    )
    return findings


def classify_run_directory(run_dir: str | Path) -> RunStatus:
    """Classify a run without guessing whether its process is still alive."""

    path = Path(run_dir)
    has_core_envelope = all((path / name).is_file() for name in CORE_RUN_ARTIFACTS)
    manifest_status, manifest_is_valid = _explicit_manifest_status(path)
    if not manifest_is_valid:
        if has_core_envelope and _statusless_core_artifacts_are_coherent(path):
            return "complete"
        return "inconsistent"
    if manifest_status == "running":
        return "running"
    if manifest_status == "failed":
        return "failed"
    if manifest_status == "completed":
        return (
            "complete"
            if has_core_envelope and _completed_artifacts_are_coherent(path)
            else "inconsistent"
        )
    return "inconsistent"


def _check_legacy_directories(repository: Path) -> list[IntegrityFinding]:
    return [
        IntegrityFinding(
            severity="error",
            code="layout.legacy_directory",
            message="Legacy top-level experiment directory must be removed.",
            path=name,
        )
        for name in _LEGACY_DIRECTORIES
        if (repository / name).is_dir()
    ]


def _check_experiments(repository: Path) -> list[IntegrityFinding]:
    experiments_dir = repository / EXPERIMENTS_DIR_NAME
    if not experiments_dir.is_dir():
        return [
            IntegrityFinding(
                severity="error",
                code="experiment.directory_missing",
                message="Canonical experiment scaffold directory does not exist.",
                path=EXPERIMENTS_DIR_NAME,
            )
        ]

    findings: list[IntegrityFinding] = []
    scaffolds: list[Path] = []
    scientific_scaffold_prefixes: dict[int, list[Path]] = {}
    for path in sorted(child for child in experiments_dir.iterdir() if child.is_file()):
        if path.name != "README.md":
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="experiment.entry_invalid",
                    message="Only README.md may live beside experiment scaffolds.",
                    path=_relative_path(repository, path),
                )
            )
    for path in sorted(child for child in experiments_dir.iterdir() if child.is_dir()):
        match = SCAFFOLD_NAME_RE.fullmatch(path.name)
        if match is None or (int(match.group(1)) == 0) != (
            path.name == SMOKE_SCAFFOLD_ID
        ):
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="scaffold.name_invalid",
                    message=(
                        "Scaffolds must be NN-<phase>-<tranche>; prefix 00 is "
                        f"reserved for {SMOKE_SCAFFOLD_ID}."
                    ),
                    path=_relative_path(repository, path),
                )
            )
            continue
        scaffolds.append(path)
        prefix = int(match.group(1))
        if prefix:
            scientific_scaffold_prefixes.setdefault(prefix, []).append(path)

    if not (experiments_dir / SMOKE_SCAFFOLD_ID).is_dir():
        findings.append(
            IntegrityFinding(
                severity="error",
                code="scaffold.smoke_missing",
                message="The reserved infrastructure-smoke scaffold is missing.",
                path=f"{EXPERIMENTS_DIR_NAME}/{SMOKE_SCAFFOLD_ID}",
            )
        )

    for prefix, paths in sorted(scientific_scaffold_prefixes.items()):
        if len(paths) > 1:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="scaffold.duplicate_prefix",
                    message=(
                        f"Scaffold prefix {prefix:02d} is used by: "
                        + ", ".join(path.name for path in paths)
                        + "."
                    ),
                    path=EXPERIMENTS_DIR_NAME,
                )
            )
    if scientific_scaffold_prefixes:
        missing = sorted(
            set(range(1, max(scientific_scaffold_prefixes) + 1))
            - set(scientific_scaffold_prefixes)
        )
        if missing:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="scaffold.numbering_gap",
                    message=(
                        "Sequential scaffold prefixes are missing: "
                        f"{_format_number_ranges(missing)}."
                    ),
                    path=EXPERIMENTS_DIR_NAME,
                )
            )

    config_prefixes: dict[int, list[Path]] = {}
    ordered_config_prefixes: list[int] = []
    for scaffold in scaffolds:
        findings.extend(_check_scaffold_shape(repository, scaffold))
        run_dir = scaffold / "run"
        if not run_dir.is_dir():
            continue
        direct_configs = sorted(
            path
            for path in run_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _YAML_SUFFIXES
        )
        deeper_configs = sorted(
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _YAML_SUFFIXES
            and path.parent != run_dir
        )
        for path in deeper_configs:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="config.location_invalid",
                    message="Configs must live directly in their scaffold run directory.",
                    path=_relative_path(repository, path),
                )
            )

        is_smoke = scaffold.name == SMOKE_SCAFFOLD_ID
        runner_path = run_dir / "runner.py"
        if is_smoke:
            if runner_path.exists():
                findings.append(
                    IntegrityFinding(
                        severity="error",
                        code="config.smoke_runner_invalid",
                        message="The infrastructure-smoke scaffold is runner-free.",
                        path=_relative_path(repository, runner_path),
                    )
                )
            if not (run_dir / "00-smoke.yaml").is_file():
                findings.append(
                    IntegrityFinding(
                        severity="error",
                        code="config.smoke_missing",
                        message="The infrastructure-smoke config is missing.",
                        path=_relative_path(repository, run_dir / "00-smoke.yaml"),
                    )
                )
        else:
            findings.extend(_check_matching_runner(repository, runner_path))
            if not direct_configs:
                findings.append(
                    IntegrityFinding(
                        severity="error",
                        code="config.scaffold_empty",
                        message="A scientific scaffold must contain at least one config.",
                        path=_relative_path(repository, run_dir),
                    )
                )

        for path in direct_configs:
            config_is_smoke = is_smoke and path.name == "00-smoke.yaml"
            if not config_is_smoke:
                filename_match = _SCIENTIFIC_CONFIG_RE.fullmatch(path.name)
                if is_smoke or filename_match is None or int(filename_match.group(1)) == 0:
                    findings.append(
                        IntegrityFinding(
                            severity="error",
                            code="config.filename_invalid",
                            message=(
                                "Scientific configs must be CCC-<case>.yaml with a "
                                "prefix of 001 or greater."
                            ),
                            path=_relative_path(repository, path),
                        )
                    )
                    continue
                prefix = int(filename_match.group(1))
                config_prefixes.setdefault(prefix, []).append(path)
                ordered_config_prefixes.append(prefix)
            tracking = _tracked_file_finding(
                repository,
                path,
                code="config.untracked",
                message="Every scaffold config must be tracked by Git.",
            )
            if tracking is not None:
                findings.append(tracking)
            finding = _validate_config_file(
                repository,
                path,
                scaffold=scaffold,
                is_smoke=config_is_smoke,
            )
            if finding is not None:
                findings.append(finding)

    for prefix, paths in sorted(config_prefixes.items()):
        if len(paths) > 1:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="config.duplicate_prefix",
                    message=(
                        f"Config prefix {prefix:03d} is used by: "
                        + ", ".join(path.name for path in paths)
                        + "."
                    ),
                    path=EXPERIMENTS_DIR_NAME,
                )
            )
    if ordered_config_prefixes != sorted(ordered_config_prefixes):
        findings.append(
            IntegrityFinding(
                severity="error",
                code="config.order_invalid",
                message="Config prefixes must increase across chronological scaffolds.",
                path=EXPERIMENTS_DIR_NAME,
            )
        )
    if config_prefixes:
        missing = sorted(set(range(1, max(config_prefixes) + 1)) - set(config_prefixes))
        if missing:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="config.numbering_gap",
                    message=(
                        "Sequential config prefixes are missing: "
                        + ", ".join(f"{number:03d}" for number in missing)
                        + "."
                    ),
                    path=EXPERIMENTS_DIR_NAME,
                )
            )
    return findings


def _check_design(repository: Path) -> list[IntegrityFinding]:
    """Validate machine-readable design and tracked training identities."""

    repository = repository.resolve()
    plan_exists = (repository / PLAN_PATH).is_file()
    catalog_exists = (repository / CATALOG_PATH).is_file()
    if not plan_exists and not catalog_exists:
        # Small unit-test repositories may exercise only artifact layout.
        return []
    if not plan_exists or not catalog_exists:
        missing = PLAN_PATH if not plan_exists else CATALOG_PATH
        return [
            IntegrityFinding(
                severity="error",
                code="design.file_missing",
                message="The normative design manifest and case catalog must exist together.",
                path=missing.as_posix(),
            )
        ]

    try:
        catalog = validate_catalog(repository)
        review = validate_reviewed_design(repository, require_reviewed=False)
    except DesignError as error:
        return [
            IntegrityFinding(
                severity="error",
                code="design.invalid",
                message=str(error),
                path=PLAN_PATH.as_posix(),
            )
        ]

    findings: list[IntegrityFinding] = []
    try:
        identities = tracked_training_identities(repository)
    except DesignError as error:
        return [
            IntegrityFinding(
                severity="error",
                code="design.config_identity_invalid",
                message=str(error),
                path=EXPERIMENTS_DIR_NAME,
            )
        ]
    by_fingerprint: dict[str, list[Path]] = {}
    for path, group_id, fingerprint in identities:
        if group_id not in catalog.groups:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="design.config_group_unknown",
                    message=f"Training config names unknown case group {group_id}.",
                    path=_relative_path(repository, path),
                )
            )
        by_fingerprint.setdefault(fingerprint, []).append(path)
    for paths in by_fingerprint.values():
        if len(paths) <= 1:
            continue
        findings.append(
            IntegrityFinding(
                severity="error",
                code="design.duplicate_fingerprint",
                message=(
                    "One scientific condition and seed is allocated more than once: "
                    + ", ".join(_relative_path(repository, path) for path in paths)
                    + "."
                ),
                path=EXPERIMENTS_DIR_NAME,
            )
        )
    if review.status == "reviewed":
        reviewed = set(review.reviewed_groups)
        for path, group_id, _fingerprint in identities:
            if group_id in reviewed:
                continue
            if _config_has_indexed_completed_evidence(repository, path):
                # Historical evidence is preserved without extending the
                # reviewed group set that grants materialization and launch
                # authority.
                continue
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="design.config_group_unreviewed",
                    message=(
                        f"Training config case group {group_id} is outside active "
                        "reviewed scope. It may be preserved only when the experiment "
                        "log indexes an exact coherent completed run with the same "
                        "immutable config snapshot; preservation does not authorize "
                        "materialization or launch."
                    ),
                    path=_relative_path(repository, path),
                )
            )
    else:
        for path, _group_id, _fingerprint in identities:
            if _config_has_indexed_completed_evidence(repository, path):
                continue
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="design.config_while_placeholder",
                    message=(
                        "A placeholder plan may preserve a scientific training config "
                        "only when the experiment log indexes an exact coherent "
                        "completed run with the same immutable config snapshot."
                    ),
                    path=_relative_path(repository, path),
                )
            )
    return findings


def _config_has_indexed_completed_evidence(
    repository: Path, config_path: Path
) -> bool:
    """Recognize historical evidence without authorizing materialization or launch."""

    root = repository.resolve()
    resolved_config = config_path.resolve()
    try:
        relative_config = resolved_config.relative_to(root).as_posix()
    except ValueError:
        return False
    config_parts = relative_config.split("/")
    if (
        len(config_parts) != 4
        or config_parts[0] != EXPERIMENTS_DIR_NAME
        or config_parts[2] != "run"
        or not config_parts[3].endswith(".yaml")
    ):
        return False

    try:
        with resolved_config.open("r", encoding="utf-8-sig") as handle:
            tracked_config = safe_load(handle) or {}
    except (OSError, UnicodeError, YAMLError):
        return False
    if not isinstance(tracked_config, dict):
        return False
    tracked_sha = complete_config_sha256(tracked_config)
    scaffold_id = config_parts[1]
    config_id = Path(config_parts[3]).stem

    experiment_log = root / "docs" / "experiment_log.md"
    if not experiment_log.is_file():
        return False
    try:
        log_text = experiment_log.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return False

    for match in _REFERENCE_RE.finditer(log_text):
        reference = match.group(1).rstrip(".,;").rstrip("/")
        if any(character in reference for character in _GLOB_CHARS):
            continue
        parts = reference.split("/")
        if (
            len(parts) != 5
            or parts[:3] != [EXPERIMENTS_DIR_NAME, scaffold_id, "raw"]
            or parts[3] != config_id
        ):
            continue
        run_dir = root / Path(reference)
        try:
            manifest = read_json(run_dir / "manifest.json")
        except (OSError, UnicodeError, ValueError):
            continue
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "completed"
            or manifest.get("mode") != "pretrain"
        ):
            continue
        if classify_run_directory(run_dir) != "complete":
            continue
        try:
            with (run_dir / "config.yaml").open(
                "r", encoding="utf-8-sig"
            ) as handle:
                run_config = safe_load(handle) or {}
        except (OSError, UnicodeError, YAMLError):
            continue
        if (
            isinstance(run_config, dict)
            and complete_config_sha256(run_config) == tracked_sha
        ):
            return True
    return False


def _check_scaffold_shape(repository: Path, scaffold: Path) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    for name in sorted(_SCAFFOLD_DIRECTORIES):
        path = scaffold / name
        if not path.is_dir():
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="scaffold.directory_missing",
                    message="Scaffold must contain run, raw, and figs directories.",
                    path=_relative_path(repository, path),
                )
            )
            continue
        if name in {"raw", "figs"}:
            keeper = path / ".gitkeep"
            if not keeper.is_file():
                findings.append(
                    IntegrityFinding(
                        severity="error",
                        code="scaffold.keeper_missing",
                        message=(
                            "Generated-output directories require a tracked "
                            ".gitkeep file."
                        ),
                        path=_relative_path(repository, keeper),
                    )
                )
            else:
                tracking = _tracked_file_finding(
                    repository,
                    keeper,
                    code="scaffold.keeper_untracked",
                    message="Scaffold directory keepers must be tracked by Git.",
                )
                if tracking is not None:
                    findings.append(tracking)
    for path in sorted(child for child in scaffold.iterdir() if child.is_dir()):
        if path.name not in _SCAFFOLD_DIRECTORIES:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="scaffold.directory_invalid",
                    message="Scaffold contains an unexpected owned directory.",
                    path=_relative_path(repository, path),
                )
            )
    for path in sorted(child for child in scaffold.iterdir() if child.is_file()):
        findings.append(
            IntegrityFinding(
                severity="error",
                code="scaffold.entry_invalid",
                message="Files belong inside run, raw, or figs, not beside them.",
                path=_relative_path(repository, path),
            )
        )
    return findings


def _check_matching_runner(repository: Path, runner_path: Path) -> list[IntegrityFinding]:
    if not runner_path.is_file():
        return [
            IntegrityFinding(
                severity="error",
                code="config.runner_missing",
                message="Scientific scaffold has no run/runner.py.",
                path=_relative_path(repository, runner_path),
            )
        ]
    finding = _tracked_file_finding(
        repository,
        runner_path,
        code="config.runner_untracked",
        message="Scientific run/runner.py must be tracked by Git.",
    )
    return [] if finding is None else [finding]


def _tracked_file_finding(
    repository: Path,
    path: Path,
    *,
    code: str,
    message: str,
) -> IntegrityFinding | None:
    try:
        require_tracked_file(repository.resolve(), path.resolve())
    except LaunchError:
        return IntegrityFinding(
            severity="error",
            code=code,
            message=message,
            path=_relative_path(repository, path),
        )
    return None


def _validate_config_file(
    repository: Path,
    path: Path,
    *,
    scaffold: Path,
    is_smoke: bool,
) -> IntegrityFinding | None:
    try:
        config = load_config(path, allow_todos=is_smoke)
        if is_smoke:
            validate_smoke_config(config)
        else:
            diagnostic_kinds = [
                kind
                for kind in (
                    "activation_histograms",
                    "weight_histograms",
                    "activation_propagation",
                    "clipping_frontier",
                )
                if kind in config
            ]
            if "training" in config and diagnostic_kinds:
                raise ConfigError(
                    "A config cannot combine training with a diagnostic workflow."
                )
            if len(diagnostic_kinds) > 1:
                raise ConfigError(
                    "A config must select exactly one diagnostic workflow."
                )
            if "training" in config:
                validate_training_config(config)
            elif diagnostic_kinds:
                validate_diagnostic_config(config, diagnostic_kinds[0])
            elif "tokenizer" in config or "preprocessing" in config:
                validate_data_config(config)
            else:
                raise ConfigError(
                    "Non-smoke configs must declare one recognized workflow section."
                )
        output_dir = str(config.get("output", {}).get("dir", ""))
        expected_output = Path(EXPERIMENTS_DIR_NAME, scaffold.name, "raw").as_posix()
        if (
            Path(output_dir).is_absolute()
            or Path(output_dir).as_posix() != expected_output
        ):
            raise ConfigError(
                "Config field output.dir must be the portable relative path "
                f"'{expected_output}'."
            )
        preprocessing = config.get("preprocessing")
        if isinstance(preprocessing, dict):
            cache_output = str(preprocessing.get("output_dir", ""))
            if (
                Path(cache_output).is_absolute()
                or Path(cache_output).as_posix() != "data/tokenized"
            ):
                raise ConfigError(
                    "Config field preprocessing.output_dir must be the relative path "
                    "'data/tokenized'."
                )
    except (ConfigError, OSError, UnicodeError, YAMLError) as error:
        return IntegrityFinding(
            severity="error",
            code="config.invalid",
            message=f"Generic config validation failed: {error}",
            path=_relative_path(repository, path),
        )
    return None


def _check_runs(
    repository: Path, *, references: set[str]
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    referenced_groups = {
        (parts[1], parts[3])
        for reference in references
        if len(parts := reference.rstrip("/").split("/")) >= 4
        and parts[0] == EXPERIMENTS_DIR_NAME
        and parts[2] == "raw"
    }
    for scaffold in _canonical_scaffolds(repository):
        raw_dir = scaffold / "raw"
        run_recipe_dir = scaffold / "run"
        if not raw_dir.is_dir():
            continue
        current_config_ids = {
            path.stem
            for path in run_recipe_dir.glob("*.yaml")
            if path.is_file()
        }
        for result_group in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
            belongs_to_current_config = any(
                result_group.name == config_id
                or result_group.name.startswith(f"{config_id}-clip-")
                for config_id in current_config_ids
            )
            if (
                not belongs_to_current_config
                and (scaffold.name, result_group.name) not in referenced_groups
            ):
                continue
            for run_dir in sorted(path for path in result_group.iterdir() if path.is_dir()):
                findings.extend(_check_run_directory(repository, run_dir))
    return findings


def _check_run_directory(repository: Path, run_dir: Path) -> list[IntegrityFinding]:
    missing = [name for name in CORE_RUN_ARTIFACTS if not (run_dir / name).is_file()]
    run_path = _relative_path(repository, run_dir)
    status = classify_run_directory(run_dir)
    if status == "complete":
        return [
            IntegrityFinding(
                severity="info",
                code="run.complete",
                message="Run has the complete core artifact envelope.",
                path=run_path,
            )
        ]

    missing_text = ", ".join(missing)
    if status == "running":
        return [
            IntegrityFinding(
                severity="warning",
                code="run.running",
                message=(
                    "Run manifest is explicitly running and must not be consumed "
                    f"as completed. Missing: {missing_text or 'none'}."
                ),
                path=run_path,
            )
        ]
    if status == "failed":
        return [
            IntegrityFinding(
                severity="warning",
                code="run.failed",
                message=(
                    "Run manifest records a failed terminal state. "
                    f"Missing: {missing_text or 'none'}."
                ),
                path=run_path,
            )
        ]
    return [
        IntegrityFinding(
            severity="error",
            code="run.inconsistent",
            message=(
                "Run has no valid explicit lifecycle manifest, or claims "
                "completion without the core artifact envelope. "
                f"Missing: {missing_text or 'none'}."
            ),
            path=run_path,
        )
    ]


def _canonical_scaffolds(repository: Path) -> list[Path]:
    experiments_dir = repository / EXPERIMENTS_DIR_NAME
    if not experiments_dir.is_dir():
        return []
    return [
        path
        for path in sorted(child for child in experiments_dir.iterdir() if child.is_dir())
        if (match := SCAFFOLD_NAME_RE.fullmatch(path.name)) is not None
        and (int(match.group(1)) == 0) == (path.name == SMOKE_SCAFFOLD_ID)
    ]


def _explicit_manifest_status(run_dir: Path) -> tuple[str | None, bool]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, False
    try:
        manifest = read_json(manifest_path)
    except (OSError, UnicodeError, ValueError):
        return None, False
    if not isinstance(manifest, dict):
        return None, False
    if manifest.get("config_id") != run_dir.parent.name:
        return None, False
    if manifest.get("run_id") != run_dir.name:
        return None, False
    if not _manifest_tranche_is_coherent(run_dir, manifest):
        return None, False
    status = manifest.get("status")
    if status is None:
        return None, False
    if status not in {"running", "failed", "completed"}:
        return None, False
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return None, False
    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            config = safe_load(handle) or {}
    except (OSError, UnicodeError, YAMLError):
        return None, False
    if not isinstance(config, dict):
        return None, False
    identity = config.get("identity")
    if manifest.get("mode") in {"pretrain", "calibrate"} and identity is None:
        return None, False
    if identity is not None:
        try:
            validate_training_identity_fields(config)
        except DesignError:
            return None, False
        if (
            manifest.get("case_group_id") != identity["group_id"]
            or manifest.get("condition_fingerprint")
            != identity["condition_fingerprint"]
            or manifest.get("training_implementation_id")
            != identity["training_implementation_id"]
            or manifest.get("config_sha256") != complete_config_sha256(config)
        ):
            return None, False
    if not _is_nonempty_string(manifest.get("started_at")):
        return None, False
    if not _is_nonempty_string(manifest.get("mode")):
        return None, False
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", git_commit) is None:
        return None, False
    if not isinstance(manifest.get("git_dirty"), bool):
        return None, False
    if status == "running":
        if "finished_at" in manifest or "failure" in manifest:
            return None, False
    else:
        if not _is_nonempty_string(manifest.get("finished_at")):
            return None, False
    if status == "failed":
        failure = manifest.get("failure")
        if not isinstance(failure, dict):
            return None, False
        if not _is_nonempty_string(failure.get("type")):
            return None, False
        if not isinstance(failure.get("message"), str):
            return None, False
    elif "failure" in manifest:
        return None, False
    return status, True


def _completed_artifacts_are_coherent(run_dir: Path) -> bool:
    try:
        manifest = read_json(run_dir / "manifest.json")
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(manifest, dict):
        return False
    mode = manifest.get("mode")
    if mode != "smoke" and manifest.get("git_dirty") is not False:
        return False
    try:
        with (run_dir / "config.yaml").open("r", encoding="utf-8-sig") as handle:
            config = safe_load(handle) or {}
        metrics = read_json(run_dir / "metrics.json")
        predictions = _read_jsonl(run_dir / "predictions.jsonl")
    except (OSError, UnicodeError, ValueError, YAMLError):
        return False
    if not isinstance(config, dict) or not isinstance(metrics, dict):
        return False
    if predictions is None:
        return False

    try:
        if mode == "smoke":
            validate_smoke_config(config)
        elif mode in {"pretrain", "calibrate"}:
            validate_training_config(config)
        elif mode == "prepare-data":
            validate_data_config(config)
        elif mode in {
            "activation-histograms",
            "weight-histograms",
            "activation-propagation",
            "clipping-frontier",
        }:
            validate_diagnostic_config(config, str(mode).replace("-", "_"))
    except ConfigError:
        return False
    specialized = {
        "activation-histograms": "activation_histograms.json",
        "weight-histograms": "weight_histograms.json",
        "activation-propagation": "activation_propagation.json",
        "clipping-frontier": "clipping_frontier.jsonl",
        "clip-sweep": "clipping_frontier.jsonl",
    }
    if mode in {"pretrain", "calibrate"}:
        events = _read_jsonl(run_dir / "events.jsonl")
        if events is None or not any(row.get("event") == "train" for row in events):
            return False
    if mode in specialized:
        artifact_path = run_dir / specialized[str(mode)]
        if mode in {"clip-sweep", "clipping-frontier"}:
            rows = _read_jsonl(artifact_path)
            if rows is None or not rows:
                return False
            if mode == "clipping-frontier":
                if predictions != rows:
                    return False
                from paper_exp.diagnostics.clipping_frontier import (
                    validate_completed_clipping_frontier_artifacts,
                )

                try:
                    validate_completed_clipping_frontier_artifacts(
                        run_dir=run_dir,
                        config=config,
                        manifest=manifest,
                        metrics=metrics,
                        rows=rows,
                        repository=_repository_for_run_dir(run_dir),
                    )
                except (
                    KeyError,
                    LaunchError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    YAMLError,
                ):
                    return False
        else:
            try:
                artifact = read_json(artifact_path)
            except (OSError, UnicodeError, ValueError):
                return False
            expected_schema = {
                "activation-histograms": 3,
                "weight-histograms": 1,
                "activation-propagation": 5,
            }[str(mode)]
            if not isinstance(artifact, dict) or artifact.get("schema_version") != expected_schema:
                return False
    if mode == "prepare-data":
        tokenized_data = manifest.get("tokenized_data")
        if not isinstance(tokenized_data, dict) or not isinstance(
            tokenized_data.get("train"), dict
        ):
            return False
    if mode not in {
        "smoke",
        "pretrain",
        "calibrate",
        "prepare-data",
        *specialized,
    }:
        return False

    if mode in {"pretrain", "calibrate"}:
        checkpoint = config.get("checkpoint") if isinstance(config, dict) else None
        if isinstance(checkpoint, dict) and checkpoint.get("save_final") is True:
            checkpoint_manifest = manifest.get("checkpoint")
            if not isinstance(checkpoint_manifest, dict) or checkpoint_manifest.get("saved") is not True:
                return False
            checkpoint_path_text = checkpoint_manifest.get("path")
            if not isinstance(checkpoint_path_text, str) or not checkpoint_path_text.strip():
                return False
            try:
                checkpoint_path = resolve_source_path(
                    checkpoint_path_text,
                    source_run=run_dir,
                )
            except (OSError, ValueError):
                return False
            if not checkpoint_path.is_dir():
                return False
            if not (checkpoint_path / "config.json").is_file():
                return False
            if not any(
                (checkpoint_path / filename).is_file()
                for filename in ("model.safetensors", "model.safetensors.index.json")
            ):
                return False
            if checkpoint.get("save_optimizer") is True and not (
                checkpoint_path / "optimizer.pt"
            ).is_file():
                return False
    return True


def _statusless_core_artifacts_are_coherent(run_dir: Path) -> bool:
    """Recognize pre-lifecycle runs without applying the current config schema."""

    try:
        manifest = read_json(run_dir / "manifest.json")
        with (run_dir / "config.yaml").open("r", encoding="utf-8-sig") as handle:
            config = safe_load(handle) or {}
        metrics = read_json(run_dir / "metrics.json")
        predictions = _read_jsonl(run_dir / "predictions.jsonl")
    except (OSError, UnicodeError, ValueError, YAMLError):
        return False
    return (
        isinstance(manifest, dict)
        and "status" not in manifest
        and manifest.get("config_id") == run_dir.parent.name
        and manifest.get("run_id") == run_dir.name
        and _manifest_tranche_is_coherent(run_dir, manifest)
        and isinstance(config, dict)
        and bool(config)
        and isinstance(metrics, dict)
        and predictions is not None
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return None
                rows.append(row)
    except (OSError, UnicodeError, ValueError):
        return None
    return rows


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _manifest_tranche_is_coherent(run_dir: Path, manifest: dict[str, Any]) -> bool:
    if run_dir.parent.parent.name != "raw":
        return True
    expected = run_dir.parents[2].name
    return manifest.get("tranche_id") in {None, expected}


def _repository_for_run_dir(run_dir: Path) -> Path:
    for parent in run_dir.resolve().parents:
        if parent.name == EXPERIMENTS_DIR_NAME:
            return parent.parent
    raise ValueError(f"Run is not inside the repository experiments tree: {run_dir}")


def _check_numbered_figures(
    repository: Path, *, references: set[str]
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []
    referenced_prefixes: dict[str, set[int]] = {}
    for reference in references:
        parts = reference.rstrip("/").split("/")
        if len(parts) < 4 or parts[0] != EXPERIMENTS_DIR_NAME or parts[2] != "figs":
            continue
        match = _NUMBERED_PREFIX_RE.match(parts[-1])
        if match is not None:
            referenced_prefixes.setdefault(parts[1], set()).add(int(match.group(1)))

    for scaffold in _canonical_scaffolds(repository):
        selected = referenced_prefixes.get(scaffold.name, set())
        if not selected:
            continue
        figure_dir = scaffold / "figs"
        prefixes: dict[int, list[Path]] = {}
        for path in sorted(figure_dir.glob("*.pdf")):
            match = _NUMBERED_PREFIX_RE.match(path.name)
            if match is not None and int(match.group(1)) in selected:
                prefixes.setdefault(int(match.group(1)), []).append(path)
        for prefix, paths in sorted(prefixes.items()):
            if len(paths) <= 1:
                continue
            findings.append(
                IntegrityFinding(
                    severity="warning",
                    code="figure.duplicate_prefix",
                    message=(
                        f"Figure prefix {prefix:02d} is used by: "
                        + ", ".join(path.name for path in paths)
                        + "."
                    ),
                    path=_relative_path(repository, figure_dir),
                )
            )
    return findings


def _check_paper_map_outputs(
    repository: Path, paper_map: Path
) -> tuple[list[IntegrityFinding], set[str]]:
    findings: list[IntegrityFinding] = []
    output_references: set[str] = set()
    text = paper_map.read_text(encoding="utf-8")

    for line_number, line in enumerate(text.splitlines(), start=1):
        cells = _markdown_table_cells(line)
        if len(cells) < 5 or cells[0] in {"Paper item", "----------"}:
            continue
        if cells[0].startswith("---"):
            continue

        output_cell = cells[4]
        if "TODO" in output_cell.upper():
            continue
        references = list(_literal_references(output_cell))
        output_references.update(references)
        if not references:
            findings.append(
                IntegrityFinding(
                    severity="warning",
                    code="paper_map.output_unindexed",
                    message=f"Paper-map row {line_number} has no literal figure or report output.",
                    path=_relative_path(repository, paper_map),
                )
            )
            continue

        for reference in references:
            if not (
                reference.startswith("report/")
                or _is_scaffold_figure_reference(reference)
            ):
                continue
            if not (repository / Path(reference)).exists():
                findings.append(
                    IntegrityFinding(
                        severity=_missing_reference_severity(reference),
                        code="paper_map.output_missing",
                        message=(
                            f"Paper-map output referenced on line {line_number} does not exist."
                        ),
                        path=reference,
                    )
                )
    return findings, output_references


def _check_document_references(
    repository: Path, document: Path, *, skip: set[str] | None = None
) -> list[IntegrityFinding]:
    if not document.is_file():
        return [
            IntegrityFinding(
                severity="error",
                code="document.missing",
                message="Required integrity-index document does not exist.",
                path=_relative_path(repository, document),
            )
        ]

    skipped = skip or set()
    findings: list[IntegrityFinding] = []
    seen: set[str] = set()
    text = document.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        for reference in _literal_references(line):
            if reference in skipped or reference in seen:
                continue
            seen.add(reference)
            if not (repository / Path(reference)).exists():
                findings.append(
                    IntegrityFinding(
                        severity=_missing_reference_severity(reference),
                        code="reference.missing",
                        message=(
                            f"Literal reference from {_relative_path(repository, document)} "
                            f"line {line_number} does not exist."
                        ),
                        path=reference,
                    )
                )
    return findings


def _literal_references(text: str):
    for code_match in _INLINE_CODE_RE.finditer(text):
        code = code_match.group(1)
        if "TODO" in code.upper():
            continue
        for reference_match in _REFERENCE_RE.finditer(code):
            reference = reference_match.group(1).rstrip(".,;")
            if any(character in reference for character in _GLOB_CHARS):
                continue
            yield reference


def _document_artifact_references(*documents: Path) -> set[str]:
    references: set[str] = set()
    for document in documents:
        if not document.is_file():
            continue
        references.update(_literal_references(document.read_text(encoding="utf-8")))
    return references


def _markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _relative_path(repository: Path, path: Path) -> str:
    try:
        return path.relative_to(repository).as_posix()
    except ValueError:
        return path.as_posix()


def _missing_reference_severity(reference: str) -> Severity:
    parts = reference.rstrip("/").split("/")
    if (
        len(parts) >= 3
        and parts[0] == EXPERIMENTS_DIR_NAME
        and parts[2] in {"raw", "figs"}
    ):
        return "warning"
    return "error"


def _is_scaffold_figure_reference(reference: str) -> bool:
    parts = reference.rstrip("/").split("/")
    return (
        len(parts) >= 4
        and parts[0] == EXPERIMENTS_DIR_NAME
        and parts[2] == "figs"
    )


def _format_number_ranges(numbers: list[int]) -> str:
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(_format_number_range(start, previous))
        start = previous = number
    ranges.append(_format_number_range(start, previous))
    return ", ".join(ranges)


def _format_number_range(start: int, end: int) -> str:
    if start == end:
        return f"{start:02d}"
    return f"{start:02d}-{end:02d}"
