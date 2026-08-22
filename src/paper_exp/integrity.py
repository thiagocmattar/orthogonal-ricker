from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from yaml import YAMLError, safe_load

from paper_exp.config import (
    CONFIG_FILE_RE,
    ConfigError,
    load_config,
    validate_data_config,
    validate_diagnostic_config,
    validate_smoke_config,
    validate_training_config,
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
_INLINE_CODE_RE = re.compile(r"`([^`\r\n]+)`")
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:configs|results|figures|report)/"
    r"[A-Za-z0-9][A-Za-z0-9._/{}*?\[\],-]*)"
)
_GLOB_CHARS = frozenset("*?[]{}")
_IGNORED_ARTIFACT_PREFIXES = ("results/", "figures/")


def check_repository(root: str | Path = ".") -> list[IntegrityFinding]:
    """Inspect repository conventions and artifact references without writing files."""

    repository = Path(root)
    findings: list[IntegrityFinding] = []
    findings.extend(_check_configs(repository))
    findings.extend(_check_runs(repository))
    findings.extend(_check_numbered_figures(repository))

    paper_map = repository / "docs" / "paper_map.md"
    paper_output_references: set[str] = set()
    if paper_map.is_file():
        output_findings, paper_output_references = _check_paper_map_outputs(
            repository, paper_map
        )
        findings.extend(output_findings)

    findings.extend(
        _check_document_references(
            repository,
            repository / "docs" / "paper_map.md",
            skip=paper_output_references,
        )
    )
    findings.extend(
        _check_document_references(
            repository, repository / "docs" / "experiment_log.md"
        )
    )
    return findings


def classify_run_directory(run_dir: str | Path) -> RunStatus:
    """Classify a run without guessing whether its process is still alive."""

    path = Path(run_dir)
    has_core_envelope = all((path / name).is_file() for name in CORE_RUN_ARTIFACTS)
    manifest_status, manifest_is_valid = _explicit_manifest_status(path)
    if not manifest_is_valid:
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


def _check_configs(repository: Path) -> list[IntegrityFinding]:
    config_dir = repository / "configs"
    if not config_dir.is_dir():
        return [
            IntegrityFinding(
                severity="error",
                code="config.directory_missing",
                message="Config directory does not exist.",
                path="configs",
            )
        ]

    findings: list[IntegrityFinding] = []
    prefixes: dict[int, list[Path]] = {}
    config_paths = sorted(
        path
        for path in config_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )

    for path in config_paths:
        prefix_match = _NUMBERED_PREFIX_RE.match(path.name)
        if prefix_match is not None:
            prefixes.setdefault(int(prefix_match.group(1)), []).append(path)

        if CONFIG_FILE_RE.fullmatch(path.name) is None:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="config.filename_invalid",
                    message=(
                        "Config filename must start with at least two digits and use "
                        "lowercase letters, digits, and hyphens."
                    ),
                    path=_relative_path(repository, path),
                )
            )
            continue

        try:
            config = load_config(
                path,
                allow_todos=(path.name == "00-smoke.yaml"),
            )
            if path.name == "00-smoke.yaml":
                validate_smoke_config(config)
            else:
                diagnostic_kinds = [
                    kind
                    for kind in (
                        "activation_histograms",
                        "weight_histograms",
                        "activation_propagation",
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
            if Path(output_dir).is_absolute() or Path(output_dir).as_posix() != "results":
                raise ConfigError("Config field output.dir must be the relative path 'results'.")
            preprocessing = config.get("preprocessing")
            if isinstance(preprocessing, dict):
                cache_output = str(preprocessing.get("output_dir", ""))
                if (
                    Path(cache_output).is_absolute()
                    or Path(cache_output).as_posix() != "data/tokenized"
                ):
                    raise ConfigError(
                        "Config field preprocessing.output_dir must be the relative path 'data/tokenized'."
                    )
        except (ConfigError, OSError, UnicodeError, YAMLError) as error:
            findings.append(
                IntegrityFinding(
                    severity="error",
                    code="config.invalid",
                    message=f"Generic config validation failed: {error}",
                    path=_relative_path(repository, path),
                )
            )

    for prefix, paths in sorted(prefixes.items()):
        if len(paths) <= 1:
            continue
        names = ", ".join(path.name for path in paths)
        findings.append(
            IntegrityFinding(
                severity="error",
                code="config.duplicate_prefix",
                message=f"Config prefix {prefix:02d} is used by: {names}.",
                path="configs",
            )
        )

    if prefixes:
        missing = sorted(set(range(1, max(prefixes) + 1)) - set(prefixes))
        if missing:
            findings.append(
                IntegrityFinding(
                    severity="warning",
                    code="config.numbering_gap",
                    message=(
                        "Sequential config prefixes are missing: "
                        f"{_format_number_ranges(missing)}."
                    ),
                    path="configs",
                )
            )
    return findings


def _check_runs(repository: Path) -> list[IntegrityFinding]:
    results_dir = repository / "results"
    if not results_dir.is_dir():
        return []

    findings: list[IntegrityFinding] = []
    for result_group in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in result_group.iterdir() if path.is_dir()):
            missing = [name for name in CORE_RUN_ARTIFACTS if not (run_dir / name).is_file()]
            run_path = _relative_path(repository, run_dir)
            status = classify_run_directory(run_dir)
            if status == "complete":
                findings.append(
                    IntegrityFinding(
                        severity="info",
                        code="run.complete",
                        message="Run has the complete core artifact envelope.",
                        path=run_path,
                    )
                )
                continue

            missing_text = ", ".join(missing)
            if status == "running":
                findings.append(
                    IntegrityFinding(
                        severity="warning",
                        code="run.running",
                        message=(
                            "Run manifest is explicitly running and must not be consumed "
                            f"as completed. Missing: {missing_text or 'none'}."
                        ),
                        path=run_path,
                    )
                )
            elif status == "failed":
                findings.append(
                    IntegrityFinding(
                        severity="warning",
                        code="run.failed",
                        message=(
                            "Run manifest records a failed terminal state. "
                            f"Missing: {missing_text or 'none'}."
                        ),
                        path=run_path,
                    )
                )
            elif status == "inconsistent":
                findings.append(
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
                )
    return findings


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
    status = manifest.get("status")
    if status is None:
        return None, False
    if status not in {"running", "failed", "completed"}:
        return None, False
    if not (run_dir / "config.yaml").is_file():
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
        }:
            validate_diagnostic_config(config, str(mode).replace("-", "_"))
    except ConfigError:
        return False
    specialized = {
        "activation-histograms": "activation_histograms.json",
        "weight-histograms": "weight_histograms.json",
        "activation-propagation": "activation_propagation.json",
        "clip-sweep": "clipping_frontier.jsonl",
    }
    if mode in {"pretrain", "calibrate"}:
        events = _read_jsonl(run_dir / "events.jsonl")
        if events is None or not any(row.get("event") == "train" for row in events):
            return False
    if mode in specialized:
        artifact_path = run_dir / specialized[str(mode)]
        if mode == "clip-sweep":
            rows = _read_jsonl(artifact_path)
            if rows is None or not rows:
                return False
        else:
            try:
                artifact = read_json(artifact_path)
            except (OSError, UnicodeError, ValueError):
                return False
            expected_schema = {
                "activation-histograms": 2,
                "weight-histograms": 1,
                "activation-propagation": 4,
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
            checkpoint_path = Path(checkpoint_path_text)
            if not checkpoint_path.is_absolute():
                checkpoint_path = run_dir.parents[2] / checkpoint_path
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


def _check_numbered_figures(repository: Path) -> list[IntegrityFinding]:
    figure_dir = repository / "figures"
    if not figure_dir.is_dir():
        return []

    prefixes: dict[int, list[Path]] = {}
    for path in sorted(figure_dir.glob("*.pdf")):
        match = _NUMBERED_PREFIX_RE.match(path.name)
        if match is not None:
            prefixes.setdefault(int(match.group(1)), []).append(path)

    findings: list[IntegrityFinding] = []
    for prefix, paths in sorted(prefixes.items()):
        if len(paths) <= 1:
            continue
        names = ", ".join(path.name for path in paths)
        findings.append(
            IntegrityFinding(
                severity="warning",
                code="figure.duplicate_prefix",
                message=f"Figure prefix {prefix:02d} is used by: {names}.",
                path="figures",
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
            if not reference.startswith(("figures/", "report/")):
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
    if reference.startswith(_IGNORED_ARTIFACT_PREFIXES):
        return "warning"
    return "error"


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
