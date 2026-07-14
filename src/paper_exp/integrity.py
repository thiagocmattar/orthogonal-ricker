from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from yaml import YAMLError

from paper_exp.config import CONFIG_FILE_RE, ConfigError, load_config
from paper_exp.run import CORE_RUN_ARTIFACTS


Severity = Literal["info", "warning", "error"]
RunStatus = Literal["event_stream", "partial", "complete"]


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
    if all((path / name).is_file() for name in CORE_RUN_ARTIFACTS):
        return "complete"
    if (path / "events.jsonl").is_file():
        return "event_stream"
    return "partial"


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
            load_config(path, allow_todos=True)
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
            if status == "event_stream":
                findings.append(
                    IntegrityFinding(
                        severity="warning",
                        code="run.event_stream_incomplete",
                        message=(
                            "Run has an event stream but not a complete artifact envelope; "
                            "it may be active or interrupted and must not be consumed as a "
                            f"completed run. Missing: {missing_text}."
                        ),
                        path=run_path,
                    )
                )
            else:
                findings.append(
                    IntegrityFinding(
                        severity="warning",
                        code="run.partial",
                        message=(
                            "Run has no event stream and its artifact envelope is incomplete; "
                            f"it is classified as partial. Missing: {missing_text}."
                        ),
                        path=run_path,
                    )
                )
    return findings


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
