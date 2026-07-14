from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from paper_exp.cli import main
from paper_exp.integrity import (
    IntegrityFinding,
    check_repository,
    classify_run_directory,
)


VALID_CONFIG = """\
experiment_name: integrity_test
model:
  provider: huggingface
  name: pythia-14m-random
  architecture: EleutherAI/pythia-14m-deduped
  initialization: random
data:
  name: JeanKaddour/minipile
  split: train
evaluation:
  metric: training_loss
run:
  seed: 0
  max_examples: 1
output:
  dir: results
"""


def test_configs_are_validated_and_numbered(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (tmp_path / "configs" / "01-duplicate.yaml").write_text(
        VALID_CONFIG, encoding="utf-8"
    )
    (tmp_path / "configs" / "bad-name.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (tmp_path / "configs" / "03-invalid.yaml").write_text(
        "experiment_name: missing_fields\n", encoding="utf-8"
    )

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "config.duplicate_prefix", "configs")
    assert _has_finding(findings, "config.filename_invalid", "configs/bad-name.yaml")
    assert _has_finding(findings, "config.invalid", "configs/03-invalid.yaml")
    assert _has_finding(findings, "config.numbering_gap", "configs")


def test_run_directories_are_classified_from_artifacts(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    result_group = tmp_path / "results" / "01-test"

    active = result_group / "001-active"
    active.mkdir(parents=True)
    (active / "events.jsonl").write_text('{"event": "train"}\n', encoding="utf-8")

    partial = result_group / "002-partial"
    partial.mkdir()
    (partial / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")

    complete = result_group / "003-complete"
    complete.mkdir()
    for artifact in ("config.yaml", "metrics.json", "predictions.jsonl"):
        (complete / artifact).write_text("{}\n", encoding="utf-8")
    (complete / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "003-complete"}\n',
        encoding="utf-8",
    )

    running = result_group / "004-running"
    running.mkdir()
    (running / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (running / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "004-running", '
        '"status": "running", "started_at": "2026-01-01T00:00:00Z"}\n',
        encoding="utf-8",
    )

    failed = result_group / "005-failed"
    failed.mkdir()
    (failed / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (failed / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "005-failed", '
        '"status": "failed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z", '
        '"failure": {"type": "RuntimeError", "message": "test"}}\n',
        encoding="utf-8",
    )

    inconsistent = result_group / "006-inconsistent"
    inconsistent.mkdir()
    (inconsistent / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (inconsistent / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "006-inconsistent", '
        '"status": "completed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z"}\n',
        encoding="utf-8",
    )

    mismatched = result_group / "007-mismatched"
    mismatched.mkdir()
    for artifact in ("config.yaml", "metrics.json", "predictions.jsonl"):
        (mismatched / artifact).write_text("{}\n", encoding="utf-8")
    (mismatched / "manifest.json").write_text(
        '{"config_id": "wrong", "run_id": "007-mismatched"}\n',
        encoding="utf-8",
    )

    completed = result_group / "008-completed"
    completed.mkdir()
    for artifact in ("config.yaml", "metrics.json", "predictions.jsonl"):
        (completed / artifact).write_text("{}\n", encoding="utf-8")
    (completed / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "008-completed", '
        '"status": "completed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z"}\n',
        encoding="utf-8",
    )

    assert classify_run_directory(active) == "event_stream"
    assert classify_run_directory(partial) == "partial"
    assert classify_run_directory(complete) == "complete"
    assert classify_run_directory(running) == "running"
    assert classify_run_directory(failed) == "failed"
    assert classify_run_directory(inconsistent) == "inconsistent"
    assert classify_run_directory(mismatched) == "inconsistent"
    assert classify_run_directory(completed) == "complete"

    findings = check_repository(tmp_path)

    active_finding = _finding(
        findings,
        "run.event_stream_incomplete",
        "results/01-test/001-active",
    )
    partial_finding = _finding(findings, "run.partial", "results/01-test/002-partial")
    complete_finding = _finding(findings, "run.complete", "results/01-test/003-complete")
    running_finding = _finding(findings, "run.running", "results/01-test/004-running")
    failed_finding = _finding(findings, "run.failed", "results/01-test/005-failed")
    inconsistent_finding = _finding(
        findings,
        "run.inconsistent",
        "results/01-test/006-inconsistent",
    )
    mismatched_finding = _finding(
        findings,
        "run.inconsistent",
        "results/01-test/007-mismatched",
    )
    completed_finding = _finding(
        findings,
        "run.complete",
        "results/01-test/008-completed",
    )
    assert active_finding.severity == "warning"
    assert partial_finding.severity == "warning"
    assert complete_finding.severity == "info"
    assert running_finding.severity == "warning"
    assert failed_finding.severity == "warning"
    assert inconsistent_finding.severity == "error"
    assert mismatched_finding.severity == "error"
    assert completed_finding.severity == "info"


def test_literal_references_and_paper_outputs_are_checked(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (tmp_path / "figures" / "01-first.pdf").write_bytes(b"pdf")
    (tmp_path / "figures" / "01-second.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Present | Test | `configs/01-valid.yaml` | TODO | `figures/01-first.pdf` |
| Missing | Test | `configs/99-missing.yaml` | `results/99-test/001-run/` | `figures/02-missing.pdf` |
| Exploratory | Test | TODO | See `results/*-sweep/` | TODO |
""",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "Literal missing report: `report/01-missing/01-missing.pdf`.\n"
        "Ignored glob: `results/01-*/001-*`.\n",
        encoding="utf-8",
    )

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "figure.duplicate_prefix", "figures")
    assert _finding(
        findings, "reference.missing", "configs/99-missing.yaml"
    ).severity == "error"
    assert _finding(
        findings, "reference.missing", "results/99-test/001-run/"
    ).severity == "warning"
    assert _finding(
        findings, "reference.missing", "report/01-missing/01-missing.pdf"
    ).severity == "error"
    assert _finding(
        findings, "paper_map.output_missing", "figures/02-missing.pdf"
    ).severity == "warning"
    assert not any(finding.path == "results/*-sweep/" for finding in findings)


def test_check_is_read_only(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    before = _tree_snapshot(tmp_path)

    check_repository(tmp_path)

    assert _tree_snapshot(tmp_path) == before


def test_check_command_reports_warnings_and_supports_strict_mode(
    tmp_path: Path, capsys
) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(
        VALID_CONFIG, encoding="utf-8"
    )
    (tmp_path / "figures" / "01-first.pdf").write_bytes(b"pdf")
    (tmp_path / "figures" / "01-second.pdf").write_bytes(b"pdf")

    assert main(["check", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "WARNING [figure.duplicate_prefix]" in output
    assert "0 error(s), 1 warning(s)" in output

    assert main(["check", "--root", str(tmp_path), "--strict"]) == 1


def test_missing_ignored_artifacts_are_strict_only_failures(
    tmp_path: Path, capsys
) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "configs" / "01-valid.yaml").write_text(
        VALID_CONFIG, encoding="utf-8"
    )
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Archived | Test | `configs/01-valid.yaml` | `results/01-test/001-run/` | `figures/01-test.pdf` |
""",
        encoding="utf-8",
    )

    assert main(["check", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "WARNING [reference.missing] results/01-test/001-run/" in output
    assert "WARNING [paper_map.output_missing] figures/01-test.pdf" in output

    assert main(["check", "--root", str(tmp_path), "--strict"]) == 1


def test_current_repository_smoke() -> None:
    repository = Path(__file__).resolve().parents[1]

    findings = check_repository(repository)

    assert all(isinstance(finding, IntegrityFinding) for finding in findings)
    assert {field.name for field in fields(IntegrityFinding)} == {
        "severity",
        "code",
        "message",
        "path",
    }
    assert all(finding.severity in {"info", "warning", "error"} for finding in findings)


def _make_repository_skeleton(root: Path) -> None:
    for relative in ("configs", "results", "figures", "report", "docs"):
        (root / relative).mkdir(parents=True)
    (root / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Pending | Test | TODO | TODO | TODO |
""",
        encoding="utf-8",
    )
    (root / "docs" / "experiment_log.md").write_text(
        "# Experiment Log\n", encoding="utf-8"
    )


def _has_finding(
    findings: list[IntegrityFinding], code: str, path: str
) -> bool:
    return any(finding.code == code and finding.path == path for finding in findings)


def _finding(
    findings: list[IntegrityFinding], code: str, path: str
) -> IntegrityFinding:
    return next(
        finding
        for finding in findings
        if finding.code == code and finding.path == path
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
