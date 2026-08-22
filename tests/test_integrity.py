from __future__ import annotations

from pathlib import Path

import pytest

import paper_exp.integrity as integrity
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
  name: test-random-model
  architecture: test/architecture
  initialization: random
data:
  name: test/dataset
  revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  split: train
  text_column: text
  max_documents: 1
tokenizer:
  name: test/tokenizer
  revision: cccccccccccccccccccccccccccccccccccccccc
preprocessing:
  output_dir: data/tokenized
  cache_id: integrity-test
  block_size: 128
  append_eos: true
  overwrite: false
validation:
  enabled: false
evaluation:
  metric: training_loss
run:
  seed: 0
  max_examples: 1
output:
  dir: results
"""


def test_configs_are_validated_and_numbered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    folder = _make_launch_folder(tmp_path, "01-a1-grid")
    _allow_tracked_runners(monkeypatch)
    (folder / "001-valid.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (folder / "001-duplicate.yaml").write_text(
        VALID_CONFIG, encoding="utf-8"
    )
    (folder / "bad-name.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (folder / "003-invalid.yaml").write_text(
        "experiment_name: missing_fields\n", encoding="utf-8"
    )

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "config.duplicate_prefix", "configs")
    assert _has_finding(
        findings,
        "config.filename_invalid",
        "configs/01-a1-grid/bad-name.yaml",
    )
    assert _has_finding(
        findings,
        "config.invalid",
        "configs/01-a1-grid/003-invalid.yaml",
    )
    assert _has_finding(findings, "config.numbering_gap", "configs")


def test_scientific_config_folders_require_convention_and_matching_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    invalid_folder = tmp_path / "configs" / "a1-grid"
    invalid_folder.mkdir()
    (invalid_folder / "001-case.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    missing_runner_folder = tmp_path / "configs" / "01-a1-grid"
    missing_runner_folder.mkdir()
    (missing_runner_folder / "002-case.yaml").write_text(
        VALID_CONFIG, encoding="utf-8"
    )

    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "config.launch_folder_invalid",
        "configs/a1-grid",
    )
    assert _has_finding(
        findings,
        "config.runner_missing",
        "runners/01-a1-grid.py",
    )

    runner_path = tmp_path / "runners" / "01-a1-grid.py"
    runner_path.write_text("", encoding="utf-8")

    def reject_untracked(_repository: Path, _path: Path) -> None:
        raise integrity.LaunchError("not tracked")

    monkeypatch.setattr(integrity, "require_tracked_file", reject_untracked)
    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "config.runner_untracked",
        "runners/01-a1-grid.py",
    )


def test_scientific_config_scope_width_and_global_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    _allow_tracked_runners(monkeypatch)
    first_folder = _make_launch_folder(tmp_path, "01-a1-grid")
    second_folder = _make_launch_folder(tmp_path, "02-b1-screen")
    (first_folder / "002-second.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (first_folder / "000-zero.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (first_folder / "03-short.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (second_folder / "001-first.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    nested = second_folder / "extra"
    nested.mkdir()
    (nested / "003-too-deep.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    root_config = tmp_path / "configs" / "003-root.yaml"
    root_config.write_text(VALID_CONFIG, encoding="utf-8")

    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "config.filename_invalid",
        "configs/01-a1-grid/03-short.yaml",
    )
    assert _has_finding(
        findings,
        "config.filename_invalid",
        "configs/01-a1-grid/000-zero.yaml",
    )
    assert _has_finding(findings, "config.order_invalid", "configs")
    assert _has_finding(
        findings,
        "config.location_invalid",
        "configs/02-b1-screen/extra/003-too-deep.yaml",
    )
    assert _has_finding(
        findings,
        "config.location_invalid",
        "configs/003-root.yaml",
    )


def test_config_prefixes_are_unique_across_launch_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    _allow_tracked_runners(monkeypatch)
    first_folder = _make_launch_folder(tmp_path, "01-a1-grid")
    second_folder = _make_launch_folder(tmp_path, "02-b1-screen")
    (first_folder / "001-first.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (second_folder / "001-second.yaml").write_text(VALID_CONFIG, encoding="utf-8")

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "config.duplicate_prefix", "configs")


def test_run_directories_are_classified_from_artifacts(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "Audit `results/01-test/`.\n", encoding="utf-8"
    )
    result_group = tmp_path / "results" / "01-test"

    active = result_group / "001-active"
    active.mkdir(parents=True)
    (active / "events.jsonl").write_text('{"event": "train"}\n', encoding="utf-8")

    partial = result_group / "002-partial"
    partial.mkdir()
    (partial / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")

    complete = result_group / "003-complete"
    complete.mkdir()
    (complete / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (complete / "metrics.json").write_text("{}\n", encoding="utf-8")
    (complete / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (complete / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "003-complete", '
        '"mode": "smoke", "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
        encoding="utf-8",
    )

    running = result_group / "004-running"
    running.mkdir()
    (running / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (running / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "004-running", '
        '"status": "running", "started_at": "2026-01-01T00:00:00Z", '
        '"mode": "smoke", "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
        encoding="utf-8",
    )

    failed = result_group / "005-failed"
    failed.mkdir()
    (failed / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (failed / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "005-failed", '
        '"status": "failed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z", '
        '"failure": {"type": "RuntimeError", "message": "test"}, '
        '"mode": "smoke", "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
        encoding="utf-8",
    )

    inconsistent = result_group / "006-inconsistent"
    inconsistent.mkdir()
    (inconsistent / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (inconsistent / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "006-inconsistent", '
        '"status": "completed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z", '
        '"mode": "smoke", "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
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
    (completed / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (completed / "metrics.json").write_text("{}\n", encoding="utf-8")
    (completed / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (completed / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "008-completed", '
        '"status": "completed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z", '
        '"mode": "smoke", "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
        encoding="utf-8",
    )

    assert classify_run_directory(active) == "inconsistent"
    assert classify_run_directory(partial) == "inconsistent"
    assert classify_run_directory(complete) == "complete"
    assert classify_run_directory(running) == "running"
    assert classify_run_directory(failed) == "failed"
    assert classify_run_directory(inconsistent) == "inconsistent"
    assert classify_run_directory(mismatched) == "inconsistent"
    assert classify_run_directory(completed) == "complete"

    findings = check_repository(tmp_path)

    active_finding = _finding(findings, "run.inconsistent", "results/01-test/001-active")
    partial_finding = _finding(findings, "run.inconsistent", "results/01-test/002-partial")
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
    assert active_finding.severity == "error"
    assert partial_finding.severity == "error"
    assert complete_finding.severity == "info"
    assert running_finding.severity == "warning"
    assert failed_finding.severity == "warning"
    assert inconsistent_finding.severity == "error"
    assert mismatched_finding.severity == "error"
    assert completed_finding.severity == "info"


def test_literal_references_and_paper_outputs_are_checked(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "figures" / "01-first.pdf").write_bytes(b"pdf")
    (tmp_path / "figures" / "01-second.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Present | Test | TODO | TODO | `figures/01-first.pdf` |
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


def test_completed_run_rejects_corrupt_core_artifact(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    run_dir = tmp_path / "results" / "01-test" / "001-corrupt"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(VALID_CONFIG, encoding="utf-8")
    (run_dir / "metrics.json").write_text("not-json\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        '{"config_id": "01-test", "run_id": "001-corrupt", '
        '"status": "completed", "started_at": "2026-01-01T00:00:00Z", '
        '"finished_at": "2026-01-01T00:01:00Z", "mode": "smoke", '
        '"git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"git_dirty": false}\n',
        encoding="utf-8",
    )

    assert classify_run_directory(run_dir) == "inconsistent"


def test_check_is_read_only(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    before = _tree_snapshot(tmp_path)

    check_repository(tmp_path)

    assert _tree_snapshot(tmp_path) == before


def test_check_command_reports_warnings_and_supports_strict_mode(
    tmp_path: Path, capsys
) -> None:
    _make_repository_skeleton(tmp_path)
    (tmp_path / "figures" / "01-first.pdf").write_bytes(b"pdf")
    (tmp_path / "figures" / "01-second.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Example | Test | TODO | TODO | `figures/01-first.pdf` |
""",
        encoding="utf-8",
    )

    assert main(["check", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "WARNING [figure.duplicate_prefix]" in output
    assert "0 error(s), 1 warning(s)" in output

    assert main(["check", "--root", str(tmp_path), "--strict"]) == 1


def _make_repository_skeleton(root: Path) -> None:
    for relative in ("configs", "runners", "results", "figures", "report", "docs"):
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


def _make_launch_folder(root: Path, name: str) -> Path:
    folder = root / "configs" / name
    folder.mkdir()
    (root / "runners" / f"{name}.py").write_text("", encoding="utf-8")
    return folder


def _allow_tracked_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integrity,
        "require_tracked_file",
        lambda _repository, _path: None,
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
