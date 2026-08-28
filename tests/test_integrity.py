from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper_exp.integrity as integrity
from paper_exp.cli import main
from paper_exp.integrity import (
    IntegrityFinding,
    check_repository,
    classify_run_directory,
)


VALID_CONFIG_TEMPLATE = """\
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
  dir: {output_dir}
"""

SMOKE_CONFIG = """\
experiment_name: harness_smoke
model:
  provider: "TODO: provider"
  name: "TODO: model"
  architecture: "TODO: architecture"
  initialization: random
data:
  name: "TODO: dataset"
  split: "TODO: split"
evaluation:
  metric: smoke_passed
run:
  seed: 0
  max_examples: 3
output:
  dir: experiments/00-infrastructure-smoke/raw
"""


@pytest.fixture(autouse=True)
def _allow_tracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integrity,
        "require_tracked_file",
        lambda _repository, _path: None,
    )


def test_configs_are_validated_tracked_and_numbered(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-grid")
    run_dir = scaffold / "run"
    _write_config(run_dir / "001-valid.yaml", scaffold)
    _write_config(run_dir / "001-duplicate.yaml", scaffold)
    _write_config(run_dir / "bad-name.yaml", scaffold)
    (run_dir / "003-invalid.yaml").write_text(
        "experiment_name: missing_fields\n", encoding="utf-8"
    )

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "config.duplicate_prefix", "experiments")
    assert _has_finding(
        findings,
        "config.filename_invalid",
        "experiments/01-a1-grid/run/bad-name.yaml",
    )
    assert _has_finding(
        findings,
        "config.invalid",
        "experiments/01-a1-grid/run/003-invalid.yaml",
    )
    assert _has_finding(findings, "config.numbering_gap", "experiments")


def test_scaffold_shape_runner_tracking_and_legacy_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    invalid = tmp_path / "experiments" / "a1-grid"
    invalid.mkdir()
    scaffold = _make_scaffold(tmp_path, "01-a1-grid")
    (scaffold / "figs" / ".gitkeep").unlink()
    (scaffold / "figs").rmdir()
    (scaffold / "unexpected").mkdir()
    (scaffold / "misplaced.txt").write_text("wrong\n", encoding="utf-8")
    _write_config(scaffold / "run" / "001-case.yaml", scaffold)
    (tmp_path / "results").mkdir()
    (tmp_path / "run-logs").mkdir()

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "scaffold.name_invalid", "experiments/a1-grid")
    assert _has_finding(
        findings,
        "scaffold.directory_missing",
        "experiments/01-a1-grid/figs",
    )
    assert _has_finding(
        findings,
        "scaffold.directory_invalid",
        "experiments/01-a1-grid/unexpected",
    )
    assert _has_finding(
        findings,
        "scaffold.entry_invalid",
        "experiments/01-a1-grid/misplaced.txt",
    )
    assert _has_finding(
        findings,
        "config.runner_missing",
        "experiments/01-a1-grid/run/runner.py",
    )
    assert _has_finding(findings, "layout.legacy_directory", "results")
    assert _has_finding(findings, "layout.legacy_directory", "run-logs")

    runner = scaffold / "run" / "runner.py"
    runner.write_text("", encoding="utf-8")

    def reject_runner(_repository: Path, path: Path) -> None:
        if path.name == "runner.py":
            raise integrity.LaunchError("not tracked")

    monkeypatch.setattr(integrity, "require_tracked_file", reject_runner)
    findings = check_repository(tmp_path)
    assert _has_finding(
        findings,
        "config.runner_untracked",
        "experiments/01-a1-grid/run/runner.py",
    )


def test_configs_and_directory_keepers_must_be_tracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-tracking-tests")
    config = scaffold / "run" / "001-case.yaml"
    _write_config(config, scaffold)

    def reject_selected(_repository: Path, path: Path) -> None:
        if path == config.resolve() or path == (scaffold / "raw" / ".gitkeep").resolve():
            raise integrity.LaunchError("not tracked")

    monkeypatch.setattr(integrity, "require_tracked_file", reject_selected)
    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "config.untracked",
        "experiments/01-tracking-tests/run/001-case.yaml",
    )
    assert _has_finding(
        findings,
        "scaffold.keeper_untracked",
        "experiments/01-tracking-tests/raw/.gitkeep",
    )

    (scaffold / "figs" / ".gitkeep").unlink()
    findings = check_repository(tmp_path)
    assert _has_finding(
        findings,
        "scaffold.keeper_missing",
        "experiments/01-tracking-tests/figs/.gitkeep",
    )


def test_scientific_config_scope_output_ownership_and_global_order(
    tmp_path: Path,
) -> None:
    _make_repository_skeleton(tmp_path)
    first = _make_scientific_scaffold(tmp_path, "01-a1-grid")
    second = _make_scientific_scaffold(tmp_path, "02-b1-screen")
    _write_config(first / "run" / "002-second.yaml", first)
    _write_config(first / "run" / "000-zero.yaml", first)
    _write_config(first / "run" / "03-short.yaml", first)
    wrong_output = _valid_config(second).replace(
        "experiments/02-b1-screen/raw",
        "experiments/01-a1-grid/raw",
    )
    (second / "run" / "001-first.yaml").write_text(
        wrong_output, encoding="utf-8"
    )
    nested = second / "run" / "extra"
    nested.mkdir()
    _write_config(nested / "003-too-deep.yaml", second)
    (tmp_path / "experiments" / "003-root.yaml").write_text(
        _valid_config(second), encoding="utf-8"
    )

    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "config.filename_invalid",
        "experiments/01-a1-grid/run/03-short.yaml",
    )
    assert _has_finding(
        findings,
        "config.filename_invalid",
        "experiments/01-a1-grid/run/000-zero.yaml",
    )
    assert _has_finding(findings, "config.order_invalid", "experiments")
    assert _has_finding(
        findings,
        "config.location_invalid",
        "experiments/02-b1-screen/run/extra/003-too-deep.yaml",
    )
    assert _has_finding(
        findings,
        "config.invalid",
        "experiments/02-b1-screen/run/001-first.yaml",
    )
    assert _has_finding(
        findings,
        "experiment.entry_invalid",
        "experiments/003-root.yaml",
    )


def test_scaffold_and_config_prefixes_are_global_and_sequential(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    first = _make_scientific_scaffold(tmp_path, "01-a1-grid")
    duplicate_scaffold = _make_scientific_scaffold(tmp_path, "01-b1-screen")
    third = _make_scientific_scaffold(tmp_path, "03-c1-confirm")
    _write_config(first / "run" / "001-first.yaml", first)
    _write_config(duplicate_scaffold / "run" / "001-second.yaml", duplicate_scaffold)
    _write_config(third / "run" / "003-third.yaml", third)

    findings = check_repository(tmp_path)

    assert _has_finding(findings, "scaffold.duplicate_prefix", "experiments")
    assert _has_finding(findings, "scaffold.numbering_gap", "experiments")
    assert _has_finding(findings, "config.duplicate_prefix", "experiments")
    assert _has_finding(findings, "config.numbering_gap", "experiments")


def test_run_directories_are_classified_from_scaffold_raw_artifacts(
    tmp_path: Path,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-run-tests")
    _write_config(scaffold / "run" / "001-test.yaml", scaffold)
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "Audit `experiments/01-run-tests/raw/001-test/`.\n",
        encoding="utf-8",
    )
    result_group = scaffold / "raw" / "001-test"

    active = result_group / "001-active"
    active.mkdir(parents=True)
    (active / "events.jsonl").write_text('{"event": "train"}\n', encoding="utf-8")

    partial = result_group / "002-partial"
    partial.mkdir()
    (partial / "config.yaml").write_text(_valid_config(scaffold), encoding="utf-8")

    complete = result_group / "003-complete"
    _write_core_run(complete, scaffold, status=None)

    running = result_group / "004-running"
    _write_lifecycle_run(running, scaffold, status="running", core=False)

    failed = result_group / "005-failed"
    _write_lifecycle_run(failed, scaffold, status="failed", core=False)

    inconsistent = result_group / "006-inconsistent"
    _write_lifecycle_run(inconsistent, scaffold, status="completed", core=False)

    mismatched = result_group / "007-mismatched"
    _write_core_run(mismatched, scaffold, status=None, config_id="wrong")

    completed = result_group / "008-completed"
    _write_lifecycle_run(completed, scaffold, status="completed", core=True)

    wrong_tranche = result_group / "009-wrong-tranche"
    _write_lifecycle_run(wrong_tranche, scaffold, status="completed", core=True)
    manifest = json.loads(
        (wrong_tranche / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["tranche_id"] = "02-other-tranche"
    (wrong_tranche / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    expected = {
        active: "inconsistent",
        partial: "inconsistent",
        complete: "complete",
        running: "running",
        failed: "failed",
        inconsistent: "inconsistent",
        mismatched: "inconsistent",
        completed: "complete",
        wrong_tranche: "inconsistent",
    }
    for run_dir, status in expected.items():
        assert classify_run_directory(run_dir) == status

    findings = check_repository(tmp_path)
    prefix = "experiments/01-run-tests/raw/001-test"
    assert _finding(findings, "run.inconsistent", f"{prefix}/001-active").severity == "error"
    assert _finding(findings, "run.inconsistent", f"{prefix}/002-partial").severity == "error"
    assert _finding(findings, "run.complete", f"{prefix}/003-complete").severity == "info"
    assert _finding(findings, "run.running", f"{prefix}/004-running").severity == "warning"
    assert _finding(findings, "run.failed", f"{prefix}/005-failed").severity == "warning"
    assert _finding(findings, "run.inconsistent", f"{prefix}/006-inconsistent").severity == "error"
    assert _finding(findings, "run.inconsistent", f"{prefix}/007-mismatched").severity == "error"
    assert _finding(findings, "run.complete", f"{prefix}/008-completed").severity == "info"
    assert _finding(
        findings, "run.inconsistent", f"{prefix}/009-wrong-tranche"
    ).severity == "error"


def test_reviewed_a2_scope_preserves_exact_indexed_completed_a1_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-history")
    config_path = scaffold / "run" / "001-a1-case.yaml"
    _write_config(config_path, scaffold)
    run_dir = scaffold / "raw" / "001-a1-case" / "001-completed"
    _write_lifecycle_run(run_dir, scaffold, status="completed", core=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["mode"] = "pretrain"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "`experiments/01-a1-history/raw/001-a1-case/001-completed/`\n",
        encoding="utf-8",
    )
    review = _mock_reviewed_a2_design(monkeypatch, tmp_path, config_path)
    monkeypatch.setattr(
        integrity, "classify_run_directory", lambda _run_dir: "complete"
    )

    findings = integrity._check_design(tmp_path)

    assert not any(
        finding.code == "design.config_group_unreviewed" for finding in findings
    )
    assert review.reviewed_groups == ("A2-relu-control", "A2-l1-screen")


def test_reviewed_a3_scope_preserves_exact_indexed_completed_a1_a2_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    historical: list[tuple[Path, str, str]] = []
    log_entries: list[str] = []
    for scaffold_name, config_id, group_id, fingerprint in (
        ("01-a1-history", "001-a1-case", "A1-lr-screen", "a" * 64),
        ("02-a2-relu-history", "012-a2-relu", "A2-relu-control", "b" * 64),
        ("02-a2-l1-history", "013-a2-l1", "A2-l1-screen", "c" * 64),
    ):
        scaffold = _make_scientific_scaffold(tmp_path, scaffold_name)
        config_path = scaffold / "run" / f"{config_id}.yaml"
        _write_config(config_path, scaffold)
        run_dir = scaffold / "raw" / config_id / "001-completed"
        _write_lifecycle_run(run_dir, scaffold, status="completed", core=True)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["mode"] = "pretrain"
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        historical.append((config_path, group_id, fingerprint))
        log_entries.append(f"`experiments/{scaffold_name}/raw/{config_id}/001-completed/`")

    (tmp_path / "docs" / "experiment_log.md").write_text(
        "\n".join(log_entries) + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / integrity.PLAN_PATH
    catalog = tmp_path / integrity.CATALOG_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("Plan status: reviewed\n", encoding="utf-8")
    catalog.write_text("case_groups: []\n", encoding="utf-8")
    review = SimpleNamespace(
        status="reviewed",
        reviewed_groups=("A3-ol1-screen",),
    )
    monkeypatch.setattr(
        integrity,
        "validate_catalog",
        lambda _repository: SimpleNamespace(
            groups={
                "A1-lr-screen": {},
                "A2-relu-control": {},
                "A2-l1-screen": {},
                "A3-ol1-screen": {},
            }
        ),
    )
    monkeypatch.setattr(
        integrity,
        "validate_reviewed_design",
        lambda _repository, require_reviewed=False: review,
    )
    monkeypatch.setattr(
        integrity,
        "tracked_training_identities",
        lambda _repository: historical,
    )
    monkeypatch.setattr(
        integrity,
        "classify_run_directory",
        lambda _run_dir: "complete",
    )

    findings = integrity._check_design(tmp_path)

    assert not any(
        finding.code == "design.config_group_unreviewed" for finding in findings
    )


@pytest.mark.parametrize("evidence_case", ["unindexed", "nonterminal", "changed"])
def test_reviewed_a2_scope_rejects_invalid_a1_historical_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_case: str,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-history")
    config_path = scaffold / "run" / "001-a1-case.yaml"
    _write_config(config_path, scaffold)
    run_dir = scaffold / "raw" / "001-a1-case" / "001-evidence"
    status = "failed" if evidence_case == "nonterminal" else "completed"
    _write_lifecycle_run(run_dir, scaffold, status=status, core=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["mode"] = "pretrain"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    if evidence_case == "changed":
        (run_dir / "config.yaml").write_text(
            _valid_config(scaffold).replace("max_examples: 1", "max_examples: 2"),
            encoding="utf-8",
        )
    log_text = (
        "No accepted run is indexed.\n"
        if evidence_case == "unindexed"
        else "`experiments/01-a1-history/raw/001-a1-case/001-evidence/`\n"
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        log_text, encoding="utf-8"
    )
    _mock_reviewed_a2_design(monkeypatch, tmp_path, config_path)
    monkeypatch.setattr(
        integrity, "classify_run_directory", lambda _run_dir: "complete"
    )

    findings = integrity._check_design(tmp_path)

    finding = _finding(
        findings,
        "design.config_group_unreviewed",
        "experiments/01-a1-history/run/001-a1-case.yaml",
    )
    assert "outside active reviewed scope" in finding.message
    assert "does not authorize materialization or launch" in finding.message


def test_placeholder_preserves_only_exact_indexed_completed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-history")
    config_path = scaffold / "run" / "001-case.yaml"
    _write_config(config_path, scaffold)
    run_dir = scaffold / "raw" / "001-case" / "001-completed"
    _write_lifecycle_run(run_dir, scaffold, status="completed", core=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["mode"] = "pretrain"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    plan = tmp_path / integrity.PLAN_PATH
    catalog = tmp_path / integrity.CATALOG_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("Plan status: placeholder\n", encoding="utf-8")
    catalog.write_text("case_groups: []\n", encoding="utf-8")
    monkeypatch.setattr(
        integrity,
        "validate_catalog",
        lambda _repository: SimpleNamespace(groups={"history": {}}),
    )
    monkeypatch.setattr(
        integrity,
        "validate_reviewed_design",
        lambda _repository, require_reviewed=False: SimpleNamespace(
            status="placeholder", reviewed_groups=()
        ),
    )
    monkeypatch.setattr(
        integrity,
        "tracked_training_identities",
        lambda _repository: [(config_path, "history", "a" * 64)],
    )
    monkeypatch.setattr(
        integrity, "classify_run_directory", lambda _run_dir: "complete"
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "[completed](../experiments/01-a1-history/raw/001-case/001-completed/)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["check", "--root", ".", "--strict"]) == 0
    findings = integrity._check_design(Path("."))
    assert not any(
        finding.code == "design.config_while_placeholder" for finding in findings
    )

    (tmp_path / "docs" / "experiment_log.md").write_text(
        "No accepted run is indexed.\n", encoding="utf-8"
    )
    findings = integrity._check_design(Path("."))
    finding = _finding(
        findings,
        "design.config_while_placeholder",
        "experiments/01-a1-history/run/001-case.yaml",
    )
    assert "exact coherent completed run" in finding.message


def test_placeholder_rejects_indexed_nonterminal_or_changed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-history")
    config_path = scaffold / "run" / "001-case.yaml"
    _write_config(config_path, scaffold)
    run_dir = scaffold / "raw" / "001-case" / "001-run"
    _write_lifecycle_run(run_dir, scaffold, status="failed", core=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["mode"] = "pretrain"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        integrity, "classify_run_directory", lambda _run_dir: "complete"
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "`experiments/01-a1-history/raw/001-case/001-run/`\n",
        encoding="utf-8",
    )

    assert not integrity._config_has_indexed_completed_evidence(
        tmp_path, config_path
    )

    completed_dir = scaffold / "raw" / "001-case" / "002-completed"
    _write_lifecycle_run(completed_dir, scaffold, status="completed", core=True)
    manifest = json.loads(
        (completed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["mode"] = "pretrain"
    (completed_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (completed_dir / "config.yaml").write_text(
        _valid_config(scaffold).replace("max_examples: 1", "max_examples: 2"),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "`experiments/01-a1-history/raw/001-case/002-completed/`\n",
        encoding="utf-8",
    )
    assert not integrity._config_has_indexed_completed_evidence(
        tmp_path, config_path
    )


def test_placeholder_rejects_completed_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-a1-history")
    config_path = scaffold / "run" / "001-case.yaml"
    _write_config(config_path, scaffold)
    run_dir = scaffold / "raw" / "001-case" / "001-calibration"
    _write_lifecycle_run(run_dir, scaffold, status="completed", core=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["mode"] = "calibrate"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "`experiments/01-a1-history/raw/001-case/001-calibration/`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        integrity, "classify_run_directory", lambda _run_dir: "complete"
    )
    assert not integrity._config_has_indexed_completed_evidence(
        tmp_path, config_path
    )


def test_literal_scaffold_references_and_paper_outputs_are_checked(
    tmp_path: Path,
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-figure-tests")
    _write_config(scaffold / "run" / "001-case.yaml", scaffold)
    (scaffold / "figs" / "01-first.pdf").write_bytes(b"pdf")
    (scaffold / "figs" / "01-second.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Present | Test | TODO | TODO | `experiments/01-figure-tests/figs/01-first.pdf` |
| Missing | Test | `experiments/99-missing/run/999-missing.yaml` | `experiments/99-missing/raw/999-test/001-run/` | `experiments/01-figure-tests/figs/02-missing.pdf` |
| Exploratory | Test | TODO | See `experiments/01-figure-tests/raw/*-sweep/` | TODO |
""",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "experiment_log.md").write_text(
        "Literal missing report: `report/01-missing/01-missing.pdf`.\n"
        "Ignored glob: `experiments/01-figure-tests/raw/001-*/001-*`.\n",
        encoding="utf-8",
    )

    findings = check_repository(tmp_path)

    assert _has_finding(
        findings,
        "figure.duplicate_prefix",
        "experiments/01-figure-tests/figs",
    )
    assert _finding(
        findings,
        "reference.missing",
        "experiments/99-missing/run/999-missing.yaml",
    ).severity == "error"
    assert _finding(
        findings,
        "reference.missing",
        "experiments/99-missing/raw/999-test/001-run/",
    ).severity == "warning"
    assert _finding(
        findings,
        "reference.missing",
        "report/01-missing/01-missing.pdf",
    ).severity == "error"
    assert _finding(
        findings,
        "paper_map.output_missing",
        "experiments/01-figure-tests/figs/02-missing.pdf",
    ).severity == "warning"
    assert not any("*" in finding.path for finding in findings)


def test_completed_checkpoint_uses_shared_source_path_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "arbitrary" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "checkpoint:\n  save_final: true\n  save_optimizer: false\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "events.jsonl").write_text('{"event": "train"}\n', encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "pretrain",
                "git_dirty": False,
                "checkpoint": {"saved": True, "path": "recorded/checkpoint"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "resolved-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    calls: list[tuple[object, Path]] = []

    def resolve(value: object, *, source_run: Path) -> Path:
        calls.append((value, source_run))
        return checkpoint

    monkeypatch.setattr(integrity, "resolve_source_path", resolve)
    monkeypatch.setattr(integrity, "validate_training_config", lambda _config: None)

    assert integrity._completed_artifacts_are_coherent(run_dir) is True
    assert calls == [("recorded/checkpoint", run_dir)]


def test_completed_run_rejects_corrupt_core_artifact(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-corrupt-tests")
    _write_config(scaffold / "run" / "001-test.yaml", scaffold)
    run_dir = scaffold / "raw" / "001-test" / "001-corrupt"
    _write_lifecycle_run(run_dir, scaffold, status="completed", core=True)
    (run_dir / "metrics.json").write_text("not-json\n", encoding="utf-8")

    assert classify_run_directory(run_dir) == "inconsistent"


def test_check_is_read_only(tmp_path: Path) -> None:
    _make_repository_skeleton(tmp_path)
    before = _tree_snapshot(tmp_path)

    check_repository(tmp_path)

    assert _tree_snapshot(tmp_path) == before


def test_check_command_reports_warnings_and_supports_strict_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_repository_skeleton(tmp_path)
    scaffold = _make_scientific_scaffold(tmp_path, "01-figure-tests")
    _write_config(scaffold / "run" / "001-case.yaml", scaffold)
    (scaffold / "figs" / "01-first.pdf").write_bytes(b"pdf")
    (scaffold / "figs" / "01-second.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "paper_map.md").write_text(
        """\
# Paper Map

| Paper item | Claim / purpose | Config | Result | Figure |
| ---------- | --------------- | ------ | ------ | ------ |
| Example | Test | TODO | TODO | `experiments/01-figure-tests/figs/01-first.pdf` |
""",
        encoding="utf-8",
    )

    assert main(["check", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "WARNING [figure.duplicate_prefix]" in output
    assert "0 error(s), 1 warning(s)" in output

    assert main(["check", "--root", str(tmp_path), "--strict"]) == 1


def _make_repository_skeleton(root: Path) -> None:
    for relative in ("experiments", "report", "docs"):
        (root / relative).mkdir(parents=True)
    smoke = _make_scaffold(root, "00-infrastructure-smoke")
    (smoke / "run" / "00-smoke.yaml").write_text(SMOKE_CONFIG, encoding="utf-8")
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


def _make_scaffold(root: Path, name: str) -> Path:
    scaffold = root / "experiments" / name
    for member in ("run", "raw", "figs"):
        (scaffold / member).mkdir(parents=True, exist_ok=True)
    for member in ("raw", "figs"):
        (scaffold / member / ".gitkeep").write_text("", encoding="utf-8")
    return scaffold


def _make_scientific_scaffold(root: Path, name: str) -> Path:
    scaffold = _make_scaffold(root, name)
    (scaffold / "run" / "runner.py").write_text("", encoding="utf-8")
    return scaffold


def _valid_config(scaffold: Path) -> str:
    return VALID_CONFIG_TEMPLATE.format(
        output_dir=f"experiments/{scaffold.name}/raw"
    )


def _write_config(path: Path, scaffold: Path) -> None:
    path.write_text(_valid_config(scaffold), encoding="utf-8")


def _manifest(
    run_dir: Path,
    *,
    status: str | None,
    config_id: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "config_id": config_id or run_dir.parent.name,
        "run_id": run_dir.name,
        "mode": "smoke",
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    if status is not None:
        value.update(
            {
                "status": status,
                "started_at": "2026-01-01T00:00:00Z",
            }
        )
        if status != "running":
            value["finished_at"] = "2026-01-01T00:01:00Z"
        if status == "failed":
            value["failure"] = {"type": "RuntimeError", "message": "test"}
    return value


def _write_core_run(
    run_dir: Path,
    scaffold: Path,
    *,
    status: str | None,
    config_id: str | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(_valid_config(scaffold), encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(run_dir, status=status, config_id=config_id)) + "\n",
        encoding="utf-8",
    )


def _write_lifecycle_run(
    run_dir: Path,
    scaffold: Path,
    *,
    status: str,
    core: bool,
) -> None:
    if core:
        _write_core_run(run_dir, scaffold, status=status)
        return
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(_valid_config(scaffold), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(run_dir, status=status)) + "\n",
        encoding="utf-8",
    )


def _mock_reviewed_a2_design(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
    historical_config: Path,
) -> SimpleNamespace:
    plan = repository / integrity.PLAN_PATH
    catalog = repository / integrity.CATALOG_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("Plan status: reviewed\n", encoding="utf-8")
    catalog.write_text("case_groups: []\n", encoding="utf-8")
    catalog_groups = {
        "A1-lr-screen": {},
        "A2-relu-control": {},
        "A2-l1-screen": {},
    }
    review = SimpleNamespace(
        status="reviewed",
        reviewed_groups=("A2-relu-control", "A2-l1-screen"),
    )
    monkeypatch.setattr(
        integrity,
        "validate_catalog",
        lambda _repository: SimpleNamespace(groups=catalog_groups),
    )
    monkeypatch.setattr(
        integrity,
        "validate_reviewed_design",
        lambda _repository, require_reviewed=False: review,
    )
    monkeypatch.setattr(
        integrity,
        "tracked_training_identities",
        lambda _repository: [(historical_config, "A1-lr-screen", "a" * 64)],
    )
    return review


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
