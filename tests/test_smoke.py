from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml
import pytest

from paper_exp.config import load_config
from paper_exp.run import run_smoke
from paper_exp.parallel import WorkerSlot


def test_smoke_run_creates_a_completed_artifact_envelope(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    run_dir = run_smoke(
        config,
        config_path=config_path,
        command="pytest smoke",
        run_id="test-run",
    )

    assert run_dir.parent.name == "01-smoke-test"
    assert run_dir.name == "001-test-run"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smoke/num_examples"] == 3
    assert metrics["smoke/passed"] is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "smoke_test"
    assert manifest["config_id"] == "01-smoke-test"
    assert manifest["run_id"] == "001-test-run"
    assert manifest["run_sequence"] == 1
    assert manifest["seed"] == 0
    assert manifest["status"] == "completed"
    assert manifest["started_at"] == manifest["timestamp"]
    assert manifest["finished_at"] >= manifest["started_at"]


def test_concurrent_smoke_is_attached_to_the_common_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    def concurrent(slots, work_root, *, require_cuda, allow_shared_gpu):
        root = Path(work_root)
        root.mkdir()
        report = {"passed": True, "attempts": [{}, {}, {}, {}]}
        report_sha256 = _write_concurrent_report(root, report)
        assert [slot.payload for slot in slots] == ["0", "1"]
        assert require_cuda is True
        assert allow_shared_gpu is False
        assert report_sha256 == sha256(
            (root / "concurrent-smoke-report.json").read_bytes()
        ).hexdigest()
        return report

    monkeypatch.setattr(
        "paper_exp.infrastructure_smoke.run_concurrent_infrastructure_smoke",
        concurrent,
    )
    run_dir = run_smoke(
        config,
        config_path=config_path,
        command="pytest concurrent smoke",
        run_id="concurrent",
        worker_slots=(WorkerSlot("gpu-0", "0"), WorkerSlot("gpu-1", "1")),
        require_cuda=True,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["smoke/concurrent_passed"] is True
    assert metrics["smoke/concurrent_attempts"] == 4
    assert manifest["infrastructure_smoke"] == {
        "report": "concurrent/concurrent-smoke-report.json",
        "sha256": sha256(
            (run_dir / "concurrent" / "concurrent-smoke-report.json").read_bytes()
        ).hexdigest(),
        "require_cuda": True,
        "allow_shared_gpu": False,
        "worker_slots": {"gpu-0": "0", "gpu-1": "1"},
    }


@pytest.mark.parametrize(
    ("case", "expected_message"),
    (
        ("malformed-sidecar", "checksum is invalid"),
        ("mismatched-sidecar", "checksum is invalid"),
        ("malformed-report", "report is malformed"),
        ("mismatched-report", "differs from its result"),
        ("failed-report", "did not pass"),
    ),
)
def test_invalid_concurrent_report_fails_the_outer_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_message: str,
) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    def concurrent(slots, work_root, *, require_cuda, allow_shared_gpu):
        del slots, require_cuda, allow_shared_gpu
        root = Path(work_root)
        root.mkdir()
        saved_report = {"passed": case != "failed-report", "attempts": []}
        returned_report = dict(saved_report)
        digest = _write_concurrent_report(root, saved_report)
        if case == "malformed-sidecar":
            (root / "concurrent-smoke-report.sha256").write_bytes(
                (digest + "\n\n").encode("ascii")
            )
        elif case == "mismatched-sidecar":
            (root / "concurrent-smoke-report.sha256").write_text(
                "a" * 64 + "\n", encoding="ascii"
            )
        elif case == "malformed-report":
            payload = b"{not-json}\n"
            (root / "concurrent-smoke-report.json").write_bytes(payload)
            (root / "concurrent-smoke-report.sha256").write_bytes(
                (sha256(payload).hexdigest() + "\n").encode("ascii")
            )
        elif case == "mismatched-report":
            returned_report["attempts"] = [{}]
        return returned_report

    monkeypatch.setattr(
        "paper_exp.infrastructure_smoke.run_concurrent_infrastructure_smoke",
        concurrent,
    )
    with pytest.raises(RuntimeError, match=expected_message):
        run_smoke(
            config,
            config_path=config_path,
            command="pytest invalid concurrent report",
            run_id=case,
            worker_slots=(WorkerSlot("gpu-0", "0"), WorkerSlot("gpu-1", "1")),
        )

    run_dir = tmp_path / "results" / "01-smoke-test" / f"001-{case}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert expected_message in manifest["failure"]["message"]
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()


@pytest.mark.parametrize(
    ("slots", "require_cuda", "allow_shared_gpu"),
    (
        ((WorkerSlot("gpu-0", "0"),), False, False),
        ((), True, False),
        (
            (WorkerSlot("gpu-0", "0"), WorkerSlot("gpu-1", "0")),
            False,
            False,
        ),
        (
            (WorkerSlot("GPU_0", "0"), WorkerSlot("gpu-1", "1")),
            False,
            False,
        ),
        (
            (WorkerSlot("gpu-0", "00"), WorkerSlot("gpu-1", "1")),
            False,
            False,
        ),
        (
            (WorkerSlot(123, "0"), WorkerSlot("gpu-1", "1")),  # type: ignore[arg-type]
            False,
            False,
        ),
    ),
)
def test_invalid_concurrent_smoke_mapping_fails_before_creating_an_attempt(
    tmp_path: Path,
    slots: tuple[WorkerSlot[str], ...],
    require_cuda: bool,
    allow_shared_gpu: bool,
) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path, allow_todos=True)

    with pytest.raises(ValueError):
        run_smoke(
            config,
            config_path=config_path,
            command="pytest invalid smoke",
            worker_slots=slots,
            require_cuda=require_cuda,
            allow_shared_gpu=allow_shared_gpu,
        )

    assert not (tmp_path / "results" / "01-smoke-test").exists()


def _write_temp_config(tmp_path: Path) -> Path:
    config = {
        "experiment_name": "smoke_test",
        "model": {
            "provider": "TODO: provider",
            "name": "TODO: model name",
            "architecture": "TODO: model architecture",
            "initialization": "random",
        },
        "data": {"name": "TODO: dataset", "split": "TODO: split"},
        "evaluation": {"metric": "smoke_pass"},
        "run": {"seed": 0, "max_examples": 3},
        "output": {"dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "01-smoke-test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_concurrent_report(root: Path, report: dict[str, object]) -> str:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256(payload).hexdigest()
    (root / "concurrent-smoke-report.json").write_bytes(payload)
    (root / "concurrent-smoke-report.sha256").write_bytes(
        (digest + "\n").encode("ascii")
    )
    return digest
