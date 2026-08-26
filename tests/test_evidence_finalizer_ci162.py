from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint, verify_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ci162_evidence_finalizer", REPO_ROOT / "tools" / "evidence_finalizer.py"
)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)

SOURCE_SHA = "f" * 40


class NumpyModel:
    def __init__(self) -> None:
        self.weights = np.asarray([0.25, -0.5, 0.75], dtype=np.float64)

    def state_dict(self):
        return {"weights": self.weights.copy()}


def _identity(step: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=SOURCE_SHA,
        model_spec={"kind": "ci162-numpy-fixture", "width": 3},
        parameter_count=3,
        tokenizer_hash="a" * 64,
        tokenizer_vocab_hash="b" * 64,
        dataset_manifest_hash="c" * 64,
        run_manifest_hash="d" * 64,
        training_config={"batch_size": 1, "max_steps": 4},
        seed=162,
        precision="float64-test",
        step=step,
        tokens_seen=step * 8,
        optimizer={"name": "none-test"},
        scheduler=None,
        environment_lock_hash="e" * 64,
    )


def _valid_d05(path: Path, step: int = 1) -> str:
    manifest = save_checkpoint(
        path,
        model=NumpyModel(),
        identity=_identity(step),
        trainer_state={"step": step},
    )
    verify_checkpoint(path)
    return manifest["checkpoint_id"]


def _phase(workspace: Path, name: str) -> Path:
    state = workspace / ".ci162-phase.json"
    finalizer.mark_phase(state, SOURCE_SHA, name)
    return state


def _finalize(workspace: Path, phase: Path, artifact: Path, status: str = "failure"):
    return finalizer.finalize_workspace(
        workspace=workspace,
        artifact_dir=artifact,
        phase_file=phase,
        source_sha=SOURCE_SHA,
        verifier_python=Path(sys.executable),
        repo_root=REPO_ROOT,
        job_status=status,
        retention_days=30,
    )


def _payload(artifact: Path, relative: str) -> Path:
    return artifact / "payload" / relative


def test_failure_injection_before_training_retains_bootstrap_and_focused_test_report(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "evidence"
    diagnostics = workspace / "ci-diagnostics"
    finalizer.write_bootstrap(diagnostics / "bootstrap.json", SOURCE_SHA, "ci162-test")
    (diagnostics / "focused-tests.xml").write_text(
        "<testsuite failures='1'><testcase name='injected'/></testsuite>\n",
        encoding="utf-8",
    )
    finalizer.write_test_status(
        diagnostics / "focused-test-status.json", SOURCE_SHA, 1, "injected-before-training"
    )
    phase = _phase(workspace, "focused_tests")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    assert report["termination_phase"] == "focused_tests"
    assert report["job_status"] == "failure"
    assert report["checkpoint_contract"]["valid_checkpoint_count"] == 0
    assert _payload(artifact, "ci-diagnostics/bootstrap.json").is_file()
    assert _payload(artifact, "ci-diagnostics/focused-tests.xml").is_file()
    assert _payload(artifact, "ci-diagnostics/focused-test-status.json").is_file()
    assert (artifact / "finalization-report.sha256").is_file()


def test_failure_injection_during_training_excludes_unanchored_curve(tmp_path: Path) -> None:
    workspace = tmp_path / "evidence"
    workspace.mkdir()
    (workspace / "train-curve.jsonl").write_text(
        json.dumps({"optimizer_step": 1, "loss": 5.0}) + "\n", encoding="utf-8"
    )
    phase = _phase(workspace, "training")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    assert report["termination_phase"] == "training"
    assert report["training_evidence"] == [
        {
            "path": "train-curve.jsonl",
            "retained": False,
            "reason": "no D05/DCP-verified committed checkpoint in the same run directory",
        }
    ]
    assert not _payload(artifact, "train-curve.jsonl").exists()


def test_failure_injection_during_checkpoint_save_retains_only_verified_d05(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "evidence"
    workspace.mkdir()
    good = workspace / "checkpoint-0001"
    good_id = _valid_d05(good)
    broken = workspace / "checkpoint-0002"
    broken.mkdir()
    (broken / "manifest.json").write_text("{}\n", encoding="utf-8")
    (workspace / "train-curve.jsonl").write_text(
        json.dumps({"optimizer_step": 1, "loss": 4.0}) + "\n", encoding="utf-8"
    )
    phase = _phase(workspace, "checkpoint_save")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    rows = {row["path"]: row for row in report["checkpoint_contract"]["checkpoints"]}
    assert rows["checkpoint-0001"]["valid"] is True
    assert rows["checkpoint-0001"]["retained"] is True
    assert rows["checkpoint-0001"]["identity"] == good_id
    assert rows["checkpoint-0002"]["valid"] is False
    assert rows["checkpoint-0002"]["retained"] is False
    assert "missing exact D05 inventory" in rows["checkpoint-0002"]["reason"]
    assert verify_checkpoint(_payload(artifact, "checkpoint-0001"))["checkpoint_id"] == good_id
    assert not _payload(artifact, "checkpoint-0002").exists()
    assert _payload(artifact, "train-curve.jsonl").is_file()


def test_failure_injection_during_report_generation_keeps_committed_checkpoint(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "evidence"
    workspace.mkdir()
    good_id = _valid_d05(workspace / "checkpoint-0004", step=4)
    (workspace / "report.preverify.json").write_text(
        json.dumps({"status": "PREVERIFY_ONLY", "checkpoint_id": good_id}) + "\n",
        encoding="utf-8",
    )
    phase = _phase(workspace, "report_generation")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    assert report["termination_phase"] == "report_generation"
    assert report["interpretation"] == "FAILURE_DIAGNOSTICS_ONLY_NO_COMPLETION_CLAIM"
    assert report["checkpoint_contract"]["valid_checkpoint_count"] == 1
    assert _payload(artifact, "checkpoint-0004").is_dir()
    assert _payload(artifact, "report.preverify.json").is_file()
    assert not _payload(artifact, "report.json").exists()


def test_dcp_candidate_requires_existing_committed_dcp_verifier(tmp_path: Path) -> None:
    workspace = tmp_path / "evidence"
    dcp = workspace / "checkpoint-dcp"
    dcp.mkdir(parents=True)
    (dcp / "scale-manifest.json").write_text("{}\n", encoding="utf-8")
    (dcp / "scale-manifest.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    (dcp / "COMMITTED").write_text("0" * 64 + "\n", encoding="ascii")
    (dcp / "payload.bin").write_bytes(b"not-a-real-dcp-checkpoint")
    phase = _phase(workspace, "checkpoint_save")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    row = report["checkpoint_contract"]["checkpoints"][0]
    assert row["kind"] == "dcp"
    assert row["valid"] is False
    assert row["retained"] is False
    assert "dcp verifier rejected checkpoint" in row["reason"]
    assert not _payload(artifact, "checkpoint-dcp").exists()


def test_privacy_gate_rejects_raw_corpus_paths_and_secret_like_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "evidence"
    (workspace / "corpus-a").mkdir(parents=True)
    (workspace / "corpus-a" / "report.json").write_text(
        json.dumps({"raw": "private source text"}) + "\n", encoding="utf-8"
    )
    workspace.mkdir(exist_ok=True)
    (workspace / "report.json").write_text(
        json.dumps({"token": "ghp_abcdefghijklmnopqrstuvwxyz123456"}) + "\n",
        encoding="utf-8",
    )
    phase = _phase(workspace, "report_generation")

    artifact = tmp_path / "artifact"
    report = _finalize(workspace, phase, artifact)

    assert not _payload(artifact, "corpus-a/report.json").exists()
    assert not _payload(artifact, "report.json").exists()
    rejected = "\n".join(item["reason"] for item in report["metadata"]["rejected"])
    assert "raw/private corpus path is forbidden" in rejected
    assert "secret-like content rejected" in rejected
