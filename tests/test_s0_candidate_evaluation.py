from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from twelve_six.s0_candidate_evaluation import (
    build_reports,
    collect_s0_candidate_evidence,
)
from twelve_six.stage_gates import evaluate_s0_integrated

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/s0_complete_evidence.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _bind_promotion(evidence: dict, *, ci_head_sha: str | None = None) -> None:
    candidate_sha = evidence["candidate"]["sha"]
    evidence["candidate"]["integrated"] = True
    evidence["promotion"] = {
        "candidate_manifest_validated": True,
        "candidate_manifest_sha256": "c" * 64,
        "candidate_ci": {
            "success": True,
            "run_id": 123456789,
            "head_sha": candidate_sha if ci_head_sha is None else ci_head_sha,
        },
        "audit_a": {
            "verdict": "PASS",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-13#exact-candidate",
        },
        "audit_b": {
            "verdict": "PASS_WITH_NOTES",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-14#exact-candidate",
        },
    }


def test_strict_collector_consumes_real_s0_and_strict_d05_checkpoint_contract() -> None:
    evidence = collect_s0_candidate_evidence(ROOT, _head(), train_steps=4)

    assert evidence["candidate"]["parameter_count"] == 10_140
    assert evidence["candidate"]["random_init"] is True
    assert evidence["dataset"]["heldout_used_for_training"] is False
    assert evidence["dataset"]["train_validation_overlap"] == 0
    assert evidence["dataset"]["distinct_train_batches"] >= 2
    assert evidence["checkpoint"]["save_load_verified"] is True
    assert evidence["checkpoint"]["resume_verified"] is True
    assert evidence["checkpoint"]["serialization_pickle"] is False
    assert evidence["checkpoint"]["packing_sha256"] == (
        evidence["provenance"]["packing_sha256"]
    )
    assert evidence["checkpoint"]["environment_lock_sha256"] == (
        evidence["provenance"]["environment_lock_sha256"]
    )
    assert evidence["generation_probes"][0]["direct_vs_reloaded_parity"] is True
    assert evidence["contamination"]["benchmark_overlap_count"] == 0
    assert evidence["contamination"]["heldout_overlap_count"] == 0

    gate_report, promotion_report = build_reports(evidence)
    assert gate_report["summary"]["counts"] == {
        "FAIL": 0,
        "NOT_TESTED": 0,
        "PASS": 15,
    }
    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["promotion_authority_status"] == "NOT_TESTED"
    assert promotion_report["promotion_eligible"] is False


def test_stale_candidate_sha_is_rejected_against_checkout_head() -> None:
    stale = "0" * 40 if _head() != "0" * 40 else "1" * 40
    with pytest.raises(ValueError, match="stale"):
        collect_s0_candidate_evidence(ROOT, stale, train_steps=4)


def test_stale_candidate_ci_head_fails_promotion() -> None:
    evidence = _fixture()
    _bind_promotion(evidence, ci_head_sha="e" * 40)
    gate_report, promotion_report = build_reports(evidence)
    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["promotion_authority_status"] == "FAIL"
    assert promotion_report["promotion_eligible"] is False
    assert any(
        "candidate_ci.head_sha does not match" in blocker
        for blocker in promotion_report["promotion_blockers"]
    )


def test_stale_audit_candidate_sha_fails_promotion() -> None:
    evidence = _fixture()
    _bind_promotion(evidence)
    evidence["promotion"]["audit_a"]["candidate_sha"] = "e" * 40
    gate_report, promotion_report = build_reports(evidence)
    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["promotion_authority_status"] == "FAIL"
    assert promotion_report["promotion_eligible"] is False
    assert any(
        "audit_a.candidate_sha does not match" in item
        for item in promotion_report["promotion_blockers"]
    )


def test_missing_candidate_ci_is_not_promotion_authority() -> None:
    evidence = _fixture()
    candidate_sha = evidence["candidate"]["sha"]
    evidence["candidate"]["integrated"] = True
    evidence["promotion"] = {
        "candidate_manifest_validated": True,
        "candidate_manifest_sha256": "c" * 64,
        "audit_a": {
            "verdict": "PASS",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-13#candidate",
        },
        "audit_b": {
            "verdict": "PASS",
            "candidate_sha": candidate_sha,
            "evidence_ref": "issue-14#candidate",
        },
    }
    gate_report, promotion_report = build_reports(evidence)
    assert gate_report["summary"]["promotion_authority_status"] == "NOT_TESTED"
    assert promotion_report["promotion_eligible"] is False


def test_split_leakage_fails_heldout_gate() -> None:
    evidence = _fixture()
    evidence["dataset"]["heldout_used_for_training"] = True
    evidence["dataset"]["train_validation_overlap"] = 1
    result = evaluate_s0_integrated(evidence)
    heldout = next(
        gate
        for gate in result["gates"]
        if gate["gate_id"] == "s0.heldout_integrity"
    )
    assert heldout["status"] == "FAIL"
    assert result["summary"]["overall_status"] == "FAIL"


def test_tokenizer_model_mismatch_fails_closed() -> None:
    evidence = _fixture()
    evidence["candidate"]["model_vocab_size"] = (
        evidence["tokenizer"]["vocab_size"] + 1
    )
    result = evaluate_s0_integrated(evidence)
    vocab = next(
        gate
        for gate in result["gates"]
        if gate["gate_id"] == "s0.tokenizer_model_vocab"
    )
    assert vocab["status"] == "FAIL"
    assert result["summary"]["evaluation_complete"] is False
