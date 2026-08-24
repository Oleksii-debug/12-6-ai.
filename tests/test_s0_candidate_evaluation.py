from __future__ import annotations

import json
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


def test_real_collector_consumes_committed_s0_splits_and_proves_checkpoint_resume() -> None:
    evidence = collect_s0_candidate_evidence(ROOT, "d" * 40, train_steps=4)

    assert evidence["candidate"]["parameter_count"] == 10_140
    assert evidence["candidate"]["random_init"] is True
    assert evidence["dataset"]["identity"] == (
        "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"
    )
    assert evidence["dataset"]["heldout_used_for_training"] is False
    assert evidence["dataset"]["train_validation_overlap"] == 0
    assert evidence["dataset"]["distinct_train_batches"] >= 2
    assert evidence["checkpoint"]["save_load_verified"] is True
    assert evidence["checkpoint"]["resume_verified"] is True
    assert evidence["checkpoint"]["serialization_pickle"] is False
    assert evidence["contamination"]["benchmark_overlap_count"] == 0
    assert evidence["contamination"]["heldout_overlap_count"] == 0
    assert evidence["provenance"]["training_consumed_paths"] == [
        "data/s0/packaged/train.jsonl"
    ]
    assert evidence["provenance"]["evaluation_only_paths"] == [
        "data/s0/packaged/validation.jsonl"
    ]


def test_real_collector_rejects_stale_or_short_candidate_identity() -> None:
    with pytest.raises(ValueError, match="candidate_sha"):
        collect_s0_candidate_evidence(ROOT, "deadbeef", train_steps=4)


def test_stale_candidate_ci_head_fails_promotion_even_when_quality_and_audits_pass() -> None:
    evidence = _fixture()
    _bind_promotion(evidence, ci_head_sha="e" * 40)

    gate_report, promotion_report = build_reports(evidence)

    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["promotion_authority_status"] == "FAIL"
    assert gate_report["summary"]["promotion_eligible"] is False
    assert promotion_report["promotion_eligible"] is False
    assert any(
        "candidate_ci.head_sha does not match" in blocker
        for blocker in promotion_report["promotion_blockers"]
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

    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["promotion_authority_status"] == "NOT_TESTED"
    assert promotion_report["promotion_eligible"] is False


def test_split_leakage_fails_heldout_gate_without_erasing_other_evidence() -> None:
    evidence = _fixture()
    evidence["dataset"]["heldout_used_for_training"] = True
    evidence["dataset"]["train_validation_overlap"] = 1

    result = evaluate_s0_integrated(evidence)
    heldout = next(gate for gate in result["gates"] if gate["gate_id"] == "s0.heldout_integrity")

    assert heldout["status"] == "FAIL"
    assert result["summary"]["overall_status"] == "FAIL"
    assert result["summary"]["evaluation_complete"] is False
