from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"
CONFIG_PATH = ROOT / "configs/data/data_bulk_code1_permissive_python_bundle_v1.json"

EXPECTED_REPORT_IDENTITY = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
EXPECTED_ARTIFACT_DIGEST = "sha256:9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac"
EXPECTED_EXECUTION_HEAD = "a045c602bfcead862f7852924fb78a5d78c992d6"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_terminal_evidence_binds_exact_contract_and_execution() -> None:
    evidence = _load(EVIDENCE_PATH)
    config = _load(CONFIG_PATH)
    assert evidence["contract_identity_sha256"] == config["contract_identity_sha256"]
    assert evidence["execution_head_sha"] == EXPECTED_EXECUTION_HEAD
    assert evidence["workflow_run_id"] == 34155446113
    assert evidence["workflow_conclusion"] == "success"
    artifact = evidence["artifact"]
    assert artifact["digest"] == EXPECTED_ARTIFACT_DIGEST
    assert artifact["retained_report_identity_sha256"] == EXPECTED_REPORT_IDENTITY


def test_terminal_evidence_arithmetic_is_exact() -> None:
    evidence = _load(EVIDENCE_PATH)
    families = evidence["families"]
    assert len(families) == 6
    assert len({row["family_id"] for row in families}) == 6
    assert sum(row["eligible_file_count"] for row in families) == 229
    assert sum(row["eligible_utf8_bytes"] for row in families) == 3_880_009
    aggregate = evidence["aggregate"]
    assert aggregate == {
        "source_family_count": 6,
        "eligible_file_count": 229,
        "eligible_utf8_bytes": 3_880_009,
    }


def test_terminal_evidence_proves_two_clean_security_checked_materializations() -> None:
    evidence = _load(EVIDENCE_PATH)
    execution = evidence["execution"]
    assert execution["independent_materializations"] == 2
    assert execution["byte_identical_reports"] is True
    assert execution["exact_source_checkout"] is True
    assert execution["exact_license_blob_validation"] is True
    assert execution["focused_tests_passed"] == 10
    assert execution["security_firewall_self_test_passed"] is True
    assert execution["credential_scan"] == "PASS_NO_HIGH_CONFIDENCE_HITS"


def test_terminal_evidence_cannot_promote_training_or_canonical_capacity() -> None:
    evidence = _load(EVIDENCE_PATH)
    boundary = evidence["claim_boundary"]
    assert boundary["source_intake_evidence_terminal"] is True
    assert boundary["automatic_canonical_capacity_credit"] is False
    assert boundary["post_global_dedup_capacity_claimed"] is False
    assert boundary["authorized_training_exposure"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["final_test_accessed"] is False
    assert boundary["paid_compute_used"] is False
