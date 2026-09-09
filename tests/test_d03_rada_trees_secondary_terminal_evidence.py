from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/d03-rada-trees/secondary-archive-terminal-role-v1.json"


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_terminal_role_evidence_binds_exact_source_execution_and_runtime() -> None:
    evidence = load_evidence()
    assert evidence["source"]["dataset"] == "uacorpus/Rada_Trees"
    assert evidence["source"]["dataset_revision"] == "1b994a5804dcda122721e8d33a03fd172cf8d867"
    assert evidence["source"]["archive_path"] == "rada_xtag_texts.7z"
    assert evidence["source"]["content_sha256"] == "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
    assert evidence["execution"]["head_sha"] == "66ad574e1fe8183efd88a592323e5fccce00bde0"
    assert evidence["execution"]["workflow_run"] == 34158830442
    assert evidence["execution"]["artifact_id"] == 10032220704
    assert evidence["runtime"]["python_version"] == "3.11.16"
    assert evidence["runtime"]["extractor_command"] in {"7zz", "7z"}


def test_terminal_role_evidence_preserves_zero_credit_and_full_stream_requirement() -> None:
    evidence = load_evidence()
    boundary = evidence["claim_boundary"]
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["final_test_payload_accessed"] is False
    assert boundary["paid_compute_used"] is False
    assert evidence["classification"]["plain_text_candidate_members"] is None
    assert evidence["classification"]["plain_text_candidate_bytes_after_exact_duplicate_collapse"] is None
    assert evidence["decision"]["training_admission_claimed"] is False
    assert evidence["decision"]["rerun_with_pinned_sha_required_before_any_capacity_credit"] is True
    assert "full_stream_scan_all_plain_text_suffix_members_without_full_archive_extraction" in evidence["next_required"]


def test_terminal_role_evidence_identity_is_canonical() -> None:
    evidence = load_evidence()
    expected = evidence.pop("evidence_identity_sha256")
    encoded = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
