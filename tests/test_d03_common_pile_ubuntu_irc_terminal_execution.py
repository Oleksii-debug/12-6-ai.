from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
EVIDENCE_PATH = REPO_ROOT / "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
CONFIG_PATH = REPO_ROOT / "configs/data/d03_common_pile_ubuntu_irc_bounded_v1.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_execution_receipt_binds_exact_source_and_run() -> None:
    evidence = load_json(EVIDENCE_PATH)
    config = load_json(CONFIG_PATH)
    source = config["source"]
    materialization = evidence["materialization"]

    assert evidence["schema_version"] == (
        "12-6.d03-common-pile-ubuntu-irc-real-execution.v1"
    )
    assert evidence["worker_issue"] == 1096
    assert evidence["incumbent_pr"] == 913
    assert evidence["execution_profile"] == "LOCAL_FREE_GITHUB_HOSTED_UBUNTU"
    assert evidence["workflow_run_id"] == 34549598850
    assert evidence["workflow_job_id"] == 103109520811
    assert evidence["workflow_conclusion"] == "success"
    assert evidence["focused_adversarial_tests"] == {
        "conclusion": "success",
        "passed": 22,
    }
    assert materialization["source_revision"] == source["revision"]
    assert materialization["source_file"] == source["file"]
    assert materialization["source_sha256"] == source["sha256"]
    assert materialization["observed_source_bytes"] <= source["max_compressed_bytes"]


def test_two_real_materializations_are_exactly_reproducible() -> None:
    evidence = load_json(EVIDENCE_PATH)
    config = load_json(CONFIG_PATH)
    materialization = evidence["materialization"]
    selection = config["selection"]

    assert materialization["status"] == (
        "TWO_INDEPENDENT_MATERIALIZATIONS_BYTE_IDENTICAL"
    )
    assert materialization["independent_build_count"] == 2
    assert materialization["independent_acquisition_count"] == 2
    assert materialization["candidate_byte_identical"] is True
    assert materialization["report_byte_identical"] is True
    assert materialization["scanned_records"] == selection["max_scanned_records"]
    assert materialization["retained_records"] == 972
    assert materialization["retained_records"] == materialization["selection_reasons"]["accepted"]
    assert sum(materialization["selection_reasons"].values()) == materialization["scanned_records"]
    assert materialization["retained_normalized_utf8_bytes"] <= (
        selection["max_retained_normalized_utf8_bytes"]
    )
    assert materialization["candidate_payload_sha256"] == (
        materialization["candidate_file_sha256"]
    )
    for key in ("candidate_payload_sha256", "candidate_file_sha256", "report_file_sha256"):
        assert SHA256_RE.fullmatch(materialization[key]) is not None


def test_execution_evidence_remains_zero_credit_and_text_free() -> None:
    evidence = load_json(EVIDENCE_PATH)
    truth = evidence["truth_boundary"]

    assert truth["durable_evidence_contains_source_text"] is False
    assert truth["source_rights_review_status"] == "REVIEW_REQUIRED"
    assert truth["canonical_corpus_admitted"] is False
    assert truth["training_eligible"] is False
    assert truth["evaluation_eligible"] is False
    assert truth["training_authorized_bytes"] == 0
    assert truth["canonical_capacity_credited"] == 0
    assert truth["family_credit_added"] == 0
    assert truth["unique_causal_loss_positions_authorized"] == 0
    assert truth["tokenizer_fit_authorized"] is False
    assert truth["optimizer_updates"] == 0
    assert truth["model_training_executed"] is False
    assert truth["learned_weights_created"] is False
    assert truth["final_test_accessed"] is False
    assert truth["paid_compute_used"] is False
    assert truth["foreign_pretrained_weights_used"] is False
    assert truth["external_llm_or_api_used_for_data_or_intelligence"] is False
    assert "text" not in evidence["materialization"]
    assert "rows" not in evidence["materialization"]
