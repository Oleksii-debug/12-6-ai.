from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "d03_pdr_attributable_subset_terminal_execution_v1.json"


def test_terminal_pdr_subset_evidence_binds_physical_replay_without_credit() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == "12-6.d03-pdr-attributable-subset-terminal-evidence.v1"
    assert evidence["execution"]["run_id"] == 36521055931
    assert evidence["execution"]["execution_head_sha"] == "76ab3d6c3fd1ade35bf371209bd89b5929201001"
    assert evidence["execution"]["independent_slots"] == ["a", "b"]
    assert evidence["execution"]["deterministic_compare_succeeded"] is True

    subset = evidence["attributable_subset"]
    assert subset["attributable_record_count"] == 498
    assert subset["excluded_record_count"] == 668
    assert subset["attributable_normalized_utf8_bytes"] == 3_727_864
    assert subset["attributable_projection_identity_sha256"] == (
        "79cff661429f365c412e3113a72a0d8b35566f19f61facca3bc643d86299e32b"
    )
    assert subset["authority_identity_sha256"] == (
        "0fd7388521c693028f999a58737ca40b4ce9640eedcb33dda8a9559d62d77832"
    )

    boundary = evidence["truth_boundary"]
    for key in (
        "pdr_source_admitted_records",
        "pdr_source_admitted_bytes",
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        assert type(boundary[key]) is int
        assert boundary[key] == 0
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
    ):
        assert boundary[key] is False
    assert boundary["source_rights_review_status"] == "REVIEW_REQUIRED"
