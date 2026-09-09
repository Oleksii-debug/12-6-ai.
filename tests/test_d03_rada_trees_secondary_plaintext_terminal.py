from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "evidence"
    / "d03-rada-trees"
    / "secondary-plaintext-full-scan-terminal-v1.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_full_plaintext_terminal_evidence_is_exact_and_fail_closed() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "12-6.d03-rada-trees-secondary-plaintext-terminal.v1"
    )

    expected_identity = evidence.pop("evidence_identity_sha256")
    assert _canonical_sha256(evidence) == expected_identity

    execution = evidence["execution"]
    assert execution["workflow_run"] == 34162348408
    assert execution["workflow_job"] == 101866579250
    assert execution["conclusion"] == "success"
    assert execution["two_independent_processing_passes_byte_identical"] is True
    assert execution["artifact_id"] == 10033726682
    assert execution["artifact_digest"] == (
        "sha256:ac52bf2aaeffc166c681181ee542626d621ee9a273540a8c24b8240f4931b56c"
    )

    source = evidence["source"]
    assert source["content_sha256"] == (
        "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
    )
    assert source["listing_identity_sha256"] == (
        "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
    )

    measurement = evidence["measurement"]
    assert measurement["selected_member_count"] == 4391
    assert measurement["selected_listed_bytes"] == 879_031_855
    assert measurement["plain_text_candidate_members_before_exact_duplicate_collapse"] == 4391
    assert measurement["plain_text_candidate_exact_unique_payload_count"] == 4385
    assert measurement["plain_text_candidate_bytes_before_exact_duplicate_collapse"] == 879_031_855
    assert measurement["plain_text_candidate_bytes_after_exact_duplicate_collapse"] == 877_983_909
    assert measurement["exact_duplicate_group_count"] == 6
    assert measurement["exact_duplicate_member_discount"] == 6
    assert measurement["exact_duplicate_byte_discount"] == 1_047_946
    assert (
        measurement["plain_text_candidate_bytes_before_exact_duplicate_collapse"]
        - measurement["plain_text_candidate_bytes_after_exact_duplicate_collapse"]
        == measurement["exact_duplicate_byte_discount"]
    )
    assert measurement["raw_member_text_persisted"] is False

    boundary = evidence["claim_boundary"]
    assert boundary["candidate_bytes_are_training_credit"] is False
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["model_training_executed"] is False
    assert boundary["final_test_payload_accessed"] is False
    assert boundary["paid_compute_used"] is False
    assert boundary["research_corpus_v1_released"] is False
