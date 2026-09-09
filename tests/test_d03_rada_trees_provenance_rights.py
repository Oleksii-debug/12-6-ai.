from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data" / "d03_rada_trees_provenance_rights_v1.json"
EVIDENCE = (
    ROOT
    / "evidence"
    / "d03-rada-trees"
    / "secondary-plaintext-provenance-rights-v1.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_rada_provenance_rights_evidence_is_hash_bound_and_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["config_sha256"] == _canonical_sha256(config)
    claimed_report_sha = evidence.pop("report_sha256")
    assert _canonical_sha256(evidence) == claimed_report_sha
    assert claimed_report_sha == (
        "7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d"
    )

    provenance = evidence["provenance"]
    assert provenance["accepted_exact_unique_members"] == 4384
    assert provenance["accepted_exact_unique_bytes"] == 877_899_128
    assert provenance["held_exact_unique_members"] == 1
    assert provenance["held_exact_unique_bytes"] == 84_781
    assert provenance["minimum_date"] == "1990-05-15"
    assert provenance["maximum_date"] == "2024-01-16"
    assert provenance["unique_session_dates"] == 2963
    assert sorted(map(int, provenance["year_member_counts"])) == list(range(1990, 2025))
    assert sum(provenance["year_member_counts"].values()) == 4384
    assert sum(provenance["year_bytes"].values()) == 877_899_128

    holds = provenance["explicit_holds"]
    assert len(holds) == 1
    assert holds[0]["path"] == (
        "texts/stenogramy-zasidannya-rnbo-vid-28-lyutogo-2014-roku.txt"
    )
    assert holds[0]["size_bytes"] == 84_781

    authorities = evidence["rights_authorities"]
    assert authorities[0]["license"] == "CC-BY-4.0"
    assert authorities[1]["condition"] == "MANDATORY_SOURCE_ATTRIBUTION"
    assert authorities[2]["condition"] == "MANDATORY_SOURCE_ATTRIBUTION"

    boundary = evidence["claim_boundary"]
    assert boundary["rights_scope_candidate_bytes_are_training_credit"] is False
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["final_test_payload_accessed"] is False
    assert boundary["paid_compute_used"] is False
