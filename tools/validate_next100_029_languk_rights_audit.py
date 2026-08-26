#!/usr/bin/env python3
"""Validate NEXT100-029 fail-closed Lang-UK rights evidence using stdlib only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "configs/data/next100_029_languk_rights_audit_v1.json"


def _identity(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "evidence_identity_sha256"}
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(path: Path = EVIDENCE) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "12-6.data-ua-languk-rights-audit.v1"
    assert payload["worker_id"] == "NEXT100-029-DATA-UA-LANGUK"
    assert payload["execution_profile"] == "LOCAL_FREE"
    assert payload["policy"]["public_or_research_availability_implies_training_rights"] is False
    assert payload["policy"]["mixed_third_party_rights_without_safe_separation"] == "REJECT"

    by_id = {item["candidate_id"]: item for item in payload["candidates"]}
    assert by_id["languk.ubertext"]["verdict"] == "REJECT_MIXED_RIGHTS"
    assert by_id["brown-uk.corpus"]["verdict"] == "REJECT_CHAIN_OF_TITLE_UNPROVEN"
    assert by_id["languk.malyuk"]["verdict"] == "REJECT_MIXED_UPSTREAM_RIGHTS"

    court = by_id["languk.court-decisions-uk.supreme-2024-5k"]
    assert court["verdict"] == "RETEST_PRIVACY_AND_BYTE_MATERIALIZATION"
    assert court["upstream_revision"] == "289c0316fc076db3e1607db6776a29df42f4ffc5"
    assert court["selected_file_origin_commit"] == "2dcac4c941b87bf9c242bdc919cef4b40f4a4813"
    assert court["selected_file_raw_sha256"] == "9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597"
    assert court["selected_file_bytes"] == 20_220_778
    assert court["license"] == "MIT"
    assert court["privacy"]["state"] == "BLOCKED_PENDING_DETERMINISTIC_SCAN"
    assert court["family_independence"]["independent_from_incumbent_rada_family"] is True
    assert court["language_quality"]["independent_language_scan_completed"] is False

    subset = court["bounded_subset_rule"]
    assert subset["exclude_file"] == "250-deanonymized-court-cases.parquet"
    assert 0 < subset["max_records"] <= 256
    assert subset["materialized_records_now"] == 0
    assert subset["materialized_training_bytes_now"] == 0
    assert subset["raw_subset_sha256"] is None
    assert subset["normalized_subset_sha256"] is None

    terminal = payload["terminal_result"]
    assert terminal["verdict"] == "RETEST_LANGUK_COURT_DECISIONS_ONLY"
    assert terminal["training_source_admitted"] is False
    assert terminal["new_uk_family_admitted"] is False
    assert terminal["materialized_training_records"] == 0
    assert terminal["materialized_training_bytes"] == 0
    assert terminal["registry_change_authorized"] is False

    expected = payload["evidence_identity_sha256"]
    actual = _identity(payload)
    assert expected == actual, (expected, actual)
    return payload


if __name__ == "__main__":
    result = validate()
    print(
        "PASS",
        result["terminal_result"]["verdict"],
        result["evidence_identity_sha256"],
    )
