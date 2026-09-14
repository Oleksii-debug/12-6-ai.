"""Fail-closed authority for known external-LLM training-data contamination.

This module does not claim that every non-blocked corpus byte is external-LLM clean.
It authenticates one independently established Nomis1864 contamination finding and
makes that exact payload, plus its known source identity, inadmissible to the D03
training-data lineage until a successor authority is rebuilt without it.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "12-6.d03-external-llm-provenance-quarantine.v1"
EXPECTED_AUTHORITY_IDENTITY_SHA256 = (
    "e9f29dd9f710fac057550e5cd671b7f412720e1ceb36568565a79909f11cf5b6"
)
BLOCKED_FAMILY = "ua.verba.public-domain.nomis1864"
BLOCKED_RECORD_ID = "ua.verba.nomis1864.bounded24"
BLOCKED_SOURCE_ID = "ua.verba.nomis1864.bounded24"
BLOCKED_PAYLOAD_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
BLOCKED_PAYLOAD_BYTES = 1659
EXPECTED_INVALIDATED_AUTHORITIES = {
    "next100_063_terminal_source_registry_v5_git_blob_sha1": (
        "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
    ),
    "next100_063_terminal_source_registry_v6_git_blob_sha1": (
        "13789effe506a815e92e4f0e22ada773d366f316"
    ),
    "next100_063_terminal_source_registry_v6_identity_sha256": (
        "c7b988081a270cd53c37f22721499568ec44f67daa998c87573b43c463642eae"
    ),
    "next100_065f_global_dedup_v8_report_sha256": (
        "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a"
    ),
    "next100_065f_global_dedup_v8_survivor_authority_sha256": (
        "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf"
    ),
}
EXPECTED_ENFORCEMENT = {
    "reject_matching_payload_sha256_under_any_record_or_source_name": True,
    "reject_known_family_record_or_source_identity_even_if_payload_changes": True,
    "silent_filtering_forbidden": True,
    "successor_authority_rebuild_required": True,
    "all_other_corpus_bytes_declared_external_llm_clean": False,
}
EXPECTED_TRUTH = {
    "current_contaminated_materialization_launch_authoritative": False,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}


class ExternalLLMProvenanceQuarantineError(ValueError):
    """Raised when quarantine authority drifts or a blocked record is present."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ExternalLLMProvenanceQuarantineError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_authority(
    authority: Mapping[str, Any],
    *,
    expected_identity_sha256: str = EXPECTED_AUTHORITY_IDENTITY_SHA256,
) -> str:
    """Authenticate the exact V1 quarantine authority and its fail-closed boundary."""
    expected_root_keys = {
        "schema_version",
        "status",
        "scope",
        "reason",
        "blocked_records",
        "invalidated_for_training_admission",
        "enforcement",
        "truth_boundary",
        "quarantine_identity_sha256",
    }
    _need(set(authority) == expected_root_keys, "quarantine authority schema is not closed")
    _need(authority.get("schema_version") == SCHEMA, "quarantine schema drift")
    _need(authority.get("status") == "ACTIVE_STOP_THE_LINE", "quarantine not active")
    _need(
        authority.get("scope") == "TRAINING_DATA_AUTHORITY_ONLY",
        "quarantine scope drift",
    )
    claimed = authority.get("quarantine_identity_sha256")
    _need(
        isinstance(claimed, str) and claimed == expected_identity_sha256,
        "quarantine identity is not independently expected",
    )
    core = dict(authority)
    del core["quarantine_identity_sha256"]
    _need(claimed == _sha256(_canonical(core)), "quarantine self-hash mismatch")

    blocked = authority.get("blocked_records")
    _need(type(blocked) is list and len(blocked) == 1, "exact blocked-record set required")
    row = blocked[0]
    _need(isinstance(row, Mapping), "blocked record must be an object")
    expected_row = {
        "family": BLOCKED_FAMILY,
        "record_id": BLOCKED_RECORD_ID,
        "source_id": BLOCKED_SOURCE_ID,
        "payload_sha256": BLOCKED_PAYLOAD_SHA256,
        "payload_bytes": BLOCKED_PAYLOAD_BYTES,
        "source_pr": 462,
        "upstream_verba_commit": "34a2c10ac35e1febad6c270a88fc8b83790407da",
        "physical_materialization_run_id": 34783440822,
        "physical_materialization_artifact_id": 10325234135,
    }
    _need(dict(row) == expected_row, "blocked Nomis1864 authority drift")
    _need(
        authority.get("invalidated_for_training_admission")
        == EXPECTED_INVALIDATED_AUTHORITIES,
        "invalidated authority set drift",
    )
    _need(authority.get("enforcement") == EXPECTED_ENFORCEMENT, "quarantine enforcement drift")
    _need(authority.get("truth_boundary") == EXPECTED_TRUTH, "quarantine truth boundary drift")
    return claimed


def reject_quarantined_records(
    records: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    *,
    expected_identity_sha256: str = EXPECTED_AUTHORITY_IDENTITY_SHA256,
) -> str:
    """Reject the known contaminated payload even if identifiers are renamed/resealed.

    The inverse is intentionally not claimed: passing this check proves only that the
    known Nomis1864 contamination is absent. It is not a whole-corpus cleanliness
    certificate.
    """
    identity = validate_authority(
        authority,
        expected_identity_sha256=expected_identity_sha256,
    )
    for index, record in enumerate(records):
        _need(isinstance(record, Mapping), f"record[{index}] must be an object")
        payload = record.get("normalized_payload")
        _need(isinstance(payload, str), f"record[{index}].normalized_payload missing")
        payload_raw = payload.encode("utf-8")
        payload_sha = _sha256(payload_raw)
        if payload_sha == BLOCKED_PAYLOAD_SHA256:
            _need(
                len(payload_raw) == BLOCKED_PAYLOAD_BYTES,
                "blocked payload SHA matched with impossible byte-count drift",
            )
            raise ExternalLLMProvenanceQuarantineError(
                f"known external-LLM payload quarantined at record[{index}]"
            )
        if (
            record.get("family") == BLOCKED_FAMILY
            or record.get("record_id") == BLOCKED_RECORD_ID
            or record.get("source_id") == BLOCKED_SOURCE_ID
        ):
            raise ExternalLLMProvenanceQuarantineError(
                f"known Nomis1864 authority identity quarantined at record[{index}]"
            )
    return identity
