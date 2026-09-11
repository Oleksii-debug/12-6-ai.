"""Fail-closed adapter from D03 post-decontam family vectors to NEXT100-106.

The adapter performs no balance allocation. It preserves the independently bound
post-decontamination family capacities while translating the canonical Ukrainian
stratum name and row shape required by the already-merged NEXT100-106 gate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from twelve_six.data.postdecontam_balance_projection_v1 import (
    ProjectionError,
    verify_family_vector,
)

NEXT100_INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
STRATUM_MAP = {"uk": "ua", "en": "en", "code": "code"}
_HEX = frozenset("0123456789abcdef")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{field} must be a non-empty string")
    return value


def _require_hex(value: Any, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX for ch in value)
    ):
        raise ProjectionError(f"{field} must be {length} lowercase hex characters")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProjectionError(f"{field} must be a positive integer")
    return value


def _verify_dedup_authority(
    document: Mapping[str, Any],
    *,
    expected_worker_id: str,
    expected_head_sha: str,
    expected_evidence_identity_sha256: str,
) -> dict[str, str]:
    required = {
        "worker_id",
        "head_sha",
        "evidence_identity_sha256",
        "terminal_verdict",
    }
    if set(document) != required:
        raise ProjectionError("dedup authority fields are not closed-world")

    worker_id = _require_nonempty_string(document.get("worker_id"), "dedup_authority.worker_id")
    head_sha = _require_hex(document.get("head_sha"), "dedup_authority.head_sha", 40)
    evidence_identity = _require_hex(
        document.get("evidence_identity_sha256"),
        "dedup_authority.evidence_identity_sha256",
        64,
    )
    if document.get("terminal_verdict") != "PASS":
        raise ProjectionError("dedup authority terminal verdict must be PASS")

    if worker_id != _require_nonempty_string(expected_worker_id, "expected_dedup_worker_id"):
        raise ProjectionError("dedup authority worker id does not match external expectation")
    if head_sha != _require_hex(expected_head_sha, "expected_dedup_head_sha", 40):
        raise ProjectionError("dedup authority head does not match external expectation")
    if evidence_identity != _require_hex(
        expected_evidence_identity_sha256,
        "expected_dedup_evidence_identity_sha256",
        64,
    ):
        raise ProjectionError("dedup authority evidence identity does not match external expectation")

    return {
        "worker_id": worker_id,
        "head_sha": head_sha,
        "evidence_identity_sha256": evidence_identity,
        "terminal_verdict": "PASS",
    }


def adapt_family_vector_to_next100_106(
    family_vector: Mapping[str, Any],
    *,
    expected_family_vector_identity_sha256: str,
    dedup_authority: Mapping[str, Any],
    expected_dedup_worker_id: str,
    expected_dedup_head_sha: str,
    expected_dedup_evidence_identity_sha256: str,
) -> dict[str, Any]:
    """Return the exact input schema consumed by merged NEXT100-106."""

    actual_family_vector_identity = verify_family_vector(family_vector)
    expected_family_vector_identity = _require_hex(
        expected_family_vector_identity_sha256,
        "expected_family_vector_identity_sha256",
        64,
    )
    if actual_family_vector_identity != expected_family_vector_identity:
        raise ProjectionError("family vector does not match independently expected identity")

    normalized_authority = _verify_dedup_authority(
        dedup_authority,
        expected_worker_id=expected_dedup_worker_id,
        expected_head_sha=expected_dedup_head_sha,
        expected_evidence_identity_sha256=expected_dedup_evidence_identity_sha256,
    )
    if (
        family_vector["dedup_evidence_identity_sha256"]
        != normalized_authority["evidence_identity_sha256"]
    ):
        raise ProjectionError(
            "dedup authority evidence does not match family-vector lineage"
        )

    rows: list[dict[str, Any]] = []
    by_stratum = defaultdict(int)
    family_count = defaultdict(int)
    seen_family_ids: set[str] = set()
    for index, row in enumerate(family_vector["families"]):
        family_id = _require_nonempty_string(row.get("family"), f"families[{index}].family")
        if family_id in seen_family_ids:
            raise ProjectionError(f"duplicate family id in adapter input: {family_id}")
        seen_family_ids.add(family_id)
        internal_stratum = row.get("stratum")
        if internal_stratum not in STRATUM_MAP:
            raise ProjectionError(f"unsupported adapter stratum for {family_id}")
        stratum = STRATUM_MAP[internal_stratum]
        unique_bytes = _require_positive_int(
            row.get("capacity_bytes"),
            f"families[{index}].capacity_bytes",
        )
        rows.append(
            {
                "family_id": family_id,
                "stratum": stratum,
                "unique_bytes": unique_bytes,
            }
        )
        by_stratum[stratum] += unique_bytes
        family_count[stratum] += 1

    rows.sort(key=lambda row: row["family_id"])
    strata = ("ua", "en", "code")
    return {
        "schema_version": NEXT100_INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": normalized_authority,
        "families": rows,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": {stratum: by_stratum[stratum] for stratum in strata},
            "family_count": {stratum: family_count[stratum] for stratum in strata},
        },
    }
