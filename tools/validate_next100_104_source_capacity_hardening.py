#!/usr/bin/env python3
"""Validate the NEXT100-104 fail-closed source-capacity correction."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_PARENT_IDENTITY = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
EXPECTED_CPYTHON_AUTHORITY = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
EXPECTED_KNOWN_BYTES = 547_842
EXPECTED_STRATA = {"uk": 100_856, "en": 150_643, "code": 296_343}


class CapacityHardeningError(ValueError):
    pass


def _fail(message: str) -> None:
    raise CapacityHardeningError(message)


def _identity(document: dict[str, Any]) -> str:
    payload = copy.deepcopy(document)
    payload.pop("authority_identity_sha256", None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(document: dict[str, Any]) -> None:
    if document.get("schema_version") != "12-6.next100-104-source-capacity-hardening.v1":
        _fail("schema mismatch")
    if document.get("local_free_only") is not True:
        _fail("LOCAL_FREE boundary removed")

    parent = document.get("parent_registry")
    if not isinstance(parent, dict):
        _fail("parent registry missing")
    if parent.get("registry_identity_sha256") != EXPECTED_PARENT_IDENTITY:
        _fail("parent registry identity drift")
    if parent.get("reported_candidate_normalized_bytes") != 565_743:
        _fail("parent reported-byte evidence drift")

    finding = document.get("finding")
    if not isinstance(finding, dict):
        _fail("finding missing")
    if finding.get("authority_identity_sha256") != EXPECTED_CPYTHON_AUTHORITY:
        _fail("CPython authority drift")
    if finding.get("source_normalized_utf8_bytes") != 17_901:
        _fail("CPython source byte evidence drift")
    if finding.get("chunk_count") != 16:
        _fail("CPython chunk count drift")
    if finding.get("accepted_chunk_count") != 14:
        _fail("CPython accepted chunk count drift")
    if finding.get("rejected_chunk_count") != 2:
        _fail("CPython rejected chunk count drift")
    if finding.get("exact_accepted_chunk_byte_total") != "NOT_MATERIALIZED":
        _fail("unknown accepted-chunk byte total was fabricated")
    if finding.get("exact_total_capacity_claim_allowed") is not False:
        _fail("exact capacity must remain fail-closed")

    corrected = document.get("corrected_fail_closed_accounting")
    if not isinstance(corrected, dict):
        _fail("corrected accounting missing")
    if corrected.get("known_exact_training_eligible_bytes_lower_bound") != EXPECTED_KNOWN_BYTES:
        _fail("known-byte lower bound mismatch")
    if corrected.get("by_stratum_known_exact_bytes_lower_bound") != EXPECTED_STRATA:
        _fail("stratum lower-bound vector mismatch")
    if sum(EXPECTED_STRATA.values()) != EXPECTED_KNOWN_BYTES:
        _fail("validator constant arithmetic broken")
    if corrected.get("candidate_exact_training_eligible_bytes") is not None:
        _fail("candidate exact bytes must remain null")
    if corrected.get("exact_target_gap_normalized_bytes") is not None:
        _fail("exact target gap must remain null")
    if corrected.get("authorized_balanced_no_replay_loss_positions") != 0:
        _fail("source capacity cannot authorize loss positions")
    if corrected.get("long_training") != "BLOCKED":
        _fail("long training must remain blocked")

    boundary = document.get("claim_boundary")
    if not isinstance(boundary, dict):
        _fail("claim boundary missing")
    if boundary.get("parent_exact_byte_total_rejected") is not True:
        _fail("parent exact-byte claim must stay rejected")
    if boundary.get("parent_family_vector_rejected") is not False:
        _fail("family vector is not rejected by this authority")
    if boundary.get("training_authorized") is not False:
        _fail("training authorization firewall violated")
    if boundary.get("loss_positions_authorized") != 0:
        _fail("loss-position firewall violated")

    expected = document.get("authority_identity_sha256")
    actual = _identity(document)
    if expected != actual:
        _fail(f"authority identity mismatch: expected={expected} actual={actual}")


def main() -> int:
    path = Path("configs/data/next100_104_source_capacity_hardening_v1.json")
    validate(json.loads(path.read_text(encoding="utf-8")))
    print("NEXT100-104 source-capacity hardening: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
