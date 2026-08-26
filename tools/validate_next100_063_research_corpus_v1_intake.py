#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-063 Research Corpus V1 intake."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.next100-063-research-corpus-v1-intake.v1"
EXPECTED_PARENT = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_BYTES = 251_572
EXPECTED_RECORDS = 14
EXPECTED_FAMILIES = {"uk": 2, "en": 2, "code": 2}
EXPECTED_SOURCE_HEADS = {
    "DATA-287-EXTERNAL-SNAPSHOT-REGISTRY-V2": "b0523ccbc4b957615aac849d476cfa851be87578",
    "NEXT100-026-DATA-UA-CABINET-MINISTRY": "40950a950b60921fd856af2719e1ae2486d9e892",
    "NEXT100-034-DATA-EN-NIST": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
}


class IntakeValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise IntakeValidationError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_intake(document: dict[str, Any]) -> None:
    if document.get("schema_version") != EXPECTED_SCHEMA:
        _fail("schema_version mismatch")
    if document.get("status") != "PREDECONTAMINATION_CANDIDATE_FROZEN":
        _fail("intake status must remain pre-decontamination")
    if document.get("local_free_only") is not True:
        _fail("LOCAL_FREE boundary removed")

    parent = document.get("parent_corpus_authority")
    if not isinstance(parent, dict) or parent.get("head_sha") != EXPECTED_PARENT:
        _fail("DATA-301 parent binding mismatch")
    if parent.get("terminal_state") != "TERMINAL_BLOCKED":
        _fail("blocked DATA-301 parent must not be rewritten as terminal success")

    source_authorities = document.get("source_authorities")
    if not isinstance(source_authorities, list) or len(source_authorities) != 3:
        _fail("exactly three source authorities are required")
    actual_heads = {}
    for authority in source_authorities:
        if not isinstance(authority, dict):
            _fail("source authority must be an object")
        worker = authority.get("worker")
        actual_heads[worker] = authority.get("head_sha")
        if authority.get("workflow_conclusion") != "success":
            _fail(f"source authority is not terminal-green: {worker}")
    if actual_heads != EXPECTED_SOURCE_HEADS:
        _fail("source authority exact-head vector mismatch")

    records = document.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        _fail("record count mismatch")

    record_ids: set[str] = set()
    hashes: set[str] = set()
    families: dict[str, set[str]] = defaultdict(set)
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            _fail("record must be an object")
        record_id = record.get("record_id")
        digest = record.get("normalized_sha256")
        stratum = record.get("stratum")
        family_id = record.get("family_id")
        normalized_bytes = record.get("normalized_bytes")
        locator = record.get("locator")
        if not isinstance(record_id, str) or not record_id:
            _fail("record_id missing")
        if record_id in record_ids:
            _fail(f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        if not isinstance(digest, str) or len(digest) != 64:
            _fail(f"invalid normalized_sha256: {record_id}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise IntakeValidationError(
                f"non-hex normalized_sha256: {record_id}"
            ) from exc
        if digest in hashes:
            _fail(f"duplicate normalized content hash: {digest}")
        hashes.add(digest)
        if stratum not in EXPECTED_FAMILIES:
            _fail(f"invalid stratum: {record_id}")
        if not isinstance(family_id, str) or not family_id:
            _fail(f"family_id missing: {record_id}")
        if (
            not isinstance(normalized_bytes, int)
            or isinstance(normalized_bytes, bool)
            or normalized_bytes <= 0
        ):
            _fail(f"normalized_bytes invalid: {record_id}")
        if not isinstance(locator, dict) or not locator:
            _fail(f"content locator missing: {record_id}")
        families[stratum].add(family_id)
        total_bytes += normalized_bytes

    if total_bytes != EXPECTED_BYTES:
        _fail(f"normalized byte total mismatch: {total_bytes}")
    if {key: len(value) for key, value in families.items()} != EXPECTED_FAMILIES:
        _fail("independent-family vector must remain exactly 2/2/2")

    summary = document.get("inventory_summary")
    if not isinstance(summary, dict):
        _fail("inventory_summary missing")
    if summary.get("record_count") != EXPECTED_RECORDS:
        _fail("summary record_count mismatch")
    if summary.get("unique_normalized_hash_count") != EXPECTED_RECORDS:
        _fail("summary unique hash count mismatch")
    if summary.get("normalized_source_bytes") != EXPECTED_BYTES:
        _fail("summary normalized byte count mismatch")
    if summary.get("g09_family_minimum_satisfied_predecontamination") is not True:
        _fail("G09 pre-decontamination family gate regressed")

    boundary = document.get("claim_boundary")
    if not isinstance(boundary, dict):
        _fail("claim_boundary missing")
    forbidden_truthy = (
        "training_authorized",
        "long_training_authorized",
        "corpus_release_claimed",
        "representative_corpus_claimed",
        "learned_20m_claimed",
    )
    for key in forbidden_truthy:
        if boundary.get(key) is not False:
            _fail(f"claim firewall violated: {key}")
    if boundary.get("authorized_unique_loss_positions") != 0:
        _fail("pre-decontamination intake cannot authorize loss positions")
    if boundary.get("final_research_corpus_v1_identity") is not None:
        _fail("final corpus identity must remain null")
    if boundary.get("shard_identity") is not None:
        _fail("shard identity must remain null")

    expected_identity = document.get("predecontamination_candidate_identity_sha256")
    if not isinstance(expected_identity, str) or len(expected_identity) != 64:
        _fail("candidate identity missing")
    identity_payload = copy.deepcopy(document)
    identity_payload.pop("predecontamination_candidate_identity_sha256", None)
    actual_identity = _canonical_sha256(identity_payload)
    if actual_identity != expected_identity:
        _fail(
            "candidate identity mismatch: "
            f"expected={expected_identity} actual={actual_identity}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="configs/data/next100_063_research_corpus_v1_intake.json",
    )
    args = parser.parse_args()
    document = json.loads(Path(args.path).read_text(encoding="utf-8"))
    validate_intake(document)
    print("NEXT100-063 intake validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
