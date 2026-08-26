#!/usr/bin/env python3
"""Run canonical Rada_Trees archive inventory from the exact HF object snapshot.

This is a fail-closed provenance bridge between PR #638 metadata evidence and
the archive inventory engine in this branch. Operator-supplied size/object IDs
are not accepted: they are derived from the self-hashed parent snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import inventory_d03_rada_trees_archive as inventory

REPO_ID = "uacorpus/Rada_Trees"
REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PRIMARY_ARCHIVE = "Rada_Trees.7z"
SECONDARY_ARCHIVE = "rada_xtag_texts.7z"
SNAPSHOT_SCHEMA = "12-6.d03-rada-trees-hf-object-identity.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class HandoffError(ValueError):
    """Raised when the metadata-to-content authority chain is not exact."""


def snapshot_identity(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hf_object_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    identity = snapshot.get("snapshot_identity_sha256")
    if not isinstance(identity, str) or HEX64.fullmatch(identity) is None:
        raise HandoffError("missing or malformed snapshot identity")

    body = dict(snapshot)
    del body["snapshot_identity_sha256"]
    if snapshot_identity(body) != identity:
        raise HandoffError("snapshot identity mismatch")

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise HandoffError("unexpected HF object snapshot schema")

    source = snapshot.get("source")
    if not isinstance(source, dict):
        raise HandoffError("snapshot source is missing")
    if source.get("repo_id") != REPO_ID or source.get("revision") != REVISION:
        raise HandoffError("snapshot dataset/revision drift")

    verification = snapshot.get("verification")
    required_verification = {
        "tree_revision_is_immutable_40hex": True,
        "git_blob_oids_bound": True,
        "xet_hashes_bound": True,
        "resolve_header_xet_hashes_match_tree": True,
    }
    if not isinstance(verification, dict):
        raise HandoffError("snapshot verification vector is missing")
    for key, expected in required_verification.items():
        if verification.get(key) is not expected:
            raise HandoffError(f"snapshot verification is not terminal: {key}")

    boundary = snapshot.get("claim_boundary")
    if not isinstance(boundary, dict):
        raise HandoffError("snapshot claim boundary is missing")
    if boundary.get("training_authorized_bytes") != 0:
        raise HandoffError("metadata snapshot cannot authorize training bytes")
    if boundary.get("archives_downloaded") is not False:
        raise HandoffError("metadata snapshot cannot claim archive download")
    if boundary.get("archive_content_sha256_verified") is not False:
        raise HandoffError("metadata snapshot cannot claim content SHA-256")

    files = snapshot.get("files")
    if not isinstance(files, list):
        raise HandoffError("snapshot file inventory is missing")
    paths = [row.get("path") for row in files if isinstance(row, dict)]
    if paths != [PRIMARY_ARCHIVE, SECONDARY_ARCHIVE]:
        raise HandoffError("snapshot archive inventory/order drift")

    primary = files[0]
    if not isinstance(primary, dict):
        raise HandoffError("primary archive row is malformed")
    size = primary.get("size_bytes")
    git_oid = primary.get("git_blob_oid")
    xet_hash = primary.get("xet_hash")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise HandoffError("primary archive exact byte size is invalid")
    if not isinstance(git_oid, str) or HEX40.fullmatch(git_oid) is None:
        raise HandoffError("primary archive Git blob OID is invalid")
    if not isinstance(xet_hash, str) or HEX64.fullmatch(xet_hash) is None:
        raise HandoffError("primary archive Xet identity is invalid")

    return {
        "snapshot_identity_sha256": identity,
        "size_bytes": size,
        "git_blob_oid": git_oid,
        "xet_hash": xet_hash,
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot read HF object snapshot: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffError("HF object snapshot root must be a JSON object")
    return value


def build_bridge_report(
    config: dict[str, Any],
    archive: Path,
    expected_sha256: str,
    snapshot: dict[str, Any],
    extractor: str,
) -> dict[str, Any]:
    handoff = validate_hf_object_snapshot(snapshot)
    inner = inventory.build_report(
        config,
        archive,
        expected_sha256,
        handoff["size_bytes"],
        handoff["xet_hash"],
        extractor,
    )

    archive_evidence = inner.get("archive")
    if not isinstance(archive_evidence, dict):
        raise HandoffError("inventory report archive evidence is missing")
    if archive_evidence.get("size_bytes") != handoff["size_bytes"]:
        raise HandoffError("inventory report size detached from parent snapshot")
    if archive_evidence.get("upstream_object_identity") != handoff["xet_hash"]:
        raise HandoffError("inventory report object identity detached from parent snapshot")

    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-hf-snapshot-inventory-bridge.v1",
        "state": "EXACT_HF_OBJECT_TO_CONTENT_INVENTORY_BOUND_CLASSIFICATION_NOT_RUN",
        "parent_hf_object_snapshot_identity_sha256": handoff[
            "snapshot_identity_sha256"
        ],
        "parent_object": {
            "path": PRIMARY_ARCHIVE,
            "size_bytes": handoff["size_bytes"],
            "git_blob_oid": handoff["git_blob_oid"],
            "xet_hash": handoff["xet_hash"],
        },
        "inventory_report": inner,
        "claim_boundary": {
            "plain_text_classification_complete": False,
            "period_provenance_stratification_complete": False,
            "member_level_rights_provenance_complete": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
        },
    }
    report["bridge_report_identity_sha256"] = inventory.sha256_bytes(
        inventory.canonical_json(report)
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--object-snapshot", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--extractor")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = inventory.load_config()
    snapshot = load_snapshot(args.object_snapshot)
    extractor = inventory.find_extractor(
        args.extractor,
        config["inventory_policy"]["accepted_extractors"],
    )
    report = build_bridge_report(
        config,
        args.archive,
        args.expected_sha256,
        snapshot,
        extractor,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = inventory.canonical_json(report)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, args.output)
    print("D03_RADA_TREES_HF_SNAPSHOT_INVENTORY_BRIDGE=BOUND_ZERO_CREDIT")
    print("BRIDGE_REPORT_IDENTITY_SHA256=" + report["bridge_report_identity_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=CLASSIFY_PLAIN_TEXT_AND_BIND_PERIOD_MEMBER_PROVENANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
