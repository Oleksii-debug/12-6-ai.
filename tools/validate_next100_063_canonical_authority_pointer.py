#!/usr/bin/env python3
"""Fail closed if downstream could consume a non-canonical NEXT100-063 registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POINTER_PATH = ROOT / "configs/data/next100_063_canonical_authority_pointer_v1.json"

EXPECTED_POINTER_ID = "3388474cfb79431f13d5c23e6a41ddb7e7c18fc817d9eb9c8b6aaf57537582fb"
EXPECTED_CANONICAL_ID = "448dd61ed3e0d78d0bca9e202529a79c02811fd67beebe4833373d0c2ab0c0a7"
EXPECTED_CANONICAL_BLOB_SHA1 = "ba25cc1d4e4acca71752408db4fd0437ea1a07ae"
EXPECTED_V1_TREE_ID = "56ab7c1e3ebf336c0ec9f51d8580f9aaad2890566d1bf7a5090e2b7f0102bfc5"
SOURCE_HEAD_SEMANTICS = "DIAGNOSTIC_ONLY_NOT_TERMINAL_AUTHORITY"
CONSUMER_RULE = (
    "DOWNSTREAM_MUST_RESOLVE_THIS_POINTER_AND_MATCH_CANONICAL_REGISTRY_IDENTITY_"
    "AND_BLOB_BEFORE_CONSUMING_NEXT100_063"
)
FORBIDDEN_IDS = {
    EXPECTED_V1_TREE_ID,
    "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526",
    "934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity_without_field(data: dict[str, Any], field: str) -> str:
    body = dict(data)
    body.pop(field, None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return the content-addressed Git blob id for one on-disk file."""

    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def is_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def validate() -> dict[str, Any]:
    pointer = load(POINTER_PATH)
    require(
        pointer.get("schema") == "12-6.next100-063-canonical-authority-pointer.v1",
        "pointer schema drift",
    )
    require(
        pointer.get("worker_id") == "NEXT100-063-CANONICAL-AUTHORITY-SELECTOR",
        "pointer worker drift",
    )
    require(pointer.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")
    require(
        pointer.get("pointer_identity_sha256") == EXPECTED_POINTER_ID,
        "pointer identity drift",
    )
    require(
        identity_without_field(pointer, "pointer_identity_sha256") == EXPECTED_POINTER_ID,
        "pointer self-hash mismatch",
    )

    # A moving PR head is useful only as an observation. It must never become the
    # canonical data identity, because a draft branch can advance while the exact
    # registry payload remains unchanged.
    observed_head = pointer.get("observed_source_pr_head_sha")
    require(is_sha1(observed_head), "observed source PR head must be a SHA-1")
    require(
        pointer.get("source_head_semantics") == SOURCE_HEAD_SEMANTICS,
        "moving source head promoted to authority",
    )
    require(
        "authority_source_head_sha" not in pointer,
        "legacy transient authority_source_head_sha must not be consumable",
    )

    canonical = pointer.get("canonical")
    require(isinstance(canonical, dict), "canonical pointer missing")
    canonical_path = ROOT / str(canonical.get("path", ""))
    require(
        canonical_path
        == ROOT / "configs/data/next100_063_terminal_source_registry_v2.json",
        "canonical path drift",
    )
    require(canonical_path.is_file(), "canonical registry missing")
    require(
        canonical.get("blob_sha1") == EXPECTED_CANONICAL_BLOB_SHA1,
        "pointer canonical blob drift",
    )
    require(
        git_blob_sha1(canonical_path) == EXPECTED_CANONICAL_BLOB_SHA1,
        "canonical registry Git blob mismatch",
    )

    registry = load(canonical_path)
    require(
        registry.get("schema_version") == canonical.get("schema_version"),
        "canonical schema mismatch",
    )
    require(
        registry.get("worker_id") == canonical.get("worker_id"),
        "canonical worker mismatch",
    )
    require(
        registry.get("registry_identity_sha256") == EXPECTED_CANONICAL_ID,
        "canonical registry identity drift",
    )
    require(
        canonical.get("registry_identity_sha256") == EXPECTED_CANONICAL_ID,
        "pointer canonical identity drift",
    )
    require(
        registry.get("pre_global_dedup_inventory", {}).get("candidate_normalized_bytes")
        == 303374,
        "canonical capacity drift",
    )
    require(canonical.get("candidate_normalized_bytes") == 303374, "pointer capacity drift")
    require(
        registry.get("downstream_gate_vector", {}).get(
            "authorized_balanced_no_replay_loss_positions"
        )
        == 0,
        "training exposure promoted",
    )
    require(
        canonical.get("authorized_balanced_no_replay_loss_positions") == 0,
        "pointer training exposure promoted",
    )

    old_path = ROOT / "configs/data/next100_063_terminal_source_registry_v1.json"
    require(old_path.is_file(), "historical V1 path unexpectedly missing")
    old = load(old_path)
    old_id = old.get("registry_identity_sha256")
    require(old_id == EXPECTED_V1_TREE_ID, "historical V1 tree identity drift")
    require(old_id in FORBIDDEN_IDS, "historical V1 unexpectedly consumable")
    require(old_id != EXPECTED_CANONICAL_ID, "V1/V2 authority collision")

    superseded = pointer.get("superseded")
    require(isinstance(superseded, list) and superseded, "superseded identity vector missing")
    declared_forbidden = {
        row.get("observed_registry_identity_sha256")
        or row.get("registry_identity_sha256")
        for row in superseded
        if isinstance(row, dict)
    }
    require(declared_forbidden == FORBIDDEN_IDS, "superseded identity set drift")
    require(
        all("DO_NOT_CONSUME" in str(row.get("status", "")) for row in superseded),
        "superseded status weakened",
    )

    require(pointer.get("consumer_rule") == CONSUMER_RULE, "consumer rule weakened")
    boundary = pointer.get("truth_boundary")
    require(isinstance(boundary, dict) and boundary, "truth boundary missing")
    require(
        all(value is False for value in boundary.values()),
        "downstream claim prematurely promoted",
    )

    return {
        "status": "PASS_CANONICAL_V2_ONLY",
        "pointer_identity_sha256": EXPECTED_POINTER_ID,
        "canonical_registry_identity_sha256": EXPECTED_CANONICAL_ID,
        "canonical_registry_blob_sha1": EXPECTED_CANONICAL_BLOB_SHA1,
        "source_head_semantics": SOURCE_HEAD_SEMANTICS,
        "observed_source_pr_head_sha": observed_head,
        "forbidden_registry_identity_sha256": sorted(FORBIDDEN_IDS),
        "candidate_normalized_bytes": 303374,
        "authorized_balanced_no_replay_loss_positions": 0,
        "next_gate": "NEXT100_063_EXACT_HEAD_TERMINAL_VALIDATION_THEN_DATA_526",
    }


def main() -> int:
    print(json.dumps(validate(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
