#!/usr/bin/env python3
"""Offline validator for NEXT100-036 OpenStax terminal source authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_WORKER = "NEXT100-036-DATA-EN-OPENSTAX"
EXPECTED_REGISTRY_SHA = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_REGISTRY_IDENTITY = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_authority(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(doc.get("worker_id") == EXPECTED_WORKER, "worker_id mismatch")
    require(doc.get("terminal_verdict") == "REJECT", "terminal verdict must be REJECT")
    require(doc.get("local_free_only") is True, "LOCAL_FREE must be true")

    registry = doc.get("bound_registry", {})
    require(registry.get("source_sha") == EXPECTED_REGISTRY_SHA, "registry source SHA mismatch")
    require(
        registry.get("registry_identity_sha256") == EXPECTED_REGISTRY_IDENTITY,
        "registry identity mismatch",
    )
    require(registry.get("source_count") == 5, "bound registry source_count must be 5")
    require(
        registry.get("independent_source_family_count") == 4,
        "bound registry family count must be 4",
    )
    require(
        registry.get("unique_normalized_bytes") == 183061,
        "bound registry normalized byte count mismatch",
    )
    require(
        registry.get("representative_corpus_claimed") is False,
        "bound registry must not claim representativeness",
    )

    candidate = doc.get("candidate", {})
    require(candidate.get("provider") == "OpenStax", "provider mismatch")
    require(candidate.get("title") == "Physics", "title mismatch")
    require(candidate.get("language") == "en", "language must be en")
    require(
        candidate.get("pinned_git_commit_sha1")
        == "dfdfd7a5356ecdd42e504de3df50d9153e33ea49",
        "OpenStax commit mismatch",
    )
    require(
        candidate.get("pinned_git_tree_sha1")
        == "b99750d59120e03f65f15da8ae012c5d5bdcfaa7",
        "OpenStax tree mismatch",
    )
    bounded = candidate.get("bounded_object", {})
    require(
        bounded.get("path") == "modules/m54467/index.cnxml",
        "bounded path mismatch",
    )
    require(
        bounded.get("git_blob_sha1")
        == "a1b45a7c27067e950a112a3746b911c5a620c01c",
        "bounded Git blob mismatch",
    )

    family_id = candidate.get("source_family_id")
    require(
        candidate.get("family_identity_sha256")
        == hashlib.sha256(str(family_id).encode("utf-8")).hexdigest(),
        "candidate family identity mismatch",
    )

    locator = {
        "source_repo": candidate.get("source_repo"),
        "pinned_git_commit_sha1": candidate.get("pinned_git_commit_sha1"),
        "pinned_git_tree_sha1": candidate.get("pinned_git_tree_sha1"),
        "path": bounded.get("path"),
        "git_blob_sha1": bounded.get("git_blob_sha1"),
    }
    require(
        bounded.get("source_locator_identity_sha256") == _canonical_sha256(locator),
        "source locator identity mismatch",
    )

    license_doc = doc.get("license", {})
    require(license_doc.get("license_id") == "CC-BY-4.0", "license must be CC-BY-4.0")
    require(
        license_doc.get("license_version") == "4.0 International",
        "license version mismatch",
    )
    require(
        license_doc.get("noncommercial_restriction") is False,
        "NC material is forbidden",
    )
    require(
        license_doc.get("no_derivatives_restriction") is False,
        "ND material is forbidden",
    )
    require(
        bool(license_doc.get("attribution_obligations")),
        "attribution obligations must be explicit",
    )
    require(
        "excluded" in str(license_doc.get("third_party_exclusion", "")).lower(),
        "third-party media exclusion must be explicit",
    )

    rights = doc.get("rights", {})
    training = rights.get("model_training", {})
    require(
        training.get("status") == "REJECT_NO_OPENSTAX_PERMISSION",
        "model training must fail closed",
    )
    require(
        training.get("separate_openstax_permission_evidenced") is False,
        "separate OpenStax permission must not be claimed",
    )
    require(
        rights.get("redistribution", {}).get("status")
        == "ALLOWED_WITH_ATTRIBUTION_FOR_CC_BY_TEXT",
        "CC BY text redistribution decision mismatch",
    )
    require(
        rights.get("evaluation", {}).get("status") == "NOT_SEPARATELY_ADMITTED",
        "evaluation must remain separate",
    )

    hashes = doc.get("source_hashes", {})
    require(
        hashes.get("bounded_object_git_blob_sha1") == bounded.get("git_blob_sha1"),
        "source hash binding mismatch",
    )
    require(hashes.get("raw_payload_sha256") is None, "raw corpus SHA must be absent")
    require(
        hashes.get("raw_payload_sha256_status")
        == "NOT_COMPUTED_NO_CORPUS_MATERIALIZATION",
        "raw SHA status mismatch",
    )

    normalization = doc.get("normalization", {})
    require(
        normalization.get("status") == "NOT_RUN_RIGHTS_REJECT",
        "normalization must not run after rights reject",
    )
    require(normalization.get("normalized_sha256") is None, "normalized SHA must be absent")
    require(normalization.get("normalized_bytes") == 0, "normalized bytes must be zero")
    require(
        bool(normalization.get("policy_if_retested")),
        "deterministic retest normalization policy must be preregistered",
    )

    require(
        doc.get("quality", {}).get("status") == "NOT_RUN_RIGHTS_REJECT",
        "quality must not be credited after rights reject",
    )
    require(
        doc.get("privacy", {}).get("status") == "NOT_RUN_NO_CORPUS_MATERIALIZATION",
        "privacy must not be credited without materialization",
    )

    dedup = doc.get("dedup", {})
    require(
        dedup.get("status") == "NOT_RUN_NO_ADMITTED_PAYLOAD",
        "dedup status mismatch",
    )
    require(dedup.get("admitted_capacity_delta_bytes") == 0, "capacity delta must be zero")
    require(dedup.get("registry_source_count_delta") == 0, "source count delta must be zero")
    require(dedup.get("registry_family_count_delta") == 0, "family count delta must be zero")

    family = doc.get("family", {})
    require(family.get("family_id") == family_id, "family id mismatch")
    require(
        family.get("family_identity_sha256") == candidate.get("family_identity_sha256"),
        "family hash mismatch",
    )
    require(
        family.get("independent_family_credit_if_rejected") == 0,
        "rejected source must add zero family credit",
    )

    boundary = doc.get("claim_boundary", {})
    require(boundary.get("training_admitted") is False, "training must not be admitted")
    require(boundary.get("capacity_added_bytes") == 0, "capacity added must be zero")
    require(boundary.get("family_credit_added") == 0, "family credit added must be zero")
    require(boundary.get("representative_claimed") is False, "representativeness forbidden")

    expected_identity = doc.get("authority_identity_sha256")
    unsigned = dict(doc)
    unsigned.pop("authority_identity_sha256", None)
    require(
        expected_identity == _canonical_sha256(unsigned),
        "authority identity SHA256 mismatch",
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="evidence/next100_036/openstax_physics_source_authority.json",
    )
    args = parser.parse_args()
    doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate_authority(doc)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("NEXT100-036 TERMINAL REJECT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
