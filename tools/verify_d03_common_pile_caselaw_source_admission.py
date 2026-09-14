#!/usr/bin/env python3
"""Fail-closed source-admission authority for the bounded D03 Caselaw candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CONFIG = Path("configs/data/d03_common_pile_caselaw_source_admission_v1.json")
SCHEMA = "12-6.d03-common-pile-caselaw-source-admission.v1"
SWARM_CONTROL_ISSUE = 723
ORIGINAL_TAKEOVER_ISSUE = 1075
REPAIR_ISSUE = 1209
SUPERSEDED_CLAIM_ISSUE = 982
GENERIC_RIGHTS_PR = 769
GENERIC_RIGHTS_HEAD = "327a5364f7729f3ffdfc2f8079d02fb7a54638d2"
GENERIC_REGISTRY_PATH = Path("configs/data/common_pile_source_rights_v1.json")
GENERIC_REGISTRY_BLOB_SHA1 = "7b4d6828288672bf25c551e85a5d7f7399e8ef0f"
REGISTRY_ID = "COMMON-PILE-SOURCE-RIGHTS-V1"
GENERIC_SOURCE_KEY = "caselaw_access_project"
PRODUCT_PR = 904
PRODUCT_SEMANTIC_COMMIT = "e63f2fd4eddbd348b57a42e1fb41cb658d0c3a82"
PRODUCT_CONFIG_PATH = "configs/data/d03_common_pile_caselaw_zero_credit_v1.json"
PRODUCT_CONFIG_BLOB_SHA1 = "e8b7e29ce2ab4e634c5f8c056eecb6b0c2579467"
PRODUCT_MATERIALIZER_PATH = "tools/materialize_d03_common_pile_caselaw.py"
PRODUCT_MATERIALIZER_BLOB_SHA1 = "a7fcaeee0302c46c43e5a6ae43da16363810f14a"
DATASET = "common-pile/caselaw_access_project"
REVISION = "31e65135501af50f8285b52489bc3b39fd0fc5d5"
ORDERING_POLICY = "PRESERVE_ORIGINAL_CAP_00044_THEN_APPEND_CAP_00043"
SOURCE_OBJECTS = (
    {
        "file": "cap_00044.jsonl.gz",
        "bytes": 9_441_419,
        "sha256": "e04bdff817b34d5fd2ab0b4aff272adfe6e863aacef7d4cd24db47e4b7c8db33",
    },
    {
        "file": "cap_00043.jsonl.gz",
        "bytes": 10_476_512,
        "sha256": "f299c45effc957e1e02d3930e68c5d6dc643eb9d585093ff7445f36095c8531f",
    },
)
FAMILY = "en.common-pile.caselaw"
ACCEPTED_SOURCE = "Caselaw Access Project"
REQUIRED_LICENSE = "Public Domain"
ALLOWED_HOSTS = ("case.law", "static.case.law")
DENIED_SOURCE_LABELS = ("Court Listener", "CourtListener")
FORBIDDEN_EDITORIAL_KEYS = (
    "annotation",
    "annotations",
    "editorial",
    "editorial_summary",
    "headnote",
    "headnotes",
    "key_number",
    "syllabus",
)
RECORD_FIELDS = ("id", "source", "added", "created", "metadata", "text")
ZERO_KEYS = (
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "unique_causal_loss_positions_authorized",
    "optimizer_updates",
)
FALSE_KEYS = (
    "tokenizer_fit_authorized",
    "training_eligible",
    "evaluation_eligible",
    "model_training_executed",
    "final_test_accessed",
    "paid_compute_used",
)
ROOT_KEYS = {
    "schema_version",
    "execution_profile",
    "project_authority",
    "candidate_binding",
    "primary_public_evidence",
    "admission_policy",
    "truth_boundary",
}
PROJECT_AUTHORITY_KEYS = {
    "swarm_control_issue",
    "takeover_issue",
    "repair_issue",
    "superseded_expired_claim_issue",
    "generic_rights_pr",
    "generic_rights_head_sha",
    "generic_registry_path",
    "generic_registry_git_blob_sha1",
    "generic_registry_id",
    "generic_source_key",
}
ADMISSION_POLICY_KEYS = {
    "decision_class",
    "common_pile_license_metadata_sufficient_alone",
    "accepted_source_exact",
    "required_metadata_license_exact",
    "required_url_scheme",
    "allowed_metadata_url_hosts",
    "explicitly_not_source_admitted",
    "forbidden_editorial_metadata_keys",
    "legal_conclusion_claimed",
    "record_level_policy_required",
    "payload_source_admission_executed",
    "admitted_payload_records",
    "admitted_payload_bytes",
}
TRUTH_BOUNDARY_KEYS = {
    "source_policy_review_complete",
    *ZERO_KEYS,
    *FALSE_KEYS,
}
PUBLIC_EVIDENCE = (
    {
        "authority": "Harvard Caselaw Access Project",
        "title": "Terms of Use",
        "url": "https://case.law/terms/",
        "accessed_utc": "2026-09-10",
        "supported_fact": (
            "CAP states that Caselaw Data and metadata made available on its site are made "
            "available under CC0 1.0; the terms also disclaim warranties and "
            "third-party-rights representations."
        ),
    },
    {
        "authority": "Common Pile",
        "title": "caselaw_access_project dataset card",
        "url": (
            "https://huggingface.co/datasets/common-pile/caselaw_access_project/"
            "blob/main/README.md"
        ),
        "accessed_utc": "2026-09-10",
        "supported_fact": (
            "The source card describes public-domain filtering and also warns that license "
            "laundering or inaccurate metadata can yield incorrect license assignments."
        ),
    },
    {
        "authority": "Common Pile",
        "title": "caselaw_access_project dataset",
        "url": "https://huggingface.co/datasets/common-pile/caselaw_access_project",
        "accessed_utc": "2026-09-10",
        "supported_fact": (
            "The dataset surface exposes id/source/metadata/text-style provenance fields "
            "used by the project record contract."
        ),
    },
    {
        "authority": "Free Law Project",
        "title": "CourtListener.com Terms of Service and Policies",
        "url": (
            "https://wiki.free.law/c/terms/courtlistener/"
            "courtlistenercom-terms-of-service-and-policies"
        ),
        "accessed_utc": "2026-09-10",
        "supported_fact": (
            "Court materials can include third-party copyrighted works, so CourtListener "
            "provenance is not treated as blanket source admission by this project."
        ),
    },
)


class AdmissionError(RuntimeError):
    """Fail-closed policy or authority error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def require_mapping(value: object, message: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), message)
    return value  # type: ignore[return-value]


def require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    require(set(value) == expected, f"{label} key drift")


def require_exact_zero(value: object, label: str) -> None:
    require(type(value) is int and value == 0, f"numeric zero boundary drift: {label}")


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _expected_binding() -> dict[str, Any]:
    return {
        "product_pr": PRODUCT_PR,
        "product_semantic_commit_sha": PRODUCT_SEMANTIC_COMMIT,
        "product_config_path": PRODUCT_CONFIG_PATH,
        "product_config_git_blob_sha1": PRODUCT_CONFIG_BLOB_SHA1,
        "product_materializer_path": PRODUCT_MATERIALIZER_PATH,
        "product_materializer_git_blob_sha1": PRODUCT_MATERIALIZER_BLOB_SHA1,
        "dataset": DATASET,
        "revision": REVISION,
        "ordering_policy": ORDERING_POLICY,
        "objects": [dict(item) for item in SOURCE_OBJECTS],
        "family": FAMILY,
    }


def _validate_project_authority(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    authority = require_mapping(policy.get("project_authority"), "project authority missing")
    require_exact_keys(authority, PROJECT_AUTHORITY_KEYS, "project authority")
    require(authority["swarm_control_issue"] == SWARM_CONTROL_ISSUE, "swarm control drift")
    require(authority["takeover_issue"] == ORIGINAL_TAKEOVER_ISSUE, "takeover issue drift")
    require(authority["repair_issue"] == REPAIR_ISSUE, "repair issue drift")
    require(
        authority["superseded_expired_claim_issue"] == SUPERSEDED_CLAIM_ISSUE,
        "superseded claim drift",
    )
    require(authority["generic_rights_pr"] == GENERIC_RIGHTS_PR, "generic rights PR drift")
    require(
        authority["generic_rights_head_sha"] == GENERIC_RIGHTS_HEAD,
        "generic rights head drift",
    )
    require(
        authority["generic_registry_path"] == str(GENERIC_REGISTRY_PATH),
        "generic registry path drift",
    )
    require(
        authority["generic_registry_git_blob_sha1"] == GENERIC_REGISTRY_BLOB_SHA1,
        "generic registry blob binding drift",
    )
    require(authority["generic_registry_id"] == REGISTRY_ID, "registry id binding drift")
    require(
        authority["generic_source_key"] == GENERIC_SOURCE_KEY,
        "generic source key binding drift",
    )
    return authority


def _validate_generic_registry(policy: Mapping[str, Any]) -> dict[str, Any]:
    _validate_project_authority(policy)
    try:
        raw = GENERIC_REGISTRY_PATH.read_bytes()
    except OSError as exc:
        raise AdmissionError("cannot load generic Common Pile rights registry") from exc
    require(
        git_blob_sha1(raw) == GENERIC_REGISTRY_BLOB_SHA1,
        "generic registry blob identity drift",
    )
    try:
        registry = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("cannot parse generic Common Pile rights registry") from exc
    require(isinstance(registry, Mapping), "generic registry root drift")
    require(registry.get("registry_id") == REGISTRY_ID, "generic registry drift")
    sources = registry.get("sources")
    require(isinstance(sources, list), "generic rights source vector missing")
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping) and item.get("key") == GENERIC_SOURCE_KEY
    ]
    require(len(matches) == 1, "generic Caselaw source missing or ambiguous")
    row = dict(matches[0])
    require(row.get("hf_dataset") == DATASET, "generic dataset drift")
    require(row.get("rights_basis_class") == "PUBLIC_DOMAIN_FILTER", "rights basis drift")
    require(row.get("project_review_status") == "REVIEW_REQUIRED", "generic review drift")
    require(row.get("canonical_training_authorized") is False, "generic registry self-authorized")
    require_exact_zero(row.get("credited_bytes"), "generic credited_bytes")
    require_exact_zero(row.get("authorized_loss_positions"), "generic authorized_loss_positions")
    require(row.get("evaluation_role") == "TRAINING_CANDIDATE_ONLY", "evaluation role drift")
    require(row.get("final_test_excluded") is True, "final-test boundary drift")
    return row


def load_policy(path: Path = CONFIG) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot load source-admission policy: {path}") from exc
    require(isinstance(policy, dict), "policy root must be object")
    require_exact_keys(policy, ROOT_KEYS, "policy root")
    require(policy["schema_version"] == SCHEMA, "source-admission schema drift")
    require(policy["execution_profile"] == "LOCAL_FREE", "LOCAL_FREE boundary drift")
    _validate_project_authority(policy)

    binding = require_mapping(policy["candidate_binding"], "candidate binding missing")
    require(dict(binding) == _expected_binding(), "candidate identity drift")

    evidence = policy["primary_public_evidence"]
    require(isinstance(evidence, list), "public evidence vector missing")
    require(evidence == [dict(item) for item in PUBLIC_EVIDENCE], "public evidence drift")

    admission = require_mapping(policy["admission_policy"], "admission policy missing")
    require_exact_keys(admission, ADMISSION_POLICY_KEYS, "admission policy")
    require(
        admission["decision_class"] == "CONDITIONAL_CAP_RECORD_SOURCE_ADMISSION",
        "decision class drift",
    )
    require(
        admission["common_pile_license_metadata_sufficient_alone"] is False,
        "license-laundering guard disabled",
    )
    require(admission["accepted_source_exact"] == ACCEPTED_SOURCE, "accepted source drift")
    require(
        admission["required_metadata_license_exact"] == REQUIRED_LICENSE,
        "required license drift",
    )
    require(admission["required_url_scheme"] == "https", "URL scheme policy drift")
    require(
        tuple(admission["allowed_metadata_url_hosts"]) == ALLOWED_HOSTS,
        "allowed CAP hosts drift",
    )
    require(
        tuple(admission["explicitly_not_source_admitted"]) == DENIED_SOURCE_LABELS,
        "denied source labels drift",
    )
    require(
        tuple(admission["forbidden_editorial_metadata_keys"]) == FORBIDDEN_EDITORIAL_KEYS,
        "editorial exclusion drift",
    )
    require(admission["legal_conclusion_claimed"] is False, "legal conclusion boundary drift")
    require(admission["record_level_policy_required"] is True, "record policy disabled")
    require(
        admission["payload_source_admission_executed"] is False,
        "payload execution fabricated",
    )
    require_exact_zero(admission["admitted_payload_records"], "admitted_payload_records")
    require_exact_zero(admission["admitted_payload_bytes"], "admitted_payload_bytes")

    truth = require_mapping(policy["truth_boundary"], "truth boundary missing")
    require_exact_keys(truth, TRUTH_BOUNDARY_KEYS, "truth boundary")
    require(truth["source_policy_review_complete"] is True, "policy review not terminal")
    for key in ZERO_KEYS:
        require_exact_zero(truth[key], key)
    for key in FALSE_KEYS:
        require(truth[key] is False, f"false truth boundary drift: {key}")
    _validate_generic_registry(policy)
    return policy


def validate_candidate_identity(identity: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    require(dict(identity) == policy["candidate_binding"], "candidate identity substitution")


def assess_record(record: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    require(set(record) == set(RECORD_FIELDS), "record field drift")
    source = record.get("source")
    if source in DENIED_SOURCE_LABELS:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "unsupported_courtlistener_source"}
    if source != ACCEPTED_SOURCE:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "source_not_exact_cap"}

    metadata = record.get("metadata")
    require(isinstance(metadata, Mapping), "metadata missing")
    if metadata.get("license") != REQUIRED_LICENSE:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "license_not_exact_public_domain"}
    metadata_keys = {str(key).lower() for key in metadata}
    if metadata_keys.intersection(FORBIDDEN_EDITORIAL_KEYS):
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "editorial_metadata_present"}

    url = metadata.get("url")
    if not isinstance(url, str):
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "cap_url_missing"}
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "cap_url_authority_mismatch"}
    if parsed.username is not None or parsed.password is not None or parsed.port is not None:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "cap_url_authority_mismatch"}

    return {
        "decision": "CONDITIONAL_SOURCE_ADMISSION",
        "reason": "exact_cap_source_license_and_authority",
        "training_authorized_bytes": 0,
        "canonical_capacity_credited": 0,
        "legal_conclusion_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=CONFIG)
    parser.add_argument("--record-json", type=Path)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    result: dict[str, Any] = {
        "policy": "VALID",
        "decision_class": policy["admission_policy"]["decision_class"],
        "training_authorized_bytes": 0,
    }
    if args.record_json is not None:
        try:
            record = json.loads(args.record_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdmissionError("cannot load record JSON") from exc
        require(isinstance(record, Mapping), "record JSON must be object")
        result["record"] = assess_record(record, policy)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
