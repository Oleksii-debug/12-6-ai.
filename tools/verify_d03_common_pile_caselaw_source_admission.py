#!/usr/bin/env python3
"""Fail-closed source-admission authority for the bounded D03 Caselaw candidate."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CONFIG = Path("configs/data/d03_common_pile_caselaw_source_admission_v1.json")
SCHEMA = "12-6.d03-common-pile-caselaw-source-admission.v1"
REGISTRY_ID = "COMMON-PILE-SOURCE-RIGHTS-V1"
GENERIC_SOURCE_KEY = "caselaw_access_project"
GENERIC_RIGHTS_HEAD = "327a5364f7729f3ffdfc2f8079d02fb7a54638d2"
PRODUCT_PR = 904
PRODUCT_HEAD = "e63f2fd4eddbd348b57a42e1fb41cb658d0c3a82"
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


class AdmissionError(RuntimeError):
    """Fail-closed policy or authority error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _expected_binding() -> dict[str, Any]:
    return {
        "product_pr": PRODUCT_PR,
        "materializer_product_head_sha": PRODUCT_HEAD,
        "dataset": DATASET,
        "revision": REVISION,
        "ordering_policy": ORDERING_POLICY,
        "objects": [dict(item) for item in SOURCE_OBJECTS],
        "family": FAMILY,
    }


def _validate_generic_registry(policy: Mapping[str, Any]) -> dict[str, Any]:
    authority = policy["project_authority"]
    require(authority["generic_rights_pr"] == 769, "generic rights PR drift")
    require(
        authority["generic_rights_head_sha"] == GENERIC_RIGHTS_HEAD,
        "generic rights head drift",
    )
    require(authority["generic_registry_id"] == REGISTRY_ID, "registry id binding drift")
    require(
        authority["generic_source_key"] == GENERIC_SOURCE_KEY,
        "generic source key binding drift",
    )
    path = Path(authority["generic_registry_path"])
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("cannot load generic Common Pile rights registry") from exc
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
    require(row.get("credited_bytes") == 0, "generic registry credited bytes")
    require(row.get("authorized_loss_positions") == 0, "generic registry credited loss")
    require(row.get("evaluation_role") == "TRAINING_CANDIDATE_ONLY", "evaluation role drift")
    require(row.get("final_test_excluded") is True, "final-test boundary drift")
    return row


def load_policy(path: Path = CONFIG) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot load source-admission policy: {path}") from exc
    require(isinstance(policy, dict), "policy root must be object")
    require(policy.get("schema_version") == SCHEMA, "source-admission schema drift")
    require(policy.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary drift")
    require(policy.get("candidate_binding") == _expected_binding(), "candidate identity drift")

    admission = policy["admission_policy"]
    require(
        admission.get("decision_class") == "CONDITIONAL_CAP_RECORD_SOURCE_ADMISSION",
        "decision class drift",
    )
    require(
        admission.get("common_pile_license_metadata_sufficient_alone") is False,
        "license-laundering guard disabled",
    )
    require(admission.get("accepted_source_exact") == ACCEPTED_SOURCE, "accepted source drift")
    require(
        admission.get("required_metadata_license_exact") == REQUIRED_LICENSE,
        "required license drift",
    )
    require(admission.get("required_url_scheme") == "https", "URL scheme policy drift")
    require(
        tuple(admission.get("allowed_metadata_url_hosts", [])) == ALLOWED_HOSTS,
        "allowed CAP hosts drift",
    )
    require(
        tuple(admission.get("explicitly_not_source_admitted", [])) == DENIED_SOURCE_LABELS,
        "denied source labels drift",
    )
    require(
        tuple(admission.get("forbidden_editorial_metadata_keys", []))
        == FORBIDDEN_EDITORIAL_KEYS,
        "editorial exclusion drift",
    )
    require(admission.get("legal_conclusion_claimed") is False, "legal conclusion boundary drift")
    require(admission.get("record_level_policy_required") is True, "record policy disabled")
    require(admission.get("payload_source_admission_executed") is False, "payload execution fabricated")
    require(admission.get("admitted_payload_records") == 0, "payload records self-credited")
    require(admission.get("admitted_payload_bytes") == 0, "payload bytes self-credited")

    truth = policy["truth_boundary"]
    require(truth.get("source_policy_review_complete") is True, "policy review not terminal")
    for key in ZERO_KEYS:
        require(truth.get(key) == 0, f"zero-credit boundary drift: {key}")
    for key in FALSE_KEYS:
        require(truth.get(key) is False, f"false truth boundary drift: {key}")
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
