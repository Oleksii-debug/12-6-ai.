#!/usr/bin/env python3
"""Fail-closed source-rights authority for the bounded Common Pile ArXiv abstracts candidate."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from twelve_six.common_pile_rights import load_and_validate as load_generic_rights

CONFIG = REPO_ROOT / "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json"
SCHEMA = "12-6.d03-common-pile-arxiv-abstracts-source-admission.v1"
REGISTRY_ID = "COMMON-PILE-SOURCE-RIGHTS-V1"
REGISTRY_IDENTITY = "b279c4a7404e0e501acd0c77842d1f71c43ea1dbcd611acab18163c64b060d4e"
GENERIC_SOURCE_KEY = "arxiv_abstracts"
GENERIC_RIGHTS_HEAD = "327a5364f7729f3ffdfc2f8079d02fb7a54638d2"
MATERIALIZER_PR = 879
MATERIALIZER_HEAD = "3a238e461adfacf5b87d6a7078bf51462825a268"
MATERIALIZER_MERGE = "32ca583e5377991bd5c0bb259436266e95e9c0d6"
DATASET = "common-pile/arxiv_abstracts"
REVISION = "46de78c48636c0b46f60049dfd1c5a3710d233f9"
SOURCE_FILE = "00003_arxiv-abstracts.jsonl.gz"
SOURCE_BYTES = 232_616_926
SOURCE_SHA256 = "3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7"
SOURCE_LABEL = "arxiv-abstracts"
RECORD_FIELDS = ("id", "text", "source", "created", "added", "metadata")
REQUIRED_LICENSE = (
    "Creative Commons Zero - Public Domain - "
    "https://creativecommons.org/publicdomain/zero/1.0/"
)
DECISION_CLASS = "CONDITIONAL_ARXIV_METADATA_CC0_SOURCE_ADMISSION"

TOP_LEVEL_KEYS = {
    "schema_version",
    "execution_profile",
    "project_authority",
    "candidate_binding",
    "primary_public_evidence",
    "admission_policy",
    "truth_boundary",
}
PROJECT_KEYS = {
    "swarm_control_issue",
    "claim_issue",
    "generic_rights_pr",
    "generic_rights_head_sha",
    "generic_registry_path",
    "generic_registry_id",
    "generic_registry_identity_sha256",
    "generic_source_key",
}
CANDIDATE_KEYS = {
    "materializer_pr",
    "materializer_product_head_sha",
    "materializer_merge_commit_sha",
    "dataset",
    "revision",
    "file",
    "bytes",
    "sha256",
    "source_label",
    "record_fields",
    "required_metadata_license_exact",
}
EVIDENCE_KEYS = {"authority", "title", "url", "accessed_utc", "supported_fact"}
ADMISSION_KEYS = {
    "decision_class",
    "project_source_policy_review_complete",
    "legal_conclusion_claimed",
    "common_pile_license_metadata_sufficient_alone",
    "primary_arxiv_policy_evidence_required",
    "record_level_policy_required",
    "accepted_source_exact",
    "required_metadata_license_exact",
    "abstract_is_arxiv_metadata_required_field",
    "arxiv_all_metadata_cc0",
    "pinned_common_pile_harvest_method",
    "payload_source_admission_executed",
    "admitted_payload_records",
    "admitted_payload_bytes",
}
TRUTH_KEYS = {
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "unique_causal_loss_positions_authorized",
    "tokenizer_fit_authorized",
    "training_eligible",
    "evaluation_eligible",
    "optimizer_updates",
    "model_training_executed",
    "learned_weights_created",
    "final_test_accessed",
    "paid_compute_used",
    "foreign_pretrained_weights_used",
}
ZERO_TRUTH_KEYS = (
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "unique_causal_loss_positions_authorized",
    "optimizer_updates",
)
FALSE_TRUTH_KEYS = (
    "tokenizer_fit_authorized",
    "training_eligible",
    "evaluation_eligible",
    "model_training_executed",
    "learned_weights_created",
    "final_test_accessed",
    "paid_compute_used",
    "foreign_pretrained_weights_used",
)

EXPECTED_EVIDENCE = [
    {
        "authority": "arXiv",
        "title": "arXiv License Information",
        "url": "https://info.arxiv.org/help/license/index.html",
        "accessed_utc": "2026-09-11",
        "supported_fact": "arXiv states that CC0 1.0 applies to all metadata.",
    },
    {
        "authority": "arXiv",
        "title": "Metadata for Required and Optional Fields",
        "url": "https://info.arxiv.org/help/prep.html",
        "accessed_utc": "2026-09-11",
        "supported_fact": "arXiv documents Abstract as a required metadata field.",
    },
    {
        "authority": "Common Pile",
        "title": (
            "ArXiv Abstracts dataset card at immutable revision "
            "46de78c48636c0b46f60049dfd1c5a3710d233f9"
        ),
        "url": (
            "https://huggingface.co/datasets/common-pile/arxiv_abstracts/blob/"
            "46de78c48636c0b46f60049dfd1c5a3710d233f9/README.md"
        ),
        "accessed_utc": "2026-09-11",
        "supported_fact": (
            "The pinned dataset card describes abstracts as arXiv metadata, says they are "
            "harvested through OAI-PMH and reproduced as-is, and warns that inaccurate "
            "licensing metadata can occur."
        ),
    },
]


class AdmissionError(RuntimeError):
    """Fail-closed policy or authority error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} key drift")


def require_zero_int(value: Any, label: str) -> None:
    require(type(value) is int and value == 0, f"{label} must be integer zero")


def expected_project_authority() -> dict[str, Any]:
    return {
        "swarm_control_issue": 723,
        "claim_issue": 1119,
        "generic_rights_pr": 769,
        "generic_rights_head_sha": GENERIC_RIGHTS_HEAD,
        "generic_registry_path": "configs/data/common_pile_source_rights_v1.json",
        "generic_registry_id": REGISTRY_ID,
        "generic_registry_identity_sha256": REGISTRY_IDENTITY,
        "generic_source_key": GENERIC_SOURCE_KEY,
    }


def expected_candidate_binding() -> dict[str, Any]:
    return {
        "materializer_pr": MATERIALIZER_PR,
        "materializer_product_head_sha": MATERIALIZER_HEAD,
        "materializer_merge_commit_sha": MATERIALIZER_MERGE,
        "dataset": DATASET,
        "revision": REVISION,
        "file": SOURCE_FILE,
        "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256,
        "source_label": SOURCE_LABEL,
        "record_fields": list(RECORD_FIELDS),
        "required_metadata_license_exact": REQUIRED_LICENSE,
    }


def validate_generic_registry(policy: Mapping[str, Any]) -> dict[str, Any]:
    authority = policy["project_authority"]
    path = (REPO_ROOT / str(authority["generic_registry_path"])).resolve()
    require(path.is_relative_to(REPO_ROOT), "generic registry escaped repository")
    try:
        registry = load_generic_rights(path)
    except Exception as exc:
        raise AdmissionError("generic Common Pile rights registry failed validation") from exc
    require(registry.get("registry_id") == REGISTRY_ID, "generic registry id drift")
    require(
        registry.get("registry_identity_sha256") == REGISTRY_IDENTITY,
        "generic registry identity drift",
    )
    sources = registry.get("sources")
    require(isinstance(sources, list), "generic rights source vector missing")
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping) and item.get("key") == GENERIC_SOURCE_KEY
    ]
    require(len(matches) == 1, "generic ArXiv abstracts source missing or ambiguous")
    row = dict(matches[0])
    require(row.get("hf_dataset") == DATASET, "generic dataset drift")
    require(row.get("rights_basis_class") == "METADATA_OPEN_LICENSE", "rights basis drift")
    signals = row.get("license_or_status_signals")
    require(isinstance(signals, list) and "CC0" in signals, "CC0 signal missing")
    require(row.get("project_review_status") == "REVIEW_REQUIRED", "generic review drift")
    require(row.get("canonical_training_authorized") is False, "generic registry self-authorized")
    require_zero_int(row.get("credited_bytes"), "generic credited_bytes")
    require_zero_int(row.get("authorized_loss_positions"), "generic authorized_loss_positions")
    require(row.get("evaluation_role") == "TRAINING_CANDIDATE_ONLY", "evaluation role drift")
    require(row.get("final_test_excluded") is True, "final-test boundary drift")
    return row


def load_policy(path: Path = CONFIG) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot load source-admission policy: {path}") from exc
    require(isinstance(policy, dict), "policy root must be object")
    require_exact_keys(policy, TOP_LEVEL_KEYS, "policy")
    require(policy.get("schema_version") == SCHEMA, "source-admission schema drift")
    require(policy.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary drift")

    project = policy.get("project_authority")
    require(isinstance(project, Mapping), "project_authority missing")
    require_exact_keys(project, PROJECT_KEYS, "project_authority")
    require(dict(project) == expected_project_authority(), "project authority drift")

    candidate = policy.get("candidate_binding")
    require(isinstance(candidate, Mapping), "candidate_binding missing")
    require_exact_keys(candidate, CANDIDATE_KEYS, "candidate_binding")
    require(dict(candidate) == expected_candidate_binding(), "candidate identity drift")

    evidence = policy.get("primary_public_evidence")
    require(isinstance(evidence, list), "primary evidence vector missing")
    require(len(evidence) == len(EXPECTED_EVIDENCE), "primary evidence count drift")
    for row in evidence:
        require(isinstance(row, Mapping), "primary evidence row must be object")
        require_exact_keys(row, EVIDENCE_KEYS, "primary evidence")
    require(evidence == EXPECTED_EVIDENCE, "primary evidence authority drift")

    admission = policy.get("admission_policy")
    require(isinstance(admission, Mapping), "admission_policy missing")
    require_exact_keys(admission, ADMISSION_KEYS, "admission_policy")
    require(admission.get("decision_class") == DECISION_CLASS, "decision class drift")
    require(admission.get("project_source_policy_review_complete") is True, "policy review incomplete")
    require(admission.get("legal_conclusion_claimed") is False, "legal conclusion boundary drift")
    require(
        admission.get("common_pile_license_metadata_sufficient_alone") is False,
        "Common Pile metadata may not self-authorize",
    )
    require(
        admission.get("primary_arxiv_policy_evidence_required") is True,
        "primary arXiv evidence requirement disabled",
    )
    require(admission.get("record_level_policy_required") is True, "record-level policy disabled")
    require(admission.get("accepted_source_exact") == SOURCE_LABEL, "accepted source drift")
    require(
        admission.get("required_metadata_license_exact") == REQUIRED_LICENSE,
        "required metadata license drift",
    )
    require(
        admission.get("abstract_is_arxiv_metadata_required_field") is True,
        "abstract metadata fact disabled",
    )
    require(admission.get("arxiv_all_metadata_cc0") is True, "arXiv metadata CC0 fact disabled")
    require(
        admission.get("pinned_common_pile_harvest_method") == "OAI_PMH_REPRODUCED_AS_IS",
        "pinned harvest provenance drift",
    )
    require(admission.get("payload_source_admission_executed") is False, "payload execution fabricated")
    require_zero_int(admission.get("admitted_payload_records"), "admitted_payload_records")
    require_zero_int(admission.get("admitted_payload_bytes"), "admitted_payload_bytes")

    truth = policy.get("truth_boundary")
    require(isinstance(truth, Mapping), "truth_boundary missing")
    require_exact_keys(truth, TRUTH_KEYS, "truth_boundary")
    for key in ZERO_TRUTH_KEYS:
        require_zero_int(truth.get(key), key)
    for key in FALSE_TRUTH_KEYS:
        require(truth.get(key) is False, f"false truth boundary drift: {key}")

    validate_generic_registry(policy)
    return policy


def validate_candidate_identity(identity: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    require(dict(identity) == policy["candidate_binding"], "candidate identity substitution")


def assess_record(record: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    require(set(record) == set(RECORD_FIELDS), "record field drift")
    if record.get("source") != SOURCE_LABEL:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "source_not_exact_arxiv_abstracts"}
    metadata = record.get("metadata")
    require(isinstance(metadata, Mapping), "metadata missing")
    if metadata.get("license") != REQUIRED_LICENSE:
        return {"decision": "NOT_SOURCE_ADMITTED", "reason": "metadata_license_not_exact_cc0"}
    require(
        policy["admission_policy"]["primary_arxiv_policy_evidence_required"] is True,
        "primary evidence not active",
    )
    return {
        "decision": "CONDITIONAL_SOURCE_ADMISSION",
        "reason": "exact_arxiv_abstract_metadata_cc0_policy",
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
        "canonical_capacity_credited": 0,
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
