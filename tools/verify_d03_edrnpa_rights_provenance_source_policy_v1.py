#!/usr/bin/env python3
"""Fail-closed EDRNPA source-level rights/provenance policy authority.

This package qualifies only a conservative source policy for the exact
already-materialized PR #910 EDRNPA candidate. It deliberately does not admit
any payload record, corpus byte, tokenizer input, optimized target, or model
training step. Record-level act-card provenance must be executed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.d03-edrnpa-rights-provenance-source-policy.v1"
POLICY_PATH = Path("configs/data/d03_edrnpa_rights_provenance_source_policy_v1.json")

PARENT_PRODUCT_HEAD = "53d7375a87fdd73c8035c2277f9d260b9a8095c6"
PARENT_PRODUCT_PATH = "tools/probe_d03_edrnpa_open_data_v1.py"
PARENT_PRODUCT_BLOB = "d08ac8d6b0c6b2497c6e511ee2fc27046532d8fd"
EXECUTION_HEAD = "37de5af9d983961a324b3c49b2903156a69e1145"
WORKFLOW_RUN = 34792525578
WORKFLOW_JOB = 103819309604

EXPECTED_AUTHORITY = {
    "swarm_control_issue": 723,
    "scientific_control_issue": 548,
    "ownership_issue": 2357,
    "parent_product_pr": 910,
    "parent_product_head_sha": PARENT_PRODUCT_HEAD,
    "parent_product_path": PARENT_PRODUCT_PATH,
    "parent_product_git_blob_sha1": PARENT_PRODUCT_BLOB,
    "physical_execution_head_sha": EXECUTION_HEAD,
    "physical_execution_workflow_run": WORKFLOW_RUN,
    "physical_execution_workflow_job": WORKFLOW_JOB,
}
EXPECTED_BINDING = {
    "dataset_id": "c98e830c-e39e-4da6-a13c-f9ba32a79bec",
    "resource_id": "5616dd04-949a-489c-8efc-54004293b238",
    "resource_updated": "2026-09-08T15:02:00+03:00",
    "resource_md5": "0ea96e1582e5584ced79be1027f0ae55",
    "source_sha256": "a87b23eff3aadd2ff2dbed24cdeaaaf37862ecebe65b5f1edad02d5a95cde431",
    "source_bytes": 611865397,
    "family_id": "ua.minjust.edrnpa.open-data",
    "selected_objects": 176,
    "selected_bytes": 4000000,
    "inventory_identity_sha256": "380939a9ef8f67dcfb55912c010941f08d35b868267116d927603e988e710cca",
    "selected_payload_identity_sha256": "9b1c7b65a71f36ebd3c99273abf3e9f4c3c1885c96432d65761275e37e0b4435",
    "report_identity_sha256": "f5668ad4ddb3ce7ed0cd0130377f803ddbcebfb5b547c6757176efa8043122a6",
    "dataset_page": "https://data.gov.ua/dataset/c98e830c-e39e-4da6-a13c-f9ba32a79bec",
    "resource_page": (
        "https://data.gov.ua/dataset/c98e830c-e39e-4da6-a13c-f9ba32a79bec/"
        "resource/5616dd04-949a-489c-8efc-54004293b238"
    ),
    "nais_metadata_page": (
        "https://nais.gov.ua/m/ediniy-derjavniy-reestr-normativno-pravovih-aktiv-196"
    ),
}
EXPECTED_EVIDENCE = [
    {
        "authority": "Єдиний державний веб-портал відкритих даних України",
        "title": "Єдиний державний реєстр нормативно-правових актів",
        "url": EXPECTED_BINDING["dataset_page"],
        "accessed_utc": "2026-09-29",
        "supported_fact": (
            "The Ministry of Justice dataset contains a normative-act card and "
            "normative-act text; the portal states that open data may be freely copied, "
            "published, distributed and used including commercially with mandatory source "
            "attribution, and labels the dataset Creative Commons Attribution."
        ),
    },
    {
        "authority": "Єдиний державний веб-портал відкритих даних України",
        "title": "25-edrnpa resource history",
        "url": EXPECTED_BINDING["resource_page"],
        "accessed_utc": "2026-09-29",
        "supported_fact": (
            "The resource history still lists the exact 2026-09-08 15:02 EEST version "
            "with MD5 0ea96e1582e5584ced79be1027f0ae55; newer resource versions exist "
            "and are not silently substituted for the pinned physical execution."
        ),
    },
    {
        "authority": "ДП НАІС",
        "title": "Єдиний державний реєстр нормативно-правових актів",
        "url": EXPECTED_BINDING["nais_metadata_page"],
        "accessed_utc": "2026-09-29",
        "supported_fact": (
            "NAIS identifies this as register dataset number 25, describes it as "
            "containing a normative-act card and normative-act text, states the language "
            "is Ukrainian, the data format is XML, and compression is ZIP."
        ),
    },
]
EXPECTED_POLICY = {
    "decision_class": "CONDITIONAL_UA_EDRNPA_NORMATIVE_TEXT_SOURCE_POLICY",
    "dataset_open_data_reuse_supported": True,
    "dataset_license_alone_sufficient_for_payload_admission": False,
    "requires_exact_pinned_resource": True,
    "requires_selected_payload_identity": True,
    "requires_record_level_act_card_binding": True,
    "requires_official_normative_act_classification": True,
    "blanket_official_document_status_for_selected_objects": False,
    "embedded_third_party_content_source_admitted": False,
    "attachment_or_media_payload_source_admitted": False,
    "ambiguous_provenance_source_admitted": False,
    "unknown_act_class_source_admitted": False,
    "record_level_provenance_classification_required": True,
    "legal_conclusion_claimed": False,
    "payload_source_admission_executed": False,
    "admitted_payload_records": 0,
    "admitted_payload_bytes": 0,
}
EXPECTED_TRUTH = {
    "source_policy_review_complete": True,
    "parent_materialization_reference_bound": True,
    "record_level_payload_provenance_executed": False,
    "canonical_corpus_admitted": False,
    "canonical_capacity_credit_bytes": 0,
    "family_count_credit_added": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "training_eligible": False,
    "evaluation_eligible": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "learned_weights_created": False,
    "final_test_accessed": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
}
EXPECTED_DOWNSTREAM = [
    "record_level_object_rights_and_provenance_classification",
    "global_cross_source_dedup",
    "fresh_reserved_evaluation_decontamination",
    "post_composition_quality_and_privacy",
    "balance_and_family_caps",
    "cluster_safe_split",
    "deterministic_packing_and_two_clean_builds",
    "positive_exact_unique_loss_accounting",
    "tokenizer_fit_authority",
    "training_authority",
]

ROOT_KEYS = {
    "schema_version",
    "execution_profile",
    "project_authority",
    "candidate_binding",
    "primary_public_evidence",
    "admission_policy",
    "truth_boundary",
    "downstream_required",
}
AUTHORITY_KEYS = set(EXPECTED_AUTHORITY)
BINDING_KEYS = set(EXPECTED_BINDING)
EVIDENCE_KEYS = {"authority", "title", "url", "accessed_utc", "supported_fact"}
POLICY_KEYS = set(EXPECTED_POLICY)
TRUTH_KEYS = set(EXPECTED_TRUTH)
NUMERIC_AUTHORITY_KEYS = {
    "swarm_control_issue",
    "scientific_control_issue",
    "ownership_issue",
    "parent_product_pr",
    "physical_execution_workflow_run",
    "physical_execution_workflow_job",
}
NUMERIC_BINDING_KEYS = {"source_bytes", "selected_objects", "selected_bytes"}
NUMERIC_POLICY_KEYS = {"admitted_payload_records", "admitted_payload_bytes"}
NUMERIC_TRUTH_KEYS = {
    "canonical_capacity_credit_bytes",
    "family_count_credit_added",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "authorized_optimized_target_exposure",
    "optimizer_updates",
}


class EdrnpaSourcePolicyError(ValueError):
    """Raised when the exact EDRNPA source-policy authority fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EdrnpaSourcePolicyError(message)


def _require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} keys must be exact")
    return value


def _require_int(value: object, *, label: str, expected: int) -> None:
    _require(type(value) is int, f"{label} must be a strict integer")
    _require(value == expected, f"{label} drift")


def _require_bool(value: object, *, label: str, expected: bool) -> None:
    _require(type(value) is bool, f"{label} must be boolean")
    _require(value is expected, f"{label} drift")


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw, usedforsecurity=False).hexdigest()


def _git_show_bytes(root: Path, revision: str, relpath: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relpath}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise EdrnpaSourcePolicyError(
            f"cannot resolve parent Product authority {revision}:{relpath}"
        ) from exc
    return result.stdout


def _validate_parent_product(root: Path) -> None:
    raw = _git_show_bytes(root, PARENT_PRODUCT_HEAD, PARENT_PRODUCT_PATH)
    _require(
        _git_blob_sha1(raw) == PARENT_PRODUCT_BLOB,
        "parent Product Git blob mismatch",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EdrnpaSourcePolicyError("parent Product is not UTF-8") from exc

    required_literals = {
        'DATASET_ID = "c98e830c-e39e-4da6-a13c-f9ba32a79bec"',
        'RESOURCE_ID = "5616dd04-949a-489c-8efc-54004293b238"',
        'RESOURCE_UPDATED = "2026-09-08T15:02:00+03:00"',
        'RESOURCE_MD5 = "0ea96e1582e5584ced79be1027f0ae55"',
        'FAMILY_ID = "ua.minjust.edrnpa.open-data"',
        '"object_level_rights_and_provenance_retest_required": True',
        '"training_rights_admitted_by_this_probe": False',
        '"training_eligible": False',
        '"evaluation_eligible": False',
    }
    for literal in required_literals:
        _require(literal in text, f"parent Product contract drift: {literal}")


def _validate_exact_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    numeric_keys: set[str],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        value = actual[key]
        if key in numeric_keys:
            _require_int(value, label=f"{label}.{key}", expected=expected_value)
        elif type(expected_value) is bool:
            _require_bool(value, label=f"{label}.{key}", expected=expected_value)
        else:
            _require(type(value) is type(expected_value), f"{label}.{key} type drift")
            _require(value == expected_value, f"{label}.{key} drift")


def validate_policy(
    policy: Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_parent_product: bool = True,
) -> dict[str, Any]:
    root_map = _require_exact_keys(policy, ROOT_KEYS, "policy root")
    _require(root_map["schema_version"] == SCHEMA, "unexpected schema")
    _require(root_map["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    authority = _require_exact_keys(
        root_map["project_authority"], AUTHORITY_KEYS, "project_authority"
    )
    _validate_exact_mapping(
        authority,
        EXPECTED_AUTHORITY,
        numeric_keys=NUMERIC_AUTHORITY_KEYS,
        label="project_authority",
    )

    binding = _require_exact_keys(
        root_map["candidate_binding"], BINDING_KEYS, "candidate_binding"
    )
    _validate_exact_mapping(
        binding,
        EXPECTED_BINDING,
        numeric_keys=NUMERIC_BINDING_KEYS,
        label="candidate_binding",
    )

    evidence = root_map["primary_public_evidence"]
    _require(type(evidence) is list, "primary_public_evidence must be a list")
    _require(len(evidence) == len(EXPECTED_EVIDENCE), "primary evidence count drift")
    for index, (actual, expected) in enumerate(
        zip(evidence, EXPECTED_EVIDENCE, strict=True)
    ):
        item = _require_exact_keys(
            actual, EVIDENCE_KEYS, f"primary_public_evidence[{index}]"
        )
        _require(item == expected, f"primary_public_evidence[{index}] drift")

    admission = _require_exact_keys(
        root_map["admission_policy"], POLICY_KEYS, "admission_policy"
    )
    _validate_exact_mapping(
        admission,
        EXPECTED_POLICY,
        numeric_keys=NUMERIC_POLICY_KEYS,
        label="admission_policy",
    )

    truth = _require_exact_keys(root_map["truth_boundary"], TRUTH_KEYS, "truth_boundary")
    _validate_exact_mapping(
        truth,
        EXPECTED_TRUTH,
        numeric_keys=NUMERIC_TRUTH_KEYS,
        label="truth_boundary",
    )

    downstream = root_map["downstream_required"]
    _require(type(downstream) is list, "downstream_required must be a list")
    _require(downstream == EXPECTED_DOWNSTREAM, "downstream_required drift")

    if verify_parent_product:
        repo_root = root or Path(__file__).resolve().parents[1]
        _validate_parent_product(repo_root)

    return dict(root_map)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EdrnpaSourcePolicyError("cannot load EDRNPA source policy") from exc
    _require(type(payload) is dict, "policy root must be an object")
    assert isinstance(payload, dict)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    validated = validate_policy(load_policy(args.policy), root=args.root)
    summary = {
        "schema_version": validated["schema_version"],
        "decision_class": validated["admission_policy"]["decision_class"],
        "source_policy_review_complete": validated["truth_boundary"][
            "source_policy_review_complete"
        ],
        "payload_source_admission_executed": validated["admission_policy"][
            "payload_source_admission_executed"
        ],
        "admitted_payload_bytes": validated["admission_policy"]["admitted_payload_bytes"],
        "authorized_optimized_target_exposure": validated["truth_boundary"][
            "authorized_optimized_target_exposure"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
