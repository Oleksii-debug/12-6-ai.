#!/usr/bin/env python3
"""Fail-closed validator for the SWARM-747 CulturaX rights/provenance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


AUDIT_ID = "AUDIT-B-CULTURAX-UA-EN-RIGHTS-PROVENANCE-V1"
LANE_KEY = "AUDIT-B|CULTURAX|INDEPENDENT-VERIFY|UA-EN-RIGHTS-PROVENANCE-V1"
BASE_MAIN_SHA = "5020afd671a3885c1b738c8b4eafe7525f630546"
REGISTRY_BLOB_SHA = "d80a60357c56eacac135f948b8a72556bb849e5a"
REGISTRY_PATH = "configs/research/open_source_reuse_registry_v2.json"

EXPECTED_COMPONENT = {
    "id": "CULTURAX",
    "kind": "dataset_candidate",
    "upstream": "https://huggingface.co/datasets/uonlp/CulturaX",
    "license": "inherits mC4 and OSCAR terms",
    "decision": "P0_UA_EN_COMPARISON_ACQUISITION",
    "canonical_training_authorized": False,
    "value": "large cleaned 167-language web corpus with Ukrainian stratum",
}

REQUIRED_EVIDENCE_IDS = {
    "CULTURAX_DATASET_CARD",
    "MC4_DATASET_CARD",
    "OSCAR_2301_DATASET_CARD",
    "COMMON_CRAWL_TERMS",
}

REQUIRED_SUCCESSOR_FRAGMENTS = {
    "immutable CulturaX dataset revision",
    "file identities",
    "per-record source, url and timestamp",
    "mC4 versus each OSCAR release",
    "source-level rights/terms",
    "Common Crawl",
    "privacy/PII",
    "reserved-evaluation decontamination",
    "cross-source deduplication",
    "unique causal-loss positions",
    "two-clean-build determinism",
}


class AuditValidationError(ValueError):
    """The audit cannot be accepted as fail-closed evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditValidationError(message)


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324 - Git object identity.


def _component_by_id(registry: dict[str, Any], component_id: str) -> dict[str, Any]:
    components = registry.get("components")
    _require(isinstance(components, list), "registry.components must be a list")
    matches = [item for item in components if isinstance(item, dict) and item.get("id") == component_id]
    _require(len(matches) == 1, f"registry must contain exactly one {component_id} component")
    return matches[0]


def _all_zero_credit(payload: dict[str, Any]) -> None:
    credit = payload["project_training_credit"]
    _require(credit["canonical_training_authorized"] is False, "training authorization must remain false")
    for field in (
        "admitted_source_bytes",
        "admitted_tokenizer_tokens",
        "authorized_unique_causal_loss_positions",
    ):
        _require(credit[field] == 0, f"{field} must remain zero")
    _require(credit["bulk_download_performed"] is False, "bulk download must remain false")
    _require(credit["corpus_payload_accessed"] is False, "corpus payload access must remain false")

    languages = payload["upstream_observation"]["languages"]
    _require(set(languages) == {"en", "uk"}, "audit must cover exactly en and uk")
    for code in ("en", "uk"):
        language = languages[code]
        _require(language["config_present"] is True, f"{code} config must be recorded as present")
        _require(language["project_admitted_bytes"] == 0, f"{code} project_admitted_bytes must be zero")
        _require(
            language["project_authorized_unique_causal_loss_positions"] == 0,
            f"{code} authorized unique causal-loss positions must be zero",
        )


def _rights_are_fail_closed(payload: dict[str, Any]) -> None:
    rights = payload["rights_and_privacy_conclusions"]
    required_false = (
        "package_or_dataset_card_license_is_blanket_source_training_authority",
        "mc4_odc_by_is_sufficient_source_rights_by_itself",
        "oscar_cc0_metadata_packaging_is_crawled_text_ownership",
        "culturax_public_availability_is_training_authority",
        "reidentification_attempt_allowed",
    )
    for field in required_false:
        _require(rights[field] is False, f"{field} must be false")

    required_true = (
        "source_level_rights_review_required",
        "common_crawl_terms_required",
        "jurisdiction_and_use_case_review_required",
        "personal_or_sensitive_information_may_remain",
        "retain_url_timestamp_source_provenance_through_intake",
    )
    for field in required_true:
        _require(rights[field] is True, f"{field} must be true")


def _firewalls_are_intact(payload: dict[str, Any]) -> None:
    firewall = payload["evaluation_and_lineage_firewall"]
    for field in (
        "benchmark_or_final_test_payload_accessed",
        "benchmark_or_final_test_payload_used_for_training",
        "foreign_pretrained_weights_imported",
        "foreign_pretrained_tokenizer_imported",
        "foreign_teacher_logits_or_synthetic_pretraining_imported",
        "base_lineage_changed",
    ):
        _require(firewall[field] is False, f"{field} must be false")

    compute = payload["compute_truth"]
    for field in (
        "paid_compute_authorized",
        "paid_compute_used",
        "gpu_provisioned",
        "model_training_executed",
        "tokenizer_fit_executed",
    ):
        _require(compute[field] is False, f"{field} must be false")


def _successor_contract_is_complete(payload: dict[str, Any]) -> None:
    contract = payload["successor_acquisition_contract"]
    requirements = contract["required_before_any_training_credit"]
    _require(isinstance(requirements, list), "successor requirements must be a list")
    joined = "\n".join(requirements)
    for fragment in REQUIRED_SUCCESSOR_FRAGMENTS:
        _require(fragment in joined, f"successor contract missing requirement fragment: {fragment}")

    _require(
        contract["promotion_rule"].startswith("No CulturaX byte receives canonical training credit"),
        "promotion rule must explicitly preserve zero credit",
    )


def validate_payload(
    payload: dict[str, Any],
    registry: dict[str, Any],
    *,
    registry_blob_sha: str | None = None,
) -> None:
    _require(payload["schema_version"] == 1, "unsupported audit schema")
    _require(payload["audit_id"] == AUDIT_ID, "unexpected audit id")
    _require(payload["verdict"] == "CHANGES_REQUIRED", "audit verdict must stay CHANGES_REQUIRED")
    _require(payload["execution_profile"] == "LOCAL_FREE", "execution profile must stay LOCAL_FREE")

    swarm = payload["swarm"]
    _require(swarm["protocol"] == "SWARM-300-V2", "swarm protocol drift")
    _require(swarm["control_issue"] == 723, "swarm control issue drift")
    _require(swarm["worker_id"] == "SWARM-747", "worker identity drift")
    _require(swarm["claim_issue"] == 747, "claim issue drift")
    _require(swarm["lane_key"] == LANE_KEY, "lane key drift")
    _require(swarm["base_main_sha"] == BASE_MAIN_SHA, "base main SHA drift")

    authority = payload["project_authority"]
    _require(authority["registry_path"] == REGISTRY_PATH, "registry path drift")
    _require(authority["registry_blob_sha"] == REGISTRY_BLOB_SHA, "audit registry blob binding drift")
    if registry_blob_sha is not None:
        _require(registry_blob_sha == REGISTRY_BLOB_SHA, "checked-out registry blob identity drift")
    _require(authority["expected_component"] == EXPECTED_COMPONENT, "embedded CULTURAX authority drift")

    policy = registry.get("canonical_policy")
    _require(isinstance(policy, dict), "registry canonical_policy is required")
    for field, expected in authority["required_canonical_policy"].items():
        _require(policy.get(field) == expected, f"registry canonical policy drift: {field}")

    live_component = _component_by_id(registry, "CULTURAX")
    _require(live_component == EXPECTED_COMPONENT, "live registry CULTURAX component drift")
    _require(live_component["canonical_training_authorized"] is False, "registry training authority must be false")

    upstream = payload["upstream_observation"]
    _require(upstream["dataset_id"] == "uonlp/CulturaX", "upstream dataset identity drift")
    _require(
        upstream["dataset_card_readme_revision"]
        == "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e",
        "CulturaX README revision drift",
    )
    declared = upstream["declared_sources"]
    _require(declared["mc4_version"] == "3.1.0", "mC4 version drift")
    _require(
        declared["oscar_releases"] == ["20.19", "21.09", "22.01", "23.01"],
        "OSCAR release lineage drift",
    )
    _require(declared["common_crawl_origin"] is True, "Common Crawl origin must be retained")
    _require(
        upstream["record_provenance_fields"] == ["timestamp", "url", "source"],
        "record provenance fields must remain explicit",
    )
    _require(
        upstream["record_source_values"] == ["mc4", "OSCAR-xxxx"],
        "record source lineage must remain explicit",
    )

    evidence = payload["primary_rights_and_privacy_evidence"]
    evidence_ids = {entry.get("id") for entry in evidence if isinstance(entry, dict)}
    _require(evidence_ids == REQUIRED_EVIDENCE_IDS, "primary rights evidence set is incomplete or drifted")

    _all_zero_credit(payload)
    _rights_are_fail_closed(payload)
    _firewalls_are_intact(payload)
    _successor_contract_is_complete(payload)

    _require(
        payload["ua_en_feasibility"]["current_training_admission"]
        == "ZERO_CREDIT_PENDING_SOURCE_LEVEL_QUALIFICATION",
        "UA/EN admission must remain zero credit",
    )


def load_and_validate(audit_path: Path, registry_path: Path) -> None:
    audit_raw = audit_path.read_bytes()
    registry_raw = registry_path.read_bytes()
    payload = json.loads(audit_raw)
    registry = json.loads(registry_raw)
    validate_payload(payload, registry, registry_blob_sha=_git_blob_sha(registry_raw))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("configs/audit/culturax_ua_en_rights_provenance_v1.json"),
    )
    parser.add_argument("--registry", type=Path, default=Path(REGISTRY_PATH))
    args = parser.parse_args()
    try:
        load_and_validate(args.audit, args.registry)
    except (AuditValidationError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: CulturaX audit remains fail-closed and bound to live project authority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
