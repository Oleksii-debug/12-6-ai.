"""Fail-closed source-rights/provenance authority for the bounded Ubuntu IRC source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.d03-common-pile-ubuntu-irc-source-rights.v1"
AUTHORITY_ID = "D03-COMMON-PILE-UBUNTU-IRC-SOURCE-RIGHTS-V1"
STATUS = "SOURCE_SPECIFIC_RIGHTS_PROVENANCE_QUALIFIED_ZERO_CREDIT"
PARENT_IDENTITY = "b279c4a7404e0e501acd0c77842d1f71c43ea1dbcd611acab18163c64b060d4e"
UPSTREAM_COMMIT = "9457f04a14cb2355ab00023420369d46ffd4a395"
SOURCE_DATASET = "common-pile/ubuntu_irc"
SOURCE_REVISION = "47d55b0534a62bf0766c621297165451969f3de9"
SOURCE_FILE = "v0/documents/00007_ubuntu.jsonl.gz"
SOURCE_SHA256 = "75e38bffcaceb00ed9ce9a63b1d9e74a70f5582b3e3f763a7ec573c7fd6c1e60"


class UbuntuIrcRightsError(ValueError):
    """Raised when the Ubuntu IRC source-rights contract is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuIrcRightsError(message)


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} keys drifted")
    return value


def _require_bool(value: Any, expected: bool, label: str) -> None:
    _require(type(value) is bool and value is expected, f"{label} must be literal {expected}")


def _require_int(value: Any, expected: int, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} must be integer {expected}")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _registry_identity(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("registry_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _validate_parent_registry(payload: dict[str, Any]) -> None:
    _require(payload.get("registry_id") == "COMMON-PILE-SOURCE-RIGHTS-V1", "wrong parent registry")
    _require(payload.get("status") == "CANDIDATE_SOURCE_RIGHTS_AUDIT", "parent status drifted")
    _require(
        payload.get("registry_identity_sha256") == PARENT_IDENTITY,
        "parent identity field drifted",
    )
    _require(_registry_identity(payload) == PARENT_IDENTITY, "parent registry bytes drifted")

    upstream = payload.get("upstream_authority")
    _require(type(upstream) is dict, "parent upstream_authority missing")
    _require(
        upstream.get("audited_code_commit") == UPSTREAM_COMMIT,
        "parent upstream commit drifted",
    )

    policy = payload.get("global_policy")
    _require(type(policy) is dict, "parent global_policy missing")
    _require_bool(
        policy.get("source_level_rights_and_provenance_required"),
        True,
        "parent source review",
    )
    _require_bool(policy.get("per_source_review_required"), True, "parent per-source review")
    _require_bool(policy.get("canonical_training_authorized"), False, "parent training authority")
    _require_int(policy.get("corpus_credit_bytes"), 0, "parent corpus credit")
    _require_int(policy.get("authorized_loss_positions"), 0, "parent loss credit")

    rows = payload.get("sources")
    _require(type(rows) is list, "parent sources must be a list")
    matches = [row for row in rows if type(row) is dict and row.get("key") == "ubuntu_irc"]
    _require(len(matches) == 1, "parent must contain exactly one ubuntu_irc row")
    row = matches[0]
    required = {
        "hf_dataset": SOURCE_DATASET,
        "domain": "online_forum",
        "audited_collector_path": "sources/ubuntu",
        "collector_path_status": "PRESENT_AT_AUDITED_CODE_COMMIT",
        "rights_basis_class": "PUBLIC_DOMAIN_ARCHIVE",
        "project_review_status": "REVIEW_REQUIRED",
        "evaluation_role": "TRAINING_CANDIDATE_ONLY",
    }
    for field, expected in required.items():
        _require(row.get(field) == expected, f"parent ubuntu_irc {field} drifted")
    _require(
        row.get("license_or_status_signals") == ["PUBLIC_DOMAIN"],
        "parent rights signal drifted",
    )
    _require_bool(
        row.get("canonical_training_authorized"),
        False,
        "parent source training authority",
    )
    _require_int(row.get("credited_bytes"), 0, "parent source credited bytes")
    _require_int(row.get("authorized_loss_positions"), 0, "parent source loss positions")
    _require_bool(row.get("final_test_excluded"), True, "parent final-test exclusion")


def validate_authority(authority: dict[str, Any], parent_registry: dict[str, Any]) -> str:
    """Validate the exact source-specific rights authority and its zero-credit boundary."""
    _require_exact_keys(
        authority,
        {
            "schema_version",
            "authority_id",
            "status",
            "project_authority",
            "parent_registry",
            "upstream_collector",
            "ubuntu_policy_evidence",
            "source_scope",
            "real_execution_reference",
            "decision",
            "truth_boundary",
        },
        "authority",
    )
    _require(authority["schema_version"] == SCHEMA_VERSION, "schema_version drifted")
    _require(authority["authority_id"] == AUTHORITY_ID, "authority_id drifted")
    _require(authority["status"] == STATUS, "status drifted")

    project = _require_exact_keys(
        authority["project_authority"],
        {"swarm_protocol", "control_issue", "worker_issue", "base_main_sha"},
        "project_authority",
    )
    _require(project["swarm_protocol"] == "SWARM-300-V2", "wrong swarm protocol")
    _require_int(project["control_issue"], 723, "control_issue")
    _require_int(project["worker_issue"], 1105, "worker_issue")
    _require(
        project["base_main_sha"] == "6ac04664a6a1a8e14be8ed9a8032cf99099503b0",
        "base main drifted",
    )

    parent = _require_exact_keys(
        authority["parent_registry"],
        {
            "path",
            "registry_id",
            "registry_identity_sha256",
            "source_key",
            "hf_dataset",
            "rights_basis_class",
            "rights_signal",
            "required_parent_review_status",
        },
        "parent_registry",
    )
    _require(
        parent["path"] == "configs/data/common_pile_source_rights_v1.json",
        "parent path drifted",
    )
    _require(parent["registry_id"] == "COMMON-PILE-SOURCE-RIGHTS-V1", "parent registry id drifted")
    _require(
        parent["registry_identity_sha256"] == PARENT_IDENTITY,
        "configured parent identity drifted",
    )
    _require(parent["source_key"] == "ubuntu_irc", "source key drifted")
    _require(parent["hf_dataset"] == SOURCE_DATASET, "configured source dataset drifted")
    _require(parent["rights_basis_class"] == "PUBLIC_DOMAIN_ARCHIVE", "rights basis drifted")
    _require(parent["rights_signal"] == "PUBLIC_DOMAIN", "rights signal drifted")
    _require(
        parent["required_parent_review_status"] == "REVIEW_REQUIRED",
        "parent review boundary drifted",
    )

    upstream = _require_exact_keys(
        authority["upstream_collector"],
        {
            "repository",
            "audited_commit",
            "collector_path",
            "readme_path",
            "readme_blob_sha1",
            "to_dolma_path",
            "to_dolma_blob_sha1",
        },
        "upstream_collector",
    )
    _require(
        upstream["repository"] == "https://github.com/r-three/common-pile",
        "upstream repository drifted",
    )
    _require(upstream["audited_commit"] == UPSTREAM_COMMIT, "upstream commit drifted")
    _require(upstream["collector_path"] == "sources/ubuntu", "collector path drifted")
    _require(upstream["readme_path"] == "sources/ubuntu/README.md", "README path drifted")
    _require(
        upstream["readme_blob_sha1"] == "98a9e0b9cfc86455955dbe4e3ef56b95659bb0a8",
        "README blob drifted",
    )
    _require(upstream["to_dolma_path"] == "sources/ubuntu/to-dolma.py", "converter path drifted")
    _require(
        upstream["to_dolma_blob_sha1"] == "a15978f5a88bf8cf356893a2e0df20feb994c966",
        "converter blob drifted",
    )

    evidence = _require_exact_keys(
        authority["ubuntu_policy_evidence"],
        {"irc_guidelines", "community_help_irc", "irc_terms_of_service"},
        "ubuntu_policy_evidence",
    )
    expected_evidence = {
        "irc_guidelines": (
            "https://wiki.ubuntu.com/IRC/Guidelines",
            "IRC/Guidelines - Ubuntu Wiki",
            "2021-09-05",
            "UBUNTU_CHANNEL_PUBLIC_DOMAIN_POLICY",
        ),
        "community_help_irc": (
            "https://help.ubuntu.com/community/InternetRelayChat",
            "InternetRelayChat - Community Help Wiki",
            "2024-03-28",
            "UBUNTU_CHANNEL_PUBLIC_DOMAIN_POLICY",
        ),
        "irc_terms_of_service": (
            "https://wiki.ubuntu.com/IRC/TermsOfService",
            "IRC/TermsOfService - Ubuntu Wiki",
            "2025-01-24",
            "PUBLIC_LOGGING_AND_STORAGE_CONSENT",
        ),
    }
    for key, (url, title, last_edited, role) in expected_evidence.items():
        item = _require_exact_keys(
            evidence[key],
            {"url", "title", "last_edited", "evidence_role", "observed_fact"},
            f"ubuntu_policy_evidence.{key}",
        )
        _require(item["url"] == url, f"{key} URL drifted")
        _require(item["title"] == title, f"{key} title drifted")
        _require(item["last_edited"] == last_edited, f"{key} edit date drifted")
        _require(item["evidence_role"] == role, f"{key} evidence role drifted")
        _require(
            type(item["observed_fact"]) is str and bool(item["observed_fact"].strip()),
            f"{key} fact missing",
        )

    scope = _require_exact_keys(
        authority["source_scope"],
        {
            "required_source_label",
            "required_metadata_license",
            "required_metadata_url_prefix",
            "non_ubuntu_irc_allowed",
            "website_wide_license_inference_allowed",
            "dataset_metadata_alone_is_sufficient",
            "policy_revalidation_required_on_drift",
        },
        "source_scope",
    )
    _require(scope["required_source_label"] == "ubuntu-chat", "source label drifted")
    _require(scope["required_metadata_license"] == "Public Domain", "metadata license drifted")
    _require(
        scope["required_metadata_url_prefix"] == "https://irclogs.ubuntu.com/",
        "origin prefix drifted",
    )
    _require_bool(scope["non_ubuntu_irc_allowed"], False, "non-Ubuntu IRC allowance")
    _require_bool(scope["website_wide_license_inference_allowed"], False, "website-wide inference")
    _require_bool(scope["dataset_metadata_alone_is_sufficient"], False, "metadata-only authority")
    _require_bool(scope["policy_revalidation_required_on_drift"], True, "policy drift revalidation")

    execution = _require_exact_keys(
        authority["real_execution_reference"],
        {
            "role",
            "product_pr",
            "product_head_sha",
            "evidence_path",
            "evidence_blob_sha1",
            "source_dataset",
            "source_revision",
            "source_file",
            "source_sha256",
            "candidate_payload_sha256",
        },
        "real_execution_reference",
    )
    _require(
        execution["role"] == "SEPARATE_REQUIRED_AUTHORITY_NOT_CONSUMED_BY_THIS_RIGHTS_DECISION",
        "execution authority boundary widened",
    )
    _require_int(execution["product_pr"], 1100, "product_pr")
    _require(
        execution["product_head_sha"] == "ab457e641d3ca5252cdbbd7fd5e2928d3c761521",
        "execution head drifted",
    )
    _require(
        execution["evidence_path"] == "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json",
        "execution evidence path drifted",
    )
    _require(
        execution["evidence_blob_sha1"] == "23c73b36f12539200a3da97b2f177b562b62196b",
        "execution evidence blob drifted",
    )
    _require(execution["source_dataset"] == SOURCE_DATASET, "execution dataset drifted")
    _require(execution["source_revision"] == SOURCE_REVISION, "execution revision drifted")
    _require(execution["source_file"] == SOURCE_FILE, "execution file drifted")
    _require(execution["source_sha256"] == SOURCE_SHA256, "execution source hash drifted")
    _require(
        execution["candidate_payload_sha256"]
        == "d6a0eaea27313e0d5146bc2a0624104541923ef06b12f01c22b3ccf6956eb6c2",
        "candidate payload hash drifted",
    )

    decision = _require_exact_keys(
        authority["decision"],
        {
            "source_scope_qualified",
            "decision_class",
            "legal_conclusion_claimed",
            "formal_public_domain_waiver_equivalence_claimed",
            "general_irc_public_domain_claimed",
            "canonical_training_authorized",
            "corpus_credit_bytes",
            "authorized_loss_positions",
            "tokenizer_fit_authorized",
            "optimizer_updates_authorized",
            "evaluation_eligible",
            "final_test_accessed",
        },
        "decision",
    )
    _require_bool(decision["source_scope_qualified"], True, "source qualification")
    _require(
        decision["decision_class"] == "PROJECT_POLICY_SOURCE_QUALIFICATION_NOT_LEGAL_OPINION",
        "decision class drifted",
    )
    for field in (
        "legal_conclusion_claimed",
        "formal_public_domain_waiver_equivalence_claimed",
        "general_irc_public_domain_claimed",
        "canonical_training_authorized",
        "tokenizer_fit_authorized",
        "optimizer_updates_authorized",
        "evaluation_eligible",
        "final_test_accessed",
    ):
        _require_bool(decision[field], False, f"decision.{field}")
    _require_int(decision["corpus_credit_bytes"], 0, "decision.corpus_credit_bytes")
    _require_int(decision["authorized_loss_positions"], 0, "decision.authorized_loss_positions")

    truth = _require_exact_keys(
        authority["truth_boundary"],
        {
            "durable_evidence_contains_source_text",
            "durable_evidence_contains_usernames_or_authors",
            "real_data_executed_by_this_package",
            "training_executed",
            "learned_weights_created",
            "paid_compute_used",
            "foreign_pretrained_weights_used",
            "external_llm_or_api_used_for_data_or_intelligence",
        },
        "truth_boundary",
    )
    for field in truth:
        _require_bool(truth[field], False, f"truth_boundary.{field}")

    _validate_parent_registry(parent_registry)
    return STATUS


def validate_candidate_metadata(source: Any, metadata: Any) -> None:
    """Validate the source metadata envelope; this never grants training authority by itself."""
    _require(source == "ubuntu-chat", "candidate source label is not ubuntu-chat")
    _require(type(metadata) is dict, "candidate metadata must be an object")
    _require(metadata.get("license") == "Public Domain", "candidate metadata license drifted")
    url = metadata.get("url")
    _require(
        type(url) is str and url.startswith("https://irclogs.ubuntu.com/"),
        "candidate origin is not Ubuntu irclogs",
    )
    channel = metadata.get("channel")
    _require(
        type(channel) is str and channel.startswith("#") and len(channel) > 1,
        "candidate channel metadata invalid",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(type(payload) is dict, f"{path} must contain a JSON object")
    assert isinstance(payload, dict)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("authority", type=Path)
    parser.add_argument("parent_registry", type=Path)
    args = parser.parse_args(argv)
    status = validate_authority(load_json(args.authority), load_json(args.parent_registry))
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
