#!/usr/bin/env python3
"""Fail-closed validator for the peS2o source-rights audit V1 contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.pes2o-source-rights-audit.v1"
EXPECTED_CODE_REPO = "https://github.com/allenai/peS2o"
EXPECTED_DATASET_REPO = "https://huggingface.co/datasets/allenai/peS2o"
EXPECTED_SOURCE_FIELDS = ["added", "created", "id", "source", "text", "version"]
EXPECTED_CHANNELS = {"S2ORC_FULLTEXT": "s2orc", "S2AG_TITLE_ABSTRACT": "s2ag"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class AuditValidationError(ValueError):
    """Raised when the audit contract would permit unsupported authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditValidationError(message)


def _require_false(mapping: dict[str, Any], key: str, context: str) -> None:
    _require(mapping.get(key) is False, f"{context}.{key} must be false")


def _require_true(mapping: dict[str, Any], key: str, context: str) -> None:
    _require(mapping.get(key) is True, f"{context}.{key} must be true")


def validate_manifest(data: dict[str, Any]) -> None:
    """Validate the V1 contract and reject any accidental authority widening."""

    _require(data.get("schema_id") == EXPECTED_SCHEMA, "unexpected schema_id")
    _require(data.get("schema_version") == 1, "schema_version must be 1")
    _require(
        data.get("status") == "REVIEW_REQUIRED_ZERO_TRAINING_CREDIT",
        "audit status must remain fail-closed",
    )

    authority = data.get("project_authority")
    _require(isinstance(authority, dict), "project_authority must be an object")
    _require(authority.get("repository") == "Oleksii-debug/12-6-ai.", "wrong repository")
    _require(HEX40.fullmatch(str(authority.get("base_sha", ""))) is not None, "bad base_sha")
    _require(authority.get("swarm_protocol") == "SWARM-300-V2", "wrong swarm protocol")
    _require(authority.get("swarm_control_issue") == 723, "wrong swarm control issue")
    _require(authority.get("worker_issue") == 745, "wrong worker issue")

    upstream = data.get("upstream_identity")
    _require(isinstance(upstream, dict), "upstream_identity must be an object")
    code = upstream.get("code_repository")
    dataset = upstream.get("dataset_repository")
    derivation = upstream.get("declared_derivation")
    rights_context = upstream.get("semantic_scholar_rights_context")
    _require(isinstance(code, dict), "code_repository must be an object")
    _require(isinstance(dataset, dict), "dataset_repository must be an object")
    _require(isinstance(derivation, dict), "declared_derivation must be an object")
    _require(isinstance(rights_context, dict), "rights context must be an object")

    _require(code.get("url") == EXPECTED_CODE_REPO, "unexpected peS2o code repository")
    _require(HEX40.fullmatch(str(code.get("commit", ""))) is not None, "bad code commit")
    _require(code.get("license") == "Apache-2.0", "code license must be Apache-2.0")
    _require(
        code.get("license_scope") == "PE_S2O_CODE_REPOSITORY_ONLY",
        "code license scope must stay code-only",
    )
    _require(dataset.get("url") == EXPECTED_DATASET_REPO, "unexpected peS2o dataset repository")
    _require(HEX40.fullmatch(str(dataset.get("revision", ""))) is not None, "bad dataset revision")
    _require(dataset.get("dataset_card_license") == "ODC-By-1.0", "unexpected dataset license")
    _require(
        dataset.get("license_scope") == "DATASET_LAYER_NOT_BLANKET_UNDERLYING_CONTENT_RIGHTS",
        "dataset license must not be widened to underlying-content authority",
    )

    _require(
        derivation.get("observed_document_fields") == EXPECTED_SOURCE_FIELDS,
        "observed peS2o record schema changed or is incomplete",
    )
    _require_false(derivation, "per_document_license_field_in_observed_schema", "declared_derivation")
    _require_true(
        rights_context,
        "third_party_content_may_have_separate_terms",
        "semantic_scholar_rights_context",
    )
    _require_true(
        rights_context,
        "semantic_scholar_cannot_grant_publisher_or_author_permissions",
        "semantic_scholar_rights_context",
    )

    channels = data.get("source_channels")
    _require(isinstance(channels, list), "source_channels must be a list")
    _require(len(channels) == len(EXPECTED_CHANNELS), "source channel set must be complete")
    by_id = {channel.get("id"): channel for channel in channels if isinstance(channel, dict)}
    _require(set(by_id) == set(EXPECTED_CHANNELS), "unexpected or missing source channel")
    for channel_id, source_value in EXPECTED_CHANNELS.items():
        channel = by_id[channel_id]
        _require(channel.get("pes2o_source_value") == source_value, f"bad source value for {channel_id}")
        _require(channel.get("rights_status") == "REVIEW_REQUIRED", f"{channel_id} rights must remain review-required")
        _require(channel.get("source_specific_license_evidence") == [], f"{channel_id} cannot fabricate license evidence")
        _require_false(channel, "training_authorized", channel_id)
        _require(channel.get("authorized_bytes") == 0, f"{channel_id}.authorized_bytes must be zero")

    rights = data.get("rights_gate")
    _require(isinstance(rights, dict), "rights_gate must be an object")
    for key in (
        "code_license_is_dataset_training_authority",
        "dataset_package_license_is_training_authority",
        "open_access_label_is_training_authority",
        "training_authorized",
    ):
        _require_false(rights, key, "rights_gate")
    for key in (
        "source_level_rights_and_provenance_required",
        "document_level_rights_evidence_required",
        "ambiguous_or_missing_rights_fail_closed",
    ):
        _require_true(rights, key, "rights_gate")
    _require(rights.get("authorized_source_bytes") == 0, "authorized_source_bytes must be zero")
    _require(rights.get("corpus_credit_bytes") == 0, "corpus_credit_bytes must be zero")

    contamination = data.get("contamination_gate")
    _require(isinstance(contamination, dict), "contamination_gate must be an object")
    for key in (
        "upstream_train_valid_split_is_project_evaluation_clearance",
        "benchmark_or_final_test_payload_accessed",
        "decontamination_executed",
        "admission_before_decontamination_allowed",
    ):
        _require_false(contamination, key, "contamination_gate")
    for key in (
        "project_reserved_evaluation_excluded",
        "exact_decontamination_required_before_admission",
        "near_decontamination_required_before_admission",
    ):
        _require_true(contamination, key, "contamination_gate")

    requirements = data.get("admission_requirements")
    _require(isinstance(requirements, list) and len(requirements) >= 7, "admission requirements are incomplete")

    boundaries = data.get("hard_boundaries")
    _require(isinstance(boundaries, dict), "hard_boundaries must be an object")
    for key in (
        "bulk_download_performed",
        "training_data_downloaded_by_this_audit",
        "tokenizer_fit_authorized",
        "paid_compute_authorized",
        "foreign_pretrained_weights_allowed",
        "final_test_data_allowed_in_training",
    ):
        _require_false(boundaries, key, "hard_boundaries")
    _require(boundaries.get("optimizer_updates_authorized") == 0, "optimizer updates must remain zero")


def load_and_validate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "audit JSON root must be an object")
    validate_manifest(data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("configs/data/pes2o_source_audit_v1.json"),
    )
    args = parser.parse_args()
    try:
        load_and_validate(args.path)
    except (AuditValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: peS2o V1 source-rights audit remains fail-closed with zero training credit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
