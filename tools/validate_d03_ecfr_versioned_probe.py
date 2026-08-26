#!/usr/bin/env python3
"""Fail-closed validator for the D03 versioned eCFR source probe.

This module performs no network access and grants no corpus or training credit.
It validates the discovery contract and can build a deterministic point-in-time
successor request envelope for later two-acquisition materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-ecfr-versioned-probe.v1"
REQUEST_SCHEMA = "12-6.d03-ecfr-versioned-request.v1"
DEFAULT_CONFIG = Path("configs/data/d03_ecfr_versioned_probe_v1.json")
BASE_GIT_SHA = "a73ab38026cb7849f478cc13ad58b93534a76e2f"
ISSUE = 672
OBSERVED_AT_UTC = "2026-08-26T20:06:00Z"
SOURCE_ID = "d03.ecfr.versioned-regulations.v1"
FAMILY_ID = "us.federal-regulations.ecfr"
PORTAL = "https://www.ecfr.gov"
API_DOCUMENTATION = "https://www.ecfr.gov/developers/documentation/api/v1"
TITLES_ENDPOINT = "https://www.ecfr.gov/api/versioner/v1/titles.json"
HISTORICAL_TEMPLATE = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
GOVINFO_DEVELOPER_HUB = "https://www.govinfo.gov/developers"
GOVINFO_XML_USER_GUIDE = (
    "https://www.govinfo.gov/bulkdata/ECFR/resources/ECFR-XML-User-Guide.pdf"
)
RIGHTS_REFERENCE = "https://www.copyright.gov/title17/92chap1.html"
TITLES_METADATA_AS_OF = "2026-08-06"
RESERVED_TITLES = (35,)

REQUIRED_SUCCESSORS = (
    "EXACT_POINT_IN_TIME_TITLE_REQUEST",
    "TWO_BYTE_IDENTICAL_ACQUISITIONS",
    "RAW_SHA256_AND_BYTE_LEDGER",
    "RIGHTS_AND_PROVENANCE_CLASSIFICATION",
    "DETERMINISTIC_XML_EXTRACTION_AND_NORMALIZATION",
    "QUALITY_AND_PRIVACY_REVIEW",
    "GLOBAL_EXACT_NEAR_FRAGMENT_AND_LINEAGE_DEDUP",
    "EVALUATION_RESERVATION_DECONTAMINATION",
    "FAMILY_CAP_AND_STRATUM_BALANCE_RETEST",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_SHARD_AND_PACK_TWO_CLEAN_BUILDS",
    "POST_PACK_UNIQUE_CAUSAL_LOSS_LEDGER",
    "TOKENIZER_AUTHORIZATION",
    "CHECKPOINT_AND_EVALUATION_GATES",
    "EXPLICIT_COMPUTE_AND_TRAINING_AUTHORIZATION",
)

FORBIDDEN_CLAIMS = (
    "CURRENT_ECFR_IS_IMMUTABLE",
    "PUBLIC_AVAILABILITY_EQUALS_TRAINING_RIGHTS",
    "GOVERNMENT_HOSTING_EQUALS_PUBLIC_DOMAIN_FOR_EVERY_EMBEDDED_OBJECT",
    "SOURCE_BYTES_EQUAL_TOKEN_COUNT",
    "SOURCE_BYTES_EQUAL_UNIQUE_CAUSAL_LOSS_POSITIONS",
    "PROBE_EQUALS_CORPUS_ADMISSION",
    "PROBE_EQUALS_TOKENIZER_AUTHORIZATION",
    "PROBE_EQUALS_TRAINING_AUTHORIZATION",
)


class ProbeValidationError(ValueError):
    """Raised when the frozen eCFR probe contract drifts or overclaims authority."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeValidationError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _exact(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    _require(mapping.get(key) == expected, f"{path}.{key} drifted")


def _zero_credit(credit: Mapping[str, Any]) -> None:
    for key in (
        "family_credit",
        "candidate_raw_bytes",
        "normalized_capacity_credited",
        "training_authorized_bytes",
        "training_authorized_loss_positions",
    ):
        value = credit.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value == 0,
            f"credit.{key} must remain integer zero at probe stage",
        )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
        "corpus_release_authorized",
    ):
        _require(credit.get(key) is False, f"credit.{key} must remain false")


def validate_probe_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact zero-credit discovery contract and return its identity."""

    _exact(config, "schema", SCHEMA, "root")
    _exact(config, "status", "PROBE_PLANNED_ZERO_CREDIT", "root")
    _exact(config, "scope", "LOCAL_FREE_DISCOVERY_AND_SUCCESSOR_REQUEST_ONLY", "root")

    authority = _mapping(config.get("authority"), "authority")
    _exact(authority, "repository", "Oleksii-debug/12-6-ai.", "authority")
    _exact(authority, "base_git_sha", BASE_GIT_SHA, "authority")
    _exact(authority, "issue", ISSUE, "authority")
    _exact(authority, "observed_at_utc", OBSERVED_AT_UTC, "authority")

    source = _mapping(config.get("source"), "source")
    expected_source = {
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "stratum": "en",
        "publisher": "Office of the Federal Register and U.S. Government Publishing Office",
        "portal": PORTAL,
        "api_documentation": API_DOCUMENTATION,
        "titles_endpoint": TITLES_ENDPOINT,
        "historical_title_endpoint_template": HISTORICAL_TEMPLATE,
        "govinfo_developer_hub": GOVINFO_DEVELOPER_HUB,
        "govinfo_xml_user_guide": GOVINFO_XML_USER_GUIDE,
        "modality": "structured_regulatory_xml",
        "language": "en",
        "titles_metadata_as_of": TITLES_METADATA_AS_OF,
        "title_min": 1,
        "title_max": 50,
        "reserved_titles_at_observation": list(RESERVED_TITLES),
        "reserved_titles_must_be_skipped": True,
    }
    for key, expected in expected_source.items():
        _exact(source, key, expected, "source")

    versioning = _mapping(config.get("versioning"), "versioning")
    expected_versioning = {
        "mutable_current_endpoint_allowed_for_capacity": False,
        "point_in_time_date_required": True,
        "exact_title_required": True,
        "request_date_must_not_exceed_titles_metadata_as_of": True,
        "two_byte_identical_acquisitions_required": True,
        "sha256_required": True,
        "byte_count_required": True,
        "content_type_required": "application/xml",
        "request_identity_fields": ["date", "title", "url"],
    }
    for key, expected in expected_versioning.items():
        _exact(versioning, key, expected, "versioning")

    rights = _mapping(config.get("rights"), "rights")
    expected_rights = {
        "status": "REVIEW_REQUIRED_BEFORE_ANY_CAPACITY",
        "primary_reference": RIGHTS_REFERENCE,
        "section": "17 U.S.C. 105",
        "government_work_rule": (
            "US_GOVERNMENT_WORKS_GENERALLY_NOT_COPYRIGHT_PROTECTED_UNDER_TITLE_17"
        ),
        "blanket_training_permission_claimed": False,
        "government_can_hold_transferred_copyrights": True,
        "covered_work_exceptions_exist": True,
        "third_party_or_contractor_material_requires_separate_basis": True,
        "incorporated_by_reference_material_requires_separate_basis": True,
        "images_media_and_external_attachments_require_separate_basis": True,
        "unknown_provenance_must_be_excluded": True,
        "issuing_agency_context_must_be_preserved": True,
    }
    for key, expected in expected_rights.items():
        _exact(rights, key, expected, "rights")

    xml_safety = _mapping(config.get("xml_safety"), "xml_safety")
    _exact(xml_safety, "doctype_allowed", False, "xml_safety")
    _exact(xml_safety, "external_entities_allowed", False, "xml_safety")
    _exact(xml_safety, "network_resolution_during_parse_allowed", False, "xml_safety")
    _exact(xml_safety, "archive_or_xml_byte_limit", 134217728, "xml_safety")
    _exact(xml_safety, "single_text_node_byte_limit", 8388608, "xml_safety")

    credit = _mapping(config.get("credit"), "credit")
    _zero_credit(credit)

    _require(
        tuple(config.get("required_successors", ())) == REQUIRED_SUCCESSORS,
        "required_successors drifted or changed order",
    )
    _require(
        tuple(config.get("forbidden_claims", ())) == FORBIDDEN_CLAIMS,
        "forbidden_claims drifted or changed order",
    )

    return {
        "schema": "12-6.d03-ecfr-versioned-probe-validation.v1",
        "status": "PASS_ZERO_CREDIT_PROBE_CONTRACT",
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "config_sha256": sha256_json(config),
        "titles_metadata_as_of": TITLES_METADATA_AS_OF,
        "reserved_titles": list(RESERVED_TITLES),
        "family_credit": 0,
        "training_authorized_bytes": 0,
        "training_authorized_loss_positions": 0,
        "model_training_authorized": False,
        "paid_compute_authorized": False,
    }


def build_successor_request(
    config: Mapping[str, Any],
    *,
    request_date: str,
    title: int,
) -> dict[str, Any]:
    """Build a deterministic point-in-time request envelope without fetching bytes."""

    validation = validate_probe_contract(config)
    _require(isinstance(request_date, str), "request date must be YYYY-MM-DD text")
    try:
        parsed_date = date.fromisoformat(request_date)
    except ValueError as exc:
        raise ProbeValidationError("request date must be an exact YYYY-MM-DD date") from exc

    metadata_as_of = date.fromisoformat(TITLES_METADATA_AS_OF)
    _require(
        parsed_date <= metadata_as_of,
        "request date is newer than the frozen eCFR titles metadata",
    )
    _require(isinstance(title, int) and not isinstance(title, bool), "title must be an integer")
    _require(1 <= title <= 50, "title must be between 1 and 50")
    _require(title not in RESERVED_TITLES, f"title {title} is reserved in frozen metadata")

    url = HISTORICAL_TEMPLATE.format(date=request_date, title=title)
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "SUCCESSOR_MATERIALIZATION_REQUEST_ONLY",
        "source_contract_sha256": validation["config_sha256"],
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "stratum": "en",
        "titles_metadata_as_of": TITLES_METADATA_AS_OF,
        "date": request_date,
        "title": title,
        "url": url,
        "required_content_type": "application/xml",
        "two_byte_identical_acquisitions_required": True,
        "raw_sha256_required": True,
        "raw_byte_count_required": True,
        "reserved_title_check_passed": True,
        "rights_and_provenance_status": "NOT_RUN",
        "normalization_status": "NOT_RUN",
        "global_dedup_status": "NOT_RUN",
        "evaluation_decontamination_status": "NOT_RUN",
        "family_credit": 0,
        "training_authorized_bytes": 0,
        "training_authorized_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "paid_compute_authorized": False,
    }
    request["request_identity_sha256"] = sha256_json(request)
    return request


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeValidationError(f"cannot load config: {path}") from exc
    return _mapping(value, "root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--date", dest="request_date")
    parser.add_argument("--title", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _load_json(args.config)
        if (args.request_date is None) != (args.title is None):
            raise ProbeValidationError("--date and --title must be supplied together")
        if args.request_date is None:
            report = validate_probe_contract(config)
        else:
            report = build_successor_request(
                config,
                request_date=args.request_date,
                title=args.title,
            )
    except ProbeValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
