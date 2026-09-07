#!/usr/bin/env python3
"""Fail-closed validator for the D03 eCFR point-in-time materialization contract."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG = Path("configs/data/d03_ecfr_point_in_time_materialization_v1.json")
SCHEMA = "12-6.d03-ecfr-point-in-time-materialization.v1"
WORKER_ID = "D03-ECFR-POINT-IN-TIME-MATERIALIZATION-V1"
EXPECTED_CONTRACT_IDENTITY = "8208e4a6569fa9f70c7181b02ab564866d72b876c5df437903a60ee0f82dd34a"
PARENT_PR = 707
PARENT_HEAD = "1c1fb68737085a532d08e57a776fe88d4dab3dcd"
PARENT_MAIN = "a53279292af68dc95e4e615542b6c8c2ea7f9ee5"
PARENT_BLOB = "f75bc56aec8e56ee41e214ffacc153e313c44518"
PARENT_IDENTITY = "79dbbe041e8270bb5fb89fa34b34807ca699056aef05df85421e3c22bd133408"
SOURCE_ID = "en.us.ecfr.regulations"
FAMILY_ID = "us.federal-regulations.ecfr"
TEMPLATE = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
METADATA_DATE = "2026-08-06"
RESERVED_TITLES = {35}
RIGHTS_EXCLUSIONS = [
    "incorporated_by_reference_material",
    "contractor_or_private_authorship",
    "transferred_copyright_material",
    "third_party_tables_images_media",
    "provenance_ambiguous_payload",
]
ZERO_FIELDS = (
    "canonical_capacity_credit_bytes",
    "family_credit",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "optimizer_updates",
)
FALSE_FIELDS = (
    "rights_and_provenance_complete",
    "quality_language_privacy_complete",
    "global_dedup_complete",
    "evaluation_decontamination_complete",
    "split_pack_complete",
    "tokenizer_fit_authorized",
    "model_training_executed",
    "final_test_accessed",
    "paid_compute_used",
    "research_corpus_v1_released",
)


class ContractError(RuntimeError):
    """Raised when the acquisition contract drifts or overclaims authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_identity(config: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(config))
    core.pop("contract_identity_sha256", None)
    return sha256(canonical_bytes(core))


def _date_allowed(value: str) -> bool:
    try:
        return date.fromisoformat(value) <= date.fromisoformat(METADATA_DATE)
    except ValueError:
        return False


def validate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == SCHEMA, "schema drift")
    require(config.get("worker_id") == WORKER_ID, "worker drift")
    require(config.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE weakened")
    declared = config.get("contract_identity_sha256")
    require(declared == EXPECTED_CONTRACT_IDENTITY, "contract identity declaration drift")
    require(contract_identity(config) == EXPECTED_CONTRACT_IDENTITY, "contract content drift")

    parent = config.get("parent_authority", {})
    expected_parent = {
        "pr": PARENT_PR,
        "head_sha": PARENT_HEAD,
        "main_merge_sha": PARENT_MAIN,
        "config_path": "configs/data/d03_ecfr_acquisition_probe_v1.json",
        "config_git_blob_sha1": PARENT_BLOB,
        "contract_identity_sha256": PARENT_IDENTITY,
    }
    require(parent == expected_parent, "parent authority drift")

    source = config.get("source", {})
    expected_source = {
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "stratum": "en",
        "publisher": "Office of the Federal Register / Government Publishing Office",
        "portal": "https://www.ecfr.gov",
        "historical_full_title_endpoint_template": TEMPLATE,
        "titles_metadata_date": METADATA_DATE,
        "title_minimum": 1,
        "title_maximum": 50,
        "reserved_titles": [35],
        "family_accounting": "ONE_CONSERVATIVE_FAMILY_UNTIL_LINEAGE_AUDIT",
    }
    require(source == expected_source, "source authority drift")

    objects = config.get("objects")
    require(isinstance(objects, list) and objects, "objects missing")
    seen: set[tuple[str, int]] = set()
    for item in objects:
        require(isinstance(item, dict), "object contract must be object")
        object_date = item.get("date")
        title = item.get("title")
        require(isinstance(object_date, str) and _date_allowed(object_date), "object date invalid")
        require(isinstance(title, int) and not isinstance(title, bool), "title must be integer")
        require(1 <= title <= 50 and title not in RESERVED_TITLES, "title unavailable")
        expected_url = TEMPLATE.format(date=object_date, title=title)
        require(item.get("url") == expected_url, "object URL detached from date/title")
        key = (object_date, title)
        require(key not in seen, "duplicate date/title object")
        seen.add(key)

    network = config.get("network_policy", {})
    expected_network = {
        "allowed_hosts": ["www.ecfr.gov", "ecfr.gov"],
        "allowed_content_types": ["application/xml", "text/xml"],
        "accept_encoding": "identity",
        "user_agent": "12-6-ai-ecfr-point-in-time-materializer/1",
        "timeout_seconds": 120,
        "max_object_bytes": 134217728,
        "max_total_bytes": 134217728,
        "redirect_same_path_required": True,
        "two_independent_acquisitions_required": True,
    }
    require(network == expected_network, "network policy drift")

    xml_policy = config.get("xml_policy", {})
    require(xml_policy == {
        "reject_doctype": True,
        "reject_entity_declarations": True,
        "extraction_scope": "ALL_CHARACTER_DATA_NO_ATTRIBUTES",
        "whitespace_policy": "UNICODE_WHITESPACE_COLLAPSE_TO_ASCII_SPACE_STRIP",
        "durable_report_contains_raw_text": False,
    }, "XML policy drift")

    rights = config.get("rights_boundary", {})
    require(rights == {
        "decision": "REVIEW_REQUIRED_ZERO_CREDIT",
        "automatic_government_work_classification_allowed": False,
        "must_exclude_or_separately_clear": RIGHTS_EXCLUSIONS,
        "training_eligibility_before_rights_review": False,
    }, "rights boundary drift")

    boundary = config.get("claim_boundary", {})
    for field in ZERO_FIELDS:
        require(boundary.get(field) == 0, f"claim must remain zero: {field}")
    for field in FALSE_FIELDS:
        require(boundary.get(field) is False, f"claim must remain false: {field}")
    require(set(boundary) == set(ZERO_FIELDS) | set(FALSE_FIELDS), "claim boundary surface drift")
    return dict(config)


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract: {exc}") from exc
    return validate_contract(config)
