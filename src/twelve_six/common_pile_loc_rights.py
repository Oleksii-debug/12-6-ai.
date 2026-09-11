"""Fail-closed source-rights qualification for Common Pile Library of Congress books."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

AUTHORITY_ID = "D03-COMMON-PILE-LOC-SOURCE-RIGHTS-V1"
SCHEMA_VERSION = "12-6.d03-common-pile-loc-source-rights.v1"
STATUS = "SOURCE_POLICY_QUALIFIED_CONDITIONAL"
POLICY_IDENTITY = "1a140049b83da211e54003da8fa48fdd3e688eb30abed4bd3c327e2d64f36c1a"

GENERIC_REGISTRY = {
    "path": "configs/data/common_pile_source_rights_v1.json",
    "blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "source_key": "library_of_congress",
    "expected_project_review_status": "REVIEW_REQUIRED",
}
UPSTREAM_CODE = {
    "repository": "https://github.com/r-three/common-pile",
    "revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "collector_path": "sources/loc_books/books.py",
    "collector_blob_sha1": "0f8633d9a0c12bd04140420d74a95d82a065119f",
    "metadata_path": "sources/loc_books/metadata.py",
    "metadata_blob_sha1": "a219f88cbe8f6cc33bd72eed6c9d75958a8efb08",
    "readme_path": "sources/loc_books/README.md",
    "readme_blob_sha1": "c364b6dec4232e128b7adee92747f0d728090714",
    "collection_url": (
        "https://www.loc.gov/collections/selected-digitized-books/?fa=language:english"
    ),
}
DATASET_CONTRACT = {
    "origin_pr": 1135,
    "origin_head": "c72bef68127c82abd70016eeaf6070ff4fe096aa",
    "origin_config_blob_sha1": "958339654703f3aab4ea9cc3686e4e4fd44a7e09",
    "dataset": "common-pile/library_of_congress",
    "revision": "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7",
    "shard_path": "data/00000_loc_books.jsonl.gz",
    "shard_compressed_bytes": 358594502,
    "shard_lfs_sha256": (
        "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
    ),
    "source_family": "en.loc.selected-digitized-books",
}
RECORD_CONTRACT = {
    "source_value": "loc_books",
    "expected_license": "Public Domain",
    "expected_language": "english",
    "record_id_role": "LCCN",
    "item_url_template": "https://www.loc.gov/item/{lccn}",
    "text_file_url_allowed_hosts": ["tile.loc.gov", "tiles.loc.gov"],
    "required_metadata_fields": ["license", "language", "year", "item_url", "text_file_url"],
}
PRIMARY_EVIDENCE = [
    {
        "authority": "Library of Congress",
        "url": (
            "https://www.loc.gov/collections/selected-digitized-books/"
            "about-this-collection/rights-and-access/"
        ),
        "title": (
            "Rights and Access | Selected Digitized Books | Digital Collections | "
            "Library of Congress"
        ),
        "accessed_utc": "2026-09-11T03:10:00Z",
        "supported_fact": (
            "Selected Digitized Books are identified by the Library of Congress as "
            "public-domain books that are free to use and reuse; the Library requests "
            "a Library of Congress credit line."
        ),
    },
    {
        "authority": "Library of Congress",
        "url": (
            "https://www.loc.gov/collections/selected-digitized-books/?fa=language:english"
        ),
        "title": "Selected Digitized Books | Digital Collections | Library of Congress",
        "accessed_utc": "2026-09-11T03:10:00Z",
        "supported_fact": (
            "The audited Common Pile metadata collector enumerates the official Selected "
            "Digitized Books collection through this English-filtered loc.gov collection endpoint."
        ),
    },
]
PROJECT_DECISION = {
    "decision": "CONDITIONAL_SOURCE_ADMISSION",
    "scope": "SOURCE_POLICY_ONLY",
    "legal_conclusion_claimed": False,
    "dataset_package_license_is_training_authority": False,
    "repository_code_license_is_dataset_license": False,
    "conditional_admission_requires_all_record_contract_checks": True,
    "collector_or_revision_drift_fails_closed": True,
    "rights_or_language_or_url_drift_fails_closed": True,
}
TRUTH_BOUNDARY = {
    "payload_source_admission_executed": False,
    "source_capacity_bytes_credited": 0,
    "canonical_corpus_admitted": False,
    "family_credit": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "evaluation_authorized_bytes": 0,
    "final_test_payload_accessed": False,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}
ROOT_KEYS = {
    "schema_version",
    "authority_id",
    "status",
    "execution_profile",
    "generic_registry",
    "upstream_code",
    "dataset_contract",
    "record_contract",
    "primary_evidence",
    "project_decision",
    "truth_boundary",
    "policy_identity_sha256",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LocSourceRightsError(ValueError):
    """Raised when the LoC source-rights authority stops being fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocSourceRightsError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def policy_identity(payload: Mapping[str, Any]) -> str:
    """Return the source-policy identity, excluding the identity field itself."""
    body = dict(payload)
    body.pop("policy_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """Return Git's immutable blob identity for raw file bytes."""
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _is_exact_type(value: object, expected_type: type[object]) -> bool:
    return value.__class__ is expected_type


def _require_exact_scalar(actual: object, expected: object, field: str) -> None:
    _require(
        _is_exact_type(actual, expected.__class__) and actual == expected,
        f"{field} drift",
    )


def _require_exact_mapping(
    actual: object,
    expected: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    _require(_is_exact_type(actual, dict), f"{field} must be an object")
    assert isinstance(actual, dict)
    _require(set(actual) == set(expected), f"{field} schema drift")
    for key, wanted in expected.items():
        value = actual[key]
        if isinstance(wanted, list):
            _require(_is_exact_type(value, list), f"{field}.{key} must be a list")
            _require(value == wanted, f"{field}.{key} drift")
        else:
            _require_exact_scalar(value, wanted, f"{field}.{key}")
    return actual


def _require_exact_evidence(actual: object) -> None:
    _require(_is_exact_type(actual, list), "primary_evidence must be a list")
    assert isinstance(actual, list)
    _require(len(actual) == len(PRIMARY_EVIDENCE), "primary_evidence count drift")
    for index, wanted in enumerate(PRIMARY_EVIDENCE):
        _require_exact_mapping(actual[index], wanted, f"primary_evidence[{index}]")
        parts = urlsplit(wanted["url"])
        _require(
            parts.scheme == "https" and parts.hostname == "www.loc.gov",
            f"primary_evidence[{index}] is not an official https loc.gov source",
        )


def validate_policy(payload: Mapping[str, Any]) -> str:
    """Validate the immutable, zero-credit LoC source-rights qualification."""
    _require(_is_exact_type(payload, dict), "policy root must be an object")
    _require(set(payload) == ROOT_KEYS, "policy schema drift")
    _require_exact_scalar(payload.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact_scalar(payload.get("authority_id"), AUTHORITY_ID, "authority_id")
    _require_exact_scalar(payload.get("status"), STATUS, "status")
    _require_exact_scalar(payload.get("execution_profile"), "LOCAL_FREE", "execution_profile")

    _require_exact_mapping(payload.get("generic_registry"), GENERIC_REGISTRY, "generic_registry")
    upstream = _require_exact_mapping(payload.get("upstream_code"), UPSTREAM_CODE, "upstream_code")
    dataset = _require_exact_mapping(
        payload.get("dataset_contract"), DATASET_CONTRACT, "dataset_contract"
    )
    _require_exact_mapping(payload.get("record_contract"), RECORD_CONTRACT, "record_contract")
    _require_exact_evidence(payload.get("primary_evidence"))
    _require_exact_mapping(payload.get("project_decision"), PROJECT_DECISION, "project_decision")
    _require_exact_mapping(payload.get("truth_boundary"), TRUTH_BOUNDARY, "truth_boundary")

    for name in ("revision", "collector_blob_sha1", "metadata_blob_sha1", "readme_blob_sha1"):
        _require(bool(HEX40.fullmatch(str(upstream[name]))), f"upstream_code.{name} must be sha1")
    for name in ("origin_head", "origin_config_blob_sha1", "revision"):
        _require(bool(HEX40.fullmatch(str(dataset[name]))), f"dataset_contract.{name} must be sha1")
    _require(
        bool(HEX64.fullmatch(str(dataset["shard_lfs_sha256"]))),
        "dataset_contract.shard_lfs_sha256 must be sha256",
    )
    _require(
        _is_exact_type(dataset["shard_compressed_bytes"], int)
        and dataset["shard_compressed_bytes"] > 0,
        "dataset_contract.shard_compressed_bytes must be a positive exact int",
    )

    identity = payload.get("policy_identity_sha256")
    _require(
        _is_exact_type(identity, str) and bool(HEX64.fullmatch(identity)),
        "invalid policy identity",
    )
    _require(identity == POLICY_IDENTITY, "policy identity constant drift")
    _require(policy_identity(payload) == identity, "policy identity mismatch")
    return identity


def _find_registry_row(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = registry.get("sources")
    _require(_is_exact_type(rows, list), "generic registry sources missing")
    assert isinstance(rows, list)
    matches = [
        row
        for row in rows
        if _is_exact_type(row, dict) and row.get("key") == GENERIC_REGISTRY["source_key"]
    ]
    _require(len(matches) == 1, "generic registry LoC row missing or duplicated")
    return matches[0]


def validate_repository_bindings(
    payload: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> None:
    """Bind the policy to the exact live-main generic registry without network access."""
    validate_policy(payload)
    root = Path(repo_root)
    registry_path = root / str(GENERIC_REGISTRY["path"])
    raw = registry_path.read_bytes()
    _require(git_blob_sha1(raw) == GENERIC_REGISTRY["blob_sha1"], "generic registry blob drift")
    registry = json.loads(raw.decode("utf-8"))
    _require(_is_exact_type(registry, dict), "generic registry root must be an object")
    row = _find_registry_row(registry)

    expected_row_fields = {
        "hf_dataset": "common-pile/library_of_congress",
        "project_review_status": "REVIEW_REQUIRED",
        "canonical_training_authorized": False,
        "credited_bytes": 0,
        "authorized_loss_positions": 0,
        "evaluation_role": "TRAINING_CANDIDATE_ONLY",
        "final_test_excluded": True,
        "collector_path_status": "PRESENT_AT_AUDITED_CODE_COMMIT",
        "audited_collector_path": "sources/loc_books/books.py",
    }
    for field, expected in expected_row_fields.items():
        _require_exact_scalar(row.get(field), expected, f"generic registry row.{field}")


def load_and_validate(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load JSON, validate exact policy, and optionally validate repository bindings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(_is_exact_type(payload, dict), "policy root must be an object")
    validate_policy(payload)
    if repo_root is not None:
        validate_repository_bindings(payload, repo_root)
    return payload
