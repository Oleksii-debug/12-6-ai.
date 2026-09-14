"""Fail-closed source-rights qualification for Common Pile arXiv abstracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

AUTHORITY_ID = "D03-COMMON-PILE-ARXIV-ABSTRACTS-SOURCE-RIGHTS-V1"
SCHEMA_VERSION = "12-6.d03-common-pile-arxiv-abstracts-source-rights.v1"
STATUS = "SOURCE_POLICY_QUALIFIED_CONDITIONAL"
POLICY_IDENTITY = "b667a71cf476282d45476babc71b9d401ffdaaa38e03b9fac8a6319614b57442"

GENERIC_REGISTRY = {
    "path": "configs/data/common_pile_source_rights_v1.json",
    "blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "source_key": "arxiv_abstracts",
    "expected_project_review_status": "REVIEW_REQUIRED",
    "expected_rights_basis_class": "METADATA_OPEN_LICENSE",
    "expected_license_signal": "CC0",
}
UPSTREAM_COMMON_PILE = {
    "repository": "https://github.com/r-three/common-pile",
    "revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "mixer_config_path": "filtering/mixer_configs/arxiv_abstracts.json",
    "mixer_config_blob_sha1": "d0ecfad4c364ef46ba3737cf8e29b961fe620590",
    "licenses_path": "common_pile/licenses.py",
    "licenses_blob_sha1": "668e3c29990b69f7f9185b67b73c949d63c76b3e",
}
DATASET_CONTRACT = {
    "origin_pr": 879,
    "origin_head": "3a238e461adfacf5b87d6a7078bf51462825a268",
    "origin_config_path": "configs/data/d03_common_pile_arxiv_abstracts_bounded_v1.json",
    "origin_config_blob_sha1": "4a819b3980634c8a2ba7de55cf854d44bcfd5757",
    "dataset": "common-pile/arxiv_abstracts",
    "revision": "46de78c48636c0b46f60049dfd1c5a3710d233f9",
    "shard_path": "00003_arxiv-abstracts.jsonl.gz",
    "shard_compressed_bytes": 232616926,
    "shard_lfs_sha256": (
        "3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7"
    ),
    "source_family": "scientific.arxiv.abstracts",
}
RECORD_CONTRACT = {
    "source_value": "arxiv-abstracts",
    "expected_metadata_license": (
        "Creative Commons Zero - Public Domain - "
        "https://creativecommons.org/publicdomain/zero/1.0/"
    ),
    "required_row_fields": ["id", "text", "source", "created", "added", "metadata"],
    "rights_scope": "DESCRIPTIVE_METADATA_INCLUDING_ABSTRACT",
}
PRIMARY_EVIDENCE = [
    {
        "authority": "arXiv",
        "url": "https://info.arxiv.org/help/api/tou.html",
        "title": "Terms of Use for arXiv APIs",
        "accessed_utc": "2026-09-11T22:10:00Z",
        "supported_fact": (
            "arXiv permits reuse of descriptive e-print metadata under CC0 1.0 and "
            "explicitly includes abstracts in descriptive metadata; this does not grant "
            "permission to redistribute full e-print content."
        ),
    }
]
PROJECT_DECISION = {
    "decision": "CONDITIONAL_SOURCE_ADMISSION",
    "scope": "SOURCE_POLICY_ONLY",
    "legal_conclusion_claimed": False,
    "dataset_package_license_is_training_authority": False,
    "repository_code_license_is_dataset_license": False,
    "cc0_scope_limited_to_descriptive_metadata_including_abstract": True,
    "full_eprint_rights_inferred": False,
    "conditional_admission_requires_exact_record_cc0": True,
    "registry_or_dataset_drift_fails_closed": True,
}
TRUTH_BOUNDARY = {
    "payload_source_admission_executed": False,
    "source_capacity_bytes_credited": 0,
    "canonical_corpus_admitted": False,
    "family_credit": False,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
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
    "upstream_common_pile",
    "dataset_contract",
    "record_contract",
    "primary_evidence",
    "project_decision",
    "truth_boundary",
    "policy_identity_sha256",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ArxivAbstractsSourceRightsError(ValueError):
    """Raised when the arXiv-abstracts rights authority stops being fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArxivAbstractsSourceRightsError(message)


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
            parts.scheme == "https" and parts.hostname == "info.arxiv.org",
            f"primary_evidence[{index}] is not an official https arxiv.org source",
        )


def validate_policy(payload: Mapping[str, Any]) -> str:
    """Validate the immutable, zero-credit arXiv-abstracts rights qualification."""
    _require(_is_exact_type(payload, dict), "policy root must be an object")
    _require(set(payload) == ROOT_KEYS, "policy schema drift")
    _require_exact_scalar(payload.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_exact_scalar(payload.get("authority_id"), AUTHORITY_ID, "authority_id")
    _require_exact_scalar(payload.get("status"), STATUS, "status")
    _require_exact_scalar(payload.get("execution_profile"), "LOCAL_FREE", "execution_profile")

    _require_exact_mapping(payload.get("generic_registry"), GENERIC_REGISTRY, "generic_registry")
    upstream = _require_exact_mapping(
        payload.get("upstream_common_pile"),
        UPSTREAM_COMMON_PILE,
        "upstream_common_pile",
    )
    dataset = _require_exact_mapping(
        payload.get("dataset_contract"),
        DATASET_CONTRACT,
        "dataset_contract",
    )
    _require_exact_mapping(payload.get("record_contract"), RECORD_CONTRACT, "record_contract")
    _require_exact_evidence(payload.get("primary_evidence"))
    _require_exact_mapping(payload.get("project_decision"), PROJECT_DECISION, "project_decision")
    _require_exact_mapping(payload.get("truth_boundary"), TRUTH_BOUNDARY, "truth_boundary")

    for name in ("revision", "mixer_config_blob_sha1", "licenses_blob_sha1"):
        _require(
            bool(HEX40.fullmatch(str(upstream[name]))),
            f"upstream_common_pile.{name} must be sha1",
        )
    for name in ("origin_head", "origin_config_blob_sha1", "revision"):
        _require(
            bool(HEX40.fullmatch(str(dataset[name]))),
            f"dataset_contract.{name} must be sha1",
        )
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
    _require(len(matches) == 1, "generic registry arXiv-abstracts row missing or duplicated")
    return matches[0]


def validate_repository_bindings(
    payload: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> None:
    """Bind the policy to exact live-main registry and bounded intake contracts."""
    validate_policy(payload)
    root = Path(repo_root)

    registry_path = root / str(GENERIC_REGISTRY["path"])
    registry_raw = registry_path.read_bytes()
    _require(
        git_blob_sha1(registry_raw) == GENERIC_REGISTRY["blob_sha1"],
        "generic registry blob drift",
    )
    registry = json.loads(registry_raw.decode("utf-8"))
    _require(_is_exact_type(registry, dict), "generic registry root must be an object")
    row = _find_registry_row(registry)

    expected_row_fields = {
        "hf_dataset": "common-pile/arxiv_abstracts",
        "rights_basis_class": "METADATA_OPEN_LICENSE",
        "license_or_status_signals": ["CC0"],
        "project_review_status": "REVIEW_REQUIRED",
        "canonical_training_authorized": False,
        "credited_bytes": 0,
        "authorized_loss_positions": 0,
        "evaluation_role": "TRAINING_CANDIDATE_ONLY",
        "final_test_excluded": True,
        "collector_path_status": "PRESENT_AT_AUDITED_CODE_COMMIT",
        "audited_collector_path": "sources/arxiv",
    }
    for field, expected in expected_row_fields.items():
        if isinstance(expected, list):
            _require(row.get(field) == expected, f"generic registry row.{field} drift")
        else:
            _require_exact_scalar(row.get(field), expected, f"generic registry row.{field}")

    intake_path = root / str(DATASET_CONTRACT["origin_config_path"])
    intake_raw = intake_path.read_bytes()
    _require(
        git_blob_sha1(intake_raw) == DATASET_CONTRACT["origin_config_blob_sha1"],
        "bounded intake config blob drift",
    )
    intake = json.loads(intake_raw.decode("utf-8"))
    _require(_is_exact_type(intake, dict), "bounded intake root must be an object")
    parent = intake.get("parent_rights")
    source = intake.get("source")
    _require(_is_exact_type(parent, dict), "bounded intake parent_rights missing")
    _require(_is_exact_type(source, dict), "bounded intake source missing")
    assert isinstance(parent, dict)
    assert isinstance(source, dict)

    intake_parent_expected = {
        "source_key": GENERIC_REGISTRY["source_key"],
        "hf_dataset": DATASET_CONTRACT["dataset"],
        "required_rights_basis_class": GENERIC_REGISTRY["expected_rights_basis_class"],
        "required_rights_signal": GENERIC_REGISTRY["expected_license_signal"],
        "required_project_review_status": GENERIC_REGISTRY["expected_project_review_status"],
    }
    for field, expected in intake_parent_expected.items():
        _require_exact_scalar(parent.get(field), expected, f"bounded intake parent_rights.{field}")

    intake_source_expected = {
        "revision": DATASET_CONTRACT["revision"],
        "file": DATASET_CONTRACT["shard_path"],
        "sha256": DATASET_CONTRACT["shard_lfs_sha256"],
        "bytes": DATASET_CONTRACT["shard_compressed_bytes"],
        "compression": "gzip",
        "expected_source_label": RECORD_CONTRACT["source_value"],
        "expected_row_fields": RECORD_CONTRACT["required_row_fields"],
        "expected_metadata_license": RECORD_CONTRACT["expected_metadata_license"],
    }
    for field, expected in intake_source_expected.items():
        if isinstance(expected, list):
            _require(source.get(field) == expected, f"bounded intake source.{field} drift")
        else:
            _require_exact_scalar(source.get(field), expected, f"bounded intake source.{field}")


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
