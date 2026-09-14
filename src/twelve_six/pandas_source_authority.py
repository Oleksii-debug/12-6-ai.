"""Validate the bounded Pandas source authority without granting corpus capacity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/data/next100_050_pandas_source_authority_v2.json")
SCHEMA_VERSION = "12-6.next100-050-pandas-source-authority.v2"
SOURCE_HEAD_SHA = "61af21e4c32bfbce124886c3c684cf85be290c19"
RECEIPT_FILE_SHA256 = "3cc3ec4a8e6efc10aeddb2510e879842d63269020d6476e07906dc79b9e5ee49"
RECEIPT_REPORT_SHA256 = "2293139ca1131842e754a119ec7b847e8ee7c0703ab4a4bd3a9ce7fb332f113e"
SOURCE = {
    "repository": "pandas-dev/pandas",
    "repository_url": "https://github.com/pandas-dev/pandas",
    "commit": "0b54092c19202b579a1114e2d74f4b8028d5d573",
    "source_id": "code.pandas-dev.pandas.core.accessor",
    "source_family": "github:pandas-dev/pandas",
    "path": "pandas/core/accessor.py",
    "size_bytes": 15837,
    "git_blob_sha1": "26560f84000f24aee377bc761db0d2686452f813",
    "raw_sha256": "d737d2c0239e0693d992a26e1ee9e56f73dd07e5fa05ff6079e70d9c3677299e",
    "normalized_sha256": "d737d2c0239e0693d992a26e1ee9e56f73dd07e5fa05ff6079e70d9c3677299e",
    "source_manifest_sha256": "ff4b75a8d956e2d0773badfda28cffbd45a1d282b7468b475e51442ee65c53f9",
}
LICENSE = {
    "license_id": "BSD-3-Clause",
    "path": "LICENSE",
    "git_blob_sha1": "bd1cc2a30c626d0bbe43ab6ec1c4744c2e2df831",
    "sha256": "9850f6b5bdef7346503065807b878deb8058b976c5c329f4c4d23bf4796ce9ae",
    "model_training": "ALLOWED",
    "redistribution": "ALLOWED",
    "evaluation": "NOT_SEPARATELY_ADMITTED",
}
ZERO_INT_FIELDS = {
    "authorized_optimized_target_exposure",
    "optimizer_updates_executed_on_real_targets",
}
FALSE_BOOL_FIELDS = {
    "current_retained_corpus_launch_authoritative",
    "tokenizer_fit_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
}


def _canonical_json_line(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python bool/int/float aliasing."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_pandas_source_authority(config: Any, receipt_bytes: bytes) -> list[str]:
    """Return fail-closed blockers for the current-main convergence authority."""
    if not isinstance(config, dict):
        return ["config_root_must_be_object"]
    errors: list[str] = []
    expected_top = {
        "schema_version",
        "execution_profile",
        "project_authority",
        "bounded_source",
        "license",
        "historical_execution",
        "historical_dedup",
        "current_composition",
        "truth_boundary",
    }
    if set(config) != expected_top:
        errors.append("config_top_level_keys_mismatch")
    if config.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if config.get("execution_profile") != "LOCAL_FREE":
        errors.append("execution_profile_must_be_local_free")
    if not _strict_json_equal(config.get("bounded_source"), SOURCE):
        errors.append("bounded_source_identity_mismatch")
    if not _strict_json_equal(config.get("license"), LICENSE):
        errors.append("license_identity_or_use_boundary_mismatch")

    project = config.get("project_authority")
    expected_project = {
        "swarm_control_issue": 723,
        "convergence_issue": 1973,
        "source_pr": 463,
        "source_audit_issue": 1959,
        "source_audit_verdict": "PASS_FOR_INTEGRATION_SOURCE_AUTHORITY",
    }
    if not _strict_json_equal(project, expected_project):
        errors.append("project_authority_mismatch")

    historical = config.get("historical_execution")
    if not isinstance(historical, dict):
        errors.append("historical_execution_missing")
        receipt_path = None
    else:
        expected_scalars = {
            "source_head_sha": SOURCE_HEAD_SHA,
            "run_id": 32998203692,
            "job_id": 98272853739,
            "artifact_id": 9617946687,
            "artifact_zip_sha256": (
                "4ed188020cd0c0a51195faad1170a0af521f15cea34ea90c8ad5e1a5dbd1a2e2"
            ),
            "receipt_file_sha256": RECEIPT_FILE_SHA256,
            "receipt_report_sha256": RECEIPT_REPORT_SHA256,
            "historical_config_git_blob_sha1": "ecd747cfdb97e9b5cf11b895477bcb4af7f77aaf",
            "historical_config_sha256": (
                "5c08d522a12e2bc9f9ed096f07739c06d455f30287340d9f096781cf76032602"
            ),
        }
        expected_historical_keys = set(expected_scalars) | {"receipt_path"}
        if set(historical) != expected_historical_keys:
            errors.append("historical_execution_keys_mismatch")
        for key, expected in expected_scalars.items():
            actual = historical.get(key)
            valid = _strict_json_equal(actual, expected)
            if not valid:
                errors.append(f"historical_{key}_mismatch")
        receipt_path = historical.get("receipt_path")
        if receipt_path != "evidence/data/next100_050_pandas_terminal_source_authority_v1.json":
            errors.append("historical_receipt_path_mismatch")

    if _sha256(receipt_bytes) != RECEIPT_FILE_SHA256:
        errors.append("historical_receipt_file_sha256_mismatch")
        receipt: Any = None
    else:
        try:
            receipt = json.loads(receipt_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            receipt = None
            errors.append("historical_receipt_json_invalid")
    if isinstance(receipt, dict):
        report_hash = receipt.get("report_sha256")
        core = dict(receipt)
        core.pop("report_sha256", None)
        if (
            report_hash != RECEIPT_REPORT_SHA256
            or _sha256(_canonical_json_line(core)) != report_hash
        ):
            errors.append("historical_receipt_report_sha256_mismatch")
        receipt_source = receipt.get("object")
        receipt_upstream = receipt.get("upstream")
        receipt_license = receipt.get("license")
        if not isinstance(receipt_source, dict) or any(
            not _strict_json_equal(receipt_source.get(key), SOURCE[key])
            for key in (
                "source_id",
                "source_family",
                "path",
                "size_bytes",
                "git_blob_sha1",
                "raw_sha256",
                "normalized_sha256",
                "source_manifest_sha256",
            )
        ):
            errors.append("historical_receipt_source_identity_mismatch")
        if not isinstance(receipt_upstream, dict) or any(
            not _strict_json_equal(receipt_upstream.get(key), SOURCE[key])
            for key in ("repository", "repository_url", "commit")
        ):
            errors.append("historical_receipt_upstream_identity_mismatch")
        if not isinstance(receipt_license, dict) or any(
            not _strict_json_equal(receipt_license.get(key), LICENSE[key])
            for key in ("license_id", "path", "git_blob_sha1", "sha256")
        ):
            errors.append("historical_receipt_license_identity_mismatch")
        if receipt.get("source_sha") != SOURCE_HEAD_SHA or receipt.get("verdict") != "ADMIT":
            errors.append("historical_receipt_verdict_or_head_mismatch")
        if receipt.get("terminal") is not True or receipt.get("local_free_only") is not True:
            errors.append("historical_receipt_execution_boundary_mismatch")
    elif "historical_receipt_json_invalid" not in errors:
        errors.append("historical_receipt_root_must_be_object")

    dedup = config.get("historical_dedup")
    expected_dedup = {
        "historical_only": True,
        "baseline_registry_path": "configs/data/data287_external_snapshot_registry_v2.json",
        "baseline_registry_git_blob_sha1": "d48e5281d8b4949d4c6f256d84518305033c2476",
        "baseline_registry_sha256": (
            "4d684ca79ac454778df7f43a8c9c710e4daac0e1e38b984c9158b9536fdc02d4"
        ),
        "exact_duplicate": False,
        "near_duplicate": False,
        "near_threshold": 0.85,
        "comparison_families": ["github:encode/httpx", "github:psf/requests"],
        "current_global_authority": False,
    }
    if not _strict_json_equal(dedup, expected_dedup):
        errors.append("historical_dedup_boundary_mismatch")

    composition = config.get("current_composition")
    if not isinstance(composition, dict):
        errors.append("current_composition_missing")
    else:
        expected_strings = {
            "global_dedup": "REQUIRED_NOT_EXECUTED_BY_THIS_AUTHORITY",
            "evaluation_firewall": "REQUIRED_AT_COMPOSITION",
            "current_retained_corpus_cleanliness": "NOT_PROVEN_BY_THIS_AUTHORITY",
            "source_authority_status": "VERIFIED_BOUNDED_SOURCE_AUTHORITY",
        }
        zero_credit_keys = {
            "canonical_capacity_credit_bytes",
            "canonical_family_credit",
            "canonical_files_credit",
        }
        if set(composition) != set(expected_strings) | zero_credit_keys:
            errors.append("current_composition_keys_mismatch")
        for key, expected in expected_strings.items():
            if not _strict_json_equal(composition.get(key), expected):
                errors.append(f"current_composition_{key}_mismatch")
        for key in zero_credit_keys:
            if not _exact_int(composition.get(key), 0):
                errors.append(f"current_composition_{key}_must_be_exact_int_zero")

    truth = config.get("truth_boundary")
    expected_truth_keys = ZERO_INT_FIELDS | FALSE_BOOL_FIELDS
    if not isinstance(truth, dict):
        errors.append("truth_boundary_missing")
    elif set(truth) != expected_truth_keys:
        errors.append("truth_boundary_keys_mismatch")
    else:
        for key in ZERO_INT_FIELDS:
            if not _exact_int(truth.get(key), 0):
                errors.append(f"{key}_must_be_exact_int_zero")
        for key in FALSE_BOOL_FIELDS:
            if truth.get(key) is not False:
                errors.append(f"{key}_must_be_false")
    return errors


def validate_pandas_source_authority_files(repo_root: str | Path = ".") -> list[str]:
    """Load the repository-bound config and receipt and return blockers."""
    root = Path(repo_root)
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    receipt_path = Path(config["historical_execution"]["receipt_path"])
    return validate_pandas_source_authority(config, (root / receipt_path).read_bytes())
