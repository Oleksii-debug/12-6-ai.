"""Validate the bounded Typer source authority without granting corpus capacity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/data/next100_052_typer_source_authority_v2.json")
SCHEMA_VERSION = "12-6.next100-052-typer-source-authority.v2"
SOURCE_HEAD_SHA = "1ad3387fa21ce208c6553c2a460573ae5648eb7b"
RECEIPT_FILE_SHA256 = "742068b814a3617e38273bd2001939a91a855309f9811b042d77eea297c6ef42"
RECEIPT_REPORT_SHA256 = "3bc54e23b63d975ff31cfbf2762679974f60ea06729ab7291e6cb765a613bf00"

SOURCE = {
    "repository": "fastapi/typer",
    "repository_url": "https://github.com/fastapi/typer",
    "repository_id": 229937405,
    "release": "0.27.1",
    "commit": "fe2aa0e2f9c853de378e60ca24ec3b256144decf",
    "source_id": "code.fastapi.typer.utils",
    "source_family": "github:fastapi/typer",
    "path": "typer/utils.py",
    "size_bytes": 7599,
    "git_blob_sha1": "addf9334d4210a9cddc9e5608ae446417d372eb0",
    "raw_sha256": "c272750d65c114c9f12c768c37f1c3627b712d16f2d0afabbb1eb76a89072272",
}
LICENSE = {
    "license_id": "MIT",
    "path": "LICENSE",
    "git_blob_sha1": "a7694736cf37716aafec14b24aa8d6316ebe07a3",
    "sha256": "58992cebcf8dfb6e40c4e2112ed12126c243666dca3912a3d78b7ecac4859d49",
    "size_bytes": 1086,
    "model_training": "ALLOWED",
    "redistribution": "ALLOWED_WITH_NOTICE",
    "evaluation": "NOT_AUTHORIZED_BY_THIS_AUTHORITY",
}
LINEAGE_EXCLUSIONS = {
    "typer/_click/**": "CLICK_DERIVED_OR_VENDORED_ZERO_TYPER_CAPACITY",
    "typer/_typing.py": "COPIED_REDUCED_FROM_PYDANTIC_1_9_2_ZERO_TYPER_CAPACITY",
    "all_other_typer_objects": "NOT_AUTHORIZED_BY_THIS_BOUNDED_SOURCE_AUTHORITY",
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


def _validate_truth_boundary(truth: Any, prefix: str) -> list[str]:
    errors: list[str] = []
    expected_keys = ZERO_INT_FIELDS | FALSE_BOOL_FIELDS
    if not isinstance(truth, dict):
        return [f"{prefix}_missing"]
    if set(truth) != expected_keys:
        errors.append(f"{prefix}_keys_mismatch")
        return errors
    for key in ZERO_INT_FIELDS:
        if not _exact_int(truth.get(key), 0):
            errors.append(f"{prefix}_{key}_must_be_exact_int_zero")
    for key in FALSE_BOOL_FIELDS:
        if truth.get(key) is not False:
            errors.append(f"{prefix}_{key}_must_be_false")
    return errors


def validate_typer_source_authority(config: Any, receipt_bytes: bytes) -> list[str]:
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
        "lineage_exclusions",
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
    if config.get("bounded_source") != SOURCE:
        errors.append("bounded_source_identity_mismatch")
    if config.get("license") != LICENSE:
        errors.append("license_identity_or_use_boundary_mismatch")
    if config.get("lineage_exclusions") != LINEAGE_EXCLUSIONS:
        errors.append("lineage_exclusions_mismatch")

    expected_project = {
        "swarm_control_issue": 723,
        "convergence_issue": 1983,
        "source_pr": 476,
        "historical_source_audit": "NOT_PERFORMED",
        "fresh_current_head_audit_required": True,
    }
    if config.get("project_authority") != expected_project:
        errors.append("project_authority_mismatch")

    historical = config.get("historical_execution")
    if not isinstance(historical, dict):
        errors.append("historical_execution_missing")
    else:
        expected_historical = {
            "source_head_sha": SOURCE_HEAD_SHA,
            "pull_request_merge_sha": "6d6dc1ce94e1e8f6f1b9e40d091f038b24ba87e1",
            "run_id": 32999414943,
            "job_id": 98276978124,
            "artifact_id": 9618793476,
            "artifact_zip_sha256": "58b444adedf11c1b79731ee98f1e4c0f7d9469c3a6a773a76ad45390eb134f41",
            "receipt_path": "evidence/data/next100_052_typer_terminal_source_receipt_v1.json",
            "receipt_file_sha256": RECEIPT_FILE_SHA256,
            "receipt_report_sha256": RECEIPT_REPORT_SHA256,
            "dedicated_verifier_conclusion": "SUCCESS",
            "historical_evidence_verdict": "ADMIT",
        }
        if set(historical) != set(expected_historical):
            errors.append("historical_execution_keys_mismatch")
        for key, expected in expected_historical.items():
            actual = historical.get(key)
            valid = (
                _exact_int(actual, expected)
                if isinstance(expected, int)
                else actual == expected
            )
            if not valid:
                errors.append(f"historical_{key}_mismatch")

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

        expected_receipt_scalars = {
            "schema_version": "12-6.next100-052-typer-terminal-source-receipt.v1",
            "authority": "EXTERNAL_REAL_CODE_SOURCE_HISTORICAL_EXECUTION_LOCAL_FREE",
            "source_pr": 476,
            "source_head_sha": SOURCE_HEAD_SHA,
            "pull_request_merge_sha": "6d6dc1ce94e1e8f6f1b9e40d091f038b24ba87e1",
            "run_id": 32999414943,
            "job_id": 98276978124,
            "artifact_id": 9618793476,
            "artifact_zip_sha256": "58b444adedf11c1b79731ee98f1e4c0f7d9469c3a6a773a76ad45390eb134f41",
            "terminal": True,
            "verdict": "ADMIT",
            "local_free_only": True,
        }
        for key, expected in expected_receipt_scalars.items():
            actual = receipt.get(key)
            if isinstance(expected, int) and not isinstance(expected, bool):
                valid = _exact_int(actual, expected)
            else:
                valid = actual == expected
            if not valid:
                errors.append(f"historical_receipt_{key}_mismatch")

        expected_upstream = {
            "repository": SOURCE["repository"],
            "repository_id": SOURCE["repository_id"],
            "release": SOURCE["release"],
            "commit": SOURCE["commit"],
        }
        if receipt.get("upstream") != expected_upstream:
            errors.append("historical_receipt_upstream_identity_mismatch")

        expected_object = {
            "source_id": SOURCE["source_id"],
            "source_family": SOURCE["source_family"],
            "path": SOURCE["path"],
            "size_bytes": SOURCE["size_bytes"],
            "git_blob_sha1": SOURCE["git_blob_sha1"],
            "raw_sha256": SOURCE["raw_sha256"],
            "parse_valid": True,
            "secret_hits": [],
            "privacy_hits": [],
        }
        if receipt.get("object") != expected_object:
            errors.append("historical_receipt_object_identity_or_gate_mismatch")
        if receipt.get("license") != LICENSE:
            errors.append("historical_receipt_license_identity_or_use_boundary_mismatch")

        expected_evaluation = {
            "authority": "EVAL-289",
            "authority_head_sha": "1c870e5e02bf48891ca599b0b3f3bfe6e84425bc",
            "active_reserved_code_objects": 0,
            "selected_object_overlap": False,
            "current_authority": False,
        }
        if receipt.get("historical_evaluation") != expected_evaluation:
            errors.append("historical_receipt_evaluation_boundary_mismatch")

        expected_inputs = {
            "manifest_git_blob_sha1": "0960b918f26709f6ff80cf53fdb63d04fc6efdca",
            "validator_git_blob_sha1": "1ec89d56b2fe85f53d82ef23c686541efd1264b5",
            "workflow_git_blob_sha1": "1b9c7cae537e0aab6983bcc2b8a22b911f97a1da",
        }
        if receipt.get("historical_inputs") != expected_inputs:
            errors.append("historical_receipt_input_identity_mismatch")

        dedup_receipt = receipt.get("historical_dedup")
        expected_dedup_receipt = {
            "failures": [],
            "evaluation_reservation_overlaps": [],
            "click": {
                "repository": "pallets/click",
                "commit": "b2e30a175449cfda909ee4fbf4a29a6a071cad53",
                "exact_blob_hits_anywhere_in_tree": [],
                "max_skeleton_jaccard": 0.11954187544738726,
                "max_skeleton_containment": 0.5634615384615385,
            },
            "fastapi": {
                "repository": "fastapi/fastapi",
                "commit": "49033471594ea5d99a80abdf1043231b7791ee49",
                "exact_blob_hits_anywhere_in_tree": [],
                "max_skeleton_jaccard": 0.1111111111111111,
                "max_skeleton_containment": 0.3,
            },
        }
        if dedup_receipt != expected_dedup_receipt:
            errors.append("historical_receipt_dedup_or_lineage_mismatch")
        errors.extend(
            _validate_truth_boundary(
                receipt.get("scientific_boundary"), "historical_receipt_scientific_boundary"
            )
        )
    elif "historical_receipt_json_invalid" not in errors:
        errors.append("historical_receipt_root_must_be_object")

    expected_historical_dedup = {
        "historical_only": True,
        "current_global_authority": False,
        "baseline_registry_authority": "DATA-287_HISTORICAL_ONLY",
        "comparison_families": [
            "github:encode/httpx",
            "github:psf/requests",
            "github:pallets/click",
            "github:fastapi/fastapi",
        ],
        "exact_duplicate": False,
        "lineage_exact_blob_collision": False,
        "near_duplicate_thresholds": {
            "python_skeleton_5gram_jaccard": 0.85,
            "python_skeleton_5gram_containment": 0.90,
        },
    }
    if config.get("historical_dedup") != expected_historical_dedup:
        errors.append("historical_dedup_boundary_mismatch")

    composition = config.get("current_composition")
    if not isinstance(composition, dict):
        errors.append("current_composition_missing")
    else:
        expected_strings = {
            "global_dedup": "REQUIRED_NOT_EXECUTED_BY_THIS_AUTHORITY",
            "evaluation_firewall": "REQUIRED_AT_COMPOSITION",
            "current_retained_corpus_cleanliness": "NOT_PROVEN_BY_THIS_AUTHORITY",
            "source_authority_status": "HISTORICAL_EXECUTION_BOUND_CURRENT_MAIN_CARRIER",
        }
        for key, expected in expected_strings.items():
            if composition.get(key) != expected:
                errors.append(f"current_composition_{key}_mismatch")
        for key in (
            "canonical_capacity_credit_bytes",
            "canonical_family_credit",
            "canonical_files_credit",
        ):
            if not _exact_int(composition.get(key), 0):
                errors.append(f"current_composition_{key}_must_be_exact_int_zero")
        if set(composition) != set(expected_strings) | {
            "canonical_capacity_credit_bytes",
            "canonical_family_credit",
            "canonical_files_credit",
        }:
            errors.append("current_composition_keys_mismatch")

    errors.extend(_validate_truth_boundary(config.get("truth_boundary"), "truth_boundary"))
    return errors


def validate_typer_source_authority_files(repo_root: str | Path = ".") -> list[str]:
    """Load the repository-bound config and receipt and return blockers."""
    root = Path(repo_root)
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    receipt_path = Path(config["historical_execution"]["receipt_path"])
    return validate_typer_source_authority(config, (root / receipt_path).read_bytes())
