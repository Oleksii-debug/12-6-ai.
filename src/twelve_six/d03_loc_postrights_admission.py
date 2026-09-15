"""Post-rights source-admission proof for the exact D03 LoC execution lineage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.common_pile_loc_rights import (
    DATASET_CONTRACT,
    POLICY_IDENTITY,
    RECORD_CONTRACT,
)
from twelve_six.common_pile_loc_rights import load_and_validate as load_rights_policy
from twelve_six.data.common_pile_loc_intake import validate_config as validate_intake_config

SCHEMA_VERSION = "12-6.d03-loc-postrights-source-admission.v1"
AUTHORITY_ID = "D03-LOC-POSTRIGHTS-SOURCE-ADMISSION-V1"
STATUS = "PAYLOAD_SOURCE_ADMISSION_EXECUTED_ZERO_CREDIT"
CONFIG_PATH = "configs/data/d03_common_pile_loc_source_admission_v1.json"

RECEIPT_PATH = "evidence/d03-loc-real-execution-v1.json"
RECEIPT_BLOB_SHA1 = "3f7d5458135d7dfb0eb64045e1af99998d710ed8"
MATERIALIZER_PATH = "tools/materialize_d03_common_pile_loc.py"
MATERIALIZER_BLOB_SHA1 = "7d121300cd241672fd3f451451c573a98db680ed"
INTAKE_CONFIG_PATH = "configs/data/d03_common_pile_loc_intake_v1.json"
INTAKE_CONFIG_BLOB_SHA1 = "958339654703f3aab4ea9cc3686e4e4fd44a7e09"
INTAKE_MODULE_PATH = "src/twelve_six/data/common_pile_loc_intake.py"
INTAKE_MODULE_BLOB_SHA1 = "f7fe4166628dd18f8b18e5e2a412319f27170ff4"

RIGHTS_POLICY_PATH = "configs/data/d03_common_pile_loc_source_rights_v1.json"
RIGHTS_POLICY_BLOB_SHA1 = "f60a97b3041d4a0ad99fdde6f45f6062bfaf24cd"
RIGHTS_MODULE_PATH = "src/twelve_six/common_pile_loc_rights.py"
RIGHTS_MODULE_BLOB_SHA1 = "56bd02f070402b533c36ee04a3708a888c5ae7c5"

PRODUCT_PR = 1276
PRODUCT_HEAD_SHA = "cdde55b75a10639e8233de7ccccaaea335b4055a"
EXECUTION_HEAD_SHA = "8ce516160b91ba48b4dd4096e0ffeea409d221de"
RUN_ID = 34606219695
JOB_ID = 103285166822
RIGHTS_PR = 1162
RIGHTS_HEAD_SHA = "7e096ffcfd17f6b0eb215037a60370f255202c02"
RIGHTS_MERGE_SHA = "3ee45353424991e471b749f807f231cad486a797"
RIGHTS_AUDIT_ISSUE = 1167
SWARM_ISSUE = 1381
SWARM_CONTROL_ISSUE = 723

ROOT_KEYS = {
    "schema_version",
    "authority_id",
    "status",
    "execution_profile",
    "swarm",
    "historical_execution",
    "rights_authority",
    "record_contract",
    "admitted_candidate",
    "proof",
    "truth_boundary",
    "downstream_required",
}

ADMITTED_CANDIDATE = {
    "source_sha256": "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f",
    "source_bytes": 358594502,
    "examined_documents": 256,
    "candidate_sha256": "f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20",
    "candidate_file_bytes": 4680968,
    "accepted_records": 28,
    "accepted_normalized_bytes": 4499908,
    "candidate_inventory_identity_sha256": (
        "d3906e11d3779b8d151753a5a126b7578d99d298b16cc6aa3aef129b55463511"
    ),
    "contract_identity_sha256": (
        "9f6fc98e3a3b720f58963b52b3f3cf3b651bdb9e35f08fa380c99f7b7112f828"
    ),
    "payload_rematerialized_in_this_package": False,
}

TRUTH_BOUNDARY = {
    "payload_source_admission_executed": True,
    "source_admitted_candidate_records": 28,
    "source_admitted_candidate_bytes": 4499908,
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "family_credit_added": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "learned_weights_created": False,
    "evaluation_authorized_bytes": 0,
    "final_test_payload_accessed": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}

DOWNSTREAM_REQUIRED = [
    "current_global_exact_near_lineage_dedup",
    "reserved_evaluation_decontamination",
    "canonical_quality_privacy",
    "family_balance_caps",
    "cluster_safe_split",
    "deterministic_packing",
    "two_clean_build_reproducibility",
    "positive_unique_loss_accounting",
    "positive_training_authority",
]


class LocPostRightsAdmissionError(ValueError):
    """Raised when the exact LoC post-rights proof stops being fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocPostRightsAdmissionError(message)


def git_blob_sha1(data: bytes) -> str:
    """Return Git's immutable SHA-1 identity for raw blob bytes."""
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _require_exact_int(value: object, expected: int, field: str) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{field} drift",
    )


def _require_exact_mapping(
    value: object,
    expected: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    _require(type(value) is dict, f"{field} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == set(expected), f"{field} schema drift")
    for key, wanted in expected.items():
        actual = value[key]
        if type(wanted) is int:
            _require_exact_int(actual, wanted, f"{field}.{key}")
        else:
            _require(
                type(actual) is type(wanted) and actual == wanted,
                f"{field}.{key} drift",
            )
    return value


def _read_bound_file(root: Path, relpath: str, expected_blob: str) -> bytes:
    path = (root / relpath).resolve()
    _require(path.is_relative_to(root), f"{relpath} escaped repository root")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LocPostRightsAdmissionError(f"cannot read {relpath}") from exc
    _require(git_blob_sha1(data) == expected_blob, f"{relpath} blob drift")
    return data


def _git_show_bound_file(
    root: Path,
    revision: str,
    relpath: str,
    expected_blob: str,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relpath}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocPostRightsAdmissionError(
            f"cannot resolve historical file {revision}:{relpath}"
        ) from exc
    data = result.stdout
    _require(git_blob_sha1(data) == expected_blob, f"historical {relpath} blob drift")
    return data


def _json_from_bytes(raw: bytes, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocPostRightsAdmissionError(f"{field} is not valid UTF-8 JSON") from exc
    _require(type(payload) is dict, f"{field} root must be an object")
    return payload


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    _require(
        receipt.get("schema_version") == "12-6.d03-loc-real-execution.v1",
        "receipt schema drift",
    )
    _require(receipt.get("execution_profile") == "LOCAL_FREE", "receipt execution profile drift")
    _require_exact_int(receipt.get("product_pr"), PRODUCT_PR, "receipt.product_pr")

    execution = receipt.get("successful_execution")
    _require(type(execution) is dict, "receipt successful_execution missing")
    assert isinstance(execution, dict)
    _require_exact_int(execution.get("workflow_run_id"), RUN_ID, "receipt.run_id")
    _require_exact_int(execution.get("job_id"), JOB_ID, "receipt.job_id")
    _require(
        execution.get("execution_head_sha") == EXECUTION_HEAD_SHA,
        "receipt execution head drift",
    )
    _require(execution.get("conclusion") == "success", "receipt execution did not succeed")

    source = receipt.get("source")
    _require(type(source) is dict, "receipt source missing")
    assert isinstance(source, dict)
    expected_source = {
        "dataset": DATASET_CONTRACT["dataset"],
        "revision": DATASET_CONTRACT["revision"],
        "shard": DATASET_CONTRACT["shard_path"],
        "source_acquisitions": 2,
        "compressed_bytes_each": DATASET_CONTRACT["shard_compressed_bytes"],
        "sha256_each": DATASET_CONTRACT["shard_lfs_sha256"],
    }
    _require_exact_mapping(source, expected_source, "receipt.source")

    materialization = receipt.get("materialization")
    _require(type(materialization) is dict, "receipt materialization missing")
    assert isinstance(materialization, dict)
    expected_materialization = {
        "examined_documents": ADMITTED_CANDIDATE["examined_documents"],
        "accepted_documents": ADMITTED_CANDIDATE["accepted_records"],
        "accepted_normalized_utf8_bytes": ADMITTED_CANDIDATE["accepted_normalized_bytes"],
        "candidate_file_bytes": ADMITTED_CANDIDATE["candidate_file_bytes"],
        "candidate_file_sha256": ADMITTED_CANDIDATE["candidate_sha256"],
        "candidate_inventory_identity_sha256": ADMITTED_CANDIDATE[
            "candidate_inventory_identity_sha256"
        ],
        "contract_identity_sha256": ADMITTED_CANDIDATE["contract_identity_sha256"],
        "full_shard_hash_verified": True,
    }
    for key, wanted in expected_materialization.items():
        actual = materialization.get(key)
        if type(wanted) is int:
            _require_exact_int(actual, wanted, f"receipt.materialization.{key}")
        else:
            _require(
                type(actual) is type(wanted) and actual == wanted,
                f"receipt.materialization.{key} drift",
            )

    determinism = receipt.get("determinism")
    _require(type(determinism) is dict, "receipt determinism missing")
    assert isinstance(determinism, dict)
    _require_exact_mapping(
        determinism,
        {
            "independent_source_acquisitions": 2,
            "source_identity_equal": True,
            "candidate_bytes_equal": True,
            "report_bytes_equal": True,
        },
        "receipt.determinism",
    )

    boundary = receipt.get("claim_boundary")
    _require(type(boundary) is dict, "receipt claim boundary missing")
    assert isinstance(boundary, dict)
    expected_boundary = {
        "real_data_executed": True,
        "candidate_status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "learned_weights_created": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }
    for key, wanted in expected_boundary.items():
        actual = boundary.get(key)
        if type(wanted) is int:
            _require_exact_int(actual, wanted, f"receipt.claim_boundary.{key}")
        else:
            _require(
                type(actual) is type(wanted) and actual == wanted,
                f"receipt.claim_boundary.{key} drift",
            )


def validate_admission(payload: Mapping[str, Any]) -> None:
    """Validate the immutable, zero-credit post-rights admission statement."""
    _require(type(payload) is dict, "admission root must be an object")
    _require(set(payload) == ROOT_KEYS, "admission schema drift")
    _require(payload["schema_version"] == SCHEMA_VERSION, "schema version drift")
    _require(payload["authority_id"] == AUTHORITY_ID, "authority id drift")
    _require(payload["status"] == STATUS, "status drift")
    _require(payload["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    _require_exact_mapping(
        payload["swarm"],
        {
            "control_issue": SWARM_CONTROL_ISSUE,
            "worker_issue": SWARM_ISSUE,
            "lane_key": (
                "D03|COMMON-PILE-LOC|POST-RIGHTS-PAYLOAD-ADMISSION|"
                "CURRENT-MAIN-CROSSBIND-V1"
            ),
        },
        "swarm",
    )
    _require_exact_mapping(
        payload["historical_execution"],
        {
            "product_pr": PRODUCT_PR,
            "product_head_sha": PRODUCT_HEAD_SHA,
            "execution_head_sha": EXECUTION_HEAD_SHA,
            "run_id": RUN_ID,
            "job_id": JOB_ID,
            "receipt_path": RECEIPT_PATH,
            "receipt_blob_sha1": RECEIPT_BLOB_SHA1,
            "materializer_path": MATERIALIZER_PATH,
            "materializer_blob_sha1": MATERIALIZER_BLOB_SHA1,
            "intake_config_path": INTAKE_CONFIG_PATH,
            "intake_config_blob_sha1": INTAKE_CONFIG_BLOB_SHA1,
            "intake_module_path": INTAKE_MODULE_PATH,
            "intake_module_blob_sha1": INTAKE_MODULE_BLOB_SHA1,
        },
        "historical_execution",
    )
    _require_exact_mapping(
        payload["rights_authority"],
        {
            "merged_pr": RIGHTS_PR,
            "product_head_sha": RIGHTS_HEAD_SHA,
            "merge_commit_sha": RIGHTS_MERGE_SHA,
            "independent_audit_issue": RIGHTS_AUDIT_ISSUE,
            "audit_verdict": "PASS_FOR_INTEGRATION_MECHANICS",
            "policy_path": RIGHTS_POLICY_PATH,
            "policy_blob_sha1": RIGHTS_POLICY_BLOB_SHA1,
            "policy_identity_sha256": POLICY_IDENTITY,
            "module_path": RIGHTS_MODULE_PATH,
            "module_blob_sha1": RIGHTS_MODULE_BLOB_SHA1,
            "decision": "CONDITIONAL_SOURCE_ADMISSION",
            "scope": "SOURCE_POLICY_ONLY",
        },
        "rights_authority",
    )
    _require_exact_mapping(payload["record_contract"], RECORD_CONTRACT, "record_contract")
    _require_exact_mapping(payload["admitted_candidate"], ADMITTED_CANDIDATE, "admitted_candidate")
    _require_exact_mapping(
        payload["proof"],
        {
            "historical_receipt_validated": True,
            "historical_execution_code_blobs_verified": True,
            "intake_contract_validated": True,
            "record_contract_alignment_verified": True,
            "rights_policy_repository_bindings_validated": True,
            "rights_decision_applied_post_hoc_to_exact_historical_lineage": True,
            "real_data_execution_repeated": False,
        },
        "proof",
    )
    _require_exact_mapping(payload["truth_boundary"], TRUTH_BOUNDARY, "truth_boundary")
    _require(payload["downstream_required"] == DOWNSTREAM_REQUIRED, "downstream gates drift")


def validate_repository_bindings(
    payload: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> None:
    """Execute the exact LoC execution+rights conjunction proof."""
    validate_admission(payload)
    root = Path(repo_root).resolve()

    receipt = _json_from_bytes(
        _read_bound_file(root, RECEIPT_PATH, RECEIPT_BLOB_SHA1),
        "historical receipt",
    )
    _validate_receipt(receipt)

    _read_bound_file(root, RIGHTS_MODULE_PATH, RIGHTS_MODULE_BLOB_SHA1)
    rights = load_rights_policy(root / RIGHTS_POLICY_PATH, repo_root=root)
    _require(
        rights["policy_identity_sha256"] == POLICY_IDENTITY,
        "rights policy identity drift",
    )

    intake_config_raw = _read_bound_file(root, INTAKE_CONFIG_PATH, INTAKE_CONFIG_BLOB_SHA1)
    intake = _json_from_bytes(intake_config_raw, "intake config")
    validate_intake_config(intake)
    _read_bound_file(root, INTAKE_MODULE_PATH, INTAKE_MODULE_BLOB_SHA1)
    _read_bound_file(root, MATERIALIZER_PATH, MATERIALIZER_BLOB_SHA1)

    _git_show_bound_file(root, EXECUTION_HEAD_SHA, MATERIALIZER_PATH, MATERIALIZER_BLOB_SHA1)
    _git_show_bound_file(root, EXECUTION_HEAD_SHA, INTAKE_CONFIG_PATH, INTAKE_CONFIG_BLOB_SHA1)
    _git_show_bound_file(root, EXECUTION_HEAD_SHA, INTAKE_MODULE_PATH, INTAKE_MODULE_BLOB_SHA1)

    source = receipt["source"]
    admitted = payload["admitted_candidate"]
    _require(
        admitted["source_sha256"] == source["sha256_each"],
        "admitted source hash does not match execution receipt",
    )
    _require(
        admitted["source_bytes"] == source["compressed_bytes_each"],
        "admitted source byte count does not match execution receipt",
    )

    upstream = intake["upstream"]
    _require(upstream["dataset"] == DATASET_CONTRACT["dataset"], "dataset contract drift")
    _require(upstream["revision"] == DATASET_CONTRACT["revision"], "revision contract drift")
    _require(upstream["shard_path"] == DATASET_CONTRACT["shard_path"], "shard path drift")
    _require(
        upstream["shard_lfs_sha256"] == DATASET_CONTRACT["shard_lfs_sha256"],
        "shard hash contract drift",
    )
    _require(
        upstream["shard_compressed_bytes"] == DATASET_CONTRACT["shard_compressed_bytes"],
        "shard byte contract drift",
    )
    _require(upstream["source_name"] == RECORD_CONTRACT["source_value"], "source label drift")

    intake_rights = intake["rights_policy"]
    _require(
        intake_rights["expected_license"] == RECORD_CONTRACT["expected_license"],
        "license contract drift",
    )
    _require(
        intake_rights["expected_language"] == RECORD_CONTRACT["expected_language"],
        "language contract drift",
    )
    _require(intake_rights["require_exact_item_url"] is True, "item URL enforcement drift")
    _require(intake_rights["require_loc_text_file_url"] is True, "text URL enforcement drift")
    _require(
        intake["source_family"] == DATASET_CONTRACT["source_family"],
        "source family contract drift",
    )
    _require(
        admitted["contract_identity_sha256"] == intake["contract_identity_sha256"],
        "executed intake contract identity drift",
    )


def load_and_validate(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the LoC post-rights source-admission authority."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocPostRightsAdmissionError(f"cannot load admission config: {path}") from exc
    _require(type(payload) is dict, "admission root must be an object")
    validate_admission(payload)
    if repo_root is not None:
        validate_repository_bindings(payload, repo_root)
    return payload
