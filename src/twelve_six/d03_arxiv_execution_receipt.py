"""Fail-closed validation for the exact D03 ArXiv LOCAL_FREE execution receipt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "12-6.d03-arxiv-real-execution.v1"
EXECUTION_HEAD_SHA = "37289b5af224c58f0b169f55f4e0f8b466b90f6d"
RUN_ID = 34563405053
RUN_ATTEMPT = 1
JOB_IDS = {
    "execution_a_job_id": 103150553265,
    "execution_b_job_id": 103150553064,
    "compare_job_id": 103150607200,
}

SOURCE = {
    "dataset": "common-pile/arxiv_abstracts",
    "revision": "46de78c48636c0b46f60049dfd1c5a3710d233f9",
    "file": "00003_arxiv-abstracts.jsonl.gz",
    "bytes": 232616926,
    "sha256": "3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7",
    "source_key": "arxiv_abstracts",
    "rights_parent_pr": 769,
    "rights_status": "REVIEW_REQUIRED",
}

OUTPUT = {
    "source_sha256": SOURCE["sha256"],
    "source_bytes": SOURCE["bytes"],
    "candidate_sha256": "21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327",
    "report_file_sha256": "1b94e8ea09f90f53fe86188e62e69f0469e96ef08d042c2fa1b88c69c24869c2",
    "report_identity_sha256": "4251b0b593500e3f91fd4c534873bbecd05992efca60d0d43d4e46c83edf6a2a",
    "disposition_sha256": "e308ae28167c9fe92abc5de1c22f53852336877d677623ce6aab075b67e0ed17",
    "rows_scanned": 4096,
    "retained_records": 1024,
    "retained_normalized_bytes": 1139552,
}

ARTIFACTS = {
    "execution_a": {
        "id": 10185121514,
        "name": "d03-arxiv-evidence-a-34563405053",
        "api_digest": "sha256:6e3be902b1ea414ca6d916a2a1c027737408c64b1a6738a9cdc01dd9b98f2283",
        "evidence_sha256": "fce52b5adaa1504d601d447951720d52651817a979471b9954e96f0cbbfbcc95",
        "expires_at": "2026-10-11T04:45:53Z",
    },
    "execution_b": {
        "id": 10185121867,
        "name": "d03-arxiv-evidence-b-34563405053",
        "api_digest": "sha256:3d1449f1b8355a2abc0e87921d0d32dbaa258099daacdd1b3957fdd1d11620c1",
        "evidence_sha256": "5435a33fcc9bd4e75ef29656e0ab05c033750ced041e6e0502cd7026dddafc18",
        "expires_at": "2026-10-11T04:45:53Z",
    },
    "compare": {
        "id": 10185125828,
        "name": "d03-arxiv-compare-34563405053",
        "api_digest": "sha256:c4059b3c8bd5fc4d2aad51e6cf8a0d025c0ded418389980916a8cc730f135211",
        "manifest_sha256": "61b5c8fe61b33ac60f211a12676afa9b92fbd86e9db411a7dcada9926aaea4ad",
        "expires_at": "2026-10-11T04:46:04Z",
    },
}

ZERO_INT_FIELDS = {
    "canonical_capacity_credited",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "family_credit_added",
    "optimizer_updates",
}
FALSE_BOOL_FIELDS = {
    "tokenizer_fit_authorized",
    "model_training_executed",
    "learned_weights_created",
    "research_corpus_released",
    "paid_compute_used",
    "foreign_pretrained_weights_used",
}


def _is_exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_sha256(value: Any, *, prefixed: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if prefixed:
        if not value.startswith("sha256:"):
            return False
        value = value.removeprefix("sha256:")
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def canonical_json_line_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_artifact(errors: list[str], name: str, value: Any) -> None:
    expected = ARTIFACTS[name]
    if not isinstance(value, dict):
        errors.append(f"artifact_{name}_missing")
        return
    if set(value) != set(expected):
        errors.append(f"artifact_{name}_keys_mismatch")
        return
    for key, expected_value in expected.items():
        actual = value.get(key)
        if key == "id" and not _is_exact_int(actual, expected_value):
            errors.append(f"artifact_{name}_{key}_mismatch")
        elif key == "api_digest" and (
            actual != expected_value or not _is_sha256(actual, prefixed=True)
        ):
            errors.append(f"artifact_{name}_{key}_mismatch")
        elif key.endswith("sha256") and (
            actual != expected_value or not _is_sha256(actual)
        ):
            errors.append(f"artifact_{name}_{key}_mismatch")
        elif key not in {"id", "api_digest"} and not key.endswith("sha256"):
            if actual != expected_value:
                errors.append(f"artifact_{name}_{key}_mismatch")


def _validate_claim_boundary(errors: list[str], value: Any) -> None:
    if not isinstance(value, dict):
        errors.append("claim_boundary_missing")
        return
    expected_keys = ZERO_INT_FIELDS | FALSE_BOOL_FIELDS | {"real_data_executed"}
    if set(value) != expected_keys:
        errors.append("claim_boundary_keys_mismatch")
        return
    if value.get("real_data_executed") is not True:
        errors.append("real_data_executed_must_be_true")
    for key in ZERO_INT_FIELDS:
        if not _is_exact_int(value.get(key), 0):
            errors.append(f"{key}_must_be_exact_int_zero")
    for key in FALSE_BOOL_FIELDS:
        if value.get(key) is not False:
            errors.append(f"{key}_must_be_false")


def _compare_artifact_digest(receipt_value: str, manifest_value: Any) -> bool:
    return isinstance(manifest_value, str) and receipt_value.removeprefix("sha256:") == manifest_value


def validate_d03_arxiv_receipt(
    receipt: Any,
    *,
    compare_manifest: Any | None = None,
) -> list[str]:
    """Return blockers; empty means the exact receipt satisfies its bounded claims."""
    if not isinstance(receipt, dict):
        return ["receipt_root_must_be_object"]

    errors: list[str] = []
    if receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if receipt.get("execution_profile") != "LOCAL_FREE":
        errors.append("execution_profile_mismatch")
    if not _is_exact_int(receipt.get("product_pr"), 1078):
        errors.append("product_pr_mismatch")
    if not _is_exact_int(receipt.get("owner_issue"), 1199):
        errors.append("owner_issue_mismatch")
    if not _is_exact_int(receipt.get("predecessor_owner_issue"), 1077):
        errors.append("predecessor_owner_issue_mismatch")
    if receipt.get("execution_head_sha") != EXECUTION_HEAD_SHA:
        errors.append("execution_head_sha_mismatch")

    workflow = receipt.get("specialist_workflow")
    if not isinstance(workflow, dict):
        errors.append("specialist_workflow_missing")
    else:
        expected_workflow = {
            "run_id": RUN_ID,
            "run_attempt": RUN_ATTEMPT,
            "conclusion": "success",
            **JOB_IDS,
        }
        if set(workflow) != set(expected_workflow):
            errors.append("specialist_workflow_keys_mismatch")
        for key, expected in expected_workflow.items():
            actual = workflow.get(key)
            if isinstance(expected, int):
                if not _is_exact_int(actual, expected):
                    errors.append(f"specialist_workflow_{key}_mismatch")
            elif actual != expected:
                errors.append(f"specialist_workflow_{key}_mismatch")

    if receipt.get("source") != SOURCE:
        errors.append("source_identity_mismatch")

    for slot in ("independent_execution_a", "independent_execution_b"):
        value = receipt.get(slot)
        if value != OUTPUT:
            errors.append(f"{slot}_mismatch")
        elif any(
            not _is_exact_int(value[key], OUTPUT[key])
            for key in ("source_bytes", "rows_scanned", "retained_records", "retained_normalized_bytes")
        ):
            errors.append(f"{slot}_numeric_type_mismatch")

    artifacts = receipt.get("run_bound_artifacts")
    if not isinstance(artifacts, dict):
        errors.append("run_bound_artifacts_missing")
    else:
        if set(artifacts) != {"retention_days", "execution_a", "execution_b", "compare"}:
            errors.append("run_bound_artifacts_keys_mismatch")
        if not _is_exact_int(artifacts.get("retention_days"), 30):
            errors.append("artifact_retention_days_mismatch")
        for name in ("execution_a", "execution_b", "compare"):
            _validate_artifact(errors, name, artifacts.get(name))

    _validate_claim_boundary(errors, receipt.get("claim_boundary"))

    evaluation = receipt.get("evaluation_boundary")
    if not isinstance(evaluation, dict) or any(value is not False for value in evaluation.values()):
        errors.append("evaluation_boundary_must_remain_false")

    content = receipt.get("content_boundary")
    if not isinstance(content, dict):
        errors.append("content_boundary_missing")
    else:
        if content.get("durable_source_payload_retained_in_git") is not False:
            errors.append("source_payload_retention_claim_invalid")
        if content.get("durable_candidate_text_retained_in_git") is not False:
            errors.append("candidate_payload_retention_claim_invalid")
        if content.get("source_and_candidate_deleted_after_hashing") is not True:
            errors.append("payload_deletion_claim_invalid")

    if compare_manifest is not None:
        if not isinstance(compare_manifest, dict):
            errors.append("compare_manifest_root_must_be_object")
        else:
            if compare_manifest.get("schema_version") != "12-6.d03-arxiv-run-bound-compare.v1":
                errors.append("compare_manifest_schema_mismatch")
            if not _is_exact_int(compare_manifest.get("run_id"), RUN_ID):
                errors.append("compare_manifest_run_id_mismatch")
            if not _is_exact_int(compare_manifest.get("run_attempt"), RUN_ATTEMPT):
                errors.append("compare_manifest_run_attempt_mismatch")
            if compare_manifest.get("execution_head_sha") != EXECUTION_HEAD_SHA:
                errors.append("compare_manifest_execution_head_mismatch")

            for name, key in (("execution_a", "execution_a_artifact"), ("execution_b", "execution_b_artifact")):
                artifact = compare_manifest.get(key)
                expected = ARTIFACTS[name]
                if not isinstance(artifact, dict):
                    errors.append(f"compare_manifest_{name}_artifact_missing")
                    continue
                if not _is_exact_int(artifact.get("id"), expected["id"]):
                    errors.append(f"compare_manifest_{name}_artifact_id_mismatch")
                if not _compare_artifact_digest(expected["api_digest"], artifact.get("digest")):
                    errors.append(f"compare_manifest_{name}_artifact_digest_mismatch")
                if artifact.get("evidence_sha256") != expected["evidence_sha256"]:
                    errors.append(f"compare_manifest_{name}_evidence_sha256_mismatch")

            deterministic = compare_manifest.get("deterministic_evidence")
            if not isinstance(deterministic, dict):
                errors.append("compare_manifest_deterministic_evidence_missing")
            else:
                manifest_output = {
                    "source_sha256": deterministic.get("source", {}).get("sha256")
                    if isinstance(deterministic.get("source"), dict)
                    else None,
                    "source_bytes": deterministic.get("source", {}).get("bytes")
                    if isinstance(deterministic.get("source"), dict)
                    else None,
                    "candidate_sha256": deterministic.get("candidate_sha256"),
                    "report_file_sha256": deterministic.get("report_file_sha256"),
                    "report_identity_sha256": deterministic.get("report_identity_sha256"),
                    "disposition_sha256": deterministic.get("disposition_sha256"),
                    "rows_scanned": deterministic.get("rows_scanned"),
                    "retained_records": deterministic.get("retained_records"),
                    "retained_normalized_bytes": deterministic.get("retained_normalized_bytes"),
                }
                if manifest_output != OUTPUT:
                    errors.append("compare_manifest_output_mismatch")

            expected_manifest_sha = ARTIFACTS["compare"]["manifest_sha256"]
            if canonical_json_line_sha256(compare_manifest) != expected_manifest_sha:
                errors.append("compare_manifest_content_sha256_mismatch")

    return sorted(set(errors))
