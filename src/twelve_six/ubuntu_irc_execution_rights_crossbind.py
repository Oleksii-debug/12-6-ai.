"""Bind repaired Ubuntu IRC execution evidence to the merged source-rights authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.ubuntu_irc_rights import validate_authority as validate_rights_authority

SCHEMA_VERSION = "12-6.d03-ubuntu-irc-repaired-execution-rights-crossbind.v1"
AUTHORITY_ID = "D03-UBUNTU-IRC-REPAIRED-EXECUTION-RIGHTS-CROSSBIND-V1"
STATUS = "REPAIRED_EXECUTION_RIGHTS_CROSSBOUND_ZERO_CREDIT"
BASE_MAIN_SHA = "44581b0779f94f2a2a614a73c3446d70666b64ae"
RIGHTS_PATH = "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
RIGHTS_BLOB_SHA1 = "e05a179c5aa8c869e918d68e5cc660ea14573fbd"
RIGHTS_AUTHORITY_ID = "D03-COMMON-PILE-UBUNTU-IRC-SOURCE-RIGHTS-V1"
RIGHTS_STATUS = "SOURCE_SPECIFIC_RIGHTS_PROVENANCE_QUALIFIED_ZERO_CREDIT"
OLD_PRODUCT_HEAD = "ab457e641d3ca5252cdbbd7fd5e2928d3c761521"
OLD_EVIDENCE_BLOB = "23c73b36f12539200a3da97b2f177b562b62196b"
REPAIRED_PRODUCT_HEAD = "1581dc99a87a38eda777988578ba0a8a9af689d8"
REPAIRED_EVIDENCE_BLOB = "9c1ceb012584e72afc6887ebd3bbba13ebf8bbf1"
EVIDENCE_PATH = "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
EXECUTION_HEAD = "3400e2cd3c62a360b25282310390640a7f20fbc9"
SOURCE_DATASET = "common-pile/ubuntu_irc"
SOURCE_REVISION = "47d55b0534a62bf0766c621297165451969f3de9"
SOURCE_FILE = "v0/documents/00007_ubuntu.jsonl.gz"
SOURCE_SHA256 = "75e38bffcaceb00ed9ce9a63b1d9e74a70f5582b3e3f763a7ec573c7fd6c1e60"
CANDIDATE_SHA256 = "d6a0eaea27313e0d5146bc2a0624104541923ef06b12f01c22b3ccf6956eb6c2"


class UbuntuIrcCrossbindError(ValueError):
    """Raised when the repaired-execution/source-rights crossbind is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuIrcCrossbindError(message)


def _expect_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} keys drifted")
    return value


def _expect_str(value: Any, expected: str, label: str) -> None:
    _require(type(value) is str and value == expected, f"{label} drifted")


def _expect_int(value: Any, expected: int, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} must be integer {expected}")


def _expect_bool(value: Any, expected: bool, label: str) -> None:
    _require(type(value) is bool and value is expected, f"{label} must be literal {expected}")


def git_blob_sha1(payload: bytes) -> str:
    """Return the Git blob SHA-1 for exact bytes."""
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UbuntuIrcCrossbindError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one UTF-8 JSON object while rejecting duplicate keys."""
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UbuntuIrcCrossbindError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} must contain a JSON object")
    assert isinstance(value, dict)
    return value


def validate_crossbind(
    crossbind: dict[str, Any],
    rights_authority: dict[str, Any],
    parent_registry: dict[str, Any],
    rights_bytes: bytes,
) -> str:
    """Validate exact immutable authorities and preserve the zero-credit boundary."""
    _expect_keys(
        crossbind,
        {
            "schema_version",
            "authority_id",
            "status",
            "project_authority",
            "rights_authority",
            "superseded_execution_reference",
            "repaired_execution_authority",
            "decision",
            "truth_boundary",
        },
        "crossbind",
    )
    _expect_str(crossbind["schema_version"], SCHEMA_VERSION, "schema_version")
    _expect_str(crossbind["authority_id"], AUTHORITY_ID, "authority_id")
    _expect_str(crossbind["status"], STATUS, "status")

    project = _expect_keys(
        crossbind["project_authority"],
        {"swarm_control_issue", "worker_issue", "base_main_sha"},
        "project_authority",
    )
    _expect_int(project["swarm_control_issue"], 723, "swarm_control_issue")
    _expect_int(project["worker_issue"], 1348, "worker_issue")
    _expect_str(project["base_main_sha"], BASE_MAIN_SHA, "base_main_sha")

    rights = _expect_keys(
        crossbind["rights_authority"],
        {"path", "git_blob_sha1", "authority_id", "status"},
        "rights_authority",
    )
    _expect_str(rights["path"], RIGHTS_PATH, "rights path")
    _expect_str(rights["git_blob_sha1"], RIGHTS_BLOB_SHA1, "rights blob")
    _expect_str(rights["authority_id"], RIGHTS_AUTHORITY_ID, "rights authority_id")
    _expect_str(rights["status"], RIGHTS_STATUS, "rights status")
    _require(git_blob_sha1(rights_bytes) == RIGHTS_BLOB_SHA1, "rights bytes drifted")
    _require(
        validate_rights_authority(rights_authority, parent_registry) == RIGHTS_STATUS,
        "rights authority did not validate",
    )

    stale = _expect_keys(
        crossbind["superseded_execution_reference"],
        {"product_pr", "product_head_sha", "evidence_path", "evidence_blob_sha1"},
        "superseded_execution_reference",
    )
    _expect_int(stale["product_pr"], 1100, "stale product_pr")
    _expect_str(stale["product_head_sha"], OLD_PRODUCT_HEAD, "stale product head")
    _expect_str(stale["evidence_path"], EVIDENCE_PATH, "stale evidence path")
    _expect_str(stale["evidence_blob_sha1"], OLD_EVIDENCE_BLOB, "stale evidence blob")

    rights_execution = rights_authority["real_execution_reference"]
    _require(type(rights_execution) is dict, "rights execution reference missing")
    for field in ("product_pr", "product_head_sha", "evidence_path", "evidence_blob_sha1"):
        _require(rights_execution.get(field) == stale[field], f"stale reference {field} mismatch")

    repaired = _expect_keys(
        crossbind["repaired_execution_authority"],
        {
            "product_pr",
            "product_head_sha",
            "evidence_path",
            "evidence_blob_sha1",
            "independent_audit_issue",
            "independent_verdict",
            "shared_ci_run_id",
            "shared_ci_conclusion",
            "real_execution_run_id",
            "real_execution_job_id",
            "execution_head_sha",
            "source_dataset",
            "source_revision",
            "source_file",
            "source_sha256",
            "candidate_payload_sha256",
            "retained_records",
            "retained_normalized_utf8_bytes",
        },
        "repaired_execution_authority",
    )
    _expect_int(repaired["product_pr"], 1100, "repaired product_pr")
    _expect_str(repaired["product_head_sha"], REPAIRED_PRODUCT_HEAD, "repaired head")
    _expect_str(repaired["evidence_path"], EVIDENCE_PATH, "repaired evidence path")
    _expect_str(repaired["evidence_blob_sha1"], REPAIRED_EVIDENCE_BLOB, "repaired evidence blob")
    _expect_int(repaired["independent_audit_issue"], 1275, "independent audit issue")
    _expect_str(
        repaired["independent_verdict"],
        "PASS_FOR_INTEGRATION_REAL_EXECUTION_EVIDENCE",
        "independent verdict",
    )
    _expect_int(repaired["shared_ci_run_id"], 34604427566, "shared CI run")
    _expect_str(repaired["shared_ci_conclusion"], "success", "shared CI conclusion")
    _expect_int(repaired["real_execution_run_id"], 34549598850, "execution run")
    _expect_int(repaired["real_execution_job_id"], 103109520811, "execution job")
    _expect_str(repaired["execution_head_sha"], EXECUTION_HEAD, "execution head")
    _expect_str(repaired["source_dataset"], SOURCE_DATASET, "source dataset")
    _expect_str(repaired["source_revision"], SOURCE_REVISION, "source revision")
    _expect_str(repaired["source_file"], SOURCE_FILE, "source file")
    _expect_str(repaired["source_sha256"], SOURCE_SHA256, "source SHA-256")
    _expect_str(repaired["candidate_payload_sha256"], CANDIDATE_SHA256, "candidate SHA-256")
    _expect_int(repaired["retained_records"], 972, "retained records")
    _expect_int(
        repaired["retained_normalized_utf8_bytes"],
        4_799_981,
        "retained normalized bytes",
    )
    _require(REPAIRED_PRODUCT_HEAD != OLD_PRODUCT_HEAD, "repaired head did not advance")
    _require(REPAIRED_EVIDENCE_BLOB != OLD_EVIDENCE_BLOB, "repaired evidence did not advance")

    for field, expected in (
        ("source_dataset", SOURCE_DATASET),
        ("source_revision", SOURCE_REVISION),
        ("source_file", SOURCE_FILE),
        ("source_sha256", SOURCE_SHA256),
        ("candidate_payload_sha256", CANDIDATE_SHA256),
    ):
        _require(rights_execution.get(field) == expected, f"rights {field} drifted")
        _require(repaired[field] == expected, f"repaired {field} drifted")

    decision = _expect_keys(
        crossbind["decision"],
        {
            "rights_authority_qualified",
            "repaired_execution_authority_qualified",
            "authority_crossbind_qualified",
            "downstream_global_dedup_candidate_input_allowed",
            "payload_source_admission_executed",
            "canonical_training_authorized",
            "canonical_capacity_credited",
            "family_credit_added",
            "training_authorized_bytes",
            "unique_causal_loss_positions_authorized",
            "tokenizer_fit_authorized",
            "optimizer_updates_authorized",
            "evaluation_eligible",
            "final_test_accessed",
        },
        "decision",
    )
    for field in (
        "rights_authority_qualified",
        "repaired_execution_authority_qualified",
        "authority_crossbind_qualified",
        "downstream_global_dedup_candidate_input_allowed",
    ):
        _expect_bool(decision[field], True, f"decision.{field}")
    for field in (
        "payload_source_admission_executed",
        "canonical_training_authorized",
        "tokenizer_fit_authorized",
        "optimizer_updates_authorized",
        "evaluation_eligible",
        "final_test_accessed",
    ):
        _expect_bool(decision[field], False, f"decision.{field}")
    for field in (
        "canonical_capacity_credited",
        "family_credit_added",
        "training_authorized_bytes",
        "unique_causal_loss_positions_authorized",
    ):
        _expect_int(decision[field], 0, f"decision.{field}")

    truth = _expect_keys(
        crossbind["truth_boundary"],
        {
            "durable_evidence_contains_source_text",
            "real_data_reexecuted_by_this_package",
            "global_dedup_executed_by_this_package",
            "decontamination_executed_by_this_package",
            "training_executed",
            "learned_weights_created",
            "paid_compute_used",
            "foreign_pretrained_weights_used",
            "external_llm_or_api_used_for_data_or_intelligence",
        },
        "truth_boundary",
    )
    for field in truth:
        _expect_bool(truth[field], False, f"truth_boundary.{field}")
    return STATUS


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    return load_json_bytes(payload, str(path)), payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crossbind", type=Path)
    parser.add_argument("rights_authority", type=Path)
    parser.add_argument("parent_registry", type=Path)
    args = parser.parse_args(argv)

    crossbind, _ = load_json(args.crossbind)
    rights, rights_bytes = load_json(args.rights_authority)
    parent, _ = load_json(args.parent_registry)
    print(validate_crossbind(crossbind, rights, parent, rights_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
