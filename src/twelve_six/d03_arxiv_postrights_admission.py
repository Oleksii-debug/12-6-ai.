"""Post-rights source-admission proof for the exact D03 ArXiv execution lineage."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.common_pile_arxiv_abstracts_rights import (
    POLICY_IDENTITY,
    RECORD_CONTRACT,
)
from twelve_six.common_pile_arxiv_abstracts_rights import (
    load_and_validate as load_rights_policy,
)
from twelve_six.d03_arxiv_execution_receipt import (
    EXECUTION_HEAD_SHA,
    OUTPUT,
    RUN_ID,
    SOURCE,
    validate_d03_arxiv_receipt,
)

SCHEMA_VERSION = "12-6.d03-arxiv-postrights-source-admission.v1"
AUTHORITY_ID = "D03-ARXIV-POSTRIGHTS-SOURCE-ADMISSION-V1"
STATUS = "PAYLOAD_SOURCE_ADMISSION_EXECUTED_ZERO_CREDIT"
CONFIG_PATH = "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json"
RECEIPT_PATH = "evidence/d03-arxiv-real-execution-v1.json"
RECEIPT_BLOB_SHA1 = "d3927dd87110aae0b1cb879498757812a3c7087d"
RECEIPT_VALIDATOR_PATH = "src/twelve_six/d03_arxiv_execution_receipt.py"
RECEIPT_VALIDATOR_BLOB_SHA1 = "4b7af5cd0e141cdb9fcefe957f2772f9a48ad73b"
MATERIALIZER_PATH = "tools/materialize_d03_common_pile_arxiv_abstracts_v1.py"
MATERIALIZER_BLOB_SHA1 = "38c244a5a2da40ed4fb67f801f02e625bc040564"
RIGHTS_POLICY_PATH = "configs/data/d03_common_pile_arxiv_abstracts_source_rights_v1.json"
RIGHTS_POLICY_BLOB_SHA1 = "31544961a99310487b2b2310adec10069065ac1c"
RIGHTS_MODULE_PATH = "src/twelve_six/common_pile_arxiv_abstracts_rights.py"
RIGHTS_MODULE_BLOB_SHA1 = "d4f48237ceae57606d69ef4eaddc2a5c4366c5dd"
PRODUCT_PR = 1078
PRODUCT_HEAD_SHA = "a4c8be72a7b7ba3a22fc49d8968d7becef1ac325"
RIGHTS_PR = 1361
SWARM_ISSUE = 1368
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

TRUTH_BOUNDARY = {
    "payload_source_admission_executed": True,
    "source_admitted_candidate_records": 1024,
    "source_admitted_candidate_bytes": 1139552,
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


class ArxivPostRightsAdmissionError(ValueError):
    """Raised when the exact post-rights admission proof stops being fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArxivPostRightsAdmissionError(message)


def git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _read_bound_file(root: Path, relpath: str, expected_blob: str) -> bytes:
    path = (root / relpath).resolve()
    _require(path.is_relative_to(root.resolve()), f"{relpath} escaped repository root")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArxivPostRightsAdmissionError(f"cannot read {relpath}") from exc
    _require(git_blob_sha1(data) == expected_blob, f"{relpath} blob drift")
    return data


def _git_show_bytes(root: Path, revision: str, relpath: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relpath}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArxivPostRightsAdmissionError(
            f"cannot resolve historical file {revision}:{relpath}"
        ) from exc
    return result.stdout


def _literal_assignment(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ArxivPostRightsAdmissionError(
                    f"historical {name} is not a literal"
                ) from exc
    raise ArxivPostRightsAdmissionError(f"historical {name} assignment missing")


def _is_false_reason_return(node: ast.If, reason: str) -> bool:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return False
    value = node.body[0].value
    if not isinstance(value, ast.Tuple) or len(value.elts) != 3:
        return False
    try:
        actual = tuple(ast.literal_eval(item) for item in value.elts)
    except (ValueError, TypeError):
        return False
    return actual == (False, reason, "")


def _is_row_source_mismatch(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.NotEq):
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Subscript)
        and isinstance(left.value, ast.Name)
        and left.value.id == "row"
        and isinstance(left.slice, ast.Constant)
        and left.slice.value == "source"
        and isinstance(right, ast.Name)
        and right.id == "SOURCE_LABEL"
    )


def _is_metadata_license_mismatch(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.NotEq):
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Attribute)
        and isinstance(left.func.value, ast.Name)
        and left.func.value.id == "metadata"
        and left.func.attr == "get"
        and len(left.args) == 1
        and isinstance(left.args[0], ast.Constant)
        and left.args[0].value == "license"
        and not left.keywords
        and isinstance(right, ast.Name)
        and right.id == "EXPECTED_LICENSE"
    )


def validate_historical_materializer(root: str | Path = ".") -> None:
    """Prove the executed materializer enforced the later-qualified CC0 contract."""
    repo_root = Path(root).resolve()
    source = _git_show_bytes(repo_root, EXECUTION_HEAD_SHA, MATERIALIZER_PATH)
    _require(
        git_blob_sha1(source) == MATERIALIZER_BLOB_SHA1,
        "historical materializer blob drift",
    )
    try:
        tree = ast.parse(source.decode("utf-8"), filename=MATERIALIZER_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ArxivPostRightsAdmissionError(
            "historical materializer cannot be parsed"
        ) from exc

    _require(
        _literal_assignment(tree, "SOURCE_REVISION") == SOURCE["revision"],
        "revision drift",
    )
    _require(
        _literal_assignment(tree, "SOURCE_FILE") == SOURCE["file"],
        "source file drift",
    )
    _require(
        _literal_assignment(tree, "SOURCE_SHA256") == SOURCE["sha256"],
        "source sha drift",
    )
    _require(
        _literal_assignment(tree, "SOURCE_BYTES") == SOURCE["bytes"],
        "source bytes drift",
    )
    _require(
        _literal_assignment(tree, "SOURCE_LABEL") == RECORD_CONTRACT["source_value"],
        "historical source label is not the qualified record contract",
    )
    _require(
        list(_literal_assignment(tree, "EXPECTED_FIELDS"))
        == RECORD_CONTRACT["required_row_fields"],
        "historical row fields are not the qualified record contract",
    )
    _require(
        _literal_assignment(tree, "EXPECTED_LICENSE")
        == RECORD_CONTRACT["expected_metadata_license"],
        "historical license is not the qualified record contract",
    )

    assess = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "assess_row"
        ),
        None,
    )
    _require(assess is not None, "historical assess_row missing")
    assert isinstance(assess, ast.FunctionDef)

    source_reject = any(
        isinstance(node, ast.If)
        and _is_row_source_mismatch(node.test)
        and _is_false_reason_return(node, "source_label_mismatch")
        for node in assess.body
    )
    license_reject = any(
        isinstance(node, ast.If)
        and _is_metadata_license_mismatch(node.test)
        and _is_false_reason_return(node, "metadata_license_mismatch")
        for node in assess.body
    )
    _require(source_reject, "historical source-label rejection predicate missing")
    _require(license_reject, "historical exact-license rejection predicate missing")


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


def validate_admission(payload: Mapping[str, Any]) -> None:
    """Validate the immutable zero-credit source-admission statement."""
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
                "D03|ARXIV|POST-RIGHTS-PAYLOAD-ADMISSION|"
                "CURRENT-MAIN-CONVERGENCE|V1"
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
            "receipt_path": RECEIPT_PATH,
            "receipt_blob_sha1": RECEIPT_BLOB_SHA1,
            "receipt_validator_path": RECEIPT_VALIDATOR_PATH,
            "receipt_validator_blob_sha1": RECEIPT_VALIDATOR_BLOB_SHA1,
            "materializer_path": MATERIALIZER_PATH,
            "materializer_blob_sha1": MATERIALIZER_BLOB_SHA1,
        },
        "historical_execution",
    )
    _require_exact_mapping(
        payload["rights_authority"],
        {
            "merged_pr": RIGHTS_PR,
            "policy_path": RIGHTS_POLICY_PATH,
            "policy_blob_sha1": RIGHTS_POLICY_BLOB_SHA1,
            "policy_identity_sha256": POLICY_IDENTITY,
            "module_path": RIGHTS_MODULE_PATH,
            "module_blob_sha1": RIGHTS_MODULE_BLOB_SHA1,
            "decision": "CONDITIONAL_SOURCE_ADMISSION",
            "scope": "DESCRIPTIVE_METADATA_INCLUDING_ABSTRACT",
        },
        "rights_authority",
    )
    _require_exact_mapping(
        payload["record_contract"],
        RECORD_CONTRACT,
        "record_contract",
    )
    _require_exact_mapping(
        payload["admitted_candidate"],
        {
            "source_sha256": SOURCE["sha256"],
            "source_bytes": SOURCE["bytes"],
            "rows_scanned": OUTPUT["rows_scanned"],
            "candidate_sha256": OUTPUT["candidate_sha256"],
            "retained_records": OUTPUT["retained_records"],
            "retained_normalized_bytes": OUTPUT["retained_normalized_bytes"],
            "payload_rematerialized_in_this_package": False,
        },
        "admitted_candidate",
    )
    _require_exact_mapping(
        payload["proof"],
        {
            "historical_receipt_validated": True,
            "historical_materializer_blob_verified": True,
            "historical_source_predicate_verified": True,
            "historical_license_predicate_verified": True,
            "rights_policy_repository_bindings_validated": True,
            "rights_decision_applied_post_hoc_to_exact_historical_lineage": True,
            "real_data_execution_repeated": False,
        },
        "proof",
    )
    _require_exact_mapping(
        payload["truth_boundary"],
        TRUTH_BOUNDARY,
        "truth_boundary",
    )
    _require(
        payload["downstream_required"] == DOWNSTREAM_REQUIRED,
        "downstream gates drift",
    )


def validate_repository_bindings(
    payload: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> None:
    """Execute the conjunction proof against exact repository/history identities."""
    validate_admission(payload)
    root = Path(repo_root).resolve()

    receipt_bytes = _read_bound_file(root, RECEIPT_PATH, RECEIPT_BLOB_SHA1)
    _read_bound_file(root, RECEIPT_VALIDATOR_PATH, RECEIPT_VALIDATOR_BLOB_SHA1)
    _read_bound_file(root, RIGHTS_POLICY_PATH, RIGHTS_POLICY_BLOB_SHA1)
    _read_bound_file(root, RIGHTS_MODULE_PATH, RIGHTS_MODULE_BLOB_SHA1)

    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArxivPostRightsAdmissionError(
            "historical receipt cannot be decoded"
        ) from exc
    blockers = validate_d03_arxiv_receipt(receipt)
    _require(blockers == [], f"historical receipt validation failed: {blockers}")

    rights_path = root / RIGHTS_POLICY_PATH
    rights = load_rights_policy(rights_path, repo_root=root)
    _require(
        rights["policy_identity_sha256"] == POLICY_IDENTITY,
        "rights policy identity drift",
    )
    validate_historical_materializer(root)

    admitted = payload["admitted_candidate"]
    _require(
        admitted["source_sha256"] == rights["dataset_contract"]["shard_lfs_sha256"],
        "admitted source does not match rights dataset contract",
    )
    _require(
        admitted["source_bytes"]
        == rights["dataset_contract"]["shard_compressed_bytes"],
        "admitted source byte count does not match rights dataset contract",
    )


def load_and_validate(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArxivPostRightsAdmissionError(
            f"cannot load admission config: {path}"
        ) from exc
    _require(type(payload) is dict, "admission root must be an object")
    validate_admission(payload)
    if repo_root is not None:
        validate_repository_bindings(payload, repo_root)
    return payload
