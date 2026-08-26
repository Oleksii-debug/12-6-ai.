"""EVAL-292 immutable external-real code selection-validation authority.

This worker intentionally publishes an empty, blocked selection set at the
2026-08-26 Wave-1 cutoff. DATA-227 authorizes its two exact code objects for
training, while terminal EVAL-233 records neither explicit evaluation use nor
reservation from training. A terminal DATA-295 policy also places both source
families in the future-training input inventory. Training permission is never
promoted to evaluation permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

WORKER_ID = "EVAL-292-CODE-SELECTION-VALIDATION-V1"
SCHEMA = "12-6.eval292-code-selection-validation.v1"
SET_SCHEMA = "12-6.eval292-code-selection-set.v1"
AUTHORITY_CUTOFF_UTC = "2026-08-26T12:02:04Z"

EVAL233_BRANCH = "eval233/real-holdout-v2-20260826"
EVAL233_HEAD = "b5512b4648cb09dd052b08884dc53f291e1ce935"
EVAL233_RUN_ID = 32957254139
EVAL233_AUTHORITY_PATH = Path("evidence/eval233/real-holdout-v2-authority.json")
EVAL233_AUTHORITY_GIT_BLOB_SHA1 = "2008570890819f32c356677e1e250707d339b53a"

DATA227_HEAD = "8ebdb2e132ed7bae5245e9d4c140752640ab9885"
DATA227_RUN_ID = 32956209865
DATA227_RIGHTS_POLICY_GIT_BLOB_SHA1 = "0ce5223a1cade10031899bf27348a1a65121d4c6"

DATA295_HEAD = "6ab35f8f0f68f1943ff612f4ab529d2d970db1d6"
DATA295_RUN_ID = 32966394993

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "code.encode.httpx._content",
        "source_family": "github:encode/httpx",
        "repository_url": "https://github.com/encode/httpx",
        "commit": "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
        "path": "httpx/_content.py",
        "content_git_blob_sha1": "6f479a0885f723b7395843d41164a87041820776",
        "content_bytes": 8161,
        "license_id": "BSD-3-Clause",
        "license_git_blob_sha1": "ab79d16a3f4c6c894c028d1f7431811e8711b42b",
        "data227_training_authorized": True,
        "evaluation_use_explicitly_authorized": False,
        "reserved_from_all_training": False,
        "present_in_terminal_data295_training_input_inventory": True,
        "selection_admitted": False,
        "blockers": [
            "NO_EXPLICIT_EVALUATION_USE_AUTHORIZATION",
            "NOT_RESERVED_FROM_TRAINING",
            "PRESENT_IN_TERMINAL_FUTURE_TRAINING_INPUT_INVENTORY",
        ],
    },
    {
        "source_id": "code.psf.requests._internal_utils",
        "source_family": "github:psf/requests",
        "repository_url": "https://github.com/psf/requests",
        "commit": "5460f467b02e49471c0fd6cfc9ca0adab6351f98",
        "path": "src/requests/_internal_utils.py",
        "content_git_blob_sha1": "0466a7d347db4ed34a37db51b75fc8e80bc06055",
        "content_bytes": 1542,
        "license_id": "Apache-2.0",
        "license_git_blob_sha1": "67db8588217f266eb561f75fae738656325deac9",
        "data227_training_authorized": True,
        "evaluation_use_explicitly_authorized": False,
        "reserved_from_all_training": False,
        "present_in_terminal_data295_training_input_inventory": True,
        "selection_admitted": False,
        "blockers": [
            "NO_EXPLICIT_EVALUATION_USE_AUTHORIZATION",
            "NOT_RESERVED_FROM_TRAINING",
            "PRESENT_IN_TERMINAL_FUTURE_TRAINING_INPUT_INVENTORY",
        ],
    },
)


class Eval292Error(RuntimeError):
    """Fail-closed EVAL-292 contract violation."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _selection_set() -> dict[str, Any]:
    unsigned = {
        "schema_version": SET_SCHEMA,
        "purpose": "selection-validation",
        "modality": "code",
        "status": "BLOCKED_NO_ELIGIBLE_CODE_OBJECTS",
        "documents": 0,
        "independent_source_families": 0,
        "unique_content_bytes": 0,
        "records": [],
        "immutable": True,
        "selection_eligible": False,
        "tokenizer_fit_eligible": False,
        "hyperparameter_selection_eligible": False,
        "training_eligible": False,
    }
    return {**unsigned, "set_identity_sha256": _hash_json(unsigned)}


def build_authority() -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "authority_cutoff_utc": AUTHORITY_CUTOFF_UTC,
        "implementation_base": {
            "branch": EVAL233_BRANCH,
            "head_sha": EVAL233_HEAD,
        },
        "inputs": {
            "data227": {
                "worker_id": "DATA-227-REAL-CODE-SOURCE-ADMISSION-V2",
                "head_sha": DATA227_HEAD,
                "dedicated_workflow_run_id": DATA227_RUN_ID,
                "dedicated_workflow_conclusion": "success",
                "rights_policy_git_blob_sha1": DATA227_RIGHTS_POLICY_GIT_BLOB_SHA1,
                "candidate_documents": 2,
                "candidate_independent_repositories": 2,
                "candidate_bytes": 9703,
                "scope": "D03/model-training admission only",
            },
            "eval233": {
                "worker_id": "EVAL-233-REAL-HOLDOUT-V2",
                "head_sha": EVAL233_HEAD,
                "dedicated_workflow_run_id": EVAL233_RUN_ID,
                "dedicated_workflow_conclusion": "success",
                "authority_git_blob_sha1": EVAL233_AUTHORITY_GIT_BLOB_SHA1,
                "code_status": (
                    "BLOCKED_DATA227_TRAINING_ONLY_NO_EVALUATION_RESERVATION"
                ),
                "evaluation_use_explicitly_authorized": False,
                "reserved_from_training": False,
            },
            "data295_future_training_inventory": {
                "worker_id": "DATA-295-BALANCE-POLICY-20M-V1",
                "head_sha": DATA295_HEAD,
                "dedicated_workflow_run_id": DATA295_RUN_ID,
                "dedicated_workflow_conclusion": "success",
                "activation_status": "BLOCKED_SOURCE_FAMILY_DIVERSITY",
                "code_input_families": [
                    "github:encode/httpx",
                    "github:psf/requests",
                ],
                "code_input_bytes": 9703,
                "note": (
                    "Terminal preregistered future-corpus policy input inventory; "
                    "not promoted here to a corpus identity."
                ),
            },
        },
        "rules": {
            "explicit_evaluation_authorization_required": True,
            "pretraining_reservation_required": True,
            "training_rights_do_not_imply_evaluation_rights": True,
            "future_training_inventory_overlap_must_equal_zero": True,
            "multiple_independent_repositories_required_when_available": True,
            "final_test_records_may_not_be_copied": True,
            "final_test_outcomes_may_not_be_inspected_for_selection_construction": True,
            "local_free_only": True,
        },
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "selection_set": _selection_set(),
        "separation": {
            "selected_source_ids": [],
            "selected_source_families": [],
            "future_training_source_id_overlap_count": 0,
            "future_training_source_family_overlap_count": 0,
            "selected_content_hash_overlap_count": 0,
            "proof_kind": "EMPTY_FAIL_CLOSED_SET",
            "final_test_records_copied": 0,
            "final_test_outcomes_inspected": False,
            "final_test_bytes_consumed": False,
        },
        "verdict": {
            "status": "BLOCKED",
            "reason": (
                "No Wave-1 code object has both explicit evaluation-use "
                "authorization and reservation from all training; both terminal "
                "DATA-227 objects are also present in the terminal DATA-295 "
                "future-training input inventory."
            ),
            "selection_documents": 0,
            "selection_independent_repositories": 0,
            "release_allowed": False,
        },
        "unblock_conditions": [
            "Publish a terminal code evaluation-rights authority for exact pinned objects.",
            (
                "Reserve those exact objects from every current and future training "
                "inventory before selection construction."
            ),
            (
                "Use at least two independent repositories when at least two eligible "
                "reserved repositories exist."
            ),
            "Prove zero source/content overlap against the frozen future-training inventory.",
            "Rebuild deterministically without reading final-test outcomes.",
        ],
    }
    return {**unsigned, "authority_identity_sha256": _hash_json(unsigned)}


def verify_eval233_boundary(repo_root: Path) -> None:
    path = repo_root / EVAL233_AUTHORITY_PATH
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Eval292Error("unable to read terminal EVAL-233 authority") from exc

    if _git_blob_sha1(raw) != EVAL233_AUTHORITY_GIT_BLOB_SHA1:
        raise Eval292Error("EVAL-233 authority Git blob drift")

    code = value.get("code")
    if not isinstance(code, dict):
        raise Eval292Error("EVAL-233 code authority missing")
    expected = {
        "data227_head_sha": DATA227_HEAD,
        "rights_policy_git_blob_sha1": DATA227_RIGHTS_POLICY_GIT_BLOB_SHA1,
        "evaluation_use_explicitly_authorized": False,
        "reserved_from_training": False,
        "status": "BLOCKED_DATA227_TRAINING_ONLY_NO_EVALUATION_RESERVATION",
    }
    for key, expected_value in expected.items():
        if code.get(key) != expected_value:
            raise Eval292Error(f"EVAL-233 code boundary drift: {key}")


def validate_authority(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA or value.get("worker_id") != WORKER_ID:
        raise Eval292Error("unsupported EVAL-292 authority identity")

    claimed_identity = value.get("authority_identity_sha256")
    unsigned = dict(value)
    unsigned.pop("authority_identity_sha256", None)
    if claimed_identity != _hash_json(unsigned):
        raise Eval292Error("authority_identity_sha256 mismatch")

    rules = value.get("rules", {})
    required_true = (
        "explicit_evaluation_authorization_required",
        "pretraining_reservation_required",
        "training_rights_do_not_imply_evaluation_rights",
        "future_training_inventory_overlap_must_equal_zero",
        "final_test_records_may_not_be_copied",
        "final_test_outcomes_may_not_be_inspected_for_selection_construction",
        "local_free_only",
    )
    if any(rules.get(key) is not True for key in required_true):
        raise Eval292Error("required fail-closed rule disabled")

    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise Eval292Error("Wave-1 candidate inventory drift")
    repositories = {candidate.get("repository_url") for candidate in candidates}
    if len(repositories) != 2:
        raise Eval292Error("candidate repositories are not independent identities")
    for candidate in candidates:
        if candidate.get("data227_training_authorized") is not True:
            raise Eval292Error("DATA-227 training-rights fact drift")
        if candidate.get("evaluation_use_explicitly_authorized") is not False:
            raise Eval292Error("evaluation authorization was fabricated")
        if candidate.get("reserved_from_all_training") is not False:
            raise Eval292Error("training reservation was fabricated")
        if candidate.get("selection_admitted") is not False:
            raise Eval292Error("ineligible code candidate admitted")
        if candidate.get("present_in_terminal_data295_training_input_inventory") is not True:
            raise Eval292Error("future-training inventory conflict was erased")

    selection = value.get("selection_set")
    if not isinstance(selection, dict):
        raise Eval292Error("selection set missing")
    selection_unsigned = dict(selection)
    claimed_set_identity = selection_unsigned.pop("set_identity_sha256", None)
    if claimed_set_identity != _hash_json(selection_unsigned):
        raise Eval292Error("set_identity_sha256 mismatch")
    expected_empty = {
        "documents": 0,
        "independent_source_families": 0,
        "unique_content_bytes": 0,
        "records": [],
        "selection_eligible": False,
        "training_eligible": False,
        "status": "BLOCKED_NO_ELIGIBLE_CODE_OBJECTS",
    }
    for key, expected_value in expected_empty.items():
        if selection.get(key) != expected_value:
            raise Eval292Error(f"blocked selection invariant changed: {key}")

    separation = value.get("separation", {})
    if any(
        separation.get(key) != 0
        for key in (
            "future_training_source_id_overlap_count",
            "future_training_source_family_overlap_count",
            "selected_content_hash_overlap_count",
            "final_test_records_copied",
        )
    ):
        raise Eval292Error("selection/training/final-test separation violated")
    if separation.get("final_test_outcomes_inspected") is not False:
        raise Eval292Error("final-test outcomes were exposed")
    if separation.get("final_test_bytes_consumed") is not False:
        raise Eval292Error("final-test bytes were consumed")
    if separation.get("selected_source_ids") != []:
        raise Eval292Error("blocked authority selected source IDs")
    if separation.get("selected_source_families") != []:
        raise Eval292Error("blocked authority selected source families")

    verdict = value.get("verdict", {})
    if verdict.get("status") != "BLOCKED" or verdict.get("release_allowed") is not False:
        raise Eval292Error("fail-closed verdict was weakened")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Eval292Error(f"unable to read JSON authority: {path}") from exc
    if not isinstance(value, dict):
        raise Eval292Error(f"expected JSON object: {path}")
    return value


def build(repo_root: Path, output_dir: Path) -> Path:
    verify_eval233_boundary(repo_root)
    authority = build_authority()
    validate_authority(authority)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_bytes(_pretty_json_bytes(authority))
    return path


def verify(repo_root: Path, manifest_path: Path) -> None:
    verify_eval233_boundary(repo_root)
    value = _load_json(manifest_path)
    validate_authority(value)
    expected = _pretty_json_bytes(build_authority())
    actual = manifest_path.read_bytes()
    if actual != expected:
        raise Eval292Error("manifest is not the deterministic canonical EVAL-292 build")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--repo-root", type=Path, default=Path("."))
    build_parser.add_argument("--output-dir", type=Path, required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--repo-root", type=Path, default=Path("."))
    verify_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        path = build(args.repo_root, args.output_dir)
        print(path)
        return 0
    verify(args.repo_root, args.manifest)
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
