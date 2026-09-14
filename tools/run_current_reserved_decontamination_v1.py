"""Execute current reserved-evaluation decontamination with exact code binding.

Payload-bearing inputs remain ephemeral. Durable publication is one atomic directory
containing only the canonical hash-only report/evidence plus a code- and input-bound
receipt. This runner never grants corpus, tokenizer, exposure, or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data.current_reserved_decontamination_v1 import (
    execute_reserved_decontamination,
    verify_execution_evidence,
)

RECEIPT_SCHEMA = "12-6.current-reserved-decontamination-run-receipt.v1"
REPORT_NAME = "decontamination_report.json"
EVIDENCE_NAME = "execution_evidence.json"
RECEIPT_NAME = "execution_receipt.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value, digest


def _load_jsonl_with_sha(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                raise ValueError(
                    f"blank JSONL line {line_number} is forbidden: {path}"
                )
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL line {line_number} must be an object: {path}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input must be non-empty: {path}")
    return rows, digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _require_git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase 40-hex Git SHA")
    return value


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_tree_is_clean(repo_root: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        check=False,
    )
    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        raise RuntimeError("unable to verify tracked working-tree cleanliness")
    return unstaged.returncode == 0 and staged.returncode == 0


def require_exact_checkout(repo_root: Path, expected_git_sha: str) -> str:
    expected = _require_git_sha(expected_git_sha, "expected implementation Git SHA")
    actual = _git_head(repo_root)
    if actual != expected:
        raise RuntimeError(
            "checked-out decontamination implementation Git head differs from "
            f"independent expectation: expected={expected} actual={actual}"
        )
    if not _tracked_tree_is_clean(repo_root):
        raise RuntimeError(
            "tracked working tree differs from the expected implementation head"
        )
    return actual


def build_run_receipt(
    *,
    implementation_git_sha: str,
    input_file_sha256: Mapping[str, str],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    report_file_bytes: bytes,
    evidence_file_bytes: bytes,
) -> dict[str, Any]:
    git_sha = _require_git_sha(implementation_git_sha, "implementation_git_sha")
    expected_input_names = {
        "training_records_jsonl",
        "training_handoff_json",
        "evaluation_records_jsonl",
        "reserved_binding_json",
    }
    if set(input_file_sha256) != expected_input_names:
        raise ValueError("receipt input-file hash set drift")
    normalized_inputs = {
        key: _require_sha256(input_file_sha256[key], f"input_file_sha256.{key}")
        for key in sorted(expected_input_names)
    }

    core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "decontamination_implementation_git_sha": git_sha,
        "input_files_sha256": normalized_inputs,
        "output_files_sha256": {
            REPORT_NAME: _sha256_bytes(report_file_bytes),
            EVIDENCE_NAME: _sha256_bytes(evidence_file_bytes),
        },
        "status": evidence.get("status"),
        "training_corpus_identity_sha256": evidence.get(
            "training_corpus_identity_sha256"
        ),
        "input_survivor_authority_sha256": evidence.get(
            "input_survivor_authority_sha256"
        ),
        "training_handoff_identity_sha256": evidence.get(
            "training_handoff_identity_sha256"
        ),
        "reserved_payload_binding_identity_sha256": evidence.get(
            "reserved_payload_binding_identity_sha256"
        ),
        "selection_validation_identity_sha256": evidence.get(
            "selection_validation_identity_sha256"
        ),
        "final_test_identity_sha256": evidence.get("final_test_identity_sha256"),
        "decontamination_report_sha256": report.get("report_sha256"),
        "execution_identity_sha256": evidence.get("execution_identity_sha256"),
        "final_test_payload_accessed_for_decontamination": evidence.get(
            "final_test_payload_accessed_for_decontamination"
        ),
        "final_test_outcomes_read": False,
        "durable_bundle_hash_only": True,
        "raw_text_persisted_in_bundle": False,
        "record_ids_persisted_in_bundle": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    receipt = deepcopy(core)
    receipt["receipt_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return receipt


def verify_run_receipt(
    receipt: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("run receipt schema drift")
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"), "receipt_identity_sha256"
    )
    body = deepcopy(dict(receipt))
    body.pop("receipt_identity_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != claimed:
        raise ValueError("run receipt self-hash mismatch")
    _require_git_sha(
        receipt.get("decontamination_implementation_git_sha"),
        "decontamination_implementation_git_sha",
    )

    inputs = receipt.get("input_files_sha256")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "training_records_jsonl",
        "training_handoff_json",
        "evaluation_records_jsonl",
        "reserved_binding_json",
    }:
        raise ValueError("run receipt input-file hash set drift")
    for name, value in inputs.items():
        _require_sha256(value, f"input_files_sha256.{name}")

    outputs = receipt.get("output_files_sha256")
    if not isinstance(outputs, Mapping) or set(outputs) != {REPORT_NAME, EVIDENCE_NAME}:
        raise ValueError("run receipt output-file hash set drift")
    expected_output_hashes = {
        REPORT_NAME: _sha256_bytes(_file_bytes(report)),
        EVIDENCE_NAME: _sha256_bytes(_file_bytes(evidence)),
    }
    if dict(outputs) != expected_output_hashes:
        raise ValueError("run receipt output-file hash mismatch")

    verify_execution_evidence(evidence, report)
    bindings = {
        "status": evidence.get("status"),
        "training_corpus_identity_sha256": evidence.get(
            "training_corpus_identity_sha256"
        ),
        "input_survivor_authority_sha256": evidence.get(
            "input_survivor_authority_sha256"
        ),
        "training_handoff_identity_sha256": evidence.get(
            "training_handoff_identity_sha256"
        ),
        "reserved_payload_binding_identity_sha256": evidence.get(
            "reserved_payload_binding_identity_sha256"
        ),
        "selection_validation_identity_sha256": evidence.get(
            "selection_validation_identity_sha256"
        ),
        "final_test_identity_sha256": evidence.get("final_test_identity_sha256"),
        "decontamination_report_sha256": report.get("report_sha256"),
        "execution_identity_sha256": evidence.get("execution_identity_sha256"),
        "final_test_payload_accessed_for_decontamination": True,
    }
    for key, expected in bindings.items():
        if receipt.get(key) != expected:
            raise ValueError(f"run receipt binding drift: {key}")

    required_false = (
        "final_test_outcomes_read",
        "raw_text_persisted_in_bundle",
        "record_ids_persisted_in_bundle",
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
    )
    for key in required_false:
        if receipt.get(key) is not False:
            raise ValueError(f"run receipt truth boundary widened: {key}")
    if receipt.get("durable_bundle_hash_only") is not True:
        raise ValueError("run receipt durable bundle is not hash-only")
    if receipt.get("authorized_training_exposure") != 0 or isinstance(
        receipt.get("authorized_training_exposure"), bool
    ):
        raise ValueError("run receipt authorized training exposure is not exact zero")


def _publish_bundle(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name in (REPORT_NAME, EVIDENCE_NAME, RECEIPT_NAME):
            payload = files[name]
            target = temporary / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output_dir.exists():
            raise FileExistsError(
                f"output bundle appeared during publication: {output_dir}"
            )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run current exact-authority reserved-evaluation decontamination and "
            "publish one atomic hash-only evidence bundle"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--training-records-jsonl", type=Path, required=True)
    parser.add_argument("--training-handoff-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--reserved-binding-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--expected-training-handoff-identity-sha256", required=True)
    parser.add_argument("--expected-reserved-binding-identity-sha256", required=True)
    parser.add_argument(
        "--expected-selection-validation-identity-sha256", required=True
    )
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument(
        "--expected-decontamination-implementation-git-sha", required=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # This gate intentionally precedes all payload reads.
    actual_head = require_exact_checkout(
        args.repo_root, args.expected_decontamination_implementation_git_sha
    )
    if args.output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {args.output_dir}")

    training_records, training_records_sha256 = _load_jsonl_with_sha(
        args.training_records_jsonl
    )
    training_handoff, training_handoff_sha256 = _load_json_with_sha(
        args.training_handoff_json
    )
    evaluation_records, evaluation_records_sha256 = _load_jsonl_with_sha(
        args.evaluation_records_jsonl
    )
    reserved_binding, reserved_binding_sha256 = _load_json_with_sha(
        args.reserved_binding_json
    )
    input_hashes = {
        "training_records_jsonl": training_records_sha256,
        "training_handoff_json": training_handoff_sha256,
        "evaluation_records_jsonl": evaluation_records_sha256,
        "reserved_binding_json": reserved_binding_sha256,
    }

    report, evidence = execute_reserved_decontamination(
        training_records,
        evaluation_records,
        training_handoff_evidence=training_handoff,
        reserved_payload_binding=reserved_binding,
        expected_inventory_identity_sha256=args.expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=(
            args.expected_training_handoff_identity_sha256
        ),
        expected_reserved_binding_identity_sha256=(
            args.expected_reserved_binding_identity_sha256
        ),
        expected_selection_validation_identity_sha256=(
            args.expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
        quarantine_cross_source_families=True,
    )
    verify_execution_evidence(evidence, report)

    report_bytes = _file_bytes(report)
    evidence_bytes = _file_bytes(evidence)
    receipt = build_run_receipt(
        implementation_git_sha=actual_head,
        input_file_sha256=input_hashes,
        report=report,
        evidence=evidence,
        report_file_bytes=report_bytes,
        evidence_file_bytes=evidence_bytes,
    )
    verify_run_receipt(receipt, report, evidence)
    receipt_bytes = _file_bytes(receipt)
    _publish_bundle(
        args.output_dir,
        {
            REPORT_NAME: report_bytes,
            EVIDENCE_NAME: evidence_bytes,
            RECEIPT_NAME: receipt_bytes,
        },
    )
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
