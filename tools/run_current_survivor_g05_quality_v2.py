#!/usr/bin/env python3
"""Execute canonical G05 over the exact real post-1247 survivor payload.

The trust direction is intentionally one-way: immutable identities are pinned in
this runner and verified against the checked-out repository. Callers may supply
only the two job-local survivor payload copies and output locations; they cannot
supply dependency heads, manifests, expected roots, or execution identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA = "12-6.current-survivor-g05-real-execution.v2"
SURVIVOR_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}
MODES = {"uk", "en", "code"}

# Independently qualified Product identities now integrated on main.
CANONICAL_G05_PRODUCT_HEAD = "8cd75ea89fa933459e75609141f95a79e1a7887f"
CANONICAL_G05_MERGE_COMMIT = "d76dc24837db37aaab77394e2d7e98a2c4f24197"
CANONICAL_G05_MODULE_PATH = Path("src/twelve_six/data/quality_execution_authority.py")
CANONICAL_G05_MODULE_BLOB_SHA1 = "c8963d2d697f3cd124a5b511d2883a93f8caf34d"

CANONICAL_SURVIVOR_PRODUCT_HEAD = "e3f0ead971091bd4a42316dfcc97aa1c3b116f66"
CANONICAL_SURVIVOR_MERGE_COMMIT = "19fde0d392efb0afe05a1b362df7b577a03edde9"
CANONICAL_SURVIVOR_EVIDENCE_PATH = Path(
    "evidence/data526/post1247/materialization_evidence.json"
)
CANONICAL_SURVIVOR_EVIDENCE_BLOB_SHA1 = "e74b6d0cba76ae8ff045298d830900261ae05775"
CANONICAL_SURVIVOR_EVIDENCE_IDENTITY = (
    "284d122a9afd4e5d4676d78a202d3001e5cf87438776022d3004d61fdf10e579"
)
CANONICAL_SURVIVOR_JSONL_SHA256 = (
    "3f60cfe55435daf53908c492be358f36d7ebbc2ebee532c921a69ba92b2f6b25"
)
CANONICAL_SURVIVOR_INVENTORY_ROOT = (
    "90e306ce74a82016c835a5e106088f7bdb586e13010301b5002ce5d55e11e27c"
)
CANONICAL_SURVIVOR_PAYLOAD_ROOT = (
    "36e30427a8f6bc089c911690af82b1c59bfec1c883f3107597ca7fb07838b8df"
)
CANONICAL_SURVIVOR_RECORDS = 272
CANONICAL_SURVIVOR_BYTES = 5_921_963
CANONICAL_SURVIVOR_SOURCE_OBJECTS = 259

# The G05 authority names this field "input_manifest_sha256". The canonical
# survivor record-inventory root is the externally frozen manifest identity.
CANONICAL_INPUT_MANIFEST_SHA256 = CANONICAL_SURVIVOR_INVENTORY_ROOT


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324 - Git object identity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_int(value: Any, name: str) -> int:
    _require(type(value) is int, f"{name} must be an exact integer")
    return value


def _require_exact_bool(value: Any, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be an exact boolean")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _checkout_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    _require(
        len(head) == 40 and all(ch in "0123456789abcdef" for ch in head),
        "actual checkout HEAD is not lowercase 40-hex",
    )
    return head


def _load_jsonl(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(raw.splitlines()):
        value = json.loads(line.decode("utf-8"))
        _require(type(value) is dict, f"row {index} must be an object")
        _require(set(value) == SURVIVOR_RECORD_KEYS, f"row {index} schema drift")
        for key in SURVIVOR_RECORD_KEYS:
            _require(
                type(value[key]) is str and value[key],
                f"row {index}.{key} must be a non-empty string",
            )
        record_id = value["record_id"]
        _require(record_id not in seen, f"duplicate record_id: {record_id}")
        _require(value["modality"] in MODES, f"row {index} modality drift")
        value["normalized_payload"].encode("utf-8", errors="strict")
        seen.add(record_id)
        rows.append(value)
    _require(rows, "survivor payload must not be empty")
    return raw, rows


def _g05_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "id": row["record_id"],
            "text": row["normalized_payload"],
            "mode": row["modality"],
        }
        for row in rows
    ]


def _g05_row_root(rows: list[dict[str, str]]) -> str:
    projection = [
        {
            "record_id": row["record_id"],
            "mode": row["modality"],
            "payload_sha256": _sha256(row["normalized_payload"].encode("utf-8")),
            "utf8_bytes": len(row["normalized_payload"].encode("utf-8")),
        }
        for row in sorted(rows, key=lambda row: row["record_id"])
    ]
    return _sha256(_cjson(projection))


def _validate_survivor_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(
        _git_blob_sha1(raw) == CANONICAL_SURVIVOR_EVIDENCE_BLOB_SHA1,
        "checked-in survivor evidence Git blob drift",
    )
    value = json.loads(raw.decode("utf-8"))
    _require(type(value) is dict, "survivor evidence root must be object")

    observed_identity = value.get("evidence_identity_sha256")
    _require(
        observed_identity == CANONICAL_SURVIVOR_EVIDENCE_IDENTITY,
        "survivor evidence identity drift",
    )
    identity_core = dict(value)
    del identity_core["evidence_identity_sha256"]
    _require(
        _sha256(_cjson(identity_core)) == CANONICAL_SURVIVOR_EVIDENCE_IDENTITY,
        "survivor evidence self-hash mismatch",
    )
    _require(
        value.get("schema_version")
        == "12-6.post1247-data526-survivor-materialization-evidence.v1",
        "survivor evidence schema drift",
    )
    _require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    result = value.get("result")
    _require(type(result) is dict, "survivor result missing")
    _require(
        result.get("record_payload_jsonl_sha256") == CANONICAL_SURVIVOR_JSONL_SHA256,
        "survivor JSONL authority drift",
    )
    _require(
        result.get("record_inventory_digest_sha256") == CANONICAL_SURVIVOR_INVENTORY_ROOT,
        "survivor record inventory drift",
    )
    _require(
        result.get("payload_inventory_digest_sha256") == CANONICAL_SURVIVOR_PAYLOAD_ROOT,
        "survivor payload inventory drift",
    )
    _require(
        _require_exact_int(result.get("record_count"), "result.record_count")
        == CANONICAL_SURVIVOR_RECORDS,
        "survivor record count drift",
    )
    _require(
        _require_exact_int(result.get("total_payload_bytes"), "result.total_payload_bytes")
        == CANONICAL_SURVIVOR_BYTES,
        "survivor byte drift",
    )
    _require(
        _require_exact_int(result.get("source_object_count"), "result.source_object_count")
        == CANONICAL_SURVIVOR_SOURCE_OBJECTS,
        "survivor source-object drift",
    )
    _require(
        _require_exact_int(
            value.get("authorized_optimized_target_exposure"),
            "authorized_optimized_target_exposure",
        )
        == 0,
        "exposure widened",
    )
    _require(
        _require_exact_bool(value.get("tokenizer_fit_authorized"), "tokenizer_fit_authorized")
        is False,
        "tokenizer boundary widened",
    )
    _require(
        _require_exact_int(value.get("optimizer_updates_executed"), "optimizer_updates_executed")
        == 0,
        "optimizer boundary widened",
    )
    _require(
        _require_exact_bool(value.get("training_executed"), "training_executed") is False,
        "training boundary widened",
    )
    _require(
        _require_exact_bool(value.get("learned_weights_created"), "learned_weights_created")
        is False,
        "learned-weight boundary widened",
    )
    _require(
        _require_exact_bool(value.get("final_test_outcomes_read"), "final_test_outcomes_read")
        is False,
        "final-test boundary widened",
    )
    _require(
        _require_exact_bool(value.get("paid_compute_used"), "paid_compute_used") is False,
        "paid-compute boundary widened",
    )
    return value


def _verify_checkout_assets(repo_root: Path) -> dict[str, str]:
    g05_raw = (repo_root / CANONICAL_G05_MODULE_PATH).read_bytes()
    g05_blob = _git_blob_sha1(g05_raw)
    _require(
        g05_blob == CANONICAL_G05_MODULE_BLOB_SHA1,
        "checked-in canonical G05 implementation Git blob drift",
    )
    evidence = repo_root / CANONICAL_SURVIVOR_EVIDENCE_PATH
    _validate_survivor_evidence(evidence)
    return {
        "g05_module_git_blob_sha1": g05_blob,
        "survivor_evidence_git_blob_sha1": _git_blob_sha1(evidence.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--independent-records-jsonl", type=Path, required=True)
    parser.add_argument("--output-authority", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    repo_root = _repo_root()
    execution_head = _checkout_head(repo_root)
    asset_bindings = _verify_checkout_assets(repo_root)
    runner_blob = _git_blob_sha1(Path(__file__).read_bytes())

    raw, rows = _load_jsonl(args.records_jsonl)
    independent_raw, independent_rows = _load_jsonl(args.independent_records_jsonl)
    _require(
        _sha256(raw) == CANONICAL_SURVIVOR_JSONL_SHA256,
        "actual survivor JSONL SHA drift",
    )
    _require(
        _sha256(independent_raw) == CANONICAL_SURVIVOR_JSONL_SHA256,
        "independent survivor JSONL SHA drift",
    )
    _require(raw == independent_raw, "independent survivor payload copies are not byte-identical")
    _require(
        len(rows) == len(independent_rows) == CANONICAL_SURVIVOR_RECORDS,
        "survivor record count drift",
    )
    actual_bytes = sum(len(row["normalized_payload"].encode("utf-8")) for row in rows)
    independent_bytes = sum(
        len(row["normalized_payload"].encode("utf-8")) for row in independent_rows
    )
    _require(
        actual_bytes == independent_bytes == CANONICAL_SURVIVOR_BYTES,
        "survivor payload byte drift",
    )

    actual_row_root = _g05_row_root(rows)
    independent_row_root = _g05_row_root(independent_rows)
    _require(
        actual_row_root == independent_row_root,
        "independent complete G05 row identity mismatch",
    )

    g05_records = _g05_rows(rows)
    authority = build_quality_execution_authority(
        g05_records,
        input_manifest_sha256=CANONICAL_INPUT_MANIFEST_SHA256,
        expected_input_rows_sha256=independent_row_root,
    )
    identity = verify_quality_execution_authority(
        authority,
        g05_records,
        expected_input_manifest_sha256=CANONICAL_INPUT_MANIFEST_SHA256,
        expected_input_rows_sha256=independent_row_root,
        expected_execution_identity_sha256=authority["execution_identity_sha256"],
    )

    counts = authority["counts"]
    byte_summary = authority["bytes"]
    _require(
        _require_exact_int(counts["records"], "counts.records")
        == CANONICAL_SURVIVOR_RECORDS,
        "G05 record count drift",
    )
    _require(
        _require_exact_int(byte_summary["input_utf8_bytes"], "bytes.input_utf8_bytes")
        == CANONICAL_SURVIVOR_BYTES,
        "G05 input bytes drift",
    )
    _require(
        _require_exact_int(byte_summary["retained_utf8_bytes"], "bytes.retained_utf8_bytes")
        + _require_exact_int(byte_summary["rejected_utf8_bytes"], "bytes.rejected_utf8_bytes")
        == CANONICAL_SURVIVOR_BYTES,
        "G05 byte conservation failure",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "runner_git_blob_sha1": runner_blob,
        "dependencies": {
            "g05_product_head": CANONICAL_G05_PRODUCT_HEAD,
            "g05_merge_commit": CANONICAL_G05_MERGE_COMMIT,
            "g05_module_git_blob_sha1": asset_bindings["g05_module_git_blob_sha1"],
            "survivor_product_head": CANONICAL_SURVIVOR_PRODUCT_HEAD,
            "survivor_merge_commit": CANONICAL_SURVIVOR_MERGE_COMMIT,
            "survivor_evidence_git_blob_sha1": asset_bindings[
                "survivor_evidence_git_blob_sha1"
            ],
            "survivor_evidence_identity_sha256": CANONICAL_SURVIVOR_EVIDENCE_IDENTITY,
            "survivor_jsonl_sha256": CANONICAL_SURVIVOR_JSONL_SHA256,
            "survivor_record_inventory_root": CANONICAL_SURVIVOR_INVENTORY_ROOT,
            "survivor_payload_inventory_root": CANONICAL_SURVIVOR_PAYLOAD_ROOT,
        },
        "g05_input_manifest_sha256": CANONICAL_INPUT_MANIFEST_SHA256,
        "g05_input_rows_sha256": independent_row_root,
        "g05_execution_identity_sha256": identity,
        "counts": counts,
        "bytes": byte_summary,
        "current_corpus_eligible": False,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
        "raw_payloads_retained_in_evidence": False,
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_cjson(evidence_core)),
    }

    authority_raw = _cjson(authority)
    evidence_raw = _cjson(evidence)
    _require(b"normalized_payload" not in authority_raw, "source text leaked into authority")
    _require(b"normalized_payload" not in evidence_raw, "source text leaked into evidence")
    args.output_authority.parent.mkdir(parents=True, exist_ok=True)
    args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
    args.output_authority.write_bytes(authority_raw)
    args.output_evidence.write_bytes(evidence_raw)

    print(
        json.dumps(
            {
                "records": counts["records"],
                "input_utf8_bytes": byte_summary["input_utf8_bytes"],
                "retained_utf8_bytes": byte_summary["retained_utf8_bytes"],
                "rejected_utf8_bytes": byte_summary["rejected_utf8_bytes"],
                "g05_input_rows_sha256": independent_row_root,
                "g05_execution_identity_sha256": identity,
                "evidence_identity_sha256": evidence["evidence_identity_sha256"],
                "execution_head_sha": execution_head,
                "runner_git_blob_sha1": runner_blob,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
