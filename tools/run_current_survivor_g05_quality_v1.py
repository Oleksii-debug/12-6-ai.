#!/usr/bin/env python3
"""Execute frozen G05 quality semantics over the exact post-1247 survivor payload.

Raw payload text is job-local. Durable outputs are text-free and retain zero
training credit. The complete G05 row root used to authorize one run is derived
from a separately rebuilt survivor copy whose exact upstream identity is pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA = "12-6.current-survivor-g05-real-execution.v1"
SURVIVOR_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}
MODES = {"uk", "en", "code"}


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: str, name: str) -> str:
    _require(
        len(value) == 64 and all(ch in "0123456789abcdef" for ch in value),
        f"{name} must be lowercase 64-hex",
    )
    return value


def _require_git_sha(value: str, name: str) -> str:
    _require(
        len(value) == 40 and all(ch in "0123456789abcdef" for ch in value),
        f"{name} must be lowercase 40-hex",
    )
    return value


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


def _validate_survivor_evidence(
    path: Path,
    *,
    expected_jsonl_sha256: str,
    expected_inventory_root: str,
    expected_payload_root: str,
    expected_records: int,
    expected_bytes: int,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, "survivor evidence root must be object")
    _require(
        value.get("schema_version")
        == "12-6.post1247-data526-survivor-materialization-evidence.v1",
        "survivor evidence schema drift",
    )
    _require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    result = value.get("result")
    _require(type(result) is dict, "survivor result missing")
    _require(
        result.get("record_payload_jsonl_sha256") == expected_jsonl_sha256,
        "survivor JSONL authority drift",
    )
    _require(
        result.get("record_inventory_digest_sha256") == expected_inventory_root,
        "survivor record inventory drift",
    )
    _require(
        result.get("payload_inventory_digest_sha256") == expected_payload_root,
        "survivor payload inventory drift",
    )
    _require(result.get("record_count") == expected_records, "survivor record count drift")
    _require(result.get("total_payload_bytes") == expected_bytes, "survivor byte drift")
    _require(value.get("authorized_optimized_target_exposure") == 0, "exposure widened")
    _require(value.get("training_executed") is False, "training boundary widened")
    _require(value.get("optimizer_updates_executed") == 0, "optimizer boundary widened")
    _require(value.get("learned_weights_created") is False, "learned-weight boundary widened")
    _require(value.get("final_test_outcomes_read") is False, "final-test boundary widened")
    _require(value.get("paid_compute_used") is False, "paid-compute boundary widened")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--expected-records-jsonl", type=Path, required=True)
    parser.add_argument("--survivor-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-survivor-jsonl-sha256", required=True)
    parser.add_argument("--expected-survivor-inventory-root", required=True)
    parser.add_argument("--expected-survivor-payload-root", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--g05-product-head", required=True)
    parser.add_argument("--survivor-product-head", required=True)
    parser.add_argument("--execution-head-sha", required=True)
    parser.add_argument("--output-authority", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    expected_jsonl = _require_sha256(
        args.expected_survivor_jsonl_sha256,
        "expected_survivor_jsonl_sha256",
    )
    expected_inventory = _require_sha256(
        args.expected_survivor_inventory_root,
        "expected_survivor_inventory_root",
    )
    expected_payload = _require_sha256(
        args.expected_survivor_payload_root,
        "expected_survivor_payload_root",
    )
    input_manifest = _require_sha256(args.input_manifest_sha256, "input_manifest_sha256")
    g05_head = _require_git_sha(args.g05_product_head, "g05_product_head")
    survivor_head = _require_git_sha(args.survivor_product_head, "survivor_product_head")
    execution_head = _require_git_sha(args.execution_head_sha, "execution_head_sha")

    raw, rows = _load_jsonl(args.records_jsonl)
    expected_raw, expected_rows = _load_jsonl(args.expected_records_jsonl)
    _require(_sha256(raw) == expected_jsonl, "actual survivor JSONL SHA drift")
    _require(_sha256(expected_raw) == expected_jsonl, "independent survivor JSONL SHA drift")
    _require(raw == expected_raw, "independent survivor payload copies are not byte-identical")
    _require(len(rows) == args.expected_records, "actual survivor record count drift")
    _require(len(expected_rows) == args.expected_records, "independent record count drift")
    actual_bytes = sum(len(row["normalized_payload"].encode("utf-8")) for row in rows)
    expected_copy_bytes = sum(
        len(row["normalized_payload"].encode("utf-8")) for row in expected_rows
    )
    _require(actual_bytes == args.expected_bytes, "actual survivor payload bytes drift")
    _require(expected_copy_bytes == args.expected_bytes, "independent payload bytes drift")

    _validate_survivor_evidence(
        args.survivor_evidence_json,
        expected_jsonl_sha256=expected_jsonl,
        expected_inventory_root=expected_inventory,
        expected_payload_root=expected_payload,
        expected_records=args.expected_records,
        expected_bytes=args.expected_bytes,
    )

    actual_row_root = _g05_row_root(rows)
    independently_expected_row_root = _g05_row_root(expected_rows)
    _require(
        actual_row_root == independently_expected_row_root,
        "independent complete G05 row identity mismatch",
    )

    g05_records = _g05_rows(rows)
    authority = build_quality_execution_authority(
        g05_records,
        input_manifest_sha256=input_manifest,
        expected_input_rows_sha256=independently_expected_row_root,
    )
    identity = verify_quality_execution_authority(
        authority,
        g05_records,
        expected_input_manifest_sha256=input_manifest,
        expected_input_rows_sha256=independently_expected_row_root,
        expected_execution_identity_sha256=authority["execution_identity_sha256"],
    )

    counts = authority["counts"]
    byte_summary = authority["bytes"]
    _require(counts["records"] == args.expected_records, "G05 record count drift")
    _require(byte_summary["input_utf8_bytes"] == args.expected_bytes, "G05 input bytes drift")
    _require(
        byte_summary["retained_utf8_bytes"] + byte_summary["rejected_utf8_bytes"]
        == args.expected_bytes,
        "G05 byte conservation failure",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "dependencies": {
            "g05_product_head": g05_head,
            "survivor_product_head": survivor_head,
            "survivor_jsonl_sha256": expected_jsonl,
            "survivor_record_inventory_root": expected_inventory,
            "survivor_payload_inventory_root": expected_payload,
        },
        "g05_input_rows_sha256": independently_expected_row_root,
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
                "g05_input_rows_sha256": independently_expected_row_root,
                "g05_execution_identity_sha256": identity,
                "evidence_identity_sha256": evidence["evidence_identity_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
