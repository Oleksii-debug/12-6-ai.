#!/usr/bin/env python3
"""Execute dependency-bound G06 privacy evidence on the exact #1366 survivor graph.

Raw payloads are read only to verify the independently retained text-free
inventory and execute the canonical privacy scanner. The emitted evidence is
strictly text-free and grants no canonical corpus/training authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data.privacy_execution_authority import (
    PrivacyExecutionAuthorityError,
    build_privacy_execution_authority,
    input_rows_sha256,
    input_rows_sha256_from_text_free_inventory,
    verify_privacy_execution_authority,
    verify_privacy_execution_root,
)

SCHEMA_VERSION = "12-6.current-survivor-g06-dependency-bound-execution.v1"
EXECUTION_PROFILE = "LOCAL_FREE"
SURVIVOR_PR = 1366
SURVIVOR_TERMINAL_PRODUCT_HEAD = "e3f0ead971091bd4a42316dfcc97aa1c3b116f66"
SURVIVOR_REAL_EXECUTION_HEAD = "b5bf78ce8e74e74876838b80294a0a7e1ee1281b"
SURVIVOR_REAL_EXECUTION_RUN = 34656696122
SURVIVOR_REAL_EXECUTION_JOB = 103450558307
SURVIVOR_ARTIFACT_ID = 10286301597
SURVIVOR_ARTIFACT_ZIP_SHA256 = (
    "a714492f3ae4e73598e91c121e5ffdb617e319e2fecd1b8c19cf62f69ab11ac5"
)
SURVIVOR_MATERIALIZATION_EVIDENCE_ID = (
    "284d122a9afd4e5d4676d78a202d3001e5cf87438776022d3004d61fdf10e579"
)
SURVIVOR_RECORD_PAYLOAD_JSONL_SHA256 = (
    "3f60cfe55435daf53908c492be358f36d7ebbc2ebee532c921a69ba92b2f6b25"
)
SURVIVOR_RECORD_INVENTORY_SHA256 = (
    "90e306ce74a82016c835a5e106088f7bdb586e13010301b5002ce5d55e11e27c"
)
SURVIVOR_PAYLOAD_INVENTORY_SHA256 = (
    "36e30427a8f6bc089c911690af82b1c59bfec1c883f3107597ca7fb07838b8df"
)
SURVIVOR_RECORD_COUNT = 272
SURVIVOR_SOURCE_OBJECT_COUNT = 259
SURVIVOR_TOTAL_PAYLOAD_BYTES = 5_921_963

_INVENTORY_KEYS = {
    "schema_version",
    "record_count",
    "total_payload_bytes",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "records",
}
_INVENTORY_ROW_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
}
_SURVIVOR_ROW_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}

_TRUTH_BOUNDARY = {
    "dependency_bound_g06_executed": True,
    "canonical_current_corpus_g06": False,
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
    "raw_payloads_retained_in_output": False,
}


class CurrentSurvivorG06Error(ValueError):
    """Raised when exact survivor or G06 execution evidence fails closed."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CurrentSurvivorG06Error(f"{name} must be a non-negative integer")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CurrentSurvivorG06Error(f"{path} root must be an object")
    return value


def _validate_inventory(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_json(path)
    if set(value) != _INVENTORY_KEYS:
        raise CurrentSurvivorG06Error("survivor inventory top-level schema drift")
    if value["schema_version"] != "12-6.data526-record-inventory.v1":
        raise CurrentSurvivorG06Error("survivor inventory schema version drift")
    rows = value["records"]
    if not isinstance(rows, list) or not rows:
        raise CurrentSurvivorG06Error("survivor inventory records missing")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _INVENTORY_ROW_KEYS:
            raise CurrentSurvivorG06Error(f"inventory row {index} schema drift")
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise CurrentSurvivorG06Error(f"inventory row {index} id malformed")
        if record_id in seen:
            raise CurrentSurvivorG06Error(f"duplicate inventory record id: {record_id}")
        for field in ("source_id", "family", "modality", "payload_sha256"):
            if not isinstance(row[field], str) or not row[field]:
                raise CurrentSurvivorG06Error(
                    f"inventory row {index} field {field} malformed"
                )
        if row["modality"] not in {"uk", "en", "code"}:
            raise CurrentSurvivorG06Error(f"inventory row {index} modality drift")
        if len(row["payload_sha256"]) != 64 or any(
            ch not in "0123456789abcdef" for ch in row["payload_sha256"]
        ):
            raise CurrentSurvivorG06Error(
                f"inventory row {index} payload hash malformed"
            )
        payload_bytes = _strict_int(row["payload_bytes"], f"inventory[{index}].payload_bytes")
        normalized.append(
            {
                "record_id": record_id,
                "source_id": row["source_id"],
                "family": row["family"],
                "modality": row["modality"],
                "payload_sha256": row["payload_sha256"],
                "payload_bytes": payload_bytes,
            }
        )
        seen.add(record_id)
        source_ids.add(row["source_id"])

    normalized.sort(key=lambda row: row["record_id"])
    full_root = _sha256(_cjson(normalized))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_root = _sha256(_cjson(payload_projection))
    record_count = _strict_int(value["record_count"], "inventory.record_count")
    total_bytes = _strict_int(
        value["total_payload_bytes"], "inventory.total_payload_bytes"
    )
    if record_count != len(normalized) or record_count != SURVIVOR_RECORD_COUNT:
        raise CurrentSurvivorG06Error("survivor record count drift")
    if total_bytes != sum(row["payload_bytes"] for row in normalized):
        raise CurrentSurvivorG06Error("survivor byte accounting mismatch")
    if total_bytes != SURVIVOR_TOTAL_PAYLOAD_BYTES:
        raise CurrentSurvivorG06Error("survivor total payload bytes drift")
    if len(source_ids) != SURVIVOR_SOURCE_OBJECT_COUNT:
        raise CurrentSurvivorG06Error("survivor source-object count drift")
    if value["record_inventory_digest_sha256"] != full_root:
        raise CurrentSurvivorG06Error("survivor inventory self-root mismatch")
    if full_root != SURVIVOR_RECORD_INVENTORY_SHA256:
        raise CurrentSurvivorG06Error("survivor record inventory authority drift")
    if value["payload_inventory_digest_sha256"] != payload_root:
        raise CurrentSurvivorG06Error("survivor payload inventory self-root mismatch")
    if payload_root != SURVIVOR_PAYLOAD_INVENTORY_SHA256:
        raise CurrentSurvivorG06Error("survivor payload inventory authority drift")

    expected_g06_root = input_rows_sha256_from_text_free_inventory(normalized)
    return {**value, "records": normalized}, expected_g06_root


def _load_survivors(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    payload_sha256 = _sha256(raw)
    if payload_sha256 != SURVIVOR_RECORD_PAYLOAD_JSONL_SHA256:
        raise CurrentSurvivorG06Error("survivor JSONL identity drift")

    inputs: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict) or set(row) != _SURVIVOR_ROW_KEYS:
            raise CurrentSurvivorG06Error(
                f"survivor row {line_number} schema drift"
            )
        record_id = row["record_id"]
        source_id = row["source_id"]
        family = row["family"]
        mode = row["modality"]
        text = row["normalized_payload"]
        for field_name, field_value in (
            ("record_id", record_id),
            ("source_id", source_id),
            ("family", family),
            ("modality", mode),
            ("normalized_payload", text),
        ):
            if not isinstance(field_value, str) or not field_value:
                raise CurrentSurvivorG06Error(
                    f"survivor row {line_number} {field_name} malformed"
                )
        if record_id in seen:
            raise CurrentSurvivorG06Error(
                f"duplicate survivor record id: {record_id}"
            )
        if mode not in {"uk", "en", "code"}:
            raise CurrentSurvivorG06Error(
                f"survivor row {line_number} modality drift"
            )
        seen.add(record_id)
        inputs.append({"id": record_id, "text": text, "mode": mode})

    if len(inputs) != SURVIVOR_RECORD_COUNT:
        raise CurrentSurvivorG06Error("survivor raw record count drift")
    return inputs, payload_sha256


def execute(
    survivor_records_jsonl: Path,
    inventory_json: Path,
) -> dict[str, Any]:
    inventory, expected_input_root = _validate_inventory(inventory_json)
    records, raw_jsonl_sha256 = _load_survivors(survivor_records_jsonl)

    observed_input_root = input_rows_sha256(records)
    if observed_input_root != expected_input_root:
        raise CurrentSurvivorG06Error(
            "raw survivors do not match independently retained text-free inventory"
        )

    authority = build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=expected_input_root,
    )
    identity = authority["execution_identity_sha256"]
    verify_privacy_execution_root(
        authority,
        expected_input_rows_sha256=expected_input_root,
        expected_execution_identity_sha256=identity,
    )
    verify_privacy_execution_authority(
        authority,
        records,
        expected_input_rows_sha256=expected_input_root,
        expected_execution_identity_sha256=identity,
    )

    core = {
        "schema_version": SCHEMA_VERSION,
        "execution_profile": EXECUTION_PROFILE,
        "dependency": {
            "survivor_pr": SURVIVOR_PR,
            "survivor_terminal_product_head": SURVIVOR_TERMINAL_PRODUCT_HEAD,
            "survivor_real_execution_head": SURVIVOR_REAL_EXECUTION_HEAD,
            "survivor_real_execution_run": SURVIVOR_REAL_EXECUTION_RUN,
            "survivor_real_execution_job": SURVIVOR_REAL_EXECUTION_JOB,
            "survivor_artifact_id": SURVIVOR_ARTIFACT_ID,
            "survivor_artifact_zip_sha256": SURVIVOR_ARTIFACT_ZIP_SHA256,
            "survivor_materialization_evidence_identity_sha256": SURVIVOR_MATERIALIZATION_EVIDENCE_ID,
            "survivor_record_payload_jsonl_sha256": raw_jsonl_sha256,
            "survivor_record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "survivor_payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
            "survivor_record_count": inventory["record_count"],
            "survivor_total_payload_bytes": inventory["total_payload_bytes"],
            "survivor_source_object_count": SURVIVOR_SOURCE_OBJECT_COUNT,
        },
        "g06_input_rows_sha256": expected_input_root,
        "privacy_execution_authority": authority,
        "truth_boundary": dict(_TRUTH_BOUNDARY),
    }
    return {**core, "evidence_identity_sha256": _sha256(_cjson(core))}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute text-free dependency-bound G06 privacy evidence over exact #1366 survivors"
    )
    parser.add_argument("--survivor-records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = execute(args.survivor_records_jsonl, args.inventory_json)
    except (CurrentSurvivorG06Error, PrivacyExecutionAuthorityError) as exc:
        raise SystemExit(f"G06 execution failed closed: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_cjson(result))
    print(result["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
