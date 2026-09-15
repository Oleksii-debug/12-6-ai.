#!/usr/bin/env python3
"""Execute canonical G05/G06 over the exact physical #2107 clean DATA526.

This is a narrow binding/execution carrier, not a second quality/privacy
framework. It consumes the exact Nomis-free DATA526 payload and text-free
inventory, reuses the canonical G05/G06 engines, and emits text-free zero-credit
execution authorities. Raw payloads are never copied into emitted evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from twelve_six.data.quality_execution_authority import (
    QualityExecutionAuthorityError,
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA = "12-6.d03-clean-g05-g06-current-data526-execution.v1"
G06_ENVELOPE_SCHEMA = "12-6.current-survivor-g06-dependency-bound-execution.v1"
EXECUTION_PROFILE = "LOCAL_FREE"

SOURCE_PR = 2107
SOURCE_INTEGRATED_MERGE_SHA = "5e8ced70926a2e7f14aad984cf552f2dc646aa59"
SOURCE_RELEASE_HEAD_SHA = "07754c5a1d61669061e608323ea35ddc093bb946"
SOURCE_PHYSICAL_EXECUTION_HEAD_SHA = "3d7dd363f6b1701694c00c76c6353077f80d1492"
SOURCE_PHYSICAL_RUN_ID = 34911721640
SOURCE_PHYSICAL_JOB_ID = 104200523132
SOURCE_ARTIFACT_ID = 10374891614
SOURCE_ARTIFACT_ZIP_SHA256 = (
    "663eec03d04252b6de574bf97ea0975703e0ba25779ca27340cb3acea941943a"
)
SOURCE_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)
SOURCE_RECORD_PAYLOAD_JSONL_SHA256 = (
    "dc22d829921890ea8c5b51cbedaae099688c37c3cb624a597973305c7fa5b2c3"
)
SOURCE_RECORD_INVENTORY_SHA256 = (
    "7d6782e91243505c01b0f2f6d6f85b5bbe3a6abf628d73721b1e0c1c77e4f352"
)
SOURCE_PAYLOAD_INVENTORY_SHA256 = (
    "59f9c5a7b5db9e45fcde2e3bab10a4adc97cb6832f906fc241edc3e35248b576"
)
SOURCE_RECORD_COUNT = 274
SOURCE_PAYLOAD_BYTES = 6_093_662
SOURCE_OBJECT_COUNT = 261

QUALITY_MODULE_PATH = Path("src/twelve_six/data/quality_execution_authority.py")
QUALITY_MODULE_GIT_BLOB_SHA1 = "c8963d2d697f3cd124a5b511d2883a93f8caf34d"
PRIVACY_EXECUTION_MODULE_PATH = Path(
    "src/twelve_six/data/privacy_execution_authority.py"
)
PRIVACY_EXECUTION_MODULE_GIT_BLOB_SHA1 = (
    "e798adf593b4bd4475acb47f017b4acc36a439a9"
)

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
_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}
_MODES = {"uk", "en", "code"}

_ZERO_TRUTH = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}
_G06_TRUTH = {
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


class CleanG05G06ExecutionError(ValueError):
    """Raised when the current-clean physical input or binding fails closed."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _root_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise CleanG05G06ExecutionError(message)


def _strict_int(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    _need(type(value) is int and value >= minimum, f"{field} must be exact integer >= {minimum}")
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
    _need(
        len(head) == 40 and all(ch in "0123456789abcdef" for ch in head),
        "checkout HEAD is not lowercase 40-hex",
    )
    return head


def _verify_checkout_assets(repo_root: Path) -> dict[str, str]:
    quality_blob = _git_blob_sha1((repo_root / QUALITY_MODULE_PATH).read_bytes())
    _need(quality_blob == QUALITY_MODULE_GIT_BLOB_SHA1, "canonical G05 module blob drift")
    privacy_blob = _git_blob_sha1(
        (repo_root / PRIVACY_EXECUTION_MODULE_PATH).read_bytes()
    )
    _need(
        privacy_blob == PRIVACY_EXECUTION_MODULE_GIT_BLOB_SHA1,
        "canonical G06 execution module blob drift",
    )
    return {
        "quality_execution_module_git_blob_sha1": quality_blob,
        "privacy_execution_module_git_blob_sha1": privacy_blob,
    }


def _load_inventory(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(type(value) is dict, "inventory root must be an object")
    _need(set(value) == _INVENTORY_KEYS, "inventory top-level schema drift")
    _need(
        value.get("schema_version") == "12-6.data526-record-inventory.v1",
        "inventory schema drift",
    )
    rows = value.get("records")
    _need(type(rows) is list and bool(rows), "inventory records missing")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        _need(
            isinstance(raw_row, Mapping) and set(raw_row) == _INVENTORY_ROW_KEYS,
            f"inventory[{index}] schema drift",
        )
        row = dict(raw_row)
        record_id = row["record_id"]
        _need(type(record_id) is str and bool(record_id), f"inventory[{index}].record_id malformed")
        _need(record_id not in seen, f"duplicate inventory record id: {record_id}")
        for field in ("source_id", "family", "modality", "payload_sha256"):
            _need(type(row[field]) is str and bool(row[field]), f"inventory[{index}].{field} malformed")
        _need(row["modality"] in _MODES, f"inventory[{index}].modality drift")
        digest = row["payload_sha256"]
        _need(
            len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
            f"inventory[{index}].payload_sha256 malformed",
        )
        row["payload_bytes"] = _strict_int(
            row["payload_bytes"], f"inventory[{index}].payload_bytes", positive=True
        )
        normalized.append(row)
        seen.add(record_id)
        source_ids.add(row["source_id"])

    normalized.sort(key=lambda row: row["record_id"])
    record_root = _sha256(_root_json(normalized))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_root = _sha256(_root_json(payload_projection))
    record_count = _strict_int(value.get("record_count"), "inventory.record_count", positive=True)
    total_bytes = _strict_int(
        value.get("total_payload_bytes"), "inventory.total_payload_bytes", positive=True
    )
    _need(record_count == len(normalized) == SOURCE_RECORD_COUNT, "clean DATA526 record count drift")
    _need(total_bytes == sum(row["payload_bytes"] for row in normalized), "inventory byte accounting mismatch")
    _need(total_bytes == SOURCE_PAYLOAD_BYTES, "clean DATA526 payload bytes drift")
    _need(len(source_ids) == SOURCE_OBJECT_COUNT, "clean DATA526 source-object count drift")
    _need(value.get("record_inventory_digest_sha256") == record_root, "inventory record self-root mismatch")
    _need(record_root == SOURCE_RECORD_INVENTORY_SHA256, "clean DATA526 record inventory authority drift")
    _need(value.get("payload_inventory_digest_sha256") == payload_root, "inventory payload self-root mismatch")
    _need(payload_root == SOURCE_PAYLOAD_INVENTORY_SHA256, "clean DATA526 payload inventory authority drift")

    canonical = {**value, "records": normalized}
    return canonical, input_rows_sha256_from_text_free_inventory(normalized)


def _load_records(path: Path, inventory: Mapping[str, Any]) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    raw_sha = _sha256(raw)
    _need(raw_sha == SOURCE_RECORD_PAYLOAD_JSONL_SHA256, "clean DATA526 JSONL identity drift")

    inv_rows = inventory.get("records")
    _need(type(inv_rows) is list, "inventory records missing")
    by_id = {row["record_id"]: row for row in inv_rows}
    inputs: list[dict[str, str]] = []
    seen: set[str] = set()
    total_bytes = 0
    source_ids: set[str] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line.decode("utf-8"))
        _need(type(value) is dict and set(value) == _RECORD_KEYS, f"records line {line_number} schema drift")
        for field in _RECORD_KEYS:
            _need(type(value[field]) is str and bool(value[field]), f"records line {line_number}.{field} malformed")
        record_id = value["record_id"]
        _need(record_id not in seen, f"duplicate record id: {record_id}")
        _need(value["modality"] in _MODES, f"records line {line_number}.modality drift")
        payload = value["normalized_payload"].encode("utf-8", errors="strict")
        expected = by_id.get(record_id)
        _need(expected is not None, f"record absent from inventory: {record_id}")
        for field in ("source_id", "family", "modality"):
            _need(value[field] == expected[field], f"record/inventory {field} drift: {record_id}")
        _need(_sha256(payload) == expected["payload_sha256"], f"record payload hash drift: {record_id}")
        _need(len(payload) == expected["payload_bytes"], f"record payload bytes drift: {record_id}")
        seen.add(record_id)
        source_ids.add(value["source_id"])
        total_bytes += len(payload)
        inputs.append({"id": record_id, "text": value["normalized_payload"], "mode": value["modality"]})

    _need(seen == set(by_id), "raw/inventory closed-world membership drift")
    _need(len(inputs) == SOURCE_RECORD_COUNT, "raw clean DATA526 record count drift")
    _need(total_bytes == SOURCE_PAYLOAD_BYTES, "raw clean DATA526 payload bytes drift")
    _need(len(source_ids) == SOURCE_OBJECT_COUNT, "raw clean DATA526 source-object count drift")
    return inputs, raw_sha


def execute(records_jsonl: Path, inventory_json: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_root = _repo_root()
    execution_head = _checkout_head(repo_root)
    bindings = _verify_checkout_assets(repo_root)
    inventory, expected_input_root = _load_inventory(inventory_json)
    records, raw_jsonl_sha = _load_records(records_jsonl, inventory)
    observed_input_root = input_rows_sha256(records)
    _need(observed_input_root == expected_input_root, "raw clean DATA526 rows do not match text-free inventory")

    g05 = build_quality_execution_authority(
        records,
        input_manifest_sha256=SOURCE_DATA526_EVIDENCE_IDENTITY_SHA256,
        expected_input_rows_sha256=expected_input_root,
    )
    g05_identity = verify_quality_execution_authority(
        g05,
        records,
        expected_input_manifest_sha256=SOURCE_DATA526_EVIDENCE_IDENTITY_SHA256,
        expected_input_rows_sha256=expected_input_root,
        expected_execution_identity_sha256=g05["execution_identity_sha256"],
    )

    g06_authority = build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=expected_input_root,
    )
    g06_identity = g06_authority["execution_identity_sha256"]
    verify_privacy_execution_root(
        g06_authority,
        expected_input_rows_sha256=expected_input_root,
        expected_execution_identity_sha256=g06_identity,
    )
    verify_privacy_execution_authority(
        g06_authority,
        records,
        expected_input_rows_sha256=expected_input_root,
        expected_execution_identity_sha256=g06_identity,
    )

    dependency = {
        "survivor_pr": SOURCE_PR,
        "survivor_terminal_product_head": SOURCE_RELEASE_HEAD_SHA,
        "survivor_real_execution_head": SOURCE_PHYSICAL_EXECUTION_HEAD_SHA,
        "survivor_real_execution_run": SOURCE_PHYSICAL_RUN_ID,
        "survivor_real_execution_job": SOURCE_PHYSICAL_JOB_ID,
        "survivor_artifact_id": SOURCE_ARTIFACT_ID,
        "survivor_artifact_zip_sha256": SOURCE_ARTIFACT_ZIP_SHA256,
        "survivor_materialization_evidence_identity_sha256": SOURCE_DATA526_EVIDENCE_IDENTITY_SHA256,
        "survivor_record_payload_jsonl_sha256": raw_jsonl_sha,
        "survivor_record_inventory_digest_sha256": inventory["record_inventory_digest_sha256"],
        "survivor_payload_inventory_digest_sha256": inventory["payload_inventory_digest_sha256"],
        "survivor_record_count": inventory["record_count"],
        "survivor_total_payload_bytes": inventory["total_payload_bytes"],
        "survivor_source_object_count": SOURCE_OBJECT_COUNT,
    }
    g06_core = {
        "schema_version": G06_ENVELOPE_SCHEMA,
        "execution_profile": EXECUTION_PROFILE,
        "dependency": dependency,
        "g06_input_rows_sha256": expected_input_root,
        "privacy_execution_authority": g06_authority,
        "truth_boundary": dict(_G06_TRUTH),
    }
    g06_envelope = {
        **g06_core,
        "evidence_identity_sha256": _sha256(_cjson(g06_core)),
    }

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": EXECUTION_PROFILE,
        "execution_head_sha": execution_head,
        "runner_git_blob_sha1": _git_blob_sha1(Path(__file__).read_bytes()),
        "bindings": {
            "source_pr": SOURCE_PR,
            "source_integrated_merge_sha": SOURCE_INTEGRATED_MERGE_SHA,
            "source_release_head_sha": SOURCE_RELEASE_HEAD_SHA,
            "source_physical_execution_head_sha": SOURCE_PHYSICAL_EXECUTION_HEAD_SHA,
            "source_physical_run_id": SOURCE_PHYSICAL_RUN_ID,
            "source_physical_job_id": SOURCE_PHYSICAL_JOB_ID,
            "source_artifact_id": SOURCE_ARTIFACT_ID,
            "source_artifact_zip_sha256": SOURCE_ARTIFACT_ZIP_SHA256,
            "source_data526_evidence_identity_sha256": SOURCE_DATA526_EVIDENCE_IDENTITY_SHA256,
            "source_record_payload_jsonl_sha256": raw_jsonl_sha,
            "source_record_inventory_sha256": SOURCE_RECORD_INVENTORY_SHA256,
            "source_payload_inventory_sha256": SOURCE_PAYLOAD_INVENTORY_SHA256,
            **bindings,
        },
        "input_rows_sha256": expected_input_root,
        "g05_execution_identity_sha256": g05_identity,
        "g06_envelope_identity_sha256": g06_envelope["evidence_identity_sha256"],
        "g06_execution_identity_sha256": g06_identity,
        "record_count": SOURCE_RECORD_COUNT,
        "payload_bytes": SOURCE_PAYLOAD_BYTES,
        "source_object_count": SOURCE_OBJECT_COUNT,
        "raw_payloads_retained_in_output": False,
        "truth_boundary": dict(_ZERO_TRUTH),
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_cjson(evidence_core)),
    }

    for name, document in (("G05", g05), ("G06", g06_envelope), ("execution", evidence)):
        raw = _cjson(document)
        _need(b"normalized_payload" not in raw, f"{name} output leaked normalized payload")
        _need(b'"text"' not in raw, f"{name} output leaked source text field")
    return g05, g06_envelope, evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute canonical G05/G06 over exact #2107 clean DATA526"
    )
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--output-g05-authority", type=Path, required=True)
    parser.add_argument("--output-g06-envelope", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    try:
        g05, g06, evidence = execute(args.records_jsonl, args.inventory_json)
    except (
        CleanG05G06ExecutionError,
        QualityExecutionAuthorityError,
        PrivacyExecutionAuthorityError,
    ) as exc:
        raise SystemExit(f"clean G05/G06 execution failed closed: {exc}") from exc

    outputs = (
        (args.output_g05_authority, g05),
        (args.output_g06_envelope, g06),
        (args.output_evidence, evidence),
    )
    for path, document in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_cjson(document))
    print(evidence["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
