#!/usr/bin/env python3
"""Replay incumbent G05/G06 on the physically reconstructed clean DATA526 graph.

The corpus bytes are reconstructed under the immutable #2107 physical authority
head, while this tracked carrier and the unchanged G05/G06 engines execute from
the current Product head. The two identities are deliberately distinct: a new
Product commit must not pretend it rematerialized the corpus historically.

Durable output is text-free and grants no corpus/training/tokenizer authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data.privacy_execution_authority import (
    build_privacy_execution_authority,
    input_rows_sha256,
    input_rows_sha256_from_text_free_inventory,
    verify_privacy_execution_authority,
    verify_privacy_execution_root,
)
from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA_VERSION = "12-6.d03-clean-g05-g06-physical-replay.v1"
EXECUTION_PROFILE = "LOCAL_FREE"

EXPECTED_RECORDS = 274
EXPECTED_SOURCE_OBJECTS = 261
EXPECTED_PAYLOAD_BYTES = 6_093_662
EXPECTED_RECORDS_JSONL_SHA256 = (
    "dc22d829921890ea8c5b51cbedaae099688c37c3cb624a597973305c7fa5b2c3"
)
EXPECTED_RECORD_INVENTORY_SHA256 = (
    "7d6782e91243505c01b0f2f6d6f85b5bbe3a6abf628d73721b1e0c1c77e4f352"
)
EXPECTED_PAYLOAD_INVENTORY_SHA256 = (
    "59f9c5a7b5db9e45fcde2e3bab10a4adc97cb6832f906fc241edc3e35248b576"
)
EXPECTED_SOURCE_REPORT_SHA256 = (
    "db72eb1d4f86cd025741efd0c612c1f2e124ce24dfa133548c375d781331c93c"
)
EXPECTED_SURVIVOR_AUTHORITY_SHA256 = (
    "e1c94f5eed4afa78a63d577fe33a67e28305c1e084c27cd1061061ad113f8ce5"
)
RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)
RETAINED_EXECUTION_HEAD = "3d7dd363f6b1701694c00c76c6353077f80d1492"
RETAINED_RUN_ID = 34911721640
RETAINED_JOB_ID = 104200523132
RETAINED_ARTIFACT_ID = 10374891614
RETAINED_ARTIFACT_ZIP_SHA256 = (
    "663eec03d04252b6de574bf97ea0975703e0ba25779ca27340cb3acea941943a"
)

NOMIS_RECORD_ID = "ua.verba.nomis1864.bounded24"
NOMIS_PAYLOAD_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
PR462_AUTHORITY_SHA256 = (
    "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7"
)

QUALITY_MODULE = Path("src/twelve_six/data/quality_execution_authority.py")
QUALITY_MODULE_BLOB_SHA1 = "4659a9d4aba49908f372250904a54361c8d8cf46"
PRIVACY_MODULE = Path("src/twelve_six/data/privacy_execution_authority.py")
PRIVACY_MODULE_BLOB_SHA1 = "9215287e81c0a82f05ec8405dc4f34c60313c193"
CLEAN_MATERIALIZER = Path("tools/materialize_d03_nomis_free_data526_successor_v1.py")
CLEAN_MATERIALIZER_BLOB_SHA1 = "d183bb83df73ca4ca528d4360048c9ccc031cf33"

_DATA526_SCHEMA = "12-6.d03-nomis-free-data526-successor-evidence.v1"
_INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}
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
_MODES = {"uk", "en", "code"}

_ZERO_FALSE_BOUNDARY = {
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
    "raw_payloads_retained_in_output": False,
    "whole_corpus_external_llm_cleanliness_claimed": False,
}


class CleanG05G06ReplayError(RuntimeError):
    """Fail-closed replay validation or execution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanG05G06ReplayError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _cjson(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _strict_int(value: Any, field: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{field} must be a non-negative exact integer",
    )
    return value


def _strict_bool(value: Any, field: str) -> bool:
    _require(type(value) is bool, f"{field} must be an exact boolean")
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanG05G06ReplayError(f"cannot read JSON {path}: {exc}") from exc
    _require(type(value) is dict, f"{path} root must be object")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _checkout_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    _require(
        len(head) == 40 and all(ch in "0123456789abcdef" for ch in head),
        "checkout HEAD must be lowercase 40-hex",
    )
    return head


def _verify_clean_checkout(root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(status == "", "tracked worktree must be clean before physical replay")
    replacements = subprocess.run(
        ["git", "-C", str(root), "for-each-ref", "--format=%(refname)", "refs/replace"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not replacements, "Git replacement refs are forbidden for physical replay")


def _verify_engine_bindings(root: Path) -> dict[str, str]:
    expected = {
        QUALITY_MODULE: QUALITY_MODULE_BLOB_SHA1,
        PRIVACY_MODULE: PRIVACY_MODULE_BLOB_SHA1,
        CLEAN_MATERIALIZER: CLEAN_MATERIALIZER_BLOB_SHA1,
    }
    observed: dict[str, str] = {}
    for path, expected_blob in expected.items():
        actual = _git_blob_sha1((root / path).read_bytes())
        _require(actual == expected_blob, f"bound Git blob drift: {path}")
        observed[str(path)] = actual
    return observed


def _validate_data526_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("schema_version") == _DATA526_SCHEMA, "DATA526 evidence schema drift")
    _require(
        value.get("execution_profile") == EXECUTION_PROFILE,
        "DATA526 execution profile drift",
    )
    _require(
        value.get("source_report_sha256") == EXPECTED_SOURCE_REPORT_SHA256,
        "clean source report authority drift",
    )
    _require(
        value.get("survivor_authority_sha256") == EXPECTED_SURVIVOR_AUTHORITY_SHA256,
        "clean survivor authority drift",
    )
    clean = value.get("clean_data526")
    _require(type(clean) is dict, "clean_data526 evidence missing")
    expected = {
        "record_count": EXPECTED_RECORDS,
        "source_object_count": EXPECTED_SOURCE_OBJECTS,
        "total_payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "record_payload_jsonl_sha256": EXPECTED_RECORDS_JSONL_SHA256,
        "record_inventory_digest_sha256": EXPECTED_RECORD_INVENTORY_SHA256,
        "payload_inventory_digest_sha256": EXPECTED_PAYLOAD_INVENTORY_SHA256,
    }
    for key, expected_value in expected.items():
        observed = clean.get(key)
        if type(expected_value) is int:
            _strict_int(observed, f"clean_data526.{key}")
        _require(observed == expected_value, f"clean_data526.{key} authority drift")

    _require(
        _strict_bool(
            value.get("raw_text_emitted_to_durable_evidence"),
            "raw_text_emitted_to_durable_evidence",
        )
        is False,
        "raw text durable-evidence boundary widened",
    )
    boundary = value.get("truth_boundary")
    _require(type(boundary) is dict, "DATA526 truth boundary missing")
    for key in (
        "corpus_released",
        "decontamination_executed_for_successor",
        "post_composition_quality_privacy_passed",
        "balance_release_claimed",
        "split_pack_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "learned_weights_created",
        "final_test_payload_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "raw_payloads_committed_to_repository",
        "raw_payloads_uploaded_as_public_evidence",
    ):
        _require(
            _strict_bool(boundary.get(key), f"truth_boundary.{key}") is False,
            f"truth boundary widened: {key}",
        )
    for key in ("authorized_training_exposure", "optimizer_updates"):
        _require(
            _strict_int(boundary.get(key), f"truth_boundary.{key}") == 0,
            f"truth boundary widened: {key}",
        )
    _require(
        _strict_bool(
            boundary.get("clean_data526_record_graph_materialized"),
            "truth_boundary.clean_data526_record_graph_materialized",
        )
        is True,
        "clean DATA526 physical materialization claim missing",
    )

    claimed_identity = value.get("evidence_identity_sha256")
    _require(
        type(claimed_identity) is str
        and len(claimed_identity) == 64
        and all(ch in "0123456789abcdef" for ch in claimed_identity),
        "DATA526 evidence identity malformed",
    )
    core = dict(value)
    del core["evidence_identity_sha256"]
    _require(
        _sha256(_canonical(core)) == claimed_identity,
        "DATA526 evidence self-hash mismatch",
    )
    return dict(clean)


def _validate_retained_materialization_authority(value: Mapping[str, Any]) -> None:
    _require(
        value.get("execution_head_sha") == RETAINED_EXECUTION_HEAD,
        "DATA526 reconstruction did not execute on retained #2107 authority head",
    )
    _require(
        value.get("evidence_identity_sha256")
        == RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
        "DATA526 retained evidence identity drift",
    )


def _g05_manifest_authority(value: Mapping[str, Any]) -> str:
    _validate_retained_materialization_authority(value)
    identity = value.get("evidence_identity_sha256")
    _require(type(identity) is str, "DATA526 retained evidence identity malformed")
    return identity


def _validate_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    value = _read_json(path)
    _require(set(value) == _INVENTORY_KEYS, "clean inventory top-level schema drift")
    _require(value["schema_version"] == _INVENTORY_SCHEMA, "clean inventory schema drift")
    rows = value["records"]
    _require(type(rows) is list and rows, "clean inventory rows missing")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        _require(
            type(raw_row) is dict and set(raw_row) == _INVENTORY_ROW_KEYS,
            f"inventory row {index} schema drift",
        )
        row = dict(raw_row)
        for field in ("record_id", "source_id", "family", "modality", "payload_sha256"):
            _require(
                type(row[field]) is str and bool(row[field]),
                f"inventory[{index}].{field} malformed",
            )
        _require(row["record_id"] not in seen, f"duplicate inventory id: {row['record_id']}")
        _require(row["modality"] in _MODES, f"inventory[{index}].modality drift")
        _require(
            len(row["payload_sha256"]) == 64
            and all(ch in "0123456789abcdef" for ch in row["payload_sha256"]),
            f"inventory[{index}].payload_sha256 malformed",
        )
        _require(
            row["record_id"] != NOMIS_RECORD_ID and row["source_id"] != NOMIS_RECORD_ID,
            "Nomis quarantined identity reappeared in clean inventory",
        )
        _require(
            row["payload_sha256"] != NOMIS_PAYLOAD_SHA256,
            "Nomis quarantined payload reappeared in clean inventory",
        )
        row["payload_bytes"] = _strict_int(
            row["payload_bytes"], f"inventory[{index}].payload_bytes"
        )
        normalized.append(row)
        seen.add(row["record_id"])
        source_ids.add(row["source_id"])

    normalized.sort(key=lambda row: row["record_id"])
    inventory_root = _sha256(_canonical(normalized))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    payload_root = _sha256(_canonical(payload_projection))
    _require(
        _strict_int(value["record_count"], "inventory.record_count") == EXPECTED_RECORDS,
        "clean inventory record-count drift",
    )
    _require(len(normalized) == EXPECTED_RECORDS, "clean inventory row-count drift")
    _require(
        _strict_int(value["total_payload_bytes"], "inventory.total_payload_bytes")
        == EXPECTED_PAYLOAD_BYTES,
        "clean inventory byte drift",
    )
    _require(
        sum(row["payload_bytes"] for row in normalized) == EXPECTED_PAYLOAD_BYTES,
        "clean inventory byte accounting mismatch",
    )
    _require(len(source_ids) == EXPECTED_SOURCE_OBJECTS, "clean source-object count drift")
    _require(inventory_root == EXPECTED_RECORD_INVENTORY_SHA256, "inventory root mismatch")
    _require(
        value["record_inventory_digest_sha256"] == inventory_root,
        "inventory self-root mismatch",
    )
    _require(payload_root == EXPECTED_PAYLOAD_INVENTORY_SHA256, "payload root mismatch")
    _require(
        value["payload_inventory_digest_sha256"] == payload_root,
        "payload inventory self-root mismatch",
    )
    return normalized, input_rows_sha256_from_text_free_inventory(normalized)


def _load_records(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    _require(_sha256(raw) == EXPECTED_RECORDS_JSONL_SHA256, "clean record JSONL hash drift")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    source_ids: set[str] = set()
    payload_bytes = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line.strip()), f"blank JSONL line {line_number}")
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_strict_object_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CleanG05G06ReplayError(f"invalid JSONL line {line_number}: {exc}") from exc
        _require(
            type(value) is dict and set(value) == _RECORD_KEYS,
            f"record row {line_number} schema drift",
        )
        for field in _RECORD_KEYS:
            _require(
                type(value[field]) is str and bool(value[field]),
                f"record[{line_number}].{field} malformed",
            )
        _require(value["record_id"] not in seen, f"duplicate record_id: {value['record_id']}")
        _require(value["modality"] in _MODES, f"record[{line_number}].modality drift")
        encoded = value["normalized_payload"].encode("utf-8", errors="strict")
        payload_bytes += len(encoded)
        seen.add(value["record_id"])
        source_ids.add(value["source_id"])
        rows.append(value)
    _require(len(rows) == EXPECTED_RECORDS, "clean record count drift")
    _require(payload_bytes == EXPECTED_PAYLOAD_BYTES, "clean payload byte drift")
    _require(len(source_ids) == EXPECTED_SOURCE_OBJECTS, "clean source-object count drift")
    return sorted(rows, key=lambda row: row["record_id"])


def _bind_records_to_inventory(
    records: list[dict[str, str]],
    inventory: list[dict[str, Any]],
) -> None:
    _require(len(records) == len(inventory), "record/inventory cardinality mismatch")
    for record, item in zip(records, inventory, strict=True):
        payload = record["normalized_payload"].encode("utf-8")
        _require(record["record_id"] == item["record_id"], "record/inventory id mismatch")
        for key in ("source_id", "family", "modality"):
            _require(record[key] == item[key], f"record/inventory {key} mismatch")
        _require(_sha256(payload) == item["payload_sha256"], "payload hash mismatch")
        _require(len(payload) == item["payload_bytes"], "payload byte mismatch")


def _quality_input_root(records: list[dict[str, str]]) -> str:
    projection = [
        {
            "record_id": row["record_id"],
            "mode": row["modality"],
            "payload_sha256": _sha256(row["normalized_payload"].encode("utf-8")),
            "utf8_bytes": len(row["normalized_payload"].encode("utf-8")),
        }
        for row in records
    ]
    return _sha256(_cjson(projection))


def _verify_shared_input_root(records: list[dict[str, str]], expected_root: str) -> str:
    observed_g06_root = input_rows_sha256(records)
    _require(observed_g06_root == expected_root, "G06 raw/inventory input root mismatch")
    quality_records = [
        {
            "record_id": row["id"],
            "source_id": "",
            "family": "",
            "modality": row["mode"],
            "normalized_payload": row["text"],
        }
        for row in records
    ]
    observed_g05_root = _quality_input_root(quality_records)
    _require(observed_g05_root == expected_root, "G05 raw/inventory input root mismatch")
    return expected_root


def _physical_identity(execution_head: str) -> dict[str, Any]:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    job = os.environ.get("GITHUB_JOB", "")
    _require(repository == "Oleksii-debug/12-6-ai.", "GITHUB_REPOSITORY authority drift")
    _require(sha == execution_head, "GitHub Product SHA does not match checkout HEAD")
    _require(run_id.isdigit() and int(run_id) > 0, "GITHUB_RUN_ID missing/malformed")
    _require(
        run_attempt.isdigit() and int(run_attempt) > 0,
        "GITHUB_RUN_ATTEMPT missing/malformed",
    )
    _require(bool(job), "GITHUB_JOB missing")
    return {
        "repository": repository,
        "workflow_run_id": int(run_id),
        "workflow_run_attempt": int(run_attempt),
        "workflow_job": job,
        "execution_head_sha": execution_head,
    }


def execute(
    *,
    records_jsonl: Path,
    inventory_json: Path,
    data526_evidence_json: Path,
) -> dict[str, Any]:
    root = _repo_root()
    execution_head = _checkout_head(root)
    _verify_clean_checkout(root)
    bindings = _verify_engine_bindings(root)
    physical = _physical_identity(execution_head)

    data526_evidence = _read_json(data526_evidence_json)
    clean = _validate_data526_evidence(data526_evidence)
    input_manifest = _g05_manifest_authority(data526_evidence)
    inventory, expected_input_root = _validate_inventory(inventory_json)
    records = _load_records(records_jsonl)
    _bind_records_to_inventory(records, inventory)

    g_records = [
        {"id": row["record_id"], "text": row["normalized_payload"], "mode": row["modality"]}
        for row in records
    ]
    shared_input_root = _verify_shared_input_root(g_records, expected_input_root)

    g05 = build_quality_execution_authority(
        g_records,
        input_manifest_sha256=input_manifest,
        expected_input_rows_sha256=shared_input_root,
    )
    g05_identity = verify_quality_execution_authority(
        g05,
        g_records,
        expected_input_manifest_sha256=input_manifest,
        expected_input_rows_sha256=shared_input_root,
        expected_execution_identity_sha256=g05["execution_identity_sha256"],
    )
    _require(g05["counts"]["records"] == EXPECTED_RECORDS, "G05 coverage drift")
    _require(
        g05["bytes"]["input_utf8_bytes"] == EXPECTED_PAYLOAD_BYTES,
        "G05 input byte drift",
    )
    _require(
        g05["bytes"]["retained_utf8_bytes"] + g05["bytes"]["rejected_utf8_bytes"]
        == EXPECTED_PAYLOAD_BYTES,
        "G05 byte conservation failure",
    )

    g06 = build_privacy_execution_authority(
        g_records,
        expected_input_rows_sha256=shared_input_root,
    )
    g06_identity = g06["execution_identity_sha256"]
    verify_privacy_execution_root(
        g06,
        expected_input_rows_sha256=shared_input_root,
        expected_execution_identity_sha256=g06_identity,
    )
    verify_privacy_execution_authority(
        g06,
        g_records,
        expected_input_rows_sha256=shared_input_root,
        expected_execution_identity_sha256=g06_identity,
    )
    _require(g06["counts"]["records"] == EXPECTED_RECORDS, "G06 coverage drift")
    _require(
        g06["total_input_utf8_bytes"] == EXPECTED_PAYLOAD_BYTES,
        "G06 input byte drift",
    )

    deterministic_projection = {
        "product_execution_head_sha": execution_head,
        "engine_bindings": bindings,
        "corpus_materialization_authority_head_sha": RETAINED_EXECUTION_HEAD,
        "data526_evidence_identity_sha256": RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
        "record_payload_jsonl_sha256": EXPECTED_RECORDS_JSONL_SHA256,
        "record_inventory_digest_sha256": EXPECTED_RECORD_INVENTORY_SHA256,
        "payload_inventory_digest_sha256": EXPECTED_PAYLOAD_INVENTORY_SHA256,
        "g05_input_manifest_sha256": input_manifest,
        "shared_input_rows_sha256": shared_input_root,
        "g05_execution_identity_sha256": g05_identity,
        "g05_execution_rows_sha256": g05["execution_rows_sha256"],
        "g05_counts": g05["counts"],
        "g05_bytes": g05["bytes"],
        "g06_execution_identity_sha256": g06_identity,
        "g06_execution_rows_sha256": g06["execution_rows_sha256"],
        "g06_counts": g06["counts"],
        "g06_total_input_utf8_bytes": g06["total_input_utf8_bytes"],
        "g06_detector_counts": g06["detector_counts"],
    }
    projection_sha256 = _sha256(_cjson(deterministic_projection))

    core = {
        "schema_version": SCHEMA_VERSION,
        "execution_profile": EXECUTION_PROFILE,
        "status": "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT",
        "physical_identity": physical,
        "replay_projection_sha256": projection_sha256,
        "engine_bindings": bindings,
        "retained_clean_authority": {
            "execution_head_sha": RETAINED_EXECUTION_HEAD,
            "workflow_run_id": RETAINED_RUN_ID,
            "workflow_job_id": RETAINED_JOB_ID,
            "artifact_id": RETAINED_ARTIFACT_ID,
            "artifact_zip_sha256": RETAINED_ARTIFACT_ZIP_SHA256,
            "data526_evidence_identity_sha256": RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
            "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "survivor_authority_sha256": EXPECTED_SURVIVOR_AUTHORITY_SHA256,
            "record_payload_jsonl_sha256": EXPECTED_RECORDS_JSONL_SHA256,
            "record_inventory_digest_sha256": EXPECTED_RECORD_INVENTORY_SHA256,
            "payload_inventory_digest_sha256": EXPECTED_PAYLOAD_INVENTORY_SHA256,
            "nomis_record_id_rejected": NOMIS_RECORD_ID,
            "nomis_payload_sha256_rejected": NOMIS_PAYLOAD_SHA256,
            "pr462_authority_sha256_not_admitted": PR462_AUTHORITY_SHA256,
        },
        "physical_clean_reconstruction": {
            "materialization_authority_head_sha": data526_evidence["execution_head_sha"],
            "evidence_identity_sha256": data526_evidence["evidence_identity_sha256"],
            **clean,
            "physical_reconstruction_verified": True,
        },
        "g05": {
            "input_manifest_sha256": input_manifest,
            "input_rows_sha256": shared_input_root,
            "execution_identity_sha256": g05_identity,
            "execution_rows_sha256": g05["execution_rows_sha256"],
            "counts": g05["counts"],
            "bytes": g05["bytes"],
            "authority": g05,
        },
        "g06": {
            "input_rows_sha256": shared_input_root,
            "execution_identity_sha256": g06_identity,
            "execution_rows_sha256": g06["execution_rows_sha256"],
            "counts": g06["counts"],
            "total_input_utf8_bytes": g06["total_input_utf8_bytes"],
            "detector_counts": g06["detector_counts"],
            "authority": g06,
        },
        "truth_boundary": dict(_ZERO_FALSE_BOUNDARY),
        "scope_note": (
            "Canonical G05/G06 authorities use the scoped whole-corpus non-claim; "
            "this replay makes no whole-corpus external-LLM cleanliness claim."
        ),
    }
    receipt = {**core, "replay_identity_sha256": _sha256(_cjson(core))}
    serialized = _cjson(receipt)
    _require(
        b"normalized_payload" not in serialized,
        "raw normalized payload leaked into replay receipt",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--data526-evidence-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = execute(
        records_jsonl=args.records_jsonl,
        inventory_json=args.inventory_json,
        data526_evidence_json=args.data526_evidence_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _require(not args.output.exists(), "replay output must be create-only")
    with args.output.open("xb") as handle:
        handle.write(_cjson(receipt))
    print(receipt["replay_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
