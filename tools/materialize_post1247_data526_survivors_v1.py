#!/usr/bin/env python3
"""Materialize the exact post-#1247 DATA-526 survivor graph.

This is a thin deletion-only successor to the canonical DATA-526 V8 record
composer. It accepts two independently rebuilt copies of the exact 275-record
pre-decontamination graph, authenticates the checked-in #924 and #1247 durable
authorities, removes every record whose SHA256(source_id UTF-8) is in the
terminal decontamination exclusion set, and emits one job-local survivor JSONL
plus text-free inventory/evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from materialize_data526_record_inventory_v1 import canonical_json, load_jsonl, materialize

PRE_EVIDENCE_GIT_BLOB_SHA1 = "d12edac3a01a37a06af11c0be11ffce6690dbb30"
DECONTAM_EVIDENCE_GIT_BLOB_SHA1 = "b85de0291064c5fb247ba80507c561281b74b510"
PRE_RECORD_PAYLOAD_SHA256 = "18a933b2e637b4cff497f4acfbff31041d4c0ba21fba397af53d213edfd7a1b7"
PRE_RECORD_INVENTORY_SHA256 = "766cc01d610373183e73a632e18674b1d7afc5aaa7e3888638029e0044b26103"
PRE_PAYLOAD_INVENTORY_SHA256 = "56ef7a457f4c4f649ad97359c2a177131632d838756d769e4b1a2ec3f3a2a278"
PRE_RECORD_COUNT = 275
PRE_SOURCE_COUNT = 262
PRE_PAYLOAD_BYTES = 6_095_321
DECONTAM_TERMINAL_IDENTITY = "c87e12081f075a3e3650500480fa83e66eff220c0fc67138bce6dc2bcdf53c6f"
DECONTAM_REPORT_SHA256 = "c3f9e1ee3a300a5fc82cbd178d040e270aa85508b9f763af0ec83c44ab1bb509"
DECONTAM_EXECUTION_SHA256 = "d530c73da082989c32b8adead9c7267f5a08c7e15dcd305826862bef8cf37961"
EXPECTED_EXCLUDED_RECORDS = 3
EXPECTED_EXCLUDED_BYTES = 173_358
EXPECTED_SURVIVOR_RECORDS = 272
EXPECTED_SURVIVOR_SOURCES = 259
EXPECTED_SURVIVOR_BYTES = 5_921_963
RECORD_KEYS = {"record_id", "source_id", "family", "modality", "normalized_payload"}


class Post1247MaterializationError(RuntimeError):
    """Raised when a scientific authority or payload invariant fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post1247MaterializationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _validated_git_sha(value: str) -> str:
    _require(len(value) == 40 and all(ch in "0123456789abcdef" for ch in value), "execution head SHA must be lowercase 40-hex")
    return value


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"{name} must be an integer >= {minimum}")
    return int(value)


def _read_json_with_blob_pin(path: Path, *, expected_blob_sha1: str) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(_git_blob_sha1(raw) == expected_blob_sha1, f"authority Git blob drift: {path}")
    value = json.loads(raw.decode("utf-8"))
    _require(isinstance(value, dict), f"authority root must be object: {path}")
    return value


def _canonical_record_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


def _validate_record_shape(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        _require(set(record) == RECORD_KEYS, f"record {index} schema drift")
        for key in RECORD_KEYS:
            _require(isinstance(record[key], str) and record[key], f"record {index} requires non-empty string {key}")


def validate_pre_graph(records: list[dict[str, Any]], authority: Mapping[str, Any]) -> dict[str, Any]:
    _validate_record_shape(records)
    raw = _canonical_record_bytes(records)
    _require(_sha256(raw) == PRE_RECORD_PAYLOAD_SHA256 == authority.get("record_payload_jsonl_sha256"), "pre-decontamination record payload identity drift")
    inventory = materialize(records)
    _require(inventory["record_count"] == PRE_RECORD_COUNT == authority.get("record_count"), "pre-decontamination record count drift")
    _require(inventory["total_payload_bytes"] == PRE_PAYLOAD_BYTES == authority.get("total_payload_bytes"), "pre-decontamination payload bytes drift")
    _require(inventory["record_inventory_digest_sha256"] == PRE_RECORD_INVENTORY_SHA256 == authority.get("record_inventory_digest_sha256"), "pre-decontamination record inventory drift")
    _require(inventory["payload_inventory_digest_sha256"] == PRE_PAYLOAD_INVENTORY_SHA256 == authority.get("payload_inventory_digest_sha256"), "pre-decontamination payload inventory drift")
    source_count = len({record["source_id"] for record in records})
    _require(source_count == PRE_SOURCE_COUNT == authority.get("source_object_count"), "pre-decontamination source count drift")
    return inventory


def validate_pre_authority(value: Mapping[str, Any]) -> None:
    _require(value.get("schema_version") == "12-6.data526-v8-record-composition-evidence.v1", "pre evidence schema drift")
    _require(value.get("decontamination_executed") is False, "pre evidence decontamination boundary weakened")
    _require(value.get("authorized_unique_optimized_targets") == 0, "pre evidence optimized-target boundary weakened")
    _require(value.get("tokenizer_fit_executed") is False, "pre evidence tokenizer boundary weakened")
    _require(value.get("training_executed") is False, "pre evidence training boundary weakened")
    _require(value.get("optimizer_updates") == 0, "pre evidence optimizer boundary weakened")
    _require(value.get("final_test_payload_accessed") is False, "pre evidence final-test boundary drift")
    validate_pre_summary(value)


def validate_pre_summary(value: Mapping[str, Any]) -> None:
    _require(value.get("record_payload_jsonl_sha256") == PRE_RECORD_PAYLOAD_SHA256, "pre evidence record payload hash drift")
    _require(value.get("record_inventory_digest_sha256") == PRE_RECORD_INVENTORY_SHA256, "pre evidence record inventory hash drift")
    _require(value.get("payload_inventory_digest_sha256") == PRE_PAYLOAD_INVENTORY_SHA256, "pre evidence payload inventory hash drift")
    _require(value.get("record_count") == PRE_RECORD_COUNT, "pre evidence count drift")
    _require(value.get("source_object_count") == PRE_SOURCE_COUNT, "pre evidence source count drift")
    _require(value.get("total_payload_bytes") == PRE_PAYLOAD_BYTES, "pre evidence byte drift")


def _validate_hash(value: Any, *, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{name} must be lowercase 64-hex")
    return value


def validate_decontam_authority(value: Mapping[str, Any]) -> set[str]:
    _require(value.get("schema_version") == "12-6.current-reserved-decontamination-terminal-evidence.v1", "decontamination evidence schema drift")
    _require(value.get("execution_profile") == "LOCAL_FREE", "decontamination execution profile drift")
    _require(value.get("workflow_conclusion") == "success", "decontamination workflow non-success")
    _require(value.get("terminal_evidence_identity_sha256") == DECONTAM_TERMINAL_IDENTITY, "decontamination terminal identity drift")
    execution = value.get("execution")
    _require(isinstance(execution, Mapping), "decontamination execution missing")
    _require(execution.get("status") == "PASS_WITH_EXCLUSIONS", "decontamination status drift")
    _require(execution.get("repeat_execution_byte_identical") is True, "decontamination repeat proof missing")
    _require(execution.get("decontamination_report_sha256") == DECONTAM_REPORT_SHA256, "decontamination report identity drift")
    _require(execution.get("execution_identity_sha256") == DECONTAM_EXECUTION_SHA256, "decontamination execution identity drift")
    counts = execution.get("counts")
    _require(isinstance(counts, Mapping), "decontamination counts missing")
    _require(_strict_int(counts.get("training_records"), name="training_records") == PRE_SOURCE_COUNT, "decontamination training source count drift")
    _require(_strict_int(counts.get("excluded_training_records"), name="excluded_training_records") == EXPECTED_EXCLUDED_RECORDS, "decontamination exclusion count drift")
    _require(_strict_int(counts.get("quarantined_source_families"), name="quarantined_source_families") == 0, "unexpected quarantined family")
    rows = execution.get("excluded_records")
    _require(isinstance(rows, list) and len(rows) == EXPECTED_EXCLUDED_RECORDS, "decontamination excluded-record rows drift")
    excluded: set[str] = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"excluded row {index} must be object")
        _require(row.get("reason") == "evaluation_connected_component", f"excluded row {index} reason drift")
        source_hash = _validate_hash(row.get("source_id_sha256"), name=f"excluded[{index}].source_id_sha256")
        _require(source_hash not in excluded, "duplicate decontamination source hash")
        excluded.add(source_hash)
        _validate_hash(row.get("source_family_sha256"), name=f"excluded[{index}].source_family_sha256")
        _require(row.get("modality") in {"uk", "en", "code"}, f"excluded row {index} modality drift")
    boundary = value.get("truth_boundary")
    _require(isinstance(boundary, Mapping), "decontamination truth boundary missing")
    _require(boundary.get("authorized_training_exposure") == 0, "decontamination training exposure boundary weakened")
    _require(boundary.get("tokenizer_fit_authorized") is False, "decontamination tokenizer boundary weakened")
    _require(boundary.get("training_executed") is False, "decontamination training boundary weakened")
    _require(boundary.get("learned_weights_created") is False, "decontamination learned-weight boundary weakened")
    _require(boundary.get("final_test_outcomes_read") is False, "decontamination final-test outcome boundary weakened")
    _require(boundary.get("paid_compute_used") is False, "decontamination paid-compute boundary weakened")
    return excluded


def derive_survivors(records: list[dict[str, Any]], excluded_source_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(excluded_source_hashes, "empty exclusion set")
    survivors: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    matched_hashes: set[str] = set()
    for record in records:
        source_hash = _sha256(record["source_id"].encode("utf-8"))
        if source_hash in excluded_source_hashes:
            excluded.append(record)
            matched_hashes.add(source_hash)
        else:
            survivors.append(record)
    _require(matched_hashes == excluded_source_hashes, "not every decontamination source hash maps to the DATA526 graph")
    return survivors, excluded


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> str:
    raw = _canonical_record_bytes(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256(raw)


def materialize_terminal(
    *,
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    pre_authority: Mapping[str, Any],
    decontam_authority: Mapping[str, Any],
    execution_head_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    validate_pre_graph(records_a, pre_authority)
    validate_pre_graph(records_b, pre_authority)
    _require(_canonical_record_bytes(records_a) == _canonical_record_bytes(records_b), "independent pre-decontamination rebuilds are not byte-identical")
    excluded_hashes = validate_decontam_authority(decontam_authority)
    survivors_a, excluded_a = derive_survivors(records_a, excluded_hashes)
    survivors_b, excluded_b = derive_survivors(records_b, excluded_hashes)
    _require(_canonical_record_bytes(survivors_a) == _canonical_record_bytes(survivors_b), "independent survivor materializations are not byte-identical")
    _require(_canonical_record_bytes(excluded_a) == _canonical_record_bytes(excluded_b), "independent exclusion projections are not byte-identical")
    survivor_inventory = materialize(survivors_a)
    excluded_inventory = materialize(excluded_a)
    _require(survivor_inventory["record_count"] == EXPECTED_SURVIVOR_RECORDS, "survivor record count does not match terminal projection")
    _require(survivor_inventory["total_payload_bytes"] == EXPECTED_SURVIVOR_BYTES, "survivor payload bytes do not match terminal projection")
    _require(len({record["source_id"] for record in survivors_a}) == EXPECTED_SURVIVOR_SOURCES, "survivor source count drift")
    _require(excluded_inventory["record_count"] == EXPECTED_EXCLUDED_RECORDS, "excluded record count drift")
    _require(excluded_inventory["total_payload_bytes"] == EXPECTED_EXCLUDED_BYTES, "excluded payload bytes drift")
    _require(PRE_PAYLOAD_BYTES - excluded_inventory["total_payload_bytes"] == survivor_inventory["total_payload_bytes"], "payload byte conservation failure")
    _require(PRE_RECORD_COUNT - excluded_inventory["record_count"] == survivor_inventory["record_count"], "record count conservation failure")

    survivor_raw = _canonical_record_bytes(survivors_a)
    core = {
        "schema_version": "12-6.post1247-data526-survivor-materialization-evidence.v1",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": _validated_git_sha(execution_head_sha),
        "pre_decontamination": {
            "authority_git_blob_sha1": PRE_EVIDENCE_GIT_BLOB_SHA1,
            "record_payload_jsonl_sha256": PRE_RECORD_PAYLOAD_SHA256,
            "record_inventory_digest_sha256": PRE_RECORD_INVENTORY_SHA256,
            "payload_inventory_digest_sha256": PRE_PAYLOAD_INVENTORY_SHA256,
            "record_count": PRE_RECORD_COUNT,
            "source_object_count": PRE_SOURCE_COUNT,
            "total_payload_bytes": PRE_PAYLOAD_BYTES,
        },
        "decontamination": {
            "authority_git_blob_sha1": DECONTAM_EVIDENCE_GIT_BLOB_SHA1,
            "terminal_evidence_identity_sha256": DECONTAM_TERMINAL_IDENTITY,
            "decontamination_report_sha256": DECONTAM_REPORT_SHA256,
            "execution_identity_sha256": DECONTAM_EXECUTION_SHA256,
            "excluded_source_id_sha256": sorted(excluded_hashes),
        },
        "result": {
            "record_payload_jsonl_sha256": _sha256(survivor_raw),
            "record_inventory_digest_sha256": survivor_inventory["record_inventory_digest_sha256"],
            "payload_inventory_digest_sha256": survivor_inventory["payload_inventory_digest_sha256"],
            "record_count": survivor_inventory["record_count"],
            "source_object_count": EXPECTED_SURVIVOR_SOURCES,
            "total_payload_bytes": survivor_inventory["total_payload_bytes"],
            "excluded_record_count": excluded_inventory["record_count"],
            "excluded_payload_bytes": excluded_inventory["total_payload_bytes"],
        },
        "repeat_pre_rebuild_byte_identical": True,
        "repeat_survivor_materialization_byte_identical": True,
        "raw_payloads_committed_to_repository": False,
        "raw_payloads_uploaded_as_public_evidence": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }
    evidence = {**core, "evidence_identity_sha256": _sha256(canonical_json(core))}
    return survivors_a, survivor_inventory, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize exact post-#1247 DATA526 survivors from two independent pre-decontamination rebuilds")
    parser.add_argument("--records-jsonl-a", type=Path, required=True)
    parser.add_argument("--records-jsonl-b", type=Path, required=True)
    parser.add_argument("--pre-materialization-evidence", type=Path, required=True)
    parser.add_argument("--decontamination-evidence", type=Path, required=True)
    parser.add_argument("--survivor-records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--execution-head-sha", required=True)
    args = parser.parse_args()

    pre = _read_json_with_blob_pin(args.pre_materialization_evidence, expected_blob_sha1=PRE_EVIDENCE_GIT_BLOB_SHA1)
    validate_pre_authority(pre)
    decontam = _read_json_with_blob_pin(args.decontamination_evidence, expected_blob_sha1=DECONTAM_EVIDENCE_GIT_BLOB_SHA1)
    records_a = load_jsonl(args.records_jsonl_a)
    records_b = load_jsonl(args.records_jsonl_b)
    survivors, inventory, evidence = materialize_terminal(
        records_a=records_a,
        records_b=records_b,
        pre_authority=pre,
        decontam_authority=decontam,
        execution_head_sha=args.execution_head_sha,
    )
    records_sha = _write_jsonl(survivors, args.survivor_records_jsonl)
    _require(records_sha == evidence["result"]["record_payload_jsonl_sha256"], "survivor write identity drift")
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.write_bytes(canonical_json(inventory) + b"\n")
    args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_json.write_bytes(canonical_json(evidence) + b"\n")
    print(evidence["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
