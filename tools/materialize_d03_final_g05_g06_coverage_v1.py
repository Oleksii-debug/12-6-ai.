#!/usr/bin/env python3
"""Materialize legacy D03 G05/G06 coverage or a native current-execution preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_G05_SCHEMA = "12-6.g05-quality-execution-authority.v1"
_G05_CLASS = "G05_QUALITY_EXECUTION_ZERO_CREDIT"
_G06_ENVELOPE_SCHEMA = "12-6.current-survivor-g06-dependency-bound-execution.v1"
_G06_SCHEMA = "12-6.g06-privacy-execution-authority.v1"
_G06_CLASS = "G06_PRIVACY_EXECUTION_ZERO_CREDIT"
_G06_TERMINAL_QUALIFICATION_SCHEMA = "12-6.g06-exact-byte-terminal-qualification.v1"
_G06_TERMINAL_QUALIFICATION_STATUS = "PASS_FOR_G06_TERMINAL_CONSUMPTION"
_PREFLIGHT_SCHEMA = "12-6.d03-current-g05-g06-composition-preflight.v1"
_G05_STATUSES = {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"}
_G06_ACTIONS = {"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"}
_G06_TERMINAL_QUALIFICATION_KEYS = {
    "schema",
    "status",
    "target_pr_number",
    "target_head_git_sha",
    "real_replay_head_git_sha",
    "real_replay_run_id",
    "real_replay_job_id",
    "final_head_ci_run_id",
    "final_head_ci_job_id",
    "g06_envelope_identity_sha256",
    "g06_execution_identity_sha256",
    "input_rows_sha256",
    "repeated_execution_evidence_sha256",
    "artifact_id",
    "artifact_zip_sha256",
    "replay_record_count",
    "replay_utf8_bytes",
    "replay_count",
    "independent_audit_issue_number",
    "independent_audit_status",
    "local_free_only",
    "head_change_invalidates",
    "truth_boundary",
    "qualification_identity_sha256",
}
_G06_TERMINAL_TRUTH = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
}


class NativeCompositionError(ValueError):
    """Raised when current native G05/G06 evidence fails closed."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise NativeCompositionError(message)


def _sha256(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{field} must be SHA-256",
    )
    return value


def _sha1(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX40.fullmatch(value) is not None,
        f"{field} must be Git SHA-1",
    )
    return value


def _int(value: Any, field: str, *, positive: bool = False) -> int:
    _need(
        type(value) is int and value >= (1 if positive else 0),
        f"{field} must be an exact integer",
    )
    return value


def _obj(value: Any, field: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _identity(document: Mapping[str, Any], field: str, expected: str) -> str:
    expected = _sha256(expected, f"expected {field}")
    claimed = _sha256(document.get(field), field)
    _need(claimed == expected, f"{field} is not independently expected")
    core = dict(document)
    del core[field]
    _need(claimed == _sha(_cjson(core)), f"{field} self-hash mismatch")
    return claimed


def _zero_credit(truth: Any, field: str) -> None:
    truth = _obj(truth, field)
    for key in (
        "current_corpus_eligible",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
    ):
        _need(truth.get(key) is False, f"{field}.{key} widened")
    for key in (
        "training_authorized_bytes",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed",
    ):
        _need(
            type(truth.get(key)) is int and truth.get(key) == 0,
            f"{field}.{key} widened",
        )


def _native_rows(
    g05: Mapping[str, Any],
    g06_envelope: Mapping[str, Any],
    *,
    expected_g05_identity: str,
    expected_g06_envelope_identity: str,
    expected_g06_identity: str,
    expected_input_root: str,
    expected_survivor_evidence: str,
    expected_survivor_jsonl: str,
    expected_record_inventory: str,
    expected_payload_inventory: str,
    expected_records: int,
    expected_bytes: int,
    expected_source_objects: int,
    expected_privacy_policy: str,
    expected_privacy_impl: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    _need(g05.get("schema_version") == _G05_SCHEMA, "G05 schema drift")
    _need(g05.get("authority_class") == _G05_CLASS, "G05 authority class drift")
    _identity(g05, "execution_identity_sha256", expected_g05_identity)
    expected_input_root = _sha256(expected_input_root, "expected input rows root")
    expected_survivor_evidence = _sha256(
        expected_survivor_evidence, "expected survivor evidence"
    )
    _need(
        g05.get("input_rows_sha256") == expected_input_root, "G05 input root drift"
    )
    _need(
        g05.get("input_manifest_sha256") == expected_survivor_evidence,
        "G05 survivor evidence drift",
    )
    _zero_credit(g05.get("truth_boundary"), "G05 truth")

    g05_rows = g05.get("records")
    _need(type(g05_rows) is list and bool(g05_rows), "G05 records missing")
    _need(
        g05.get("execution_rows_sha256") == _sha(_cjson(g05_rows)),
        "G05 execution rows root drift",
    )
    g05_by_id: dict[str, Mapping[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    total_bytes = retained_bytes = rejected_bytes = 0
    for index, row in enumerate(g05_rows):
        row = _obj(row, f"G05 row[{index}]")
        record_id = row.get("record_id")
        _need(
            type(record_id) is str
            and bool(record_id)
            and record_id not in g05_by_id,
            "G05 record id drift",
        )
        _need(
            row.get("status") in _G05_STATUSES,
            f"G05 status drift: {record_id}",
        )
        _sha256(row.get("payload_sha256"), f"G05 payload {record_id}")
        nbytes = _int(row.get("utf8_bytes"), f"G05 bytes {record_id}", positive=True)
        retained = _int(
            row.get("retained_utf8_bytes"), f"G05 retained bytes {record_id}"
        )
        rejected = _int(
            row.get("rejected_utf8_bytes"), f"G05 rejected bytes {record_id}"
        )
        _need(
            retained + rejected == nbytes,
            f"G05 byte accounting drift: {record_id}",
        )
        g05_by_id[record_id] = row
        status_counts[row["status"]] += 1
        total_bytes += nbytes
        retained_bytes += retained
        rejected_bytes += rejected
    _need(list(g05_by_id) == sorted(g05_by_id), "G05 records are not sorted")
    counts = _obj(g05.get("counts"), "G05 counts")
    for key, value in {
        "records": len(g05_rows),
        "retain_all": status_counts["RETAIN_ALL"],
        "retain_partial": status_counts["RETAIN_PARTIAL"],
        "reject_document": status_counts["REJECT_DOCUMENT"],
    }.items():
        _need(
            _int(counts.get(key), f"G05 counts.{key}") == value,
            f"G05 count drift: {key}",
        )
    byte_summary = _obj(g05.get("bytes"), "G05 bytes")
    for key, value in {
        "input_utf8_bytes": total_bytes,
        "retained_utf8_bytes": retained_bytes,
        "rejected_utf8_bytes": rejected_bytes,
    }.items():
        _need(
            _int(byte_summary.get(key), f"G05 bytes.{key}") == value,
            f"G05 byte drift: {key}",
        )

    _need(
        g06_envelope.get("schema_version") == _G06_ENVELOPE_SCHEMA,
        "G06 envelope schema drift",
    )
    _need(
        g06_envelope.get("execution_profile") == "LOCAL_FREE",
        "G06 execution profile drift",
    )
    _identity(
        g06_envelope, "evidence_identity_sha256", expected_g06_envelope_identity
    )
    _need(
        g06_envelope.get("g06_input_rows_sha256") == expected_input_root,
        "G06 envelope input root drift",
    )
    _zero_credit(g06_envelope.get("truth_boundary"), "G06 envelope truth")
    dependency = _obj(g06_envelope.get("dependency"), "G06 dependency")
    exact = {
        "survivor_materialization_evidence_identity_sha256": expected_survivor_evidence,
        "survivor_record_payload_jsonl_sha256": _sha256(
            expected_survivor_jsonl, "expected survivor JSONL"
        ),
        "survivor_record_inventory_digest_sha256": _sha256(
            expected_record_inventory, "expected record inventory"
        ),
        "survivor_payload_inventory_digest_sha256": _sha256(
            expected_payload_inventory, "expected payload inventory"
        ),
    }
    for key, value in exact.items():
        _need(dependency.get(key) == value, f"G06 dependency drift: {key}")
    for key, value in {
        "survivor_record_count": expected_records,
        "survivor_total_payload_bytes": expected_bytes,
        "survivor_source_object_count": expected_source_objects,
    }.items():
        _need(
            _int(dependency.get(key), f"G06 dependency.{key}", positive=True)
            == _int(value, f"expected {key}", positive=True),
            f"G06 dependency drift: {key}",
        )

    g06 = _obj(g06_envelope.get("privacy_execution_authority"), "G06 authority")
    _need(g06.get("schema_version") == _G06_SCHEMA, "G06 schema drift")
    _need(g06.get("authority_class") == _G06_CLASS, "G06 authority class drift")
    _identity(g06, "execution_identity_sha256", expected_g06_identity)
    _need(
        g06.get("expected_input_rows_sha256") == expected_input_root,
        "G06 expected input root drift",
    )
    _need(
        g06.get("observed_input_rows_sha256") == expected_input_root,
        "G06 observed input root drift",
    )
    binding = _obj(g06.get("privacy_binding"), "G06 privacy binding")
    _need(
        binding.get("policy_sha256")
        == _sha256(expected_privacy_policy, "expected privacy policy"),
        "G06 privacy policy drift",
    )
    _need(
        binding.get("implementation_git_blob_sha1")
        == _sha1(expected_privacy_impl, "expected privacy implementation"),
        "G06 privacy implementation drift",
    )
    _zero_credit(g06.get("truth_boundary"), "G06 truth")

    g06_rows = g06.get("records")
    _need(type(g06_rows) is list and bool(g06_rows), "G06 records missing")
    _need(
        g06.get("execution_rows_sha256") == _sha(_cjson(g06_rows)),
        "G06 execution rows root drift",
    )
    g06_by_id: dict[str, Mapping[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    g06_bytes = 0
    for index, row in enumerate(g06_rows):
        row = _obj(row, f"G06 row[{index}]")
        record_id = row.get("record_id")
        _need(
            type(record_id) is str
            and bool(record_id)
            and record_id not in g06_by_id,
            "G06 record id drift",
        )
        _need(
            row.get("action") in _G06_ACTIONS,
            f"G06 action drift: {record_id}",
        )
        _sha256(row.get("payload_sha256"), f"G06 payload {record_id}")
        nbytes = _int(row.get("utf8_bytes"), f"G06 bytes {record_id}", positive=True)
        g06_by_id[record_id] = row
        action_counts[row["action"]] += 1
        g06_bytes += nbytes
    _need(list(g06_by_id) == sorted(g06_by_id), "G06 records are not sorted")
    counts = _obj(g06.get("counts"), "G06 counts")
    for key, value in {
        "records": len(g06_rows),
        "allow": action_counts["ALLOW"],
        "redact": action_counts["REDACT"],
        "quarantine": action_counts["QUARANTINE"],
        "exclude": action_counts["EXCLUDE"],
    }.items():
        _need(
            _int(counts.get(key), f"G06 counts.{key}") == value,
            f"G06 count drift: {key}",
        )
    _need(
        _int(g06.get("total_input_utf8_bytes"), "G06 total bytes") == g06_bytes,
        "G06 byte drift",
    )
    _need(
        len(g05_by_id) == expected_records and total_bytes == expected_bytes,
        "G05 survivor accounting drift",
    )
    _need(
        len(g06_by_id) == expected_records and g06_bytes == expected_bytes,
        "G06 survivor accounting drift",
    )
    return g05_by_id, g06_by_id


def _terminal_g06_qualification(
    document: Mapping[str, Any] | None,
    *,
    expected_identity: str | None,
    expected_g06_envelope_identity: str,
    expected_g06_execution_identity: str,
    expected_input_rows_sha256: str,
    expected_record_count: int,
    expected_utf8_bytes: int,
) -> str | None:
    """Verify a distinct externally expected terminal G06 qualification receipt."""
    if document is None and expected_identity is None:
        return None
    _need(
        document is not None and expected_identity is not None,
        "terminal G06 qualification document and expected identity must be provided together",
    )
    qualification = _obj(document, "terminal G06 qualification")
    _need(
        set(qualification) == _G06_TERMINAL_QUALIFICATION_KEYS,
        "terminal G06 qualification schema is not closed",
    )
    _need(
        qualification.get("schema") == _G06_TERMINAL_QUALIFICATION_SCHEMA,
        "terminal G06 qualification schema drift",
    )
    _need(
        qualification.get("status") == _G06_TERMINAL_QUALIFICATION_STATUS,
        "terminal G06 qualification is not PASS",
    )
    identity = _identity(
        qualification,
        "qualification_identity_sha256",
        expected_identity,
    )
    _need(
        qualification.get("g06_envelope_identity_sha256")
        == _sha256(expected_g06_envelope_identity, "expected G06 envelope identity"),
        "terminal G06 qualification envelope lineage drift",
    )
    _need(
        qualification.get("g06_execution_identity_sha256")
        == _sha256(expected_g06_execution_identity, "expected G06 execution identity"),
        "terminal G06 qualification execution lineage drift",
    )
    _need(
        qualification.get("input_rows_sha256")
        == _sha256(expected_input_rows_sha256, "expected input rows root"),
        "terminal G06 qualification input lineage drift",
    )
    _need(
        qualification.get("repeated_execution_evidence_sha256")
        == qualification.get("g06_envelope_identity_sha256"),
        "terminal G06 qualification replay evidence drift",
    )
    _sha1(qualification.get("target_head_git_sha"), "terminal G06 target head")
    _sha1(
        qualification.get("real_replay_head_git_sha"),
        "terminal G06 real replay head",
    )
    for field in (
        "target_pr_number",
        "real_replay_run_id",
        "real_replay_job_id",
        "final_head_ci_run_id",
        "final_head_ci_job_id",
        "artifact_id",
        "independent_audit_issue_number",
    ):
        _int(qualification.get(field), f"terminal G06 qualification.{field}", positive=True)
    _sha256(
        qualification.get("artifact_zip_sha256"),
        "terminal G06 qualification artifact ZIP",
    )
    _need(
        _int(
            qualification.get("replay_record_count"),
            "terminal G06 qualification.replay_record_count",
            positive=True,
        )
        == _int(expected_record_count, "expected terminal G06 record count", positive=True),
        "terminal G06 qualification record-count drift",
    )
    _need(
        _int(
            qualification.get("replay_utf8_bytes"),
            "terminal G06 qualification.replay_utf8_bytes",
            positive=True,
        )
        == _int(expected_utf8_bytes, "expected terminal G06 UTF-8 bytes", positive=True),
        "terminal G06 qualification byte-count drift",
    )
    _need(
        _int(
            qualification.get("replay_count"),
            "terminal G06 qualification.replay_count",
            positive=True,
        )
        >= 2,
        "terminal G06 qualification requires two independent replays",
    )
    _need(
        qualification.get("independent_audit_status")
        == "PASS_FOR_INTEGRATION_RELEASED",
        "terminal G06 qualification independent audit is not released PASS",
    )
    _need(
        qualification.get("local_free_only") is True,
        "terminal G06 qualification is not LOCAL_FREE",
    )
    _need(
        qualification.get("head_change_invalidates") is True,
        "terminal G06 qualification does not fail closed on head change",
    )
    _need(
        qualification.get("truth_boundary") == _G06_TERMINAL_TRUTH,
        "terminal G06 qualification truth boundary drift",
    )
    return identity


def build_native_current_execution_preflight(
    *,
    g05_authority: Mapping[str, Any],
    g06_execution_envelope: Mapping[str, Any],
    expected_g05_execution_identity_sha256: str,
    expected_g06_envelope_identity_sha256: str,
    expected_g06_execution_identity_sha256: str,
    g06_terminal_qualification: Mapping[str, Any] | None,
    expected_g06_terminal_qualification_identity_sha256: str | None,
    expected_input_rows_sha256: str,
    expected_survivor_evidence_identity_sha256: str,
    expected_survivor_jsonl_sha256: str,
    expected_survivor_record_inventory_sha256: str,
    expected_survivor_payload_inventory_sha256: str,
    expected_survivor_record_count: int,
    expected_survivor_payload_bytes: int,
    expected_survivor_source_object_count: int,
    expected_privacy_policy_sha256: str,
    expected_privacy_implementation_git_blob_sha1: str,
) -> dict[str, Any]:
    g05_rows, g06_rows = _native_rows(
        g05_authority,
        g06_execution_envelope,
        expected_g05_identity=expected_g05_execution_identity_sha256,
        expected_g06_envelope_identity=expected_g06_envelope_identity_sha256,
        expected_g06_identity=expected_g06_execution_identity_sha256,
        expected_input_root=expected_input_rows_sha256,
        expected_survivor_evidence=expected_survivor_evidence_identity_sha256,
        expected_survivor_jsonl=expected_survivor_jsonl_sha256,
        expected_record_inventory=expected_survivor_record_inventory_sha256,
        expected_payload_inventory=expected_survivor_payload_inventory_sha256,
        expected_records=expected_survivor_record_count,
        expected_bytes=expected_survivor_payload_bytes,
        expected_source_objects=expected_survivor_source_object_count,
        expected_privacy_policy=expected_privacy_policy_sha256,
        expected_privacy_impl=expected_privacy_implementation_git_blob_sha1,
    )
    terminal_qualification_identity = _terminal_g06_qualification(
        g06_terminal_qualification,
        expected_identity=expected_g06_terminal_qualification_identity_sha256,
        expected_g06_envelope_identity=expected_g06_envelope_identity_sha256,
        expected_g06_execution_identity=expected_g06_execution_identity_sha256,
        expected_input_rows_sha256=expected_input_rows_sha256,
        expected_record_count=expected_survivor_record_count,
        expected_utf8_bytes=expected_survivor_payload_bytes,
    )
    _need(set(g05_rows) == set(g06_rows), "G05/G06 record-set drift")
    matrix_count: Counter[tuple[str, str]] = Counter()
    matrix_bytes: Counter[tuple[str, str]] = Counter()
    drop: set[str] = set()
    partial: set[str] = set()
    redact: set[str] = set()
    unchanged: set[str] = set()
    for record_id in sorted(g05_rows):
        g05 = g05_rows[record_id]
        g06 = g06_rows[record_id]
        binding = (
            g05.get("mode"),
            g05.get("payload_sha256"),
            g05.get("utf8_bytes"),
        )
        _need(
            binding
            == (
                g06.get("mode"),
                g06.get("payload_sha256"),
                g06.get("utf8_bytes"),
            ),
            f"G05/G06 payload binding drift: {record_id}",
        )
        status, action = g05["status"], g06["action"]
        matrix_count[(status, action)] += 1
        matrix_bytes[(status, action)] += g05["utf8_bytes"]
        if status == "REJECT_DOCUMENT" or action in {"QUARANTINE", "EXCLUDE"}:
            drop.add(record_id)
        elif status == "RETAIN_PARTIAL":
            partial.add(record_id)
        if action == "REDACT" and record_id not in drop:
            redact.add(record_id)
        if status == "RETAIN_ALL" and action == "ALLOW":
            unchanged.add(record_id)

    terminal = terminal_qualification_identity is not None
    blockers = []
    if not terminal:
        blockers.append("G06_EXACT_BYTE_EXECUTION_AUTHORITY_NOT_TERMINAL")
    if drop:
        blockers.append("POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED")
    if partial:
        blockers.append("G05_PARTIAL_MATERIALIZATION_REQUIRED")
    if redact:
        blockers.append("G06_REDACTION_MATERIALIZATION_REQUIRED")
    matrix = [
        {
            "g05_status": status,
            "g06_action": action,
            "record_count": matrix_count[(status, action)],
            "input_utf8_bytes": matrix_bytes[(status, action)],
        }
        for status, action in sorted(matrix_count)
    ]

    def record_hash(value: str) -> str:
        return _sha(value.encode("utf-8"))

    truth = {
        "current_corpus_eligible": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }
    core = {
        "schema": _PREFLIGHT_SCHEMA,
        "status": (
            "BLOCKED_CURRENT_G05_G06_COMPOSITION"
            if blockers
            else "READY_FOR_FINAL_COVERAGE_BINDING"
        ),
        "input_rows_sha256": expected_input_rows_sha256,
        "survivor_evidence_identity_sha256": (
            expected_survivor_evidence_identity_sha256
        ),
        "survivor_jsonl_sha256": expected_survivor_jsonl_sha256,
        "survivor_record_inventory_sha256": (
            expected_survivor_record_inventory_sha256
        ),
        "survivor_payload_inventory_sha256": (
            expected_survivor_payload_inventory_sha256
        ),
        "g05_execution_identity_sha256": expected_g05_execution_identity_sha256,
        "g06_envelope_identity_sha256": expected_g06_envelope_identity_sha256,
        "g06_execution_identity_sha256": expected_g06_execution_identity_sha256,
        "g06_terminal_qualification_identity_sha256": (
            terminal_qualification_identity
        ),
        "g06_exact_byte_execution_terminal": terminal,
        "input_record_count": expected_survivor_record_count,
        "input_utf8_bytes": expected_survivor_payload_bytes,
        "decision_matrix": matrix,
        "drop_record_id_sha256": sorted(record_hash(value) for value in drop),
        "g05_partial_record_id_sha256": sorted(
            record_hash(value) for value in partial
        ),
        "g06_redaction_record_id_sha256": sorted(
            record_hash(value) for value in redact
        ),
        "unchanged_allow_record_id_sha256": sorted(
            record_hash(value) for value in unchanged
        ),
        "drop_record_count": len(drop),
        "g05_partial_record_count": len(partial),
        "g06_redaction_record_count": len(redact),
        "unchanged_allow_record_count": len(unchanged),
        "unchanged_allow_input_utf8_bytes": sum(
            g05_rows[value]["utf8_bytes"] for value in unchanged
        ),
        "blockers": sorted(blockers),
        "truth_boundary": truth,
    }
    return {
        **core,
        "composition_preflight_identity_sha256": _sha(_cjson(core)),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-current-execution-preflight", action="store_true")
    parser.add_argument("--retained-inventory", type=Path)
    parser.add_argument("--expected-retained-inventory-identity")
    parser.add_argument("--decontamination-binding", type=Path)
    parser.add_argument("--expected-decontamination-authority")
    parser.add_argument("--expected-records-jsonl-sha256")
    parser.add_argument("--qualification-authority", action="append", type=Path)
    parser.add_argument("--expected-qualification-authority-identity", action="append")
    parser.add_argument("--expected-privacy-policy-identity")
    parser.add_argument("--expected-privacy-implementation-git-blob")
    parser.add_argument("--g05-authority", type=Path)
    parser.add_argument("--expected-g05-execution-identity")
    parser.add_argument("--g06-execution-envelope", type=Path)
    parser.add_argument("--expected-g06-envelope-identity")
    parser.add_argument("--expected-g06-execution-identity")
    parser.add_argument("--g06-terminal-qualification", type=Path)
    parser.add_argument("--expected-g06-terminal-qualification-identity")
    parser.add_argument("--expected-input-rows-sha256")
    parser.add_argument("--expected-survivor-evidence-identity")
    parser.add_argument("--expected-survivor-jsonl-sha256")
    parser.add_argument("--expected-survivor-record-inventory-sha256")
    parser.add_argument("--expected-survivor-payload-inventory-sha256")
    parser.add_argument("--expected-survivor-record-count", type=int)
    parser.add_argument("--expected-survivor-payload-bytes", type=int)
    parser.add_argument("--expected-survivor-source-object-count", type=int)
    parser.add_argument("--expected-privacy-policy-sha256")
    parser.add_argument("--expected-privacy-implementation-git-blob-sha1")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _required(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    names: tuple[str, ...],
) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        parser.error(
            "missing arguments for selected mode: " + ", ".join(missing)
        )


def _legacy(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    from twelve_six.data.final_g05_g06_coverage_v1 import (
        build_final_g05_g06_coverage,
    )

    names = (
        "retained_inventory",
        "expected_retained_inventory_identity",
        "decontamination_binding",
        "expected_decontamination_authority",
        "expected_records_jsonl_sha256",
        "qualification_authority",
        "expected_qualification_authority_identity",
        "expected_privacy_policy_identity",
        "expected_privacy_implementation_git_blob",
    )
    _required(parser, args, names)
    result = build_final_g05_g06_coverage(
        retained_inventory=_load(args.retained_inventory),
        expected_retained_inventory_identity_sha256=(
            args.expected_retained_inventory_identity
        ),
        decontamination_binding=_load(args.decontamination_binding),
        expected_decontamination_authority_sha256=(
            args.expected_decontamination_authority
        ),
        expected_records_jsonl_sha256=args.expected_records_jsonl_sha256,
        qualification_authorities=[
            _load(path) for path in args.qualification_authority
        ],
        expected_qualification_authority_identities_sha256=(
            args.expected_qualification_authority_identity
        ),
        expected_privacy_policy_identity_sha256=(
            args.expected_privacy_policy_identity
        ),
        expected_privacy_implementation_git_blob_sha=(
            args.expected_privacy_implementation_git_blob
        ),
    )
    return result, result["g05_g06_coverage_identity_sha256"]


def _native(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    names = (
        "g05_authority",
        "expected_g05_execution_identity",
        "g06_execution_envelope",
        "expected_g06_envelope_identity",
        "expected_g06_execution_identity",
        "expected_input_rows_sha256",
        "expected_survivor_evidence_identity",
        "expected_survivor_jsonl_sha256",
        "expected_survivor_record_inventory_sha256",
        "expected_survivor_payload_inventory_sha256",
        "expected_survivor_record_count",
        "expected_survivor_payload_bytes",
        "expected_survivor_source_object_count",
        "expected_privacy_policy_sha256",
        "expected_privacy_implementation_git_blob_sha1",
    )
    _required(parser, args, names)
    result = build_native_current_execution_preflight(
        g05_authority=_load(args.g05_authority),
        g06_execution_envelope=_load(args.g06_execution_envelope),
        expected_g05_execution_identity_sha256=(
            args.expected_g05_execution_identity
        ),
        expected_g06_envelope_identity_sha256=(
            args.expected_g06_envelope_identity
        ),
        expected_g06_execution_identity_sha256=(
            args.expected_g06_execution_identity
        ),
        g06_terminal_qualification=(
            _load(args.g06_terminal_qualification)
            if args.g06_terminal_qualification is not None
            else None
        ),
        expected_g06_terminal_qualification_identity_sha256=(
            args.expected_g06_terminal_qualification_identity
        ),
        expected_input_rows_sha256=args.expected_input_rows_sha256,
        expected_survivor_evidence_identity_sha256=(
            args.expected_survivor_evidence_identity
        ),
        expected_survivor_jsonl_sha256=args.expected_survivor_jsonl_sha256,
        expected_survivor_record_inventory_sha256=(
            args.expected_survivor_record_inventory_sha256
        ),
        expected_survivor_payload_inventory_sha256=(
            args.expected_survivor_payload_inventory_sha256
        ),
        expected_survivor_record_count=args.expected_survivor_record_count,
        expected_survivor_payload_bytes=args.expected_survivor_payload_bytes,
        expected_survivor_source_object_count=(
            args.expected_survivor_source_object_count
        ),
        expected_privacy_policy_sha256=args.expected_privacy_policy_sha256,
        expected_privacy_implementation_git_blob_sha1=(
            args.expected_privacy_implementation_git_blob_sha1
        ),
    )
    return result, result["composition_preflight_identity_sha256"]


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    result, identity = (
        _native(parser, args)
        if args.native_current_execution_preflight
        else _legacy(parser, args)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
