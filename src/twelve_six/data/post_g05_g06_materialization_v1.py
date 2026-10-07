"""Deterministic post-G05/G06 payload materialization for the current D03 survivor graph.

This module does not define quality/privacy policy. It consumes exact upstream
G05/G06 decisions, physically omits excluded records, applies REDACT spans from
the exact privacy implementation authenticated by G06, and emits a text-free
zero-credit materialization receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-post-g05-g06-materialization.v1"
PREFLIGHT_SCHEMA = "12-6.d03-current-g05-g06-composition-preflight.v1"
G05_SCHEMA = "12-6.g05-quality-execution-authority.v1"
G05_CLASS = "G05_QUALITY_EXECUTION_ZERO_CREDIT"
G06_ENVELOPE_SCHEMA = "12-6.current-survivor-g06-dependency-bound-execution.v1"
G06_SCHEMA = "12-6.g06-privacy-execution-authority.v1"
G06_CLASS = "G06_PRIVACY_EXECUTION_ZERO_CREDIT"
G06_TERMINAL_QUALIFICATION_SCHEMA = "12-6.g06-exact-byte-terminal-qualification.v1"
G06_TERMINAL_QUALIFICATION_STATUS = "PASS_FOR_G06_TERMINAL_CONSUMPTION"
G06_TERMINAL_QUALIFICATION_KEYS = {
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
G06_TERMINAL_TRUTH = {
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
PRIVACY_MODULE = "twelve_six.data.privacy_filter_v3"
PRIVACY_SOURCE_RELATIVE_PATH = "privacy_filter_v3.py"
PRIVACY_BINDING_KEYS = {
    "module",
    "relative_path",
    "implementation_git_blob_sha1",
    "policy_schema_version",
    "policy_sha256",
}
RECORD_KEYS = {"record_id", "source_id", "family", "modality", "normalized_payload"}
REDACTION_MARKER = "<redacted>"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class PostG05G06MaterializationError(ValueError):
    """Raised when an authority or payload invariant fails closed."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PostG05G06MaterializationError(message)


def _cjson(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _hash64(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{field} must be SHA-256",
    )
    return value


def _hash40(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX40.fullmatch(value) is not None,
        f"{field} must be Git SHA-1",
    )
    return value


def _int(value: Any, field: str, *, minimum: int = 0) -> int:
    _need(
        type(value) is int and value >= minimum,
        f"{field} must be an exact integer >= {minimum}",
    )
    return value


def _obj(value: Any, field: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _identity(document: Mapping[str, Any], field: str, expected: str) -> str:
    expected = _hash64(expected, f"expected {field}")
    claimed = _hash64(document.get(field), field)
    _need(claimed == expected, f"{field} is not independently expected")
    core = dict(document)
    del core[field]
    _need(claimed == _sha256(_cjson(core)), f"{field} self-hash mismatch")
    return claimed


def _zero_credit(value: Any, field: str) -> None:
    truth = _obj(value, field)
    false_fields = (
        "current_corpus_eligible",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
    )
    for key in false_fields:
        _need(truth.get(key) is False, f"{field}.{key} missing or widened")
    for key in ("training_authorized_bytes", "authorized_optimized_target_exposure"):
        _need(
            type(truth.get(key)) is int and truth.get(key) == 0,
            f"{field}.{key} missing or widened",
        )
    optimizer_fields = (
        "optimizer_updates_executed",
        "optimizer_updates_executed_on_real_targets",
    )
    present = [key for key in optimizer_fields if key in truth]
    _need(bool(present), f"{field} optimizer-update boundary missing")
    for key in present:
        _need(
            type(truth.get(key)) is int and truth.get(key) == 0,
            f"{field}.{key} widened",
        )


def canonical_record_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_cjson(record) for record in records)


def materialize_record_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        _need(set(record) == RECORD_KEYS, f"record[{index}] schema drift")
        for key in RECORD_KEYS:
            _need(
                isinstance(record.get(key), str) and bool(record[key]),
                f"record[{index}].{key} missing",
            )
        record_id = record["record_id"]
        _need(record_id not in seen, f"duplicate record_id: {record_id}")
        seen.add(record_id)
        payload_raw = record["normalized_payload"].encode("utf-8")
        rows.append(
            {
                "record_id": record_id,
                "source_id": record["source_id"],
                "family": record["family"],
                "modality": record["modality"],
                "payload_sha256": _sha256(payload_raw),
                "payload_bytes": len(payload_raw),
            }
        )
    rows.sort(key=lambda item: item["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": _sha256(_cjson(rows, newline=False)),
        "payload_inventory_digest_sha256": _sha256(
            _cjson(payload_projection, newline=False)
        ),
        "records": rows,
    }


def _validate_input_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    _need(bool(records), "input records missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        _need(isinstance(record, Mapping), f"record[{index}] must be an object")
        _need(set(record) == RECORD_KEYS, f"record[{index}] schema drift")
        for key in RECORD_KEYS:
            _need(
                isinstance(record.get(key), str) and bool(record[key]),
                f"record[{index}].{key} missing",
            )
        record_id = record["record_id"]
        _need(record_id not in by_id, f"duplicate input record_id: {record_id}")
        by_id[record_id] = record
    return by_id, materialize_record_inventory(records)


def _validate_preflight(
    value: Mapping[str, Any],
    *,
    expected_identity: str,
    expected_g05_identity: str,
    expected_g06_envelope_identity: str,
    expected_g06_identity: str,
    expected_g06_terminal_qualification_identity: str,
) -> str:
    _need(value.get("schema") == PREFLIGHT_SCHEMA, "composition preflight schema drift")
    _identity(value, "composition_preflight_identity_sha256", expected_identity)
    _need(
        value.get("status") == "BLOCKED_CURRENT_G05_G06_COMPOSITION",
        "unexpected preflight status",
    )
    _need(
        value.get("g06_exact_byte_execution_terminal") is True,
        "G06 exact-byte authority is not terminal",
    )
    _need(
        value.get("g05_execution_identity_sha256") == expected_g05_identity,
        "preflight G05 identity drift",
    )
    _need(
        value.get("g06_envelope_identity_sha256") == expected_g06_envelope_identity,
        "preflight G06 envelope drift",
    )
    _need(
        value.get("g06_execution_identity_sha256") == expected_g06_identity,
        "preflight G06 identity drift",
    )
    expected_qualification = _hash64(
        expected_g06_terminal_qualification_identity,
        "expected terminal G06 qualification identity",
    )
    _need(
        value.get("g06_terminal_qualification_identity_sha256")
        == expected_qualification,
        "preflight terminal G06 qualification identity drift",
    )
    input_rows_sha256 = _hash64(
        value.get("input_rows_sha256"),
        "preflight input rows root",
    )
    _need(
        _int(value.get("g05_partial_record_count"), "g05_partial_record_count") == 0,
        "G05 partial materialization has no byte-span authority",
    )
    blockers = value.get("blockers")
    _need(type(blockers) is list, "preflight blockers missing")
    _need(len(blockers) == len(set(blockers)), "duplicate preflight blocker")
    expected_blockers: set[str] = set()
    if _int(value.get("drop_record_count"), "drop_record_count") > 0:
        expected_blockers.add("POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED")
    if _int(value.get("g06_redaction_record_count"), "g06_redaction_record_count") > 0:
        expected_blockers.add("G06_REDACTION_MATERIALIZATION_REQUIRED")
    _need(bool(expected_blockers), "no post-G05/G06 materialization work remains")
    _need(
        set(blockers) == expected_blockers,
        "preflight contains an unconsumed upstream blocker",
    )
    _zero_credit(value.get("truth_boundary"), "preflight truth")
    return input_rows_sha256


def _terminal_g06_qualification(
    document: Mapping[str, Any] | None,
    *,
    expected_identity: str | None,
    expected_g06_envelope_identity: str,
    expected_g06_execution_identity: str,
    expected_input_rows_sha256: str,
    expected_record_count: int,
    expected_utf8_bytes: int,
) -> str:
    """Verify PR1041's distinct externally expected terminal G06 qualification."""
    _need(
        document is not None and expected_identity is not None,
        "terminal G06 qualification document and expected identity must be provided together",
    )
    qualification = _obj(document, "terminal G06 qualification")
    _need(
        set(qualification) == G06_TERMINAL_QUALIFICATION_KEYS,
        "terminal G06 qualification schema is not closed",
    )
    _need(
        qualification.get("schema") == G06_TERMINAL_QUALIFICATION_SCHEMA,
        "terminal G06 qualification schema drift",
    )
    _need(
        qualification.get("status") == G06_TERMINAL_QUALIFICATION_STATUS,
        "terminal G06 qualification is not PASS",
    )
    identity = _identity(
        qualification,
        "qualification_identity_sha256",
        expected_identity,
    )
    _need(
        qualification.get("g06_envelope_identity_sha256")
        == _hash64(expected_g06_envelope_identity, "expected G06 envelope identity"),
        "terminal G06 qualification envelope lineage drift",
    )
    _need(
        qualification.get("g06_execution_identity_sha256")
        == _hash64(expected_g06_execution_identity, "expected G06 execution identity"),
        "terminal G06 qualification execution lineage drift",
    )
    _need(
        qualification.get("input_rows_sha256")
        == _hash64(expected_input_rows_sha256, "expected input rows root"),
        "terminal G06 qualification input lineage drift",
    )
    _need(
        qualification.get("repeated_execution_evidence_sha256")
        == qualification.get("g06_envelope_identity_sha256"),
        "terminal G06 qualification replay evidence drift",
    )
    _hash40(qualification.get("target_head_git_sha"), "terminal G06 target head")
    _hash40(
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
        _int(
            qualification.get(field),
            f"terminal G06 qualification.{field}",
            minimum=1,
        )
    _hash64(
        qualification.get("artifact_zip_sha256"),
        "terminal G06 qualification artifact ZIP",
    )
    _need(
        _int(
            qualification.get("replay_record_count"),
            "terminal G06 qualification.replay_record_count",
            minimum=1,
        )
        == _int(expected_record_count, "expected terminal G06 record count", minimum=1),
        "terminal G06 qualification record-count drift",
    )
    _need(
        _int(
            qualification.get("replay_utf8_bytes"),
            "terminal G06 qualification.replay_utf8_bytes",
            minimum=1,
        )
        == _int(expected_utf8_bytes, "expected terminal G06 UTF-8 bytes", minimum=1),
        "terminal G06 qualification byte-count drift",
    )
    _need(
        _int(
            qualification.get("replay_count"),
            "terminal G06 qualification.replay_count",
            minimum=1,
        )
        >= 2,
        "terminal G06 qualification requires two independent replays",
    )
    _need(
        qualification.get("independent_audit_status") == "PASS_FOR_INTEGRATION_RELEASED",
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
        qualification.get("truth_boundary") == G06_TERMINAL_TRUTH,
        "terminal G06 qualification truth boundary drift",
    )
    return identity


def _validate_privacy_binding(value: Any) -> dict[str, str]:
    binding = _obj(value, "G06 privacy_binding")
    _need(set(binding) == PRIVACY_BINDING_KEYS, "G06 privacy_binding schema drift")
    module = binding.get("module")
    relative_path = binding.get("relative_path")
    policy_schema = binding.get("policy_schema_version")
    _need(module == PRIVACY_MODULE, "G06 privacy_binding module drift")
    _need(
        relative_path == PRIVACY_SOURCE_RELATIVE_PATH,
        "G06 privacy_binding relative path drift",
    )
    _need(
        isinstance(policy_schema, str) and bool(policy_schema),
        "G06 privacy_binding policy schema malformed",
    )
    return {
        "module": module,
        "relative_path": relative_path,
        "implementation_git_blob_sha1": _hash40(
            binding.get("implementation_git_blob_sha1"),
            "G06 privacy_binding implementation blob",
        ),
        "policy_schema_version": policy_schema,
        "policy_sha256": _hash64(
            binding.get("policy_sha256"),
            "G06 privacy_binding policy identity",
        ),
    }


def _rows_from_authorities(
    g05: Mapping[str, Any],
    g06_envelope: Mapping[str, Any],
    *,
    expected_g05_identity: str,
    expected_g06_envelope_identity: str,
    expected_g06_identity: str,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, str],
]:
    _need(g05.get("schema_version") == G05_SCHEMA, "G05 schema drift")
    _need(g05.get("authority_class") == G05_CLASS, "G05 authority class drift")
    _identity(g05, "execution_identity_sha256", expected_g05_identity)
    _zero_credit(g05.get("truth_boundary"), "G05 truth")
    g05_rows = g05.get("records")
    _need(type(g05_rows) is list and bool(g05_rows), "G05 records missing")
    _need(
        g05.get("execution_rows_sha256") == _sha256(_cjson(g05_rows)),
        "G05 rows root drift",
    )

    _need(
        g06_envelope.get("schema_version") == G06_ENVELOPE_SCHEMA,
        "G06 envelope schema drift",
    )
    _need(
        g06_envelope.get("execution_profile") == "LOCAL_FREE",
        "G06 execution profile drift",
    )
    _identity(
        g06_envelope,
        "evidence_identity_sha256",
        expected_g06_envelope_identity,
    )
    _zero_credit(g06_envelope.get("truth_boundary"), "G06 envelope truth")
    g06 = _obj(g06_envelope.get("privacy_execution_authority"), "G06 authority")
    _need(g06.get("schema_version") == G06_SCHEMA, "G06 schema drift")
    _need(g06.get("authority_class") == G06_CLASS, "G06 authority class drift")
    _identity(g06, "execution_identity_sha256", expected_g06_identity)
    _zero_credit(g06.get("truth_boundary"), "G06 truth")
    privacy_binding = _validate_privacy_binding(g06.get("privacy_binding"))
    g06_rows = g06.get("records")
    _need(type(g06_rows) is list and bool(g06_rows), "G06 records missing")
    _need(
        g06.get("execution_rows_sha256") == _sha256(_cjson(g06_rows)),
        "G06 rows root drift",
    )

    def index(rows: list[Any], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for number, raw in enumerate(rows):
            row = _obj(raw, f"{label}[{number}]")
            record_id = row.get("record_id")
            _need(
                isinstance(record_id, str) and bool(record_id),
                f"{label} record_id missing",
            )
            _need(record_id not in result, f"duplicate {label} record_id: {record_id}")
            result[record_id] = row
        _need(list(result) == sorted(result), f"{label} rows are not sorted")
        return result

    g05_by_id = index(g05_rows, "G05")
    g06_by_id = index(g06_rows, "G06")
    _need(set(g05_by_id) == set(g06_by_id), "G05/G06 record-set drift")
    return g05_by_id, g06_by_id, privacy_binding


def _materializer_implementation_blob(expected_git_blob_sha1: str) -> str:
    expected_blob = _hash40(
        expected_git_blob_sha1,
        "expected materializer implementation Git blob",
    )
    try:
        source = Path(__file__).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise PostG05G06MaterializationError(
            "cannot resolve running materializer implementation"
        ) from exc
    actual_blob = _git_blob_sha1(source)
    _need(
        actual_blob == expected_blob,
        "materializer implementation Git blob drift",
    )
    return actual_blob


def _privacy_runtime(
    path: Path,
    *,
    privacy_binding: Mapping[str, str],
) -> tuple[Callable[[str], Sequence[Any]], Callable[[bytes | str], Any], str]:
    _need(path.name == privacy_binding["relative_path"], "privacy source path drift")
    source = path.read_bytes()
    expected_blob = privacy_binding["implementation_git_blob_sha1"]
    actual_blob = _git_blob_sha1(source)
    _need(actual_blob == expected_blob, "privacy source Git blob drift")
    private_name = f"_twelve_six_privacy_{actual_blob}"
    module = types.ModuleType(private_name)
    module.__file__ = str(path)
    previous = sys.modules.get(private_name)
    _need(previous is None, "private privacy namespace collision")
    sys.modules[private_name] = module
    try:
        exec(  # noqa: S102 - exact G06-authenticated Git-blob-verified source bytes
            compile(source, str(path), "exec"), module.__dict__
        )
    finally:
        sys.modules.pop(private_name, None)
    detect = module.__dict__.get("detect")
    scan = module.__dict__.get("hash_safe_scan")
    manifest = module.__dict__.get("policy_manifest")
    _need(
        callable(detect) and callable(scan) and callable(manifest),
        "privacy source API drift",
    )
    policy = manifest()
    _need(isinstance(policy, Mapping), "privacy policy manifest missing")
    _need(
        policy.get("policy_sha256") == privacy_binding["policy_sha256"],
        "privacy policy identity drift",
    )
    return detect, scan, actual_blob


def _redact_text(text: str, findings: Sequence[Any]) -> str:
    _need(bool(findings), "REDACT action produced no findings")
    spans: list[tuple[int, int]] = []
    for index, finding in enumerate(findings):
        action = getattr(finding, "action", None)
        start = getattr(finding, "start", None)
        end = getattr(finding, "end", None)
        _need(
            action == "REDACT",
            f"REDACT row has non-REDACT finding[{index}]",
        )
        _need(
            type(start) is int and type(end) is int,
            f"finding[{index}] span must be exact integers",
        )
        _need(0 <= start < end <= len(text), f"finding[{index}] span out of range")
        spans.append((start, end))
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append(REDACTION_MARKER)
        cursor = end
    out.append(text[cursor:])
    result = "".join(out)
    _need(result != text, "REDACT materialization did not change payload")
    return result


def _record_hash(record_id: str) -> str:
    return _sha256(record_id.encode("utf-8"))


def materialize_post_g05_g06(
    *,
    records: Sequence[Mapping[str, Any]],
    composition_preflight: Mapping[str, Any],
    expected_composition_preflight_identity_sha256: str,
    g05_authority: Mapping[str, Any],
    expected_g05_execution_identity_sha256: str,
    g06_execution_envelope: Mapping[str, Any],
    expected_g06_envelope_identity_sha256: str,
    expected_g06_execution_identity_sha256: str,
    g06_terminal_qualification: Mapping[str, Any] | None,
    expected_g06_terminal_qualification_identity_sha256: str | None,
    privacy_source_path: Path,
    execution_head_sha: str,
    expected_materializer_implementation_git_blob_sha1: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Materialize exact post-G05/G06 records and a text-free zero-credit receipt."""
    execution_head_sha = _hash40(execution_head_sha, "execution head SHA")
    materializer_blob = _materializer_implementation_blob(
        expected_materializer_implementation_git_blob_sha1
    )
    expected_g05 = _hash64(
        expected_g05_execution_identity_sha256,
        "expected G05 identity",
    )
    expected_g06_envelope = _hash64(
        expected_g06_envelope_identity_sha256,
        "expected G06 envelope identity",
    )
    expected_g06 = _hash64(
        expected_g06_execution_identity_sha256,
        "expected G06 identity",
    )
    expected_qualification = _hash64(
        expected_g06_terminal_qualification_identity_sha256,
        "expected terminal G06 qualification identity",
    )
    input_rows_sha256 = _validate_preflight(
        composition_preflight,
        expected_identity=expected_composition_preflight_identity_sha256,
        expected_g05_identity=expected_g05,
        expected_g06_envelope_identity=expected_g06_envelope,
        expected_g06_identity=expected_g06,
        expected_g06_terminal_qualification_identity=expected_qualification,
    )
    qualification_identity = _terminal_g06_qualification(
        g06_terminal_qualification,
        expected_identity=expected_qualification,
        expected_g06_envelope_identity=expected_g06_envelope,
        expected_g06_execution_identity=expected_g06,
        expected_input_rows_sha256=input_rows_sha256,
        expected_record_count=_int(
            composition_preflight.get("input_record_count"),
            "preflight input record count",
            minimum=1,
        ),
        expected_utf8_bytes=_int(
            composition_preflight.get("input_utf8_bytes"),
            "preflight input UTF-8 bytes",
            minimum=1,
        ),
    )
    _need(
        qualification_identity
        == composition_preflight.get("g06_terminal_qualification_identity_sha256"),
        "terminal G06 qualification/preflight lineage drift",
    )

    by_id, input_inventory = _validate_input_records(records)
    g05_rows, g06_rows, privacy_binding = _rows_from_authorities(
        g05_authority,
        g06_execution_envelope,
        expected_g05_identity=expected_g05,
        expected_g06_envelope_identity=expected_g06_envelope,
        expected_g06_identity=expected_g06,
    )
    _need(set(by_id) == set(g05_rows), "payload/G05 record-set drift")
    _need(
        input_inventory["record_count"]
        == composition_preflight.get("input_record_count"),
        "preflight record count drift",
    )
    _need(
        input_inventory["total_payload_bytes"]
        == composition_preflight.get("input_utf8_bytes"),
        "preflight payload byte drift",
    )
    input_raw = canonical_record_bytes(records)
    _need(
        composition_preflight.get("survivor_jsonl_sha256") == _sha256(input_raw),
        "preflight survivor JSONL identity drift",
    )
    _need(
        composition_preflight.get("survivor_record_inventory_sha256")
        == input_inventory["record_inventory_digest_sha256"],
        "preflight survivor record inventory drift",
    )
    _need(
        composition_preflight.get("survivor_payload_inventory_sha256")
        == input_inventory["payload_inventory_digest_sha256"],
        "preflight survivor payload inventory drift",
    )

    detect, scan, privacy_blob = _privacy_runtime(
        privacy_source_path,
        privacy_binding=privacy_binding,
    )
    output: list[dict[str, Any]] = []
    drop_ids: list[str] = []
    redact_ids: list[str] = []
    unchanged_ids: list[str] = []
    dropped_bytes = redacted_input_bytes = unchanged_bytes = 0

    for record in records:
        record_id = record["record_id"]
        g05 = g05_rows[record_id]
        g06 = g06_rows[record_id]
        payload = record["normalized_payload"]
        payload_raw = payload.encode("utf-8")
        payload_sha = _sha256(payload_raw)
        payload_bytes = len(payload_raw)
        for label, row in (("G05", g05), ("G06", g06)):
            _need(
                row.get("payload_sha256") == payload_sha,
                f"{label} payload hash drift: {record_id}",
            )
            _need(
                _int(
                    row.get("utf8_bytes"),
                    f"{label} bytes {record_id}",
                    minimum=1,
                )
                == payload_bytes,
                f"{label} payload byte drift: {record_id}",
            )
        _need(
            (g05.get("mode"), g05.get("payload_sha256"), g05.get("utf8_bytes"))
            == (g06.get("mode"), g06.get("payload_sha256"), g06.get("utf8_bytes")),
            f"G05/G06 payload binding drift: {record_id}",
        )
        status = g05.get("status")
        action = g06.get("action")
        _need(
            status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"},
            f"G05 status drift: {record_id}",
        )
        _need(
            action in {"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"},
            f"G06 action drift: {record_id}",
        )
        _need(
            status != "RETAIN_PARTIAL",
            f"G05 partial row lacks byte-span authority: {record_id}",
        )
        if status == "REJECT_DOCUMENT" or action in {"QUARANTINE", "EXCLUDE"}:
            drop_ids.append(record_id)
            dropped_bytes += payload_bytes
            continue
        if action == "REDACT":
            transformed = _redact_text(payload, detect(payload))
            rescanned = scan(transformed.encode("utf-8"))
            _need(
                getattr(rescanned, "action", None) == "ALLOW",
                f"redacted output is not privacy-clean: {record_id}",
            )
            changed = dict(record)
            changed["normalized_payload"] = transformed
            output.append(changed)
            redact_ids.append(record_id)
            redacted_input_bytes += payload_bytes
            continue
        _need(
            status == "RETAIN_ALL" and action == "ALLOW",
            f"unhandled decision pair: {record_id}",
        )
        output.append(dict(record))
        unchanged_ids.append(record_id)
        unchanged_bytes += payload_bytes

    _need(
        len(output) == len(records) - len(drop_ids),
        "record-count conservation failure",
    )
    _need(
        len(drop_ids) + len(redact_ids) + len(unchanged_ids) == len(records),
        "decision partition failure",
    )
    output_inventory = materialize_record_inventory(output)
    output_raw = canonical_record_bytes(output)
    expected_drop = composition_preflight.get("drop_record_count")
    expected_redact = composition_preflight.get("g06_redaction_record_count")
    expected_unchanged = composition_preflight.get("unchanged_allow_record_count")
    _need(len(drop_ids) == expected_drop, "drop count disagrees with preflight")
    _need(
        len(redact_ids) == expected_redact,
        "redaction count disagrees with preflight",
    )
    _need(
        len(unchanged_ids) == expected_unchanged,
        "unchanged count disagrees with preflight",
    )
    _need(
        sorted(_record_hash(value) for value in drop_ids)
        == composition_preflight.get("drop_record_id_sha256"),
        "drop set disagrees with preflight",
    )
    _need(
        sorted(_record_hash(value) for value in redact_ids)
        == composition_preflight.get("g06_redaction_record_id_sha256"),
        "redaction set disagrees with preflight",
    )
    _need(
        sorted(_record_hash(value) for value in unchanged_ids)
        == composition_preflight.get("unchanged_allow_record_id_sha256"),
        "unchanged set disagrees with preflight",
    )
    _need(
        unchanged_bytes
        == composition_preflight.get("unchanged_allow_input_utf8_bytes"),
        "unchanged byte total drift",
    )

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
        "schema_version": SCHEMA,
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head_sha,
        "materializer_implementation_git_blob_sha1": materializer_blob,
        "input": {
            "composition_preflight_identity_sha256": (
                expected_composition_preflight_identity_sha256
            ),
            "g05_execution_identity_sha256": expected_g05,
            "g06_envelope_identity_sha256": expected_g06_envelope,
            "g06_execution_identity_sha256": expected_g06,
            "g06_terminal_qualification_identity_sha256": qualification_identity,
            "input_rows_sha256": input_rows_sha256,
            "privacy_binding": dict(privacy_binding),
            "privacy_implementation_git_blob_sha1": privacy_blob,
            "record_payload_jsonl_sha256": _sha256(input_raw),
            "record_inventory_digest_sha256": input_inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": input_inventory[
                "payload_inventory_digest_sha256"
            ],
            "record_count": input_inventory["record_count"],
            "source_object_count": len({record["source_id"] for record in records}),
            "total_payload_bytes": input_inventory["total_payload_bytes"],
        },
        "transform": {
            "redaction_marker_sha256": _sha256(REDACTION_MARKER.encode("utf-8")),
            "drop_record_count": len(drop_ids),
            "drop_input_payload_bytes": dropped_bytes,
            "redact_record_count": len(redact_ids),
            "redact_input_payload_bytes": redacted_input_bytes,
            "unchanged_record_count": len(unchanged_ids),
            "unchanged_input_payload_bytes": unchanged_bytes,
            "drop_record_id_sha256": sorted(_record_hash(value) for value in drop_ids),
            "redact_record_id_sha256": sorted(
                _record_hash(value) for value in redact_ids
            ),
            "unchanged_record_id_sha256": sorted(
                _record_hash(value) for value in unchanged_ids
            ),
        },
        "result": {
            "record_payload_jsonl_sha256": _sha256(output_raw),
            "record_inventory_digest_sha256": output_inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": output_inventory[
                "payload_inventory_digest_sha256"
            ],
            "record_count": output_inventory["record_count"],
            "source_object_count": len({record["source_id"] for record in output}),
            "total_payload_bytes": output_inventory["total_payload_bytes"],
        },
        "consumed_blockers": sorted(composition_preflight.get("blockers", [])),
        "remaining_materialization_blockers": [],
        "truth_boundary": truth,
    }
    evidence = {
        **core,
        "materialization_identity_sha256": _sha256(_cjson(core)),
    }
    return output, output_inventory, evidence
