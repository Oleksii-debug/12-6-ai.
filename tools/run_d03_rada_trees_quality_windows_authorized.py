#!/usr/bin/env python3
"""Authorize the merged #958 Rada_Trees quality-window materializer from #916 evidence.

This is a thin fail-closed authority wrapper around the incumbent materializer.  It
adds no quality or privacy rule: it binds the candidate to the independently
verified terminal #916 handoff report, rejects any candidate-schema widening, then
delegates the actual quality-window/privacy work to the canonical #958 tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-trees-quality-window-authorized-run.v1"
UPSTREAM_SCHEMA = "12-6.d03-rada-trees-rights-scoped-handoff-report.v1"
EXPECTED_UPSTREAM_REPORT_SHA256 = (
    "83b2cc636aa4cf9a754cd55534035c44782722d2b457755c00f6780112ebfffa"
)
EXPECTED_FAMILY = "ua.rada.open-data.plenary-transcripts"
EXPECTED_DATASET = "uacorpus/Rada_Trees"
EXPECTED_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_ARCHIVE = "rada_xtag_texts.7z"
EXPECTED_ARCHIVE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
EXPECTED_ARCHIVE_BYTES = 697_768_591
EXPECTED_RIGHTS_STATUS = "RIGHTS_SCOPE_SUPPORTED_DATED_PARLIAMENT_TRANSCRIPT_CANDIDATE"
EXPECTED_RIGHTS_CONFIG_SHA256 = "34da44a047c5e0d562ee6a86987cb66e3ed266e1c1f09af95e63c38c219fe1a3"
EXPECTED_RIGHTS_REPORT_SHA256 = "7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d"
EXPECTED_FULL_SCAN_EVIDENCE_ID = "b25ff7e7948e2fa215ee8d059ea5594025693e105389b7095f5250808ecc3a9d"
EXPECTED_ACCEPTED_INVENTORY = "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
EXPECTED_HELD_INVENTORY = "566760e10157cd835ed0879abb37f052b57d31cff6af358a81ff717f4f7f59d9"
EXPECTED_CANDIDATE_RECORDS = 4_384
EXPECTED_ACCEPTED_SOURCE_PAYLOAD_BYTES = 877_899_128
EXPECTED_SAFE_RESULT = "RADA_TREES_RIGHTS_SCOPED_HANDOFF_ZERO_CREDIT"

EXPECTED_CANDIDATE_KEYS = frozenset(
    {
        "record_id",
        "source_family",
        "source_dataset",
        "source_revision",
        "source_archive",
        "source_path",
        "session_date",
        "source_payload_sha256",
        "source_payload_bytes",
        "decoded_encoding",
        "decoded_text_utf8_sha256",
        "decoded_text_utf8_bytes",
        "rights_scope_status",
        "attribution_required",
        "language_quality_privacy_complete",
        "global_dedup_complete",
        "reserved_evaluation_decontamination_complete",
        "training_eligible",
        "evaluation_eligible",
        "text",
    }
)

_TOOL = Path(__file__).with_name("materialize_d03_rada_trees_quality_windows.py")
_SPEC = importlib.util.spec_from_file_location("d03_rada_quality_windows_incumbent", _TOOL)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - repository corruption
    raise RuntimeError("cannot load incumbent Rada quality-window materializer")
_delegate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_delegate)


class AuthorizationError(RuntimeError):
    """Fail-closed Rada_Trees authority error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.casefold())
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationError(f"cannot load {label}") from exc
    require(isinstance(value, dict), f"{label} root must be an object")
    require(raw == canonical_bytes(value), f"{label} serialization is not canonical")
    return value


def _validate_upstream_handoff(path: Path) -> dict[str, Any]:
    report = _load_canonical_json(path, "upstream #916 handoff report")
    require(report.get("schema_version") == UPSTREAM_SCHEMA, "upstream handoff schema drift")
    require(report.get("execution_profile") == "LOCAL_FREE", "upstream execution profile drift")

    claimed = report.get("report_sha256")
    require(_is_sha256(claimed), "upstream handoff report identity missing")
    require(
        str(claimed).casefold() == EXPECTED_UPSTREAM_REPORT_SHA256,
        "upstream handoff report is not the independently verified #916 authority",
    )
    core = dict(report)
    core.pop("report_sha256", None)
    require(
        sha256_bytes(canonical_bytes(core)) == str(claimed).casefold(),
        "upstream handoff report self-hash invalid",
    )

    source = report.get("source")
    require(isinstance(source, Mapping), "upstream source authority missing")
    expected_source = {
        "dataset": EXPECTED_DATASET,
        "revision": EXPECTED_REVISION,
        "archive": EXPECTED_ARCHIVE,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "family": EXPECTED_FAMILY,
    }
    for key, expected in expected_source.items():
        require(source.get(key) == expected, f"upstream source authority drift: {key}")

    parent = report.get("parent_authority")
    require(isinstance(parent, Mapping), "upstream parent authority missing")
    expected_parent = {
        "full_scan_evidence_identity_sha256": EXPECTED_FULL_SCAN_EVIDENCE_ID,
        "rights_config_sha256": EXPECTED_RIGHTS_CONFIG_SHA256,
        "rights_report_sha256": EXPECTED_RIGHTS_REPORT_SHA256,
        "accepted_path_inventory_sha256": EXPECTED_ACCEPTED_INVENTORY,
        "held_path_inventory_sha256": EXPECTED_HELD_INVENTORY,
    }
    for key, expected in expected_parent.items():
        require(parent.get(key) == expected, f"upstream parent authority drift: {key}")

    materialization = report.get("materialization")
    require(isinstance(materialization, Mapping), "upstream materialization authority missing")
    candidate_sha = materialization.get("candidate_jsonl_sha256")
    require(_is_sha256(candidate_sha), "upstream candidate identity missing")
    require(
        materialization.get("candidate_records") == EXPECTED_CANDIDATE_RECORDS,
        "upstream candidate record count drift",
    )
    require(
        materialization.get("accepted_source_payload_bytes")
        == EXPECTED_ACCEPTED_SOURCE_PAYLOAD_BYTES,
        "upstream accepted source byte total drift",
    )
    candidate_file_bytes = materialization.get("candidate_jsonl_bytes")
    decoded_text_bytes = materialization.get("decoded_text_utf8_bytes")
    require(type(candidate_file_bytes) is int and candidate_file_bytes > 0, "candidate file byte authority missing")
    require(type(decoded_text_bytes) is int and decoded_text_bytes > 0, "decoded text byte authority missing")
    require(
        materialization.get("source_native_document_boundaries_preserved") is True,
        "source-native boundary authority drift",
    )
    require(
        materialization.get("record_text_persisted_in_report") is False,
        "upstream report payload-retention boundary drift",
    )

    boundary = report.get("claim_boundary")
    require(isinstance(boundary, Mapping), "upstream claim boundary missing")
    for key in ("training_authorized_bytes", "unique_causal_loss_positions_authorized", "optimizer_updates"):
        require(boundary.get(key) == 0, f"upstream zero-credit boundary drift: {key}")
    for key in (
        "candidate_jsonl_is_canonical_corpus",
        "language_quality_privacy_complete",
        "global_dedup_complete",
        "reserved_evaluation_decontamination_complete",
        "family_caps_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
    ):
        require(boundary.get(key) is False, f"upstream fail-closed boundary drift: {key}")
    require(report.get("safe_result") == EXPECTED_SAFE_RESULT, "upstream safe-result drift")

    return {
        "report_sha256": str(claimed).casefold(),
        "candidate_jsonl_sha256": str(candidate_sha).casefold(),
        "candidate_jsonl_bytes": candidate_file_bytes,
        "candidate_records": EXPECTED_CANDIDATE_RECORDS,
        "accepted_source_payload_bytes": EXPECTED_ACCEPTED_SOURCE_PAYLOAD_BYTES,
        "decoded_text_utf8_bytes": decoded_text_bytes,
    }


def _validate_candidate(path: Path, authority: Mapping[str, Any]) -> None:
    require(path.is_file() and not path.is_symlink(), "candidate must be a regular file")
    actual_sha, actual_bytes = _sha256_file(path)
    require(actual_sha == authority["candidate_jsonl_sha256"], "candidate SHA does not match #916 authority")
    require(actual_bytes == authority["candidate_jsonl_bytes"], "candidate file bytes do not match #916 authority")

    seen: set[str] = set()
    records = source_bytes = decoded_bytes = 0
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, 1):
            require(line.strip(), f"blank candidate JSONL line at {line_no}")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuthorizationError(f"invalid candidate JSONL at line {line_no}") from exc
            require(isinstance(row, dict), f"candidate row {line_no} must be an object")
            unexpected = set(row) - EXPECTED_CANDIDATE_KEYS
            missing = EXPECTED_CANDIDATE_KEYS - set(row)
            require(not unexpected, f"candidate schema has unknown keys at line {line_no}: {sorted(unexpected)}")
            require(not missing, f"candidate schema missing keys at line {line_no}: {sorted(missing)}")

            record_id = row["record_id"]
            source_path = row["source_path"]
            source_sha = row["source_payload_sha256"]
            source_nbytes = row["source_payload_bytes"]
            text = row["text"]
            require(isinstance(source_path, str) and source_path, f"source path missing at line {line_no}")
            require(_is_sha256(source_sha), f"source payload SHA malformed at line {line_no}")
            require(type(source_nbytes) is int and source_nbytes > 0, f"source payload bytes malformed at line {line_no}")
            require(
                record_id == f"rada-trees:{source_sha}:{source_path}",
                f"record/source identity drift at line {line_no}",
            )
            require(record_id not in seen, f"duplicate record_id: {record_id}")
            seen.add(record_id)
            require(row["source_family"] == EXPECTED_FAMILY, f"source family drift at line {line_no}")
            require(row["source_dataset"] == EXPECTED_DATASET, f"source dataset drift at line {line_no}")
            require(row["source_revision"] == EXPECTED_REVISION, f"source revision drift at line {line_no}")
            require(row["source_archive"] == EXPECTED_ARCHIVE, f"source archive drift at line {line_no}")
            require(row["rights_scope_status"] == EXPECTED_RIGHTS_STATUS, f"rights scope drift at line {line_no}")
            require(row["attribution_required"] is True, f"attribution boundary drift at line {line_no}")
            require(isinstance(row["decoded_encoding"], str) and row["decoded_encoding"], f"decoded encoding missing at line {line_no}")
            try:
                date.fromisoformat(str(row["session_date"]))
            except ValueError as exc:
                raise AuthorizationError(f"session date malformed at line {line_no}") from exc
            require(isinstance(text, str), f"text missing at line {line_no}")
            text_raw = text.encode("utf-8")
            require(row["decoded_text_utf8_sha256"] == sha256_bytes(text_raw), f"decoded text SHA drift at line {line_no}")
            require(row["decoded_text_utf8_bytes"] == len(text_raw), f"decoded text bytes drift at line {line_no}")
            for key in (
                "language_quality_privacy_complete",
                "global_dedup_complete",
                "reserved_evaluation_decontamination_complete",
                "training_eligible",
                "evaluation_eligible",
            ):
                require(row[key] is False, f"premature candidate authority at line {line_no}: {key}")
            records += 1
            source_bytes += source_nbytes
            decoded_bytes += len(text_raw)

    require(records == authority["candidate_records"], "candidate record count does not match #916 authority")
    require(source_bytes == authority["accepted_source_payload_bytes"], "candidate source bytes do not match #916 authority")
    require(decoded_bytes == authority["decoded_text_utf8_bytes"], "candidate decoded bytes do not match #916 authority")


def _validate_paths(
    candidate: Path,
    upstream: Path,
    output: Path,
    delegate_report: Path,
    authority_report: Path,
) -> None:
    resolved = {path.resolve(strict=False) for path in (candidate, upstream, output, delegate_report, authority_report)}
    require(len(resolved) == 5, "all input/output/report paths must be distinct")
    require(not authority_report.is_symlink(), "authority report path must not be a symlink")
    require(not authority_report.exists(), "authority report already exists")


def materialize(
    candidate: Path,
    upstream_handoff_report: Path,
    output: Path,
    materialization_report: Path,
    authority_report: Path,
) -> dict[str, Any]:
    _validate_paths(candidate, upstream_handoff_report, output, materialization_report, authority_report)
    authority = _validate_upstream_handoff(upstream_handoff_report)
    _validate_candidate(candidate, authority)

    delegated = _delegate.materialize(
        candidate,
        output,
        materialization_report,
        expected_candidate_sha256=authority["candidate_jsonl_sha256"],
    )
    require(isinstance(delegated, Mapping), "incumbent materializer returned malformed report")
    require(
        delegated.get("input_candidate_jsonl_sha256") == authority["candidate_jsonl_sha256"],
        "delegated materializer candidate binding drift",
    )
    delegate_report_sha = delegated.get("report_sha256")
    require(_is_sha256(delegate_report_sha), "delegated materialization report identity missing")
    privacy = delegated.get("privacy")
    boundary = delegated.get("claim_boundary")
    require(isinstance(privacy, Mapping), "delegated privacy evidence missing")
    require(isinstance(boundary, Mapping), "delegated claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "delegated training credit drift")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "delegated loss-position credit drift")
    require(boundary.get("optimizer_updates") == 0, "delegated optimizer boundary drift")
    require(boundary.get("model_training_executed") is False, "delegated training boundary drift")
    require(boundary.get("paid_compute_used") is False, "delegated paid-compute boundary drift")

    core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "upstream_handoff_report_sha256": authority["report_sha256"],
        "input_candidate_jsonl_sha256": authority["candidate_jsonl_sha256"],
        "input_candidate_records": authority["candidate_records"],
        "input_accepted_source_payload_bytes": authority["accepted_source_payload_bytes"],
        "delegated_materialization_report_sha256": str(delegate_report_sha).casefold(),
        "output_jsonl_sha256": delegated.get("output_jsonl_sha256"),
        "output_records": delegated.get("output_records"),
        "quality_policy_sha256": delegated.get("quality", {}).get("policy_sha256"),
        "privacy_policy_sha256": privacy.get("policy_sha256"),
        "authority_repairs": {
            "candidate_identity_bound_to_terminal_handoff": True,
            "immutable_source_and_rights_parent_bound": True,
            "candidate_schema_closed_before_privacy": True,
            "unknown_parent_fields_can_reach_survivors": False,
            "quality_or_privacy_algorithm_reimplemented": False,
        },
        "claim_boundary": {
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "safe_result": "RADA_TREES_QUALITY_WINDOWS_AUTHORITY_BOUND_ZERO_CREDIT",
    }
    report = {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}
    authority_report.parent.mkdir(parents=True, exist_ok=True)
    authority_report.write_bytes(canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--upstream-handoff-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--authority-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize(
            args.candidate_jsonl,
            args.upstream_handoff_report,
            args.output_jsonl,
            args.materialization_report,
            args.authority_report,
        )
    except (AuthorizationError, OSError, ValueError, TypeError, KeyError, _delegate.MaterializationError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_TREES_QUALITY_WINDOWS_AUTHORIZED=PASS_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
