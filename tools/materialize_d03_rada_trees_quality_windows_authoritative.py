#!/usr/bin/env python3
"""Authority-bound successor for Rada_Trees quality-window materialization.

This wrapper closes the upstream-authority and open-schema seams found by #975.
It delegates all quality/privacy mechanics to the incumbent materializer and adds
only fail-closed binding to the exact #916 handoff evidence and candidate schema.
It does not repair or redefine privacy detector semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import materialize_d03_rada_trees_quality_windows as base

SCHEMA = "12-6.d03-rada-trees-quality-window-authority-bound.v1"
UPSTREAM_SCHEMA = "12-6.d03-rada-trees-rights-scoped-handoff-report.v1"
UPSTREAM_SAFE_RESULT = "RADA_TREES_RIGHTS_SCOPED_HANDOFF_ZERO_CREDIT"

EXPECTED_ARCHIVE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
EXPECTED_ARCHIVE_BYTES = 697_768_591
EXPECTED_RECORDS = 4_384
EXPECTED_SOURCE_BYTES = 877_899_128
EXPECTED_FULL_SCAN_EVIDENCE_ID = "b25ff7e7948e2fa215ee8d059ea5594025693e105389b7095f5250808ecc3a9d"
EXPECTED_RIGHTS_CONFIG_SHA256 = "34da44a047c5e0d562ee6a86987cb66e3ed266e1c1f09af95e63c38c219fe1a3"
EXPECTED_RIGHTS_REPORT_SHA256 = "7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d"
EXPECTED_ACCEPTED_INVENTORY_SHA256 = "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
EXPECTED_HELD_INVENTORY_SHA256 = "566760e10157cd835ed0879abb37f052b57d31cff6af358a81ff717f4f7f59d9"
EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA = "a84395daf6f07726d6a730cf6ccb04c35bdc4e67"
PRIVACY_REPAIR_REFERENCE_MERGE_SHA = "82a2cacfec2670feff957d64904f1b8fe310dbd4"
PRIVACY_SOURCE_RELATIVE_PATH = Path("src/twelve_six/data/privacy_filter_v3.py")

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


class AuthorityBindingError(RuntimeError):
    """Raised when exact upstream/candidate authority cannot be reproduced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityBindingError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _git_blob_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), "privacy implementation must be a regular file")
    payload = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _load_bound_mechanics() -> tuple[tuple[Any, ...], str]:
    """Load incumbent mechanics once and bind the exact imported privacy source."""
    mechanics = base._load_mechanics()
    require(len(mechanics) == 4, "incumbent mechanics tuple drift")
    scan_privacy = mechanics[2]
    privacy_policy = mechanics[3]

    privacy_module = importlib.import_module(base.PRIVACY_MODULE)
    module_file = getattr(privacy_module, "__file__", None)
    require(isinstance(module_file, str) and module_file, "imported privacy module has no source path")
    try:
        observed_path = Path(module_file).resolve(strict=True)
        expected_path = (TOOLS.parent / PRIVACY_SOURCE_RELATIVE_PATH).resolve(strict=True)
    except OSError as exc:
        raise AuthorityBindingError("cannot resolve imported privacy implementation path") from exc
    require(
        observed_path == expected_path,
        "imported privacy implementation resolved outside canonical repository source",
    )
    observed_blob = _git_blob_sha(observed_path)
    require(
        observed_blob == EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA,
        "canonical privacy repair implementation drift",
    )
    require(
        scan_privacy is getattr(privacy_module, "hash_safe_scan", None),
        "executed privacy scan is not from the bound canonical module",
    )
    require(
        privacy_policy is getattr(privacy_module, "policy_manifest", None),
        "executed privacy policy provider is not from the bound canonical module",
    )
    return mechanics, observed_blob


def _load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityBindingError(f"cannot load {label}") from exc
    require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _validate_upstream_report(
    path: Path, expected_report_sha256: str
) -> tuple[str, str]:
    require(_is_sha256(expected_report_sha256), "expected upstream report SHA-256 is malformed")
    report = _load_json(path, "upstream handoff report")
    claimed = report.get("report_sha256")
    require(_is_sha256(claimed), "upstream report self-hash missing")
    core = dict(report)
    core.pop("report_sha256", None)
    observed = base.sha256_bytes(base.canonical_bytes(core))
    require(observed == claimed, "upstream handoff report self-hash invalid")
    require(observed == expected_report_sha256, "upstream handoff report identity mismatch")
    require(report.get("schema_version") == UPSTREAM_SCHEMA, "upstream handoff schema drift")
    require(report.get("execution_profile") == "LOCAL_FREE", "upstream execution profile drift")
    require(report.get("safe_result") == UPSTREAM_SAFE_RESULT, "upstream safe-result drift")

    source = _mapping(report.get("source"), "upstream source")
    require(source.get("dataset") == base.EXPECTED_DATASET, "upstream dataset drift")
    require(source.get("revision") == base.EXPECTED_REVISION, "upstream revision drift")
    require(source.get("archive") == base.EXPECTED_ARCHIVE, "upstream archive drift")
    require(source.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256, "upstream archive SHA drift")
    require(source.get("archive_bytes") == EXPECTED_ARCHIVE_BYTES, "upstream archive byte drift")
    require(source.get("family") == base.EXPECTED_FAMILY, "upstream family drift")

    parent = _mapping(report.get("parent_authority"), "upstream parent authority")
    require(
        parent.get("full_scan_evidence_identity_sha256") == EXPECTED_FULL_SCAN_EVIDENCE_ID,
        "upstream full-scan authority drift",
    )
    require(
        parent.get("rights_config_sha256") == EXPECTED_RIGHTS_CONFIG_SHA256,
        "upstream rights-config authority drift",
    )
    require(
        parent.get("rights_report_sha256") == EXPECTED_RIGHTS_REPORT_SHA256,
        "upstream rights-report authority drift",
    )
    require(
        parent.get("accepted_path_inventory_sha256") == EXPECTED_ACCEPTED_INVENTORY_SHA256,
        "upstream accepted-inventory authority drift",
    )
    require(
        parent.get("held_path_inventory_sha256") == EXPECTED_HELD_INVENTORY_SHA256,
        "upstream held-inventory authority drift",
    )

    materialization = _mapping(report.get("materialization"), "upstream materialization")
    candidate_sha = materialization.get("candidate_jsonl_sha256")
    require(_is_sha256(candidate_sha), "upstream candidate SHA-256 missing")
    require(materialization.get("candidate_records") == EXPECTED_RECORDS, "upstream record count drift")
    require(
        materialization.get("accepted_source_payload_bytes") == EXPECTED_SOURCE_BYTES,
        "upstream accepted source-byte total drift",
    )
    require(materialization.get("held_records_not_emitted") == 1, "upstream held-record count drift")
    require(
        materialization.get("source_native_document_boundaries_preserved") is True,
        "upstream source-native boundary drift",
    )
    require(
        materialization.get("record_text_persisted_in_report") is False,
        "upstream report payload-retention drift",
    )

    boundary = _mapping(report.get("claim_boundary"), "upstream claim boundary")
    require(boundary.get("candidate_jsonl_is_canonical_corpus") is False, "upstream corpus claim widened")
    require(boundary.get("training_authorized_bytes") == 0, "upstream training credit drift")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "upstream loss-position credit drift",
    )
    require(boundary.get("tokenizer_fit_authorized") is False, "upstream tokenizer authority drift")
    require(boundary.get("optimizer_updates") == 0, "upstream optimizer authority drift")
    require(boundary.get("model_training_executed") is False, "upstream training-execution drift")
    require(boundary.get("final_test_payload_accessed") is False, "upstream final-test boundary drift")
    require(boundary.get("paid_compute_used") is False, "upstream paid-compute boundary drift")
    return observed, str(candidate_sha)


def _candidate_identity(path: Path) -> tuple[str, int, int]:
    require(path.is_file() and not path.is_symlink(), "candidate must be a regular file")
    digest = hashlib.sha256()
    records = 0
    source_bytes = 0
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, 1):
            require(line.strip(), f"blank candidate JSONL line at {line_no}")
            digest.update(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuthorityBindingError(f"invalid candidate JSONL at line {line_no}") from exc
            require(isinstance(row, dict), f"candidate record {line_no} must be an object")
            require(
                set(row) == EXPECTED_CANDIDATE_KEYS,
                f"candidate schema/keyset drift at line {line_no}",
            )
            record_id, _ = base._validate_row(row, line_no)
            require(record_id not in seen, f"duplicate candidate record_id: {record_id}")
            seen.add(record_id)
            source_path = row.get("source_path")
            session_date = row.get("session_date")
            source_sha = row.get("source_payload_sha256")
            source_nbytes = row.get("source_payload_bytes")
            decoded_encoding = row.get("decoded_encoding")
            require(isinstance(source_path, str) and source_path, f"source path missing: {record_id}")
            require(isinstance(session_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", session_date) is not None, f"session date malformed: {record_id}")
            require(_is_sha256(source_sha), f"source payload SHA malformed: {record_id}")
            require(type(source_nbytes) is int and source_nbytes > 0, f"source payload bytes malformed: {record_id}")
            require(isinstance(decoded_encoding, str) and decoded_encoding, f"decoded encoding missing: {record_id}")
            records += 1
            source_bytes += int(source_nbytes)
    require(records == EXPECTED_RECORDS, "candidate record count does not match upstream authority")
    require(source_bytes == EXPECTED_SOURCE_BYTES, "candidate source-byte total does not match upstream authority")
    return digest.hexdigest(), records, source_bytes


def materialize_authoritative(
    candidate: Path,
    upstream_report: Path,
    output: Path,
    report_path: Path,
    *,
    expected_upstream_report_sha256: str,
) -> dict[str, Any]:
    """Bind exact upstream and executed privacy mechanics, then materialize."""
    require(not output.exists(), "output JSONL already exists")
    require(not report_path.exists(), "authority-bound report already exists")
    require(not output.is_symlink(), "output JSONL path must not be a symlink")
    require(not report_path.is_symlink(), "report path must not be a symlink")
    resolved = {
        candidate.resolve(strict=False),
        upstream_report.resolve(strict=False),
        output.resolve(strict=False),
        report_path.resolve(strict=False),
    }
    require(len(resolved) == 4, "candidate, upstream report, output, and report paths must be distinct")

    bound_mechanics, privacy_blob_sha = _load_bound_mechanics()
    upstream_sha, expected_candidate_sha = _validate_upstream_report(
        upstream_report, expected_upstream_report_sha256
    )
    observed_candidate_sha, records, source_bytes = _candidate_identity(candidate)
    require(
        observed_candidate_sha == expected_candidate_sha,
        "candidate JSONL identity does not match bound upstream handoff report",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    mechanics_report = report_path.with_suffix(report_path.suffix + ".mechanics.partial")
    final_partial = report_path.with_suffix(report_path.suffix + ".partial")
    require(not mechanics_report.exists(), "stale mechanics partial report exists")
    require(not final_partial.exists(), "stale authority-bound partial report exists")
    published_output = False
    original_loader = base._load_mechanics
    try:
        base._load_mechanics = lambda: bound_mechanics
        mechanics = base.materialize(
            candidate,
            output,
            mechanics_report,
            expected_candidate_sha256=expected_candidate_sha,
        )
        published_output = output.exists()
        require(
            mechanics.get("input_candidate_jsonl_sha256") == observed_candidate_sha,
            "incumbent materializer input binding drift",
        )
        claimed_mechanics_sha = mechanics.get("report_sha256")
        require(_is_sha256(claimed_mechanics_sha), "incumbent materializer report identity missing")
        mechanics_core = dict(mechanics)
        mechanics_core.pop("report_sha256", None)
        require(
            base.sha256_bytes(base.canonical_bytes(mechanics_core)) == claimed_mechanics_sha,
            "incumbent materializer report self-hash invalid",
        )

        core = dict(mechanics_core)
        core["schema_version"] = SCHEMA
        core["authority_binding"] = {
            "upstream_handoff_schema_version": UPSTREAM_SCHEMA,
            "upstream_handoff_report_sha256": upstream_sha,
            "candidate_jsonl_sha256": observed_candidate_sha,
            "candidate_record_count": records,
            "candidate_source_payload_bytes": source_bytes,
            "candidate_exact_keyset_enforced": True,
            "unknown_candidate_fields_rejected": True,
            "incumbent_materializer_report_sha256": claimed_mechanics_sha,
            "privacy_repair_reference_merge_sha": PRIVACY_REPAIR_REFERENCE_MERGE_SHA,
            "privacy_filter_v3_git_blob_sha": privacy_blob_sha,
            "privacy_filter_v3_resolved_path": PRIVACY_SOURCE_RELATIVE_PATH.as_posix(),
            "executed_privacy_mechanics_pinned": True,
        }
        boundary = dict(_mapping(core.get("claim_boundary"), "materializer claim boundary"))
        boundary["upstream_handoff_authority_bound"] = True
        boundary["candidate_schema_exact"] = True
        boundary["canonical_privacy_repair_bound"] = True
        core["claim_boundary"] = boundary
        core["safe_result"] = "RADA_TREES_QUALITY_WINDOWS_AUTHORITY_BOUND_ZERO_CREDIT"
        report = {**core, "report_sha256": base.sha256_bytes(base.canonical_bytes(core))}
        final_partial.write_bytes(base.canonical_bytes(report))
        final_partial.replace(report_path)
        mechanics_report.unlink(missing_ok=True)
        return report
    except Exception:
        mechanics_report.unlink(missing_ok=True)
        final_partial.unlink(missing_ok=True)
        if published_output:
            output.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    finally:
        base._load_mechanics = original_loader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--upstream-handoff-report", type=Path, required=True)
    parser.add_argument("--expected-upstream-report-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize_authoritative(
            args.candidate_jsonl,
            args.upstream_handoff_report,
            args.output_jsonl,
            args.report,
            expected_upstream_report_sha256=args.expected_upstream_report_sha256,
        )
    except (AuthorityBindingError, base.MaterializationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_TREES_QUALITY_WINDOWS_AUTHORITY_BOUND=PASS_ZERO_CREDIT")
    print("OUTPUT_RECORDS=" + str(report["output_records"]))
    print("UPSTREAM_REPORT_SHA256=" + report["authority_binding"]["upstream_handoff_report_sha256"])
    print("PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA=" + report["authority_binding"]["privacy_filter_v3_git_blob_sha"])
    print("REPORT_SHA256=" + report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
