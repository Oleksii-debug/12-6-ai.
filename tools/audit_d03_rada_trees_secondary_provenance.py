#!/usr/bin/env python3
"""Fail-closed text-free provenance audit for terminal Rada_Trees plaintext scan.

This consumes the text-free full-scan report emitted by the Rada_Trees secondary
archive scanner. It never needs member text. It binds the exact source/report and
classifies exact-deduplicated survivors by source-path provenance. Only the regular
`texts/YYYY-MM-DD__*.txt` plenary-session shape is admitted to the Rada parliamentary
family. Nonmatching paths are quarantined rather than silently inheriting that family.

This audit grants no corpus or training authority; downstream rights, language,
quality, privacy, global dedup, decontamination, balance, split and packing gates remain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-trees-secondary-provenance.v1"
INPUT_SCHEMA = "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1"
DATASET = "uacorpus/Rada_Trees"
DATASET_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
ARCHIVE = "rada_xtag_texts.7z"
ARCHIVE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
INPUT_REPORT_SHA256 = "689dbea2b5975c1d073dde7fefa814af694378d12a88adcb33b1caf2e0ffa291"
MEMBER_METADATA_SHA256 = "9d6ba19977d91a0cd8068b7921790daa9c13e6040a797e4f5fb683d4fca40008"
EXPECTED_MEMBER_COUNT = 4_391
EXPECTED_CANDIDATE_BYTES = 879_031_855
EXPECTED_UNIQUE_COUNT = 4_385
EXPECTED_UNIQUE_BYTES = 877_983_909
SESSION_RE = re.compile(r"^texts/(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})__(?P<label>.+)\.txt$")


class ProvenanceAuditError(RuntimeError):
    """Raised when exact input identity or provenance invariants drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceAuditError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_scan_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(report, dict), "scan report root must be an object")
    require(report.get("schema_version") == INPUT_SCHEMA, "scan report schema drift")
    require(report.get("report_sha256") == INPUT_REPORT_SHA256, "scan report identity drift")

    source = report.get("source")
    require(isinstance(source, dict), "scan source block missing")
    require(source.get("dataset") == DATASET, "dataset drift")
    require(source.get("dataset_revision") == DATASET_REVISION, "dataset revision drift")
    require(source.get("archive_path") == ARCHIVE, "archive path drift")
    require(source.get("content_sha256") == ARCHIVE_SHA256, "archive content identity drift")

    boundary = report.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "input training credit must remain zero")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "input loss credit must remain zero")
    require(boundary.get("tokenizer_fit_authorized") is False, "input tokenizer boundary weakened")
    require(boundary.get("model_training_executed") is False, "input model-training boundary weakened")
    require(boundary.get("final_test_payload_accessed") is False, "input final-test boundary weakened")
    require(boundary.get("paid_compute_used") is False, "input paid-compute boundary weakened")

    classification = report.get("classification")
    require(isinstance(classification, dict), "classification block missing")
    metadata = classification.get("member_metadata")
    require(isinstance(metadata, list), "member metadata missing")
    require(len(metadata) == EXPECTED_MEMBER_COUNT, "member count drift")
    require(sha256_json(metadata) == MEMBER_METADATA_SHA256, "member metadata identity drift")
    require(classification.get("plain_text_candidate_members") == EXPECTED_MEMBER_COUNT, "candidate member count drift")
    require(
        classification.get("plain_text_candidate_bytes_before_exact_duplicate_collapse") == EXPECTED_CANDIDATE_BYTES,
        "candidate byte total drift",
    )
    require(
        classification.get("plain_text_candidate_exact_unique_payload_count") == EXPECTED_UNIQUE_COUNT,
        "unique candidate count drift",
    )
    require(
        classification.get("plain_text_candidate_bytes_after_exact_duplicate_collapse") == EXPECTED_UNIQUE_BYTES,
        "unique candidate byte total drift",
    )
    return report


def exact_hash_survivors(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        require(isinstance(item, dict), "member metadata entry must be an object")
        require(item.get("classification") == "PLAIN_TEXT_CANDIDATE", "non-candidate member in terminal candidate inventory")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        require(isinstance(path, str) and path.startswith("texts/") and path.endswith(".txt"), "member path outside plaintext scope")
        require(isinstance(digest, str) and len(digest) == 64, f"invalid member SHA for {path}")
        require(isinstance(size, int) and size >= 0, f"invalid member size for {path}")
        hints = item.get("path_year_hints")
        require(isinstance(hints, list) and len(hints) == 1 and isinstance(hints[0], int), f"ambiguous year hint for {path}")
        groups[digest].append(item)
    survivors = [min(group, key=lambda row: row["path"]) for group in groups.values()]
    survivors.sort(key=lambda row: row["path"])
    require(len(survivors) == EXPECTED_UNIQUE_COUNT, "exact-hash survivor count drift")
    require(sum(int(row["size_bytes"]) for row in survivors) == EXPECTED_UNIQUE_BYTES, "exact-hash survivor bytes drift")
    return survivors


def classify_path(row: dict[str, Any]) -> tuple[str, int | None, str]:
    path = str(row["path"])
    match = SESSION_RE.fullmatch(path)
    if not match:
        return "QUARANTINE_NON_PARLIAMENT_PATH", None, "path_not_dated_plenary_session_shape"
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    try:
        dt.date(year, month, day)
    except ValueError as exc:
        raise ProvenanceAuditError(f"invalid session date in {path}: {exc}") from exc
    require(1990 <= year <= 2024, f"session year outside declared source period: {path}")
    require(row["path_year_hints"] == [year], f"path-year hint disagrees with session date: {path}")
    return "RADA_PLENARY_SESSION_PATH_CANDIDATE", year, "dated_session_shape_and_year_hint_agree"


def build_audit(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["classification"]["member_metadata"]
    survivors = exact_hash_survivors(metadata)
    disposition_counts: Counter[str] = Counter()
    disposition_bytes: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    year_bytes: Counter[int] = Counter()
    quarantined: list[dict[str, Any]] = []
    admitted_identity_rows: list[dict[str, Any]] = []

    for row in survivors:
        disposition, year, reason = classify_path(row)
        size = int(row["size_bytes"])
        disposition_counts[disposition] += 1
        disposition_bytes[disposition] += size
        if disposition == "RADA_PLENARY_SESSION_PATH_CANDIDATE":
            assert year is not None
            year_counts[year] += 1
            year_bytes[year] += size
            admitted_identity_rows.append({"path": row["path"], "sha256": row["sha256"], "size_bytes": size, "year": year})
        else:
            quarantined.append({"path": row["path"], "sha256": row["sha256"], "size_bytes": size, "reason": reason})

    admitted_count = disposition_counts["RADA_PLENARY_SESSION_PATH_CANDIDATE"]
    admitted_bytes = disposition_bytes["RADA_PLENARY_SESSION_PATH_CANDIDATE"]
    quarantine_count = disposition_counts["QUARANTINE_NON_PARLIAMENT_PATH"]
    quarantine_bytes = disposition_bytes["QUARANTINE_NON_PARLIAMENT_PATH"]
    require(admitted_count + quarantine_count == EXPECTED_UNIQUE_COUNT, "survivor disposition count mismatch")
    require(admitted_bytes + quarantine_bytes == EXPECTED_UNIQUE_BYTES, "survivor disposition byte mismatch")

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": "D03-RADA-TREES-PERIOD-SESSION-PROVENANCE-20260907",
        "execution_profile": "LOCAL_FREE_TEXT_FREE_ARTIFACT_REPLAY",
        "source_binding": {
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "archive_path": ARCHIVE,
            "archive_sha256": ARCHIVE_SHA256,
            "full_scan_report_sha256": INPUT_REPORT_SHA256,
            "member_metadata_identity_sha256": MEMBER_METADATA_SHA256,
        },
        "exact_duplicate_survivors": {
            "count": EXPECTED_UNIQUE_COUNT,
            "bytes": EXPECTED_UNIQUE_BYTES,
        },
        "provenance_disposition": {
            "rada_plenary_session_path_candidate_count": admitted_count,
            "rada_plenary_session_path_candidate_bytes": admitted_bytes,
            "quarantine_non_parliament_path_count": quarantine_count,
            "quarantine_non_parliament_path_bytes": quarantine_bytes,
            "admitted_member_identity_sha256": sha256_json(admitted_identity_rows),
            "quarantined_members": quarantined,
            "year_member_counts": {str(year): year_counts[year] for year in sorted(year_counts)},
            "year_bytes": {str(year): year_bytes[year] for year in sorted(year_bytes)},
            "first_year": min(year_counts) if year_counts else None,
            "last_year": max(year_counts) if year_counts else None,
        },
        "decision": {
            "state": "PROVENANCE_PATH_SCOPE_PASS_WITH_QUARANTINE_ZERO_CREDIT",
            "parliament_family_candidate_bytes_after_exact_dedup_and_path_scope": admitted_bytes,
            "training_admission_claimed": False,
            "next_required": [
                "rights_scope_revalidation_and_attribution_contract",
                "language_quality_privacy_on_exact_provenance_survivors",
                "global_exact_near_lineage_dedup",
                "reserved_evaluation_decontamination",
                "family_cap_mix_recompute",
            ],
        },
        "claim_boundary": {
            "provenance_candidate_bytes_are_training_credit": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    core["evidence_identity_sha256"] = sha256_json(core)
    return core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-scan-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_audit(load_scan_report(args.full_scan_report))
    except (OSError, ValueError, json.JSONDecodeError, ProvenanceAuditError) as exc:
        print(f"PROVENANCE_AUDIT_FAIL: {exc}")
        return 2
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
