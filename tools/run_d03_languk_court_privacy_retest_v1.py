"""Execute a bounded, zero-credit Lang-UK Supreme Court privacy/quality retest.

This carrier consumes one exact pinned parquet object, executes the incumbent
G06 privacy and G05 quality mechanics, and emits only text-free evidence.
It does not admit a source, grant corpus/tokenizer/training authority, perform
global deduplication, or read evaluation outcomes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.privacy_execution_authority import (
    build_privacy_execution_authority,
    input_rows_sha256,
    verify_privacy_execution_authority,
)
from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

SCHEMA = "12-6.d03-languk-court-privacy-retest.v1"
PARENT_AUDIT_HEAD = "0e682d8a63719063b7661600bf46e775b58ce823"
PARENT_AUDIT_ID = "96e1c85bdeaad908689b47252832ab9674897d3a3dff3e73804ca80b492ad9be"
PARENT_AUDIT_BLOB = "c88a2884f77f2c9747b56d7f08edfdf7b6478af2"
CANDIDATE_ID = "languk.court-decisions-uk.supreme-2024-5k"
DATASET = "lang-uk/court-decisions-uk"
REVISION = "289c0316fc076db3e1607db6776a29df42f4ffc5"
SOURCE_FILE = "2024-5K-supreme-court-decisions-deduplicated.parquet"
EXCLUDED_FILE = "250-deanonymized-court-cases.parquet"
SOURCE_SHA256 = "9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597"
SOURCE_BYTES = 20_220_778
FAMILY = "ua.languk.supreme-court-decisions"
MAX_RECORDS = 256
PYARROW_VERSION = "25.0.1"

EXPECTED_DEPENDENCY_BLOBS = {
    "src/twelve_six/data/document_quality.py": "b1461263034b4fb9510479b20c9697e22faa5f97",
    "src/twelve_six/data/quality_granularity.py": "513523b86824c423cad97352b3abb3d1241531b9",
    "src/twelve_six/data/quality_execution_authority.py": "4659a9d4aba49908f372250904a54361c8d8cf46",
    "src/twelve_six/data/privacy_execution_authority.py": "9215287e81c0a82f05ec8405dc4f34c60313c193",
    "src/twelve_six/data/privacy_filter_v3.py": "bcc5938395724f6728ab212f98b39f2334b0f37d",
}

_TRUTH = {
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "optimizer_updates_executed": 0,
    "tokenizer_fit_authorized": False,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "scale_promotion_authorized": False,
    "registry_change_authorized": False,
    "source_family_admitted": False,
}


class LangUkCourtRetestError(RuntimeError):
    """Fail-closed execution error."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise LangUkCourtRetestError(message)


def canonical(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hashed(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**copy.deepcopy(dict(core)), field: sha256(canonical(core))}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(dict(value), newline=True))


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if proc.returncode:
        detail = proc.stderr.strip() or proc.stdout.strip() or str(proc.returncode)
        raise LangUkCourtRetestError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def bind_execution_head(expected_head: str) -> dict[str, str]:
    need(git("rev-parse", "HEAD") == expected_head, "execution HEAD drift")
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_DEPENDENCY_BLOBS.items():
        actual = git("rev-parse", f"HEAD:{path}")
        need(actual == expected, f"dependency blob drift: {path}")
        observed[path] = actual
    return observed


def _verify_parent_identity(value: Mapping[str, Any]) -> None:
    claimed = value.get("evidence_identity_sha256")
    need(claimed == PARENT_AUDIT_ID, "parent audit identity drift")
    core = dict(value)
    core.pop("evidence_identity_sha256", None)
    need(sha256(canonical(core)) == claimed, "parent audit self-hash mismatch")


def verify_parent_audit(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LangUkCourtRetestError("cannot load parent rights audit") from exc
    need(type(value) is dict, "parent audit root must be object")
    need(git("hash-object", str(path)) == PARENT_AUDIT_BLOB, "parent audit blob drift")
    _verify_parent_identity(value)

    candidates = value.get("candidates")
    need(type(candidates) is list, "parent candidate list missing")
    matches = [
        row for row in candidates
        if isinstance(row, Mapping) and row.get("candidate_id") == CANDIDATE_ID
    ]
    need(len(matches) == 1, "exact Lang-UK candidate missing or duplicated")
    candidate = matches[0]
    need(
        candidate.get("verdict") == "RETEST_PRIVACY_AND_BYTE_MATERIALIZATION",
        "parent verdict widened or changed",
    )
    need(candidate.get("dataset_repo") == DATASET, "dataset identity drift")
    need(candidate.get("upstream_revision") == REVISION, "dataset revision drift")
    need(candidate.get("selected_file") == SOURCE_FILE, "selected file drift")
    need(candidate.get("selected_file_raw_sha256") == SOURCE_SHA256, "source SHA drift")
    need(candidate.get("selected_file_bytes") == SOURCE_BYTES, "source size drift")
    need(candidate.get("license") == "MIT", "dataset license-label drift")
    family = candidate.get("family_independence")
    need(isinstance(family, Mapping), "family evidence missing")
    need(family.get("proposed_family_id") == FAMILY, "proposed family drift")
    need(
        family.get("independent_from_incumbent_rada_family") is True,
        "primary family independence not established by parent audit",
    )
    subset = candidate.get("bounded_subset_rule")
    need(isinstance(subset, Mapping), "bounded subset contract missing")
    need(subset.get("allowed_source_file") == SOURCE_FILE, "allowed file drift")
    need(subset.get("exclude_file") == EXCLUDED_FILE, "excluded file drift")
    need(subset.get("max_records") == MAX_RECORDS, "bounded row cap drift")
    return candidate


def normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFKC", text)


def stable_id_key(value: str) -> tuple[int, Any, str]:
    if value.isascii() and value.isdigit():
        return (0, int(value), value)
    return (1, value, value)


def record_id(source_id: str) -> str:
    lineage = f"{DATASET}@{REVISION}:{SOURCE_FILE}#{source_id}".encode("utf-8")
    return "languk-court-" + sha256(lineage)


def _quality_input_rows_sha(records: Sequence[Mapping[str, str]]) -> str:
    normalized = sorted(records, key=lambda row: row["id"])
    rows = [
        {
            "record_id": row["id"],
            "mode": row["mode"],
            "payload_sha256": sha256(row["text"].encode("utf-8")),
            "utf8_bytes": len(row["text"].encode("utf-8")),
        }
        for row in normalized
    ]
    return sha256(canonical(rows, newline=True))


def ua_stats(texts: Sequence[str]) -> dict[str, int]:
    joined = "".join(texts)
    alpha = sum(ch.isalpha() for ch in joined)
    cyr = sum("\u0400" <= ch <= "\u052f" for ch in joined)
    latin = sum(("a" <= ch.casefold() <= "z") for ch in joined)
    specific_chars = frozenset("іїєґІЇЄҐ")
    specific = sum(ch in specific_chars for ch in joined)
    with_specific = sum(any(ch in specific_chars for ch in text) for text in texts)
    return {
        "records": len(texts),
        "unicode_characters": len(joined),
        "alphabetic_characters": alpha,
        "cyrillic_characters": cyr,
        "latin_ascii_letters": latin,
        "ukrainian_specific_letters": specific,
        "records_with_ukrainian_specific_letters": with_specific,
        "cyrillic_per_million_alpha": (cyr * 1_000_000 // alpha) if alpha else 0,
        "latin_per_million_alpha": (latin * 1_000_000 // alpha) if alpha else 0,
    }


def load_source(path: Path) -> list[dict[str, str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LangUkCourtRetestError("cannot read pinned parquet") from exc
    need(len(raw) == SOURCE_BYTES, "source byte count drift")
    need(sha256(raw) == SOURCE_SHA256, "source SHA-256 drift")

    try:
        import pyarrow
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise LangUkCourtRetestError("exact PyArrow runtime unavailable") from exc
    need(pyarrow.__version__ == PYARROW_VERSION, "PyArrow version drift")

    try:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        need({"id", "text"}.issubset(names), "parquet id/text schema missing")
        table = parquet.read(columns=["id", "text"])
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
    except LangUkCourtRetestError:
        raise
    except Exception as exc:
        raise LangUkCourtRetestError("cannot materialize exact parquet id/text columns") from exc

    need(len(ids) == len(texts) and bool(ids), "parquet row count invalid")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, (raw_id, raw_text) in enumerate(zip(ids, texts, strict=True)):
        need(raw_id is not None, f"row {index}: id missing")
        source_id = str(raw_id)
        need(bool(source_id), f"row {index}: id empty")
        need(source_id not in seen, f"duplicate source id: {source_id}")
        need(isinstance(raw_text, str) and bool(raw_text), f"row {index}: text missing")
        try:
            raw_text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise LangUkCourtRetestError(f"row {index}: text is not strict UTF-8") from exc
        seen.add(source_id)
        rows.append(
            {
                "source_id": source_id,
                "raw_text": raw_text,
                "normalized_text": normalize_text(raw_text),
                "record_id": record_id(source_id),
            }
        )
    return sorted(rows, key=lambda row: stable_id_key(row["source_id"]))


def run(args: argparse.Namespace) -> None:
    dependency_blobs = bind_execution_head(args.expected_execution_head)
    verify_parent_audit(args.parent_audit)
    source_rows = load_source(args.source_parquet)

    privacy_inputs = [
        {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
        for row in source_rows
    ]
    privacy_input_root = input_rows_sha256(privacy_inputs)
    privacy = build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
    )
    verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
        expected_execution_identity_sha256=privacy["execution_identity_sha256"],
    )
    privacy_by_id = {
        row["record_id"]: row
        for row in privacy["records"]
    }
    allow_rows = [
        row for row in source_rows
        if privacy_by_id[row["record_id"]]["action"] == "ALLOW"
    ]
    selected = allow_rows[:MAX_RECORDS]

    raw_projection = [
        {"id": row["source_id"], "text": row["raw_text"]}
        for row in selected
    ]
    normalized_projection = [
        {"id": row["source_id"], "text": row["normalized_text"]}
        for row in selected
    ]
    raw_subset_bytes = canonical(raw_projection, newline=True)
    normalized_subset_bytes = canonical(normalized_projection, newline=True)

    selected_manifest_rows = [
        {
            "record_id": row["record_id"],
            "source_id_sha256": sha256(row["source_id"].encode("utf-8")),
            "raw_payload_sha256": sha256(row["raw_text"].encode("utf-8")),
            "normalized_payload_sha256": sha256(row["normalized_text"].encode("utf-8")),
            "normalized_payload_bytes": len(row["normalized_text"].encode("utf-8")),
        }
        for row in selected
    ]
    selected_manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_file": SOURCE_FILE,
        "source_family": FAMILY,
        "family_admitted": False,
        "max_records": MAX_RECORDS,
        "selection_rule": "privacy_action_ALLOW_then_ascending_stable_source_id",
        "records": selected_manifest_rows,
    }
    selected_manifest_id = sha256(canonical(selected_manifest))

    quality: dict[str, Any] | None = None
    if selected:
        quality_inputs = [
            {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
            for row in selected
        ]
        quality_input_root = _quality_input_rows_sha(quality_inputs)
        quality = build_quality_execution_authority(
            quality_inputs,
            input_manifest_sha256=selected_manifest_id,
            expected_input_rows_sha256=quality_input_root,
        )
        verify_quality_execution_authority(
            quality,
            quality_inputs,
            expected_input_manifest_sha256=selected_manifest_id,
            expected_input_rows_sha256=quality_input_root,
            expected_execution_identity_sha256=quality["execution_identity_sha256"],
        )
    else:
        quality_input_root = None

    actions = privacy["counts"]
    detectors = privacy["detector_counts"]
    selected_texts = [row["normalized_text"] for row in selected]
    report_core = {
        "schema_version": SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "parent_rights_audit": {
            "pr": 447,
            "head_sha": PARENT_AUDIT_HEAD,
            "evidence_identity_sha256": PARENT_AUDIT_ID,
            "verdict": "RETEST_PRIVACY_AND_BYTE_MATERIALIZATION",
        },
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "file": SOURCE_FILE,
            "excluded_file": EXCLUDED_FILE,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
        },
        "dependency_git_blobs": dependency_blobs,
        "materialization": {
            "source_records": len(source_rows),
            "privacy_input_rows_sha256": privacy_input_root,
            "privacy_execution_identity_sha256": privacy["execution_identity_sha256"],
            "privacy_action_counts": actions,
            "privacy_detector_counts": detectors,
            "privacy_coverage_claim": (
                "deterministic high-confidence patterns; zero findings is not proof of zero PII/secrets"
            ),
            "allow_records_available": len(allow_rows),
            "selected_records": len(selected),
            "selection_cap": MAX_RECORDS,
            "selected_manifest_identity_sha256": selected_manifest_id,
            "raw_subset_sha256": sha256(raw_subset_bytes),
            "raw_subset_canonical_bytes": len(raw_subset_bytes),
            "normalized_subset_sha256": sha256(normalized_subset_bytes),
            "normalized_subset_canonical_bytes": len(normalized_subset_bytes),
            "quality_input_rows_sha256": quality_input_root,
            "quality_execution_identity_sha256": (
                quality["execution_identity_sha256"] if quality is not None else None
            ),
            "quality_counts": quality["counts"] if quality is not None else None,
            "quality_bytes": quality["bytes"] if quality is not None else None,
            "ukrainian_language_statistics": ua_stats(selected_texts),
        },
        "family": {
            "proposed_family_id": FAMILY,
            "primary_lineage_distinct_from_rada": True,
            "family_admitted": False,
            "cross_family_global_dedup_required": True,
        },
        "next_required": [
            "global_exact_near_lineage_dedup_against_current_terminal_sources",
            "reserved_evaluation_decontamination",
            "post_dedup_G05_G06_reexecution",
            "late_registry_refresh",
            "family_cap_balance_recompute",
        ],
        "durable_evidence_hash_only": True,
        "truth_boundary": dict(_TRUTH),
    }
    report = self_hashed(report_core, "report_identity_sha256")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "privacy-authority.json", privacy)
    if quality is not None:
        write_json(args.output_dir / "quality-authority.json", quality)
    write_json(args.output_dir / "report.json", report)

    print(
        json.dumps(
            {
                "status": "RETEST_EXECUTED_ZERO_CREDIT",
                "source_records": len(source_rows),
                "privacy_actions": actions,
                "selected_records": len(selected),
                "report_identity_sha256": report["report_identity_sha256"],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--expected-execution-head", required=True)
    run_cmd.add_argument("--parent-audit", type=Path, required=True)
    run_cmd.add_argument("--source-parquet", type=Path, required=True)
    run_cmd.add_argument("--output-dir", type=Path, required=True)
    run_cmd.set_defaults(func=run)
    return value


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
