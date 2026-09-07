#!/usr/bin/env python3
"""Byte-exact, zero-credit retest for the bounded Lang-UK Supreme Court source."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/d03_languk_supreme_court_retest_v1.json")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MARKER_MAP = {
    "person_count": "ОСОБА_",
    "address_count": "АДРЕСА_",
    "information_count": "ІНФОРМАЦІЯ_",
    "number_count": "НОМЕР_",
}


class RetestError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RetestError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    _require(cfg["schema_version"] == "12-6.d03-languk-supreme-court-retest.v1", "schema drift")
    src = cfg["source"]
    _require(src["dataset"] == "lang-uk/court-decisions-uk", "dataset drift")
    _require(src["revision"] == "2dcac4c941b87bf9c242bdc919cef4b40f4a4813", "revision drift")
    _require(src["sha256"] == "9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597", "source SHA drift")
    _require(src["bytes"] == 20220778, "source byte count drift")
    _require(src["excluded_file"] == "250-deanonymized-court-cases.parquet", "excluded-file drift")
    _require(cfg["selection"]["max_records"] == 256, "selection bound drift")
    boundary = cfg["claim_boundary"]
    for key in ("training_authorized_bytes", "canonical_capacity_credited", "family_credit_added", "optimizer_updates"):
        _require(boundary[key] == 0, f"zero-credit boundary drift: {key}")
    for key in ("tokenizer_fit_authorized", "model_training_executed", "final_test_accessed", "paid_compute_used", "research_corpus_v1_released"):
        _require(boundary[key] is False, f"false boundary drift: {key}")
    return cfg


def download_exact(cfg: dict[str, Any], output: Path) -> None:
    src = cfg["source"]
    request = urllib.request.Request(src["url"], headers={"User-Agent": "12-6-D03-LangUK-Retest/1"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as dst:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            _require(dst.tell() <= src["bytes"], "download exceeds pinned byte count")
    raw = output.read_bytes()
    _require(len(raw) == src["bytes"], f"source bytes mismatch: {len(raw)}")
    _require(_sha256(raw) == src["sha256"], "source SHA-256 mismatch")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _marker_count(text: str, prefix: str) -> int:
    return len(set(re.findall(re.escape(prefix) + r"\d+", text)))


def assess_row(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str, str]:
    text = row.get(cfg["selection"]["text_column"])
    if not isinstance(text, str):
        return False, "missing_text", ""
    if CONTROL_RE.search(text):
        return False, "control_character", ""
    if EMAIL_RE.search(text):
        return False, "email", ""
    if PHONE_RE.search(text):
        return False, "phone", ""

    observed_count_fields = 0
    for field, prefix in MARKER_MAP.items():
        if field not in row:
            continue
        observed_count_fields += 1
        expected = row[field]
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
            return False, f"invalid_{field}", ""
        if _marker_count(text, prefix) != expected:
            return False, f"marker_mismatch_{field}", ""
    if cfg["privacy"]["require_anonymization_marker_consistency"] and observed_count_fields == 0:
        return False, "marker_count_schema_missing", ""

    normalized = normalize(text)
    if len(normalized) < cfg["quality"]["min_chars"]:
        return False, "too_short", ""
    alpha = [c for c in normalized if c.isalpha()]
    cyr = [c for c in alpha if "CYRILLIC" in unicodedata.name(c, "")]
    ratio = len(cyr) / max(1, len(alpha))
    if ratio < cfg["quality"]["min_cyrillic_alpha_ratio"]:
        return False, "low_cyrillic_ratio", ""
    if cfg["quality"]["require_ukrainian_specific_letter"] and not re.search(r"[іїєґІЇЄҐ]", normalized):
        return False, "no_ukrainian_specific_letter", ""
    return True, "accepted", normalized


def select_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    id_col = next((name for name in cfg["selection"]["id_candidates"] if rows and name in rows[0]), None)
    _require(id_col is not None, "no stable id column found")
    ordered = sorted(rows, key=lambda row: str(row.get(id_col, "")))
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for row in ordered:
        rid = str(row.get(id_col, ""))
        if not rid or rid in seen_ids:
            reasons["invalid_or_duplicate_id"] += 1
            continue
        seen_ids.add(rid)
        ok, reason, text = assess_row(row, cfg)
        reasons[reason] += 1
        if not ok:
            continue
        digest = _sha256(text.encode())
        if digest in seen_hashes:
            reasons["exact_normalized_duplicate"] += 1
            continue
        seen_hashes.add(digest)
        accepted.append({"record_id": rid, "normalized_sha256": digest, "normalized_bytes": len(text.encode()), "text": text})
        if len(accepted) >= cfg["selection"]["max_records"]:
            break
    return accepted, reasons


def materialize(parquet_path: Path, output_jsonl: Path, report_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    raw = parquet_path.read_bytes()
    _require(len(raw) == cfg["source"]["bytes"], "local source byte count mismatch")
    _require(_sha256(raw) == cfg["source"]["sha256"], "local source SHA-256 mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RetestError("pyarrow is required for real parquet execution") from exc
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    _require(rows, "parquet contains no rows")
    accepted, reasons = select_rows(rows, cfg)
    _require(accepted, "privacy/quality retest retained zero rows")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(row) + b"\n" for row in accepted)
    output_jsonl.write_bytes(payload)
    report = {
        "schema_version": "12-6.d03-languk-supreme-court-retest-report.v1",
        "source_sha256": cfg["source"]["sha256"],
        "source_bytes": cfg["source"]["bytes"],
        "source_family": cfg["source"]["family"],
        "parquet_columns": table.column_names,
        "parquet_rows": table.num_rows,
        "retained_records": len(accepted),
        "retained_normalized_bytes": sum(row["normalized_bytes"] for row in accepted),
        "retained_jsonl_sha256": _sha256(payload),
        "disposition_counts": dict(sorted(reasons.items())),
        "rejected_text_emitted": False,
        "rejected_hashes_emitted": False,
        "training_authorized_bytes": 0,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "safe_result": "LANGUK_SUPREME_COURT_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_DEDUP_DECONTAM_REQUIRED",
    }
    unsigned = _canonical(report)
    report["report_identity_sha256"] = _sha256(unsigned)
    report_path.write_bytes(_canonical(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if args.download:
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        download_exact(cfg, args.parquet)
    report = materialize(args.parquet, args.output_jsonl, args.report, cfg)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
