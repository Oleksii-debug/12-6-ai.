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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/d03_languk_supreme_court_retest_v1.json")
SOURCE_DATASET = "lang-uk/court-decisions-uk"
SOURCE_REVISION = "2dcac4c941b87bf9c242bdc919cef4b40f4a4813"
SOURCE_FILE = "2024-5K-supreme-court-decisions-deduplicated.parquet"
SOURCE_SHA256 = "9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597"
SOURCE_BYTES = 20_220_778
PYARROW_VERSION = "17.0.0"
EXPECTED_FIELDS = (
    "id",
    "text",
    "number_count",
    "information_count",
    "person_count",
    "address_count",
    "sum_of_unique_entities",
    "number_occurrences",
    "information_occurrences",
    "person_occurrences",
    "address_occurrences",
    "__index_level_0__",
)
OCCURRENCE_FIELDS = {
    "number": ("number_count", "number_occurrences", re.compile(r"НОМЕР_[1-9][0-9]*")),
    "information": (
        "information_count",
        "information_occurrences",
        re.compile(r"ІНФОРМАЦІЯ_[1-9][0-9]*"),
    ),
    "person": ("person_count", "person_occurrences", re.compile(r"ОСОБА_[1-9][0-9]*")),
    "address": (
        "address_count",
        "address_occurrences",
        re.compile(r"АДРЕСА_[1-9][0-9]*"),
    ),
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class RetestError(RuntimeError):
    """Fail-closed Lang-UK retest error."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RetestError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise RetestError(f"cannot read source parquet: {path}") from exc
    return size, digest.hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetestError(f"cannot load config: {path}") from exc
    _require(isinstance(cfg, dict), "config root must be an object")
    _require(
        cfg.get("schema_version") == "12-6.d03-languk-supreme-court-retest.v1",
        "schema drift",
    )
    _require(cfg.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    src = cfg.get("source")
    _require(isinstance(src, Mapping), "source config missing")
    expected_source = {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "url": (
            "https://huggingface.co/datasets/lang-uk/court-decisions-uk/resolve/"
            f"{SOURCE_REVISION}/{SOURCE_FILE}"
        ),
        "sha256": SOURCE_SHA256,
        "bytes": SOURCE_BYTES,
        "excluded_file": "250-deanonymized-court-cases.parquet",
        "family": "ua.languk.supreme-court-decisions",
    }
    for key, expected in expected_source.items():
        _require(src.get(key) == expected, f"source identity drift: {key}")

    selection = cfg.get("selection")
    _require(isinstance(selection, Mapping), "selection config missing")
    _require(selection.get("max_records") == 256, "selection bound drift")
    _require(
        selection.get("order") == "numeric_id_ascending_after_privacy_pass",
        "selection order drift",
    )
    _require(selection.get("text_column") == "text", "text column drift")
    _require(selection.get("id_column") == "id", "id column drift")

    schema = cfg.get("parquet_schema")
    _require(isinstance(schema, Mapping), "parquet schema contract missing")
    _require(
        schema.get("required_fields") == list(EXPECTED_FIELDS),
        "parquet field contract drift",
    )
    _require(
        schema.get("reject_extra_fields") is True,
        "extra-field rejection disabled",
    )

    privacy = cfg.get("privacy")
    _require(isinstance(privacy, Mapping), "privacy config missing")
    required_privacy = {
        "reject_email": True,
        "reject_phone": True,
        "reject_control_characters": True,
        "require_anonymization_marker_consistency": True,
        "verify_occurrence_spans": True,
        "require_complete_occurrence_annotation": True,
        "rejected_text_emitted": False,
        "rejected_hashes_emitted": False,
    }
    for key, expected in required_privacy.items():
        _require(privacy.get(key) is expected, f"privacy boundary drift: {key}")
    _require(
        privacy.get("marker_prefixes")
        == ["ОСОБА_", "АДРЕСА_", "ІНФОРМАЦІЯ_", "НОМЕР_"],
        "marker-prefix vector drift",
    )

    quality = cfg.get("quality")
    _require(isinstance(quality, Mapping), "quality config missing")
    _require(quality.get("min_chars") == 200, "min_chars drift")
    _require(
        quality.get("min_cyrillic_alpha_ratio") == 0.70,
        "Cyrillic threshold drift",
    )
    _require(
        quality.get("require_ukrainian_specific_letter") is True,
        "UA letter gate disabled",
    )

    runtime = cfg.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime contract missing")
    _require(runtime.get("pyarrow") == PYARROW_VERSION, "PyArrow version drift")

    boundary = cfg.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim boundary missing")
    for key in (
        "training_authorized_bytes",
        "canonical_capacity_credited",
        "family_credit_added",
        "optimizer_updates",
    ):
        _require(boundary.get(key) == 0, f"zero-credit boundary drift: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_accessed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        _require(boundary.get(key) is False, f"false boundary drift: {key}")
    return cfg


def download_exact(cfg: Mapping[str, Any], output: Path) -> None:
    src = cfg["source"]
    request = urllib.request.Request(
        src["url"],
        headers={"User-Agent": "12-6-D03-LangUK-Retest/1"},
    )
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as dst,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                _require(total <= src["bytes"], "download exceeds pinned byte count")
                digest.update(chunk)
                dst.write(chunk)
        _require(total == src["bytes"], f"source bytes mismatch: {total}")
        _require(digest.hexdigest() == src["sha256"], "source SHA-256 mismatch")
        partial.replace(output)
    except OSError as exc:
        raise RetestError("exact source download failed") from exc
    finally:
        if partial.exists():
            partial.unlink()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _validate_occurrences(row: Mapping[str, Any], text: str) -> None:
    nonzero_types = 0
    for count_field, occurrence_field, pattern in OCCURRENCE_FIELDS.values():
        count = row.get(count_field)
        occurrences = row.get(occurrence_field)
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0,
            f"invalid_{count_field}",
        )
        _require(isinstance(occurrences, list), f"invalid_{occurrence_field}")
        _require(len(occurrences) == count, f"occurrence_count_mismatch_{count_field}")
        nonzero_types += int(count > 0)

        annotated: list[tuple[int, int, str]] = []
        for item in occurrences:
            _require(
                isinstance(item, Mapping) and set(item) == {"start", "end", "text"},
                f"occurrence_schema_drift_{occurrence_field}",
            )
            start = item.get("start")
            end = item.get("end")
            token = item.get("text")
            _require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(text),
                f"invalid_occurrence_span_{occurrence_field}",
            )
            _require(
                isinstance(token, str) and pattern.fullmatch(token) is not None,
                f"invalid_occurrence_token_{occurrence_field}",
            )
            _require(
                text[start:end] == token,
                f"occurrence_span_text_mismatch_{occurrence_field}",
            )
            annotated.append((start, end, token))

        expected = [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]
        _require(
            sorted(annotated) == expected,
            f"untracked_or_stale_occurrences_{occurrence_field}",
        )

    _require(
        row.get("sum_of_unique_entities") == nonzero_types,
        "sum_of_unique_entities_inconsistent",
    )


def assess_row(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str, str]:
    _require(set(row) == set(EXPECTED_FIELDS), "parquet row field drift")
    source_id = row.get("id")
    _require(
        isinstance(source_id, str) and source_id.isdigit(),
        "source id must be decimal string",
    )
    index = row.get("__index_level_0__")
    _require(
        isinstance(index, int) and not isinstance(index, bool) and index >= 0,
        "source index invalid",
    )
    text = row.get(cfg["selection"]["text_column"])
    _require(isinstance(text, str), "text field must be string")
    _validate_occurrences(row, text)

    if CONTROL_RE.search(text):
        return False, "control_character", ""
    if EMAIL_RE.search(text):
        return False, "email", ""
    if PHONE_RE.search(text):
        return False, "phone", ""

    normalized = normalize(text)
    if len(normalized) < cfg["quality"]["min_chars"]:
        return False, "too_short", ""
    alpha = [char for char in normalized if char.isalpha()]
    cyr = [char for char in alpha if "CYRILLIC" in unicodedata.name(char, "")]
    ratio = len(cyr) / max(1, len(alpha))
    if ratio < cfg["quality"]["min_cyrillic_alpha_ratio"]:
        return False, "low_cyrillic_ratio", ""
    if cfg["quality"]["require_ukrainian_specific_letter"] and not re.search(
        r"[іїєґІЇЄҐ]", normalized
    ):
        return False, "no_ukrainian_specific_letter", ""
    return True, "accepted", normalized


def select_rows(
    rows: list[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    _require(rows, "parquet contains no rows")
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "parquet row must be object")
        _require(set(row) == set(EXPECTED_FIELDS), "parquet row field drift")
        raw_id = row.get("id")
        _require(
            isinstance(raw_id, str) and raw_id.isdigit(),
            "source id must be decimal string",
        )
        numeric_id = int(raw_id)
        _require(numeric_id not in by_id, f"duplicate source id: {raw_id}")
        by_id[numeric_id] = row

    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    for numeric_id in sorted(by_id):
        row = by_id[numeric_id]
        ok, reason, text = assess_row(row, cfg)
        reasons[reason] += 1
        if not ok:
            continue
        digest = _sha256(text.encode("utf-8"))
        if digest in seen_hashes:
            reasons["exact_normalized_duplicate"] += 1
            continue
        seen_hashes.add(digest)
        accepted.append(
            {
                "record_id": str(row["id"]),
                "normalized_sha256": digest,
                "normalized_bytes": len(text.encode("utf-8")),
                "text": text,
            }
        )
        if len(accepted) >= cfg["selection"]["max_records"]:
            break
    return accepted, reasons


def materialize(
    parquet_path: Path,
    output_jsonl: Path,
    report_path: Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    size, digest = _sha256_file(parquet_path)
    _require(size == cfg["source"]["bytes"], "local source byte count mismatch")
    _require(digest == cfg["source"]["sha256"], "local source SHA-256 mismatch")
    try:
        import pyarrow  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RetestError("pyarrow is required for real parquet execution") from exc
    _require(pyarrow.__version__ == PYARROW_VERSION, "PyArrow runtime version drift")
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        raise RetestError("cannot read exact source parquet") from exc
    _require(tuple(table.column_names) == EXPECTED_FIELDS, "parquet schema drift")
    rows = table.to_pylist()
    accepted, reasons = select_rows(rows, cfg)
    _require(accepted, "privacy/quality retest retained zero rows")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(row) + b"\n" for row in accepted)
    output_jsonl.write_bytes(payload)
    report = {
        "schema_version": "12-6.d03-languk-supreme-court-retest-report.v1",
        "source_sha256": cfg["source"]["sha256"],
        "source_bytes": cfg["source"]["bytes"],
        "source_family": cfg["source"]["family"],
        "config_identity_sha256": _sha256(_canonical(dict(cfg))),
        "parquet_columns": table.column_names,
        "parquet_rows": table.num_rows,
        "retained_records": len(accepted),
        "retained_normalized_bytes": sum(row["normalized_bytes"] for row in accepted),
        "retained_jsonl_sha256": _sha256(payload),
        "disposition_counts": dict(sorted(reasons.items())),
        "exact_schema_validated": True,
        "occurrence_spans_validated": True,
        "complete_placeholder_annotation_validated": True,
        "universal_pii_absence_claimed": False,
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
        "safe_result": (
            "LANGUK_SUPREME_COURT_PRIVACY_FILTERED_CANDIDATE_ONLY_"
            "DOWNSTREAM_DEDUP_DECONTAM_REQUIRED"
        ),
    }
    report["report_identity_sha256"] = _sha256(_canonical(report))
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
        download_exact(cfg, args.parquet)
    report = materialize(args.parquet, args.output_jsonl, args.report, cfg)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
