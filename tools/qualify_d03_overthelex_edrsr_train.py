#!/usr/bin/env python3
"""Zero-credit qualification for immutable overthelex EDRSR train facts only."""
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

CONFIG = Path("configs/data/d03_overthelex_edrsr_train_candidate_v1.json")
SCHEMA_VERSION = "12-6.d03-overthelex-edrsr-train-candidate.v1"
DATASET = "overthelex/ua-case-outcome"
REVISION = "a175ffa122b1f5b04d25ed415a30a925d692b5a9"
SOURCE_FILE = "train.parquet"
SOURCE_SHA256 = "e5706a2a35a0131224dc75eac8345296e11518688c4072dc996b25a55a5b7793"
SOURCE_BYTES = 49_826_322
SOURCE_ROWS = 11_561
SOURCE_FAMILY = "ua.edrsr.case-outcome.train"
PYARROW_VERSION = "17.0.0"
MAX_CANDIDATE_BYTES = 6_000_000
EXPECTED_FIELDS = (
    "doc_id",
    "justice_kind",
    "justice_kind_name",
    "judgment_code",
    "category_code",
    "court_code",
    "judge",
    "adjudication_date",
    "facts",
    "dispositive",
    "outcome",
    "epoch",
    "year",
    "full_text_length",
)
EXCLUDED_SURFACES = {
    "validation": {
        "file": "validation.parquet",
        "sha256": "45559c8e6465f6ef7909674ca94edf2a3a917c60c864d7afde11256fa1cd7d27",
        "bytes": 6_412_051,
    },
    "test": {
        "file": "test.parquet",
        "sha256": "c569b8c6c6a386e75951c5bfc66b436e7e0606a3b55ad642e988d22b9c65654f",
        "bytes": 6_283_692,
    },
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LEGACY_MARKER_RE = re.compile(r"(?:ОСОБА|АДРЕСА|ІНФОРМАЦІЯ|НОМЕР)_[1-9][0-9]*")


class QualificationError(RuntimeError):
    """Fail-closed source qualification error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise QualificationError(f"cannot read source parquet: {path}") from exc
    return size, digest.hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot load config: {path}") from exc
    _require(isinstance(cfg, dict), "config root must be an object")
    _require(cfg.get("schema_version") == SCHEMA_VERSION, "schema drift")
    _require(cfg.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    source = cfg.get("source")
    _require(isinstance(source, Mapping), "source contract missing")
    expected_source = {
        "dataset": DATASET,
        "revision": REVISION,
        "file": SOURCE_FILE,
        "split": "train",
        "url": (
            "https://huggingface.co/datasets/overthelex/ua-case-outcome/resolve/"
            f"{REVISION}/{SOURCE_FILE}"
        ),
        "sha256": SOURCE_SHA256,
        "bytes": SOURCE_BYTES,
        "rows": SOURCE_ROWS,
        "license": "CC-BY-4.0",
        "attribution": (
            "Volodymyr Ovcharov / overthelex; source: Unified State Register "
            "of Court Decisions of Ukraine (EDRSR)"
        ),
        "family": SOURCE_FAMILY,
    }
    for key, expected in expected_source.items():
        _require(source.get(key) == expected, f"source identity drift: {key}")

    excluded = cfg.get("excluded_evaluation_surfaces")
    _require(isinstance(excluded, list) and len(excluded) == 2, "evaluation surface drift")
    observed_excluded = {row.get("split"): row for row in excluded if isinstance(row, Mapping)}
    _require(set(observed_excluded) == set(EXCLUDED_SURFACES), "excluded split set drift")
    for split, expected in EXCLUDED_SURFACES.items():
        row = observed_excluded[split]
        for key, value in expected.items():
            _require(row.get(key) == value, f"excluded {split} identity drift: {key}")

    selection = cfg.get("selection")
    _require(isinstance(selection, Mapping), "selection contract missing")
    _require(selection.get("text_column") == "facts", "text column drift")
    _require(selection.get("id_column") == "doc_id", "id column drift")
    _require(
        selection.get("order") == "numeric_doc_id_ascending_after_privacy_quality_pass",
        "selection order drift",
    )
    _require(
        selection.get("max_candidate_normalized_bytes") == MAX_CANDIDATE_BYTES,
        "candidate byte cap drift",
    )

    schema = cfg.get("parquet_schema")
    _require(isinstance(schema, Mapping), "parquet schema contract missing")
    _require(schema.get("required_fields") == list(EXPECTED_FIELDS), "parquet field drift")
    _require(schema.get("reject_extra_fields") is True, "extra-field rejection disabled")

    privacy = cfg.get("privacy")
    _require(isinstance(privacy, Mapping), "privacy contract missing")
    required_privacy = {
        "source_claims_personal_data_anonymized": True,
        "reject_legacy_anonymization_markers": True,
        "reject_email": True,
        "reject_phone": True,
        "reject_control_characters": True,
        "universal_pii_absence_claimed": False,
        "rejected_text_emitted": False,
        "rejected_hashes_emitted": False,
    }
    for key, expected in required_privacy.items():
        _require(privacy.get(key) is expected, f"privacy boundary drift: {key}")
    _require(
        privacy.get("normalized_placeholder_tokens")
        == ["[PERSON]", "[ADDRESS]", "[NUMBER]", "[INFO]"],
        "placeholder token contract drift",
    )

    quality = cfg.get("quality")
    _require(isinstance(quality, Mapping), "quality contract missing")
    _require(quality.get("min_chars") == 200, "min_chars drift")
    _require(quality.get("min_cyrillic_alpha_ratio") == 0.70, "Cyrillic threshold drift")
    _require(quality.get("require_ukrainian_specific_letter") is True, "UA letter gate disabled")

    evaluation = cfg.get("evaluation_boundary")
    _require(isinstance(evaluation, Mapping), "evaluation boundary missing")
    _require(evaluation.get("source_has_benchmark_intended_use") is True, "benchmark flag drift")
    _require(evaluation.get("training_split_only") is True, "training split boundary drift")
    _require(evaluation.get("excluded_splits") == ["validation", "test"], "excluded split order drift")
    _require(
        evaluation.get("reserved_evaluation_decontamination_required") is True,
        "reserved evaluation decontamination requirement disabled",
    )
    _require(evaluation.get("final_test_payload_accessed") is False, "final-test payload boundary drift")
    _require(evaluation.get("final_test_outcomes_read") is False, "final-test outcome boundary drift")

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
        "research_corpus_v1_released",
        "paid_compute_used",
    ):
        _require(boundary.get(key) is False, f"false boundary drift: {key}")
    return cfg


def download_exact(cfg: Mapping[str, Any], output: Path) -> None:
    source = cfg["source"]
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "12-6-D03-EDRSR-Qualification/1"}
    )
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as dst:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                _require(total <= source["bytes"], "download exceeds pinned byte count")
                digest.update(chunk)
                dst.write(chunk)
        _require(total == source["bytes"], f"source bytes mismatch: {total}")
        _require(digest.hexdigest() == source["sha256"], "source SHA-256 mismatch")
        partial.replace(output)
    except OSError as exc:
        raise QualificationError("exact source download failed") from exc
    finally:
        if partial.exists():
            partial.unlink()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def assess_row(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str, str]:
    _require(set(row) == set(EXPECTED_FIELDS), "parquet row field drift")
    doc_id = row.get("doc_id")
    _require(
        isinstance(doc_id, int) and not isinstance(doc_id, bool) and doc_id > 0,
        "doc_id must be positive integer",
    )
    facts = row.get("facts")
    _require(isinstance(facts, str), "facts field must be string")

    if CONTROL_RE.search(facts):
        return False, "control_character", ""
    if EMAIL_RE.search(facts):
        return False, "email", ""
    if PHONE_RE.search(facts):
        return False, "phone", ""
    if LEGACY_MARKER_RE.search(facts):
        return False, "legacy_anonymization_marker", ""

    normalized = normalize(facts)
    if len(normalized) < cfg["quality"]["min_chars"]:
        return False, "too_short", ""
    alpha = [char for char in normalized if char.isalpha()]
    cyrillic = [char for char in alpha if "CYRILLIC" in unicodedata.name(char, "")]
    ratio = len(cyrillic) / max(1, len(alpha))
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
        doc_id = row.get("doc_id")
        _require(
            isinstance(doc_id, int) and not isinstance(doc_id, bool) and doc_id > 0,
            "doc_id must be positive integer",
        )
        _require(doc_id not in by_id, f"duplicate doc_id: {doc_id}")
        by_id[doc_id] = row

    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    retained_bytes = 0
    for doc_id in sorted(by_id):
        ok, reason, text = assess_row(by_id[doc_id], cfg)
        if not ok:
            reasons[reason] += 1
            continue
        digest = _sha256(text.encode("utf-8"))
        if digest in seen_hashes:
            reasons["exact_normalized_duplicate"] += 1
            continue
        payload_bytes = len(text.encode("utf-8"))
        if retained_bytes + payload_bytes > cfg["selection"]["max_candidate_normalized_bytes"]:
            reasons["candidate_byte_cap_reached"] += 1
            break
        seen_hashes.add(digest)
        retained_bytes += payload_bytes
        accepted.append(
            {
                "record_id": str(doc_id),
                "source_family": SOURCE_FAMILY,
                "language": "uk",
                "modality": "text",
                "normalized_sha256": digest,
                "normalized_bytes": payload_bytes,
                "text": text,
            }
        )
        reasons["accepted"] += 1
    return accepted, reasons


def materialize(
    parquet_path: Path,
    output_jsonl: Path,
    report_path: Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    size, digest = _sha256_file(parquet_path)
    _require(size == SOURCE_BYTES, "local source byte count mismatch")
    _require(digest == SOURCE_SHA256, "local source SHA-256 mismatch")
    try:
        import pyarrow  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise QualificationError("pyarrow is required for real parquet execution") from exc
    _require(pyarrow.__version__ == PYARROW_VERSION, "PyArrow runtime version drift")
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        raise QualificationError("cannot read exact source parquet") from exc
    _require(tuple(table.column_names) == EXPECTED_FIELDS, "parquet schema drift")
    _require(table.num_rows == SOURCE_ROWS, "parquet row-count drift")
    accepted, reasons = select_rows(table.to_pylist(), cfg)
    _require(accepted, "qualification retained zero candidate rows")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(row) + b"\n" for row in accepted)
    output_jsonl.write_bytes(payload)
    report = {
        "schema_version": "12-6.d03-overthelex-edrsr-train-candidate-report.v1",
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "source_rows": SOURCE_ROWS,
        "source_family": SOURCE_FAMILY,
        "source_license": "CC-BY-4.0",
        "training_split_only": True,
        "excluded_evaluation_splits": ["validation", "test"],
        "retained_records": len(accepted),
        "retained_normalized_bytes": sum(row["normalized_bytes"] for row in accepted),
        "retained_jsonl_sha256": _sha256(payload),
        "disposition_counts": dict(sorted(reasons.items())),
        "facts_column_only": True,
        "dispositive_emitted": False,
        "outcome_emitted": False,
        "judge_metadata_emitted": False,
        "universal_pii_absence_claimed": False,
        "rejected_text_emitted": False,
        "rejected_hashes_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_authorized_bytes": 0,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_payload_accessed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "safe_result": "EDRSR_TRAIN_FACTS_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED",
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
