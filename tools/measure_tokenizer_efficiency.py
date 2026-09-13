#!/usr/bin/env python3
"""Measure tokenizer-unit efficiency on an exact JSONL text slice.

Input JSONL rows must contain:
  language: one of uk, en, code
  text: non-empty Unicode string
Optional:
  domain: short label
  reference_subword_tokens: positive integer from a separately identified tokenizer

This tool does not train or select a tokenizer. It produces deterministic unit
measurements that prevent raw UTF-8 byte positions from being compared directly
with source-reported/subword token counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ALLOWED_LANGUAGES = {"uk", "en", "code"}
CONTEXT_BYTES = 1024


class CalibrationError(ValueError):
    """Raised when calibration input is incomplete or unit-ambiguous."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_bucket() -> dict[str, int]:
    return {
        "documents": 0,
        "unicode_codepoints": 0,
        "utf8_bytes": 0,
        "ascii_bytes": 0,
        "reference_subword_tokens": 0,
        "documents_with_reference_subword_tokens": 0,
    }


def _finish_bucket(raw: dict[str, int]) -> dict[str, Any]:
    chars = raw["unicode_codepoints"]
    byte_count = raw["utf8_bytes"]
    ref = raw["reference_subword_tokens"]
    with_ref = raw["documents_with_reference_subword_tokens"]
    documents = raw["documents"]
    require(documents > 0, "cannot summarize an empty bucket")
    require(chars > 0 and byte_count > 0, "text metrics must be non-zero")

    result: dict[str, Any] = dict(raw)
    result["utf8_bytes_per_codepoint"] = round(byte_count / chars, 6)
    result["estimated_codepoints_per_1024_byte_context"] = round(
        CONTEXT_BYTES * chars / byte_count, 3
    )
    result["ascii_byte_fraction"] = round(raw["ascii_bytes"] / byte_count, 6)
    if with_ref == documents:
        require(ref > 0, "reference token count must be positive when complete")
        result["utf8_bytes_per_reference_subword_token"] = round(byte_count / ref, 6)
        result["codepoints_per_reference_subword_token"] = round(chars / ref, 6)
        result["reference_subword_coverage"] = "COMPLETE"
    elif with_ref == 0:
        result["utf8_bytes_per_reference_subword_token"] = None
        result["codepoints_per_reference_subword_token"] = None
        result["reference_subword_coverage"] = "ABSENT"
    else:
        raise CalibrationError(
            "reference_subword_tokens must be supplied for all or none of the rows in a bucket"
        )
    return result


def measure(path: Path, *, corpus_identity: str) -> dict[str, Any]:
    require(path.is_file(), f"input file does not exist: {path}")
    require(bool(corpus_identity.strip()), "corpus_identity must be non-empty")

    buckets: dict[str, dict[str, int]] = defaultdict(_new_bucket)
    total = _new_bucket()
    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"invalid JSON at line {line_number}: {exc}") from exc
            require(isinstance(row, dict), f"line {line_number}: row must be an object")
            language = row.get("language")
            text = row.get("text")
            require(language in ALLOWED_LANGUAGES, f"line {line_number}: invalid language")
            require(isinstance(text, str) and bool(text), f"line {line_number}: text must be non-empty")

            encoded = text.encode("utf-8")
            chars = len(text)
            byte_count = len(encoded)
            ascii_bytes = sum(value < 128 for value in encoded)
            ref = row.get("reference_subword_tokens")
            if ref is not None:
                require(type(ref) is int and ref > 0, f"line {line_number}: reference_subword_tokens must be a positive integer")

            for bucket in (buckets[language], total):
                bucket["documents"] += 1
                bucket["unicode_codepoints"] += chars
                bucket["utf8_bytes"] += byte_count
                bucket["ascii_bytes"] += ascii_bytes
                if ref is not None:
                    bucket["reference_subword_tokens"] += ref
                    bucket["documents_with_reference_subword_tokens"] += 1
            rows += 1

    require(rows > 0, "input JSONL contains no records")
    missing = sorted(ALLOWED_LANGUAGES - set(buckets))
    require(not missing, "calibration slice must contain uk, en and code strata; missing: " + ",".join(missing))

    return {
        "schema": "12-6.tokenizer-efficiency-report.v1",
        "input": {
            "path": str(path),
            "sha256": sha256_file(path),
            "corpus_identity": corpus_identity,
            "rows": rows,
        },
        "current_tokenizer": {
            "version": "s0-byte-v1",
            "type": "utf8-byte",
            "loss_position_unit": "UTF8_BYTE_POSITION",
            "context_bytes": CONTEXT_BYTES,
        },
        "unit_policy": {
            "direct_byte_to_subword_scaling_law_conversion_allowed": False,
            "reference_subword_counts_are_measurements_only": True,
        },
        "by_language": {name: _finish_bucket(buckets[name]) for name in sorted(buckets)},
        "overall": _finish_bucket(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--corpus-identity", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = measure(args.input_jsonl, corpus_identity=args.corpus_identity)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
