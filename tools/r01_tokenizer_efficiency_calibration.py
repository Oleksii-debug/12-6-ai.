#!/usr/bin/env python3
"""Deterministic tokenizer-efficiency calibration for R01.

This tool measures byte/token geometry only. It never derives a training budget,
authorizes compute, or reads evaluation outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

STRATA = ("UA", "EN", "CODE")
SCHEMA = "12-6.r01-tokenizer-efficiency-calibration.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def calibrate(payload: dict[str, Any]) -> dict[str, Any]:
    corpus_identity = payload.get("corpus_identity")
    tokenizer_identity = payload.get("tokenizer_identity")
    records = payload.get("records")

    if not isinstance(corpus_identity, str) or not corpus_identity.strip():
        raise ValueError("corpus_identity must be a non-empty string")
    if not isinstance(tokenizer_identity, str) or not tokenizer_identity.strip():
        raise ValueError("tokenizer_identity must be a non-empty string")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen: set[str] = set()
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"utf8_bytes": 0, "content_tokens": 0})

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        record_id = record.get("record_id")
        stratum = record.get("stratum")
        utf8_bytes = record.get("utf8_bytes")
        content_tokens = record.get("content_tokens")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"records[{index}].record_id must be a non-empty string")
        if record_id in seen:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        if stratum not in STRATA:
            raise ValueError(f"records[{index}].stratum must be one of {STRATA}")
        if isinstance(utf8_bytes, bool) or not isinstance(utf8_bytes, int) or utf8_bytes <= 0:
            raise ValueError(f"records[{index}].utf8_bytes must be a positive integer")
        if (
            isinstance(content_tokens, bool)
            or not isinstance(content_tokens, int)
            or content_tokens <= 0
        ):
            raise ValueError(f"records[{index}].content_tokens must be a positive integer")

        totals[stratum]["utf8_bytes"] += utf8_bytes
        totals[stratum]["content_tokens"] += content_tokens

    missing = [stratum for stratum in STRATA if stratum not in totals]
    if missing:
        raise ValueError(f"all UA/EN/CODE strata are required; missing: {','.join(missing)}")

    by_stratum: dict[str, dict[str, float | int]] = {}
    overall_bytes = 0
    overall_tokens = 0
    for stratum in STRATA:
        raw_bytes = totals[stratum]["utf8_bytes"]
        tokens = totals[stratum]["content_tokens"]
        overall_bytes += raw_bytes
        overall_tokens += tokens
        by_stratum[stratum] = {
            "records": sum(1 for item in records if item["stratum"] == stratum),
            "utf8_bytes": raw_bytes,
            "content_tokens": tokens,
            "bytes_per_content_token": raw_bytes / tokens,
            "content_tokens_per_1000_utf8_bytes": tokens * 1000.0 / raw_bytes,
        }

    result = {
        "schema": SCHEMA,
        "corpus_identity": corpus_identity,
        "tokenizer_identity": tokenizer_identity,
        "record_count": len(records),
        "by_stratum": by_stratum,
        "overall": {
            "utf8_bytes": overall_bytes,
            "content_tokens": overall_tokens,
            "bytes_per_content_token": overall_bytes / overall_tokens,
            "content_tokens_per_1000_utf8_bytes": overall_tokens * 1000.0 / overall_bytes,
        },
        "truth_boundary": {
            "measurement_only": True,
            "training_budget_derived": False,
            "long_training_authorized": False,
            "paid_compute_authorized": False,
            "semantic_context_equivalence_claimed": False,
            "flop_equivalence_claimed": False,
        },
    }
    result["result_identity_sha256"] = _sha256(result)
    return result


def main(argv: list[str]) -> int:
    if len(argv) not in {2, 3}:
        print("usage: r01_tokenizer_efficiency_calibration.py INPUT.json [OUTPUT.json]")
        return 2
    payload = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print("FAIL: input root must be an object")
        return 1
    try:
        result = calibrate(payload)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if len(argv) == 3:
        Path(argv[2]).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
