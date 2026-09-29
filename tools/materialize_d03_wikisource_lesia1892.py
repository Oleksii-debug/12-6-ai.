#!/usr/bin/env python3
"""Bounded LOCAL_FREE current-main materializer for the qualified Lesia 1892 Wikisource edition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from twelve_six.data.wikisource_pd_contract import (
    WikisourceIntakeError,
    validate_control_contract,
)
from twelve_six.data.wikisource_pd_edition import materialize_live


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/data/d03_wikisource_lesia1892_current_main_v1.json"


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WikisourceIntakeError("control contract has duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise WikisourceIntakeError(f"control contract has non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise WikisourceIntakeError("control contract has non-finite JSON number")
    return parsed


def load_strict_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise WikisourceIntakeError(f"invalid control contract JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise WikisourceIntakeError("control contract must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=112)
    args = parser.parse_args()
    try:
        contract = load_strict_json_object(args.contract.read_text(encoding="utf-8"))
        validate_control_contract(contract)
        result = materialize_live(max_pages=args.max_pages)
    except (OSError, ValueError, WikisourceIntakeError) as exc:
        parser.error(str(exc))
    args.candidate_out.write_bytes(result.candidate_jsonl)
    args.report_out.write_text(
        json.dumps(result.report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
