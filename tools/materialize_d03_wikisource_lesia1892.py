#!/usr/bin/env python3
"""Bounded LOCAL_FREE current-main materializer for the qualified Lesia 1892 Wikisource edition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.wikisource_pd_contract import (
    WikisourceIntakeError,
    validate_control_contract,
)
from twelve_six.data.wikisource_pd_edition import materialize_live


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/data/d03_wikisource_lesia1892_current_main_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=112)
    args = parser.parse_args()
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise WikisourceIntakeError("control contract must be a JSON object")
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
