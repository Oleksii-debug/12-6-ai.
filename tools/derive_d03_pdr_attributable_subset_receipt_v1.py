#!/usr/bin/env python3
"""Derive a text-free PDR attributable-subset receipt from exact #1076 replay files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.d03_pdr_attributable_subset import PdrSubsetError, derive_exact_replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--attribution-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = derive_exact_replay(args.candidate, args.sidecar, args.attribution_report)
    except (OSError, UnicodeError, json.JSONDecodeError, PdrSubsetError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
