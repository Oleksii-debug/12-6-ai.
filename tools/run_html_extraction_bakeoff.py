#!/usr/bin/env python3
"""Execute and validate the SWARM-742 HTML extraction bake-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.html_extraction_bakeoff import load_contract, run_bakeoff, validate_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        default="configs/research/html_extraction_bakeoff_v1.json",
        help="Frozen benchmark contract JSON",
    )
    parser.add_argument("--output", help="Optional evidence JSON output path")
    parser.add_argument(
        "--allow-retest",
        action="store_true",
        help="Return zero for a truthful RETEST state (useful for constrained local environments)",
    )
    args = parser.parse_args()

    contract = load_contract(args.contract)
    report = run_bakeoff(contract)
    validate_report(report, contract)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if report["terminal_state"].startswith("RETEST_") and not args.allow_retest:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
