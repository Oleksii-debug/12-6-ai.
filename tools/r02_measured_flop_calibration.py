#!/usr/bin/env python3
"""Build or verify R02 measured-FLOP equal-budget evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.measured_flop_calibration import (
    build_report,
    load_json,
    validate_contract,
    verify_report,
)


def _write(value: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-contract")
    validate.add_argument("--contract", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--contract", required=True)
    analyze.add_argument("--measurement", action="append", default=[])
    analyze.add_argument("--report", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "validate-contract":
        validate_contract(load_json(args.contract))
        return 0
    if args.command == "analyze":
        contract = load_json(args.contract)
        measurements = [load_json(path) for path in args.measurement]
        report = build_report(contract, measurements)
        verify_report(report)
        _write(report, args.report)
        return 0
    verify_report(load_json(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
