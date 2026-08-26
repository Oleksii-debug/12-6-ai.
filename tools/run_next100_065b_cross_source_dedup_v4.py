#!/usr/bin/env python3
"""Run or verify NEXT100-065B converged cross-source dedup V4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_capacity_audit_v4 import audit_live, verify_report, write_report


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--base-inventory", required=True)
    run.add_argument("--extension", required=True)
    run.add_argument("--report", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "run":
        report = audit_live(_load(args.base_inventory), _load(args.extension))
        verify_report(report)
        write_report(report, args.report)
        return 0

    verify_report(_load(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
