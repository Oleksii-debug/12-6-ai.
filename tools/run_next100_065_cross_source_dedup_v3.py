#!/usr/bin/env python3
"""Run or verify the NEXT100-065 cross-source deduplication V3 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_capacity_audit_v3 import (
    audit_live,
    verify_report,
    write_report,
)


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--inventory", required=True)
    run.add_argument("--report", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "run":
        report = audit_live(_load(args.inventory))
        verify_report(report)
        write_report(report, args.report)
        return 0
    report = _load(args.report)
    verify_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
