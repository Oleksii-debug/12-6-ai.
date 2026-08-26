#!/usr/bin/env python3
"""Run or verify the NEXT100-071 successor global cross-source dedup audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.successor_cross_source_dedup import (
    audit_successor_live,
    verify_successor_report,
    write_successor_report,
)


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--base-inventory", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--pdftotext", default="pdftotext")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "run":
        report = audit_successor_live(
            _load(args.base_inventory),
            _load(args.config),
            pdftotext=args.pdftotext,
        )
        verify_successor_report(report)
        write_successor_report(report, args.report)
        return 0

    verify_successor_report(_load(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
