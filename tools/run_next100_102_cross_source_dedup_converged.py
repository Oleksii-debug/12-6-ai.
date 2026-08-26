#!/usr/bin/env python3
"""Run or verify NEXT100-102 converged global cross-source dedup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_capacity_audit_v4 import audit_live, verify_report, write_report


def _load(path: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--repo-root", default=".")
    verify = sub.add_parser("verify")
    verify.add_argument("--report", required=True)
    args = parser.parse_args()

    if args.command == "run":
        report = audit_live(_load(args.config), repo_root=args.repo_root)
        verify_report(report)
        write_report(report, args.report)
        return 0
    verify_report(_load(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
