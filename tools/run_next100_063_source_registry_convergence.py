#!/usr/bin/env python3
"""Run or verify NEXT100-063 scoped source-registry convergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.source_registry_convergence_v1 import (
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
    run.add_argument("--config", required=True)
    run.add_argument("--repo-root", default=".")
    run.add_argument("--report", required=True)

    verify = subparsers.add_parser("verify")
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
