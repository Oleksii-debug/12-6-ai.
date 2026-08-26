"""Run or verify the DATA-298 cross-source capacity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_capacity_audit import audit_live, verify_report, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--inventory", required=True)
    run.add_argument("--report", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "run":
        inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        report = audit_live(inventory)
        verify_report(report)
        write_report(report, args.report)
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "source_count": report["source_count"],
                    "report_sha256": report["report_sha256"],
                    "scopes": report["scopes"],
                },
                sort_keys=True,
            )
        )
        return 0

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    verify_report(report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
