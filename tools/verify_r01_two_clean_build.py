"""Verify two independently produced R01 build roots byte-for-byte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.two_clean_build import (
    canonical_json_bytes,
    compare_clean_builds,
    validate_report,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--root-a", type=Path, required=True)
    parser.add_argument("--root-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        print("refusing to overwrite existing output")
        return 2

    result = compare_clean_builds(
        _load_json(args.binding),
        args.root_a,
        args.root_b,
    )
    if not result.identical or result.report is None:
        print(json.dumps({"status": "BLOCKED", "blockers": list(result.blockers)}))
        return 1

    report_errors = validate_report(result.report)
    if report_errors:
        print(json.dumps({"status": "BLOCKED", "blockers": report_errors}))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result.report))
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "build_tree_sha256": result.report["build_tree_sha256"],
                "report_sha256": result.report["report_sha256"],
                "file_count": result.report["file_count"],
                "total_bytes": result.report["total_bytes"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
