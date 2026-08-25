#!/usr/bin/env python3
"""Compact DATA-105 retained evidence without changing analytical authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    analysis = report.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("DATA-105 report analysis must be an object")
    taxonomy = analysis.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise ValueError("DATA-105 report taxonomy must be an object")
    records = taxonomy.get("records")
    if not isinstance(records, list):
        raise ValueError("DATA-105 taxonomy records must be a list before compaction")
    expected_taxonomy_sha = analysis.get("taxonomy_sha256")
    actual_taxonomy_sha = canonical_sha256(taxonomy)
    if expected_taxonomy_sha != actual_taxonomy_sha:
        raise ValueError("taxonomy_sha256 does not bind the full pre-compaction taxonomy")

    analysis["taxonomy"] = {
        "schema_version": taxonomy.get("schema_version"),
        "derivation": taxonomy.get("derivation"),
        "record_count": len(records),
        "records_omitted_from_retained_report": True,
        "full_taxonomy_sha256": actual_taxonomy_sha,
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = compact_report(args.report)
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "taxonomy_sha256": report["analysis"]["taxonomy_sha256"],
                "taxonomy_record_count": report["analysis"]["taxonomy"]["record_count"],
                "selected_policy": report["recommendation"]["selected_policy"],
                "verdict": report["recommendation"]["verdict"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
