#!/usr/bin/env python3
"""Sanitize a Gitleaks JSON report without retaining secret material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_SCHEMA = "12-6.gitleaks-sanitized-findings.v1"
_SAFE_FIELDS = (
    "RuleID",
    "Description",
    "File",
    "StartLine",
    "EndLine",
    "Commit",
    "Author",
    "Date",
    "Message",
    "Tags",
    "Fingerprint",
)
_FORBIDDEN_FIELDS = {"Secret", "Match"}


def sanitize_findings(payload: Any) -> dict[str, Any]:
    """Return metadata-only findings and reject malformed report shapes."""
    if not isinstance(payload, list):
        raise ValueError("Gitleaks report must be a JSON array")

    sanitized: list[dict[str, Any]] = []
    for index, finding in enumerate(payload):
        if not isinstance(finding, dict):
            raise ValueError(f"finding {index} must be a JSON object")
        if not _FORBIDDEN_FIELDS.issubset(finding):
            missing = sorted(_FORBIDDEN_FIELDS.difference(finding))
            raise ValueError(f"finding {index} is missing expected sensitive fields: {missing}")

        item = {field: finding[field] for field in _SAFE_FIELDS if field in finding}
        if not item.get("RuleID") or not item.get("File") or not item.get("Commit"):
            raise ValueError(f"finding {index} lacks rule/file/commit identity")
        sanitized.append(item)

    return {
        "schema": _SCHEMA,
        "finding_count": len(sanitized),
        "findings": sanitized,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    sanitized = sanitize_findings(payload)
    args.output.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"sanitized_gitleaks_findings={sanitized['finding_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
