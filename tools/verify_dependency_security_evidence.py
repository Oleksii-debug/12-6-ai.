#!/usr/bin/env python3
"""Verify retained dependency-security evidence without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integration.dependency_security import validate_security_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-age-hours", type=float, default=168.0)
    parser.add_argument("--require-no-review-findings", action="store_true")
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    validated = validate_security_evidence(
        root=args.root,
        evidence=evidence,
        expected_source_sha=args.source_sha,
        max_age_hours=args.max_age_hours,
    )
    print(f"evidence_sha256={validated['evidence_sha256']}")
    print(f"status={validated['status']}")
    if (
        args.require_no_review_findings
        and validated["status"] != "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
