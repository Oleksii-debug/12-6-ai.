"""Verify retained dependency-security evidence without network access."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

from _dependency_contract_loader import load_dependency_contracts

_, SECURITY = load_dependency_contracts()
RELEASE = importlib.import_module("twelve_six.integration.release_fixture")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--max-age-hours", type=float, default=168.0)
    parser.add_argument("--require-no-review-findings", action="store_true")
    args = parser.parse_args()
    actual_sha = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != args.source_sha:
        raise ValueError("evidence source SHA does not match checked-out HEAD")

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    validated = SECURITY.validate_security_evidence(
        root=args.root,
        evidence=evidence,
        expected_source_sha=args.source_sha,
        max_age_hours=args.max_age_hours,
    )
    print(f"evidence_sha256={validated['evidence_sha256']}")
    sbom = SECURITY.build_lock_sbom(root=args.root, source_sha=args.source_sha)
    notices = json.loads(args.notices.read_text(encoding="utf-8"))
    RELEASE.verify_notice_inventory(sbom, notices)
    print(f"notices_sha256={notices['notices_sha256']}")
    print(f"status={validated['status']}")
    if (
        args.require_no_review_findings
        and validated["status"] != "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
