"""Create a metadata-only Gitleaks report that is safe to retain in CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _integration_bootstrap import load_integration_module

ROOT = Path(__file__).resolve().parents[1]
_SECRET_HISTORY = load_integration_module(ROOT, "secret_history")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip all secret-bearing fields from a Gitleaks JSON report."
    )
    parser.add_argument("raw_report", type=Path)
    parser.add_argument("sanitized_report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = json.loads(args.raw_report.read_text(encoding="utf-8"))
        sanitized = _SECRET_HISTORY.sanitize_gitleaks_findings(payload)
    except (OSError, json.JSONDecodeError, _SECRET_HISTORY.SecretHistoryReportError) as exc:
        print(f"gitleaks_report_sanitize=fail reason={exc}")
        return 1

    args.sanitized_report.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "gitleaks_report_sanitize=pass "
        f"finding_count={sanitized['finding_count']} "
        f"output={args.sanitized_report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
