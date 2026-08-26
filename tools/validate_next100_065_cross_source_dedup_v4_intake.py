#!/usr/bin/env python3
"""Validate the NEXT100-065 V4 object-level dedup intake snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_dedup_v4_intake import evaluate_v4_intake

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/next100_065_cross_source_dedup_v4_intake_v1.json"
DEFAULT_CONVERGENCE = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--expect-status")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    convergence = json.loads(args.convergence.read_text(encoding="utf-8"))
    report = evaluate_v4_intake(config, convergence)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    if args.expect_status is not None and report["status"] != args.expect_status:
        print(f"expected status {args.expect_status}, got {report['status']}")
        return 2
    if args.require_ready and not report["ready_for_global_dedup_object_comparison"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
