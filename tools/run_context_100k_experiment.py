#!/usr/bin/env python3
"""Run or compare MODEL-17 100K context conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.context_100k_experiment import (
    compare_context_conditions,
    run_context_condition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/model17_context_100k.json"),
    )
    parser.add_argument("--context", type=int, choices=(128, 256))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-128", type=Path)
    parser.add_argument("--compare-256", type=Path)
    args = parser.parse_args()

    if args.compare_128 or args.compare_256:
        if not args.compare_128 or not args.compare_256 or args.context is not None:
            parser.error("comparison requires both --compare-128 and --compare-256 only")
        left = json.loads(args.compare_128.read_text(encoding="utf-8"))
        right = json.loads(args.compare_256.read_text(encoding="utf-8"))
        report = compare_context_conditions(left, right)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        if args.source_sha is None or args.context is None:
            parser.error("condition run requires --source-sha and --context")
        report = run_context_condition(
            repo_root=Path(".").resolve(),
            source_sha=args.source_sha,
            config_path=args.config,
            context_length=args.context,
            output_path=args.output,
        )
    print(json.dumps({
        "report_sha256": report.get("report_sha256", report.get("result_sha256")),
        "recommendation": report.get("recommendation"),
        "context_length": report.get("context_length"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
