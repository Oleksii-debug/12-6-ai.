#!/usr/bin/env python3
"""Execute the MODEL-16 matched-capacity tied-vs-untied experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.weight_tying_experiment import run_weight_tying_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/model16_weight_tying_500k.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model16/weight_tying_500k.json"),
    )
    args = parser.parse_args()
    report = run_weight_tying_experiment(
        repo_root=Path(".").resolve(),
        source_sha=args.source_sha,
        config_path=args.config,
        output_path=args.output,
    )
    summary = {
        "report_sha256": report["report_sha256"],
        "parameter_match": report["parameter_match"],
        "aggregates": report["aggregates"],
        "recommendation": report["recommendation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
