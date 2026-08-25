#!/usr/bin/env python3
"""Execute MODEL-19 initialization seed matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.init_seed_experiment import run_init_seed_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/model19_init_seeds_100k_500k.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model19/init_seeds_100k_500k.json"),
    )
    args = parser.parse_args()
    report = run_init_seed_experiment(
        repo_root=Path(".").resolve(),
        source_sha=args.source_sha,
        config_path=args.config,
        output_path=args.output,
    )
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "recommendation": report["recommendation"],
        "classifications": report["classifications"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
