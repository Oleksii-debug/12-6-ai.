#!/usr/bin/env python3
"""Execute and retain TRAIN-43/45 stability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.stability_schedule_experiments import (
    run_stability_schedule_experiments,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True)
    parser.add_argument(
        "--plan",
        default="configs/runs/train43_45_stability.experimental.json",
    )
    parser.add_argument("--output", default="train43-45-stability-evidence.json")
    parser.add_argument("--summary-output", default="train43-45-stability-summary.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    environment = json.loads(
        Path(args.locked_environment_evidence).read_text(encoding="utf-8")
    )
    evidence, summary = run_stability_schedule_experiments(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=environment,
        plan_path=args.plan,
    )
    Path(args.output).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
