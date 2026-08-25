"""Run and validate the TRAIN-49 AdamW epsilon experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.adam_epsilon_experiment import run_adam_epsilon_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True)
    parser.add_argument(
        "--plan",
        default="configs/runs/adam_epsilon_500k.experimental.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    locked_environment = json.loads(
        Path(args.locked_environment_evidence).read_text(encoding="utf-8")
    )
    evidence, summary = run_adam_epsilon_experiment(
        root,
        source_sha=args.source_sha,
        locked_environment_evidence=locked_environment,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
