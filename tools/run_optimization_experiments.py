"""Execute the committed next-scale optimization experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.optimization_experiments import run_optimization_experiments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/runs/optimizer_experiments.experimental.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence, summary = run_optimization_experiments(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        plan_path=args.plan,
        seed=args.seed,
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "authority": evidence["authority"],
                "source_sha": evidence["identity"]["source_sha"],
                "evidence_sha256": evidence["evidence_sha256"],
                "output": str(args.output),
                "summary_output": str(args.summary_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
