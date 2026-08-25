"""Run one canonical S2 or S3 scale-execution evidence probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s2_s3_scale_execution import run_scale_execution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("S2", "S3"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence = run_scale_execution(
        Path.cwd(),
        stage_name=args.stage,
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "authority": evidence["authority"],
                "stage": evidence["identity"]["stage"],
                "source_sha": evidence["identity"]["source_sha"],
                "parameter_count": evidence["identity"]["parameter_count"],
                "optimizer_steps": evidence["training"]["optimizer_steps"],
                "optimized_tokens": evidence["training"]["optimized_tokens"],
                "wall_seconds_training_only": evidence["runtime"][
                    "wall_seconds_training_only"
                ],
                "optimized_tokens_per_wall_second": evidence["runtime"][
                    "optimized_tokens_per_wall_second"
                ],
                "observed_tensor_bytes_with_snapshot": evidence["resources"][
                    "observed_tensor_bytes_with_snapshot"
                ],
                "evidence_sha256": evidence["evidence_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
