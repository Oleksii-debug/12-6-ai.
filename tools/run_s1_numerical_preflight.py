#!/usr/bin/env python3
"""Run the current S1 engineering-candidate numerical preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s1_preflight import run_s1_numerical_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence = run_s1_numerical_preflight(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "authority": evidence["authority"],
                "source_sha": evidence["identity"]["source_sha"],
                "parameter_count": evidence["identity"]["parameter_count"],
                "fp32_final_validation_loss": evidence["profiles"]["fp32"][
                    "final_validation_loss"
                ],
                "bf16_final_validation_loss": evidence["profiles"]["bf16"][
                    "final_validation_loss"
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
