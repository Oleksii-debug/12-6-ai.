from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.precision_learning import run_precision_learning_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired TRAIN-60 CPU precision learning curves")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence = run_precision_learning_experiment(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
    )
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "output": str(args.output),
        "actual_tokens": evidence["profiles"]["fp32"]["optimized_tokens"],
        "fp32_final_validation_bpb": evidence["profiles"]["fp32"]["curve"][-1][
            "validation"
        ]["bpb"],
        "bf16_final_validation_bpb": evidence["profiles"]["bf16"]["curve"][-1][
            "validation"
        ]["bpb"],
        "within_tolerance": evidence["comparison"]["within_tolerance"],
        "recommendation": evidence["recommendation"]["decision"],
        "evidence_sha256": evidence["evidence_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
