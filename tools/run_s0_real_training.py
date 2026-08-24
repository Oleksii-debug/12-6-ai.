"""Execute exact S0 real-training evidence and write canonical JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_evidence import run_s0_training_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    evidence = run_s0_training_evidence(
        root,
        source_sha=args.source_sha,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "identity_sha256": evidence["identity_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "optimized_tokens": evidence["training"]["optimized_tokens"],
        "initial_train_loss": evidence["training"]["initial_train_loss"],
        "final_train_loss": evidence["training"]["final_train_loss"],
        "initial_validation_loss": evidence["training"]["initial_validation_loss"],
        "final_validation_loss": evidence["training"]["final_validation_loss"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
