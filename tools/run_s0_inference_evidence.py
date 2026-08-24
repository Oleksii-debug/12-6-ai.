from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.s0_evidence import collect_s0_trained_inference_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retained trained-checkpoint S0 inference evidence."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = collect_s0_trained_inference_evidence(
        args.repo_root,
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "source_sha": evidence["identity"]["source_sha"],
                "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
                "evidence_sha256": evidence["evidence_sha256"],
                "parity": evidence["parity"]["passed"],
                "seeded_sampling_repeatable": evidence["generation"][
                    "seeded_sampling"
                ]["repeatable"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
