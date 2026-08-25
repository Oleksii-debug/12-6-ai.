#!/usr/bin/env python3
"""Canonical LEARN-01 entry point: seed model initialization before importing the run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from twelve_six.real_100k_training import run_experiment, validate_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    # TwelveSixDecoder initializes weights during construction, before Trainer can seed.
    # Seed here so the canonical executable binds random initialization to --seed.
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    report = run_experiment(
        repo_root=args.repo_root.resolve(),
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        eval_every=args.eval_every,
        seed=args.seed,
        learning_rate=args.learning_rate,
        torch_threads=args.torch_threads,
    )
    validate_report(report, expected_source_sha=args.source_sha)
    summary = {
        "model_identity_sha256": report["model"]["model_identity_sha256"],
        "parameter_count": report["model"]["parameter_count"],
        "optimized_tokens": report["training"]["optimized_tokens"],
        "initial_validation_loss": report["validation_gate"]["initial_loss"],
        "best_validation_loss": report["validation_gate"]["best_loss"],
        "final_validation_loss": report["validation_gate"]["final_loss"],
        "optimized_tokens_per_train_second": report["training"][
            "optimized_tokens_per_train_second"
        ],
        "report_sha256": report["report_sha256"],
    }
    import json

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
