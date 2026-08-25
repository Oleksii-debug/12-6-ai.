"""MODEL08 depth/width execution wrapper with retained training-loss telemetry.

The scientific runner lives in :mod:`twelve_six.fixed_token_research`.  This
wrapper does not implement another decoder or optimizer.  It instruments the
incumbent Trainer inside the experiment process so MODEL08 retains the training
loss explicitly required by its evidence contract while selection remains based
on held-out validation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, ClassVar

from . import fixed_token_research as research
from .training import Trainer as CanonicalTrainer


class RecordingTrainer(CanonicalTrainer):
    """Canonical Trainer plus process-local loss observations."""

    observations: ClassVar[list[dict[str, Any]]] = []

    def train_microbatch(self, batch: Any):  # type: ignore[override]
        metrics = super().train_microbatch(batch)
        self.observations.append(
            {
                "optimizer_step": int(metrics.optimizer_step),
                "cumulative_optimized_tokens": int(self.tokens_seen),
                "valid_causal_tokens": int(metrics.tokens),
                "train_loss": float(metrics.loss),
                "update_loss": (
                    None if metrics.update_loss is None else float(metrics.update_loss)
                ),
            }
        )
        return metrics


def _weighted_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "steps": 0,
            "optimized_tokens": 0,
            "token_weighted_mean": None,
            "min_step_loss": None,
            "max_step_loss": None,
            "last_step_loss": None,
        }
    total_tokens = sum(int(row["valid_causal_tokens"]) for row in rows)
    weighted = sum(
        float(row["train_loss"]) * int(row["valid_causal_tokens"]) for row in rows
    )
    losses = [float(row["train_loss"]) for row in rows]
    return {
        "steps": len(rows),
        "optimized_tokens": total_tokens,
        "token_weighted_mean": weighted / total_tokens,
        "min_step_loss": min(losses),
        "max_step_loss": max(losses),
        "last_step_loss": losses[-1],
    }


def _segment_summaries(
    rows: list[dict[str, Any]], budgets: tuple[int, ...]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lower = 0
    for budget in budgets:
        segment = [
            row
            for row in rows
            if lower < int(row["cumulative_optimized_tokens"]) <= budget
        ]
        summary = _weighted_summary(segment)
        summary["segment_start_optimized_tokens"] = lower
        summary["segment_end_optimized_tokens"] = budget
        if int(summary["optimized_tokens"]) != budget - lower:
            raise RuntimeError(
                "MODEL08 training-loss telemetry token drift: "
                f"segment {lower}->{budget} recorded {summary['optimized_tokens']}"
            )
        result.append(summary)
        lower = budget
    return result


def run_model08_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    candidate_id: str,
    output_path: Path,
    checkpoint_dir: Path,
    token_budgets: tuple[int, ...] = research.DEFAULT_BUDGETS,
    batch_size: int = research.DEFAULT_BATCH_SIZE,
    sequence_length: int = research.DEFAULT_SEQUENCE_LENGTH,
    seed: int = research.DEFAULT_SEED,
    torch_threads: int = research.DEFAULT_THREADS,
) -> dict[str, Any]:
    budgets = research._validate_budgets(token_budgets)
    RecordingTrainer.observations = []
    original_trainer = research.Trainer
    research.Trainer = RecordingTrainer
    try:
        report = research.run_candidate(
            repo_root=repo_root,
            source_sha=source_sha,
            family="depth_width_100k",
            candidate_id=candidate_id,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            token_budgets=budgets,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
            torch_threads=torch_threads,
            exercise_resume=True,
        )
    finally:
        research.Trainer = original_trainer

    observations = list(RecordingTrainer.observations)
    if not observations:
        raise RuntimeError("MODEL08 captured no training-loss observations")
    if int(observations[-1]["cumulative_optimized_tokens"]) != budgets[-1]:
        raise RuntimeError("MODEL08 final training-loss token ledger drift")
    if any(not math.isfinite(float(row["train_loss"])) for row in observations):
        raise RuntimeError("MODEL08 captured non-finite training loss")

    report["training_loss_telemetry"] = {
        "definition": (
            "Per-optimizer-step causal cross entropy over exactly the valid optimized "
            "targets in that step; token-weighted summaries are diagnostics only and "
            "are never substituted for held-out validation."
        ),
        "all_steps": observations,
        "overall": _weighted_summary(observations),
        "segments": _segment_summaries(observations, budgets),
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = research._canonical_hash(report)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    research._validate_candidate(report, expected_source_sha=source_sha)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--token-budgets", type=int, nargs="+", default=list(research.DEFAULT_BUDGETS)
    )
    parser.add_argument("--batch-size", type=int, default=research.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--sequence-length", type=int, default=research.DEFAULT_SEQUENCE_LENGTH
    )
    parser.add_argument("--seed", type=int, default=research.DEFAULT_SEED)
    parser.add_argument("--torch-threads", type=int, default=research.DEFAULT_THREADS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_model08_candidate(
        repo_root=args.repo_root.resolve(),
        source_sha=args.source_sha,
        candidate_id=args.candidate_id,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        token_budgets=tuple(args.token_budgets),
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "training_loss": report["training_loss_telemetry"]["overall"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
