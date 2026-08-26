#!/usr/bin/env python3
"""SCALE-191 preregistered LR x gradient-clipping transfer ablation.

This is a LOCAL_FREE research harness stacked on SCALE-190. It keeps the
model, tokenizer, corpus, packing, evaluation, context, batch, optimizer
family, Adam betas/epsilon, weight decay, precision and token checkpoints
fixed. Only learning rate and gradient clip norm vary.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import scale190_three_million_bridge as scale190

SCHEMA = "12-6.scale191.optimizer-transfer.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_ONLY_NOT_PROMOTION"
BASE_LR = 3e-4
BASE_CLIP_NORM = 1.0
LR_FACTORS = (0.5, 1.0, 2.0)
CLIP_FACTORS = (0.5, 1.0, 2.0)
SEEDS = (1337, 1338)
CHECKPOINT_STEPS = scale190.CHECKPOINT_STEPS
CHECKPOINT_TOKENS = scale190.CHECKPOINT_TOKENS
BASELINE_EXPECTED_BPB = {
    1337: (3.63065595487007, 2.8779210625587925, 3.6269801849983665),
    1338: (3.6899818981527384, 2.9178113452297696, 4.0184479394803585),
}
BASELINE_PARITY_TOLERANCE_BPB = 1e-6


@dataclass(frozen=True, slots=True)
class TrialSpec:
    lr_factor: float
    clip_factor: float

    @property
    def learning_rate(self) -> float:
        return BASE_LR * self.lr_factor

    @property
    def clip_norm(self) -> float:
        return BASE_CLIP_NORM * self.clip_factor

    @property
    def trial_id(self) -> str:
        return f"lr-{self.lr_factor:g}x_clip-{self.clip_factor:g}x"

    @property
    def is_baseline(self) -> bool:
        return self.lr_factor == 1.0 and self.clip_factor == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "lr_factor": self.lr_factor,
            "clip_factor": self.clip_factor,
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.clip_norm,
            "is_scale190_baseline": self.is_baseline,
        }


def trial_matrix() -> tuple[TrialSpec, ...]:
    return tuple(
        TrialSpec(lr_factor=lr_factor, clip_factor=clip_factor)
        for lr_factor in LR_FACTORS
        for clip_factor in CLIP_FACTORS
    )


def build_preregistration() -> dict[str, Any]:
    model_spec = scale190.ModelSpec(**scale190.BRIDGE_SPEC_DICT)
    fixed = {
        "model_identity_sha256": model_spec.identity_sha256(),
        "parameter_count": model_spec.parameter_count(),
        "init_identity_sha256": scale190.INIT_SHA256,
        "tokenizer_id": scale190.TOKENIZER_ID,
        "tokenizer_config_sha256": scale190.TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": scale190.TOKENIZER_VOCAB_SHA256,
        "corpus_dataset_id": scale190.CORPUS_DATASET_ID,
        "corpus_identity_sha256": scale190.CORPUS_IDENTITY_SHA256,
        "corpus_manifest_sha256": scale190.CORPUS_MANIFEST_SHA256,
        "train_jsonl_sha256": scale190.TRAIN_JSONL_SHA256,
        "evaluation_jsonl_sha256": scale190.VALIDATION_JSONL_SHA256,
        "packing_id": scale190.PACKING_ID,
        "context": model_spec.max_seq_len,
        "batch_size": scale190.BATCH_SIZE,
        "sequence_length": scale190.SEQUENCE_LENGTH,
        "causal_tokens_per_update": scale190.TOKEN_QUANTUM,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "precision": "fp32",
        "deterministic_algorithms": True,
    }
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "parent_evidence": {
            "pr": 352,
            "head_sha": "2f1ce640aa8a881f77aba7722d3bed8a00d633b7",
            "observed_problem": (
                "SCALE-190 late held-out BPB reversal after 65,772 tokens with "
                ">94% gradient clipping in both seeds"
            ),
        },
        "hypotheses": {
            "h_lr": "late reversal is sensitive to AdamW learning-rate transfer",
            "h_clip": "clip norm 1.0 materially constrains the 3.2M trajectory",
            "h_interaction": "learning rate and clip norm interact at fixed model/data",
        },
        "fixed_identity": fixed,
        "varied_only": ["learning_rate", "gradient_clip_norm"],
        "grid_rule": "symmetric log2 factors 0.5x/1x/2x around SCALE-190 baseline",
        "trials": [trial.to_dict() for trial in trial_matrix()],
        "seeds": list(SEEDS),
        "optimized_token_checkpoints": list(CHECKPOINT_TOKENS),
        "optimizer_steps": list(CHECKPOINT_STEPS),
        "primary_metric": "two-seed mean heldout_bpb at 131292 optimized tokens",
        "secondary_metrics": [
            "two-seed mean late_delta_bpb_65772_to_131292",
            "two-seed mean clip_fraction",
            "raw_gradient_norm",
            "train_loss",
        ],
        "selection_rule": (
            "Require complete finite trajectories for both preregistered seeds; rank lower "
            "mean final BPB first, then lower mean late BPB delta, then lower mean clip fraction. "
            "Treat the ranking as repeated-tiny-fixture research evidence only."
        ),
        "baseline_parity": {
            "required_before_interpreting_grid": True,
            "absolute_bpb_tolerance": BASELINE_PARITY_TOLERANCE_BPB,
            "expected_by_seed": {
                str(seed): list(values) for seed, values in BASELINE_EXPECTED_BPB.items()
            },
        },
        "truth_boundary": {
            "repeated_tiny_fixture": True,
            "broad_corpus_claim": False,
            "universal_optimizer_transfer_claim": False,
            "model341_20m_training_authorized": False,
            "stage_promotion": False,
            "paid_compute": False,
        },
    }


def validate_preregistration(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected SCALE-191 schema")
    trials = plan.get("trials")
    if not isinstance(trials, list) or len(trials) != 9:
        raise ValueError("SCALE-191 requires the full 3x3 LR x clip grid")
    pairs = {(row["lr_factor"], row["clip_factor"]) for row in trials}
    expected = {(lr, clip) for lr in LR_FACTORS for clip in CLIP_FACTORS}
    if pairs != expected:
        raise ValueError("trial matrix drift")
    baseline_rows = [row for row in trials if row["is_scale190_baseline"]]
    if len(baseline_rows) != 1:
        raise ValueError("exactly one SCALE-190 baseline cell is required")
    if plan.get("varied_only") != ["learning_rate", "gradient_clip_norm"]:
        raise ValueError("optimizer ablation changed an unregistered variable")
    fixed = plan.get("fixed_identity")
    if not isinstance(fixed, dict):
        raise TypeError("fixed_identity must be a mapping")
    if fixed.get("parameter_count") != 3_221_184:
        raise ValueError("SCALE-190 bridge parameter identity drift")


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().double()
        squared += torch.sum(grad * grad)
    return float(torch.sqrt(squared).item())


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("metrics cannot be empty")
    return {
        "min": min(values),
        "max": max(values),
        "mean": fmean(values),
        "last": values[-1],
    }


def run_trial(
    *,
    repo_root: Path,
    source_sha: str,
    trial: TrialSpec,
    seed: int,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    _, validation_records, stream = scale190.assert_data(repo_root)
    model_spec = scale190.ModelSpec(**scale190.BRIDGE_SPEC_DICT)
    random.seed(seed)
    torch.manual_seed(seed)
    model = scale190.Decoder(model_spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    clip_events = 0
    raw_grad_norms: list[float] = []
    train_losses: list[float] = []
    trajectory: list[dict[str, Any]] = []
    checkpoint_set = set(CHECKPOINT_STEPS)
    for step_zero in range(CHECKPOINT_STEPS[-1]):
        batch = scale190.make_batch(stream, step_zero)
        model.train()
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, model_spec.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at optimizer step {step_zero + 1}")
        (loss * scale190.TOKEN_QUANTUM).backward()
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(
                    f"non-finite gradient at optimizer step {step_zero + 1}"
                )
            parameter.grad.div_(scale190.TOKEN_QUANTUM)
        raw_grad_norm = _gradient_norm(model)
        if raw_grad_norm > trial.clip_norm:
            clip_events += 1
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), trial.clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(float(loss.item()))
        raw_grad_norms.append(raw_grad_norm)
        step = step_zero + 1
        if step not in checkpoint_set:
            continue
        validation_loss, validation_tokens = scale190.validation_loss(
            model, validation_records
        )
        bpb = validation_loss / scale190.LN2
        trajectory.append(
            {
                "optimizer_step": step,
                "optimized_tokens": step * scale190.TOKEN_QUANTUM,
                "heldout_loss_nats": validation_loss,
                "heldout_bpb": bpb,
                "validation_tokens": validation_tokens,
                "raw_gradient_norm": raw_grad_norm,
                "clip_events_cumulative": clip_events,
                "clip_fraction_cumulative": clip_events / step,
                "train_loss_window": _stats(train_losses),
                "raw_gradient_norm_window": _stats(raw_grad_norms),
            }
        )
    if [row["optimized_tokens"] for row in trajectory] != list(CHECKPOINT_TOKENS):
        raise RuntimeError("checkpoint trajectory drift")
    bpbs = [float(row["heldout_bpb"]) for row in trajectory]
    baseline_parity = None
    if trial.is_baseline:
        expected = BASELINE_EXPECTED_BPB[seed]
        deltas = [observed - target for observed, target in zip(bpbs, expected, strict=True)]
        baseline_parity = {
            "expected_bpb": list(expected),
            "observed_bpb": bpbs,
            "delta_bpb": deltas,
            "max_abs_delta_bpb": max(abs(delta) for delta in deltas),
            "pass": max(abs(delta) for delta in deltas) <= BASELINE_PARITY_TOLERANCE_BPB,
        }
    return {
        "schema": "12-6.scale191.trial-result.v1",
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "trial": trial.to_dict(),
        "seed": seed,
        "fixed_identity": build_preregistration()["fixed_identity"],
        "trajectory": trajectory,
        "late_delta_bpb_65772_to_131292": bpbs[-1] - bpbs[1],
        "final_clip_fraction": clip_events / CHECKPOINT_STEPS[-1],
        "baseline_parity": baseline_parity,
        "truth_boundary": build_preregistration()["truth_boundary"],
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        trial = result.get("trial")
        if not isinstance(trial, dict):
            raise TypeError("trial result is missing trial metadata")
        grouped.setdefault(str(trial["trial_id"]), []).append(result)
    expected_ids = {trial.trial_id for trial in trial_matrix()}
    if set(grouped) != expected_ids:
        missing = sorted(expected_ids - set(grouped))
        extra = sorted(set(grouped) - expected_ids)
        raise ValueError(f"incomplete trial matrix: missing={missing}, extra={extra}")
    rows = []
    baseline_parity_pass = True
    for trial in trial_matrix():
        trial_results = grouped[trial.trial_id]
        seeds = {int(result["seed"]) for result in trial_results}
        if seeds != set(SEEDS):
            raise ValueError(f"{trial.trial_id} must contain both preregistered seeds")
        final_bpbs = []
        late_deltas = []
        clip_fractions = []
        for result in trial_results:
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, list) or len(trajectory) != 3:
                raise ValueError(f"incomplete trajectory for {trial.trial_id}")
            final_bpb = float(trajectory[-1]["heldout_bpb"])
            late_delta = float(result["late_delta_bpb_65772_to_131292"])
            clip_fraction = float(result["final_clip_fraction"])
            if not all(
                math.isfinite(value) for value in (final_bpb, late_delta, clip_fraction)
            ):
                raise ValueError(f"non-finite metric for {trial.trial_id}")
            final_bpbs.append(final_bpb)
            late_deltas.append(late_delta)
            clip_fractions.append(clip_fraction)
            if trial.is_baseline:
                parity = result.get("baseline_parity")
                baseline_parity_pass = baseline_parity_pass and bool(
                    isinstance(parity, dict) and parity.get("pass")
                )
        rows.append(
            {
                **trial.to_dict(),
                "mean_final_bpb": fmean(final_bpbs),
                "sample_sd_final_bpb": stdev(final_bpbs),
                "mean_late_delta_bpb": fmean(late_deltas),
                "mean_clip_fraction": fmean(clip_fractions),
            }
        )
    if not baseline_parity_pass:
        raise RuntimeError("SCALE-190 baseline parity failed; optimizer grid is not interpretable")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["mean_final_bpb"],
            row["mean_late_delta_bpb"],
            row["mean_clip_fraction"],
        ),
    )
    baseline = next(row for row in rows if row["is_scale190_baseline"])
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["delta_mean_final_bpb_vs_baseline"] = (
            row["mean_final_bpb"] - baseline["mean_final_bpb"]
        )
    return {
        "schema": "12-6.scale191.aggregate.v1",
        "authority": AUTHORITY,
        "baseline_parity_pass": baseline_parity_pass,
        "selection_rule": build_preregistration()["selection_rule"],
        "ranking": ranked,
        "winner": ranked[0],
        "truth_boundary": build_preregistration()["truth_boundary"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trial_from_args(args: argparse.Namespace) -> TrialSpec:
    trial = TrialSpec(lr_factor=args.lr_factor, clip_factor=args.clip_factor)
    if trial not in trial_matrix():
        raise ValueError("trial must be one of the preregistered 0.5x/1x/2x grid cells")
    return trial


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", required=True)

    trial_parser = subparsers.add_parser("trial")
    trial_parser.add_argument("--repo-root", required=True)
    trial_parser.add_argument("--source-sha", required=True)
    trial_parser.add_argument("--lr-factor", type=float, required=True)
    trial_parser.add_argument("--clip-factor", type=float, required=True)
    trial_parser.add_argument("--seed", type=int, required=True)
    trial_parser.add_argument("--output", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--inputs", nargs="+", required=True)
    aggregate_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        plan = build_preregistration()
        validate_preregistration(plan)
        _write_json(Path(args.output), plan)
        return 0
    if args.command == "trial":
        result = run_trial(
            repo_root=Path(args.repo_root),
            source_sha=args.source_sha,
            trial=_trial_from_args(args),
            seed=args.seed,
        )
        _write_json(Path(args.output), result)
        return 0
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    aggregate = aggregate_results(results)
    _write_json(Path(args.output), aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
