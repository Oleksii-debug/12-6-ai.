"""TRAIN-43 controlled warmup ablation for approximately 100K and 1M Base models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

from .optimization_experiments import (
    OptimizationRecipe,
    _batch_trace_sha256,
    _evaluate,
    _model_fingerprint,
    _parameter_l2,
    _snapshot,
    _stable_model,
    _tensor_batches,
    _update_metrics,
)
from .trainer import Trainer

PLAN_SCHEMA = "12-6.warmup-ablation-plan.v1"
REPORT_SCHEMA = "12-6.warmup-ablation-evidence.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_WARMUP_EVIDENCE_PROVISIONAL"
REPOSITORY = "Oleksii-debug/12-6-ai."
EXPECTED_COUNTS = {"S1": 107_856, "S2": 1_066_112}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(plan, dict), "plan root must be an object")
    _require(plan.get("schema_version") == PLAN_SCHEMA, "wrong warmup plan schema")
    _require(float(plan.get("learning_rate")) == 3e-4, "TRAIN-43 LR must remain 3e-4")
    optimizer = plan.get("optimizer")
    _require(isinstance(optimizer, dict), "optimizer block missing")
    _require(float(optimizer.get("weight_decay")) == 0.0, "weight decay must remain zero")
    _require(tuple(float(x) for x in optimizer.get("betas", ())) == (0.9, 0.95), "betas drift")
    _require(float(optimizer.get("eps")) == 1e-8, "epsilon drift")
    _require(optimizer.get("scheduler") == "linear_warmup", "scheduler must isolate warmup")
    _require(float(optimizer.get("gradient_clip_norm")) == 1.0, "clip norm drift")
    fractions = tuple(float(x) for x in plan.get("warmup_fractions", ()))
    _require(fractions == (0.0, 0.02, 0.05), "warmup grid must be 0/2%/5%")
    horizon = plan.get("schedule_horizon_steps")
    _require(isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0, "bad horizon")
    stages = plan.get("stages")
    _require(isinstance(stages, list) and len(stages) == 2, "S1 and S2 stage plans required")
    seen: set[str] = set()
    for stage in stages:
        _require(isinstance(stage, dict), "stage plan must be an object")
        name = stage.get("stage")
        _require(name in EXPECTED_COUNTS and name not in seen, "bad or duplicate stage")
        seen.add(str(name))
        for field in ("execution_steps", "batch_size", "sequence_length"):
            value = stage.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"bad {field}")
        _require(horizon >= 4 * int(stage["execution_steps"]), "scheduler horizon must be independent and >=4x execution")
    _require(seen == set(EXPECTED_COUNTS), "plan must cover S1 and S2")
    early = plan.get("early_window_steps")
    eval_every = plan.get("evaluation_interval_steps")
    _require(isinstance(early, int) and early > 0, "bad early window")
    _require(isinstance(eval_every, int) and eval_every > 0, "bad evaluation interval")
    return plan


def _recipe(plan: dict[str, Any], fraction: float) -> OptimizationRecipe:
    optimizer = plan["optimizer"]
    return OptimizationRecipe(
        name=f"train43_warmup_{fraction:.2f}",
        optimizer="adamw",
        learning_rate=float(plan["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        scheduler="linear_warmup",
        warmup_fraction=fraction,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        decay_embeddings=True,
        precision="fp32",
    )


def materialized_warmup_steps(plan: dict[str, Any]) -> dict[float, int]:
    horizon = int(plan["schedule_horizon_steps"])
    seed = int(plan["seed"])
    result: dict[float, int] = {}
    for fraction in (float(x) for x in plan["warmup_fractions"]):
        _, materialized = _recipe(plan, fraction).materialize(
            schedule_horizon_steps=horizon,
            seed=seed,
        )
        result[fraction] = int(materialized["warmup_steps"])
    return result


def _run_candidate(
    *,
    repo_root: Path,
    stage_plan: dict[str, Any],
    plan: dict[str, Any],
    fraction: float,
    train_batches: list[dict[str, torch.Tensor]],
    validation_batches: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    stage_name = str(stage_plan["stage"])
    stage = load_stage_config(repo_root / str(stage_plan["config"]))
    _require(stage.expected_parameters == EXPECTED_COUNTS[stage_name], "stage parameter count drift")
    seed = int(plan["seed"])
    execution_steps = int(stage_plan["execution_steps"])
    horizon = int(plan["schedule_horizon_steps"])
    early_window = min(int(plan["early_window_steps"]), execution_steps)
    eval_every = int(plan["evaluation_interval_steps"])
    recipe = _recipe(plan, fraction)
    config, materialized = recipe.materialize(schedule_horizon_steps=horizon, seed=seed)

    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_model_sha = _model_fingerprint(model)
    initial_train_loss = _evaluate(model, train_batches)
    initial_validation_loss = _evaluate(model, validation_batches)
    trainer = Trainer(model, config, device="cpu")

    progression: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = [
        {"step": 0, "optimized_tokens": 0, "loss": initial_validation_loss}
    ]
    failure: dict[str, Any] | None = None
    for step, batch in enumerate(islice(cycle(train_batches), execution_steps), start=1):
        before = _snapshot(model)
        parameter_l2_before = _parameter_l2(model)
        try:
            metrics = trainer.train_microbatch(batch)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failure = {"step": step, "exception_type": type(exc).__name__, "message": str(exc)[:240]}
            break
        update = _update_metrics(model, before, parameter_l2_before=parameter_l2_before)
        grad_norm = None if metrics.grad_norm is None else float(metrics.grad_norm)
        finite = _stable_model(model) and math.isfinite(float(metrics.loss))
        progression.append(
            {
                "step": step,
                "optimized_tokens": trainer.tokens_seen,
                "train_loss": float(metrics.loss),
                "learning_rate": float(metrics.learning_rate),
                "gradient_norm": grad_norm,
                "clip_activated": grad_norm is not None and grad_norm > 1.0,
                **update,
            }
        )
        if not finite:
            failure = {"step": step, "exception_type": "NonFiniteState"}
            break
        if step <= early_window or step % eval_every == 0 or step == execution_steps:
            tokens_before = trainer.tokens_seen
            optimizer_step_before = trainer.optimizer_step
            validation_loss = _evaluate(model, validation_batches)
            _require(trainer.tokens_seen == tokens_before, "validation changed token accounting")
            _require(trainer.optimizer_step == optimizer_step_before, "validation changed optimizer step")
            held_out.append(
                {"step": step, "optimized_tokens": tokens_before, "loss": validation_loss}
            )
            model.train()

    gradients = [float(row["gradient_norm"]) for row in progression if row["gradient_norm"] is not None]
    updates = [float(row["relative_update_l2"]) for row in progression]
    early_rows = [row for row in progression if int(row["step"]) <= early_window]
    early_held_out = [row for row in held_out if 0 < int(row["step"]) <= early_window]
    recovery = next(
        (
            int(row["optimized_tokens"])
            for row in held_out
            if int(row["step"]) > 0 and float(row["loss"]) <= initial_validation_loss + 1e-12
        ),
        None,
    )
    status = "PASS" if failure is None and len(progression) == execution_steps else "FAIL"
    final_validation = held_out[-1]["loss"] if held_out[-1]["step"] == execution_steps else None
    early_gradients = [float(row["gradient_norm"]) for row in early_rows if row["gradient_norm"] is not None]
    early_updates = [float(row["relative_update_l2"]) for row in early_rows]
    train_spike = max(
        0.0,
        max((float(row["train_loss"]) for row in early_rows), default=initial_train_loss) - initial_train_loss,
    )
    validation_spike = max(
        0.0,
        max((float(row["loss"]) for row in early_held_out), default=initial_validation_loss) - initial_validation_loss,
    )
    return {
        "warmup_fraction": fraction,
        "warmup_steps": int(materialized["warmup_steps"]),
        "schedule_horizon_steps": horizon,
        "status": status,
        "failure": failure,
        "initial_model_sha256": initial_model_sha,
        "batch_trace_sha256": _batch_trace_sha256(train_batches, execution_steps),
        "initial_train_loss": initial_train_loss,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation,
        "training_loss_first": progression[0]["train_loss"] if progression else None,
        "training_loss_last": progression[-1]["train_loss"] if progression else None,
        "early_train_loss_spike_nats": train_spike,
        "early_validation_loss_spike_nats": validation_spike,
        "early_gradient_norm_max": max(early_gradients) if early_gradients else None,
        "gradient_norm_max": max(gradients) if gradients else None,
        "gradient_norm_median": statistics.median(gradients) if gradients else None,
        "early_clip_frequency": (
            sum(bool(row["clip_activated"]) for row in early_rows) / len(early_rows) if early_rows else None
        ),
        "clip_frequency": (
            sum(bool(row["clip_activated"]) for row in progression) / len(progression) if progression else None
        ),
        "early_relative_update_l2_max": max(early_updates) if early_updates else None,
        "relative_update_l2_max": max(updates) if updates else None,
        "relative_update_l2_median": statistics.median(updates) if updates else None,
        "recovery_tokens_from_initial_validation": recovery,
        "steps_completed": len(progression),
        "optimized_tokens": trainer.tokens_seen,
        "progression": progression,
        "held_out": held_out,
    }


def _material_improvements(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    improved: list[str] = []
    base_spike = float(baseline["early_validation_loss_spike_nats"])
    cand_spike = float(candidate["early_validation_loss_spike_nats"])
    if base_spike > 1e-8 and cand_spike <= 0.9 * base_spike:
        improved.append("early_validation_loss_spike")
    for key, label in (
        ("early_gradient_norm_max", "early_gradient_norm"),
        ("early_relative_update_l2_max", "early_update_ratio"),
    ):
        base = baseline[key]
        cand = candidate[key]
        if base is not None and cand is not None and float(cand) <= 0.9 * float(base):
            improved.append(label)
    base_clip = baseline["early_clip_frequency"]
    cand_clip = candidate["early_clip_frequency"]
    if base_clip is not None and cand_clip is not None and float(cand_clip) <= float(base_clip) - 0.10:
        improved.append("early_clip_frequency")
    base_recovery = baseline["recovery_tokens_from_initial_validation"]
    cand_recovery = candidate["recovery_tokens_from_initial_validation"]
    if base_recovery is None and cand_recovery is not None:
        improved.append("recovery_tokens")
    elif base_recovery is not None and cand_recovery is not None and int(cand_recovery) <= 0.9 * int(base_recovery):
        improved.append("recovery_tokens")
    return improved


def provisional_rule(stage_results: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = (0.0, 0.02, 0.05)
    qualification: dict[str, dict[str, Any]] = {}
    for stage in stage_results:
        name = str(stage["stage"])
        by_fraction = {float(row["warmup_fraction"]): row for row in stage["results"]}
        baseline = by_fraction[0.0]
        stage_q: dict[str, Any] = {}
        for fraction in fractions[1:]:
            candidate = by_fraction[fraction]
            improvements = _material_improvements(candidate, baseline)
            final_ok = (
                candidate["status"] == "PASS"
                and baseline["status"] == "PASS"
                and candidate["final_validation_loss"] is not None
                and baseline["final_validation_loss"] is not None
                and float(candidate["final_validation_loss"]) <= 1.01 * float(baseline["final_validation_loss"])
            )
            stage_q[f"{fraction:.2f}"] = {
                "material_improvements": improvements,
                "final_validation_within_1pct": final_ok,
                "qualifies": final_ok and len(improvements) >= 2,
            }
        qualification[name] = stage_q
    selected = 0.0
    for fraction in fractions[1:]:
        if all(bool(qualification[stage][f"{fraction:.2f}"]["qualifies"]) for stage in EXPECTED_COUNTS):
            selected = fraction
            break
    if selected == 0.0:
        text = (
            "At AdamW peak LR 3e-4 with betas 0.9/0.95, zero weight decay and clip=1.0, "
            "use no warmup provisionally for the controlled 100K-1M family; reopen warmup when "
            "LR, batch geometry, precision or model scale materially changes."
        )
    else:
        text = (
            f"Use {selected:.0%} linear warmup of the intended scheduler horizon provisionally at "
            "100K-1M for the fixed 3e-4 AdamW recipe; keep the horizon independent of short probes."
        )
    return {
        "selected_warmup_fraction": selected,
        "qualification": qualification,
        "rule": text,
        "selection_contract": (
            "Choose the smallest nonzero warmup that materially improves at least two early-stability "
            "signals on both scales while keeping final held-out loss within 1% of no-warmup; otherwise use zero."
        ),
    }


def run_experiment(*, repo_root: Path, source_sha: str, plan_path: Path, output_path: Path) -> dict[str, Any]:
    _require(_git_head(repo_root) == source_sha, "exact source checkout mismatch")
    plan = load_plan(plan_path)
    torch.set_num_threads(int(plan["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "controlled tokenizer drift")
    stage_results: list[dict[str, Any]] = []
    for stage_plan in plan["stages"]:
        train_batches, train_ids, _, train_max = _tensor_batches(
            repo_root,
            split="train",
            tokenizer=tokenizer,
            batch_size=int(stage_plan["batch_size"]),
            sequence_length=int(stage_plan["sequence_length"]),
        )
        validation_batches, validation_ids, validation_tokens, validation_max = _tensor_batches(
            repo_root,
            split="validation",
            tokenizer=tokenizer,
            batch_size=int(stage_plan["batch_size"]),
            sequence_length=int(stage_plan["sequence_length"]),
        )
        _require(not (set(train_ids) & set(validation_ids)), "train/validation record overlap")
        _require(max(train_max, validation_max) < 256, "fixture token exceeds byte vocab")
        results = [
            _run_candidate(
                repo_root=repo_root,
                stage_plan=stage_plan,
                plan=plan,
                fraction=float(fraction),
                train_batches=train_batches,
                validation_batches=validation_batches,
            )
            for fraction in plan["warmup_fractions"]
        ]
        _require(len({row["initial_model_sha256"] for row in results}) == 1, "initialization drift")
        _require(len({row["batch_trace_sha256"] for row in results}) == 1, "batch trace drift")
        stage_results.append(
            {
                "stage": stage_plan["stage"],
                "parameter_count": EXPECTED_COUNTS[str(stage_plan["stage"])],
                "validation_scoreable_tokens": validation_tokens,
                "results": results,
            }
        )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "paid_compute": False},
        "plan": plan,
        "stage_results": stage_results,
        "provisional_warmup_rule": provisional_rule(stage_results),
        "truth_boundary": {
            "representative_corpus_claim": False,
            "quality_or_capability_claim": False,
            "paid_compute_authorized_or_used": False,
            "optimizer_betas_changed_across_candidates": False,
            "weight_decay_changed_across_candidates": False,
            "scheduler_horizon_equals_short_execution_by_accident": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_report(report, expected_source_sha=source_sha)
    output_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "wrong report schema")
    source = report.get("source")
    _require(isinstance(source, dict), "source missing")
    if expected_source_sha is not None:
        _require(source.get("git_sha") == expected_source_sha, "source SHA mismatch")
    stages = report.get("stage_results")
    _require(isinstance(stages, list) and len(stages) == 2, "two scale results required")
    _require({row["stage"] for row in stages} == set(EXPECTED_COUNTS), "stage result drift")
    for stage in stages:
        results = stage.get("results")
        _require(isinstance(results, list) and len(results) == 3, "three warmup candidates required")
        _require({float(row["warmup_fraction"]) for row in results} == {0.0, 0.02, 0.05}, "warmup grid drift")
        _require(len({row["initial_model_sha256"] for row in results}) == 1, "init mismatch")
        _require(len({row["batch_trace_sha256"] for row in results}) == 1, "batch mismatch")
        for row in results:
            _require(row["schedule_horizon_steps"] > row["steps_completed"], "horizon collapsed to probe length")
            if row["status"] == "PASS":
                _require(row["final_validation_loss"] is not None, "PASS missing validation")
                _require(math.isfinite(float(row["gradient_norm_max"])), "non-finite gradient")
                _require(math.isfinite(float(row["relative_update_l2_max"])), "non-finite update ratio")
    expected = dict(report)
    claimed_hash = expected.pop("report_sha256", None)
    _require(isinstance(claimed_hash, str) and claimed_hash == _canonical_hash(expected), "report hash mismatch")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha": report["source"]["git_sha"],
        "learning_rate": report["plan"]["learning_rate"],
        "warmup_steps": materialized_warmup_steps(report["plan"]),
        "stages": [
            {
                "stage": stage["stage"],
                "parameter_count": stage["parameter_count"],
                "results": [
                    {
                        "warmup_fraction": row["warmup_fraction"],
                        "warmup_steps": row["warmup_steps"],
                        "early_validation_loss_spike_nats": row["early_validation_loss_spike_nats"],
                        "early_gradient_norm_max": row["early_gradient_norm_max"],
                        "early_clip_frequency": row["early_clip_frequency"],
                        "early_relative_update_l2_max": row["early_relative_update_l2_max"],
                        "recovery_tokens": row["recovery_tokens_from_initial_validation"],
                        "final_validation_loss": row["final_validation_loss"],
                    }
                    for row in stage["results"]
                ],
            }
            for stage in report["stage_results"]
        ],
        "provisional_warmup_rule": report["provisional_warmup_rule"],
        "report_sha256": report["report_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--summary-output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            plan_path=args.plan,
            output_path=args.output,
        )
        args.summary_output.write_text(
            json.dumps(_summary(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
