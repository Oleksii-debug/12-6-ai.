"""Controlled learning-rate range experiment for the 12-6 small Base family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

from .optimization_experiments import (
    OptimizationRecipe,
    _batch_trace_sha256,
    _build_experiment_optimizer,
    _evaluate,
    _model_fingerprint,
    _parameter_l2,
    _snapshot,
    _stable_model,
    _tensor_batches,
    _update_metrics,
)
from .s0_evidence_contract import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    INIT_SPEC_SHA256,
    PACKING_CONFIG_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
)
from .trainer import Trainer

PLAN_SCHEMA = "12-6.lr-range-plan.v1"
REPORT_SCHEMA = "12-6.lr-range-evidence.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_LR_RANGE_EVIDENCE_PROVISIONAL"
REPOSITORY = "Oleksii-debug/12-6-ai."
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

_GEOMETRIES: tuple[dict[str, int], ...] = (
    {
        "d_model": 48,
        "n_layers": 3,
        "n_heads": 4,
        "n_kv_heads": 4,
        "head_dim": 12,
        "d_ff": 128,
    },
    {
        "d_model": 72,
        "n_layers": 4,
        "n_heads": 6,
        "n_kv_heads": 6,
        "head_dim": 12,
        "d_ff": 192,
    },
    {
        "d_model": 96,
        "n_layers": 4,
        "n_heads": 6,
        "n_kv_heads": 6,
        "head_dim": 16,
        "d_ff": 256,
    },
)
_EXPECTED_COUNTS = (95_568, 267_912, 467_808)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _model_spec(geometry: Mapping[str, int]) -> ModelSpec:
    head_dim = geometry["head_dim"]
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=geometry["d_model"],
        n_layers=geometry["n_layers"],
        n_heads=geometry["n_heads"],
        n_kv_heads=geometry["n_kv_heads"],
        head_dim=head_dim,
        d_ff=geometry["d_ff"],
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=head_dim,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def controlled_family() -> tuple[ModelSpec, ...]:
    """Return the established RESEARCH41 100K-500K fixed-control family."""
    specs = tuple(_model_spec(geometry) for geometry in _GEOMETRIES)
    counts = tuple(spec.parameter_count() for spec in specs)
    _require(counts == _EXPECTED_COUNTS, f"controlled family drift: {counts!r}")
    _require({spec.vocab_size for spec in specs} == {256}, "vocabulary drift")
    _require({spec.max_seq_len for spec in specs} == {256}, "context drift")
    return specs


def load_plan(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "plan root must be an object")
    _require(raw.get("schema_version") == PLAN_SCHEMA, "wrong plan schema")
    rates = raw.get("learning_rates")
    _require(isinstance(rates, list) and 4 <= len(rates) <= 6, "LR grid must contain 4-6 points")
    learning_rates = tuple(float(value) for value in rates)
    _require(all(math.isfinite(value) and value > 0 for value in learning_rates), "invalid LR")
    _require(tuple(sorted(learning_rates)) == learning_rates, "LRs must be increasing")
    _require(len(set(learning_rates)) == len(learning_rates), "LRs must be unique")
    default_lr = float(raw.get("current_default_learning_rate"))
    _require(default_lr in learning_rates, "current default LR must be included")
    log_gaps = [
        math.log10(right) - math.log10(left)
        for left, right in zip(learning_rates, learning_rates[1:])
    ]
    _require(max(log_gaps) - min(log_gaps) < 0.08, "LR points are not approximately log-spaced")
    execution = raw.get("execution")
    _require(isinstance(execution, dict), "execution block missing")
    for field in (
        "execution_steps",
        "schedule_horizon_steps",
        "evaluation_interval_steps",
        "batch_size",
        "sequence_length",
        "torch_threads",
    ):
        value = execution.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"invalid {field}")
    _require(
        execution["schedule_horizon_steps"] >= 3 * execution["execution_steps"],
        "cosine horizon must be at least 3x the short experiment",
    )
    _require(
        execution["execution_steps"] % execution["evaluation_interval_steps"] == 0,
        "evaluation interval must divide execution steps",
    )
    model_family = raw.get("model_family")
    _require(isinstance(model_family, dict), "model family block missing")
    _require(tuple(model_family.get("parameter_counts", ())) == _EXPECTED_COUNTS, "model family drift")
    return raw


def _recipe(plan: Mapping[str, Any], learning_rate: float) -> OptimizationRecipe:
    optimizer = plan["optimizer"]
    return OptimizationRecipe(
        name=f"train42_lr_{learning_rate:.0e}",
        optimizer="adamw",
        learning_rate=learning_rate,
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        scheduler=str(optimizer["scheduler"]),  # type: ignore[arg-type]
        warmup_fraction=float(optimizer["warmup_fraction"]),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        decay_embeddings=bool(optimizer["decay_embeddings"]),
        precision="fp32",
    )


def _run_candidate(
    *,
    spec: ModelSpec,
    init_spec: InitSpec,
    learning_rate: float,
    plan: Mapping[str, Any],
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    validation_batches: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    execution = plan["execution"]
    seed = int(execution["seed"])
    execution_steps = int(execution["execution_steps"])
    schedule_horizon = int(execution["schedule_horizon_steps"])
    eval_interval = int(execution["evaluation_interval_steps"])
    recipe = _recipe(plan, learning_rate)
    config, materialized_recipe = recipe.materialize(
        schedule_horizon_steps=schedule_horizon,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    initial_model_sha256 = _model_fingerprint(model)
    initial_validation_loss = _evaluate(model, validation_batches)
    optimizer = _build_experiment_optimizer(model, config, recipe)
    trainer = Trainer(model, config, device="cpu", optimizer=optimizer)
    batch_list = list(train_batches)
    _require(bool(batch_list), "training batch trace is empty")

    progression: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = [{"step": 0, "loss": initial_validation_loss}]
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    for step in range(1, execution_steps + 1):
        batch = batch_list[(step - 1) % len(batch_list)]
        before = _snapshot(model)
        parameter_l2_before = _parameter_l2(model)
        try:
            metrics = trainer.train_microbatch(batch)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failure = {
                "step": step,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:300],
            }
            break
        update = _update_metrics(model, before, parameter_l2_before=parameter_l2_before)
        finite = _stable_model(model) and math.isfinite(float(metrics.loss))
        grad_norm = None if metrics.grad_norm is None else float(metrics.grad_norm)
        progression.append(
            {
                "step": step,
                "train_loss": float(metrics.loss),
                "learning_rate": float(metrics.learning_rate),
                "gradient_norm": grad_norm,
                "clip_activated": (
                    grad_norm is not None
                    and recipe.gradient_clip_norm is not None
                    and grad_norm > recipe.gradient_clip_norm
                ),
                "parameters_finite": finite,
                **update,
            }
        )
        if not finite:
            failure = {"step": step, "exception_type": "NonFiniteState"}
            break
        if step % eval_interval == 0:
            tokens_before = trainer.tokens_seen
            optimizer_step_before = trainer.optimizer_step
            validation_loss = _evaluate(model, validation_batches)
            _require(trainer.tokens_seen == tokens_before, "evaluation changed optimized-token count")
            _require(trainer.optimizer_step == optimizer_step_before, "evaluation changed optimizer step")
            held_out.append({"step": step, "loss": validation_loss})

    elapsed = time.perf_counter() - started
    final_validation_loss = held_out[-1]["loss"] if held_out[-1]["step"] == execution_steps else None
    gradients = [float(item["gradient_norm"]) for item in progression if item["gradient_norm"] is not None]
    relative_updates = [float(item["relative_update_l2"]) for item in progression]
    learning_rates = [float(item["learning_rate"]) for item in progression]
    return {
        "learning_rate": learning_rate,
        "recipe": materialized_recipe,
        "initial_model_sha256": initial_model_sha256,
        "batch_trace_sha256": _batch_trace_sha256(batch_list, execution_steps),
        "status": "PASS" if failure is None and len(progression) == execution_steps else "FAIL",
        "failure": failure,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "held_out": held_out,
        "train_loss_first": progression[0]["train_loss"] if progression else None,
        "train_loss_last": progression[-1]["train_loss"] if progression else None,
        "gradient_norm_median": statistics.median(gradients) if gradients else None,
        "gradient_norm_max": max(gradients) if gradients else None,
        "clip_frequency": (
            sum(bool(item["clip_activated"]) for item in progression) / len(progression)
            if progression
            else None
        ),
        "relative_update_l2_median": statistics.median(relative_updates) if relative_updates else None,
        "relative_update_l2_max": max(relative_updates) if relative_updates else None,
        "peak_observed_learning_rate": max(learning_rates) if learning_rates else None,
        "final_observed_learning_rate": learning_rates[-1] if learning_rates else None,
        "wall_seconds": elapsed,
        "steps_completed": len(progression),
        "progression": progression,
    }


def _classify_model_results(
    results: Sequence[dict[str, Any]],
    *,
    default_lr: float,
    plan: Mapping[str, Any],
) -> None:
    classification = plan["classification"]
    default = next(item for item in results if float(item["learning_rate"]) == default_lr)
    default_improvement = (
        float(default["initial_validation_loss"]) - float(default["final_validation_loss"])
        if default["status"] == "PASS" and default["final_validation_loss"] is not None
        else 0.0
    )
    for result in results:
        final_loss = result["final_validation_loss"]
        initial_loss = float(result["initial_validation_loss"])
        failed = result["status"] != "PASS" or final_loss is None
        regression = (
            not failed
            and float(final_loss)
            > initial_loss * (1.0 + float(classification["unstable_validation_regression_fraction"]))
        )
        excessive_update = (
            result["relative_update_l2_max"] is not None
            and float(result["relative_update_l2_max"])
            > float(classification["unstable_relative_update_l2_max"])
        )
        if failed or regression or excessive_update:
            label = "unstable"
        else:
            improvement = initial_loss - float(final_loss)
            slow_threshold = max(
                0.0,
                default_improvement
                * float(classification["too_slow_fraction_of_default_validation_improvement"]),
            )
            if float(result["learning_rate"]) < default_lr and improvement < slow_threshold:
                label = "too_slow"
            else:
                label = "healthy"
        result["classification"] = label


def _aggregate_summary(model_runs: Sequence[Mapping[str, Any]], default_lr: float) -> dict[str, Any]:
    rates = sorted({float(result["learning_rate"]) for run in model_runs for result in run["results"]})
    consensus: dict[str, str] = {}
    for rate in rates:
        labels = [
            next(
                result["classification"]
                for result in run["results"]
                if float(result["learning_rate"]) == rate
            )
            for run in model_runs
        ]
        if "unstable" in labels:
            consensus[f"{rate:.12g}"] = "unstable"
        elif labels.count("too_slow") >= 2:
            consensus[f"{rate:.12g}"] = "too_slow"
        elif all(label == "healthy" for label in labels):
            consensus[f"{rate:.12g}"] = "healthy"
        else:
            consensus[f"{rate:.12g}"] = "mixed"
    default_healthy = consensus.get(f"{default_lr:.12g}") == "healthy"
    healthy_rates = [rate for rate in rates if consensus[f"{rate:.12g}"] == "healthy"]
    _require(bool(healthy_rates), "no family-wide healthy LR candidate")
    transfer_lr = default_lr if default_healthy else min(healthy_rates)
    return {
        "consensus_by_learning_rate": consensus,
        "provisional_1m_learning_rate": transfer_lr,
        "transfer_rule": (
            "Hold the current 3e-4 AdamW peak LR through approximately 1M when it is healthy "
            "across the fixed-control 100K-500K family; do not promote a more aggressive LR from "
            "the tiny repeated fixture alone."
            if default_healthy
            else "Use the lowest family-wide healthy LR provisionally at approximately 1M."
        ),
        "aggressive_lr_promotion_authorized": False,
    }


def run_lr_range(
    *,
    repo_root: Path,
    source_sha: str,
    plan_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require(_GIT_SHA.fullmatch(source_sha) is not None, "source SHA must be lowercase 40-hex")
    _require(_git_head(repo_root) == source_sha, "exact checkout mismatch")
    plan = load_plan(plan_path)
    execution = plan["execution"]
    torch.set_num_threads(int(execution["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "controlled byte tokenizer drift")
    train_batches, train_ids, _, train_max_id = _tensor_batches(
        repo_root,
        split="train",
        tokenizer=tokenizer,
        batch_size=int(execution["batch_size"]),
        sequence_length=int(execution["sequence_length"]),
    )
    validation_batches, validation_ids, validation_tokens, validation_max_id = _tensor_batches(
        repo_root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=int(execution["batch_size"]),
        sequence_length=int(execution["sequence_length"]),
    )
    _require(not (set(train_ids) & set(validation_ids)), "train/validation record overlap")
    _require(max(train_max_id, validation_max_id) < 256, "fixture token exceeds controlled vocab")
    init_spec = InitSpec()
    _require(init_spec.identity_sha256() == INIT_SPEC_SHA256, "InitSpec drift")
    learning_rates = tuple(float(value) for value in plan["learning_rates"])
    default_lr = float(plan["current_default_learning_rate"])
    model_runs: list[dict[str, Any]] = []
    for spec in controlled_family():
        results = [
            _run_candidate(
                spec=spec,
                init_spec=init_spec,
                learning_rate=rate,
                plan=plan,
                train_batches=train_batches,
                validation_batches=validation_batches,
            )
            for rate in learning_rates
        ]
        _require(len({item["initial_model_sha256"] for item in results}) == 1, "initialization drift")
        _require(len({item["batch_trace_sha256"] for item in results}) == 1, "batch-trace drift")
        _classify_model_results(results, default_lr=default_lr, plan=plan)
        model_runs.append(
            {
                "parameter_count": spec.parameter_count(),
                "modelspec_sha256": spec.identity_sha256(),
                "shared_initial_model_sha256": results[0]["initial_model_sha256"],
                "shared_batch_trace_sha256": results[0]["batch_trace_sha256"],
                "results": results,
            }
        )
    summary = _aggregate_summary(model_runs, default_lr)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "paid_compute": False,
        },
        "plan": {
            "path": str(plan_path.relative_to(repo_root)),
            "file_sha256": _file_sha256(plan_path),
            "learning_rates": list(learning_rates),
            "current_default_learning_rate": default_lr,
            "execution": execution,
        },
        "fixed_identities": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "init_spec_sha256": INIT_SPEC_SHA256,
            "validation_scoreable_tokens": validation_tokens,
        },
        "model_runs": model_runs,
        "summary": summary,
        "truth_boundary": {
            "model_changed": False,
            "data_changed": False,
            "tokenizer_changed": False,
            "optimizer_family_changed": False,
            "quality_or_capability_claim": False,
            "paid_compute_authorized_or_used": False,
            "approximately_1m_executed": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_report(report, expected_source_sha=source_sha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: Mapping[str, Any], *, expected_source_sha: str | None = None) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "wrong report schema")
    _require(report.get("authority") == AUTHORITY, "wrong report authority")
    source = report.get("source")
    _require(isinstance(source, Mapping), "source missing")
    source_sha = source.get("git_sha")
    _require(isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None, "bad source SHA")
    if expected_source_sha is not None:
        _require(source_sha == expected_source_sha, "source SHA mismatch")
    model_runs = report.get("model_runs")
    _require(isinstance(model_runs, list) and len(model_runs) == 3, "expected three model runs")
    _require(tuple(int(run["parameter_count"]) for run in model_runs) == _EXPECTED_COUNTS, "model family drift")
    plan = report.get("plan")
    _require(isinstance(plan, Mapping), "plan missing")
    learning_rates = plan.get("learning_rates")
    _require(isinstance(learning_rates, list) and 4 <= len(learning_rates) <= 6, "bad LR grid")
    for run in model_runs:
        results = run.get("results")
        _require(isinstance(results, list) and len(results) == len(learning_rates), "LR result count drift")
        _require(len({result["initial_model_sha256"] for result in results}) == 1, "init drift")
        _require(len({result["batch_trace_sha256"] for result in results}) == 1, "batch trace drift")
        for result in results:
            _require(result.get("classification") in {"too_slow", "healthy", "unstable"}, "bad classification")
            if result.get("status") == "PASS":
                _require(result.get("final_validation_loss") is not None, "PASS missing held-out loss")
                _require(math.isfinite(float(result["gradient_norm_max"])), "non-finite gradient norm")
                _require(math.isfinite(float(result["relative_update_l2_max"])), "non-finite update ratio")
                peak = float(result["peak_observed_learning_rate"])
                final = float(result["final_observed_learning_rate"])
                _require(final >= 0.5 * peak, "cosine schedule collapsed during short experiment")
    truth = report.get("truth_boundary")
    _require(isinstance(truth, Mapping), "truth boundary missing")
    for field in (
        "model_changed",
        "data_changed",
        "tokenizer_changed",
        "optimizer_family_changed",
        "quality_or_capability_claim",
        "paid_compute_authorized_or_used",
        "approximately_1m_executed",
    ):
        _require(truth.get(field) is False, f"truth boundary weakened: {field}")
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    _require(supplied_hash == _canonical_hash(unsigned), "report self-hash mismatch")


def summary_view(report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in report["model_runs"]:
        for result in run["results"]:
            rows.append(
                {
                    "parameter_count": run["parameter_count"],
                    "learning_rate": result["learning_rate"],
                    "classification": result["classification"],
                    "initial_validation_loss": result["initial_validation_loss"],
                    "final_validation_loss": result["final_validation_loss"],
                    "train_loss_first": result["train_loss_first"],
                    "train_loss_last": result["train_loss_last"],
                    "gradient_norm_max": result["gradient_norm_max"],
                    "clip_frequency": result["clip_frequency"],
                    "relative_update_l2_median": result["relative_update_l2_median"],
                    "relative_update_l2_max": result["relative_update_l2_max"],
                    "failure": result["failure"],
                }
            )
    return {
        "schema_version": "12-6.lr-range-summary.v1",
        "source_sha": report["source"]["git_sha"],
        "report_sha256": report["report_sha256"],
        "rows": rows,
        "consensus_by_learning_rate": report["summary"]["consensus_by_learning_rate"],
        "provisional_1m_learning_rate": report["summary"]["provisional_1m_learning_rate"],
        "transfer_rule": report["summary"]["transfer_rule"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--plan", type=Path, default=Path("configs/runs/train42_lr_range.experimental.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--summary-output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        root = args.repo_root.resolve()
        plan_path = args.plan if args.plan.is_absolute() else root / args.plan
        output_path = args.output if args.output.is_absolute() else root / args.output
        report = run_lr_range(
            repo_root=root,
            source_sha=args.source_sha,
            plan_path=plan_path,
            output_path=output_path,
        )
        summary_path = args.summary_output if args.summary_output.is_absolute() else root / args.summary_output
        summary_path.write_text(
            json.dumps(summary_view(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), "report must be an object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
