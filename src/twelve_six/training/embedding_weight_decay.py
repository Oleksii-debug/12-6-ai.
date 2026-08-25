"""TRAIN-44 controlled AdamW weight-decay grouping experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import platform
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

from .lr_range_experiment import controlled_family
from .optimization_experiments import (
    OptimizationRecipe,
    _batch_trace_sha256,
    _build_experiment_optimizer,
    _evaluate,
    _model_fingerprint,
    _optimizer_state_tensor_bytes,
    _tensor_batches,
)
from .trainer import Trainer

PLAN_SCHEMA = "12-6.embedding-weight-decay-plan.v1"
REPORT_SCHEMA = "12-6.embedding-weight-decay-evidence.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_EMBEDDING_WEIGHT_DECAY_EVIDENCE_PROVISIONAL"
REPOSITORY = "Oleksii-debug/12-6-ai."
EXPECTED_PARAMETERS = 467_808
EMBEDDING_PARAMETER_NAME = "token_embedding.weight"


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
    _require(plan.get("schema_version") == PLAN_SCHEMA, "wrong TRAIN-44 plan schema")
    _require(float(plan.get("learning_rate")) == 3e-4, "learning rate must remain 3e-4")
    optimizer = plan.get("optimizer")
    _require(isinstance(optimizer, dict), "optimizer block missing")
    _require(float(optimizer.get("weight_decay")) == 0.1, "weight decay must remain 0.1")
    _require(tuple(float(x) for x in optimizer.get("betas", ())) == (0.9, 0.95), "betas drift")
    _require(float(optimizer.get("eps")) == 1e-8, "epsilon drift")
    _require(optimizer.get("scheduler") == "constant", "scheduler must remain constant")
    _require(float(optimizer.get("gradient_clip_norm")) == 1.0, "clip norm drift")
    conditions = plan.get("conditions")
    _require(conditions == ["all_parameters_decayed", "embedding_excluded"], "condition grid drift")
    _require(int(plan.get("parameter_count")) == EXPECTED_PARAMETERS, "fixed model parameter count drift")
    steps = plan.get("execution_steps")
    checkpoint = plan.get("checkpoint_step")
    eval_every = plan.get("evaluation_interval_steps")
    _require(isinstance(steps, int) and steps >= 256, "run is too short to expose weight-decay effects")
    _require(isinstance(checkpoint, int) and 0 < checkpoint < steps, "bad checkpoint step")
    _require(isinstance(eval_every, int) and eval_every > 0 and steps % eval_every == 0, "bad eval interval")
    _require(checkpoint % eval_every == 0, "checkpoint must land on evaluation boundary")
    return plan


def fixed_model_spec() -> ModelSpec:
    spec = controlled_family()[2]
    _require(spec.parameter_count() == EXPECTED_PARAMETERS, "controlled 468K model drift")
    _require(spec.tie_word_embeddings, "TRAIN-44 requires tied embedding/output")
    _require(spec.vocab_size == 256 and spec.max_seq_len == 256, "fixed-control identity drift")
    return spec


def _recipe(plan: dict[str, Any], *, exclude_embedding: bool) -> OptimizationRecipe:
    optimizer = plan["optimizer"]
    return OptimizationRecipe(
        name="train44_embedding_excluded" if exclude_embedding else "train44_all_parameters_decayed",
        optimizer="adamw",
        learning_rate=float(plan["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        scheduler="constant",
        warmup_fraction=0.0,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        decay_embeddings=not exclude_embedding,
        precision="fp32",
    )


def _build_trainer(
    model: TwelveSixDecoder,
    *,
    plan: dict[str, Any],
    exclude_embedding: bool,
    max_steps: int,
) -> Trainer:
    recipe = _recipe(plan, exclude_embedding=exclude_embedding)
    config, _ = recipe.materialize(
        schedule_horizon_steps=max_steps,
        seed=int(plan["seed"]),
    )
    optimizer = _build_experiment_optimizer(model, config, recipe)
    return Trainer(model, config, device="cpu", optimizer=optimizer)


def _parameter_group_norms(model: TwelveSixDecoder) -> dict[str, float]:
    embedding = model.token_embedding.weight
    embedding_sq = float(torch.sum(embedding.detach().float() ** 2).item())
    other_sq = 0.0
    for parameter in model.parameters():
        if parameter is embedding:
            continue
        value = parameter.detach().float()
        other_sq += float(torch.sum(value * value).item())
    return {
        "embedding_l2": math.sqrt(embedding_sq),
        "non_embedding_l2": math.sqrt(other_sq),
    }


def _snapshot(model: TwelveSixDecoder) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _group_update_ratios(model: TwelveSixDecoder, before: Mapping[str, Tensor]) -> dict[str, float]:
    embedding = model.token_embedding.weight
    embedding_before = before[EMBEDDING_PARAMETER_NAME].float()
    embedding_delta = embedding.detach().float() - embedding_before
    embedding_before_norm = float(torch.linalg.vector_norm(embedding_before).item())
    embedding_update_norm = float(torch.linalg.vector_norm(embedding_delta).item())
    other_before_sq = 0.0
    other_delta_sq = 0.0
    for name, parameter in model.named_parameters():
        if parameter is embedding:
            continue
        old = before[name].float()
        delta = parameter.detach().float() - old
        other_before_sq += float(torch.sum(old * old).item())
        other_delta_sq += float(torch.sum(delta * delta).item())
    other_before_norm = math.sqrt(other_before_sq)
    return {
        "embedding_update_l2": embedding_update_norm,
        "embedding_relative_update_l2": (
            embedding_update_norm / embedding_before_norm if embedding_before_norm > 0.0 else math.inf
        ),
        "non_embedding_update_l2": math.sqrt(other_delta_sq),
        "non_embedding_relative_update_l2": (
            math.sqrt(other_delta_sq) / other_before_norm if other_before_norm > 0.0 else math.inf
        ),
    }


def _optimizer_serialized_bytes(trainer: Trainer) -> int:
    buffer = io.BytesIO()
    torch.save(trainer.optimizer.state_dict(), buffer)
    return len(buffer.getvalue())


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
            and all(_recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _model_states_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def resume_regression(
    *,
    spec: ModelSpec,
    init_spec: InitSpec,
    plan: dict[str, Any],
    train_batches: Sequence[Mapping[str, Tensor]],
    exclude_embedding: bool,
) -> dict[str, Any]:
    seed = int(plan["seed"])
    total_steps = 8
    split_step = 4
    batches = list(islice(cycle(train_batches), total_steps))
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = _build_trainer(
        model,
        plan=plan,
        exclude_embedding=exclude_embedding,
        max_steps=total_steps,
    )
    for batch in batches[:split_step]:
        trainer.train_microbatch(batch)
    checkpoint_model = copy.deepcopy(model.state_dict())
    checkpoint_trainer = trainer.state_dict()
    groups_at_checkpoint = [float(group["weight_decay"]) for group in trainer.optimizer.param_groups]
    for batch in batches[split_step:]:
        trainer.train_microbatch(batch)
    reference_model = copy.deepcopy(model.state_dict())
    reference_state = trainer.state_dict()

    torch.manual_seed(seed)
    resumed_model = TwelveSixDecoder(spec, init_spec)
    resumed_model.load_state_dict(checkpoint_model)
    resumed_trainer = _build_trainer(
        resumed_model,
        plan=plan,
        exclude_embedding=exclude_embedding,
        max_steps=total_steps,
    )
    resumed_trainer.load_state_dict(checkpoint_trainer)
    groups_after_load = [float(group["weight_decay"]) for group in resumed_trainer.optimizer.param_groups]
    for batch in batches[split_step:]:
        resumed_trainer.train_microbatch(batch)
    resumed_state = resumed_trainer.state_dict()
    model_match = _model_states_equal(reference_model, resumed_model.state_dict())
    optimizer_match = _recursive_equal(reference_state.optimizer, resumed_state.optimizer)
    scheduler_match = _recursive_equal(reference_state.scheduler, resumed_state.scheduler)
    counters_match = (
        reference_state.micro_step == resumed_state.micro_step
        and reference_state.optimizer_step == resumed_state.optimizer_step
        and reference_state.tokens_seen == resumed_state.tokens_seen
    )
    return {
        "passed": model_match and optimizer_match and scheduler_match and counters_match,
        "exact_final_model_match": model_match,
        "exact_optimizer_state_match": optimizer_match,
        "exact_scheduler_state_match": scheduler_match,
        "exact_counter_match": counters_match,
        "group_weight_decays_at_checkpoint": groups_at_checkpoint,
        "group_weight_decays_after_load": groups_after_load,
    }


def _run_candidate(
    *,
    spec: ModelSpec,
    init_spec: InitSpec,
    plan: dict[str, Any],
    train_batches: list[dict[str, Tensor]],
    validation_batches: list[dict[str, Tensor]],
    exclude_embedding: bool,
) -> dict[str, Any]:
    seed = int(plan["seed"])
    steps = int(plan["execution_steps"])
    checkpoint_step = int(plan["checkpoint_step"])
    eval_every = int(plan["evaluation_interval_steps"])
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    initial_model_sha = _model_fingerprint(model)
    initial_norms = _parameter_group_norms(model)
    initial_validation_loss = _evaluate(model, validation_batches)
    trainer = _build_trainer(
        model,
        plan=plan,
        exclude_embedding=exclude_embedding,
        max_steps=steps,
    )
    progression: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = [
        {
            "step": 0,
            "optimized_tokens": 0,
            "validation_loss": initial_validation_loss,
            "validation_bpb": initial_validation_loss / math.log(2.0),
        }
    ]
    resume_event: dict[str, Any] | None = None
    batch_list = list(islice(cycle(train_batches), steps))
    for step, batch in enumerate(batch_list, start=1):
        before = _snapshot(model)
        metrics = trainer.train_microbatch(batch)
        updates = _group_update_ratios(model, before)
        progression.append(
            {
                "step": step,
                "optimized_tokens": trainer.tokens_seen,
                "training_loss": float(metrics.loss),
                "gradient_norm": None if metrics.grad_norm is None else float(metrics.grad_norm),
                "learning_rate": float(metrics.learning_rate),
                **updates,
            }
        )
        if step % eval_every == 0:
            validation_loss = _evaluate(model, validation_batches)
            held_out.append(
                {
                    "step": step,
                    "optimized_tokens": trainer.tokens_seen,
                    "validation_loss": validation_loss,
                    "validation_bpb": validation_loss / math.log(2.0),
                }
            )
            model.train()
        if step == checkpoint_step:
            checkpoint_model = copy.deepcopy(model.state_dict())
            checkpoint_trainer = trainer.state_dict()
            groups_before = [float(group["weight_decay"]) for group in trainer.optimizer.param_groups]
            fresh_model = TwelveSixDecoder(spec, init_spec)
            fresh_model.load_state_dict(checkpoint_model)
            fresh_trainer = _build_trainer(
                fresh_model,
                plan=plan,
                exclude_embedding=exclude_embedding,
                max_steps=steps,
            )
            fresh_trainer.load_state_dict(checkpoint_trainer)
            groups_after = [float(group["weight_decay"]) for group in fresh_trainer.optimizer.param_groups]
            _require(groups_after == groups_before, "parameter groups changed across resume")
            resume_event = {
                "step": checkpoint_step,
                "optimized_tokens": fresh_trainer.tokens_seen,
                "group_weight_decays_before": groups_before,
                "group_weight_decays_after": groups_after,
                "model_sha256_before": _model_fingerprint(model),
                "model_sha256_after": _model_fingerprint(fresh_model),
            }
            _require(
                resume_event["model_sha256_before"] == resume_event["model_sha256_after"],
                "model state changed across resume",
            )
            model = fresh_model
            trainer = fresh_trainer

    final_norms = _parameter_group_norms(model)
    embedding_updates = [float(row["embedding_relative_update_l2"]) for row in progression]
    other_updates = [float(row["non_embedding_relative_update_l2"]) for row in progression]
    losses = [float(row["training_loss"]) for row in progression]
    _require(resume_event is not None, "long run did not exercise checkpoint/resume")
    return {
        "condition": "embedding_excluded" if exclude_embedding else "all_parameters_decayed",
        "embedding_excluded_from_weight_decay": exclude_embedding,
        "status": "PASS",
        "initial_model_sha256": initial_model_sha,
        "batch_trace_sha256": _batch_trace_sha256(train_batches, steps),
        "optimizer_group_weight_decays": [float(group["weight_decay"]) for group in trainer.optimizer.param_groups],
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "training_loss_median": statistics.median(losses),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": held_out[-1]["validation_loss"],
        "final_validation_bpb": held_out[-1]["validation_bpb"],
        "embedding_norm_initial": initial_norms["embedding_l2"],
        "embedding_norm_final": final_norms["embedding_l2"],
        "embedding_norm_relative_change": final_norms["embedding_l2"] / initial_norms["embedding_l2"] - 1.0,
        "non_embedding_norm_initial": initial_norms["non_embedding_l2"],
        "non_embedding_norm_final": final_norms["non_embedding_l2"],
        "non_embedding_norm_relative_change": final_norms["non_embedding_l2"] / initial_norms["non_embedding_l2"] - 1.0,
        "embedding_relative_update_l2_median": statistics.median(embedding_updates),
        "embedding_relative_update_l2_max": max(embedding_updates),
        "non_embedding_relative_update_l2_median": statistics.median(other_updates),
        "non_embedding_relative_update_l2_max": max(other_updates),
        "optimizer_state_tensor_bytes": _optimizer_state_tensor_bytes(trainer),
        "optimizer_state_serialized_bytes": _optimizer_serialized_bytes(trainer),
        "optimized_tokens": trainer.tokens_seen,
        "resume_event": resume_event,
        "held_out": held_out,
        "progression": progression,
    }


def recommendation(results: list[dict[str, Any]], resume_checks: dict[str, Any]) -> dict[str, Any]:
    by_name = {row["condition"]: row for row in results}
    all_decay = by_name["all_parameters_decayed"]
    excluded = by_name["embedding_excluded"]
    _require(all(bool(check["passed"]) for check in resume_checks.values()), "resume regression failed")
    all_bpb = float(all_decay["final_validation_bpb"])
    excluded_bpb = float(excluded["final_validation_bpb"])
    improvement_fraction = (all_bpb - excluded_bpb) / all_bpb
    if excluded_bpb <= 1.005 * all_bpb:
        selected = "embedding_excluded"
        rule = (
            "Exclude the tied token embedding/output matrix from AdamW weight decay provisionally; "
            "keep weight decay on non-embedding parameters. The exclusion is retained when held-out BPB "
            "is no worse than 0.5% versus all-decay and grouped checkpoint/resume is exact."
        )
    else:
        selected = "all_parameters_decayed"
        rule = (
            "Decay the tied embedding with the rest of the model provisionally because all-parameter decay "
            "improved held-out BPB by more than the predeclared 0.5% tolerance on this controlled run."
        )
    return {
        "selected_default_grouping": selected,
        "excluded_minus_all_decay_bpb_fraction": -improvement_fraction,
        "rule": rule,
        "selection_tolerance_fraction": 0.005,
        "single_seed_fixture_scoped": True,
    }


def run_experiment(*, repo_root: Path, source_sha: str, plan_path: Path, output_path: Path) -> dict[str, Any]:
    _require(_git_head(repo_root) == source_sha, "exact source checkout mismatch")
    plan = load_plan(plan_path)
    torch.set_num_threads(int(plan["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    spec = fixed_model_spec()
    init_spec = InitSpec()
    tokenizer = ByteTokenizer()
    train_batches, train_ids, _, train_max = _tensor_batches(
        repo_root,
        split="train",
        tokenizer=tokenizer,
        batch_size=int(plan["batch_size"]),
        sequence_length=int(plan["sequence_length"]),
    )
    validation_batches, validation_ids, validation_tokens, validation_max = _tensor_batches(
        repo_root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=int(plan["batch_size"]),
        sequence_length=int(plan["sequence_length"]),
    )
    _require(not (set(train_ids) & set(validation_ids)), "train/validation record overlap")
    _require(max(train_max, validation_max) < spec.vocab_size, "fixture token exceeds vocab")
    resume_checks = {
        condition: resume_regression(
            spec=spec,
            init_spec=init_spec,
            plan=plan,
            train_batches=train_batches,
            exclude_embedding=(condition == "embedding_excluded"),
        )
        for condition in plan["conditions"]
    }
    results = [
        _run_candidate(
            spec=spec,
            init_spec=init_spec,
            plan=plan,
            train_batches=train_batches,
            validation_batches=validation_batches,
            exclude_embedding=(condition == "embedding_excluded"),
        )
        for condition in plan["conditions"]
    ]
    _require(len({row["initial_model_sha256"] for row in results}) == 1, "initialization drift")
    _require(len({row["batch_trace_sha256"] for row in results}) == 1, "batch trace drift")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "paid_compute": False},
        "plan": plan,
        "model": {
            "parameter_count": spec.parameter_count(),
            "modelspec_sha256": spec.identity_sha256(),
            "tied_embedding_output": spec.tie_word_embeddings,
            "embedding_parameter_name": EMBEDDING_PARAMETER_NAME,
            "validation_scoreable_tokens": validation_tokens,
        },
        "resume_regressions": resume_checks,
        "results": results,
        "provisional_grouping": recommendation(results, resume_checks),
        "truth_boundary": {
            "trainer_reimplemented": False,
            "optimizer_family_changed": False,
            "learning_rate_changed_across_conditions": False,
            "betas_changed_across_conditions": False,
            "weight_decay_coefficient_changed_across_conditions": False,
            "only_embedding_decay_membership_changed": True,
            "representative_corpus_claim": False,
            "paid_compute_authorized_or_used": False,
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
    model = report.get("model")
    _require(isinstance(model, dict) and model.get("parameter_count") == EXPECTED_PARAMETERS, "model drift")
    _require(model.get("tied_embedding_output") is True, "embedding must be tied")
    resume = report.get("resume_regressions")
    _require(isinstance(resume, dict) and set(resume) == {"all_parameters_decayed", "embedding_excluded"}, "resume checks missing")
    _require(all(check.get("passed") is True for check in resume.values()), "resume regression failed")
    results = report.get("results")
    _require(isinstance(results, list) and len(results) == 2, "two conditions required")
    _require({row["condition"] for row in results} == {"all_parameters_decayed", "embedding_excluded"}, "condition drift")
    _require(len({row["initial_model_sha256"] for row in results}) == 1, "init drift")
    _require(len({row["batch_trace_sha256"] for row in results}) == 1, "batch trace drift")
    for row in results:
        _require(row.get("status") == "PASS", "candidate failed")
        _require(math.isfinite(float(row["final_validation_bpb"])), "non-finite BPB")
        _require(int(row["optimized_tokens"]) > 0, "no optimized tokens")
        _require(row["resume_event"]["model_sha256_before"] == row["resume_event"]["model_sha256_after"], "resume model mismatch")
    expected = dict(report)
    claimed_hash = expected.pop("report_sha256", None)
    _require(isinstance(claimed_hash, str) and claimed_hash == _canonical_hash(expected), "report hash mismatch")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha": report["source"]["git_sha"],
        "parameter_count": report["model"]["parameter_count"],
        "optimized_tokens_per_condition": [row["optimized_tokens"] for row in report["results"]],
        "results": [
            {
                "condition": row["condition"],
                "final_validation_bpb": row["final_validation_bpb"],
                "training_loss_last": row["training_loss_last"],
                "embedding_norm_initial": row["embedding_norm_initial"],
                "embedding_norm_final": row["embedding_norm_final"],
                "non_embedding_norm_initial": row["non_embedding_norm_initial"],
                "non_embedding_norm_final": row["non_embedding_norm_final"],
                "embedding_relative_update_l2_median": row["embedding_relative_update_l2_median"],
                "non_embedding_relative_update_l2_median": row["non_embedding_relative_update_l2_median"],
                "optimizer_state_tensor_bytes": row["optimizer_state_tensor_bytes"],
                "optimizer_state_serialized_bytes": row["optimizer_state_serialized_bytes"],
                "optimizer_group_weight_decays": row["optimizer_group_weight_decays"],
            }
            for row in report["results"]
        ],
        "resume_regressions": report["resume_regressions"],
        "provisional_grouping": report["provisional_grouping"],
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
