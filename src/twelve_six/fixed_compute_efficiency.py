"""Equal-compute successor to the RESEARCH06 exact-token scaling experiment.

RESEARCH07 deliberately imports RESEARCH06's exact aligned causal-pair packing,
held-out evaluator, data split, controlled ModelSpecs, initialization family and
optimizer recipe.  Only the stopping budget changes: each candidate receives the
largest integer number of valid causal loss tokens whose explicit ``6 * N * T``
proxy does not exceed one shared LOCAL_FREE engineering-compute envelope.

CPU wall time is measured separately and is never treated as a portable compute
unit or as evidence about paid accelerator economics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from . import fixed_token_efficiency as ft
from .checkpoint.core import CheckpointIdentity, hash_json
from .checkpoint.trainer_adapter import load_trainer_checkpoint, save_trainer_checkpoint
from .model import TwelveSixDecoder
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH
from .training import Trainer

SCHEMA = "12-6.fixed-compute-efficiency.v1"
MODEL_SCHEMA = "12-6.fixed-compute-efficiency-model.v1"
PARTIAL_SCHEMA = "12-6.fixed-compute-efficiency-partial.v1"
AUTHORITY = "LOCAL_FREE_FIXED_COMPUTE_GENERALIZATION_EVIDENCE_NOT_PROMOTION"
DEFAULT_COMPUTE_BUDGET = 105_000_000_000
EXPECTED_COUNTS = (95_568, 267_912, 467_808, 1_037_696)
COMPUTE_PROXY = "6 * trainable_parameters * optimized_causal_loss_tokens"


def _canonical_hash(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("report_sha256", None)
    return hash_json(body)


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["report_sha256"] = _canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return body


def _read_hashed_json(path: Path, *, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise RuntimeError(f"{path} schema mismatch")
    recorded = payload.get("report_sha256")
    if not isinstance(recorded, str) or recorded != _canonical_hash(payload):
        raise RuntimeError(f"{path} self-hash mismatch")
    return payload


def budget_plan(*, parameters: int, compute_budget: int) -> dict[str, int]:
    """Return the maximal exact causal-token target that stays inside the proxy budget."""
    if parameters <= 0 or compute_budget <= 0:
        raise ValueError("parameters and compute_budget must be positive")
    compute_per_token = 6 * parameters
    optimized_tokens = compute_budget // compute_per_token
    if optimized_tokens < 2:
        raise ValueError("compute budget must fund at least two valid causal loss tokens")
    compute_proxy = compute_per_token * optimized_tokens
    remainder = compute_budget - compute_proxy
    if not 0 <= remainder < compute_per_token:
        raise RuntimeError("compute-budget flooring drift")
    return {
        "requested_compute_budget": compute_budget,
        "compute_per_optimized_token": compute_per_token,
        "optimized_tokens": optimized_tokens,
        "compute_proxy": compute_proxy,
        "compute_remainder": remainder,
    }


def _compute_bundle(
    *,
    repo_root: Path,
    source_sha: str,
    model_index: int,
    compute_budget: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    specs = ft.controlled_specs()
    counts = tuple(spec.parameter_count() for spec in specs)
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"RESEARCH41 control-family drift: {counts!r}")
    if not 0 <= model_index < len(specs):
        raise ValueError(f"model_index must be in [0,{len(specs) - 1}]")
    spec = specs[model_index]
    plan = budget_plan(parameters=spec.parameter_count(), compute_budget=compute_budget)
    data = ft._load_control_data(repo_root)
    init_spec = ft.InitSpec()
    trainer_config = ft._trainer_config(
        final_tokens=plan["optimized_tokens"],
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    training_config = {
        "recipe": "research41-fixed-control-adamw-v1",
        "trainer": asdict(trainer_config),
        "init_spec_sha256": init_spec.identity_sha256(),
        "data": {
            "split_identity": data["split_identity"],
            "packing_sha256": ft.PACKING_SHA256,
            "packing_version": ft.PACKING_VERSION,
        },
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "budget_mode": "fixed_compute",
        "compute_proxy_definition": COMPUTE_PROXY,
        "requested_compute_budget": compute_budget,
        "target_optimized_tokens": plan["optimized_tokens"],
        "actual_compute_proxy": plan["compute_proxy"],
        "compute_remainder": plan["compute_remainder"],
    }
    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
        "dataset_manifest_hash": data["dataset_manifest_hash"],
        "training_config": training_config,
    }
    controls_hash = hash_json(run_manifest)
    return {
        "spec": spec,
        "plan": plan,
        "data": data,
        "init_spec": init_spec,
        "trainer_config": trainer_config,
        "training_config": training_config,
        "run_manifest_hash": controls_hash,
        "controls_hash": controls_hash,
    }


def _checkpoint_identity(
    *, source_sha: str, bundle: dict[str, Any], trainer: Trainer
) -> CheckpointIdentity:
    config = bundle["trainer_config"]
    spec = bundle["spec"]
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=bundle["data"]["dataset_manifest_hash"],
        run_manifest_hash=bundle["run_manifest_hash"],
        training_config=bundle["training_config"],
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler=None,
    )


def _base_state(*, initial_loss: float, validation_tokens: int) -> dict[str, Any]:
    return {
        "initial_validation_loss": initial_loss,
        "initial_validation_bpb": initial_loss / math.log(2.0),
        "validation_tokens": validation_tokens,
        "training_wall_seconds": 0.0,
        "grad_norms": [],
        "clip_count": 0,
        "update_ratios": [],
        "last_train_loss": None,
    }


def _train_until(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    bundle: dict[str, Any],
    stop_tokens: int,
    state: dict[str, Any],
    batch_size: int,
    sequence_length: int,
) -> None:
    if stop_tokens <= trainer.tokens_seen:
        raise ValueError("stop_tokens must exceed current optimized-token count")
    capacity = batch_size * sequence_length
    stream = bundle["data"]["train_stream"]
    clip_threshold = bundle["trainer_config"].gradient_clip_norm
    while trainer.tokens_seen < stop_tokens:
        before_tokens = trainer.tokens_seen
        valid_pairs = min(capacity, stop_tokens - before_tokens)
        planned_after = before_tokens + valid_pairs
        sample_update = (
            trainer.optimizer_step == 0
            or (trainer.optimizer_step + 1) % 32 == 0
            or planned_after == stop_tokens
        )
        snapshot = ft._parameter_snapshot(model) if sample_update else None
        batch = ft._make_pair_batch(
            stream,
            causal_offset=before_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_pairs=valid_pairs,
        )
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        state["training_wall_seconds"] += time.perf_counter() - started
        if metrics.grad_norm is None:
            raise RuntimeError("committed optimizer update did not report gradient norm")
        grad_norm = float(metrics.grad_norm)
        state["grad_norms"].append(grad_norm)
        if clip_threshold is not None and grad_norm > clip_threshold:
            state["clip_count"] += 1
        if snapshot is not None:
            state["update_ratios"].append(
                {
                    "optimizer_step": trainer.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "relative_l2_update": ft._relative_update_ratio(model, snapshot),
                }
            )
        state["last_train_loss"] = float(metrics.loss)
        ft._assert_token_transition(
            before=before_tokens,
            metrics_tokens=metrics.tokens,
            after=trainer.tokens_seen,
            requested=valid_pairs,
        )
        if trainer.tokens_seen > stop_tokens:
            raise RuntimeError("optimized-token budget overshoot")
    expected_steps = math.ceil(stop_tokens / capacity)
    if trainer.optimizer_step != expected_steps:
        raise RuntimeError(
            f"optimizer-step drift: expected {expected_steps}, observed {trainer.optimizer_step}"
        )


def _evaluate(
    *, model: TwelveSixDecoder, trainer: Trainer, bundle: dict[str, Any], expected_tokens: int
) -> tuple[float, float]:
    loss, validation_tokens, elapsed = ft._evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
        expected_validation_tokens=expected_tokens,
    )
    if validation_tokens != expected_tokens:
        raise RuntimeError("held-out validation target count drift")
    return loss, elapsed


def start_model(
    *,
    repo_root: Path,
    source_sha: str,
    model_index: int,
    compute_budget: int,
    partial_path: Path,
    checkpoint_dir: Path,
    batch_size: int = 4,
    sequence_length: int = 64,
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    ft._assert_exact_source(repo_root, source_sha)
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint destination already exists: {checkpoint_dir}")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    bundle = _compute_bundle(
        repo_root=repo_root,
        source_sha=source_sha,
        model_index=model_index,
        compute_budget=compute_budget,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    target_tokens = int(bundle["plan"]["optimized_tokens"])
    resume_tokens = max(1, target_tokens // 2)
    if resume_tokens >= target_tokens:
        raise RuntimeError("compute target cannot support a distinct resume boundary")

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    phase_started = time.perf_counter()
    initial_loss, validation_tokens, initial_eval_seconds = ft._evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
    )
    if trainer.tokens_seen != 0 or trainer.optimizer_step != 0:
        raise RuntimeError("initial evaluation contaminated optimization accounting")
    state = _base_state(initial_loss=initial_loss, validation_tokens=validation_tokens)
    state["initial_evaluation_wall_seconds"] = initial_eval_seconds
    _train_until(
        model=model,
        trainer=trainer,
        bundle=bundle,
        stop_tokens=resume_tokens,
        state=state,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    resume_loss, resume_eval_seconds = _evaluate(
        model=model,
        trainer=trainer,
        bundle=bundle,
        expected_tokens=validation_tokens,
    )
    trainer.assert_checkpoint_safe()
    save_started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=_checkpoint_identity(source_sha=source_sha, bundle=bundle, trainer=trainer),
    )
    checkpoint_save_seconds = time.perf_counter() - save_started
    start_wall_seconds = time.perf_counter() - phase_started
    payload = {
        "schema": PARTIAL_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "model_index": model_index,
        "parameters": bundle["spec"].parameter_count(),
        "model_identity_sha256": bundle["spec"].identity_sha256(),
        "controls_hash": bundle["controls_hash"],
        "compute_budget": compute_budget,
        "plan": bundle["plan"],
        "resume_tokens": resume_tokens,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "start_pid": os.getpid(),
        "state": state,
        "resume_validation_loss": resume_loss,
        "resume_evaluation_wall_seconds": resume_eval_seconds,
        "start_wall_seconds": start_wall_seconds,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "checkpoint_id": manifest["checkpoint_id"],
            "save_seconds": checkpoint_save_seconds,
            "bytes": ft._directory_bytes(checkpoint_dir),
        },
        "start_memory": {
            "parameter_bytes": ft._parameter_bytes(model),
            "optimizer_tensor_bytes": ft._tensor_tree_bytes(trainer.optimizer.state_dict()),
            "process_peak_rss_bytes": ft._peak_rss_bytes(),
        },
    }
    return _write_hashed_json(partial_path, payload)


def resume_model(
    *,
    repo_root: Path,
    source_sha: str,
    partial_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    torch_threads: int = 2,
) -> dict[str, Any]:
    ft._assert_exact_source(repo_root, source_sha)
    partial = _read_hashed_json(partial_path, schema=PARTIAL_SCHEMA)
    if partial["source_sha"] != source_sha:
        raise RuntimeError("partial evidence source SHA mismatch")
    if int(partial["start_pid"]) == os.getpid():
        raise RuntimeError("resume must execute in a fresh process")
    model_index = int(partial["model_index"])
    compute_budget = int(partial["compute_budget"])
    batch_size = int(partial["batch_size"])
    sequence_length = int(partial["sequence_length"])
    seed = int(partial["seed"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    bundle = _compute_bundle(
        repo_root=repo_root,
        source_sha=source_sha,
        model_index=model_index,
        compute_budget=compute_budget,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    if bundle["controls_hash"] != partial["controls_hash"]:
        raise RuntimeError("scientific controls drifted across fresh-process resume")
    if bundle["plan"] != partial["plan"]:
        raise RuntimeError("compute budget plan drifted across fresh-process resume")

    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    phase_started = time.perf_counter()
    load_started = time.perf_counter()
    load_result = load_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=bundle["spec"].identity_sha256(),
        expected_init_spec_hash=bundle["init_spec"].identity_sha256(),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=bundle["data"]["dataset_manifest_hash"],
        expected_split_identity=bundle["data"]["split_identity"],
        expected_packing_hash=ft.PACKING_SHA256,
        expected_packing_version=ft.PACKING_VERSION,
        expected_run_manifest_hash=bundle["run_manifest_hash"],
        expected_training_config_hash=hash_json(bundle["training_config"]),
        expected_seed=seed,
    )
    checkpoint_load_seconds = time.perf_counter() - load_started
    resume_tokens = int(partial["resume_tokens"])
    if trainer.tokens_seen != resume_tokens:
        raise RuntimeError("checkpoint optimized-token counter drift")
    if int(load_result.manifest["identity"]["tokens_seen"]) != resume_tokens:
        raise RuntimeError("verified checkpoint identity token count mismatch")

    state = dict(partial["state"])
    state["grad_norms"] = list(state["grad_norms"])
    state["update_ratios"] = list(state["update_ratios"])
    resume_loss, resume_eval_seconds = _evaluate(
        model=model,
        trainer=trainer,
        bundle=bundle,
        expected_tokens=int(state["validation_tokens"]),
    )
    resume_validation_abs_diff = abs(float(partial["resume_validation_loss"]) - resume_loss)
    if resume_validation_abs_diff > 1e-12:
        raise RuntimeError(
            "fresh-process resume changed held-out validation: "
            f"abs_diff={resume_validation_abs_diff:.3e}"
        )

    target_tokens = int(bundle["plan"]["optimized_tokens"])
    _train_until(
        model=model,
        trainer=trainer,
        bundle=bundle,
        stop_tokens=target_tokens,
        state=state,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    final_loss, final_eval_seconds = _evaluate(
        model=model,
        trainer=trainer,
        bundle=bundle,
        expected_tokens=int(state["validation_tokens"]),
    )
    if trainer.tokens_seen != target_tokens:
        raise RuntimeError("final optimized-token target drift")
    parameters = bundle["spec"].parameter_count()
    actual_compute = 6 * parameters * trainer.tokens_seen
    if actual_compute != int(bundle["plan"]["compute_proxy"]):
        raise RuntimeError("final compute-proxy accounting drift")
    if actual_compute > compute_budget:
        raise RuntimeError("candidate exceeded fixed compute envelope")
    if compute_budget - actual_compute >= 6 * parameters:
        raise RuntimeError("candidate stopped before maximal one-token compute boundary")

    resume_process_wall_seconds = time.perf_counter() - phase_started
    total_wall_seconds = float(partial["start_wall_seconds"]) + resume_process_wall_seconds
    training_wall_seconds = float(state["training_wall_seconds"])
    grad_norms = [float(value) for value in state["grad_norms"]]
    update_ratios = [float(item["relative_l2_update"]) for item in state["update_ratios"]]
    if not grad_norms or not update_ratios or state["last_train_loss"] is None:
        raise RuntimeError("required optimization telemetry is empty")
    final_memory = {
        "parameter_bytes": ft._parameter_bytes(model),
        "optimizer_tensor_bytes": ft._tensor_tree_bytes(trainer.optimizer.state_dict()),
        "process_peak_rss_bytes": ft._peak_rss_bytes(),
    }
    peak_process_rss = max(
        int(partial["start_memory"]["process_peak_rss_bytes"]),
        int(final_memory["process_peak_rss_bytes"]),
    )
    initial_loss = float(state["initial_validation_loss"])
    validation_improvement = initial_loss - final_loss
    checkpoint = partial["checkpoint"]
    result = {
        "schema": MODEL_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "model_index": model_index,
        "parameters": parameters,
        "model_identity_sha256": bundle["spec"].identity_sha256(),
        "controls_hash": bundle["controls_hash"],
        "compute_proxy_definition": COMPUTE_PROXY,
        "budget": bundle["plan"],
        "accounting": {
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
            "evaluation_tokens_per_pass": int(state["validation_tokens"]),
            "evaluation_optimized_tokens": 0,
            "exact_token_target": True,
            "maximal_compute_floor": True,
        },
        "generalization": {
            "initial_validation_loss": initial_loss,
            "initial_validation_bpb": float(state["initial_validation_bpb"]),
            "resume_validation_loss": resume_loss,
            "resume_validation_abs_diff": resume_validation_abs_diff,
            "validation_loss": final_loss,
            "validation_bpb": final_loss / math.log(2.0),
            "validation_improvement": validation_improvement,
        },
        "optimization": {
            "final_train_loss": float(state["last_train_loss"]),
            "gradient_norm_min": min(grad_norms),
            "gradient_norm_mean": statistics.fmean(grad_norms),
            "gradient_norm_max": max(grad_norms),
            "clip_count": int(state["clip_count"]),
            "clip_fraction": int(state["clip_count"]) / len(grad_norms),
            "sampled_update_ratio_min": min(update_ratios),
            "sampled_update_ratio_mean": statistics.fmean(update_ratios),
            "sampled_update_ratio_max": max(update_ratios),
            "sampled_update_ratios": state["update_ratios"],
        },
        "timing": {
            "training_wall_seconds": training_wall_seconds,
            "end_to_end_wall_seconds": total_wall_seconds,
            "optimized_tokens_per_training_second": trainer.tokens_seen / training_wall_seconds,
            "initial_evaluation_wall_seconds": float(state["initial_evaluation_wall_seconds"]),
            "resume_evaluation_wall_seconds": resume_eval_seconds,
            "final_evaluation_wall_seconds": final_eval_seconds,
        },
        "memory": {
            "parameter_bytes": final_memory["parameter_bytes"],
            "optimizer_tensor_bytes": final_memory["optimizer_tensor_bytes"],
            "peak_process_rss_bytes": peak_process_rss,
            "rss_scope": "max of isolated start and resume process high-water marks",
        },
        "checkpoint_resume": {
            "fresh_process": True,
            "resume_tokens": resume_tokens,
            "validation_abs_diff": resume_validation_abs_diff,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_bytes": int(checkpoint["bytes"]),
            "checkpoint_save_seconds": float(checkpoint["save_seconds"]),
            "checkpoint_load_seconds": checkpoint_load_seconds,
        },
        "scheduler_semantics": {
            "name": bundle["trainer_config"].scheduler,
            "warmup_steps": bundle["trainer_config"].warmup_steps,
            "learning_rate": bundle["trainer_config"].learning_rate,
            "max_steps": bundle["trainer_config"].max_steps,
            "note": (
                "constant scheduler with zero warmup; candidate-specific max_steps is only "
                "the exact stopping bound and does not reshape learning rate"
            ),
        },
        "efficiency": {
            "validation_improvement_per_compute_proxy": validation_improvement / actual_compute,
            "validation_improvement_per_training_wall_second": (
                validation_improvement / training_wall_seconds
            ),
        },
        "truth_boundary": {
            "held_out_validation_is_decision_metric": True,
            "train_loss_is_generalization_metric": False,
            "cpu_wall_time_is_universal_compute": False,
            "paid_compute": False,
            "gpu_economics_extrapolation": False,
            "broad_corpus_claim": False,
            "stage_promotion": False,
        },
    }
    return _write_hashed_json(output_path, result)


def _rank(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=not lower_is_better)
    return [
        {"rank": rank, "parameters": int(row["parameters"]), "score": float(row[key])}
        for rank, row in enumerate(ordered, 1)
    ]


def aggregate_models(
    *, inputs: list[Path], output_path: Path, expected_source_sha: str
) -> dict[str, Any]:
    models = [_read_hashed_json(path, schema=MODEL_SCHEMA) for path in inputs]
    models.sort(key=lambda model: int(model["parameters"]))
    if [int(model["parameters"]) for model in models] != list(EXPECTED_COUNTS):
        raise RuntimeError("aggregate model family drift")
    if any(model["source_sha"] != expected_source_sha for model in models):
        raise RuntimeError("aggregate source SHA mismatch")
    budgets = {int(model["budget"]["requested_compute_budget"]) for model in models}
    if len(budgets) != 1:
        raise RuntimeError("candidates did not share one compute envelope")
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.append(
            {
                "parameters": int(model["parameters"]),
                "optimized_tokens": int(model["accounting"]["optimized_tokens"]),
                "compute_proxy": int(model["budget"]["compute_proxy"]),
                "compute_remainder": int(model["budget"]["compute_remainder"]),
                "final_train_loss": float(model["optimization"]["final_train_loss"]),
                "validation_loss": float(model["generalization"]["validation_loss"]),
                "validation_bpb": float(model["generalization"]["validation_bpb"]),
                "validation_improvement": float(
                    model["generalization"]["validation_improvement"]
                ),
                "training_wall_seconds": float(model["timing"]["training_wall_seconds"]),
                "end_to_end_wall_seconds": float(model["timing"]["end_to_end_wall_seconds"]),
                "peak_process_rss_bytes": int(model["memory"]["peak_process_rss_bytes"]),
                "checkpoint_bytes": int(model["checkpoint_resume"]["checkpoint_bytes"]),
                "checkpoint_save_seconds": float(
                    model["checkpoint_resume"]["checkpoint_save_seconds"]
                ),
                "checkpoint_load_seconds": float(
                    model["checkpoint_resume"]["checkpoint_load_seconds"]
                ),
                "validation_improvement_per_compute_proxy": float(
                    model["efficiency"]["validation_improvement_per_compute_proxy"]
                ),
                "validation_improvement_per_training_wall_second": float(
                    model["efficiency"]["validation_improvement_per_training_wall_second"]
                ),
            }
        )
    rankings = {
        "best_validation": _rank(rows, "validation_loss", lower_is_better=True),
        "validation_improvement_per_compute": _rank(
            rows, "validation_improvement_per_compute_proxy", lower_is_better=False
        ),
        "validation_improvement_per_training_wall_second": _rank(
            rows,
            "validation_improvement_per_training_wall_second",
            lower_is_better=False,
        ),
    }
    winner = rankings["best_validation"][0]
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": expected_source_sha,
        "incumbent": "RESEARCH06 exact-token harness / RESEARCH41 fixed-control family",
        "compute_proxy_definition": COMPUTE_PROXY,
        "requested_compute_budget": next(iter(budgets)),
        "parameter_counts": list(EXPECTED_COUNTS),
        "matrix": rows,
        "rankings": rankings,
        "recommendation": {
            "strongest_held_out_under_equal_engineering_compute_parameters": int(
                winner["parameters"]
            ),
            "selection_rule": "lowest held-out validation loss under the common 6*N*T envelope",
            "fixture_scoped": True,
        },
        "scientific_controls": {
            "tokenizer": "RESEARCH06/RESEARCH41 byte tokenizer",
            "corpus_split": "data/s0/packaged train and held-out validation",
            "packing": ft.PACKING_VERSION,
            "context": 256,
            "training_sequence_length": 64,
            "batch_size": 4,
            "init_family": "InitSpec v1",
            "optimizer": "AdamW lr=3e-4 betas=(0.9,0.95) eps=1e-8 wd=0",
            "precision": "fp32",
            "seed": 1337,
            "scheduler": "constant, warmup_steps=0",
        },
        "truth_boundary": {
            "tiny_recycled_project_fixture": True,
            "held_out_generalization_only": True,
            "cpu_wall_time_is_universal_compute": False,
            "paid_compute": False,
            "gpu_economics_extrapolation": False,
            "representative_corpus_claim": False,
            "stage_promotion": False,
        },
    }
    return _write_hashed_json(output_path, report)


def validate_report(payload: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    schema = payload.get("schema")
    if schema not in {MODEL_SCHEMA, SCHEMA}:
        raise RuntimeError(f"unsupported report schema: {schema!r}")
    if payload.get("authority") != AUTHORITY:
        raise RuntimeError("report authority mismatch")
    if expected_source_sha is not None and payload.get("source_sha") != expected_source_sha:
        raise RuntimeError("report source SHA mismatch")
    recorded = payload.get("report_sha256")
    if not isinstance(recorded, str) or recorded != _canonical_hash(payload):
        raise RuntimeError("report self-hash mismatch")
    if schema == MODEL_SCHEMA:
        parameters = int(payload["parameters"])
        if parameters not in EXPECTED_COUNTS:
            raise RuntimeError("candidate parameter count outside fixed-control family")
        budget = payload["budget"]
        optimized_tokens = int(payload["accounting"]["optimized_tokens"])
        expected_plan = budget_plan(
            parameters=parameters,
            compute_budget=int(budget["requested_compute_budget"]),
        )
        if expected_plan != budget:
            raise RuntimeError("candidate compute plan drift")
        if optimized_tokens != int(budget["optimized_tokens"]):
            raise RuntimeError("candidate optimized-token accounting drift")
        if int(payload["accounting"]["evaluation_optimized_tokens"]) != 0:
            raise RuntimeError("evaluation contaminated optimized-token accounting")
        if payload["checkpoint_resume"].get("fresh_process") is not True:
            raise RuntimeError("fresh-process resume evidence missing")
        if float(payload["checkpoint_resume"]["validation_abs_diff"]) > 1e-12:
            raise RuntimeError("checkpoint resume validation parity failed")
        if not math.isfinite(float(payload["generalization"]["validation_loss"])):
            raise RuntimeError("non-finite held-out validation loss")
        if not math.isfinite(float(payload["generalization"]["validation_bpb"])):
            raise RuntimeError("non-finite held-out BPB")
        if payload["truth_boundary"].get("train_loss_is_generalization_metric") is not False:
            raise RuntimeError("train loss incorrectly promoted to generalization metric")
        return
    rows = payload.get("matrix")
    if not isinstance(rows, list) or len(rows) != 4:
        raise RuntimeError("aggregate must contain four candidates")
    if [int(row["parameters"]) for row in rows] != list(EXPECTED_COUNTS):
        raise RuntimeError("aggregate parameter family drift")
    compute_budget = int(payload["requested_compute_budget"])
    previous_tokens = None
    for row in rows:
        parameters = int(row["parameters"])
        plan = budget_plan(parameters=parameters, compute_budget=compute_budget)
        if int(row["optimized_tokens"]) != plan["optimized_tokens"]:
            raise RuntimeError("aggregate optimized-token target drift")
        if int(row["compute_proxy"]) != plan["compute_proxy"]:
            raise RuntimeError("aggregate compute proxy drift")
        if previous_tokens is not None and int(row["optimized_tokens"]) >= previous_tokens:
            raise RuntimeError("larger candidate did not receive fewer optimized tokens")
        previous_tokens = int(row["optimized_tokens"])
    if payload["truth_boundary"].get("cpu_wall_time_is_universal_compute") is not False:
        raise RuntimeError("CPU wall time truth boundary weakened")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--repo-root", type=Path, default=Path("."))
    start.add_argument("--source-sha", required=True)
    start.add_argument("--model-index", type=int, required=True)
    start.add_argument("--compute-budget", type=int, default=DEFAULT_COMPUTE_BUDGET)
    start.add_argument("--partial", type=Path, required=True)
    start.add_argument("--checkpoint-dir", type=Path, required=True)
    start.add_argument("--torch-threads", type=int, default=2)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--repo-root", type=Path, default=Path("."))
    resume.add_argument("--source-sha", required=True)
    resume.add_argument("--partial", type=Path, required=True)
    resume.add_argument("--checkpoint-dir", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--torch-threads", type=int, default=2)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--inputs", nargs=4, type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--expected-source-sha", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        payload = start_model(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            model_index=args.model_index,
            compute_budget=args.compute_budget,
            partial_path=args.partial,
            checkpoint_dir=args.checkpoint_dir,
            torch_threads=args.torch_threads,
        )
        print(
            json.dumps(
                {
                    "parameters": payload["parameters"],
                    "optimized_tokens_at_resume": payload["resume_tokens"],
                    "target_optimized_tokens": payload["plan"]["optimized_tokens"],
                    "compute_proxy": payload["plan"]["compute_proxy"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "resume":
        payload = resume_model(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            partial_path=args.partial,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        validate_report(payload, expected_source_sha=args.source_sha)
        print(
            json.dumps(
                {
                    "parameters": payload["parameters"],
                    "optimized_tokens": payload["accounting"]["optimized_tokens"],
                    "compute_proxy": payload["budget"]["compute_proxy"],
                    "validation_loss": payload["generalization"]["validation_loss"],
                    "validation_bpb": payload["generalization"]["validation_bpb"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "aggregate":
        payload = aggregate_models(
            inputs=args.inputs,
            output_path=args.output,
            expected_source_sha=args.expected_source_sha,
        )
        validate_report(payload, expected_source_sha=args.expected_source_sha)
        print(
            json.dumps(
                {
                    "matrix": payload["matrix"],
                    "rankings": payload["rankings"],
                    "recommendation": payload["recommendation"],
                    "report_sha256": payload["report_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("report must be a JSON object")
    validate_report(payload, expected_source_sha=args.expected_source_sha)
    print(f"{payload['schema']}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
