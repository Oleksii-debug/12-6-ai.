"""Strict fixed-token and fixed-compute scaling experiments for RESEARCH41.

This module reuses the exact RESEARCH41 model family, tokenizer, corpus split,
packing schedule, initialization, and optimizer recipe. It adds budget control,
exact token accounting, checkpoint/resume evidence, and efficiency rankings.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from .checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    COMPUTE_PROXY,
    PACKING_ID,
    TOKENIZER_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
    controlled_specs,
)
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.scaling-budget-experiment.v1"
AUTHORITY = "LOCAL_FREE_FIXED_BUDGET_SCALING_NOT_PROMOTION_OR_PAID_COMPUTE"
FIXED_TOKEN_BUDGET = 65_772
FIXED_COMPUTE_BUDGET = 105_000_000_000
PARAMETER_COUNTS = (95_568, 267_912, 467_808, 1_037_696)


def _token_quantum(*, batch_size: int, sequence_length: int) -> int:
    if batch_size <= 0 or sequence_length < 2 or sequence_length > 256:
        raise ValueError("invalid batch_size or sequence_length")
    return batch_size * (sequence_length - 1)


def _budget_plan(
    *,
    mode: str,
    budget: int,
    parameters: int,
    token_quantum: int,
) -> dict[str, int]:
    if budget <= 0 or parameters <= 0 or token_quantum <= 0:
        raise ValueError("budget, parameters, and token_quantum must be positive")
    compute_per_update = 6 * parameters * token_quantum
    if mode == "fixed_tokens":
        if budget % token_quantum:
            raise ValueError(
                "fixed token budget is not exactly representable by the frozen "
                f"optimizer-update token quantum: budget={budget}, quantum={token_quantum}"
            )
        optimizer_steps = budget // token_quantum
        optimized_tokens = budget
        compute_budget = 6 * parameters * budget
        compute_remainder = 0
    elif mode == "fixed_compute":
        optimizer_steps = budget // compute_per_update
        if optimizer_steps <= 0:
            raise ValueError("compute budget cannot fund one complete optimizer update")
        optimized_tokens = optimizer_steps * token_quantum
        compute_budget = budget
        compute_remainder = budget - (6 * parameters * optimized_tokens)
        if not 0 <= compute_remainder < compute_per_update:
            raise RuntimeError("fixed-compute flooring drift")
    else:
        raise ValueError(f"unsupported budget mode: {mode!r}")
    if optimizer_steps < 2:
        raise ValueError("budget must fund at least two updates so exact resume is exercised")
    actual_compute = 6 * parameters * optimized_tokens
    return {
        "requested_budget": budget,
        "optimizer_steps": optimizer_steps,
        "optimized_tokens": optimized_tokens,
        "compute_per_update": compute_per_update,
        "compute_proxy": actual_compute,
        "compute_budget": compute_budget,
        "compute_remainder": compute_remainder,
    }


def _rss_hwm_bytes() -> tuple[int | None, str]:
    try:
        import resource
    except ImportError:
        return None, "UNAVAILABLE_ON_PLATFORM"
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value, "getrusage.ru_maxrss_bytes"
    return value * 1024, "getrusage.ru_maxrss_kib_converted_to_bytes"


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_tensor_bytes(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _assert_exact_state(left: Any, right: Any, *, path: str = "state") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"resume state type drift at {path}")
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"resume tensor drift at {path}")
        return
    if is_dataclass(left) or is_dataclass(right):
        if not (is_dataclass(left) and is_dataclass(right)) or type(left) is not type(right):
            raise RuntimeError(f"resume dataclass drift at {path}")
        for field in fields(left):
            _assert_exact_state(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise RuntimeError(f"resume mapping type drift at {path}")
        if set(left) != set(right):
            raise RuntimeError(f"resume mapping-key drift at {path}")
        for key in left:
            _assert_exact_state(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"resume sequence drift at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact_state(left_item, right_item, path=f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"resume scalar drift at {path}: {left!r} != {right!r}")


def _parameter_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _parameter_delta_stats(
    initial: Mapping[str, torch.Tensor], model: TwelveSixDecoder
) -> dict[str, float]:
    delta_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    changed = 0
    elements = 0
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu()
        baseline = initial[name]
        delta = current - baseline
        delta_sq += float(torch.sum(delta.double() * delta.double()).item())
        initial_sq += float(torch.sum(baseline.double() * baseline.double()).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        elements += delta.numel()
    delta_l2 = math.sqrt(delta_sq)
    initial_l2 = math.sqrt(initial_sq)
    return {
        "delta_l2": delta_l2,
        "initial_weight_l2": initial_l2,
        "delta_to_initial_l2_ratio": delta_l2 / initial_l2 if initial_l2 else math.inf,
        "max_abs_delta": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": elements,
    }


def _update_ratio(before: Sequence[torch.Tensor], model: TwelveSixDecoder) -> float:
    delta_sq = 0.0
    weight_sq = 0.0
    for baseline, parameter in zip(before, model.parameters(), strict=True):
        current = parameter.detach()
        delta = current - baseline
        delta_sq += float(torch.sum(delta.double() * delta.double()).item())
        weight_sq += float(torch.sum(current.double() * current.double()).item())
    denominator = math.sqrt(weight_sq)
    return math.sqrt(delta_sq) / denominator if denominator else math.inf


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("statistics require at least one value")
    return {
        "min": min(values),
        "max": max(values),
        "mean": fmean(values),
        "last": values[-1],
    }


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: Any,
    trainer: Any,
    dataset_manifest_sha256: str,
    run_manifest_hash: str,
    seed: int,
) -> CheckpointIdentity:
    config = trainer.config
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=dataset_manifest_sha256,
        run_manifest_hash=run_manifest_hash,
        training_config=asdict(config),
        seed=seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": list(config.betas),
            "eps": config.eps,
        },
        scheduler=None,
    )


def _load_data(repo_root: Path) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    return {
        "tokenizer": tokenizer,
        "train_records": train_records,
        "validation_records": validation_records,
        "train_stream": _byte_stream(train_records, tokenizer),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "manifest_path": manifest_path,
        "train_path": train_path,
        "validation_path": validation_path,
        "overlap": overlap,
    }


def _run_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    mode: str,
    budget: int,
    candidate_index: int,
    checkpoint_root: Path,
    batch_size: int = 4,
    sequence_length: int = 64,
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("candidate exact-checkout mismatch")
    specs = controlled_specs()
    if not 0 <= candidate_index < len(specs):
        raise ValueError("candidate_index is outside the controlled family")
    spec = specs[candidate_index]
    parameters = spec.parameter_count()
    quantum = _token_quantum(batch_size=batch_size, sequence_length=sequence_length)
    plan = _budget_plan(
        mode=mode,
        budget=budget,
        parameters=parameters,
        token_quantum=quantum,
    )
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    data = _load_data(repo_root)
    tokenizer = data["tokenizer"]
    validation_records = data["validation_records"]
    train_stream = data["train_stream"]
    dataset_manifest_sha256 = _file_sha256(data["manifest_path"])
    config = _trainer_config(max_steps=plan["optimizer_steps"], seed=seed)
    init_spec = InitSpec()

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")
    initial_parameters = _parameter_snapshot(model)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    candidate_started = time.perf_counter()
    eval_started = time.perf_counter()
    tokens_before_eval = trainer.tokens_seen
    initial_validation_loss, validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    initial_eval_seconds = time.perf_counter() - eval_started
    if trainer.tokens_seen != tokens_before_eval:
        raise RuntimeError("evaluation mutated optimized-token accounting")

    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "mode": mode,
        "requested_budget": budget,
        "parameters": parameters,
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "packing_id": PACKING_ID,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "token_quantum": quantum,
        "optimizer_steps": plan["optimizer_steps"],
        "seed": seed,
    }
    run_manifest_hash = hash_json(run_manifest)
    resume_step = max(1, plan["optimizer_steps"] // 2)
    checkpoint_path = checkpoint_root / f"{mode}-{parameters}-step-{resume_step}"
    grad_norms: list[float] = []
    update_ratios: list[float] = []
    update_losses: list[float] = []
    train_losses: list[float] = []
    optimization_wall_seconds = 0.0
    checkpoint_save_seconds = 0.0
    checkpoint_load_seconds = 0.0
    checkpoint_bytes = 0
    resume_exact = False

    for update_index in range(plan["optimizer_steps"]):
        batch = _make_batch(
            train_stream,
            step=update_index,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        before_update = [parameter.detach().clone() for parameter in model.parameters()]
        train_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        optimization_wall_seconds += time.perf_counter() - train_started
        expected_step = update_index + 1
        expected_tokens = expected_step * quantum
        if metrics.tokens != quantum:
            raise RuntimeError(
                f"causal-token quantum drift: expected {quantum}, observed {metrics.tokens}"
            )
        if not metrics.optimizer_stepped or trainer.optimizer_step != expected_step:
            raise RuntimeError("optimizer-step accounting drift")
        if trainer.tokens_seen != expected_tokens:
            raise RuntimeError(
                f"optimized-token accounting drift: expected {expected_tokens}, "
                f"observed {trainer.tokens_seen}"
            )
        if metrics.grad_norm is None or metrics.update_loss is None:
            raise RuntimeError("missing committed gradient/update statistics")
        grad_norms.append(float(metrics.grad_norm))
        update_losses.append(float(metrics.update_loss))
        train_losses.append(float(metrics.loss))
        update_ratios.append(_update_ratio(before_update, model))

        if trainer.optimizer_step == resume_step:
            before_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            before_trainer_state = trainer.state_dict()
            identity = _checkpoint_identity(
                source_sha=source_sha,
                spec=spec,
                trainer=trainer,
                dataset_manifest_sha256=dataset_manifest_sha256,
                run_manifest_hash=run_manifest_hash,
                seed=seed,
            )
            save_started = time.perf_counter()
            save_trainer_checkpoint(
                checkpoint_path,
                model=model,
                trainer=trainer,
                identity=identity,
            )
            checkpoint_save_seconds = time.perf_counter() - save_started
            checkpoint_bytes = _directory_bytes(checkpoint_path)

            del trainer
            del model
            gc.collect()
            random.seed(seed)
            torch.manual_seed(seed)
            model = TwelveSixDecoder(spec, init_spec)
            fresh_trainer = Trainer(model, config, device="cpu")
            load_started = time.perf_counter()
            load_trainer_checkpoint(
                checkpoint_path,
                model=model,
                trainer=fresh_trainer,
                restore_rng=True,
                expected_git_sha=source_sha,
                expected_model_spec_hash=hash_json(spec.to_dict()),
                expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
                expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
                expected_dataset_manifest_hash=dataset_manifest_sha256,
                expected_run_manifest_hash=run_manifest_hash,
                expected_seed=seed,
            )
            checkpoint_load_seconds = time.perf_counter() - load_started
            for name, tensor in model.state_dict().items():
                if not torch.equal(tensor.detach().cpu(), before_model_state[name]):
                    raise RuntimeError(f"checkpoint model resume drift at {name}")
            _assert_exact_state(before_trainer_state, fresh_trainer.state_dict())
            if fresh_trainer.tokens_seen != resume_step * quantum:
                raise RuntimeError("checkpoint optimized-token counter drift")
            trainer = fresh_trainer
            resume_exact = True

    if trainer.tokens_seen != plan["optimized_tokens"]:
        raise RuntimeError(
            f"final optimized-token drift: expected {plan['optimized_tokens']}, "
            f"observed {trainer.tokens_seen}"
        )
    if trainer.optimizer_step != plan["optimizer_steps"]:
        raise RuntimeError("final optimizer-step drift")
    if 6 * parameters * trainer.tokens_seen != plan["compute_proxy"]:
        raise RuntimeError("final compute-proxy drift")
    tokens_before_final_eval = trainer.tokens_seen
    eval_started = time.perf_counter()
    validation_loss, checked_validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    final_eval_seconds = time.perf_counter() - eval_started
    if checked_validation_tokens != validation_tokens:
        raise RuntimeError("validation target count drift")
    if trainer.tokens_seen != tokens_before_final_eval:
        raise RuntimeError("final evaluation counted as optimized tokens")

    movement = _parameter_delta_stats(initial_parameters, model)
    optimizer_state_tensor_bytes = _tensor_bytes(trainer.state_dict().optimizer)
    clip_threshold = float(config.gradient_clip_norm or math.inf)
    clip_events = sum(value > clip_threshold for value in grad_norms)
    peak_rss_bytes, peak_rss_method = _rss_hwm_bytes()
    candidate_total_wall_seconds = time.perf_counter() - candidate_started
    validation_improvement = initial_validation_loss - validation_loss
    bpb = validation_loss / math.log(2.0)
    initial_bpb = initial_validation_loss / math.log(2.0)

    return {
        "schema": "12-6.scaling-budget-candidate.v1",
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "mode": mode,
        "parameters": parameters,
        "model_identity_sha256": spec.identity_sha256(),
        "model_spec": spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "budget": plan,
        "accounting": {
            "valid_causal_tokens_per_update": quantum,
            "optimized_tokens": trainer.tokens_seen,
            "evaluation_tokens_per_pass": validation_tokens,
            "evaluation_passes": 2,
            "evaluation_tokens_counted_as_optimized": 0,
            "optimizer_steps": trainer.optimizer_step,
            "token_drift_detected": False,
        },
        "generalization": {
            "initial_validation_loss": initial_validation_loss,
            "validation_loss": validation_loss,
            "initial_bpb": initial_bpb,
            "bpb": bpb,
            "validation_improvement": validation_improvement,
            "validation_tokens": validation_tokens,
        },
        "optimization": {
            "final_train_loss": train_losses[-1],
            "train_loss_mean": fmean(train_losses),
            "update_loss_mean": fmean(update_losses),
            "gradient_norm": _stats(grad_norms),
            "update_to_weight_l2_ratio": _stats(update_ratios),
            "clip_events": clip_events,
            "clip_fraction": clip_events / len(grad_norms),
            "parameter_movement": movement,
        },
        "timing": {
            "optimization_wall_seconds": optimization_wall_seconds,
            "candidate_total_wall_seconds": candidate_total_wall_seconds,
            "initial_validation_seconds": initial_eval_seconds,
            "final_validation_seconds": final_eval_seconds,
            "optimized_tokens_per_optimization_second": (
                trainer.tokens_seen / optimization_wall_seconds
            ),
            "mean_update_seconds": optimization_wall_seconds / trainer.optimizer_step,
            "measurement_note": (
                "optimization_wall_seconds times only Trainer.train_microbatch; "
                "candidate_total_wall_seconds also includes evaluation, update-ratio "
                "measurement, checkpoint save/load, and resume verification"
            ),
        },
        "memory": {
            "peak_process_rss_bytes": peak_rss_bytes,
            "peak_process_rss_method": peak_rss_method,
            "model_parameter_tensor_bytes": parameter_bytes,
            "optimizer_state_tensor_bytes": optimizer_state_tensor_bytes,
            "peak_rss_includes_resume_verification_overhead": True,
        },
        "checkpoint_resume": {
            "resume_exercised": True,
            "resume_exact": resume_exact,
            "resume_optimizer_step": resume_step,
            "resume_optimized_tokens": resume_step * quantum,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "checkpoint_path": str(checkpoint_path),
        },
        "efficiency": {
            "validation_improvement_per_parameter": validation_improvement / parameters,
            "validation_improvement_per_compute_proxy": (
                validation_improvement / plan["compute_proxy"]
            ),
            "validation_improvement_per_optimization_wall_second": (
                validation_improvement / optimization_wall_seconds
            ),
        },
        "truth_boundary": {
            "held_out_generalization_only_for_ranking": True,
            "cpu_wall_time_is_universal_compute": False,
            "paid_compute": False,
            "stage_promotion": False,
            "broad_corpus_claim": False,
        },
    }


def _candidate_command(
    *,
    repo_root: Path,
    source_sha: str,
    mode: str,
    budget: int,
    candidate_index: int,
    output: Path,
    checkpoint_root: Path,
    batch_size: int,
    sequence_length: int,
    seed: int,
    torch_threads: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "twelve_six.scaling_budget",
        "candidate",
        "--repo-root",
        str(repo_root),
        "--source-sha",
        source_sha,
        "--mode",
        mode,
        "--budget",
        str(budget),
        "--candidate-index",
        str(candidate_index),
        "--output",
        str(output),
        "--checkpoint-root",
        str(checkpoint_root),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--seed",
        str(seed),
        "--torch-threads",
        str(torch_threads),
    ]


def _rank(
    rows: Sequence[dict[str, Any]], key: str, *, lower_is_better: bool
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row[key]),
        reverse=not lower_is_better,
    )
    return [
        {
            "rank": index,
            "parameters": int(row["parameters"]),
            "score": float(row[key]),
        }
        for index, row in enumerate(ranked, 1)
    ]


def _regime_summary(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "parameters": candidate["parameters"],
                "validation_loss": candidate["generalization"]["validation_loss"],
                "bpb": candidate["generalization"]["bpb"],
                "validation_improvement": candidate["generalization"]["validation_improvement"],
                "validation_improvement_per_parameter": candidate["efficiency"][
                    "validation_improvement_per_parameter"
                ],
                "validation_improvement_per_compute_proxy": candidate["efficiency"][
                    "validation_improvement_per_compute_proxy"
                ],
                "validation_improvement_per_optimization_wall_second": candidate[
                    "efficiency"
                ]["validation_improvement_per_optimization_wall_second"],
                "optimized_tokens": candidate["accounting"]["optimized_tokens"],
                "compute_proxy": candidate["budget"]["compute_proxy"],
                "optimization_wall_seconds": candidate["timing"]["optimization_wall_seconds"],
                "candidate_total_wall_seconds": candidate["timing"][
                    "candidate_total_wall_seconds"
                ],
                "peak_process_rss_bytes": candidate["memory"]["peak_process_rss_bytes"],
                "final_train_loss": candidate["optimization"]["final_train_loss"],
                "checkpoint_bytes": candidate["checkpoint_resume"]["checkpoint_bytes"],
                "checkpoint_save_seconds": candidate["checkpoint_resume"][
                    "checkpoint_save_seconds"
                ],
                "checkpoint_load_seconds": candidate["checkpoint_resume"][
                    "checkpoint_load_seconds"
                ],
            }
        )
    return {
        "matrix": rows,
        "rankings": {
            "best_validation": _rank(rows, "validation_loss", lower_is_better=True),
            "validation_improvement_per_parameter": _rank(
                rows, "validation_improvement_per_parameter", lower_is_better=False
            ),
            "validation_improvement_per_compute": _rank(
                rows, "validation_improvement_per_compute_proxy", lower_is_better=False
            ),
            "validation_improvement_per_wall_second": _rank(
                rows,
                "validation_improvement_per_optimization_wall_second",
                lower_is_better=False,
            ),
        },
    }


def run_matrix(
    *,
    repo_root: Path,
    source_sha: str,
    output_path: Path,
    evidence_root: Path,
    fixed_token_budget: int = FIXED_TOKEN_BUDGET,
    fixed_compute_budget: int = FIXED_COMPUTE_BUDGET,
    batch_size: int = 4,
    sequence_length: int = 64,
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("matrix exact-checkout mismatch")
    quantum = _token_quantum(batch_size=batch_size, sequence_length=sequence_length)
    specs = controlled_specs()
    if tuple(spec.parameter_count() for spec in specs) != PARAMETER_COUNTS:
        raise RuntimeError("controlled parameter family drift")
    for spec in specs:
        _budget_plan(
            mode="fixed_tokens",
            budget=fixed_token_budget,
            parameters=spec.parameter_count(),
            token_quantum=quantum,
        )
        _budget_plan(
            mode="fixed_compute",
            budget=fixed_compute_budget,
            parameters=spec.parameter_count(),
            token_quantum=quantum,
        )

    evidence_root.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict[str, Any]]] = {"fixed_tokens": [], "fixed_compute": []}
    for mode, budget in (
        ("fixed_tokens", fixed_token_budget),
        ("fixed_compute", fixed_compute_budget),
    ):
        for index, spec in enumerate(specs):
            candidate_output = evidence_root / f"{mode}-{spec.parameter_count()}.json"
            command = _candidate_command(
                repo_root=repo_root,
                source_sha=source_sha,
                mode=mode,
                budget=budget,
                candidate_index=index,
                output=candidate_output,
                checkpoint_root=evidence_root / "checkpoints",
                batch_size=batch_size,
                sequence_length=sequence_length,
                seed=seed,
                torch_threads=torch_threads,
            )
            subprocess.run(command, cwd=repo_root, check=True)
            candidate = json.loads(candidate_output.read_text(encoding="utf-8"))
            candidates[mode].append(candidate)

    data = _load_data(repo_root)
    manifest = data["manifest"]
    base_recipe = asdict(_trainer_config(max_steps=2, seed=seed))
    base_recipe.pop("max_steps")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "incumbent": "RESEARCH41 PR #162",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads_per_candidate": torch_threads,
            "paid_compute": False,
            "candidate_process_isolation": True,
        },
        "controls": {
            "canonical_base": "random_init",
            "parameter_counts": list(PARAMETER_COUNTS),
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "vocab_size": 256,
            "model_max_seq_len": 256,
            "training_sequence_length": sequence_length,
            "batch_size": batch_size,
            "valid_causal_tokens_per_optimizer_update": quantum,
            "packing_id": PACKING_ID,
            "seed": seed,
            "optimizer_recipe_excluding_budget_max_steps": base_recipe,
            "scheduler_budget_semantics": (
                "constant scheduler with warmup_steps=0; max_steps is a stopping bound "
                "only and does not change learning rate"
            ),
            "compute_proxy_definition": COMPUTE_PROXY,
            "fixed_token_budget": fixed_token_budget,
            "fixed_compute_budget": fixed_compute_budget,
            "fixed_compute_stop_rule": (
                "floor budget to the largest complete optimizer update; report remainder; "
                "never exceed the requested 6*N*T envelope"
            ),
        },
        "data": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
            "manifest_sha256": _file_sha256(data["manifest_path"]),
            "train_jsonl_sha256": _file_sha256(data["train_path"]),
            "validation_jsonl_sha256": _file_sha256(data["validation_path"]),
            "train_record_count": len(data["train_records"]),
            "validation_record_count": len(data["validation_records"]),
            "train_validation_record_overlap": data["overlap"],
            "repeated_fixture": True,
            "scope_warning": (
                "The project-authored S0 fixture is tiny and cyclically recycled. "
                "These are controlled local generalization measurements, not broad-corpus scaling."
            ),
        },
        "fixed_tokens": {
            "budget": fixed_token_budget,
            "candidates": candidates["fixed_tokens"],
            **_regime_summary(candidates["fixed_tokens"]),
        },
        "fixed_compute": {
            "budget": fixed_compute_budget,
            "candidates": candidates["fixed_compute"],
            **_regime_summary(candidates["fixed_compute"]),
        },
        "ranking_basis": {
            "generalization_metric": "held-out validation loss; train loss is diagnostic only",
            "bpb_definition": "validation cross-entropy nats / ln(2) for byte tokenizer",
            "wall_efficiency_denominator": "optimization_wall_seconds only",
            "cpu_wall_time_is_hardware_specific": True,
        },
        "truth_boundary": {
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
            "cpu_wall_time_is_universal_hardware_compute": False,
            "gpu_economics_extrapolation": False,
            "broad_corpus_quality_claim": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("fixed-budget report schema/authority mismatch")
    source = report.get("source", {})
    if source.get("repository") != "Oleksii-debug/12-6-ai.":
        raise ValueError("repository identity drift")
    if expected_source_sha is not None and source.get("git_sha") != expected_source_sha:
        raise ValueError("source SHA mismatch")
    controls = report.get("controls", {})
    if controls.get("parameter_counts") != list(PARAMETER_COUNTS):
        raise ValueError("controlled parameter family drift")
    if controls.get("tokenizer_id") != TOKENIZER_ID:
        raise ValueError("tokenizer identity drift")
    if controls.get("tokenizer_config_sha256") != BYTE_TOKENIZER_HASH:
        raise ValueError("tokenizer config hash drift")
    if controls.get("tokenizer_vocab_sha256") != BYTE_VOCAB_HASH:
        raise ValueError("tokenizer vocabulary hash drift")
    if controls.get("model_max_seq_len") != 256:
        raise ValueError("controlled context drift")
    if report.get("data", {}).get("train_validation_record_overlap") != []:
        raise ValueError("held-out split isolation failed")
    quantum = int(controls["valid_causal_tokens_per_optimizer_update"])
    token_budget = int(controls["fixed_token_budget"])
    compute_budget = int(controls["fixed_compute_budget"])
    if token_budget % quantum:
        raise ValueError("fixed-token budget is not exact")
    for mode in ("fixed_tokens", "fixed_compute"):
        candidates = report.get(mode, {}).get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"{mode} must contain four candidates")
        if [int(candidate["parameters"]) for candidate in candidates] != list(PARAMETER_COUNTS):
            raise ValueError(f"{mode} parameter order/family drift")
        for candidate in candidates:
            parameters = int(candidate["parameters"])
            accounting = candidate["accounting"]
            budget = candidate["budget"]
            optimized_tokens = int(accounting["optimized_tokens"])
            if optimized_tokens != int(budget["optimized_tokens"]):
                raise ValueError("reported optimized token drift")
            if optimized_tokens % quantum:
                raise ValueError("optimized tokens are not a complete-update quantum")
            expected_compute = 6 * parameters * optimized_tokens
            if int(budget["compute_proxy"]) != expected_compute:
                raise ValueError("compute proxy drift")
            if accounting.get("evaluation_tokens_counted_as_optimized") != 0:
                raise ValueError("evaluation tokens contaminated optimization budget")
            if accounting.get("token_drift_detected") is not False:
                raise ValueError("candidate reported token drift")
            resume = candidate["checkpoint_resume"]
            if resume.get("resume_exercised") is not True or resume.get("resume_exact") is not True:
                raise ValueError("exact resume evidence missing")
            generalization = candidate["generalization"]
            if not math.isfinite(float(generalization["validation_loss"])):
                raise ValueError("non-finite held-out validation loss")
            if not math.isfinite(float(generalization["bpb"])):
                raise ValueError("non-finite BPB")
            if mode == "fixed_tokens" and optimized_tokens != token_budget:
                raise ValueError("fixed-token candidates did not receive identical exact tokens")
            if mode == "fixed_compute":
                if expected_compute > compute_budget:
                    raise ValueError("fixed-compute candidate exceeded envelope")
                step_compute = int(budget["compute_per_update"])
                remainder = compute_budget - expected_compute
                if remainder != int(budget["compute_remainder"]):
                    raise ValueError("compute remainder drift")
                if not 0 <= remainder < step_compute:
                    raise ValueError("fixed-compute stop boundary is not maximal")
    truth = report.get("truth_boundary", {})
    required_false = (
        "stage_freeze",
        "promotion_authority",
        "paid_compute_authority",
        "cpu_wall_time_is_universal_hardware_compute",
        "gpu_economics_extrapolation",
        "broad_corpus_quality_claim",
    )
    if any(truth.get(key) is not False for key in required_false):
        raise ValueError("truth boundary weakened")
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("fixed-budget report self-hash mismatch")


def _write_candidate(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report["report_sha256"] = _canonical_hash(report)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--repo-root", type=Path, default=Path("."))
    candidate.add_argument("--source-sha", required=True)
    candidate.add_argument("--mode", choices=("fixed_tokens", "fixed_compute"), required=True)
    candidate.add_argument("--budget", type=int, required=True)
    candidate.add_argument("--candidate-index", type=int, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--checkpoint-root", type=Path, required=True)
    candidate.add_argument("--batch-size", type=int, default=4)
    candidate.add_argument("--sequence-length", type=int, default=64)
    candidate.add_argument("--seed", type=int, default=1337)
    candidate.add_argument("--torch-threads", type=int, default=2)

    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--fixed-token-budget", type=int, default=FIXED_TOKEN_BUDGET)
    run.add_argument("--fixed-compute-budget", type=int, default=FIXED_COMPUTE_BUDGET)
    run.add_argument("--torch-threads", type=int, default=2)

    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "candidate":
        report = _run_candidate(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            mode=args.mode,
            budget=args.budget,
            candidate_index=args.candidate_index,
            checkpoint_root=args.checkpoint_root.resolve(),
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=args.seed,
            torch_threads=args.torch_threads,
        )
        _write_candidate(args.output, report)
        print(
            json.dumps(
                {
                    "mode": report["mode"],
                    "parameters": report["parameters"],
                    "optimized_tokens": report["accounting"]["optimized_tokens"],
                    "compute_proxy": report["budget"]["compute_proxy"],
                    "validation_loss": report["generalization"]["validation_loss"],
                    "bpb": report["generalization"]["bpb"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        report = run_matrix(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
            evidence_root=args.evidence_root.resolve(),
            fixed_token_budget=args.fixed_token_budget,
            fixed_compute_budget=args.fixed_compute_budget,
            torch_threads=args.torch_threads,
        )
        validate_report(report, expected_source_sha=args.source_sha)
        print(
            json.dumps(
                {
                    "fixed_tokens": report["fixed_tokens"]["rankings"],
                    "fixed_compute": report["fixed_compute"]["rankings"],
                    "report_sha256": report["report_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("report must be a JSON object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
