#!/usr/bin/env python3
"""RESEARCH-123 / MILESTONE-100 convergence harness.

This file is deliberately an orchestration layer. It reuses the incumbent 12-6
ModelSpec/TwelveSixDecoder, DATA-21/22 bounded rights-aware intake, ByteTokenizer,
packer, D02 Trainer, and D05 checkpoint adapter. It does not implement competing
model, tokenizer, data, trainer, checkpoint, or inference systems.

Truth boundary: the currently available real corpus is the three-object DATA-21/22
bounded sample. It is real and training-approved, but explicitly not a representative
broad pretraining corpus. Therefore this run can produce useful local recycling/T/N
evidence and a genuinely learned Base checkpoint, but it MUST NOT mark the exact
MILESTONE-100 representative-corpus requirement PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from twelve_six.training.batch_noise_probe import (
    _real_corpus_records,
    _tensor_batches_from_records,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import Trainer

SCHEMA_VERSION = "12-6.research123-real-tn-scaling.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
SEED = 1337
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
LOSS_TOKENS_PER_STEP = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
TARGET_TN_RATIOS = (1.0 / 32.0, 1.0 / 8.0, 1.0 / 2.0, 2.0)
BOOTSTRAP_SAMPLES = 400
CURVE_BOOTSTRAP_SAMPLES = 300


class Research123Error(RuntimeError):
    """Fail-closed RESEARCH-123 contract failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Research123Error(message)


def _spec(
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    d_ff: int,
) -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=head_dim,
        d_ff=d_ff,
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


def model_family() -> tuple[tuple[str, ModelSpec], ...]:
    """Established RESEARCH41 fixed-vocab/fixed-context MHA family."""
    family = (
        ("100k", _spec(d_model=48, n_layers=3, n_heads=4, head_dim=12, d_ff=128)),
        ("250k", _spec(d_model=72, n_layers=4, n_heads=6, head_dim=12, d_ff=192)),
        ("500k", _spec(d_model=96, n_layers=4, n_heads=6, head_dim=16, d_ff=256)),
        ("1m", _spec(d_model=128, n_layers=5, n_heads=8, head_dim=16, d_ff=352)),
    )
    expected = (95_568, 267_912, 467_808, 1_037_696)
    actual = tuple(spec.parameter_count() for _, spec in family)
    _require(actual == expected, f"fixed-control parameter family drift: {actual}")
    return family


def _trainer_config(max_steps: int, *, seed: int = SEED) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return header + value.numpy().tobytes()


def _state_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(b"tensor:")
            digest.update(_tensor_digest(item))
            return
        if is_dataclass(item) and not isinstance(item, type):
            visit(asdict(item))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda current: repr(current)):
                digest.update(repr(key).encode("utf-8"))
                digest.update(b"=")
                visit(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
            return
        digest.update(type(item).__name__.encode())
        digest.update(b":")
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _model_digest(model: TwelveSixDecoder) -> str:
    return _state_digest(model.state_dict())


def _batch_trace_digest(batches: Sequence[Mapping[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for index, batch in enumerate(batches):
        digest.update(str(index).encode())
        digest.update(_tensor_digest(batch["input_ids"]))
        digest.update(_tensor_digest(batch["labels"]))
    return digest.hexdigest()


def _lock_bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "requirements/locks/linux-x86_64/runtime.lock.txt",
        "requirements/locks/linux-x86_64/dev.lock.txt",
    ):
        path = root / relative
        _require(path.is_file(), f"runtime lock missing: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _bootstrap_ci(
    observations: Sequence[tuple[float, int]],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    if len(observations) <= 1:
        value = observations[0][0] / math.log(2.0)
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [observations[rng.randrange(len(observations))] for _ in observations]
        total_tokens = sum(tokens for _, tokens in draw)
        weighted = sum(loss * tokens for loss, tokens in draw)
        estimates.append((weighted / total_tokens) / math.log(2.0))
    estimates.sort()
    lo = estimates[max(0, int(0.025 * len(estimates)) - 1)]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return lo, hi


@torch.no_grad()
def _evaluate_nonmutating(
    model: TwelveSixDecoder,
    trainer: Trainer,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    before_model = _model_digest(model)
    before_trainer = _state_digest(trainer.state_dict())
    before_mode = bool(model.training)
    before_tokens = trainer.tokens_seen
    before_step = trainer.optimizer_step

    observations: list[tuple[float, int]] = []
    weighted_loss = 0.0
    token_count = 0
    correct = 0
    model.eval()
    for batch in batches:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = model(input_ids).logits
        loss = causal_lm_loss(logits, labels)
        _require(torch.isfinite(loss).item(), "evaluation produced non-finite loss")
        targets = labels[:, 1:]
        predictions = logits[:, :-1, :].argmax(dim=-1)
        valid = targets.ne(-100)
        tokens = int(valid.sum().item())
        _require(tokens > 0, "evaluation batch has zero scoreable tokens")
        batch_loss = float(loss.item())
        observations.append((batch_loss, tokens))
        weighted_loss += batch_loss * tokens
        token_count += tokens
        correct += int((predictions.eq(targets) & valid).sum().item())

    model.train(before_mode)
    after_model = _model_digest(model)
    after_trainer = _state_digest(trainer.state_dict())
    _require(before_model == after_model, "evaluation mutated model state")
    _require(before_trainer == after_trainer, "evaluation mutated Trainer/optimizer state")
    _require(before_tokens == trainer.tokens_seen, "evaluation mutated optimized-token ledger")
    _require(before_step == trainer.optimizer_step, "evaluation mutated optimizer-step ledger")
    _require(bool(model.training) == before_mode, "evaluation failed to restore model mode")

    loss_nats = weighted_loss / token_count
    bpb = loss_nats / math.log(2.0)
    ci_low, ci_high = _bootstrap_ci(observations, seed=bootstrap_seed)
    return {
        "loss_nats": loss_nats,
        "bpb": bpb,
        "bpb_bootstrap_95ci": [ci_low, ci_high],
        "scoreable_tokens": token_count,
        "next_token_accuracy": correct / token_count,
        "evaluation_optimized_tokens": 0,
        "non_mutation_proof": {
            "model_sha256_before": before_model,
            "model_sha256_after": after_model,
            "trainer_sha256_before": before_trainer,
            "trainer_sha256_after": after_trainer,
            "tokens_seen_before": before_tokens,
            "tokens_seen_after": trainer.tokens_seen,
            "optimizer_step_before": before_step,
            "optimizer_step_after": trainer.optimizer_step,
            "pass": True,
        },
    }


@torch.no_grad()
def _generate_snapshot(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt)
    _require(bool(ids), "generation prompt encoded empty")
    input_ids = torch.tensor([ids], dtype=torch.long)
    before = _model_digest(model)
    before_mode = bool(model.training)
    model.eval()
    generated = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    model.train(before_mode)
    after = _model_digest(model)
    _require(before == after, "first-party generation mutated model weights")
    all_ids = generated[0].tolist()
    continuation = all_ids[len(ids) :]
    return {
        "prompt": prompt,
        "prompt_token_ids": ids,
        "continuation_token_ids": continuation,
        "continuation_text_utf8_replace": tokenizer.decode(continuation, errors="replace"),
        "first_party_api": "TwelveSixDecoder.generate",
        "greedy": True,
        "model_non_mutation": before == after,
    }


def _ratio_budgets(parameter_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = 0
    for requested_ratio in TARGET_TN_RATIOS:
        requested_tokens = parameter_count * requested_ratio
        steps = max(1, int(round(requested_tokens / LOSS_TOKENS_PER_STEP)))
        steps = max(steps, previous + 1)
        tokens = steps * LOSS_TOKENS_PER_STEP
        result.append(
            {
                "requested_t_per_n": requested_ratio,
                "requested_optimized_tokens": requested_tokens,
                "optimizer_steps": steps,
                "optimized_tokens": tokens,
                "realized_t_per_n": tokens / parameter_count,
                "absolute_ratio_error": abs(tokens / parameter_count - requested_ratio),
            }
        )
        previous = steps
    return result


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    tokenizer: ByteTokenizer,
    data: Mapping[str, Any],
    run_manifest_hash: str,
    config: TrainerConfig,
    trainer: Trainer,
    lock_hash: str,
) -> CheckpointIdentity:
    training_config = {
        "trainer": asdict(config),
        "init_spec_sha256": init_spec.identity_sha256(),
        "data": {
            "split_identity": data["dataset_identity_sha256"],
            "packing_version": "incumbent.iter_packed_examples.seq64.labels",
            "packing_sha256": data["training_trace_sha256"],
        },
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=str(data["intake_manifest_sha256"]),
        run_manifest_hash=run_manifest_hash,
        training_config=training_config,
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
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        scheduler={"name": config.scheduler, "warmup_steps": config.warmup_steps},
        environment_lock_hash=lock_hash,
    )


def _weight_delta(initial: Mapping[str, torch.Tensor], model: TwelveSixDecoder) -> dict[str, Any]:
    squared = 0.0
    maximum = 0.0
    changed = 0
    total = 0
    for name, parameter in model.state_dict().items():
        if not torch.is_tensor(parameter):
            continue
        delta = parameter.detach().cpu().float() - initial[name].float()
        squared += float((delta * delta).sum().item())
        maximum = max(maximum, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": maximum,
        "changed_elements": changed,
        "total_state_elements": total,
        "nonzero_change": changed > 0 and maximum > 0.0,
    }


def _run_candidate(
    *,
    root: Path,
    output_dir: Path,
    source_sha: str,
    label: str,
    spec: ModelSpec,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    validation_batches: Sequence[Mapping[str, torch.Tensor]],
    tokenizer: ByteTokenizer,
    data: dict[str, Any],
    train_prompt: str,
    validation_prompt: str,
    lock_hash: str,
) -> dict[str, Any]:
    parameter_count = spec.parameter_count()
    budgets = _ratio_budgets(parameter_count)
    max_steps = budgets[-1]["optimizer_steps"]
    config = _trainer_config(max_steps)
    init_spec = InitSpec()

    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init_spec)
    _require(sum(p.numel() for p in model.parameters() if p.requires_grad) == parameter_count, "actual parameter count drift")
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if torch.is_tensor(value)}
    initial_model_sha = _model_digest(model)
    trainer = Trainer(model, config, device="cpu")

    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "label": label,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "parameter_count": parameter_count,
        "tokenizer_version": BYTE_TOKENIZER_VERSION,
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
        "dataset_identity_sha256": data["dataset_identity_sha256"],
        "intake_manifest_sha256": data["intake_manifest_sha256"],
        "training_trace_sha256": data["training_trace_sha256"],
        "target_t_per_n_ratios": list(TARGET_TN_RATIOS),
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "loss_tokens_per_optimizer_step": LOSS_TOKENS_PER_STEP,
        "trainer_config": asdict(config),
        "seed": SEED,
        "precision": "fp32",
        "runtime_lock_bundle_sha256": lock_hash,
        "paid_compute": False,
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
    }
    run_manifest_hash = hash_json(run_identity)

    baseline_train = _evaluate_nonmutating(
        model,
        trainer,
        train_batches,
        bootstrap_seed=SEED + parameter_count + 1,
    )
    baseline_validation = _evaluate_nonmutating(
        model,
        trainer,
        validation_batches,
        bootstrap_seed=SEED + parameter_count + 2,
    )
    initial_generation = {
        "train_prefix": _generate_snapshot(model, tokenizer, train_prompt),
        "validation_prefix": _generate_snapshot(model, tokenizer, validation_prompt),
    }

    unique_train_tokens = int(baseline_train["scoreable_tokens"])
    points: list[dict[str, Any]] = []
    checkpoint_root = output_dir / "checkpoints" / label
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    step_wall_start = time.perf_counter()
    last_wall = step_wall_start
    batch_count = len(train_batches)
    _require(batch_count > 0, "training produced zero batches")

    for budget_index, budget in enumerate(budgets):
        target_steps = int(budget["optimizer_steps"])
        recent_update_losses: list[float] = []
        while trainer.optimizer_step < target_steps:
            batch = train_batches[trainer.optimizer_step % batch_count]
            metrics = trainer.train_microbatch(batch)
            _require(metrics.optimizer_stepped, "accumulation-1 candidate failed to optimizer-step")
            _require(metrics.tokens == LOSS_TOKENS_PER_STEP, "optimized-token step size drift")
            if metrics.update_loss is not None:
                recent_update_losses.append(float(metrics.update_loss))
                if len(recent_update_losses) > 64:
                    recent_update_losses.pop(0)

        _require(trainer.tokens_seen == target_steps * LOSS_TOKENS_PER_STEP, "optimized-token ledger drift")
        interval_wall = time.perf_counter() - last_wall
        cumulative_wall = time.perf_counter() - step_wall_start
        last_wall = time.perf_counter()

        train_eval = _evaluate_nonmutating(
            model,
            trainer,
            train_batches,
            bootstrap_seed=SEED + parameter_count + 100 + budget_index * 2,
        )
        validation_eval = _evaluate_nonmutating(
            model,
            trainer,
            validation_batches,
            bootstrap_seed=SEED + parameter_count + 101 + budget_index * 2,
        )
        generation = {
            "train_prefix": _generate_snapshot(model, tokenizer, train_prompt),
            "validation_prefix": _generate_snapshot(model, tokenizer, validation_prompt),
        }

        checkpoint_dir = checkpoint_root / f"tn-{budget_index:02d}-tokens-{trainer.tokens_seen}"
        identity = _checkpoint_identity(
            source_sha=source_sha,
            spec=spec,
            init_spec=init_spec,
            tokenizer=tokenizer,
            data=data,
            run_manifest_hash=run_manifest_hash,
            config=config,
            trainer=trainer,
            lock_hash=lock_hash,
        )
        save_trainer_checkpoint(
            checkpoint_dir,
            model=model,
            trainer=trainer,
            identity=identity,
            overwrite=False,
        )
        manifest_path = checkpoint_dir / "manifest.json"
        _require(manifest_path.is_file(), "checkpoint manifest missing after save")

        recent_bpb = None
        if recent_update_losses:
            recent_bpb = sum(recent_update_losses) / len(recent_update_losses) / math.log(2.0)
        points.append(
            {
                **budget,
                "training_eval": train_eval,
                "validation_eval": validation_eval,
                "generalization_gap_bpb": validation_eval["bpb"] - train_eval["bpb"],
                "memorization": {
                    "train_next_token_accuracy": train_eval["next_token_accuracy"],
                    "validation_next_token_accuracy": validation_eval["next_token_accuracy"],
                    "accuracy_gap": train_eval["next_token_accuracy"] - validation_eval["next_token_accuracy"],
                    "corpus_recycle_factor": trainer.tokens_seen / unique_train_tokens,
                    "train_minus_validation_bpb": train_eval["bpb"] - validation_eval["bpb"],
                },
                "recent_optimizer_update_bpb": recent_bpb,
                "wall": {
                    "interval_training_seconds": interval_wall,
                    "cumulative_training_seconds": cumulative_wall,
                    "cumulative_optimized_tokens_per_second": trainer.tokens_seen / cumulative_wall,
                },
                "compute_proxy_6NT": 6 * parameter_count * trainer.tokens_seen,
                "checkpoint": {
                    "relative_path": str(checkpoint_dir.relative_to(output_dir)),
                    "manifest_sha256": sha256_file(manifest_path),
                    "model_sha256": _model_digest(model),
                    "trainer_state_sha256": _state_digest(trainer.state_dict()),
                },
                "generation": generation,
            }
        )

    movement = _weight_delta(initial_state, model)
    _require(movement["nonzero_change"], f"{label} did not learn: zero parameter movement")
    _require(points[-1]["training_eval"]["bpb"] < baseline_train["bpb"], f"{label} train BPB failed to decrease")
    best_validation = min(point["validation_eval"]["bpb"] for point in points)
    _require(best_validation < baseline_validation["bpb"], f"{label} held-out BPB never improved")

    reversals: list[dict[str, Any]] = []
    previous_point: dict[str, Any] | None = None
    for point in points:
        if previous_point is not None:
            train_delta = point["training_eval"]["bpb"] - previous_point["training_eval"]["bpb"]
            validation_delta = point["validation_eval"]["bpb"] - previous_point["validation_eval"]["bpb"]
            gap_delta = point["generalization_gap_bpb"] - previous_point["generalization_gap_bpb"]
            if train_delta < 0.0 and validation_delta > 0.0:
                reversals.append(
                    {
                        "from_t_per_n": previous_point["realized_t_per_n"],
                        "to_t_per_n": point["realized_t_per_n"],
                        "train_bpb_delta": train_delta,
                        "validation_bpb_delta": validation_delta,
                        "generalization_gap_delta": gap_delta,
                        "signal": "TRAIN_IMPROVES_WHILE_HELDOUT_WORSENS",
                    }
                )
        previous_point = point

    return {
        "label": label,
        "parameter_count": parameter_count,
        "model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec": init_spec.to_dict(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "random_initialization": {
            "seed": SEED,
            "model_state_sha256": initial_model_sha,
            "foreign_pretrained_weights_loaded": False,
        },
        "run_manifest_sha256": run_manifest_hash,
        "training_trace_sha256": data["training_trace_sha256"],
        "baseline": {
            "optimized_tokens": 0,
            "training_eval": baseline_train,
            "validation_eval": baseline_validation,
            "generation": initial_generation,
        },
        "points": points,
        "weight_movement": movement,
        "overfit_reversals": reversals,
        "final_model_sha256": _model_digest(model),
        "final_trainer_state_sha256": _state_digest(trainer.state_dict()),
        "final_optimized_tokens": trainer.tokens_seen,
        "final_optimizer_steps": trainer.optimizer_step,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise Research123Error("percentile requires values")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _fit_curve(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[list[float]] = []
    observed: list[float] = []
    standard_errors: list[float] = []
    for candidate in candidates:
        n = float(candidate["parameter_count"])
        ln_n = math.log(n)
        for point in candidate["points"]:
            ratio = float(point["realized_t_per_n"])
            ln_r = math.log(ratio)
            rows.append([1.0, ln_n, ln_r, ln_r * ln_r, ln_n * ln_r])
            bpb = float(point["validation_eval"]["bpb"])
            observed.append(bpb)
            ci = point["validation_eval"]["bpb_bootstrap_95ci"]
            se = max((float(ci[1]) - float(ci[0])) / (2.0 * 1.96), 1e-6)
            standard_errors.append(se)

    x = torch.tensor(rows, dtype=torch.float64)
    y = torch.tensor(observed, dtype=torch.float64)
    coefficients = torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    fitted = x @ coefficients
    residuals = y - fitted
    residual_rmse = float(torch.sqrt(torch.mean(residuals * residuals)).item())

    rng = random.Random(SEED + 123_000)
    boot_coeffs: list[list[float]] = []
    for _ in range(CURVE_BOOTSTRAP_SAMPLES):
        sampled = torch.tensor(
            [rng.gauss(mean, se) for mean, se in zip(observed, standard_errors, strict=True)],
            dtype=torch.float64,
        )
        solution = torch.linalg.lstsq(x, sampled[:, None]).solution[:, 0]
        boot_coeffs.append([float(value.item()) for value in solution])

    names = ["intercept", "ln_N", "ln_T_per_N", "ln_T_per_N_squared", "ln_N_x_ln_T_per_N"]
    coefficient_report: dict[str, Any] = {}
    for index, name in enumerate(names):
        samples = [row[index] for row in boot_coeffs]
        coefficient_report[name] = {
            "estimate": float(coefficients[index].item()),
            "parametric_eval_bootstrap_95ci": [
                _percentile(samples, 0.025),
                _percentile(samples, 0.975),
            ],
        }
    return {
        "form": "validation_BPB = b0 + bN*ln(N) + bR*ln(T/N) + bR2*ln(T/N)^2 + bNR*ln(N)*ln(T/N)",
        "coefficients": coefficient_report,
        "observations": len(observed),
        "residual_rmse_bpb": residual_rmse,
        "uncertainty_scope": (
            "Point uncertainty is a deterministic block bootstrap over held-out packed batches. "
            "Coefficient intervals are a parametric propagation of those point errors only; "
            "they do not include training-seed, source-selection, or broad-corpus uncertainty."
        ),
        "universal_scaling_law_claim": False,
        "extrapolation_authority": "NONE_OUTSIDE_OBSERVED_95K_TO_1.04M_AND_TN_GRID",
    }


def _recommend(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    large_model_best_ratios: list[float] = []
    for candidate in candidates:
        points = list(candidate["points"])
        best_index = min(range(len(points)), key=lambda index: points[index]["validation_eval"]["bpb"])
        best = points[best_index]
        status = "INTERIOR_OR_EARLY_BEST"
        if best_index == len(points) - 1:
            status = "GRID_EDGE_STILL_BEST_NO_SATURATION_CLAIM"
        reversal_after_best = any(
            point["validation_eval"]["bpb"] > best["validation_eval"]["bpb"]
            and point["training_eval"]["bpb"] < best["training_eval"]["bpb"]
            for point in points[best_index + 1 :]
        )
        per_model[str(candidate["label"])] = {
            "parameter_count": candidate["parameter_count"],
            "recommended_observed_checkpoint_tokens": best["optimized_tokens"],
            "recommended_observed_t_per_n": best["realized_t_per_n"],
            "heldout_bpb": best["validation_eval"]["bpb"],
            "training_bpb": best["training_eval"]["bpb"],
            "generalization_gap_bpb": best["generalization_gap_bpb"],
            "checkpoint": best["checkpoint"],
            "status": status,
            "overfit_reversal_after_best": reversal_after_best,
            "real_bounded_corpus_only": True,
        }
        if str(candidate["label"]) in {"500k", "1m"}:
            large_model_best_ratios.append(float(best["realized_t_per_n"]))

    transfer_ratio = sorted(large_model_best_ratios)[len(large_model_best_ratios) // 2]
    initial_10m_tokens = int(round(10_000_000 * transfer_ratio))
    return {
        "per_model": per_model,
        "initial_10m_research_only_transfer": {
            "provisional_t_per_n": transfer_ratio,
            "provisional_optimized_tokens": initial_10m_tokens,
            "derivation": "upper-median observed best T/N from 500K and 1M bounded-real runs",
            "status": "NOT_LAUNCH_AUTHORITY_RETEST_ON_REPRESENTATIVE_CORPUS_AND_TARGET_HARDWARE",
            "paid_compute_authorized": False,
        },
        "selection_basis": "lowest held-out BPB on the observed common T/N grid; train loss never selects a checkpoint",
    }


def _run_resume_child(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"])
    output_dir = Path(plan["output_dir"])
    intake_dir = output_dir / "resume-intake"
    tokenizer = ByteTokenizer()
    train_records, validation_records, data = _real_corpus_records(root, intake_dir)
    train_batches = _tensor_batches_from_records(
        train_records,
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=True,
    )
    validation_batches = _tensor_batches_from_records(
        validation_records,
        split="validation",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=False,
    )
    data["training_trace_sha256"] = _batch_trace_digest(train_batches)
    _require(data["dataset_identity_sha256"] == plan["dataset_identity_sha256"], "resume child dataset identity drift")
    _require(data["intake_manifest_sha256"] == plan["intake_manifest_sha256"], "resume child intake manifest drift")
    _require(data["training_trace_sha256"] == plan["training_trace_sha256"], "resume child batch trace drift")

    spec = ModelSpec.from_dict(plan["model_spec"])
    init_spec = InitSpec.from_dict(plan["init_spec"])
    config = TrainerConfig(**plan["trainer_config"])
    torch.manual_seed(config.seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")
    load_trainer_checkpoint(
        plan["checkpoint_path"],
        model=model,
        trainer=trainer,
        expected_git_sha=plan["source_sha"],
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=plan["intake_manifest_sha256"],
        expected_run_manifest_hash=plan["run_manifest_sha256"],
        expected_seed=config.seed,
    )
    _require(trainer.optimizer_step == plan["resume_optimizer_step"], "resume child optimizer-step restore drift")
    _require(trainer.tokens_seen == plan["resume_optimized_tokens"], "resume child optimized-token restore drift")

    final_steps = int(plan["final_optimizer_steps"])
    while trainer.optimizer_step < final_steps:
        batch = train_batches[trainer.optimizer_step % len(train_batches)]
        metrics = trainer.train_microbatch(batch)
        _require(metrics.optimizer_stepped, "resume child did not optimizer-step")

    final_train = _evaluate_nonmutating(
        model,
        trainer,
        train_batches,
        bootstrap_seed=SEED + 900_001,
    )
    final_validation = _evaluate_nonmutating(
        model,
        trainer,
        validation_batches,
        bootstrap_seed=SEED + 900_002,
    )
    child_report = {
        "fresh_process": True,
        "pid": os.getpid(),
        "loaded_checkpoint": plan["checkpoint_path"],
        "final_optimizer_steps": trainer.optimizer_step,
        "final_optimized_tokens": trainer.tokens_seen,
        "final_model_sha256": _model_digest(model),
        "final_trainer_state_sha256": _state_digest(trainer.state_dict()),
        "final_training_bpb": final_train["bpb"],
        "final_validation_bpb": final_validation["bpb"],
    }
    Path(plan["child_report_path"]).write_text(
        json.dumps(child_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _prove_fresh_process_resume(
    *,
    root: Path,
    output_dir: Path,
    source_sha: str,
    candidate: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    points = list(candidate["points"])
    _require(len(points) >= 4, "resume proof requires four T/N points")
    resume_point = points[2]
    final_point = points[-1]
    checkpoint_path = output_dir / resume_point["checkpoint"]["relative_path"]
    plan_path = output_dir / "fresh-process-resume-plan.json"
    child_report_path = output_dir / "fresh-process-resume-child.json"
    config = _trainer_config(int(final_point["optimizer_steps"]))
    plan = {
        "root": str(root),
        "output_dir": str(output_dir),
        "source_sha": source_sha,
        "checkpoint_path": str(checkpoint_path),
        "child_report_path": str(child_report_path),
        "dataset_identity_sha256": data["dataset_identity_sha256"],
        "intake_manifest_sha256": data["intake_manifest_sha256"],
        "training_trace_sha256": data["training_trace_sha256"],
        "run_manifest_sha256": candidate["run_manifest_sha256"],
        "model_spec": candidate["model_spec"],
        "init_spec": candidate["init_spec"],
        "trainer_config": asdict(config),
        "resume_optimizer_step": resume_point["optimizer_steps"],
        "resume_optimized_tokens": resume_point["optimized_tokens"],
        "final_optimizer_steps": final_point["optimizer_steps"],
    }
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--resume-child", str(plan_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"fresh-process resume child failed: {completed.stderr[-4000:]}")
    _require(child_report_path.is_file(), "fresh-process resume child report missing")
    child = json.loads(child_report_path.read_text(encoding="utf-8"))
    exact_model = child["final_model_sha256"] == candidate["final_model_sha256"]
    exact_trainer = child["final_trainer_state_sha256"] == candidate["final_trainer_state_sha256"]
    exact_tokens = child["final_optimized_tokens"] == candidate["final_optimized_tokens"]
    _require(exact_model, "fresh-process resumed final model differs from uninterrupted trajectory")
    _require(exact_trainer, "fresh-process resumed Trainer state differs from uninterrupted trajectory")
    _require(exact_tokens, "fresh-process resumed optimized-token ledger differs")
    return {
        "status": "PASS_EXACT",
        "candidate": candidate["label"],
        "resume_from_t_per_n": resume_point["realized_t_per_n"],
        "resume_from_optimized_tokens": resume_point["optimized_tokens"],
        "fresh_process_pid": child["pid"],
        "parent_pid": os.getpid(),
        "pid_differs": child["pid"] != os.getpid(),
        "final_model_exact": exact_model,
        "final_trainer_state_exact": exact_trainer,
        "final_optimized_tokens_exact": exact_tokens,
        "child_report": str(child_report_path.relative_to(output_dir)),
    }


def _machine_manifest(source_sha: str, lock_hash: str) -> dict[str, Any]:
    cpu_model = None
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = None
    return {
        "source_sha": source_sha,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "runtime_lock_bundle_sha256": lock_hash,
        "paid_compute": False,
    }


def run_experiment(*, source_sha: str, output_dir: Path, torch_threads: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    _require(len(source_sha) == 40 and all(ch in "0123456789abcdef" for ch in source_sha), "source SHA must be exact 40-hex")
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_hash = _lock_bundle_hash(root)

    tokenizer = ByteTokenizer()
    intake_dir = output_dir / "intake"
    train_records, validation_records, data = _real_corpus_records(root, intake_dir)
    _require(data["real_external_source_bytes"] is True, "RESEARCH-123 requires real external-source bytes")
    _require(data["project_authored_synthetic_fixture"] is False, "synthetic corpus is forbidden for RESEARCH-123")
    _require(data["all_accepted_records_approved_for_model_training"] is True, "real corpus rights gate failed")
    _require(data["representative_broad_pretraining_corpus"] is False, "truth boundary unexpectedly changed; re-review corpus before promotion")

    train_batches = _tensor_batches_from_records(
        train_records,
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=True,
    )
    validation_batches = _tensor_batches_from_records(
        validation_records,
        split="validation",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=False,
    )
    data["training_trace_sha256"] = _batch_trace_digest(train_batches)
    data["validation_trace_sha256"] = _batch_trace_digest(validation_batches)
    data["training_unique_scoreable_tokens"] = len(train_batches) * LOSS_TOKENS_PER_STEP
    data["fixed_recipe"] = {
        "tokenizer": BYTE_TOKENIZER_VERSION,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "packing": "incumbent.iter_packed_examples + batch_examples + collate_rows(labels)",
        "stream_recycling": "deterministic cyclic batch order",
        "validation_document_isolated": True,
    }

    train_prompt = train_records[0].text[:24]
    validation_prompt = validation_records[0].text[:24]
    candidates: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for label, spec in model_family():
        candidate_dir = output_dir / "candidate-work" / label
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate = _run_candidate(
            root=root,
            output_dir=output_dir,
            source_sha=source_sha,
            label=label,
            spec=spec,
            train_batches=train_batches,
            validation_batches=validation_batches,
            tokenizer=tokenizer,
            data=data,
            train_prompt=train_prompt,
            validation_prompt=validation_prompt,
            lock_hash=lock_hash,
        )
        candidates.append(candidate)

    resume_proof = _prove_fresh_process_resume(
        root=root,
        output_dir=output_dir,
        source_sha=source_sha,
        candidate=candidates[0],
        data=data,
    )
    curve = _fit_curve(candidates)
    recommendation = _recommend(candidates)

    selected_candidate: Mapping[str, Any] | None = None
    selected_point: Mapping[str, Any] | None = None
    for candidate in candidates:
        for point in candidate["points"]:
            if selected_point is None or point["validation_eval"]["bpb"] < selected_point["validation_eval"]["bpb"]:
                selected_candidate = candidate
                selected_point = point
    _require(selected_candidate is not None and selected_point is not None, "failed to select learned checkpoint")

    all_train_decrease = all(
        candidate["points"][-1]["training_eval"]["bpb"] < candidate["baseline"]["training_eval"]["bpb"]
        for candidate in candidates
    )
    all_val_improve_somewhere = all(
        min(point["validation_eval"]["bpb"] for point in candidate["points"])
        < candidate["baseline"]["validation_eval"]["bpb"]
        for candidate in candidates
    )
    _require(all_train_decrease, "one or more models failed train-BPB decrease proof")
    _require(all_val_improve_somewhere, "one or more models failed held-out-BPB improvement proof")

    report = {
        "schema_version": SCHEMA_VERSION,
        "authority": "LOCAL_FREE_REAL_BOUNDED_CORPUS_TN_EVIDENCE_NOT_REPRESENTATIVE_SCALING_LAW",
        "source": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "branch_expected": "research123/real-tn-scaling-20260826",
        },
        "constraints": {
            "paid_compute": False,
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "broad_intelligence_claim": False,
            "evaluation_tokens_in_optimized_token_accounting": 0,
        },
        "component_selection": {
            "model": "ModelSpec + TwelveSixDecoder fixed RESEARCH41 MHA family",
            "tokenizer": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "corpus": "DATA-21/22 bounded real rights-aware intake",
            "streaming_packing": "iter_packed_examples + batch_examples + collate_rows",
            "trainer_optimizer": "D02 Trainer + AdamW fp32 constant 3e-4 betas 0.9/0.95 clip 1.0",
            "checkpoint_resume": "D05 save_trainer_checkpoint/load_trainer_checkpoint",
            "heldout_evaluation": "document-isolated validation with explicit non-mutation fingerprints",
            "first_party_inference": "TwelveSixDecoder.generate greedy",
        },
        "data": data,
        "tokenizer": {
            "version": BYTE_TOKENIZER_VERSION,
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
            "vocab_size": tokenizer.vocab_size,
        },
        "common_t_per_n_design": {
            "requested_ratios": list(TARGET_TN_RATIOS),
            "log_spacing_factor": 4.0,
            "loss_tokens_per_full_optimizer_step": LOSS_TOKENS_PER_STEP,
            "evaluation_excluded": True,
        },
        "candidates": candidates,
        "fresh_process_resume": resume_proof,
        "descriptive_curve": curve,
        "recommendation": recommendation,
        "selected_learned_base_checkpoint": {
            "label": selected_candidate["label"],
            "parameter_count": selected_candidate["parameter_count"],
            "optimized_tokens": selected_point["optimized_tokens"],
            "realized_t_per_n": selected_point["realized_t_per_n"],
            "training_bpb": selected_point["training_eval"]["bpb"],
            "heldout_bpb": selected_point["validation_eval"]["bpb"],
            "checkpoint": selected_point["checkpoint"],
            "generation_before": selected_candidate["baseline"]["generation"],
            "generation_after": selected_point["generation"],
        },
        "proof_summary": {
            "random_initialization": True,
            "exact_parameter_counts": [candidate["parameter_count"] for candidate in candidates],
            "real_training_eligible_corpus": True,
            "representative_broad_corpus": False,
            "versioned_tokenizer": True,
            "train_bpb_decreased_all_models": all_train_decrease,
            "heldout_bpb_improved_somewhere_all_models": all_val_improve_somewhere,
            "multiple_checkpoints_per_model": all(len(candidate["points"]) == len(TARGET_TN_RATIOS) for candidate in candidates),
            "fresh_process_resume_exact": resume_proof["status"] == "PASS_EXACT",
            "evaluation_non_mutation": True,
            "generation_before_after": True,
            "retained_exact_checkpoint": True,
        },
        "milestone100_status": {
            "status": "BLOCKED_ON_REPRESENTATIVE_CORPUS",
            "reason": (
                "The strongest live rights-approved real incumbent is only the DATA-21/22 three-object bounded sample. "
                "It is explicitly non-representative, so the exact user milestone cannot honestly be declared PASS."
            ),
            "genuinely_learned_base_artifact_present": True,
            "artifact_scope": "real-bounded-corpus experimental Base only",
        },
        "research123_status": {
            "status": "PASS_IF_THIS_REPORT_VALIDATES",
            "scope": "descriptive T/N evidence for this exact bounded real corpus and small-model family only",
            "universal_scaling_law": False,
        },
        "machine_manifest": _machine_manifest(source_sha, lock_hash),
        "total_experiment_wall_seconds": time.perf_counter() - run_start,
        "reproduction_command": (
            f"python tools/research123_real_tn_scaling.py --source-sha {source_sha} "
            "--output-dir evidence/research123 --torch-threads 2"
        ),
    }
    report["report_sha256"] = hash_json(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA_VERSION, "report schema drift")
    proof = report.get("proof_summary")
    _require(isinstance(proof, Mapping), "proof summary missing")
    required_true = (
        "random_initialization",
        "versioned_tokenizer",
        "train_bpb_decreased_all_models",
        "heldout_bpb_improved_somewhere_all_models",
        "multiple_checkpoints_per_model",
        "fresh_process_resume_exact",
        "evaluation_non_mutation",
        "generation_before_after",
        "retained_exact_checkpoint",
    )
    for field in required_true:
        _require(proof.get(field) is True, f"proof field failed: {field}")
    _require(proof.get("real_training_eligible_corpus") is True, "real corpus proof failed")
    _require(proof.get("representative_broad_corpus") is False, "representative-corpus truth boundary must remain false")
    _require(report["milestone100_status"]["status"] == "BLOCKED_ON_REPRESENTATIVE_CORPUS", "milestone truth boundary drift")
    _require(len(report.get("candidates", [])) == 4, "expected four model sizes")
    for candidate in report["candidates"]:
        _require(len(candidate["points"]) == 4, "candidate missing common T/N points")
        for point in candidate["points"]:
            _require(point["training_eval"]["evaluation_optimized_tokens"] == 0, "train evaluation entered optimized ledger")
            _require(point["validation_eval"]["evaluation_optimized_tokens"] == 0, "validation entered optimized ledger")
            _require(point["training_eval"]["non_mutation_proof"]["pass"] is True, "training evaluation mutated state")
            _require(point["validation_eval"]["non_mutation_proof"]["pass"] is True, "validation evaluation mutated state")
    recorded_hash = report.get("report_sha256")
    _require(isinstance(recorded_hash, str) and len(recorded_hash) == 64, "report hash missing")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(hash_json(core) == recorded_hash, "report self-hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha")
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/research123"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resume-child", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()

    if args.resume_child is not None:
        return _run_resume_child(args.resume_child)
    if args.validate is not None:
        report = json.loads(args.validate.read_text(encoding="utf-8"))
        validate_report(report)
        print(json.dumps({"status": "PASS", "report_sha256": report["report_sha256"]}, sort_keys=True))
        return 0

    _require(args.source_sha is not None, "--source-sha is required")
    _require(args.torch_threads >= 1, "--torch-threads must be >=1")
    report = run_experiment(
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        torch_threads=args.torch_threads,
    )
    validate_report(report)
    output = args.output or (args.output_dir / "research123-real-tn-scaling.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS_REAL_BOUNDED_ONLY",
                "report": str(output),
                "report_sha256": report["report_sha256"],
                "selected": report["selected_learned_base_checkpoint"],
                "milestone100_status": report["milestone100_status"],
                "recommendation": report["recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
