"""Strict fixed-token scaling and ~1M MHA/GQA research for 12-6 AI.

RESEARCH-06 extends the terminal-green RESEARCH41 controlled family. MODEL-10
reuses the incumbent model-native GQA/KV-cache implementation from MODEL-35.
All evidence is LOCAL_FREE CPU research evidence on the tiny project-authored
S0 fixture; it is not stage promotion or broad-corpus capability evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID,
    TOKENIZER_ID,
    _byte_stream,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _validation_loss,
    controlled_specs,
)
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.fixed-token-scaling-gqa.v1"
CANDIDATE_SCHEMA = "12-6.fixed-token-candidate.v1"
AUTHORITY = "LOCAL_FREE_FIXED_TOKEN_HELDOUT_RESEARCH_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
MASKING_VERSION = "research06-exact-valid-causal-targets-v1"
FIXED_PLAN_PATH = "configs/experiments/research06_fixed_token_scaling_v1.json"
GQA_PLAN_PATH = "configs/experiments/model10_1m_mha_gqa_v1.json"
_HEX40 = set("0123456789abcdef")


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _exact_sha(value: str) -> bool:
    return len(value) == 40 and value == value.lower() and all(ch in _HEX40 for ch in value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _model_from_geometry(geometry: Mapping[str, Any]) -> ModelSpec:
    base = controlled_specs()[-1].to_dict()
    for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff"):
        if key in geometry:
            base[key] = int(geometry[key])
    base["rope_rotary_dim"] = int(base["head_dim"])
    return ModelSpec.from_dict(base)


def _load_plans(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, ModelSpec]]:
    fixed = _load_json(repo_root / FIXED_PLAN_PATH)
    attention = _load_json(repo_root / GQA_PLAN_PATH)
    if fixed.get("schema") != "12-6.fixed-token-scaling-plan.v1":
        raise ValueError("fixed-token plan schema drift")
    if attention.get("schema") != "12-6.model10-gqa1m-plan.v1":
        raise ValueError("MODEL-10 plan schema drift")

    incumbent_specs = controlled_specs()
    labels = fixed.get("candidate_labels")
    if labels != ["P100K_MHA", "P250K_MHA", "P500K_MHA", "P1M_MHA"]:
        raise ValueError("fixed-token candidate label family drift")
    expected_counts = [95_568, 267_912, 467_808, 1_037_696]
    if [spec.parameter_count() for spec in incumbent_specs] != expected_counts:
        raise RuntimeError("RESEARCH41 controlled family parameter drift")
    candidates = dict(zip(labels, incumbent_specs, strict=True))

    if attention.get("mha_control_label") != "P1M_MHA":
        raise ValueError("MODEL-10 must reuse the fixed-control 1M MHA")
    gqa_label = str(attention.get("gqa_candidate_label"))
    if gqa_label != "P1M_GQA4":
        raise ValueError("unexpected MODEL-10 GQA label")
    gqa = _model_from_geometry(attention["gqa_geometry"])
    candidates[gqa_label] = gqa
    expected_gqa = int(attention["expected_parameters"])
    if gqa.parameter_count() != expected_gqa:
        raise RuntimeError(
            f"MODEL-10 GQA parameter drift: {gqa.parameter_count()} != {expected_gqa}"
        )
    mha = candidates["P1M_MHA"]
    relative_gap = abs(gqa.parameter_count() - mha.parameter_count()) / mha.parameter_count()
    if relative_gap > float(attention["max_parameter_relative_gap"]):
        raise RuntimeError("MODEL-10 parameter matching tolerance exceeded")
    if gqa.n_kv_heads >= gqa.n_heads:
        raise RuntimeError("MODEL-10 GQA candidate must reduce K/V heads")
    return fixed, attention, candidates


def strict_step_plan(token_budgets: tuple[int, ...], capacity: int) -> list[dict[str, int | bool]]:
    """Plan optimizer steps that land exactly on every cumulative token budget."""
    if not token_budgets or tuple(sorted(set(token_budgets))) != token_budgets:
        raise ValueError("token budgets must be strictly increasing and unique")
    if token_budgets[0] <= 0 or capacity <= 0:
        raise ValueError("token budgets and capacity must be positive")
    plan: list[dict[str, int | bool]] = []
    seen = 0
    for budget in token_budgets:
        if budget <= seen:
            raise ValueError("token budget must exceed prior cumulative total")
        while seen < budget:
            valid = min(capacity, budget - seen)
            seen += valid
            plan.append(
                {
                    "optimizer_step": len(plan) + 1,
                    "valid_loss_tokens": valid,
                    "cumulative_optimized_tokens": seen,
                    "budget_boundary": seen == budget,
                }
            )
    return plan


def make_exact_causal_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
    valid_loss_tokens: int,
) -> dict[str, torch.Tensor]:
    """Build aligned next-token targets with exactly N valid causal loss pairs."""
    capacity = batch_size * (sequence_length - 1)
    if not 1 <= valid_loss_tokens <= capacity:
        raise ValueError(f"valid_loss_tokens must be in [1, {capacity}]")
    input_ids = _make_batch(
        stream,
        step=step,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    target_ids = torch.zeros_like(input_ids)
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, -1] = input_ids[:, -1]
    loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
    remaining = valid_loss_tokens
    for row in range(batch_size):
        take = min(remaining, sequence_length - 1)
        if take:
            loss_mask[row, :take] = 1
            remaining -= take
    if remaining != 0 or int(loss_mask.sum().item()) != valid_loss_tokens:
        raise RuntimeError("exact causal loss-mask construction drift")
    if bool(loss_mask[:, -1].any().item()):
        raise RuntimeError("final sequence position cannot be a valid shifted causal pair")
    return {"input_ids": input_ids, "target_ids": target_ids, "loss_mask": loss_mask}


def _batch_trace_hash(batch: Mapping[str, torch.Tensor], *, step: int) -> str:
    digest = hashlib.sha256(str(step).encode("ascii"))
    for key in ("input_ids", "target_ids", "loss_mask"):
        tensor = batch[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _combine_trace_hashes(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _tensor_l2(model: TwelveSixDecoder) -> float:
    squared = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float()
        squared += float(torch.sum(value * value).item())
    return math.sqrt(squared)


def _weight_delta(
    model: TwelveSixDecoder, before: Mapping[str, torch.Tensor]
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    l2 = math.sqrt(squared)
    return {
        "l2": l2,
        "max_abs": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
    }


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _nested_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) and not isinstance(left, type):
        left = asdict(left)
    if is_dataclass(right) and not isinstance(right, type):
        right = asdict(right)
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(_nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _models_equal(left: TwelveSixDecoder, right: TwelveSixDecoder) -> bool:
    a = left.state_dict()
    b = right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _rss_bytes(field: str) -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith(field + ":"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _trainer_config(*, max_steps: int, seed: int, fixed: Mapping[str, Any]) -> TrainerConfig:
    optimizer = fixed["optimizer"]
    return TrainerConfig(
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=tuple(float(v) for v in optimizer["betas"]),
        eps=float(optimizer["eps"]),
        max_steps=max_steps,
        warmup_steps=int(optimizer["warmup_steps"]),
        scheduler=str(optimizer["scheduler"]),
        gradient_accumulation_steps=1,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    trainer: Trainer,
    tokenizer: ByteTokenizer,
    dataset_manifest_sha: str,
    train_sha: str,
    environment_lock_sha: str,
    plan_hash: str,
    candidate_label: str,
) -> CheckpointIdentity:
    packing_hash = hash_json({"packing_id": PACKING_ID, "masking_version": MASKING_VERSION})
    training_config = {
        "authority": AUTHORITY,
        "candidate_label": candidate_label,
        "init_spec_sha256": init_spec.identity_sha256(),
        "fixed_token_plan_sha256": plan_hash,
        "data": {
            "split_identity": f"controlled-s0-train:{train_sha}",
            "packing_sha256": packing_hash,
            "packing_version": MASKING_VERSION,
        },
        "trainer": asdict(trainer.config),
    }
    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "candidate_label": candidate_label,
        "model_spec_sha256": spec.identity_sha256(),
        "fixed_token_plan_sha256": plan_hash,
        "step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_manifest_sha,
        run_manifest_hash=hash_json(run_manifest),
        training_config=training_config,
        seed=trainer.config.seed,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "lr": trainer.config.learning_rate,
            "betas": list(trainer.config.betas),
            "eps": trainer.config.eps,
            "weight_decay": trainer.config.weight_decay,
        },
        scheduler={"name": trainer.config.scheduler},
        environment_lock_hash=environment_lock_sha,
    )


def _validate_without_optimization(
    model: TwelveSixDecoder,
    trainer: Trainer,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> tuple[float, int, float]:
    tokens_before = trainer.tokens_seen
    step_before = trainer.optimizer_step
    started = time.perf_counter()
    loss, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    seconds = time.perf_counter() - started
    if trainer.tokens_seen != tokens_before or trainer.optimizer_step != step_before:
        raise RuntimeError("evaluation mutated optimized-token or optimizer-step counters")
    return loss, validation_tokens, seconds


def _run_plan(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    stream: bytes,
    step_plan: list[dict[str, int | bool]],
    batch_size: int,
    sequence_length: int,
    start_index: int,
    end_index: int,
    trace_hashes: list[str] | None = None,
    collect_metrics: bool = False,
) -> dict[str, Any]:
    losses: list[float] = []
    grad_norms: list[float] = []
    step_seconds: list[float] = []
    clipped = 0
    token_sum = 0
    for index in range(start_index, end_index):
        planned = step_plan[index]
        valid = int(planned["valid_loss_tokens"])
        batch = make_exact_causal_batch(
            stream,
            step=index,
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_loss_tokens=valid,
        )
        if trace_hashes is not None:
            trace_hashes.append(_batch_trace_hash(batch, step=index))
        before_tokens = trainer.tokens_seen
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        elapsed = time.perf_counter() - started
        if not metrics.optimizer_stepped or metrics.grad_norm is None:
            raise RuntimeError("fixed-token plan requires one committed optimizer step per batch")
        if metrics.tokens != valid or trainer.tokens_seen - before_tokens != valid:
            raise RuntimeError("valid causal loss-token accounting drift")
        if trainer.tokens_seen != int(planned["cumulative_optimized_tokens"]):
            raise RuntimeError("cumulative optimized-token accounting drift")
        token_sum += metrics.tokens
        if collect_metrics:
            losses.append(float(metrics.loss))
            grad_norms.append(float(metrics.grad_norm))
            step_seconds.append(elapsed)
            if (
                trainer.config.gradient_clip_norm is not None
                and metrics.grad_norm > trainer.config.gradient_clip_norm
            ):
                clipped += 1
    trainer.assert_checkpoint_safe()
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "step_seconds": step_seconds,
        "clip_count": clipped,
        "optimized_tokens_observed": token_sum,
    }


def _cache_bytes(cache: Any) -> int:
    return int(
        sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            for layer in cache.layers
        )
    )


@torch.no_grad()
def _generation_probe(
    model: TwelveSixDecoder, *, prompt_len: int = 64, steps: int = 8
) -> dict[str, Any]:
    model.eval()
    row = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0) % min(model.spec.vocab_size, 64)
    cached_out, cache = model.prefill_kv_cache(row)
    cached_tokens: list[int] = []
    for index in range(steps):
        token = int(torch.argmax(cached_out.logits[:, -1, :], dim=-1).item())
        cached_tokens.append(token)
        if index + 1 < steps:
            cached_out, cache = model.decode_one_with_kv_cache(
                torch.tensor([[token]], dtype=torch.long), cache
            )
    stateless = row.clone()
    stateless_tokens: list[int] = []
    for _ in range(steps):
        output = model(stateless)
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        stateless_tokens.append(int(next_token.item()))
        stateless = torch.cat((stateless, next_token), dim=1)
    if cached_tokens != stateless_tokens:
        raise RuntimeError("cached/stateless greedy generation diverged")
    spec = model.spec
    return {
        "cached_stateless_greedy_exact": True,
        "generated_token_ids": cached_tokens,
        "actual_cache_bytes": _cache_bytes(cache),
        "cache_sequence_length": cache.sequence_length,
        "full_context_kv_cache_bf16_bytes": (
            2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * spec.max_seq_len * 2
        ),
        "full_context_kv_cache_fp32_bytes": (
            2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * spec.max_seq_len * 4
        ),
        "cpu_attention_execution": (
            "MHA_DIRECT_REFERENCE"
            if spec.n_kv_heads == spec.n_heads
            else "EXPLICIT_EXPANDED_KV_REFERENCE_FALLBACK"
        ),
        "gpu_speed_inference_authorized": False,
    }


def run_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    candidate_label: str,
    output_dir: Path,
    torch_threads: int,
) -> dict[str, Any]:
    if not _exact_sha(source_sha):
        raise ValueError("source_sha must be lowercase 40-hex")
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact checkout mismatch")
    fixed, _, candidates = _load_plans(repo_root)
    if candidate_label not in candidates:
        raise ValueError(f"unknown candidate: {candidate_label}")
    spec = candidates[candidate_label]
    init_spec = InitSpec()
    budgets = tuple(int(v) for v in fixed["token_budgets"])
    batch_size = int(fixed["batch_size"])
    sequence_length = int(fixed["sequence_length"])
    seed = int(fixed["seed"])
    capacity = batch_size * (sequence_length - 1)
    step_plan = strict_step_plan(budgets, capacity)
    plan_hash = _canonical_hash(step_plan)
    resume_budget = int(fixed["resume_budget"])
    resume_index = next(
        index + 1
        for index, item in enumerate(step_plan)
        if int(item["cumulative_optimized_tokens"]) == resume_budget
    )
    max_steps = len(step_plan)
    config = _trainer_config(max_steps=max_steps, seed=seed, fixed=fixed)

    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    environment_path = repo_root / "requirements/locks/index.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    overlap = sorted(
        {str(r["id"]) for r in train_records}
        & {str(r["id"]) for r in validation_records}
    )
    if overlap:
        raise RuntimeError(f"train/validation overlap: {overlap!r}")
    stream = _byte_stream(train_records, tokenizer)
    dataset_manifest_sha = sha256_file(manifest_path)
    train_sha = sha256_file(train_path)
    environment_sha = sha256_file(environment_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "resume-checkpoint"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    rss_start = _rss_bytes("VmRSS")
    random.seed(seed)
    torch.manual_seed(seed)
    primary_model = TwelveSixDecoder(spec, init_spec)
    actual_parameters = sum(p.numel() for p in primary_model.parameters() if p.requires_grad)
    if actual_parameters != spec.parameter_count():
        raise RuntimeError("instantiated parameter count drift")
    initial_snapshot = _snapshot(primary_model)
    initial_weight_l2 = _tensor_l2(primary_model)
    primary_trainer = Trainer(primary_model, config, device="cpu")
    rss_after_init = _rss_bytes("VmRSS")
    initial_loss, validation_tokens, initial_eval_seconds = _validate_without_optimization(
        primary_model, primary_trainer, validation_records, tokenizer
    )

    trace_hashes: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    training_losses: list[float] = []
    grad_norms: list[float] = []
    step_seconds: list[float] = []
    clip_count = 0
    evaluation_seconds = initial_eval_seconds
    train_seconds = 0.0
    checkpoint_seconds = 0.0
    trajectory_started = time.perf_counter()

    start_index = 0
    for boundary_budget in budgets:
        boundary_index = next(
            index + 1
            for index, item in enumerate(step_plan)
            if int(item["cumulative_optimized_tokens"]) == boundary_budget
        )
        segment = _run_plan(
            model=primary_model,
            trainer=primary_trainer,
            stream=stream,
            step_plan=step_plan,
            batch_size=batch_size,
            sequence_length=sequence_length,
            start_index=start_index,
            end_index=boundary_index,
            trace_hashes=trace_hashes,
            collect_metrics=True,
        )
        training_losses.extend(segment["losses"])
        grad_norms.extend(segment["grad_norms"])
        step_seconds.extend(segment["step_seconds"])
        clip_count += int(segment["clip_count"])
        train_seconds += sum(segment["step_seconds"])
        if primary_trainer.tokens_seen != boundary_budget:
            raise RuntimeError("failed to land exactly on fixed token budget")
        validation_loss, checked_tokens, eval_seconds = _validate_without_optimization(
            primary_model, primary_trainer, validation_records, tokenizer
        )
        if checked_tokens != validation_tokens:
            raise RuntimeError("validation target count drift")
        evaluation_seconds += eval_seconds
        checkpoint = {
            "requested_token_budget": boundary_budget,
            "optimized_tokens": primary_trainer.tokens_seen,
            "optimizer_steps": primary_trainer.optimizer_step,
            "compute_proxy": 6 * actual_parameters * primary_trainer.tokens_seen,
            "validation_loss": validation_loss,
            "validation_bpb": validation_loss / math.log(2.0),
            "validation_loss_tokens": checked_tokens,
            "evaluation_optimized_tokens": 0,
            "last_train_loss": training_losses[-1],
            "last_grad_norm": grad_norms[-1],
        }
        checkpoints.append(checkpoint)

        if boundary_budget == resume_budget:
            identity = _checkpoint_identity(
                source_sha=source_sha,
                spec=spec,
                init_spec=init_spec,
                trainer=primary_trainer,
                tokenizer=tokenizer,
                dataset_manifest_sha=dataset_manifest_sha,
                train_sha=train_sha,
                environment_lock_sha=environment_sha,
                plan_hash=plan_hash,
                candidate_label=candidate_label,
            )
            started = time.perf_counter()
            manifest = save_trainer_checkpoint(
                checkpoint_dir,
                model=primary_model,
                trainer=primary_trainer,
                identity=identity,
            )
            del primary_trainer, primary_model
            gc.collect()
            restored_model = TwelveSixDecoder(spec, init_spec)
            restored_trainer = Trainer(restored_model, config, device="cpu")
            packing_hash = hash_json({"packing_id": PACKING_ID, "masking_version": MASKING_VERSION})
            load_trainer_checkpoint(
                checkpoint_dir,
                model=restored_model,
                trainer=restored_trainer,
                restore_rng=True,
                expected_git_sha=source_sha,
                expected_model_spec_hash=spec.identity_sha256(),
                expected_init_spec_hash=init_spec.identity_sha256(),
                expected_tokenizer_hash=tokenizer.identity.config_sha256,
                expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
                expected_dataset_manifest_hash=dataset_manifest_sha,
                expected_split_identity=f"controlled-s0-train:{train_sha}",
                expected_packing_hash=packing_hash,
                expected_packing_version=MASKING_VERSION,
                expected_run_manifest_hash=identity.run_manifest_hash,
                expected_training_config_hash=hash_json(identity.training_config),
                expected_environment_lock_hash=environment_sha,
                expected_seed=seed,
            )
            checkpoint_seconds += time.perf_counter() - started
            primary_model = restored_model
            primary_trainer = restored_trainer
            if (
                primary_trainer.tokens_seen != resume_budget
                or primary_trainer.optimizer_step != resume_index
            ):
                raise RuntimeError("checkpoint did not restore exact token/step boundary")
            resume_manifest = manifest
        start_index = boundary_index

    trajectory_wall = time.perf_counter() - trajectory_started + initial_eval_seconds
    if primary_trainer.tokens_seen != budgets[-1]:
        raise RuntimeError("final exact-token budget drift")
    if sum(int(item["valid_loss_tokens"]) for item in step_plan) != primary_trainer.tokens_seen:
        raise RuntimeError("step-plan token sum drift")
    if len(trace_hashes) != len(step_plan):
        raise RuntimeError("token trace did not cover every optimizer step")

    final_delta = _weight_delta(primary_model, initial_snapshot)
    final_weight_l2 = _tensor_l2(primary_model)
    final_delta["relative_to_initial_weight_l2"] = (
        float(final_delta["l2"]) / initial_weight_l2 if initial_weight_l2 else None
    )
    model_parameter_bytes = sum(p.numel() * p.element_size() for p in primary_model.parameters())
    optimizer_tensor_bytes = _tensor_bytes(primary_trainer.optimizer.state_dict())
    rss_after_train = _rss_bytes("VmRSS")
    peak_rss = _rss_bytes("VmHWM")

    random.seed(seed)
    torch.manual_seed(seed)
    control_model = TwelveSixDecoder(spec, init_spec)
    control_trainer = Trainer(control_model, config, device="cpu")
    control_trace: list[str] = []
    _run_plan(
        model=control_model,
        trainer=control_trainer,
        stream=stream,
        step_plan=step_plan,
        batch_size=batch_size,
        sequence_length=sequence_length,
        start_index=0,
        end_index=len(step_plan),
        trace_hashes=control_trace,
        collect_metrics=False,
    )
    trace_exact = trace_hashes == control_trace
    resume_model_exact = _models_equal(primary_model, control_model)
    resume_trainer_exact = _nested_equal(primary_trainer.state_dict(), control_trainer.state_dict())
    if not trace_exact or not resume_model_exact or not resume_trainer_exact:
        raise RuntimeError("resumed trajectory diverged from uninterrupted exact-token control")

    generation = None
    if candidate_label in {"P1M_MHA", "P1M_GQA4"}:
        generation = _generation_probe(primary_model)

    report: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "candidate_label": candidate_label,
        "model": {
            "model_spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": init_spec.identity_sha256(),
            "parameters": actual_parameters,
            "parameter_breakdown": spec.parameter_breakdown(),
        },
        "controls": {
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "vocab_size": tokenizer.vocab_size,
            "model_max_seq_len": spec.max_seq_len,
            "training_sequence_length": sequence_length,
            "batch_size": batch_size,
            "seed": seed,
            "optimizer": asdict(config),
            "packing_id": PACKING_ID,
            "masking_version": MASKING_VERSION,
            "token_budgets": list(budgets),
            "fixed_token_plan_sha256": plan_hash,
            "fixed_token_step_count": len(step_plan),
            "full_step_valid_loss_token_capacity": capacity,
        },
        "data": {
            "manifest_sha256": _file_sha256(manifest_path),
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "train_validation_record_overlap": overlap,
            "unique_train_stream_bytes": len(stream),
            "repeated_fixture": True,
        },
        "token_accounting": {
            "optimized_tokens_final": primary_trainer.tokens_seen,
            "expected_optimized_tokens_final": budgets[-1],
            "evaluation_loss_tokens_per_pass": validation_tokens,
            "evaluation_optimized_tokens_total": 0,
            "all_budget_boundaries_exact": all(
                int(point["optimized_tokens"]) == int(point["requested_token_budget"])
                for point in checkpoints
            ),
            "batch_trace_sha256": _combine_trace_hashes(trace_hashes),
            "resume_control_trace_exact": trace_exact,
        },
        "heldout": {
            "initial_validation_loss": initial_loss,
            "initial_validation_bpb": initial_loss / math.log(2.0),
            "checkpoints": checkpoints,
            "final_validation_loss": checkpoints[-1]["validation_loss"],
            "final_validation_bpb": checkpoints[-1]["validation_bpb"],
            "validation_loss_improvement": initial_loss - checkpoints[-1]["validation_loss"],
        },
        "optimization": {
            "train_loss_first": training_losses[0],
            "train_loss_last": training_losses[-1],
            "gradient_norm_min": min(grad_norms),
            "gradient_norm_median": statistics.median(grad_norms),
            "gradient_norm_max": max(grad_norms),
            "clip_count": clip_count,
            "clip_frequency": clip_count / len(grad_norms),
            "parameter_update": final_delta,
            "initial_weight_l2": initial_weight_l2,
            "final_weight_l2": final_weight_l2,
        },
        "timing": {
            "primary_trajectory_wall_seconds": trajectory_wall,
            "training_step_wall_seconds": train_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "checkpoint_save_reload_wall_seconds": checkpoint_seconds,
            "median_train_step_seconds": statistics.median(step_seconds),
            "p95_train_step_seconds": sorted(step_seconds)[
                max(0, math.ceil(0.95 * len(step_seconds)) - 1)
            ],
            "optimized_tokens_per_training_step_second": budgets[-1] / train_seconds,
        },
        "memory": {
            "rss_bytes_process_start": rss_start,
            "rss_bytes_after_model_init": rss_after_init,
            "rss_bytes_after_training": rss_after_train,
            "peak_rss_bytes_fresh_candidate_process_including_instrumentation": peak_rss,
            "model_parameter_tensor_bytes": model_parameter_bytes,
            "optimizer_tensor_bytes": optimizer_tensor_bytes,
            "model_plus_optimizer_tensor_bytes": model_parameter_bytes + optimizer_tensor_bytes,
        },
        "resume": {
            "resume_budget": resume_budget,
            "checkpoint_id": resume_manifest["checkpoint_id"],
            "checkpoint_directory_bytes": _directory_bytes(checkpoint_dir),
            "fresh_object_load": True,
            "model_state_exact_vs_uninterrupted": resume_model_exact,
            "trainer_optimizer_state_exact_vs_uninterrupted": resume_trainer_exact,
        },
        "attention_probe": generation,
        "truth_boundary": {
            "fixture_scope": "TINY_PROJECT_AUTHORED_S0_HELDOUT_ONLY",
            "broad_corpus_generalization_claim": False,
            "gpu_performance_claim": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    (output_dir / "candidate-evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _ranking_rows(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        if item["candidate_label"] == "P1M_GQA4":
            continue
        n = int(item["model"]["parameters"])
        final = float(item["heldout"]["final_validation_loss"])
        improvement = float(item["heldout"]["validation_loss_improvement"])
        tokens = int(item["token_accounting"]["optimized_tokens_final"])
        compute = 6 * n * tokens
        wall = float(item["timing"]["primary_trajectory_wall_seconds"])
        rows.append(
            {
                "candidate_label": item["candidate_label"],
                "parameters": n,
                "final_validation_loss": final,
                "final_validation_bpb": float(item["heldout"]["final_validation_bpb"]),
                "validation_improvement": improvement,
                "compute_proxy": compute,
                "primary_trajectory_wall_seconds": wall,
                "improvement_per_parameter": improvement / n,
                "improvement_per_compute_proxy_unit": improvement / compute,
                "improvement_per_wall_second": improvement / wall,
            }
        )
    return {
        "best_validation": sorted(rows, key=lambda row: row["final_validation_loss"]),
        "validation_improvement_per_parameter": sorted(
            rows, key=lambda row: row["improvement_per_parameter"], reverse=True
        ),
        "validation_improvement_per_compute": sorted(
            rows,
            key=lambda row: row["improvement_per_compute_proxy_unit"],
            reverse=True,
        ),
        "validation_improvement_per_wall_second": sorted(
            rows, key=lambda row: row["improvement_per_wall_second"], reverse=True
        ),
    }


def aggregate_reports(
    *, repo_root: Path, source_sha: str, input_dir: Path, output_path: Path
) -> dict[str, Any]:
    fixed, attention, _ = _load_plans(repo_root)
    reports: list[dict[str, Any]] = []
    for label in [*fixed["candidate_labels"], attention["gqa_candidate_label"]]:
        path = input_dir / label / "candidate-evidence.json"
        report = _load_json(path)
        validate_candidate(report, expected_source_sha=source_sha)
        reports.append(report)
    trace_hashes = {item["token_accounting"]["batch_trace_sha256"] for item in reports}
    if len(trace_hashes) != 1:
        raise RuntimeError("candidate data/token traces are not identical")
    control_fingerprints = {
        _canonical_hash(
            {
                "controls": item["controls"],
                "data": item["data"],
            }
        )
        for item in reports
    }
    if len(control_fingerprints) != 1:
        raise RuntimeError("fixed scientific controls drifted between candidates")

    rankings = _ranking_rows(reports)
    scale_by_label = {item["candidate_label"]: item for item in reports}
    best = rankings["best_validation"][0]
    tolerance = float(fixed["primary_vehicle_max_relative_loss_gap"])
    eligible = [
        row
        for row in rankings["best_validation"]
        if row["final_validation_loss"] <= best["final_validation_loss"] * (1.0 + tolerance)
    ]
    primary = min(eligible, key=lambda row: row["parameters"])

    mha = scale_by_label["P1M_MHA"]
    gqa = scale_by_label["P1M_GQA4"]
    mha_loss = float(mha["heldout"]["final_validation_loss"])
    gqa_loss = float(gqa["heldout"]["final_validation_loss"])
    relative_quality_delta = (gqa_loss - mha_loss) / mha_loss
    mha_cache = int(mha["attention_probe"]["full_context_kv_cache_bf16_bytes"])
    gqa_cache = int(gqa["attention_probe"]["full_context_kv_cache_bf16_bytes"])
    quality_guardrail = float(attention["quality_relative_loss_guardrail"])
    if relative_quality_delta > quality_guardrail:
        attention_recommendation = "RETAIN_MHA_AT_1M_GQA_HELDOUT_WORSE_THAN_GUARDRAIL"
    elif relative_quality_delta < -quality_guardrail:
        attention_recommendation = "GQA_QUALITY_SIGNAL_FAVORS_REPLICATION_NOT_DEFAULT_PROMOTION"
    else:
        attention_recommendation = "RETAIN_MHA_DEFAULT_AT_1M_GQA_QUALIFIED_COMPARATOR"

    aggregate: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "paid_compute": False,
        },
        "incumbents": {
            "research41_exact_green_sha": fixed["research41_exact_green_sha"],
            "model35_exact_green_sha": attention["model35_exact_green_sha"],
        },
        "fixed_controls": reports[0]["controls"],
        "fixed_data": reports[0]["data"],
        "candidate_summaries": [
            {
                "candidate_label": item["candidate_label"],
                "parameters": item["model"]["parameters"],
                "final_validation_loss": item["heldout"]["final_validation_loss"],
                "final_validation_bpb": item["heldout"]["final_validation_bpb"],
                "validation_improvement": item["heldout"]["validation_loss_improvement"],
                "compute_proxy": (
                    6
                    * int(item["model"]["parameters"])
                    * int(item["token_accounting"]["optimized_tokens_final"])
                ),
                "primary_trajectory_wall_seconds": item["timing"][
                    "primary_trajectory_wall_seconds"
                ],
                "peak_rss_bytes": item["memory"][
                    "peak_rss_bytes_fresh_candidate_process_including_instrumentation"
                ],
                "gradient_norm_median": item["optimization"]["gradient_norm_median"],
                "clip_frequency": item["optimization"]["clip_frequency"],
                "relative_parameter_update": item["optimization"]["parameter_update"][
                    "relative_to_initial_weight_l2"
                ],
                "resume_exact": (
                    item["resume"]["model_state_exact_vs_uninterrupted"]
                    and item["resume"]["trainer_optimizer_state_exact_vs_uninterrupted"]
                ),
            }
            for item in reports
        ],
        "rankings": rankings,
        "primary_small_model_recommendation": {
            "candidate_label": primary["candidate_label"],
            "parameters": primary["parameters"],
            "rule": (
                "Choose the smallest fixed-control MHA candidate whose final held-out loss is "
                f"within {100*tolerance:.1f}% of the best observed final held-out loss."
            ),
            "stage_freeze": False,
        },
        "model10_1m_attention": {
            "mha_parameters": mha["model"]["parameters"],
            "gqa_parameters": gqa["model"]["parameters"],
            "parameter_relative_gap": (
                abs(
                    int(gqa["model"]["parameters"])
                    - int(mha["model"]["parameters"])
                )
                / int(mha["model"]["parameters"])
            ),
            "mha_final_validation_loss": mha_loss,
            "gqa_final_validation_loss": gqa_loss,
            "gqa_relative_validation_loss_delta": relative_quality_delta,
            "mha_full_context_bf16_kv_bytes": mha_cache,
            "gqa_full_context_bf16_kv_bytes": gqa_cache,
            "gqa_kv_cache_fraction_of_mha": gqa_cache / mha_cache,
            "cached_generation_exact_mha": mha["attention_probe"]["cached_stateless_greedy_exact"],
            "cached_generation_exact_gqa": gqa["attention_probe"]["cached_stateless_greedy_exact"],
            "cpu_gqa_execution_path": gqa["attention_probe"]["cpu_attention_execution"],
            "gpu_speed_inference_authorized": False,
            "stage_recommendation": attention_recommendation,
        },
        "truth_boundary": {
            "heldout_generalization_scope": (
                "ONLY_THE_FROZEN_TWO_RECORD_PROJECT_AUTHORED_S0_VALIDATION_SPLIT"
            ),
            "train_loss_used_for_ranking": False,
            "evaluation_tokens_counted_as_optimized": False,
            "exact_fixed_token_budgets": True,
            "broad_language_quality_claim": False,
            "gpu_speed_claim": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute": False,
        },
    }
    aggregate["report_sha256"] = _canonical_hash(aggregate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def validate_candidate(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != CANDIDATE_SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("candidate schema/authority mismatch")
    source_sha = report.get("source_sha")
    if not isinstance(source_sha, str) or not _exact_sha(source_sha):
        raise ValueError("candidate source SHA invalid")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("candidate source SHA mismatch")
    accounting = report["token_accounting"]
    if accounting.get("optimized_tokens_final") != accounting.get(
        "expected_optimized_tokens_final"
    ):
        raise ValueError("final fixed-token accounting drift")
    if accounting.get("evaluation_optimized_tokens_total") != 0:
        raise ValueError("evaluation tokens were counted as optimized")
    if accounting.get("all_budget_boundaries_exact") is not True:
        raise ValueError("one or more token budgets were not exact")
    for point in report["heldout"]["checkpoints"]:
        if int(point["optimized_tokens"]) != int(point["requested_token_budget"]):
            raise ValueError("checkpoint optimized-token budget drift")
        if int(point["evaluation_optimized_tokens"]) != 0:
            raise ValueError("evaluation optimized-token drift")
        expected_compute = 6 * int(report["model"]["parameters"]) * int(point["optimized_tokens"])
        if int(point["compute_proxy"]) != expected_compute:
            raise ValueError("compute proxy drift")
        if not math.isfinite(float(point["validation_loss"])):
            raise ValueError("non-finite held-out loss")
    resume = report["resume"]
    if resume.get("model_state_exact_vs_uninterrupted") is not True:
        raise ValueError("model resume equality failed")
    if resume.get("trainer_optimizer_state_exact_vs_uninterrupted") is not True:
        raise ValueError("trainer resume equality failed")
    truth = report["truth_boundary"]
    for key in (
        "broad_corpus_generalization_claim",
        "gpu_performance_claim",
        "stage_freeze",
        "promotion_authority",
        "paid_compute",
    ):
        if truth.get(key) is not False:
            raise ValueError("candidate truth boundary weakened")
    supplied = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied != _canonical_hash(unsigned):
        raise ValueError("candidate self-hash mismatch")


def validate_aggregate(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("aggregate schema/authority mismatch")
    source = report.get("source", {})
    if source.get("repository") != REPOSITORY:
        raise ValueError("repository identity drift")
    source_sha = source.get("git_sha")
    if not isinstance(source_sha, str) or not _exact_sha(source_sha):
        raise ValueError("aggregate source SHA invalid")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("aggregate source SHA mismatch")
    if len(report.get("candidate_summaries", [])) != 5:
        raise ValueError("expected four scaling candidates plus one GQA comparator")
    truth = report["truth_boundary"]
    if truth.get("train_loss_used_for_ranking") is not False:
        raise ValueError("train loss cannot substitute for held-out ranking")
    if truth.get("evaluation_tokens_counted_as_optimized") is not False:
        raise ValueError("evaluation token accounting overclaim")
    if truth.get("exact_fixed_token_budgets") is not True:
        raise ValueError("fixed token budgets not established")
    for key in (
        "broad_language_quality_claim",
        "gpu_speed_claim",
        "stage_freeze",
        "promotion_authority",
        "paid_compute",
    ):
        if truth.get(key) is not False:
            raise ValueError("aggregate truth boundary weakened")
    supplied = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied != _canonical_hash(unsigned):
        raise ValueError("aggregate self-hash mismatch")


def _run_orchestrator(
    *, repo_root: Path, source_sha: str, output_dir: Path, output_path: Path, torch_threads: int
) -> dict[str, Any]:
    fixed, attention, _ = _load_plans(repo_root)
    labels = [*fixed["candidate_labels"], attention["gqa_candidate_label"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        candidate_dir = output_dir / label
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        command = [
            sys.executable,
            "-m",
            "twelve_six.fixed_token_experiment",
            "run-candidate",
            "--repo-root",
            str(repo_root),
            "--source-sha",
            source_sha,
            "--candidate",
            label,
            "--output-dir",
            str(candidate_dir),
            "--torch-threads",
            str(torch_threads),
        ]
        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"candidate process failed: {label} rc={completed.returncode}")
    report = aggregate_reports(
        repo_root=repo_root,
        source_sha=source_sha,
        input_dir=output_dir,
        output_path=output_path,
    )
    validate_aggregate(report, expected_source_sha=source_sha)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--torch-threads", type=int, default=2)
    candidate = sub.add_parser("run-candidate")
    candidate.add_argument("--repo-root", type=Path, default=Path("."))
    candidate.add_argument("--source-sha", required=True)
    candidate.add_argument("--candidate", required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    candidate.add_argument("--torch-threads", type=int, default=2)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = _run_orchestrator(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_dir=args.output_dir.resolve(),
            output_path=args.report.resolve(),
            torch_threads=args.torch_threads,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "run-candidate":
        report = run_candidate(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            candidate_label=args.candidate,
            output_dir=args.output_dir.resolve(),
            torch_threads=args.torch_threads,
        )
        validate_candidate(report, expected_source_sha=args.source_sha)
        print(
            json.dumps(
                {"candidate": args.candidate, "report_sha256": report["report_sha256"]},
                sort_keys=True,
            )
        )
        return 0
    report = _load_json(args.report)
    validate_aggregate(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
