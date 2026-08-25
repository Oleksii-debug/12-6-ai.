"""Deep LOCAL_FREE evidence run for the RESEARCH41 467,808-parameter control point.

This module deliberately imports RESEARCH41's private control helpers rather than
forking tokenizer, data-order, ModelSpec, InitSpec, or optimizer semantics. It is a
stacked experiment surface: the shared 65K-token prefix must remain bit-for-bit
comparable with the 95K/268K/468K/1.04M controlled sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
    controlled_specs,
    validate_report as validate_scaling_report,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer

SCHEMA = "12-6.scaling-500k-evidence.v1"
AUTHORITY = "LOCAL_FREE_FIXED_CONTROL_500K_EVIDENCE_NOT_PROMOTION_OR_PAID_COMPUTE_AUTHORIZATION"
TARGET_PARAMETERS = 467_808
SHARED_COMPARISON_BUDGET = 65_536
DEFAULT_TOKEN_BUDGETS = (4_096, 16_384, 65_536, 131_072, 262_144)
DEFAULT_SEEDS = (1337, 1338)
GENERATION_PROMPTS = ("The ", "def ")


def _target_spec():
    matches = [spec for spec in controlled_specs() if spec.parameter_count() == TARGET_PARAMETERS]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {TARGET_PARAMETERS}-parameter RESEARCH41 spec")
    return matches[0]


def _bpb(loss_nats: float) -> float:
    """Exact bits/byte because s0-byte-v1 maps every token to exactly one byte."""
    if not math.isfinite(loss_nats):
        raise ValueError("loss must be finite")
    return loss_nats / math.log(2.0)


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(node: Any) -> None:
        if is_dataclass(node) and not isinstance(node, type):
            visit(asdict(node))
            return
        if isinstance(node, torch.Tensor):
            tensor = node.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b":")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b":")
            digest.update(tensor.numpy().tobytes(order="C"))
            digest.update(b";")
            return
        if isinstance(node, dict):
            digest.update(b"{")
            for key in sorted(node, key=lambda item: str(item)):
                visit(str(key))
                visit(node[key])
            digest.update(b"}")
            return
        if isinstance(node, (list, tuple)):
            digest.update(b"[")
            for item in node:
                visit(item)
            digest.update(b"]")
            return
        payload = json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b";")

    visit(value)
    return digest.hexdigest()


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_delta_stats(
    model: torch.nn.Module,
    *,
    previous: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, float]:
    update_sq = 0.0
    previous_sq = 0.0
    movement_sq = 0.0
    initial_sq = 0.0
    max_abs_update = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().cpu().float()
        prior = previous[name].float()
        origin = initial[name].float()
        update = current - prior
        movement = current - origin
        update_sq += float(torch.sum(update * update).item())
        previous_sq += float(torch.sum(prior * prior).item())
        movement_sq += float(torch.sum(movement * movement).item())
        initial_sq += float(torch.sum(origin * origin).item())
        if update.numel():
            max_abs_update = max(max_abs_update, float(torch.max(torch.abs(update)).item()))
    update_l2 = math.sqrt(update_sq)
    previous_l2 = math.sqrt(previous_sq)
    movement_l2 = math.sqrt(movement_sq)
    initial_l2 = math.sqrt(initial_sq)
    return {
        "update_l2": update_l2,
        "weight_l2_before_update": previous_l2,
        "update_to_weight_ratio": update_l2 / previous_l2 if previous_l2 else 0.0,
        "max_abs_update": max_abs_update,
        "movement_l2_from_init": movement_l2,
        "initial_weight_l2": initial_l2,
        "movement_to_initial_weight_ratio": movement_l2 / initial_l2 if initial_l2 else 0.0,
    }


@torch.no_grad()
def _generation_snapshot(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    *,
    prompt: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("generation prompt must encode to at least one token")
    token_ids = list(prompt_ids)
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = token_ids[-model.spec.max_seq_len :]
        input_ids = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits
        token_ids.append(int(torch.argmax(logits[0, -1, :]).item()))
    model.train(was_training)
    generated = token_ids[len(prompt_ids) :]
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_bytes_hex": bytes(generated).hex(),
        "decoded_utf8_replace": tokenizer.decode(generated, errors="replace"),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy_argmax",
    }


def _all_generations(model: TwelveSixDecoder, tokenizer: ByteTokenizer) -> list[dict[str, Any]]:
    return [
        _generation_snapshot(model, tokenizer, prompt=prompt)
        for prompt in GENERATION_PROMPTS
    ]


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024


def _checkpoint_identity(
    *,
    source_sha: str,
    spec,
    init_spec: InitSpec,
    trainer: Trainer,
    dataset_manifest_hash: str,
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
        dataset_manifest_hash=dataset_manifest_hash,
        run_manifest_hash=run_manifest_hash,
        training_config={
            "trainer": asdict(config),
            "init_spec_sha256": init_spec.identity_sha256(),
            "data": {
                "packing_version": PACKING_ID,
                "packing_sha256": _canonical_hash(
                    {
                        "packing_id": PACKING_ID,
                        "batch_size": 4,
                        "sequence_length": 64,
                    }
                ),
            },
        },
        seed=seed,
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
        scheduler=None,
    )


def _checkpoint_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _baseline_comparison(
    baseline: dict[str, Any],
    seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    shared = {
        int(point["parameters"]): point
        for point in baseline["observations"]
        if int(point["requested_token_budget"]) == SHARED_COMPARISON_BUDGET
    }
    required = (95_568, 267_912, TARGET_PARAMETERS, 1_037_696)
    if tuple(sorted(shared)) != tuple(sorted(required)):
        raise RuntimeError(f"baseline shared-budget model set drifted: {sorted(shared)}")

    seed1337 = next(run for run in seed_runs if int(run["seed"]) == 1337)
    seed1337_shared = next(
        point
        for point in seed1337["held_out_curve"]
        if int(point["requested_token_budget"]) == SHARED_COMPARISON_BUDGET
    )
    incumbent_468 = shared[TARGET_PARAMETERS]
    exact_seed1337_match = (
        int(seed1337_shared["optimized_tokens"]) == int(incumbent_468["optimized_tokens"])
        and float(seed1337_shared["validation_loss"]) == float(incumbent_468["validation_loss"])
    )
    if not exact_seed1337_match:
        raise RuntimeError("extended seed-1337 run drifted from the RESEARCH41 468K control point")

    shared_seed_points = []
    for run in seed_runs:
        point = next(
            item
            for item in run["held_out_curve"]
            if int(item["requested_token_budget"]) == SHARED_COMPARISON_BUDGET
        )
        shared_seed_points.append(point)
    mean_468_loss = sum(float(point["validation_loss"]) for point in shared_seed_points) / len(
        shared_seed_points
    )
    mean_468_bpb = _bpb(mean_468_loss)

    p95 = shared[95_568]
    p268 = shared[267_912]
    p468 = shared[TARGET_PARAMETERS]
    p1m = shared[1_037_696]

    def marginal(a: dict[str, Any], b_loss: float, b_parameters: int) -> dict[str, Any]:
        delta_parameters = b_parameters - int(a["parameters"])
        reduction = float(a["validation_loss"]) - b_loss
        return {
            "from_parameters": int(a["parameters"]),
            "to_parameters": b_parameters,
            "extra_parameters": delta_parameters,
            "validation_loss_reduction": reduction,
            "bpb_reduction": _bpb(float(a["validation_loss"])) - _bpb(b_loss),
            "validation_loss_reduction_per_100k_extra_parameters": reduction
            / delta_parameters
            * 100_000.0,
        }

    return {
        "shared_requested_token_budget": SHARED_COMPARISON_BUDGET,
        "shared_optimized_tokens": int(p468["optimized_tokens"]),
        "seed_1337_exact_incumbent_match": exact_seed1337_match,
        "baseline_points": {
            str(parameters): {
                "validation_loss": float(point["validation_loss"]),
                "validation_bpb": _bpb(float(point["validation_loss"])),
                "optimized_tokens": int(point["optimized_tokens"]),
            }
            for parameters, point in sorted(shared.items())
        },
        "target_468k_two_seed_mean": {
            "seeds": [int(run["seed"]) for run in seed_runs],
            "validation_loss": mean_468_loss,
            "validation_bpb": mean_468_bpb,
            "individual_validation_losses": [
                float(point["validation_loss"]) for point in shared_seed_points
            ],
        },
        "marginal_95k_to_268k": marginal(p95, float(p268["validation_loss"]), 267_912),
        "marginal_268k_to_468k_seed1337": marginal(
            p268, float(p468["validation_loss"]), TARGET_PARAMETERS
        ),
        "marginal_468k_to_1m_seed1337": marginal(
            p468, float(p1m["validation_loss"]), 1_037_696
        ),
        "interpretation_boundary": (
            "Only the first 65,536 requested optimized tokens are shared across the four "
            "RESEARCH41 model sizes. The 131K/262K points diagnose the 468K model only."
        ),
    }


def run_500k_evidence(
    *,
    repo_root: Path,
    source_sha: str,
    baseline_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    token_budgets: tuple[int, ...] = DEFAULT_TOKEN_BUDGETS,
    batch_size: int = 4,
    sequence_length: int = 64,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch for 500K evidence run")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("at least two unique non-negative seeds are required")
    if tuple(sorted(set(token_budgets))) != token_budgets:
        raise ValueError("token_budgets must be strictly increasing and unique")
    if SHARED_COMPARISON_BUDGET not in token_budgets:
        raise ValueError("token budgets must retain the 65,536-token shared comparison point")
    if batch_size != 4 or sequence_length != 64:
        raise ValueError("RESEARCH41 batch geometry is fixed at batch_size=4, sequence_length=64")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_scaling_report(baseline, expected_source_sha=source_sha)

    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    spec = _target_spec()
    init_spec = InitSpec()
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
    train_stream = _byte_stream(train_records, tokenizer)
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps = math.ceil(token_budgets[-1] / tokens_per_step)
    dataset_manifest_hash = _file_sha256(manifest_path)
    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "tokenizer_id": BYTE_TOKENIZER_VERSION,
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
        "packing_id": PACKING_ID,
        "train_jsonl_sha256": _file_sha256(train_path),
        "validation_jsonl_sha256": _file_sha256(validation_path),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "tokens_per_optimizer_step": tokens_per_step,
        "token_budgets": list(token_budgets),
        "seeds": list(seeds),
    }
    run_manifest_hash = _canonical_hash(run_manifest)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        trainer_config = _trainer_config(max_steps=max_steps, seed=seed)
        model = TwelveSixDecoder(spec, init_spec)
        trainer = Trainer(model, trainer_config, device="cpu")
        initial_model_state_sha256 = _model_state_sha256(model)
        initial_parameters = _parameter_snapshot(model)
        previous_parameters = {name: value.clone() for name, value in initial_parameters.items()}
        initial_validation_loss, validation_tokens = _validation_loss(
            model, validation_records, tokenizer
        )
        generations: list[dict[str, Any]] = [
            {
                "requested_token_budget": 0,
                "optimized_tokens": 0,
                "snapshots": _all_generations(model, tokenizer),
            }
        ]
        held_out_curve: list[dict[str, Any]] = [
            {
                "requested_token_budget": 0,
                "optimized_tokens": 0,
                "optimizer_steps": 0,
                "validation_loss": initial_validation_loss,
                "validation_bpb": _bpb(initial_validation_loss),
                "validation_tokens": validation_tokens,
            }
        ]
        train_curve: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        next_budget_index = 0
        optimizer_wall_seconds = 0.0
        batch_trace = hashlib.sha256()
        run_started = time.perf_counter()

        for step in range(max_steps):
            batch = _make_batch(
                train_stream,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            batch_trace.update(batch.numpy().tobytes(order="C"))
            step_started = time.perf_counter()
            metrics = trainer.train_microbatch({"input_ids": batch})
            optimizer_wall_seconds += time.perf_counter() - step_started
            delta = _parameter_delta_stats(
                model,
                previous=previous_parameters,
                initial=initial_parameters,
            )
            previous_parameters = _parameter_snapshot(model)
            train_curve.append(
                {
                    "optimizer_step": metrics.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "train_loss": metrics.update_loss,
                    "grad_norm_pre_clip": metrics.grad_norm,
                    "learning_rate": metrics.learning_rate,
                    **delta,
                }
            )

            while (
                next_budget_index < len(token_budgets)
                and trainer.tokens_seen >= token_budgets[next_budget_index]
            ):
                requested_budget = token_budgets[next_budget_index]
                validation_loss, checked_tokens = _validation_loss(
                    model, validation_records, tokenizer
                )
                if checked_tokens != validation_tokens:
                    raise RuntimeError("validation token count drifted")
                held_out_curve.append(
                    {
                        "requested_token_budget": requested_budget,
                        "optimized_tokens": trainer.tokens_seen,
                        "optimizer_steps": trainer.optimizer_step,
                        "validation_loss": validation_loss,
                        "validation_bpb": _bpb(validation_loss),
                        "validation_tokens": checked_tokens,
                    }
                )
                current_generations = _all_generations(model, tokenizer)
                generations.append(
                    {
                        "requested_token_budget": requested_budget,
                        "optimized_tokens": trainer.tokens_seen,
                        "snapshots": current_generations,
                    }
                )
                checkpoint_dir = checkpoint_root / f"seed-{seed}" / f"tokens-{requested_budget}"
                identity = _checkpoint_identity(
                    source_sha=source_sha,
                    spec=spec,
                    init_spec=init_spec,
                    trainer=trainer,
                    dataset_manifest_hash=dataset_manifest_hash,
                    run_manifest_hash=run_manifest_hash,
                    seed=seed,
                )
                save_started = time.perf_counter()
                save_trainer_checkpoint(
                    checkpoint_dir,
                    model=model,
                    trainer=trainer,
                    identity=identity,
                )
                save_seconds = time.perf_counter() - save_started
                checkpoint_records.append(
                    {
                        "requested_token_budget": requested_budget,
                        "optimized_tokens": trainer.tokens_seen,
                        "optimizer_steps": trainer.optimizer_step,
                        "path": checkpoint_dir.relative_to(repo_root).as_posix()
                        if checkpoint_dir.is_relative_to(repo_root)
                        else str(checkpoint_dir),
                        "bytes": _checkpoint_size_bytes(checkpoint_dir),
                        "save_seconds": save_seconds,
                        "manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
                        "model_state_sha256": _model_state_sha256(model),
                        "trainer_state_sha256": _tree_sha256(trainer.state_dict()),
                        "validation_loss": validation_loss,
                        "generations": current_generations,
                    }
                )
                next_budget_index += 1

        if next_budget_index != len(token_budgets):
            raise RuntimeError("training ended before all token checkpoints were observed")
        run_wall_seconds = time.perf_counter() - run_started

        reload_evidence: list[dict[str, Any]] = []
        for record in checkpoint_records:
            checkpoint_dir = checkpoint_root / f"seed-{seed}" / f"tokens-{record['requested_token_budget']}"
            fresh_model = TwelveSixDecoder(spec, init_spec)
            fresh_trainer = Trainer(fresh_model, trainer_config, device="cpu")
            load_started = time.perf_counter()
            load_trainer_checkpoint(
                checkpoint_dir,
                model=fresh_model,
                trainer=fresh_trainer,
                restore_rng=False,
                expected_git_sha=source_sha,
                expected_model_spec_hash=spec.identity_sha256(),
                expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
                expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
                expected_dataset_manifest_hash=dataset_manifest_hash,
                expected_run_manifest_hash=run_manifest_hash,
                expected_seed=seed,
            )
            load_seconds = time.perf_counter() - load_started
            reload_loss, reload_validation_tokens = _validation_loss(
                fresh_model, validation_records, tokenizer
            )
            reload_generations = _all_generations(fresh_model, tokenizer)
            model_equal = _model_state_sha256(fresh_model) == record["model_state_sha256"]
            trainer_equal = _tree_sha256(fresh_trainer.state_dict()) == record["trainer_state_sha256"]
            loss_equal = reload_loss == record["validation_loss"]
            generation_equal = reload_generations == record["generations"]
            counters_equal = (
                fresh_trainer.optimizer_step == record["optimizer_steps"]
                and fresh_trainer.tokens_seen == record["optimized_tokens"]
            )
            if not all((model_equal, trainer_equal, loss_equal, generation_equal, counters_equal)):
                raise RuntimeError(
                    f"checkpoint/reload equality failed for seed={seed}, "
                    f"budget={record['requested_token_budget']}"
                )
            reload_evidence.append(
                {
                    "requested_token_budget": record["requested_token_budget"],
                    "load_seconds": load_seconds,
                    "reload_validation_tokens": reload_validation_tokens,
                    "model_state_equal": model_equal,
                    "trainer_state_equal": trainer_equal,
                    "validation_loss_equal": loss_equal,
                    "generation_equal": generation_equal,
                    "trainer_counters_equal": counters_equal,
                }
            )

        seed_runs.append(
            {
                "seed": seed,
                "initial_model_state_sha256": initial_model_state_sha256,
                "batch_trace_sha256": batch_trace.hexdigest(),
                "initial_validation_loss": initial_validation_loss,
                "initial_validation_bpb": _bpb(initial_validation_loss),
                "validation_tokens": validation_tokens,
                "train_curve": train_curve,
                "held_out_curve": held_out_curve,
                "generation_snapshots": generations,
                "checkpoints": checkpoint_records,
                "checkpoint_reload_equality": reload_evidence,
                "optimizer_wall_seconds": optimizer_wall_seconds,
                "run_wall_seconds": run_wall_seconds,
                "optimized_tokens": trainer.tokens_seen,
                "optimizer_steps": trainer.optimizer_step,
                "optimizer_only_tokens_per_second": trainer.tokens_seen / optimizer_wall_seconds,
                "end_to_end_tokens_per_second": trainer.tokens_seen / run_wall_seconds,
                "peak_rss_bytes": _peak_rss_bytes(),
                "final_model_state_sha256": _model_state_sha256(model),
                "final_parameter_movement": train_curve[-1],
            }
        )

    traces = {run["batch_trace_sha256"] for run in seed_runs}
    if len(traces) != 1:
        raise RuntimeError("multi-seed runs did not consume the exact same controlled token sequence")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha},
        "paid_compute": False,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "init_spec": init_spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "controls": {
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "packing_id": PACKING_ID,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens_per_optimizer_step": tokens_per_step,
            "token_budgets": list(token_budgets),
            "seeds": list(seeds),
            "same_batch_trace_across_seeds": True,
            "batch_trace_sha256": next(iter(traces)),
            "optimizer_by_seed": {
                str(seed): asdict(_trainer_config(max_steps=max_steps, seed=seed)) for seed in seeds
            },
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_hash,
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "train_validation_record_overlap": overlap,
            "unique_train_stream_bytes": len(train_stream),
            "validation_never_used_for_training": True,
            "repeated_fixture": True,
        },
        "run_manifest": run_manifest,
        "run_manifest_sha256": run_manifest_hash,
        "seed_runs": seed_runs,
        "parameter_efficiency": _baseline_comparison(baseline, seed_runs),
        "truth_boundary": {
            "shared_cross_scale_budget_ends_at_requested_tokens": SHARED_COMPARISON_BUDGET,
            "extended_468k_only_budgets": [
                budget for budget in token_budgets if budget > SHARED_COMPARISON_BUDGET
            ],
            "project_authored_tiny_fixture": True,
            "broad_corpus_scaling_claim": False,
            "paid_compute_authority": False,
            "alignment_or_sft": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_500k_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("unexpected 500K evidence schema/authority")
    if report.get("parameters") != TARGET_PARAMETERS:
        raise ValueError("target parameter count drift")
    if expected_source_sha is not None and report.get("source", {}).get("git_sha") != expected_source_sha:
        raise ValueError("source SHA mismatch")
    if report.get("paid_compute") is not False:
        raise ValueError("paid_compute must remain false")
    controls = report.get("controls", {})
    if controls.get("tokenizer_id") != BYTE_TOKENIZER_VERSION:
        raise ValueError("tokenizer identity drift")
    if controls.get("tokenizer_config_sha256") != BYTE_TOKENIZER_HASH:
        raise ValueError("tokenizer config hash drift")
    if controls.get("tokenizer_vocab_sha256") != BYTE_VOCAB_HASH:
        raise ValueError("tokenizer vocab hash drift")
    if controls.get("packing_id") != PACKING_ID:
        raise ValueError("packing identity drift")
    if controls.get("same_batch_trace_across_seeds") is not True:
        raise ValueError("multi-seed batch trace equality not proven")
    if report.get("data", {}).get("train_validation_record_overlap") != []:
        raise ValueError("train/validation overlap detected")
    if report.get("data", {}).get("validation_never_used_for_training") is not True:
        raise ValueError("validation isolation not asserted")
    seed_runs = report.get("seed_runs")
    if not isinstance(seed_runs, list) or len(seed_runs) < 2:
        raise ValueError("at least two seed runs are required")
    for seed_run in seed_runs:
        reloads = seed_run.get("checkpoint_reload_equality", [])
        if len(reloads) < 3:
            raise ValueError("at least three checkpoint/reload proofs are required per seed")
        for proof in reloads:
            for field in (
                "model_state_equal",
                "trainer_state_equal",
                "validation_loss_equal",
                "generation_equal",
                "trainer_counters_equal",
            ):
                if proof.get(field) is not True:
                    raise ValueError(f"checkpoint equality field failed: {field}")
    comparison = report.get("parameter_efficiency", {})
    if comparison.get("seed_1337_exact_incumbent_match") is not True:
        raise ValueError("seed 1337 no longer matches incumbent RESEARCH41 control")
    observed_hash = report.get("report_sha256")
    if not isinstance(observed_hash, str):
        raise ValueError("missing report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if _canonical_hash(unhashed) != observed_hash:
        raise ValueError("report_sha256 mismatch")


def _parse_int_tuple(values: list[str]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--seeds", nargs="+", default=[str(value) for value in DEFAULT_SEEDS])
    run.add_argument(
        "--token-budgets",
        nargs="+",
        default=[str(value) for value in DEFAULT_TOKEN_BUDGETS],
    )
    run.add_argument("--torch-threads", type=int, default=2)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)

    if args.command == "run":
        report = run_500k_evidence(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            baseline_path=args.baseline,
            output_path=args.output,
            checkpoint_root=args.checkpoint_root,
            seeds=_parse_int_tuple(args.seeds),
            token_budgets=_parse_int_tuple(args.token_budgets),
            torch_threads=args.torch_threads,
        )
        validate_500k_report(report, expected_source_sha=args.source_sha)
        print(json.dumps({
            "report_sha256": report["report_sha256"],
            "parameters": report["parameters"],
            "parameter_efficiency": report["parameter_efficiency"],
            "seed_summaries": [
                {
                    "seed": run["seed"],
                    "optimized_tokens": run["optimized_tokens"],
                    "optimizer_only_tokens_per_second": run["optimizer_only_tokens_per_second"],
                    "end_to_end_tokens_per_second": run["end_to_end_tokens_per_second"],
                    "final_validation_loss": run["held_out_curve"][-1]["validation_loss"],
                    "final_validation_bpb": run["held_out_curve"][-1]["validation_bpb"],
                }
                for run in report["seed_runs"]
            ],
        }, indent=2, sort_keys=True))
        return 0

    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_500k_report(report, expected_source_sha=args.expected_source_sha)
    print(json.dumps({"report_sha256": report["report_sha256"], "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
