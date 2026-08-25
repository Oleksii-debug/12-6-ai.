"""Matched-parameter tied-vs-untied embedding/output experiment.

Research-only LOCAL_FREE evidence. Canonical stage configs are never mutated.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.checkpoint.core import CheckpointIdentity, hash_json, save_checkpoint
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.scaling_experiment import (
    BYTE_TOKENIZER_HASH,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
    _byte_stream,
    _file_sha256,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
)

SCHEMA = "12-6.weight-tying-experiment.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_EVIDENCE_NOT_CANONICAL_ARCHITECTURE_CHANGE"


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _global_grad_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt().item())


def _grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().double().norm().cpu().item())


def _grad_cosine(left: torch.nn.Parameter, right: torch.nn.Parameter) -> float | None:
    if left.grad is None or right.grad is None:
        return None
    a = left.grad.detach().double().reshape(-1)
    b = right.grad.detach().double().reshape(-1)
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return None
    return float(torch.dot(a, b).item() / denom)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected weight-tying experiment schema")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("exactly two candidates are required")
    return payload


def validate_matched_candidates(
    tied: ModelSpec,
    untied: ModelSpec,
    *,
    max_relative_parameter_delta: float,
) -> None:
    if not tied.tie_word_embeddings or untied.tie_word_embeddings:
        raise ValueError("candidate A must be tied and candidate B must be untied")
    invariant_fields = (
        "schema_version",
        "vocab_size",
        "max_seq_len",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "activation",
        "norm_kind",
        "norm_placement",
        "norm_eps",
        "position_embedding",
        "rope_theta",
        "rope_rotary_dim",
        "attention_bias",
        "mlp_bias",
        "attention_dropout",
        "final_norm",
        "lm_head_bias",
    )
    for field in invariant_fields:
        if getattr(tied, field) != getattr(untied, field):
            raise ValueError(f"matched experiment invariant drift: {field}")
    if untied.d_ff >= tied.d_ff:
        raise ValueError("untied candidate must rebalance capacity downward through d_ff")
    average = (tied.parameter_count() + untied.parameter_count()) / 2.0
    relative = abs(tied.parameter_count() - untied.parameter_count()) / average
    if relative > max_relative_parameter_delta:
        raise ValueError(
            f"parameter mismatch {relative:.6%} exceeds {max_relative_parameter_delta:.6%}"
        )
    if tied.identity_sha256() == untied.identity_sha256():
        raise ValueError("tied and untied ModelSpec identities must differ")


def _run_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    output_root: Path,
    label: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    manifest_hash: str,
    seed: int,
    token_budget: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    stream = _byte_stream(train_records, tokenizer)
    tokens_per_step = batch_size * (sequence_length - 1)
    if token_budget % tokens_per_step != 0:
        raise ValueError("token_budget must divide exactly by causal loss tokens per step")
    steps = token_budget // tokens_per_step

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    config = _trainer_config(max_steps=steps, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    initial_heldout_loss, heldout_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    losses: list[float] = []
    global_grad_norms: list[float] = []
    embedding_grad_norms: list[float] = []
    lm_head_grad_norms: list[float] = []
    clip_factors: list[float] = []
    first_grad_cosine: float | None = None
    step_seconds: list[float] = []

    model.train()
    for step in range(steps):
        batch = _make_batch(
            stream,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        logits = model(batch).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, spec.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        grad_norm = _global_grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        emb_norm = _grad_norm(model.token_embedding.weight)
        head_norm = _grad_norm(model.lm_head.weight)
        if step == 0:
            first_grad_cosine = _grad_cosine(
                model.token_embedding.weight, model.lm_head.weight
            )
        clip = config.gradient_clip_norm
        factor = 1.0 if clip is None else min(1.0, clip / max(grad_norm, 1e-30))
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach().item()))
        global_grad_norms.append(grad_norm)
        embedding_grad_norms.append(emb_norm)
        lm_head_grad_norms.append(head_norm)
        clip_factors.append(factor)

    final_heldout_loss, final_heldout_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    if heldout_tokens != final_heldout_tokens:
        raise RuntimeError("held-out token count drifted")

    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "label": label,
        "seed": seed,
        "token_budget": token_budget,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "init_spec": init_spec.to_dict(),
        "training_config": asdict(config),
    }
    checkpoint_dir = output_root / f"checkpoint-{label}-seed{seed}"
    identity = CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=manifest_hash,
        run_manifest_hash=hash_json(run_manifest),
        training_config=asdict(config),
        seed=seed,
        precision=config.precision,
        step=steps,
        tokens_seen=token_budget,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler={"name": "constant"},
    )
    manifest = save_checkpoint(
        checkpoint_dir,
        model=model,
        identity=identity,
        optimizer=optimizer,
        trainer_state={"step": steps, "optimized_tokens": token_budget},
    )
    checkpoint_total_bytes = _dir_bytes(checkpoint_dir)
    weights_bytes = int(manifest["files"]["weights.safetensors"]["bytes"])
    manifest_model_hash = str(manifest["identity"]["model_spec_hash"])
    if manifest_model_hash != spec.identity_sha256():
        raise RuntimeError("checkpoint ModelSpec identity drift")
    checkpoint_id = str(manifest["checkpoint_id"])
    shutil.rmtree(checkpoint_dir)

    tail = losses[-min(8, len(losses)) :]
    return {
        "label": label,
        "seed": seed,
        "parameters": spec.parameter_count(),
        "parameter_breakdown": spec.parameter_breakdown(),
        "model_identity_sha256": spec.identity_sha256(),
        "tie_word_embeddings": spec.tie_word_embeddings,
        "embedding_lm_head_parameter_alias": (
            model.token_embedding.weight is model.lm_head.weight
        ),
        "state_dict_keys": sorted(model.state_dict().keys()),
        "optimized_tokens": token_budget,
        "steps": steps,
        "heldout_tokens": heldout_tokens,
        "initial_heldout_loss": initial_heldout_loss,
        "final_heldout_loss": final_heldout_loss,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "training_loss_tail_mean": _mean(tail),
        "pre_clip_global_grad_norm_mean": _mean(global_grad_norms),
        "pre_clip_global_grad_norm_max": max(global_grad_norms),
        "embedding_grad_norm_mean": _mean(embedding_grad_norms),
        "lm_head_grad_norm_mean": _mean(lm_head_grad_norms),
        "first_embedding_lm_head_grad_cosine": first_grad_cosine,
        "clip_fraction": sum(value < 1.0 for value in clip_factors) / len(clip_factors),
        "mean_step_seconds": _mean(step_seconds),
        "seconds_per_optimized_token": sum(step_seconds) / token_budget,
        "checkpoint_total_bytes": checkpoint_total_bytes,
        "checkpoint_weights_bytes": weights_bytes,
        "checkpoint_id": checkpoint_id,
        "checkpoint_model_spec_hash": manifest_model_hash,
    }


def run_weight_tying_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch")
    payload = _load_config(config_path)
    train = payload["training"]
    candidates: dict[str, ModelSpec] = {}
    for item in payload["candidates"]:
        label = str(item["label"])
        spec = ModelSpec.from_dict(item["model"])
        if spec.parameter_count() != int(item["expected_parameters"]):
            raise ValueError(f"{label} parameter count drift")
        if spec.identity_sha256() != str(item["expected_model_identity_sha256"]):
            raise ValueError(f"{label} ModelSpec identity drift")
        candidates[label] = spec
    tied = candidates["A_tied"]
    untied = candidates["B_untied"]
    validate_matched_candidates(
        tied,
        untied,
        max_relative_parameter_delta=float(payload["max_relative_parameter_delta"]),
    )

    init_spec = InitSpec.from_dict(payload["init"])
    if init_spec.identity_sha256() != str(payload["expected_init_identity_sha256"]):
        raise ValueError("InitSpec identity drift")
    seeds = tuple(int(seed) for seed in train["seeds"])
    token_budget = int(train["optimized_token_budget"])
    batch_size = int(train["batch_size"])
    sequence_length = int(train["sequence_length"])
    torch_threads = int(train["torch_threads"])
    if sequence_length > min(tied.max_seq_len, untied.max_seq_len):
        raise ValueError("training sequence exceeds candidate context")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    if {str(item["id"]) for item in train_records} & {
        str(item["id"]) for item in validation_records
    }:
        raise RuntimeError("train/validation overlap")
    manifest_hash = _file_sha256(manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_path.parent / "model16-checkpoints-tmp"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for label in ("A_tied", "B_untied"):
                runs.append(
                    _run_candidate(
                        repo_root=repo_root,
                        source_sha=source_sha,
                        output_root=checkpoint_root,
                        label=label,
                        spec=candidates[label],
                        init_spec=init_spec,
                        train_records=train_records,
                        validation_records=validation_records,
                        manifest_hash=manifest_hash,
                        seed=seed,
                        token_budget=token_budget,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                    )
                )
    finally:
        shutil.rmtree(checkpoint_root, ignore_errors=True)

    aggregates: dict[str, Any] = {}
    for label in ("A_tied", "B_untied"):
        items = [item for item in runs if item["label"] == label]
        aggregates[label] = {
            "parameters": items[0]["parameters"],
            "model_identity_sha256": items[0]["model_identity_sha256"],
            "heldout_loss_mean": _mean([float(item["final_heldout_loss"]) for item in items]),
            "training_loss_tail_mean": _mean(
                [float(item["training_loss_tail_mean"]) for item in items]
            ),
            "gradient_norm_mean": _mean(
                [float(item["pre_clip_global_grad_norm_mean"]) for item in items]
            ),
            "clip_fraction_mean": _mean([float(item["clip_fraction"]) for item in items]),
            "mean_step_seconds": _mean([float(item["mean_step_seconds"]) for item in items]),
            "checkpoint_total_bytes_mean": _mean(
                [float(item["checkpoint_total_bytes"]) for item in items]
            ),
            "checkpoint_weights_bytes_mean": _mean(
                [float(item["checkpoint_weights_bytes"]) for item in items]
            ),
        }

    winner = min(
        ("A_tied", "B_untied"),
        key=lambda label: float(aggregates[label]["heldout_loss_mean"]),
    )
    average_params = (tied.parameter_count() + untied.parameter_count()) / 2.0
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "config_path": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": manifest_hash,
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
        "parameter_match": {
            "A_tied": tied.parameter_count(),
            "B_untied": untied.parameter_count(),
            "absolute_delta": abs(tied.parameter_count() - untied.parameter_count()),
            "relative_delta": abs(tied.parameter_count() - untied.parameter_count())
            / average_params,
        },
        "runs": runs,
        "aggregates": aggregates,
        "recommendation": {
            "winner_on_this_controlled_local_free_fixture": winner,
            "canonical_architecture_changed": False,
            "promotion_authorized": False,
            "note": (
                "Prefer the lower mean held-out loss only as experimental evidence; "
                "the untied candidate is not advantaged by extra total parameters."
            ),
        },
        "truth_boundary": {
            "paid_compute_used": False,
            "canonical_modelspec_changed": False,
            "canonical_stage_configs_changed": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
