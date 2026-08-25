"""Controlled attention-geometry experiments for MODEL-35.

This module intentionally reuses the canonical model, Trainer and model-native KV cache.
It is engineering evidence only; timing results are host-specific and are not stage or
capability claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.loss import causal_lm_loss

_SCHEMA = "12-6.attention-geometry-experiment.v1"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_checkout(repo_root: Path, source_sha: str) -> None:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be a lowercase 40-hex Git SHA")
    actual = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != source_sha:
        raise RuntimeError(f"exact checkout mismatch: expected {source_sha}, got {actual}")


def _candidate_paths(repo_root: Path) -> list[tuple[str, Path]]:
    stage_dir = repo_root / "configs" / "stages"
    return [
        ("S2-MHA4", stage_dir / "s2_1m.json"),
        ("S2-GQA2", stage_dir / "s2_1m_gqa2.candidate.json"),
        ("S2-MQA1", stage_dir / "s2_1m_mqa1.candidate.json"),
        ("S3-MHA8", stage_dir / "s3_10m.json"),
        ("S3-GQA4", stage_dir / "s3_10m_gqa4.candidate.json"),
        ("S3-GQA2", stage_dir / "s3_10m_gqa2.candidate.json"),
        ("S3-MQA1", stage_dir / "s3_10m_mqa1.candidate.json"),
    ]


def _make_batch(
    spec: ModelSpec,
    *,
    batch_size: int,
    seq_len: int,
    offset: int,
) -> dict[str, torch.Tensor]:
    if seq_len > spec.max_seq_len:
        raise ValueError("experiment sequence exceeds ModelSpec max_seq_len")
    row = torch.arange(seq_len, dtype=torch.long)
    rows = []
    pattern_vocab = min(spec.vocab_size, 64)
    for batch_index in range(batch_size):
        rows.append((row + offset + 7 * batch_index) % pattern_vocab)
    input_ids = torch.stack(rows, dim=0)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


@torch.no_grad()
def _loss(model: TwelveSixDecoder, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    logits = model(batch["input_ids"]).logits
    return float(causal_lm_loss(logits, batch["labels"]).float().item())


def _cache_bytes(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.key.numel() * layer.key.element_size()
        total += layer.value.numel() * layer.value.element_size()
    return int(total)


def _full_context_cache_bytes(
    spec: ModelSpec,
    *,
    bytes_per_element: int,
    batch_size: int = 1,
) -> int:
    return (
        2
        * spec.n_layers
        * spec.n_kv_heads
        * spec.head_dim
        * spec.max_seq_len
        * bytes_per_element
        * batch_size
    )


@torch.no_grad()
def _generation_probe(
    model: TwelveSixDecoder,
    *,
    prompt_len: int,
    generation_steps: int,
) -> dict[str, Any]:
    model.eval()
    prompt = _make_batch(model.spec, batch_size=1, seq_len=prompt_len, offset=3)["input_ids"]

    start = time.perf_counter()
    cached_out, cache = model.prefill_kv_cache(prompt)
    cached_tokens: list[int] = []
    for index in range(generation_steps):
        token = int(torch.argmax(cached_out.logits[:, -1, :], dim=-1).item())
        cached_tokens.append(token)
        if index + 1 < generation_steps:
            next_token = torch.tensor([[token]], dtype=torch.long)
            cached_out, cache = model.decode_one_with_kv_cache(next_token, cache)
    cached_seconds = time.perf_counter() - start

    generated = prompt.clone()
    stateless_tokens: list[int] = []
    start = time.perf_counter()
    for _ in range(generation_steps):
        out = model(generated)
        token_tensor = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        stateless_tokens.append(int(token_tensor.item()))
        generated = torch.cat((generated, token_tensor), dim=1)
    stateless_seconds = time.perf_counter() - start

    if cached_tokens != stateless_tokens:
        raise RuntimeError("cached and stateless greedy generation diverged")

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": generation_steps,
        "cached_tokens_per_second": generation_steps / cached_seconds,
        "stateless_tokens_per_second": generation_steps / stateless_seconds,
        "cached_seconds": cached_seconds,
        "stateless_seconds": stateless_seconds,
        "cache_sequence_length": cache.sequence_length,
        "actual_cache_bytes": _cache_bytes(cache),
        "decoder_input_positions": {
            "cached": prompt_len + max(generation_steps - 1, 0),
            "stateless_full_prefix": sum(prompt_len + index for index in range(generation_steps)),
        },
        "greedy_tokens_exact": True,
    }


@torch.no_grad()
def _native_gqa_probe(
    model: TwelveSixDecoder,
    *,
    seq_len: int = 32,
    repeats: int = 8,
) -> dict[str, Any]:
    spec = model.spec
    if spec.n_kv_heads == spec.n_heads:
        return {"status": "NOT_APPLICABLE_MHA"}

    model.eval()
    attention = model.blocks[0].attn
    x = torch.randn(1, seq_len, spec.d_model)
    q, k, v = attention._project_qkv(x, position_offset=0)
    repeat_output = attention._attend(q, k, v, is_causal=True)

    try:
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        return {
            "status": "NOT_AVAILABLE",
            "exception_type": type(exc).__name__,
        }

    def native_attend() -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(1, seq_len, spec.q_dim)
        return attention.out_proj(attended)

    native_output = native_attend()
    max_abs_error = float((repeat_output - native_output).abs().max().item())

    for _ in range(2):
        attention._attend(q, k, v, is_causal=True)
        native_attend()

    start = time.perf_counter()
    for _ in range(repeats):
        attention._attend(q, k, v, is_causal=True)
    repeat_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        native_attend()
    native_seconds = time.perf_counter() - start

    return {
        "status": "AVAILABLE",
        "max_abs_output_error": max_abs_error,
        "repeat_expand_median_proxy_ms": 1000.0 * repeat_seconds / repeats,
        "native_gqa_median_proxy_ms": 1000.0 * native_seconds / repeats,
        "repeat_over_native_time_ratio": repeat_seconds / native_seconds,
        "stored_kv_heads": spec.n_kv_heads,
        "temporary_repeat_heads": spec.n_heads,
        "current_path_materializes_repeated_kv": True,
        "timing_scope": "HOST_SPECIFIC_MICROBENCH_NOT_RUNTIME_CLAIM",
    }


def _run_seed(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    seq_len: int,
    generation_steps: int,
) -> dict[str, Any]:
    stage = load_stage_config(config_path)
    spec = stage.model

    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, stage.init)
    train_config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    trainer = Trainer(model, train_config, device="cpu")
    initial_batch = _make_batch(spec, batch_size=2, seq_len=seq_len, offset=0)
    initial_loss = _loss(model, initial_batch)

    losses: list[float] = []
    grad_norms: list[float] = []
    step_seconds: list[float] = []
    for step in range(steps):
        batch = _make_batch(spec, batch_size=2, seq_len=seq_len, offset=step)
        start = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        step_seconds.append(time.perf_counter() - start)
        losses.append(metrics.loss)
        if metrics.grad_norm is not None:
            grad_norms.append(metrics.grad_norm)

    final_batch = _make_batch(spec, batch_size=2, seq_len=seq_len, offset=steps)
    final_loss = _loss(model, final_batch)
    generation = _generation_probe(
        model,
        prompt_len=min(seq_len, 64),
        generation_steps=generation_steps,
    )
    native_gqa = _native_gqa_probe(model)

    return {
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_trace": losses,
        "first_grad_norm": None if not grad_norms else grad_norms[0],
        "max_grad_norm": None if not grad_norms else max(grad_norms),
        "median_step_seconds": statistics.median(step_seconds),
        "generation": generation,
        "native_gqa_probe": native_gqa,
    }


def _aggregate(
    label: str,
    config_path: Path,
    seed_runs: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    stage = load_stage_config(config_path)
    spec = stage.model
    breakdown = spec.parameter_breakdown()
    full_cache = {
        "bf16_bytes": _full_context_cache_bytes(spec, bytes_per_element=2),
        "fp32_bytes": _full_context_cache_bytes(spec, bytes_per_element=4),
    }
    return {
        "label": label,
        "config_path": str(config_path.relative_to(repo_root)),
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "parameter_breakdown": breakdown,
        "attention_geometry": {
            "n_query_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "queries_per_kv_head": spec.n_heads // spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "q_projection_width": spec.q_dim,
            "kv_projection_width": spec.kv_dim,
        },
        "full_context_kv_cache": full_cache,
        "inference_work": {
            "attention_score_heads": spec.n_heads,
            "kv_projection_width": spec.kv_dim,
            "current_repeat_expand_temporary_head_multiplier": spec.n_heads / spec.n_kv_heads,
            "note": (
                "GQA does not reduce query-head score count at fixed Hq; it reduces K/V "
                "projection and cache width. The current Product path repeat-expands K/V "
                "before SDPA."
            ),
        },
        "seed_runs": seed_runs,
        "aggregate": {
            "mean_initial_loss": statistics.fmean(run["initial_loss"] for run in seed_runs),
            "mean_final_loss": statistics.fmean(run["final_loss"] for run in seed_runs),
            "mean_first_grad_norm": statistics.fmean(
                run["first_grad_norm"]
                for run in seed_runs
                if run["first_grad_norm"] is not None
            ),
            "median_step_seconds": statistics.median(
                run["median_step_seconds"] for run in seed_runs
            ),
            "mean_cached_generation_tokens_per_second": statistics.fmean(
                run["generation"]["cached_tokens_per_second"] for run in seed_runs
            ),
            "mean_stateless_generation_tokens_per_second": statistics.fmean(
                run["generation"]["stateless_tokens_per_second"] for run in seed_runs
            ),
        },
    }


def collect(
    *,
    repo_root: Path,
    source_sha: str,
    seeds: list[int],
    s2_steps: int,
    s3_steps: int,
    generation_steps: int,
) -> dict[str, Any]:
    _require_checkout(repo_root, source_sha)
    if not seeds:
        raise ValueError("at least one seed is required")
    if min(s2_steps, s3_steps, generation_steps) <= 0:
        raise ValueError("steps must be positive")

    results = []
    for label, path in _candidate_paths(repo_root):
        is_s2 = label.startswith("S2-")
        steps = s2_steps if is_s2 else s3_steps
        seq_len = 64 if is_s2 else 48
        runs = [
            _run_seed(
                config_path=path,
                seed=seed,
                steps=steps,
                seq_len=seq_len,
                generation_steps=generation_steps,
            )
            for seed in seeds
        ]
        results.append(_aggregate(label, path, runs, repo_root=repo_root))

    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "source_sha": source_sha,
        "torch_version": torch.__version__,
        "device": "cpu",
        "precision": "fp32",
        "seeds": seeds,
        "experiment": {
            "s2_steps": s2_steps,
            "s3_steps": s3_steps,
            "generation_steps": generation_steps,
            "data": "deterministic_structured_synthetic_mechanics_fixture",
            "optimizer": "canonical Trainer AdamW defaults except explicit weight_decay=0.0",
        },
        "results": results,
        "recommendation_boundary": {
            "s0_changed": False,
            "capability_claim": False,
            "quality_claim": False,
            "gpu_performance_claim": False,
            "promotion_authority": False,
            "paid_compute": False,
            "authority": "LOCAL_FREE_CPU_MECHANICS_ONLY",
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--s2-steps", type=int, default=6)
    parser.add_argument("--s3-steps", type=int, default=4)
    parser.add_argument("--generation-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = collect(
        repo_root=args.repo_root.resolve(),
        source_sha=args.source_sha,
        seeds=args.seeds,
        s2_steps=args.s2_steps,
        s3_steps=args.s3_steps,
        generation_steps=args.generation_steps,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
