"""MODEL-119 matched LOCAL_FREE Q/K normalization experiment.

This runner is intentionally research-only. It leaves canonical model semantics,
checkpoint loading, Trainer code, GQA geometry and optimizer defaults untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.data.corpus_v01 import authored_text, norm, sha, split_for
from twelve_six.model import ModelSpec, TwelveSixDecoder, apply_rope
from twelve_six.qk_norm_research import (
    ResearchModelSpec,
    build_research_decoder,
    qk_rms_normalize,
)

BASE_SHA = "b9bc147e0a08181b91798c2515cac7a79c66791c"
DATA25_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
MIXTURE = (
    "uk", "en", "uk", "code", "en", "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk", "en", "uk", "code", "en", "uk",
)
LN2 = math.log(2.0)


def _base_spec(scale: str) -> ModelSpec:
    if scale == "1m":
        return ModelSpec(
            schema_version=1, vocab_size=256, max_seq_len=256,
            d_model=128, n_layers=5, n_heads=8, n_kv_heads=8,
            head_dim=16, d_ff=352, rope_rotary_dim=16,
        )
    if scale == "10m":
        return ModelSpec(
            schema_version=1, vocab_size=256, max_seq_len=256,
            d_model=320, n_layers=6, n_heads=8, n_kv_heads=8,
            head_dim=40, d_ff=1280, rope_rotary_dim=40,
        )
    raise ValueError(scale)


def _spec(scale: str, qk: bool) -> ResearchModelSpec:
    return ResearchModelSpec.from_base(_base_spec(scale), research_qk_norm=qk)


def _record_id(stratum: str, index: int, raw: str) -> str:
    source_id = f"project-authored:{stratum}:corpus-v01"
    digest = sha(f"{source_id}\0{0.1}\0{index}\0{raw}".encode())[:24]
    return f"{source_id}:{digest}"


def _trace(candidates: int = 5000) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, Any]]:
    train = {name: bytearray() for name in ("uk", "en", "code")}
    validation = {name: bytearray() for name in ("uk", "en", "code")}
    for index in range(candidates):
        for stratum in ("uk", "en", "code"):
            raw = authored_text(stratum, index)
            text = norm(raw, stratum == "code")
            rid = _record_id(stratum, index, raw)
            split = split_for(rid, "data25-corpus-v01-20260825", 500)
            destination = validation if split == "validation" else train
            destination[stratum].extend(text.encode("utf-8") + b"\n")
    train_bytes = {name: bytes(value) for name, value in train.items()}
    val_bytes = {name: bytes(value) for name, value in validation.items()}
    identity = {
        "base_corpus_identity_sha256": DATA25_ID,
        "subset_candidates_per_stratum": candidates,
        "packing": "cyclic-contiguous-byte-stream-per-stratum-research-only",
        "train_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in train_bytes.items()
        },
        "validation_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in val_bytes.items()
        },
    }
    return train_bytes, val_bytes, identity


def _batch(streams: dict[str, bytes], step: int, batch_size: int, seq: int) -> torch.Tensor:
    stratum = MIXTURE[step % len(MIXTURE)]
    stream = streams[stratum]
    base = (step * batch_size * seq * 17 + {"uk": 0, "en": 131, "code": 271}[stratum]) % len(stream)
    rows = []
    for batch_index in range(batch_size):
        start = (base + batch_index * seq) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(seq)])
    return torch.tensor(rows, dtype=torch.long)


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + value.numpy().tobytes())
    return digest.hexdigest()


def _grad_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum().item())
    return math.sqrt(squared)


def _layer_grad_norms(model: TwelveSixDecoder) -> list[float]:
    result = []
    for block in model.blocks:
        squared = 0.0
        for parameter in block.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().double().square().sum().item())
        result.append(math.sqrt(squared))
    return result


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    streams: dict[str, bytes],
    *,
    batch_size: int,
    seq: int,
    batches_per_stratum: int,
) -> dict[str, Any]:
    before = _state_hash(model)
    training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    try:
        for source_index, stratum in enumerate(("uk", "en", "code")):
            stream = streams[stratum]
            for batch_index in range(batches_per_stratum):
                base = (batch_index * batch_size * seq * 29 + source_index * 101) % len(stream)
                rows = [
                    [stream[(base + row * seq + offset) % len(stream)] for offset in range(seq)]
                    for row in range(batch_size)
                ]
                ids = torch.tensor(rows, dtype=torch.long)
                logits = model(ids).logits[:, :-1, :]
                targets = ids[:, 1:]
                total_nll += float(
                    F.cross_entropy(
                        logits.reshape(-1, 256), targets.reshape(-1), reduction="sum"
                    ).item()
                )
                total_tokens += targets.numel()
    finally:
        model.train(training)
    after = _state_hash(model)
    if before != after:
        raise RuntimeError("evaluation mutated model state")
    return {
        "bpb": total_nll / LN2 / total_tokens,
        "tokens": total_tokens,
        "non_mutation": True,
    }


@torch.no_grad()
def _attention_stats(model: TwelveSixDecoder, ids: torch.Tensor) -> dict[str, Any]:
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(model.blocks):
        handles.append(
            block.attn.register_forward_pre_hook(
                lambda module, args, index=index: inputs.__setitem__(index, args[0].detach())
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda module, args, output, index=index: outputs.__setitem__(index, output.detach())
            )
        )
    training = model.training
    model.eval()
    try:
        model(ids)
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)

    layers = []
    for index, block in enumerate(model.blocks):
        x = inputs[index]
        attention = block.attn
        batch, seq, _ = x.shape
        q = attention.q_proj(x).view(batch, seq, attention.n_heads, attention.head_dim).transpose(1, 2)
        k = attention.k_proj(x).view(batch, seq, attention.n_kv_heads, attention.head_dim).transpose(1, 2)
        cos, sin = attention.rope.cos_sin(seq, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin, attention.rotary_dim)
        k = apply_rope(k, cos, sin, attention.rotary_dim)
        if model.spec.research_qk_norm:
            q = qk_rms_normalize(q, model.spec.research_qk_norm_eps)
            k = qk_rms_normalize(k, model.spec.research_qk_norm_eps)
        if attention.n_kv_heads != attention.n_heads:
            k = k.repeat_interleave(attention.n_heads // attention.n_kv_heads, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(attention.head_dim)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        valid = scores[..., causal]
        masked = scores.masked_fill(~causal.view(1, 1, seq, seq), float("-inf"))
        probabilities = torch.softmax(masked, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(-1)
        layers.append(
            {
                "layer": index,
                "attention_logit_rms": float(valid.square().mean().sqrt().item()),
                "attention_logit_max_abs": float(valid.abs().max().item()),
                "softmax_entropy": float(entropy.mean().item()),
                "activation_rms": float(outputs[index].float().square().mean().sqrt().item()),
            }
        )
    return {
        "layers": layers,
        "logit_rms_mean": statistics.mean(row["attention_logit_rms"] for row in layers),
        "logit_max_abs": max(row["attention_logit_max_abs"] for row in layers),
        "entropy_mean": statistics.mean(row["softmax_entropy"] for row in layers),
    }


@torch.no_grad()
def _cache_parity(model: TwelveSixDecoder, ids: torch.Tensor) -> dict[str, Any]:
    caches = [{"k": None, "v": None} for _ in model.blocks]
    maximum = 0.0
    for position in range(ids.shape[1]):
        x = model.token_embedding(ids[:, position : position + 1])
        for layer_index, block in enumerate(model.blocks):
            attention = block.attn
            normalized = block.attn_norm(x)
            batch = normalized.shape[0]
            q = attention.q_proj(normalized).view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
            k = attention.k_proj(normalized).view(batch, 1, attention.n_kv_heads, attention.head_dim).transpose(1, 2)
            v = attention.v_proj(normalized).view(batch, 1, attention.n_kv_heads, attention.head_dim).transpose(1, 2)
            cos, sin = attention.rope.cos_sin(position + 1, device=x.device, dtype=q.dtype)
            q = apply_rope(q, cos[-1:], sin[-1:], attention.rotary_dim)
            k = apply_rope(k, cos[-1:], sin[-1:], attention.rotary_dim)
            if model.spec.research_qk_norm:
                q = qk_rms_normalize(q, model.spec.research_qk_norm_eps)
                k = qk_rms_normalize(k, model.spec.research_qk_norm_eps)
            cache = caches[layer_index]
            cache["k"] = k if cache["k"] is None else torch.cat((cache["k"], k), dim=2)
            cache["v"] = v if cache["v"] is None else torch.cat((cache["v"], v), dim=2)
            kk, vv = cache["k"], cache["v"]
            if attention.n_kv_heads != attention.n_heads:
                repeats = attention.n_heads // attention.n_kv_heads
                kk = kk.repeat_interleave(repeats, dim=1)
                vv = vv.repeat_interleave(repeats, dim=1)
            weights = torch.softmax(
                torch.matmul(q, kk.transpose(-2, -1)).float() / math.sqrt(attention.head_dim),
                dim=-1,
            ).to(vv.dtype)
            attended = torch.matmul(weights, vv).transpose(1, 2).contiguous().view(batch, 1, attention.q_dim)
            x = x + attention.out_proj(attended)
            x = x + block.mlp(block.mlp_norm(x))
        cached_logits = model.lm_head(model.final_norm(x))[:, -1, :]
        stateless_logits = model(ids[:, : position + 1]).logits[:, -1, :]
        maximum = max(maximum, float((cached_logits - stateless_logits).abs().max().item()))
    return {"max_abs_error": maximum, "passed": maximum < 2e-5}


def _run_cell(
    spec: ResearchModelSpec,
    *,
    seed: int,
    train_streams: dict[str, bytes],
    validation_streams: dict[str, bytes],
    steps: int,
    batch_size: int,
    seq: int,
    eval_batches: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_research_decoder(spec)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    fixed = _batch(validation_streams, 0, min(batch_size, 2), seq)
    initial_eval = _evaluate(
        model, validation_streams, batch_size=min(batch_size, 2), seq=seq,
        batches_per_stratum=eval_batches,
    )
    initial_attention = _attention_stats(model, fixed)
    losses: list[float] = []
    gradient_norms: list[float] = []
    clipped = 0
    update_ratios = []
    final_layer_gradient_norms = []
    sample_steps = {1, max(1, steps // 2), steps}
    started = time.perf_counter()
    for step in range(steps):
        ids = _batch(train_streams, step, batch_size, seq)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids).logits[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1))
        loss.backward()
        gradient_norm = _grad_norm(model)
        gradient_norms.append(gradient_norm)
        clipped += int(gradient_norm > 1.0)
        final_layer_gradient_norms = _layer_grad_norms(model)
        snapshot = None
        denominator = None
        if step + 1 in sample_steps:
            snapshot = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            denominator = math.sqrt(
                sum(float(value.double().square().sum().item()) for value in snapshot.values())
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if snapshot is not None:
            delta = 0.0
            for name, parameter in model.named_parameters():
                delta += float(
                    (parameter.detach().double() - snapshot[name].double()).square().sum().item()
                )
            update_ratios.append(math.sqrt(delta) / max(denominator, 1e-30))
    elapsed = time.perf_counter() - started
    final_eval = _evaluate(
        model, validation_streams, batch_size=min(batch_size, 2), seq=seq,
        batches_per_stratum=eval_batches,
    )
    final_attention = _attention_stats(model, fixed)
    parity = _cache_parity(model, _batch(validation_streams, 3, 1, min(seq, 16)))
    window = min(16, max(1, steps // 4))
    return {
        "seed": seed,
        "candidate": spec.research_qk_norm,
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "initial_bpb": initial_eval["bpb"],
        "final_bpb": final_eval["bpb"],
        "train_loss_first_mean": statistics.mean(losses[:window]),
        "train_loss_last_mean": statistics.mean(losses[-window:]),
        "gradient_norm_mean": statistics.mean(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "final_layer_gradient_norms": final_layer_gradient_norms,
        "clip_frequency": clipped / steps,
        "update_ratio_mean": statistics.mean(update_ratios),
        "initial_attention": initial_attention,
        "final_attention": final_attention,
        "tokens_per_second": steps * batch_size * (seq - 1) / elapsed,
        "cache_stateless_parity": parity,
        "evaluation_non_mutation": initial_eval["non_mutation"] and final_eval["non_mutation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-1m", type=int, default=96)
    parser.add_argument("--run-10m-probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(5, torch.get_num_threads()))
    train_streams, validation_streams, trace = _trace()
    cells = []
    for seed in (1337, 2027, 4099):
        for qk in (False, True):
            cells.append(
                _run_cell(
                    _spec("1m", qk), seed=seed, train_streams=train_streams,
                    validation_streams=validation_streams, steps=args.steps_1m,
                    batch_size=4, seq=64, eval_batches=8,
                )
            )
    if args.run_10m_probe:
        for qk in (False, True):
            cell = _run_cell(
                _spec("10m", qk), seed=1337, train_streams=train_streams,
                validation_streams=validation_streams, steps=8,
                batch_size=1, seq=32, eval_batches=1,
            )
            cell["short_probe_only"] = True
            cells.append(cell)
    report = {
        "schema": "12-6.model119-qk-norm.runner.v1",
        "authority": "LOCAL_FREE_RESEARCH_NOT_ARCHITECTURE_PROMOTION",
        "source": {"repository": "Oleksii-debug/12-6-ai.", "base_sha": BASE_SHA},
        "trace": trace,
        "optimizer": {
            "name": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8,
            "weight_decay": 0.0, "gradient_clip_norm": 1.0, "precision": "fp32",
        },
        "machine": {
            "python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
        },
        "cells": cells,
        "truth_boundary": {
            "paid_compute": False,
            "external_training_data": False,
            "representative_external_corpus_claim": False,
            "architecture_promotion_authority": False,
        },
    }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
