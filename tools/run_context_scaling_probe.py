#!/usr/bin/env python3
"""Run bounded context-length mechanics probes on the current 12-6 model.

This is engineering evidence, not a long-context quality evaluation. The tool
creates identity-distinct ModelSpecs with larger ``max_seq_len`` while leaving
canonical stage files untouched, then measures one deterministic training step
and model-native KV-cache prefill/decode mechanics.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.context_scaling import context_probe_spec, estimate_context_cost
from twelve_six.model import TwelveSixDecoder, load_stage_config


def _peak_rss_kib() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _grad_norm(model: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().pow(2).sum().cpu()
    return math.sqrt(float(squared))


def run_probe(
    *,
    stage_config: Path,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if sequence_length < 2:
        raise ValueError("context probe sequence_length must be at least two")

    stage = load_stage_config(stage_config)
    spec = context_probe_spec(stage.model, max_seq_len=sequence_length)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    construct_started = time.perf_counter()
    model = TwelveSixDecoder(spec, stage.init).to(device)
    _sync(device)
    construct_seconds = time.perf_counter() - construct_started

    input_ids = torch.randint(
        0,
        spec.vocab_size,
        (1, sequence_length),
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    _sync(device)
    forward_started = time.perf_counter()
    logits = model(input_ids).logits
    _sync(device)
    forward_seconds = time.perf_counter() - forward_started

    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, spec.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    _sync(device)
    backward_started = time.perf_counter()
    loss.backward()
    _sync(device)
    backward_seconds = time.perf_counter() - backward_started
    grad_norm = _grad_norm(model)

    _sync(device)
    update_started = time.perf_counter()
    optimizer.step()
    _sync(device)
    update_seconds = time.perf_counter() - update_started

    # Exercise the incumbent incremental path all the way to the configured
    # context boundary. Prefill S-1 positions, then decode the final token at
    # absolute RoPE position S-1. The returned cache must contain S positions.
    model.eval()
    prefill_ids = input_ids[:, :-1]
    decode_ids = input_ids[:, -1:]
    _sync(device)
    prefill_started = time.perf_counter()
    _, prefix_cache = model.prefill_kv_cache(prefill_ids)
    _sync(device)
    prefill_seconds = time.perf_counter() - prefill_started

    _sync(device)
    decode_started = time.perf_counter()
    _, final_cache = model.decode_one_with_kv_cache(decode_ids, prefix_cache)
    _sync(device)
    decode_seconds = time.perf_counter() - decode_started
    if final_cache.sequence_length != sequence_length:
        raise RuntimeError("incremental decode did not reach the requested context length")

    cache_bytes_actual = sum(
        layer.key.numel() * layer.key.element_size()
        + layer.value.numel() * layer.value.element_size()
        for layer in final_cache.layers
    )

    element_bytes = next(model.parameters()).element_size()
    cost = estimate_context_cost(
        spec,
        sequence_length=sequence_length,
        batch_size=1,
        activation_element_bytes=element_bytes,
        kv_element_bytes=element_bytes,
    )
    if cache_bytes_actual != cost.kv_cache_bytes:
        raise RuntimeError(
            "model-native KV cache byte count disagrees with context cost model: "
            f"actual={cache_bytes_actual} formula={cost.kv_cache_bytes}"
        )

    return {
        "stage": stage.stage,
        "canonical_model_identity_sha256": stage.model.identity_sha256(),
        "probe_model_identity_sha256": spec.identity_sha256(),
        "canonical_max_seq_len": stage.model.max_seq_len,
        "probe_max_seq_len": sequence_length,
        "parameters": spec.parameter_count(),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": grad_norm,
        "construct_seconds": construct_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_update_seconds": update_seconds,
        "kv_prefill_sequence_length": sequence_length - 1,
        "kv_prefill_seconds": prefill_seconds,
        "kv_decode_one_seconds": decode_seconds,
        "kv_final_sequence_length": final_cache.sequence_length,
        "kv_cache_bytes_actual": cache_bytes_actual,
        "context_cost": asdict(cost),
        "peak_rss_kib": _peak_rss_kib(),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def _default_matrix(mode: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if mode == "quick":
        return (
            ("configs/stages/s0_10k.json", (128, 256)),
            ("configs/stages/s1_100k.json", (256, 512)),
            ("configs/stages/s2_1m.json", (512, 1024)),
        )
    return (
        ("configs/stages/s0_10k.json", (128, 256, 512, 1024)),
        ("configs/stages/s1_100k.json", (256, 512, 1024, 2048, 4096)),
        ("configs/stages/s2_1m.json", (512, 1024, 2048, 4096)),
        ("configs/stages/s3_10m.json", (1024, 2048)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=126)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    records: list[dict[str, Any]] = []
    for stage_path, contexts in _default_matrix(args.mode):
        for sequence_length in contexts:
            records.append(
                run_probe(
                    stage_config=Path(stage_path),
                    sequence_length=sequence_length,
                    seed=args.seed,
                    device=device,
                )
            )

    payload = {
        "schema_version": 1,
        "claim_boundary": (
            "mechanical context-scaling evidence only; not trained/evaluated long-context capability"
        ),
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
