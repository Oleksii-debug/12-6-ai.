from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder


def _model() -> TwelveSixDecoder:
    torch.manual_seed(226)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
    )
    model = TwelveSixDecoder(spec, InitSpec())
    model.eval()
    return model


def _dynamic_bytes(cache: object) -> int:
    layers = cache.layers
    return sum(
        layer.key.numel() * layer.key.element_size()
        + layer.value.numel() * layer.value.element_size()
        for layer in layers
    )


def _source_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parity_probe(
    model: TwelveSixDecoder,
    prompt: torch.Tensor,
    decode_tokens: list[int],
) -> dict[str, Any]:
    static_cache = allocate_static_kv_cache(model, batch_size=1)
    static_prompt = prefill_static_kv_cache(model, prompt, static_cache)
    dynamic_prompt, dynamic_cache = model.prefill_kv_cache(prompt)
    stateless_prompt = model(prompt)

    max_abs_static_dynamic = float(
        (static_prompt.logits - dynamic_prompt.logits).abs().max().item()
    )
    max_abs_static_stateless = float(
        (static_prompt.logits - stateless_prompt.logits).abs().max().item()
    )
    sequence = prompt
    for token_id in decode_tokens:
        token = torch.tensor([[token_id]], dtype=torch.long)
        static_output = decode_one_with_static_kv_cache(model, token, static_cache)
        dynamic_output, dynamic_cache = model.decode_one_with_kv_cache(token, dynamic_cache)
        sequence = torch.cat((sequence, token), dim=1)
        stateless_output = model(sequence).logits[:, -1:, :]
        max_abs_static_dynamic = max(
            max_abs_static_dynamic,
            float((static_output.logits - dynamic_output.logits).abs().max().item()),
        )
        max_abs_static_stateless = max(
            max_abs_static_stateless,
            float((static_output.logits - stateless_output).abs().max().item()),
        )

    return {
        "static_vs_dynamic_max_abs_logits": max_abs_static_dynamic,
        "static_vs_stateless_max_abs_logits": max_abs_static_stateless,
        "rtol": 1e-6,
        "atol": 1e-6,
        "pass": max_abs_static_dynamic <= 1e-6 and max_abs_static_stateless <= 1e-6,
    }


def _time_static_decode(
    model: TwelveSixDecoder,
    prompt: torch.Tensor,
    decode_tokens: list[int],
    repetitions: int,
) -> tuple[list[float], dict[str, Any]]:
    cache = allocate_static_kv_cache(model, batch_size=1)
    allocated = cache.allocated_bytes
    initial_storage = cache.storage_signature
    times: list[float] = []
    initial_logical = 0
    final_logical = 0
    for repetition in range(repetitions):
        prefill_static_kv_cache(model, prompt, cache)
        if repetition == 0:
            initial_logical = cache.logical_bytes
        start = time.perf_counter_ns()
        for token_id in decode_tokens:
            decode_one_with_static_kv_cache(
                model,
                torch.tensor([[token_id]], dtype=torch.long),
                cache,
            )
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1_000_000_000)
        final_logical = cache.logical_bytes
        if cache.allocated_bytes != allocated:
            raise RuntimeError("static KV backing byte count changed during decode")
        if cache.storage_signature != initial_storage:
            raise RuntimeError("static KV backing storage identity changed during decode")
    return times, {
        "allocated_bytes": allocated,
        "prefill_logical_bytes": initial_logical,
        "final_logical_bytes": final_logical,
        "allocation_growth_bytes": 0,
        "backing_storage_stable": True,
    }


def _time_dynamic_decode(
    model: TwelveSixDecoder,
    prompt: torch.Tensor,
    decode_tokens: list[int],
    repetitions: int,
) -> tuple[list[float], dict[str, Any]]:
    times: list[float] = []
    prefill_bytes = 0
    final_bytes = 0
    for repetition in range(repetitions):
        _, cache = model.prefill_kv_cache(prompt)
        if repetition == 0:
            prefill_bytes = _dynamic_bytes(cache)
        start = time.perf_counter_ns()
        for token_id in decode_tokens:
            _, cache = model.decode_one_with_kv_cache(
                torch.tensor([[token_id]], dtype=torch.long),
                cache,
            )
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1_000_000_000)
        final_bytes = _dynamic_bytes(cache)
    return times, {
        "prefill_bytes": prefill_bytes,
        "final_bytes": final_bytes,
        "allocation_growth_bytes": final_bytes - prefill_bytes,
    }


def _time_stateless_work(
    model: TwelveSixDecoder,
    prompt: torch.Tensor,
    decode_tokens: list[int],
    repetitions: int,
) -> list[float]:
    times: list[float] = []
    for _ in range(repetitions):
        sequence = prompt
        start = time.perf_counter_ns()
        for token_id in decode_tokens:
            sequence = torch.cat(
                (sequence, torch.tensor([[token_id]], dtype=torch.long)),
                dim=1,
            )
            model(sequence)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1_000_000_000)
    return times


def _timing_summary(values: list[float], generated_positions: int) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "median_seconds": median,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "median_decode_positions_per_second": generated_positions / median,
    }


def benchmark(*, repetitions: int) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    torch.set_num_threads(1)
    model = _model()
    prompt_ids = list(range(16, 32))
    decode_tokens = list(range(32, 48))
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    parity = _parity_probe(model, prompt, decode_tokens)
    static_times, static_cache = _time_static_decode(
        model,
        prompt,
        decode_tokens,
        repetitions,
    )
    dynamic_times, dynamic_cache = _time_dynamic_decode(
        model,
        prompt,
        decode_tokens,
        repetitions,
    )
    stateless_times = _time_stateless_work(
        model,
        prompt,
        decode_tokens,
        repetitions,
    )

    decode_count = len(decode_tokens)
    cached_input_positions = len(prompt_ids) + decode_count
    stateless_input_positions = len(prompt_ids) + sum(
        len(prompt_ids) + step for step in range(1, decode_count + 1)
    )
    return {
        "schema": "12-6.perf226-static-kv-benchmark.v1",
        "worker_id": "PERF-226-STATIC-KV-CACHE-V1",
        "source_sha": _source_sha(),
        "device": "cpu",
        "torch_version": torch.__version__,
        "paid_compute": False,
        "paged_attention": False,
        "custom_cuda_kernel": False,
        "geometry": {
            "batch_size": 1,
            "query_heads": model.spec.n_heads,
            "kv_heads": model.spec.n_kv_heads,
            "head_dim": model.spec.head_dim,
            "capacity": model.spec.max_seq_len,
            "prompt_tokens": len(prompt_ids),
            "decode_tokens": decode_count,
        },
        "parity": parity,
        "cache": {
            "static": static_cache,
            "dynamic": dynamic_cache,
        },
        "model_call_work": {
            "static_model_calls": 1 + decode_count,
            "dynamic_model_calls": 1 + decode_count,
            "stateless_model_calls": 1 + decode_count,
            "static_input_positions": cached_input_positions,
            "dynamic_input_positions": cached_input_positions,
            "stateless_input_positions": stateless_input_positions,
            "stateless_to_static_input_position_ratio": (
                stateless_input_positions / cached_input_positions
            ),
        },
        "cpu_decode_time": {
            "repetitions": repetitions,
            "static": _timing_summary(static_times, decode_count),
            "dynamic": _timing_summary(dynamic_times, decode_count),
            "stateless": _timing_summary(stateless_times, decode_count),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(repetitions=args.repetitions)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["parity"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
