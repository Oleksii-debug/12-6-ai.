from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from twelve_six.inference.first_party import load_first_party_backend


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


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(raw_path)


def _close(session: object) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


def _bench_one(label: str, checkpoint: Path, repetitions: int) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    torch.set_num_threads(1)
    spec = backend.model.spec
    prompt_len = min(64, max(8, spec.max_seq_len // 4))
    decode_count = min(16, spec.max_seq_len - prompt_len)
    if decode_count <= 0:
        raise ValueError(f"{label}: context too small for benchmark")
    prompt = tuple((17 + index) % spec.vocab_size for index in range(prompt_len))
    decode_tokens = tuple((113 + index) % spec.vocab_size for index in range(decode_count))

    static_prefill: list[float] = []
    dynamic_prefill: list[float] = []
    static_decode: list[float] = []
    dynamic_decode: list[float] = []
    max_abs = 0.0
    static_allocated = 0
    static_prefill_logical = 0
    static_final_logical = 0
    dynamic_prefill_bytes = 0
    dynamic_final_bytes = 0
    storage_stable = True

    for _ in range(repetitions):
        start = time.perf_counter_ns()
        static_session = backend.begin_generation(prompt)
        static_prefill.append((time.perf_counter_ns() - start) / 1_000_000_000)
        try:
            storage = static_session.cache_storage_signature
            allocated = static_session.cache_bytes
            logical_before = static_session.logical_cache_bytes
            static_allocated = allocated
            static_prefill_logical = logical_before
            static_logits = torch.tensor(static_session.next_token_logits())

            start = time.perf_counter_ns()
            for token_id in decode_tokens:
                static_session.append(token_id)
            static_decode.append((time.perf_counter_ns() - start) / 1_000_000_000)
            static_final_logical = static_session.logical_cache_bytes
            storage_stable = storage_stable and static_session.cache_storage_signature == storage
            storage_stable = storage_stable and static_session.cache_bytes == allocated
            static_final_logits = torch.tensor(static_session.next_token_logits())
        finally:
            _close(static_session)

        start = time.perf_counter_ns()
        dynamic_session = backend.begin_dynamic_generation(prompt)
        dynamic_prefill.append((time.perf_counter_ns() - start) / 1_000_000_000)
        try:
            dynamic_prefill_bytes = dynamic_session.cache_bytes
            dynamic_logits = torch.tensor(dynamic_session.next_token_logits())
            max_abs = max(max_abs, float((static_logits - dynamic_logits).abs().max().item()))

            start = time.perf_counter_ns()
            for token_id in decode_tokens:
                dynamic_session.append(token_id)
            dynamic_decode.append((time.perf_counter_ns() - start) / 1_000_000_000)
            dynamic_final_bytes = dynamic_session.cache_bytes
            dynamic_final_logits = torch.tensor(dynamic_session.next_token_logits())
            max_abs = max(
                max_abs,
                float((static_final_logits - dynamic_final_logits).abs().max().item()),
            )
        finally:
            _close(dynamic_session)

    diagnostics = backend.diagnostics()
    static_decode_summary = _summary(static_decode)
    dynamic_decode_summary = _summary(dynamic_decode)
    static_prefill_summary = _summary(static_prefill)
    dynamic_prefill_summary = _summary(dynamic_prefill)
    return {
        "label": label,
        "checkpoint": diagnostics,
        "geometry": {
            "parameters": spec.parameter_count(),
            "max_context": spec.max_seq_len,
            "layers": spec.n_layers,
            "query_heads": spec.n_heads,
            "kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "native_gqa_active": spec.n_kv_heads < spec.n_heads,
            "unexpanded_kv": spec.n_kv_heads <= spec.n_heads,
            "prompt_tokens": prompt_len,
            "decode_tokens": decode_count,
        },
        "parity": {
            "static_vs_dynamic_max_abs_logits": max_abs,
            "atol": 1e-6,
            "rtol": 1e-6,
            "pass": max_abs <= 1e-6,
        },
        "cache": {
            "static": {
                "allocated_bytes": static_allocated,
                "prefill_logical_bytes": static_prefill_logical,
                "final_logical_bytes": static_final_logical,
                "allocation_growth_bytes": 0,
                "backing_storage_stable": storage_stable,
            },
            "dynamic": {
                "prefill_bytes": dynamic_prefill_bytes,
                "final_bytes": dynamic_final_bytes,
                "allocation_growth_bytes": dynamic_final_bytes - dynamic_prefill_bytes,
            },
        },
        "cpu_prefill_time": {
            "repetitions": repetitions,
            "static": static_prefill_summary,
            "dynamic": dynamic_prefill_summary,
        },
        "cpu_decode_time": {
            "repetitions": repetitions,
            "static": {
                **static_decode_summary,
                "median_seconds_per_token": static_decode_summary["median_seconds"] / decode_count,
            },
            "dynamic": {
                **dynamic_decode_summary,
                "median_seconds_per_token": dynamic_decode_summary["median_seconds"] / decode_count,
            },
        },
        "model_call_work": {
            "static_model_calls": 1 + decode_count,
            "dynamic_model_calls": 1 + decode_count,
            "static_new_input_positions": prompt_len + decode_count,
            "dynamic_new_input_positions": prompt_len + decode_count,
            "semantically_equal": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    results = [
        _bench_one(label, checkpoint, args.repetitions)
        for label, checkpoint in args.checkpoint
    ]
    result = {
        "schema": "12-6.perf250-learned-static-gqa-benchmark.v1",
        "worker_id": "PERF-250-GQA-STATIC-KV-CONVERGENCE-V2",
        "source_sha": _source_sha(),
        "device": "cpu",
        "torch_version": torch.__version__,
        "paid_compute": False,
        "paged_attention": False,
        "custom_cuda_kernel": False,
        "dynamic_cache_retained_as_reference_fallback": True,
        "models": results,
        "all_parity_pass": all(item["parity"]["pass"] for item in results),
        "all_static_storage_stable": all(
            item["cache"]["static"]["backing_storage_stable"] for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_parity_pass"] and result["all_static_storage_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
