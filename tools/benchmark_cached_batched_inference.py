from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from twelve_six.inference.batching import (
    BatchGenerationRequest,
    CachedBatchGenerationOutput,
    generate_batch_cached,
)
from twelve_six.inference.contracts import GenerationConfig, GenerationResult
from twelve_six.inference.generation import generate
from twelve_six.integration.torch_batching import S0TorchBatchedInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

_SCHEMA = "12-6.cached-batched-inference-benchmark.v1"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_source_sha(value: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("source SHA must be exactly 40 lowercase hexadecimal characters")
    return value


def _requests(prompt_length: int) -> tuple[BatchGenerationRequest, ...]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    max_new_tokens = (8, 4, 6, 8)
    prompt_bytes = ("a", "b", "c", "d")
    return tuple(
        BatchGenerationRequest(
            prompt=character * prompt_length,
            config=GenerationConfig(max_new_tokens=budget),
        )
        for character, budget in zip(prompt_bytes, max_new_tokens, strict=True)
    )


def _independent_generate(
    backend: S0TorchBatchedInferenceBackend,
    requests: tuple[BatchGenerationRequest, ...],
) -> tuple[GenerationResult, ...]:
    return tuple(generate(backend, request.prompt, request.config) for request in requests)


def _batched_generate(
    backend: S0TorchBatchedInferenceBackend,
    requests: tuple[BatchGenerationRequest, ...],
) -> CachedBatchGenerationOutput:
    return generate_batch_cached(backend, requests, max_batch_size=len(requests))


def _median_seconds(callable_obj: Callable[[], object], repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_obj()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def _independent_concurrent_peak_cache_bytes(
    backend: S0TorchBatchedInferenceBackend,
    results: tuple[GenerationResult, ...],
) -> int:
    total = 0
    for result in results:
        sequence_length = len(result.prompt_token_ids) + max(
            0,
            len(result.generated_token_ids) - 1,
        )
        total += backend.estimate_cache_bytes(sequence_length, batch_size=1)
    return total


def _stage_benchmark(
    repo_root: Path,
    *,
    config_name: str,
    prompt_length: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    config = load_stage_config(repo_root / "configs" / "stages" / config_name)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(config.model, config.init)
    backend = S0TorchBatchedInferenceBackend(model, ByteTokenizer())
    requests = _requests(prompt_length)

    independent = _independent_generate(backend, requests)
    batched = _batched_generate(backend, requests)
    if batched.results != independent:
        raise RuntimeError("cached batched generation diverged from independent requests")
    if backend.active_generation_sessions != 0:
        raise RuntimeError("generation session leaked after cached batching")

    independent_peak_cache_bytes = _independent_concurrent_peak_cache_bytes(
        backend,
        independent,
    )
    batched_peak_cache_bytes = batched.stats.peak_cache_bytes
    if independent_peak_cache_bytes <= 0 or batched_peak_cache_bytes <= 0:
        raise RuntimeError("cache payload measurement must be positive")

    _independent_generate(backend, requests)
    _batched_generate(backend, requests)
    independent_seconds = _median_seconds(
        lambda: _independent_generate(backend, requests),
        repeats,
    )
    batched_seconds = _median_seconds(
        lambda: _batched_generate(backend, requests),
        repeats,
    )

    stats = batched.stats
    if stats.model_batch_calls <= 0 or stats.scheduled_cached_input_positions <= 0:
        raise RuntimeError("cached batching produced invalid work accounting")
    if stats.independent_stateless_input_positions <= 0:
        raise RuntimeError("stateless work baseline must be positive")

    return {
        "stage": config.stage,
        "config": config_name,
        "parameter_count": config.model.parameter_count(),
        "max_seq_len": config.model.max_seq_len,
        "batch_size": len(requests),
        "prompt_length": prompt_length,
        "max_new_tokens_by_request": [request.config.max_new_tokens for request in requests],
        "semantic_parity": {
            "greedy_results_exact": True,
            "independent_request_count": len(independent),
        },
        "latency_seconds": {
            "independent_cached_median": independent_seconds,
            "batched_cached_median": batched_seconds,
        },
        "latency_ratio_independent_over_batched": independent_seconds / batched_seconds,
        "work_accounting": {
            "independent_cached_model_calls": stats.independent_cached_model_calls,
            "batched_cached_model_calls": stats.model_batch_calls,
            "prefill_batch_calls": stats.prefill_batch_calls,
            "decode_batch_calls": stats.decode_batch_calls,
            "model_call_reduction_ratio": (
                stats.independent_cached_model_calls / stats.model_batch_calls
            ),
            "independent_stateless_input_positions": (
                stats.independent_stateless_input_positions
            ),
            "logical_cached_input_positions": stats.logical_cached_input_positions,
            "scheduled_cached_input_positions": stats.scheduled_cached_input_positions,
            "retired_row_decode_positions": stats.retired_row_decode_positions,
            "position_reduction_ratio_vs_independent_stateless": (
                stats.independent_stateless_input_positions
                / stats.scheduled_cached_input_positions
            ),
        },
        "cache_memory_accounting": {
            "batched_peak_logical_kv_payload_bytes": batched_peak_cache_bytes,
            "independent_concurrent_peak_logical_kv_payload_bytes": (
                independent_peak_cache_bytes
            ),
            "fixed_row_payload_ratio_vs_independent_concurrent": (
                batched_peak_cache_bytes / independent_peak_cache_bytes
            ),
            "element_size_bytes": next(model.parameters()).element_size(),
        },
    }


def build_report(
    repo_root: Path,
    *,
    source_sha: str,
    threads: int,
    s0_repeats: int,
    s3_repeats: int,
) -> dict[str, Any]:
    source_sha = _validate_source_sha(source_sha)
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)

    stages = [
        _stage_benchmark(
            repo_root,
            config_name="s0_10k.json",
            prompt_length=32,
            repeats=s0_repeats,
            seed=20260826,
        ),
        _stage_benchmark(
            repo_root,
            config_name="s3_10m.json",
            prompt_length=64,
            repeats=s3_repeats,
            seed=20260827,
        ),
    ]
    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "source_sha": source_sha,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": threads,
        },
        "stages": stages,
        "scheduler_contract": {
            "exact_prompt_length_buckets": True,
            "fixed_cache_rows_until_bucket_drain": True,
            "completed_rows_consume_rng": False,
            "ragged_cache_compaction": False,
            "semantic_padding_in_live_rows": False,
        },
        "truth_boundary": {
            "local_free_cpu_measurement": True,
            "gpu_benchmark": False,
            "kv_cache_used": True,
            "paged_attention_used": False,
            "continuous_batching_claim": False,
            "public_server_throughput_or_sla": False,
            "cache_bytes_are_logical_tensor_payload_not_allocator_peak": True,
            "promotion_authority": False,
        },
    }
    report["report_sha256"] = _sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--s0-repeats", type=int, default=20)
    parser.add_argument("--s3-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        args.repo_root.resolve(),
        source_sha=args.source_sha,
        threads=args.threads,
        s0_repeats=args.s0_repeats,
        s3_repeats=args.s3_repeats,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
