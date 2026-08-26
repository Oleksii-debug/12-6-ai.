from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder

WORKER_ID = "PERF-350-20M-CPU-SERVING-PLAN"
SCHEMA = "12-6.perf350.cpu-serving-plan.v1"
MECHANICS_SPEC = ModelSpec(
    schema_version=1,
    vocab_size=256,
    max_seq_len=1024,
    d_model=256,
    n_layers=24,
    n_heads=8,
    n_kv_heads=2,
    head_dim=32,
    d_ff=864,
    rope_rotary_dim=32,
)
INIT_SPEC = InitSpec()
REPRESENTATIVE_PROMPT = 128
REPRESENTATIVE_DECODE = 32
THREAD_TIE_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class Case:
    label: str
    threads: int
    batch_size: int
    prompt_tokens: int
    decode_tokens: int
    cache_mode: str
    repetitions: int = 3
    warmups: int = 1


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": _median(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _ru_maxrss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def _host_fingerprint() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    mem_total = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": affinity,
        "memory_total_bytes": mem_total,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def candidate_threads(host: dict[str, Any]) -> list[int]:
    available = host.get("cpu_affinity_count") or host.get("logical_cpu_count") or 1
    available = int(available)
    candidates = {1}
    for value in (2, 4, 8):
        if value <= available:
            candidates.add(value)
    candidates.add(min(8, available))
    return sorted(candidates)


def select_threads(results: list[dict[str, Any]]) -> int:
    if not results:
        raise ValueError("thread results must be non-empty")
    best = max(float(item["decode"]["aggregate_tokens_per_second"]) for item in results)
    eligible = [
        int(item["case"]["threads"])
        for item in results
        if float(item["decode"]["aggregate_tokens_per_second"])
        >= best * (1.0 - THREAD_TIE_FRACTION)
    ]
    return min(eligible)


def _build_model() -> TwelveSixDecoder:
    torch.manual_seed(350)
    model = TwelveSixDecoder(MECHANICS_SPEC, INIT_SPEC)
    model.eval()
    return model


def _run_once(
    model: TwelveSixDecoder, case: Case
) -> tuple[float, float, float, int, int, int]:
    prompt = torch.arange(
        case.batch_size * case.prompt_tokens,
        dtype=torch.long,
    ).reshape(case.batch_size, case.prompt_tokens) % MECHANICS_SPEC.vocab_size
    next_tokens = torch.arange(case.batch_size, dtype=torch.long).reshape(case.batch_size, 1)
    next_tokens %= MECHANICS_SPEC.vocab_size

    allocation_seconds = 0.0
    if case.cache_mode == "static":
        start = time.perf_counter_ns()
        cache = allocate_static_kv_cache(
            model,
            batch_size=case.batch_size,
            capacity=MECHANICS_SPEC.max_seq_len,
        )
        allocation_seconds = (time.perf_counter_ns() - start) / 1e9
        start = time.perf_counter_ns()
        prefill_static_kv_cache(model, prompt, cache)
        prefill_seconds = (time.perf_counter_ns() - start) / 1e9
        cache_bytes = cache.allocated_bytes
        storage_signature = cache.storage_signature
        logical_prefill_bytes = cache.logical_bytes
        start = time.perf_counter_ns()
        for step in range(case.decode_tokens):
            token = (next_tokens + step) % MECHANICS_SPEC.vocab_size
            decode_one_with_static_kv_cache(model, token, cache)
        decode_seconds = (time.perf_counter_ns() - start) / 1e9
        logical_final_bytes = cache.logical_bytes
        if cache.allocated_bytes != cache_bytes:
            raise RuntimeError("static KV backing allocation changed during decode")
        if cache.storage_signature != storage_signature:
            raise RuntimeError("static KV backing storage identity changed during decode")
        return (
            allocation_seconds,
            prefill_seconds,
            decode_seconds,
            cache_bytes,
            0,
            logical_final_bytes - logical_prefill_bytes,
        )

    if case.cache_mode == "dynamic":
        start = time.perf_counter_ns()
        _, dynamic_cache = model.prefill_kv_cache(prompt)
        prefill_seconds = (time.perf_counter_ns() - start) / 1e9
        prefill_bytes = sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            for layer in dynamic_cache.layers
        )
        start = time.perf_counter_ns()
        for step in range(case.decode_tokens):
            token = (next_tokens + step) % MECHANICS_SPEC.vocab_size
            _, dynamic_cache = model.decode_one_with_kv_cache(token, dynamic_cache)
        decode_seconds = (time.perf_counter_ns() - start) / 1e9
        final_bytes = sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            for layer in dynamic_cache.layers
        )
        growth = final_bytes - prefill_bytes
        return 0.0, prefill_seconds, decode_seconds, final_bytes, growth, growth

    raise ValueError(f"unsupported cache mode: {case.cache_mode}")


def run_case(case: Case) -> dict[str, Any]:
    if case.prompt_tokens + case.decode_tokens > MECHANICS_SPEC.max_seq_len:
        raise ValueError("case exceeds mechanics ModelSpec context")
    torch.set_num_threads(case.threads)
    torch.set_num_interop_threads(1)
    model = _build_model()
    baseline_rss = _ru_maxrss_bytes()

    for _ in range(case.warmups):
        _run_once(model, case)

    allocations: list[float] = []
    prefills: list[float] = []
    decodes: list[float] = []
    cache_bytes = 0
    cache_growth_bytes = 0
    logical_growth_bytes = 0
    for _ in range(case.repetitions):
        allocation, prefill, decode, case_cache_bytes, growth, logical_growth = _run_once(
            model, case
        )
        allocations.append(allocation)
        prefills.append(prefill)
        decodes.append(decode)
        cache_bytes = case_cache_bytes
        cache_growth_bytes = growth
        logical_growth_bytes = logical_growth

    peak_rss = _ru_maxrss_bytes()
    decode_median = _median(decodes)
    prefill_median = _median(prefills)
    generated = case.batch_size * case.decode_tokens
    prompt_positions = case.batch_size * case.prompt_tokens
    return {
        "case": asdict(case),
        "model": {
            "parameters": MECHANICS_SPEC.parameter_count(),
            "model_spec_sha256": MECHANICS_SPEC.identity_sha256(),
        },
        "allocation_seconds": _summary(allocations),
        "prefill": {
            "seconds": _summary(prefills),
            "aggregate_prompt_positions_per_second": prompt_positions / prefill_median,
        },
        "decode": {
            "seconds": _summary(decodes),
            "seconds_per_generated_token_per_sequence": decode_median / case.decode_tokens,
            "aggregate_tokens_per_second": generated / decode_median,
        },
        "cache": {
            "reported_bytes_after_decode": cache_bytes,
            "physical_decode_growth_bytes": cache_growth_bytes,
            "logical_payload_growth_bytes": logical_growth_bytes,
            "static_allocation_stable": case.cache_mode == "static",
        },
        "rss": {
            "baseline_after_model_bytes": baseline_rss,
            "peak_process_bytes": peak_rss,
            "incremental_peak_after_model_bytes": max(0, peak_rss - baseline_rss),
        },
    }


def _invoke_case(case: Case) -> dict[str, Any]:
    payload = json.dumps(asdict(case), separators=(",", ":"))
    completed = subprocess.run(
        [sys.executable, __file__, "--worker-case", payload],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _same_case(
    *,
    label: str,
    threads: int,
    batch_size: int = 1,
    prompt_tokens: int = REPRESENTATIVE_PROMPT,
    decode_tokens: int = REPRESENTATIVE_DECODE,
    cache_mode: str = "static",
) -> Case:
    return Case(
        label=label,
        threads=threads,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        cache_mode=cache_mode,
    )


def build_plan() -> dict[str, Any]:
    host = _host_fingerprint()
    if host["cuda_available"] or host["torch_cuda_version"] is not None:
        raise RuntimeError("PERF-350 is CPU-only and must not execute with a CUDA runtime")

    thread_results = [
        _invoke_case(_same_case(label=f"threads_{threads}", threads=threads))
        for threads in candidate_threads(host)
    ]
    selected_threads = select_threads(thread_results)

    batch_results = [
        _invoke_case(
            _same_case(
                label=f"batch_{batch}",
                threads=selected_threads,
                batch_size=batch,
            )
        )
        for batch in (1, 2, 4)
    ]

    prefill_results = [
        _invoke_case(
            _same_case(
                label=f"prefill_{prompt}",
                threads=selected_threads,
                prompt_tokens=prompt,
                decode_tokens=16,
            )
        )
        for prompt in (32, 128, 256, 512)
    ]

    cache_results = [
        _invoke_case(
            _same_case(
                label=f"cache_{cache_mode}",
                threads=selected_threads,
                prompt_tokens=128,
                decode_tokens=64,
                cache_mode=cache_mode,
            )
        )
        for cache_mode in ("static", "dynamic")
    ]

    generation_results = [
        _invoke_case(
            _same_case(
                label=f"generation_{decode}",
                threads=selected_threads,
                prompt_tokens=128,
                decode_tokens=decode,
            )
        )
        for decode in (16, 64, 128)
    ]

    static_case = next(
        item for item in cache_results if item["case"]["cache_mode"] == "static"
    )
    dynamic_case = next(
        item for item in cache_results if item["case"]["cache_mode"] == "dynamic"
    )

    return {
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "scope": "MECHANICS_ONLY_FUTURE_LEARNED_20M",
        "paid_compute": False,
        "local_free_only": True,
        "hardware_extrapolation_allowed": False,
        "learned_20m_checkpoint_used": False,
        "mechanics_surrogate": {
            "reason": (
                "Primary learned 20M checkpoint is not yet available; benchmark reuses "
                "accepted 10M GQA geometry and doubles depth only for serving mechanics "
                "measurement."
            ),
            "model_spec": MECHANICS_SPEC.to_dict(),
            "model_spec_sha256": MECHANICS_SPEC.identity_sha256(),
            "parameters": MECHANICS_SPEC.parameter_count(),
            "delta_from_20m_fraction": (
                MECHANICS_SPEC.parameter_count() - 20_000_000
            )
            / 20_000_000,
            "init_spec_sha256": INIT_SPEC.identity_sha256(),
        },
        "host": host,
        "measurements": {
            "threads": thread_results,
            "batch": batch_results,
            "prefill": prefill_results,
            "cache": cache_results,
            "generation_length": generation_results,
        },
        "defaults_for_this_exact_host_only": {
            "torch_intraop_threads": selected_threads,
            "torch_interop_threads": 1,
            "interactive_batch_size": 1,
            "representative_prefill_tokens": REPRESENTATIVE_PROMPT,
            "kv_cache": "static",
            "dynamic_kv_role": "reference_fallback",
            "default_max_new_tokens": 64,
            "max_context_tokens": MECHANICS_SPEC.max_seq_len,
            "static_kv_bytes_batch1": static_case["cache"]["reported_bytes_after_decode"],
            "static_kv_peak_rss_batch1_bytes": static_case["rss"]["peak_process_bytes"],
            "dynamic_kv_decode_growth_bytes_128_plus_64": dynamic_case["cache"][
                "physical_decode_growth_bytes"
            ],
        },
        "default_rationale": {
            "threads": (
                "Highest representative batch-1 static decode throughput, with a 5% tie "
                "band resolved toward fewer threads."
            ),
            "batch": (
                "Batch 1 remains the interactive default because this benchmark measures "
                "equal-length coalesced mechanics, not queueing delay or a continuous "
                "scheduler; batch 2/4 are capacity measurements only."
            ),
            "prefill": (
                "128 tokens is a benchmark/default planning point, not an input restriction; "
                "all measured prompt lengths and the ModelSpec context bound remain supported."
            ),
            "kv_cache": (
                "Static KV is the serving default for fixed allocation and zero decode-time "
                "cache growth; dynamic cache remains a parity/reference fallback. Latency is "
                "reported rather than assumed better."
            ),
            "generation_length": (
                "64 new tokens is the bounded middle operational default; 16/64/128 costs "
                "are measured. This is not a language-quality optimum and callers may "
                "override it within context."
            ),
        },
        "not_claimed": [
            "learned 20M quality or behavior",
            "other CPU models or memory sizes",
            "Windows performance",
            "GPU performance",
            "public serving SLA",
            "continuous or ragged batching",
            "a final primary 20M architecture",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.worker_case:
        case = Case(**json.loads(args.worker_case))
        print(json.dumps(run_case(case), sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --worker-case is used")

    result = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
