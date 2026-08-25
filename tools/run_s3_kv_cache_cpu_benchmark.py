from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch

from twelve_six.integration.s0_runtime import kv_cache_payload_bytes
from twelve_six.model import TwelveSixDecoder, load_stage_config

SCHEMA = "12-6.s3-kv-cache-cpu-benchmark.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("KV-cache benchmark requires a Git checkout") from exc


def _validate_source_sha(value: str) -> None:
    valid = len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError("source SHA must be a full lowercase Git object id")


def _timed(call: Callable[[], float]) -> tuple[float, float]:
    started = time.perf_counter()
    checksum = call()
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("benchmark produced a non-positive elapsed time")
    if not math.isfinite(checksum):
        raise FloatingPointError("benchmark checksum is non-finite")
    return elapsed, checksum


def collect_benchmark(
    repo_root: Path,
    *,
    source_sha: str,
    prompt_length: int = 64,
    decode_steps: int = 8,
    repeats: int = 3,
    threads: int = 1,
    seed: int = 20260825,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    _validate_source_sha(source_sha)
    if _git_head(repo_root) != source_sha:
        raise ValueError("source SHA does not equal checkout HEAD")
    for name, value in {
        "prompt_length": prompt_length,
        "decode_steps": decode_steps,
        "repeats": repeats,
        "threads": threads,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    stage = load_stage_config(repo_root / "configs/stages/s3_10m.json")
    final_sequence_length = prompt_length + decode_steps - 1
    if final_sequence_length > stage.model.max_seq_len:
        raise ValueError("benchmark prompt plus decode steps exceed S3 context")

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init).cpu().eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    prompt = torch.randint(
        0,
        stage.model.vocab_size,
        (1, prompt_length),
        dtype=torch.long,
        generator=generator,
    )
    decode_tokens = torch.randint(
        0,
        stage.model.vocab_size,
        (1, max(0, decode_steps - 1)),
        dtype=torch.long,
        generator=generator,
    )

    @torch.inference_mode()
    def stateless_full_prefix() -> float:
        sequence = prompt
        checksum = 0.0
        for step_index in range(decode_steps):
            logits = model(sequence).logits[:, -1, :]
            checksum += float(logits[0, step_index % logits.shape[-1]])
            if step_index + 1 < decode_steps:
                sequence = torch.cat(
                    (sequence, decode_tokens[:, step_index : step_index + 1]),
                    dim=1,
                )
        return checksum

    @torch.inference_mode()
    def cached_incremental() -> float:
        output, cache = model.prefill_kv_cache(prompt)
        checksum = float(output.logits[0, -1, 0])
        for step_index in range(1, decode_steps):
            token = decode_tokens[:, step_index - 1 : step_index]
            output, cache = model.decode_one_with_kv_cache(token, cache)
            checksum += float(output.logits[0, -1, step_index % output.logits.shape[-1]])
        return checksum

    stateless_full_prefix()
    cached_incremental()

    stateless_seconds: list[float] = []
    cached_seconds: list[float] = []
    stateless_checksums: list[float] = []
    cached_checksums: list[float] = []
    for repeat_index in range(repeats):
        order = (
            (cached_incremental, cached_seconds, cached_checksums),
            (stateless_full_prefix, stateless_seconds, stateless_checksums),
        )
        if repeat_index % 2:
            order = tuple(reversed(order))
        for call, timings, checksums in order:
            elapsed, checksum = _timed(call)
            timings.append(elapsed)
            checksums.append(checksum)

    stateless_median = statistics.median(stateless_seconds)
    cached_median = statistics.median(cached_seconds)
    speedup = stateless_median / cached_median
    if not math.isfinite(speedup) or speedup <= 0.0:
        raise RuntimeError("benchmark produced an invalid speedup ratio")

    stateless_positions = sum(prompt_length + step for step in range(decode_steps))
    cached_positions = prompt_length + decode_steps - 1
    fp32_context_bytes = kv_cache_payload_bytes(
        stage.model,
        stage.model.max_seq_len,
        element_size_bytes=4,
    )
    bf16_context_bytes = kv_cache_payload_bytes(
        stage.model,
        stage.model.max_seq_len,
        element_size_bytes=2,
    )

    return {
        "schema": SCHEMA,
        "authority": "LOCAL_FREE_CPU_BENCHMARK_NOT_CAPACITY_OR_PROMOTION",
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": stage.stage,
        "parameter_count": stage.expected_parameters,
        "model_spec_sha256": stage.model.identity_sha256(),
        "device": "cpu",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": threads,
        "seed": seed,
        "prompt_length": prompt_length,
        "decode_logits_steps": decode_steps,
        "repeats": repeats,
        "median_seconds": {
            "stateless_full_prefix": stateless_median,
            "cached_incremental": cached_median,
        },
        "latency_speedup_ratio": speedup,
        "decoder_input_positions": {
            "stateless_full_prefix": stateless_positions,
            "cached_incremental": cached_positions,
            "work_reduction_ratio": stateless_positions / cached_positions,
        },
        "kv_cache_payload_bytes_at_max_context": {
            "fp32": fp32_context_bytes,
            "bf16_or_fp16": bf16_context_bytes,
        },
        "checksums": {
            "stateless": stateless_checksums,
            "cached": cached_checksums,
        },
        "truth_boundary": {
            "cpu_only": True,
            "gpu_latency": False,
            "serving_throughput": False,
            "capacity_or_sla": False,
            "paid_compute": False,
            "promotion_authority": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark S3 full-prefix vs KV-cache CPU decode.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {args.output}")
    report = collect_benchmark(
        args.repo_root,
        source_sha=args.source_sha,
        prompt_length=args.prompt_length,
        decode_steps=args.decode_steps,
        repeats=args.repeats,
        threads=args.threads,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "s3-kv-cache-cpu-benchmark: COMPLETE "
        f"speedup={report['latency_speedup_ratio']:.3f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
