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

from twelve_six.integration.torch_batching import right_padded_next_token_logits
from twelve_six.model import TwelveSixDecoder, load_stage_config

_SCHEMA = "12-6.batched-raw-base-benchmark.v1"
_LOGIT_TOLERANCE = 1e-4


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


def _synthetic_rows(vocab_size: int, lengths: tuple[int, ...]) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_index, length in enumerate(lengths):
        rows.append(
            [
                1 + ((position * 37 + row_index * 53) % (vocab_size - 1))
                for position in range(length)
            ]
        )
    return rows


def _sequential_logits(
    model: TwelveSixDecoder,
    rows: list[list[int]],
) -> list[list[float]]:
    outputs: list[list[float]] = []
    for row in rows:
        values, _ = right_padded_next_token_logits(model, [row])
        outputs.append(values[0])
    return outputs


def _max_abs_error(reference: list[list[float]], candidate: list[list[float]]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("batch result size mismatch")
    maximum = 0.0
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if len(reference_row) != len(candidate_row):
            raise ValueError("logit vocabulary width mismatch")
        for expected, actual in zip(reference_row, candidate_row, strict=True):
            maximum = max(maximum, abs(expected - actual))
    return maximum


def _median_seconds(callable_obj: Callable[[], object], repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_obj()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def _stage_benchmark(
    repo_root: Path,
    *,
    config_name: str,
    lengths: tuple[int, ...],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    config = load_stage_config(repo_root / "configs" / "stages" / config_name)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(config.model, config.init)
    model.eval()
    rows = _synthetic_rows(config.model.vocab_size, lengths)

    reference = _sequential_logits(model, rows)
    candidate, call_stats = right_padded_next_token_logits(model, rows)
    max_abs_error = _max_abs_error(reference, candidate)
    if max_abs_error > _LOGIT_TOLERANCE:
        raise RuntimeError(
            f"batched logits drifted from independent canonical forwards: {max_abs_error}"
        )

    _sequential_logits(model, rows)
    right_padded_next_token_logits(model, rows)

    sequential_seconds = _median_seconds(lambda: _sequential_logits(model, rows), repeats)
    batched_seconds = _median_seconds(
        lambda: right_padded_next_token_logits(model, rows),
        repeats,
    )
    batch_size = len(rows)
    speedup = sequential_seconds / batched_seconds

    return {
        "stage": config.stage,
        "config": config_name,
        "parameter_count": config.model.parameter_count(),
        "vocab_size": config.model.vocab_size,
        "max_seq_len": config.model.max_seq_len,
        "batch_size": batch_size,
        "sequence_lengths": list(lengths),
        "semantic_parity": {
            "max_abs_logit_error": max_abs_error,
            "tolerance": _LOGIT_TOLERANCE,
        },
        "latency_seconds": {
            "sequential_median": sequential_seconds,
            "batched_median": batched_seconds,
        },
        "requests_per_second": {
            "sequential": batch_size / sequential_seconds,
            "batched": batch_size / batched_seconds,
        },
        "batched_speedup_vs_sequential": speedup,
        "batch_memory_accounting": {
            "logical_input_positions": call_stats.logical_input_positions,
            "padded_input_positions": call_stats.padded_input_positions,
            "right_padding_positions": call_stats.right_padding_positions,
            "input_tensor_bytes": call_stats.input_tensor_bytes,
            "full_output_logits_bytes": call_stats.output_logits_bytes,
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
            lengths=(8, 13, 21, 29),
            repeats=s0_repeats,
            seed=20260825,
        ),
        _stage_benchmark(
            repo_root,
            config_name="s3_10m.json",
            lengths=(16, 23, 31, 40),
            repeats=s3_repeats,
            seed=20260826,
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
        "truth_boundary": {
            "local_free_cpu_measurement": True,
            "gpu_benchmark": False,
            "public_server_throughput_or_sla": False,
            "kv_cache_used": False,
            "paged_attention_used": False,
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
