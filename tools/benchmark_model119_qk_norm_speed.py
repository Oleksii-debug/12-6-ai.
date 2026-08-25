"""Paired single-thread training-speed benchmark for MODEL-119.

Uses the authoritative DATA-25 source-version correction before constructing the
same research trace as the main experiment. Alternating pair order reduces
systematic warm-cache/order bias. This benchmark is LOCAL_FREE CPU evidence.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_model119_qk_norm as experiment
from twelve_six.data.corpus_v01 import sha
from twelve_six.qk_norm_research import build_research_decoder


def _record_id(stratum: str, index: int, raw: str) -> str:
    source_id = f"project-authored:{stratum}:corpus-v01"
    source_version = "0.1.0"
    digest = sha(f"{source_id}\0{source_version}\0{index}\0{raw}".encode())[:24]
    return f"{source_id}:{digest}"


experiment._record_id = _record_id


def benchmark(flag: bool, *, seed: int, streams: dict[str, bytes], warmup: int, steps: int) -> float:
    torch.manual_seed(seed)
    model = build_research_decoder(experiment._spec("1m", flag))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )

    def train_step(step: int) -> None:
        ids = experiment._batch(streams, step, 2, 32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids).logits[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for step in range(warmup):
        train_step(step)
    started = time.perf_counter()
    for step in range(warmup, warmup + steps):
        train_step(step)
    elapsed = time.perf_counter() - started
    return steps * 2 * 31 / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(1)
    streams, _, trace = experiment._trace()
    rows = []
    for repetition in range(args.repetitions):
        seed = 8100 + repetition
        order = (False, True) if repetition % 2 == 0 else (True, False)
        values = {}
        for flag in order:
            values[flag] = benchmark(
                flag, seed=seed, streams=streams, warmup=args.warmup, steps=args.steps
            )
        rows.append(
            {
                "rep": repetition,
                "control_tps": values[False],
                "qk_tps": values[True],
                "ratio": values[True] / values[False],
            }
        )
    ratios = [row["ratio"] for row in rows]
    report = {
        "schema": "12-6.model119-qk-norm.speed.v1",
        "authority": "LOCAL_FREE_RESEARCH_NOT_ARCHITECTURE_PROMOTION",
        "trace": trace,
        "threads": 1,
        "batch_size": 2,
        "sequence_length": 32,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "rows": rows,
        "ratio_mean": statistics.mean(ratios),
        "ratio_median": statistics.median(ratios),
        "slowdown_mean": 1.0 - statistics.mean(ratios),
        "slowdown_median": 1.0 - statistics.median(ratios),
        "material_cost_gate_fraction": 0.05,
        "material_cost_gate_failed": 1.0 - statistics.median(ratios) > 0.05,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
