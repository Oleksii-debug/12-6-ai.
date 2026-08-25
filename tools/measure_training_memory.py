#!/usr/bin/env python3
"""Measure live Trainer/AdamW state in isolated LOCAL_FREE CPU processes."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.training import (
    Trainer,
    TrainerConfig,
    measure_training_tensor_memory,
    process_rss_bytes,
    scaler_state_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
SCALES = ("100k", "500k", "1m", "10m")
RSS_FIELDS = (
    "rss_before_optimizer_init",
    "rss_after_optimizer_init",
    "rss_optimizer_init_delta",
    "rss_before_first_update",
    "rss_pre_optimizer_step_after_backward",
    "rss_after_optimizer_step",
    "rss_after_first_update",
    "rss_first_update_delta",
)


def model_for(scale: str):
    if scale == "500k":
        spec = ModelSpec(
            schema_version=1,
            vocab_size=256,
            max_seq_len=256,
            d_model=96,
            n_layers=4,
            n_heads=6,
            n_kv_heads=6,
            head_dim=16,
            d_ff=256,
            rope_rotary_dim=16,
        )
        assert spec.parameter_count() == 467_808
        return TwelveSixDecoder(spec, InitSpec()), 467_808, "research41_467808_control"
    path = {
        "100k": ROOT / "configs/stages/s1_100k.json",
        "1m": ROOT / "configs/stages/s2_1m.json",
        "10m": ROOT / "configs/stages/s3_10m.json",
    }[scale]
    stage = load_stage_config(path)
    return TwelveSixDecoder(stage.model, stage.init), stage.expected_parameters, stage.stage


def measure_one(scale: str, sequence_length: int, precision: str) -> dict:
    torch.manual_seed(1337)
    model, expected, source = model_for(scale)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise RuntimeError(f"{scale}: parameter count {count} != {expected}")

    rss0 = process_rss_bytes()
    trainer = Trainer(model, TrainerConfig(max_steps=1, precision=precision), device="cpu")
    rss1 = process_rss_bytes()
    initialized = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)

    seq_len = min(sequence_length, model.spec.max_seq_len)
    ids = torch.arange(seq_len, dtype=torch.long).view(1, -1) % model.spec.vocab_size
    batch = {"input_ids": ids, "labels": ids.clone()}
    rss2 = process_rss_bytes()

    captured = {}
    original_step = trainer.optimizer.step

    def measured_step(*args, **kwargs):
        captured["pre"] = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
        captured["rss_pre"] = process_rss_bytes()
        result = original_step(*args, **kwargs)
        captured["post"] = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
        captured["rss_post"] = process_rss_bytes()
        return result

    trainer.optimizer.step = measured_step  # type: ignore[method-assign]
    metrics = trainer.train_microbatch(batch)
    rss3 = process_rss_bytes()
    final = measure_training_tensor_memory(model, trainer.optimizer, trainer.scaler)
    pre = asdict(captured["pre"])
    post = asdict(captured["post"])

    return {
        "scale": scale,
        "geometry_source": source,
        "parameters": count,
        "precision": precision,
        "batch_size": 1,
        "sequence_length": seq_len,
        "parameter_tensor_bytes": initialized.parameter_bytes,
        "gradient_bytes_pre_step": pre["gradient_bytes"],
        "adam_moment_bytes_after_first_step": post["adam_moment_bytes"],
        "optimizer_other_tensor_bytes": post["optimizer_other_tensor_bytes"],
        "scaler": scaler_state_metadata(trainer.scaler),
        "optimizer_state_empty_after_init": (
            initialized.adam_moment_bytes == 0
            and initialized.optimizer_other_tensor_bytes == 0
        ),
        "gradients_released_after_update": final.gradient_bytes == 0,
        "rss_before_optimizer_init": rss0,
        "rss_after_optimizer_init": rss1,
        "rss_optimizer_init_delta": rss1 - rss0,
        "rss_before_first_update": rss2,
        "rss_pre_optimizer_step_after_backward": captured["rss_pre"],
        "rss_after_optimizer_step": captured["rss_post"],
        "rss_after_first_update": rss3,
        "rss_first_update_delta": rss3 - rss2,
        "loss": metrics.loss,
        "analytical_fp32": {
            "parameter_tensor_bytes": 4 * count,
            "gradient_bytes": 4 * count,
            "adam_moment_bytes": 8 * count,
            "master_weight_bytes": 0,
        },
    }


def isolated(scale: str, sequence_length: int, precision: str) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-scale",
        scale,
        "--sequence-length",
        str(sequence_length),
        "--precision",
        precision,
    ]
    done = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-scale", choices=SCALES)
    parser.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.worker_scale:
        print(json.dumps(measure_one(args.worker_scale, args.sequence_length, args.precision)))
        return 0
    if args.repeats < 1 or args.sequence_length < 2:
        raise ValueError("repeats must be >= 1 and sequence_length must be >= 2")

    results = {}
    for scale in args.scales:
        runs = [isolated(scale, args.sequence_length, args.precision) for _ in range(args.repeats)]
        first = runs[0]
        rss = {
            field: {
                "median": int(statistics.median(run[field] for run in runs)),
                "min": min(run[field] for run in runs),
                "max": max(run[field] for run in runs),
            }
            for field in RSS_FIELDS
        }
        results[scale] = {
            key: first[key]
            for key in first
            if key not in RSS_FIELDS and key != "loss"
        }
        results[scale]["rss"] = rss
        results[scale]["raw_runs"] = runs

    report = {
        "schema": "12-6.train57-optimizer-memory.v1",
        "authority": "LOCAL_FREE_CPU_MEMORY_EVIDENCE_NOT_PAID_COMPUTE_AUTHORIZATION",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "method": {
            "isolated_process_per_observation": True,
            "repeats": args.repeats,
            "sequence_length": args.sequence_length,
            "batch_size": 1,
            "precision": args.precision,
            "rss_source": "/proc/self/statm",
        },
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
