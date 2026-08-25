#!/usr/bin/env python3
"""Bounded memory/compute benchmark for SCALE-143 activation checkpointing.

Uses the canonical 12-6 model and PyTorch's maintained checkpoint wrapper. The
benchmark is deliberately single-process; FSDP2/DCP compatibility is covered by
separate integration tests so memory timing is not conflated with collectives.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import torch

from twelve_six.distributed.activation_checkpointing import apply_activation_checkpointing
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.loss import causal_lm_loss


def _current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_kib) * 1024


class _RSSSampler:
    def __init__(self, interval_seconds: float = 0.002) -> None:
        self.interval_seconds = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.peak = _current_rss_bytes()

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, _current_rss_bytes())
                time.sleep(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, _current_rss_bytes())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype(name: str) -> torch.dtype:
    values = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _build_model(stage_path: Path, policy: str, *, device: torch.device, dtype: torch.dtype):
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
    plan = apply_activation_checkpointing(model, policy)  # type: ignore[arg-type]
    return stage, model, plan


def _make_ids(stage, *, batch_size: int, sequence_length: int, seed: int, device):
    if sequence_length > stage.model.max_seq_len:
        raise ValueError("sequence_length exceeds ModelSpec.max_seq_len")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(
        0,
        stage.model.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    return ids.to(device)


def benchmark(
    stage_path: Path,
    policy: str,
    *,
    sequence_length: int,
    batch_size: int,
    measured_steps: int,
    dtype_name: str,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    if measured_steps < 1:
        raise ValueError("measured_steps must be >= 1")
    torch.manual_seed(seed)
    device = _device(device_name)
    dtype = _dtype(dtype_name)
    stage, model, plan = _build_model(stage_path, policy, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Materialize lazy Adam state with the minimum legal context so the measured
    # memory peak is not dominated by first-use optimizer-state allocation.
    warm_ids = _make_ids(
        stage,
        batch_size=batch_size,
        sequence_length=2,
        seed=seed + 1,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    warm_loss = causal_lm_loss(model(warm_ids).logits, warm_ids)
    warm_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del warm_loss, warm_ids
    gc.collect()
    _sync(device)

    ids = _make_ids(
        stage,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 101,
        device=device,
    )
    baseline_rss = _current_rss_bytes()
    rows: list[dict[str, float | int]] = []

    for _ in range(measured_steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with _RSSSampler() as sampler:
            _sync(device)
            start = time.perf_counter()
            output = model(ids)
            _sync(device)
            forward_end = time.perf_counter()
            loss = causal_lm_loss(output.logits, ids)
            loss.backward()
            _sync(device)
            backward_end = time.perf_counter()
            optimizer.step()
            _sync(device)
            step_end = time.perf_counter()

        valid_tokens = batch_size * (sequence_length - 1)
        row: dict[str, float | int] = {
            "loss": float(loss.detach().float().cpu()),
            "forward_seconds": forward_end - start,
            "backward_seconds": backward_end - forward_end,
            "optimizer_seconds": step_end - backward_end,
            "step_seconds": step_end - start,
            "tokens_per_second": valid_tokens / (step_end - start),
            "peak_cpu_rss_bytes": sampler.peak,
        }
        if device.type == "cuda":
            row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            row["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        rows.append(row)
        del output, loss

    result: dict[str, Any] = {
        "schema_version": "12-6.activation-checkpoint-benchmark.v1",
        "stage_config": str(stage_path),
        "stage": stage.stage,
        "model_spec_sha256": stage.model.identity_sha256(),
        "parameter_count": stage.model.parameter_count(),
        "policy": policy,
        "checkpointed_block_indices": list(plan.checkpointed_block_indices),
        "checkpoint_library": plan.library,
        "checkpoint_implementation": plan.implementation,
        "device": str(device),
        "dtype": dtype_name,
        "logical_cpu_count": os.cpu_count(),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "valid_tokens_per_step": batch_size * (sequence_length - 1),
        "measured_steps": measured_steps,
        "baseline_cpu_rss_bytes": baseline_rss,
        "peak_cpu_rss_bytes": max(int(row["peak_cpu_rss_bytes"]) for row in rows),
        "median_forward_seconds": statistics.median(
            float(row["forward_seconds"]) for row in rows
        ),
        "median_backward_seconds": statistics.median(
            float(row["backward_seconds"]) for row in rows
        ),
        "median_optimizer_seconds": statistics.median(
            float(row["optimizer_seconds"]) for row in rows
        ),
        "median_step_seconds": statistics.median(float(row["step_seconds"]) for row in rows),
        "median_tokens_per_second": statistics.median(
            float(row["tokens_per_second"]) for row in rows
        ),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = max(
            int(row["peak_cuda_allocated_bytes"]) for row in rows
        )
        result["peak_cuda_reserved_bytes"] = max(
            int(row["peak_cuda_reserved_bytes"]) for row in rows
        )
    else:
        result["peak_cuda_allocated_bytes"] = None
        result["peak_cuda_reserved_bytes"] = None
    return result


def parity(
    stage_path: Path,
    *,
    sequence_length: int,
    batch_size: int,
    dtype_name: str,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    device = _device(device_name)
    dtype = _dtype(dtype_name)
    torch.manual_seed(seed)
    stage = load_stage_config(stage_path)
    control = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
    state = {name: value.detach().clone() for name, value in control.state_dict().items()}
    del control
    ids = _make_ids(
        stage,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 9,
        device=device,
    )

    outputs: dict[str, Any] = {}
    reference_logits = None
    reference_gradients = None
    for policy in ("none", "every_other_block", "per_block"):
        _, model, plan = _build_model(stage_path, policy, device=device, dtype=dtype)
        model.load_state_dict(state)
        logits = model(ids).logits
        loss = causal_lm_loss(logits, ids)
        loss.backward()
        gradients = {
            _canonical_parameter_name(name): parameter.grad.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        logits_fp32 = logits.detach().float()
        if reference_logits is None:
            reference_logits = logits_fp32.clone()
            reference_gradients = gradients
            logit_delta = 0.0
            gradient_delta = 0.0
        else:
            assert reference_gradients is not None
            logit_delta = float((logits_fp32 - reference_logits).abs().max().cpu())
            gradient_delta = max(
                float((gradients[name] - reference_gradients[name]).abs().max().cpu())
                for name in reference_gradients
            )
        outputs[policy] = {
            "checkpointed_blocks": plan.checkpointed_blocks,
            "loss": float(loss.detach().float().cpu()),
            "max_abs_logit_delta_vs_none": logit_delta,
            "max_abs_gradient_delta_vs_none": gradient_delta,
        }
        del model, logits, loss, gradients

    tolerance = {
        "fp32": {"rtol": 1e-6, "atol": 1e-7},
        "bf16": {"rtol": 2e-2, "atol": 2e-2},
        "fp16": {"rtol": 5e-3, "atol": 5e-3},
    }[dtype_name]
    return {
        "schema_version": "12-6.activation-checkpoint-parity.v1",
        "stage": stage.stage,
        "model_spec_sha256": stage.model.identity_sha256(),
        "parameter_count": stage.model.parameter_count(),
        "device": str(device),
        "dtype": dtype_name,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "tolerance": tolerance,
        "results": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("none", "every_other_block", "per_block"),
        default="none",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.parity:
        result = parity(
            args.stage_config,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
            seed=args.seed,
        )
    else:
        result = benchmark(
            args.stage_config,
            args.policy,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            measured_steps=args.measured_steps,
            dtype_name=args.dtype,
            device_name=args.device,
            seed=args.seed,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
