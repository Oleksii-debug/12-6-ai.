from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from twelve_six.attention_perf import (
    expanded_kv_tensor_bytes,
    kv_tensor_bytes,
    sdpa_expanded_reference,
    sdpa_native_gqa,
)


@dataclass(frozen=True, slots=True)
class StageCase:
    stage: str
    parameters: int
    query_heads: int
    kv_heads: int
    head_dim: int
    configured_sequence_length: int
    source: str


STAGES = {
    "s3": StageCase("S3", 10_059_840, 8, 8, 40, 1024, "canonical S3 config"),
    "s4": StageCase("S4", 100_384_512, 12, 12, 64, 2048, "PR #37 engineering candidate"),
    "s5": StageCase("S5", 400_598_016, 16, 4, 64, 4096, "PR #37 engineering candidate"),
}

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _backend_context(name: str):
    if name == "auto":
        return nullcontext()
    return sdpa_kernel(BACKENDS[name])


def _make_qkv(
    case: StageCase,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(1234)
    q_source = torch.randn(
        batch_size,
        sequence_length,
        case.query_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    k_source = torch.randn(
        batch_size,
        sequence_length,
        case.kv_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    v_source = torch.randn(
        batch_size,
        sequence_length,
        case.kv_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
    )
    q = q_source.transpose(1, 2).detach().requires_grad_(requires_grad)
    k = k_source.transpose(1, 2).detach().requires_grad_(requires_grad)
    v = v_source.transpose(1, 2).detach().requires_grad_(requires_grad)
    return q, k, v


def _one_step(
    fn: Callable[..., Tensor],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    phase: str,
) -> Tensor:
    output = fn(q, k, v, is_causal=True)
    if phase == "forward-backward":
        output.square().mean().backward()
    return output


def _clear_grads(q: Tensor, k: Tensor, v: Tensor) -> None:
    for tensor in (q, k, v):
        tensor.grad = None


def _time_variant(
    fn: Callable[..., Tensor],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    phase: str,
    backend: str,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | None]:
    device = q.device
    for _ in range(warmup):
        _clear_grads(q, k, v)
        with _backend_context(backend):
            _one_step(fn, q, k, v, phase=phase)
        _sync(device)

    samples: list[float] = []
    peak_bytes: int | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    for _ in range(iterations):
        _clear_grads(q, k, v)
        _sync(device)
        start = time.perf_counter()
        with _backend_context(backend):
            _one_step(fn, q, k, v, phase=phase)
        _sync(device)
        samples.append(time.perf_counter() - start)
    if device.type == "cuda":
        peak_bytes = max(0, torch.cuda.max_memory_allocated(device) - baseline)

    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "iterations": iterations,
        "peak_incremental_allocated_bytes": peak_bytes,
    }


def _parity_error(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    backend: str,
) -> float:
    with torch.no_grad(), _backend_context(backend):
        expected = sdpa_expanded_reference(q, k, v, is_causal=True)
    with torch.no_grad(), _backend_context(backend):
        actual = sdpa_native_gqa(q, k, v, is_causal=True)
    return float((actual.float() - expected.float()).abs().max().item())


def _compile_probe(q: Tensor, k: Tensor, v: Tensor) -> dict[str, object]:
    if not hasattr(torch, "compile"):
        return {"status": "UNAVAILABLE"}

    def call(q_arg: Tensor, k_arg: Tensor, v_arg: Tensor) -> Tensor:
        return sdpa_native_gqa(q_arg, k_arg, v_arg, is_causal=True)

    try:
        compiled = torch.compile(call, backend="eager", fullgraph=True)
        with torch.no_grad():
            expected = call(q, k, v)
            actual = compiled(q, k, v)
        error = float((actual.float() - expected.float()).abs().max().item())
        return {"status": "PASS", "max_abs_error": error}
    except Exception as exc:  # benchmark diagnostic: retain exception type, not user data
        return {"status": "FAIL", "exception_type": type(exc).__name__}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return DTYPES[name]
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def run_case(
    case: StageCase,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    phase: str,
    backend: str,
    warmup: int,
    iterations: int,
    compile_check: bool,
) -> dict[str, object]:
    requires_grad = phase == "forward-backward"
    q, k, v = _make_qkv(
        case,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("benchmark must preserve canonical last-dimension-contiguous layout")

    parity_error = _parity_error(q.detach(), k.detach(), v.detach(), backend=backend)
    expanded = _time_variant(
        sdpa_expanded_reference,
        q,
        k,
        v,
        phase=phase,
        backend=backend,
        warmup=warmup,
        iterations=iterations,
    )
    native = _time_variant(
        sdpa_native_gqa,
        q,
        k,
        v,
        phase=phase,
        backend=backend,
        warmup=warmup,
        iterations=iterations,
    )
    expanded_median = float(expanded["median_seconds"])
    native_median = float(native["median_seconds"])
    native_kv_bytes = kv_tensor_bytes(
        batch_size=batch_size,
        kv_heads=case.kv_heads,
        sequence_length=sequence_length,
        head_dim=case.head_dim,
        dtype=dtype,
    )
    expanded_kv_bytes = expanded_kv_tensor_bytes(
        batch_size=batch_size,
        query_heads=case.query_heads,
        sequence_length=sequence_length,
        head_dim=case.head_dim,
        dtype=dtype,
    )
    full_native_kv_bytes = kv_tensor_bytes(
        batch_size=batch_size,
        kv_heads=case.kv_heads,
        sequence_length=case.configured_sequence_length,
        head_dim=case.head_dim,
        dtype=dtype,
    )
    full_expanded_kv_bytes = expanded_kv_tensor_bytes(
        batch_size=batch_size,
        query_heads=case.query_heads,
        sequence_length=case.configured_sequence_length,
        head_dim=case.head_dim,
        dtype=dtype,
    )
    report: dict[str, object] = {
        "case": asdict(case),
        "benchmark_sequence_length": sequence_length,
        "batch_size": batch_size,
        "phase": phase,
        "qkv_layout": "[batch,heads,sequence,head_dim] from transpose; inner dim contiguous",
        "is_causal": True,
        "attn_mask": None,
        "max_abs_parity_error": parity_error,
        "expanded_reference": expanded,
        "native_gqa": native,
        "native_speedup_x": expanded_median / native_median,
        "benchmark_kv_pair_bytes": {
            "unexpanded": native_kv_bytes,
            "expanded_reference": expanded_kv_bytes,
            "materialization_ratio": expanded_kv_bytes / native_kv_bytes,
        },
        "configured_context_kv_pair_bytes": {
            "unexpanded": full_native_kv_bytes,
            "expanded_reference": full_expanded_kv_bytes,
            "materialization_ratio": full_expanded_kv_bytes / full_native_kv_bytes,
        },
    }
    if compile_check:
        report["compile_fullgraph_eager"] = _compile_probe(q.detach(), k.detach(), v.detach())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark current expanded-K/V SDPA against native PyTorch GQA SDPA."
    )
    parser.add_argument("--stage", choices=["s3", "s4", "s5", "all"], default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument(
        "--phase", choices=["forward", "forward-backward"], default="forward"
    )
    parser.add_argument(
        "--backend", choices=["auto", *BACKENDS], default="auto",
        help="Use auto for production-like dispatch; force a backend only for diagnostics.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override configured stage context. CPU defaults to min(configured, 1024).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--compile-check", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("batch-size/iterations must be positive and warmup non-negative")
    if args.sequence_length is not None and args.sequence_length <= 0:
        raise SystemExit("sequence-length must be positive")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    selected = list(STAGES.values()) if args.stage == "all" else [STAGES[args.stage]]
    reports = []
    for case in selected:
        sequence_length = args.sequence_length
        if sequence_length is None:
            sequence_length = (
                case.configured_sequence_length
                if device.type == "cuda"
                else min(case.configured_sequence_length, 1024)
            )
        reports.append(
            run_case(
                case,
                batch_size=args.batch_size,
                sequence_length=sequence_length,
                device=device,
                dtype=dtype,
                phase=args.phase,
                backend=args.backend,
                warmup=args.warmup,
                iterations=args.iterations,
                compile_check=args.compile_check,
            )
        )

    payload = {
        "schema_version": "12-6.attention-sdpa-benchmark.v1",
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "dtype": str(dtype),
        "backend": args.backend,
        "authority": (
            "GPU_KERNEL_EVIDENCE" if device.type == "cuda" else "CPU_DISPATCH_EVIDENCE_ONLY"
        ),
        "results": reports,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
