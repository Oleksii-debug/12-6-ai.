#!/usr/bin/env python3
"""Fail-closed CUDA precision qualification preflight for TRAIN-129.

This tool never provisions compute. It only inspects the current process and a local
checkout, emits a machine-readable report, and exits non-zero unless every launch gate
for a future device-bound precision experiment is satisfied.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

SCHEMA = "12-6.train129-cuda-precision-preflight.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
MILESTONE100_HEAD = "b9bc147e0a08181b91798c2515cac7a79c66791c"
TRAIN15_HEAD = "d6b71a7a18a6ac8cf9ec47fa181264b9df55b7eb"
TRAIN60_HEAD = "30c2f56a86177ce9fa3f25167389c86fa396d9a8"
TRAIN60_PRECISION_BLOB = "63b35f695ab4b7d4bf35a777c884d0233f9cc24e"
TRAIN60_TRAINER_BLOB = "0ec579154521a9f11b2167f9f2611a2a05064c52"


def run(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def git(repo: Path, *args: str) -> str | None:
    code, out, _ = run(["git", *args], cwd=repo)
    return out if code == 0 else None


def git_blob(repo: Path, path: str) -> str | None:
    return git(repo, "hash-object", path) if (repo / path).is_file() else None


def is_ancestor(repo: Path, ancestor: str) -> bool:
    code, _, _ = run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], cwd=repo)
    return code == 0


def nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"path": exe, "rows": [], "query_error": None}
    if exe is None:
        return result
    queries = [
        "index,name,uuid,driver_version,memory.total,pci.bus_id,compute_cap",
        "index,name,uuid,driver_version,memory.total,pci.bus_id",
    ]
    for fields in queries:
        code, out, err = run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if code != 0:
            result["query_error"] = err or out or f"exit={code}"
            continue
        names = fields.split(",")
        rows = []
        for line in out.splitlines():
            values = [v.strip() for v in line.split(",")]
            if len(values) == len(names):
                rows.append(dict(zip(names, values)))
        result["rows"] = rows
        result["query_error"] = None
        return result
    return result


def native_bf16(device_index: int) -> tuple[bool, str]:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if probe is None:
        return False, "torch.cuda.is_bf16_supported unavailable"
    try:
        with torch.cuda.device(device_index):
            try:
                ok = bool(probe(including_emulation=False))
            except TypeError:
                ok = bool(probe())
        return ok, "native CUDA bf16 probe"
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return False, f"bf16 probe error: {type(exc).__name__}: {exc}"


def fp16_scaler_probe(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": False, "passed": False}
    if device.type != "cuda":
        result["reason"] = "no CUDA device"
        return result
    result["attempted"] = True
    try:
        torch.cuda.empty_cache()
        torch.manual_seed(129)
        model = torch.nn.Linear(16, 16).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        x = torch.randn(8, 16, device=device)
        y = torch.randn(8, 16, device=device)
        opt.zero_grad(set_to_none=True)
        before = float(scaler.get_scale())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        finite_grads = all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters()
        )
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(device)
        after = float(scaler.get_scale())
        result.update(
            passed=bool(torch.isfinite(loss).item()) and finite_grads and before > 0 and after > 0,
            loss=float(loss.detach().item()),
            finite_gradients=finite_grads,
            scaler_enabled=bool(scaler.is_enabled()),
            scale_before=before,
            scale_after=after,
        )
    except Exception as exc:  # pragma: no cover - hardware-dependent
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def bf16_kernel_probe(device: torch.device, eligible: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": False, "passed": False}
    if device.type != "cuda":
        result["reason"] = "no CUDA device"
        return result
    if not eligible:
        result["reason"] = "native bf16 not proven"
        return result
    result["attempted"] = True
    try:
        x = torch.randn(32, 32, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = x @ x.T
        torch.cuda.synchronize(device)
        result.update(passed=bool(torch.isfinite(y).all().item()), output_dtype=str(y.dtype))
    except Exception as exc:  # pragma: no cover - hardware-dependent
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--authorization",
        choices=["UNAUTHORIZED", "AUTHORIZED_PREPROVISIONED_FREE"],
        default="UNAUTHORIZED",
        help="Explicit operator attestation. This tool never provisions or purchases compute.",
    )
    ns = ap.parse_args()
    repo = ns.repo_root.resolve()

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "policy": {
            "no_compute_provisioning": True,
            "no_paid_compute": True,
            "single_visible_cuda_required": True,
            "authorization_required": "AUTHORIZED_PREPROVISIONED_FREE",
            "precision_order": ["fp32", "bf16_if_native", "fp16_gradscaler_if_probe_passes"],
            "promotion_rule": "quality_and_numerical_acceptance_before_throughput",
            "cross_precision_bit_identity_required": False,
        },
        "incumbents": {
            "milestone100_learned_head": MILESTONE100_HEAD,
            "train15_single_gpu_head": TRAIN15_HEAD,
            "train60_precision_head": TRAIN60_HEAD,
            "required_train60_precision_blob": TRAIN60_PRECISION_BLOB,
            "required_train60_trainer_blob": TRAIN60_TRAINER_BLOB,
        },
        "acceptance": {
            "all_states_finite": True,
            "final_train_bpb_abs_delta_vs_fp32_max": 0.08,
            "final_validation_bpb_abs_delta_vs_fp32_max": 0.08,
            "max_validation_curve_bpb_abs_delta_vs_fp32_max": 0.12,
            "gradient_norm_median_relative_delta_max": 0.10,
            "gradient_norm_p95_relative_delta_max": 0.25,
            "update_norm_median_relative_delta_max": 0.10,
            "update_norm_p95_relative_delta_max": 0.25,
            "same_precision_reload_logits_max_abs": 1e-6,
            "fp16_scaler_required": True,
            "fp16_nonpositive_scale_allowed": False,
            "bf16_scaler_allowed": False,
            "checkpoint_reload_required": True,
            "post_reload_inference_required": True,
            "peak_allocated_and_reserved_vram_required": True,
            "synchronized_throughput_required": True,
        },
    }

    head = git(repo, "rev-parse", "HEAD") if (repo / ".git").exists() else None
    precision_blob = git_blob(repo, "src/twelve_six/training/precision.py") if head else None
    trainer_blob = git_blob(repo, "src/twelve_six/training/trainer.py") if head else None
    integration = {
        "git_head": head,
        "milestone100_ancestor": bool(head and is_ancestor(repo, MILESTONE100_HEAD)),
        "train15_ancestor": bool(head and is_ancestor(repo, TRAIN15_HEAD)),
        "train60_ancestor": bool(head and is_ancestor(repo, TRAIN60_HEAD)),
        "precision_blob": precision_blob,
        "precision_blob_exact_train60": precision_blob == TRAIN60_PRECISION_BLOB,
        "trainer_blob": trainer_blob,
        "trainer_blob_exact_train60": trainer_blob == TRAIN60_TRAINER_BLOB,
        "single_gpu_runner_present": (repo / "tools/run_single_gpu_pilot.py").is_file(),
        "milestone100_runner_present": (repo / "src/twelve_six/milestone100_first_learned.py").is_file(),
    }
    integration["passed"] = all(
        integration[k]
        for k in (
            "milestone100_ancestor",
            "train15_ancestor",
            "train60_ancestor",
            "precision_blob_exact_train60",
            "trainer_blob_exact_train60",
            "single_gpu_runner_present",
            "milestone100_runner_present",
        )
    )
    report["integration"] = integration

    smi = nvidia_smi()
    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    device: torch.device = torch.device("cuda:0") if cuda_available and count > 0 else torch.device("cpu")
    torch_device: dict[str, Any] | None = None
    bf16_ok = False
    bf16_reason = "no CUDA device"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        bf16_ok, bf16_reason = native_bf16(0)
        torch_device = {
            "index": 0,
            "name": props.name,
            "compute_capability": [int(props.major), int(props.minor)],
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }
    fp16_probe = fp16_scaler_probe(device)
    bf16_probe = bf16_kernel_probe(device, bf16_ok)
    hardware = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices_env": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "cuda_is_available": cuda_available,
        "visible_cuda_devices": count,
        "torch_device": torch_device,
        "nvidia_smi": smi,
        "native_bf16_supported": bf16_ok,
        "native_bf16_reason": bf16_reason,
        "bf16_kernel_probe": bf16_probe,
        "fp16_gradscaler_probe": fp16_probe,
    }
    report["hardware"] = hardware

    exact_identity = (
        device.type == "cuda"
        and count == 1
        and smi["path"] is not None
        and len(smi["rows"]) == 1
        and torch_device is not None
    )
    authorization_ok = ns.authorization == "AUTHORIZED_PREPROVISIONED_FREE"
    report["authorization"] = {
        "value": ns.authorization,
        "passed": authorization_ok,
        "operator_attests_already_attached_free_or_authorized_device": authorization_ok,
    }

    learned_scale = {
        "current_substantial_learned_base_parameters": 95_568,
        "requested_cuda_comparison_scale": "~1M_or_~10M_learned",
        "qualified_learned_scale_checkpoint_present": False,
        "reason": (
            "Current MILESTONE-100 is ~100K learned; existing ~1M/~10M paths are mechanics/"
            "scale candidates, not an already-qualified substantial learned checkpoint."
        ),
    }
    report["learned_scale_gate"] = learned_scale

    gates = {
        "repository_integration": bool(integration["passed"]),
        "explicit_authorization": authorization_ok,
        "cuda_visible": cuda_available and count > 0,
        "exactly_one_visible_cuda": count == 1,
        "exact_gpu_identity": exact_identity,
        "fp32_reference_eligible": cuda_available and count == 1,
        "bf16_eligible": cuda_available and count == 1 and bf16_ok and bool(bf16_probe.get("passed")),
        "fp16_gradscaler_eligible": cuda_available and count == 1 and bool(fp16_probe.get("passed")),
        "learned_1m_or_10m_checkpoint": False,
    }
    report["gates"] = gates

    hard_launch_gates = [
        "repository_integration",
        "explicit_authorization",
        "cuda_visible",
        "exactly_one_visible_cuda",
        "exact_gpu_identity",
        "fp32_reference_eligible",
        "learned_1m_or_10m_checkpoint",
    ]
    passed = all(gates[k] for k in hard_launch_gates)
    report["verdict"] = {
        "status": "READY_FOR_DEVICE_BOUND_COMPARISON" if passed else "NOT_RUN_NO_CUDA_PASS",
        "cuda_pass": False,
        "device_qualified_precision_recommendation": None,
        "hardware_scope": smi["rows"][0] if exact_identity else None,
        "missing_hard_gates": [k for k in hard_launch_gates if not gates[k]],
        "note": "A CUDA PASS can only be issued after the full fp32/bf16/fp16 learned comparison, never by this preflight alone.",
    }

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if ns.output:
        ns.output.parent.mkdir(parents=True, exist_ok=True)
        ns.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
