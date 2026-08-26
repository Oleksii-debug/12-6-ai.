#!/usr/bin/env python3
"""SCALE-205 target-GPU validation for the accepted SCALE-143 checkpoint policy.

This tool never implements recomputation itself. It consumes SCALE-143's
``apply_activation_checkpointing`` and benchmark semantics and fails closed when a
real CUDA device or the exact D08 CUDA runtime identity is absent.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import torch

SCHEMA = "12-6.scale205.activation-checkpoint-gpu.v1"
EXPECTED_TORCH = "2.13.0"
EXPECTED_TORCH_CUDA = "13.0"
HEADROOM_FRACTION = 0.80


def _source_sha() -> str | None:
    value = os.environ.get("GITHUB_SHA") or os.environ.get("SOURCE_SHA")
    return value.lower() if value else None


def _device_probe() -> dict[str, Any]:
    row: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available():
        row.update(
            {
                "device_index": None,
                "device_name": None,
                "compute_capability": None,
                "total_hbm_bytes": None,
                "free_hbm_bytes_preflight": None,
            }
        )
        return row
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    row.update(
        {
            "device_index": 0,
            "device_name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_hbm_bytes": int(total_bytes),
            "free_hbm_bytes_preflight": int(free_bytes),
        }
    )
    return row


def _runtime_identity_ok(probe: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    torch_base = str(probe["torch"]).split("+", 1)[0]
    if torch_base != EXPECTED_TORCH:
        errors.append(f"torch={probe['torch']} expected base {EXPECTED_TORCH}")
    if probe["torch_cuda"] != EXPECTED_TORCH_CUDA:
        errors.append(f"torch_cuda={probe['torch_cuda']} expected {EXPECTED_TORCH_CUDA}")
    return not errors, errors


def _headroom_decision(peak_reserved_bytes: int, usable_hbm_bytes: int) -> dict[str, Any]:
    if usable_hbm_bytes <= 0:
        raise ValueError("usable_hbm_bytes must be positive")
    fraction = peak_reserved_bytes / usable_hbm_bytes
    return {
        "usable_hbm_bytes": int(usable_hbm_bytes),
        "headroom_fraction_limit": HEADROOM_FRACTION,
        "peak_reserved_fraction_of_usable": fraction,
        "rule_selection": "none" if fraction <= HEADROOM_FRACTION else "per_block",
    }


def _oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _pair_parity(
    stage_path: Path, *, sequence_length: int, dtype_name: str, seed: int
) -> dict[str, Any]:
    from twelve_six.distributed.activation_checkpointing import apply_activation_checkpointing
    from twelve_six.model import TwelveSixDecoder, load_stage_config
    from twelve_six.training.loss import causal_lm_loss

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    stage = load_stage_config(stage_path)
    control = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in control.state_dict().items()
    }
    del control
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    ids = torch.randint(
        0,
        stage.model.vocab_size,
        (1, sequence_length),
        generator=generator,
    ).to(device)

    reference_logits: torch.Tensor | None = None
    reference_gradients: dict[str, torch.Tensor] | None = None
    results: dict[str, Any] = {}
    for policy in ("none", "per_block"):
        model = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
        apply_activation_checkpointing(model, policy)
        model.load_state_dict(initial_state)
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
                float(
                    (gradients[name] - reference_gradients[name]).abs().max().cpu()
                )
                for name in reference_gradients
            )
        results[policy] = {
            "loss": float(loss.detach().float().cpu()),
            "max_abs_logit_delta_vs_none": logit_delta,
            "max_abs_gradient_delta_vs_none": gradient_delta,
        }
        del model, logits, loss, gradients
        _cleanup_cuda()

    tolerance = {
        "bf16": {"atol": 2e-2, "rtol": 2e-2},
        "fp32": {"atol": 1e-7, "rtol": 1e-6},
    }[dtype_name]
    maximum = max(
        results["per_block"]["max_abs_logit_delta_vs_none"],
        results["per_block"]["max_abs_gradient_delta_vs_none"],
    )
    return {
        "policies": ["none", "per_block"],
        "sequence_length": sequence_length,
        "dtype": dtype_name,
        "tolerance": tolerance,
        "results": results,
        "passed": maximum <= tolerance["atol"],
    }


def _dcp_roundtrip(
    stage_path: Path, *, sequence_length: int, dtype_name: str, seed: int
) -> dict[str, Any]:
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save

    from twelve_six.distributed.activation_checkpointing import apply_activation_checkpointing
    from twelve_six.model import TwelveSixDecoder, load_stage_config
    from twelve_six.training.loss import causal_lm_loss

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    device = torch.device("cuda:0")
    stage = load_stage_config(stage_path)
    torch.manual_seed(seed + 31)
    torch.cuda.manual_seed_all(seed + 31)
    model = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
    plan = apply_activation_checkpointing(model, "per_block")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    length = min(sequence_length, 128)
    ids = torch.randint(0, stage.model.vocab_size, (1, length), device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = causal_lm_loss(model(ids).logits, ids)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    probe_name, probe_parameter = list(model.named_parameters())[0]
    expected = probe_parameter.detach().clone()
    key_count = len(model.state_dict())
    with tempfile.TemporaryDirectory(prefix="scale205-dcp-") as directory:
        checkpoint_id = str(Path(directory) / "checkpoint")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dcp_save(state, checkpoint_id=checkpoint_id)
        with torch.no_grad():
            probe_parameter.zero_()
        restore_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        dcp_load(restore_state, checkpoint_id=checkpoint_id)
        restored = torch.equal(probe_parameter.detach(), expected)
        files = sorted(
            str(path.relative_to(directory))
            for path in Path(directory).rglob("*")
            if path.is_file()
        )

    del model, optimizer, loss, ids, expected
    _cleanup_cuda()
    return {
        "policy": "per_block",
        "checkpointed_blocks": plan.checkpointed_blocks,
        "model_state_key_count": key_count,
        "probe_parameter": _canonical_parameter_name(probe_name),
        "probe_parameter_restored_exactly": restored,
        "dcp_files": files,
        "passed": bool(restored and files),
    }


def _run_benchmark(
    stage_path: Path,
    policy: str,
    *,
    sequence_length: int,
    measured_steps: int,
    dtype_name: str,
    seed: int,
) -> dict[str, Any]:
    from tools.benchmark_activation_checkpointing import benchmark

    return benchmark(
        stage_path,
        policy,
        sequence_length=sequence_length,
        batch_size=1,
        measured_steps=measured_steps,
        dtype_name=dtype_name,
        device_name="cuda:0",
        seed=seed,
    )


def _matched_comparison(
    stage_path: Path,
    *,
    contexts: list[int],
    measured_steps: int,
    dtype_name: str,
    seed: int,
    usable_hbm_bytes: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for context in contexts:
        pair: dict[str, Any] = {"sequence_length": context, "policies": {}}
        common = True
        for policy in ("none", "per_block"):
            try:
                measurement = _run_benchmark(
                    stage_path,
                    policy,
                    sequence_length=context,
                    measured_steps=measured_steps,
                    dtype_name=dtype_name,
                    seed=seed,
                )
                pair["policies"][policy] = {
                    "status": "PASS",
                    "measurement": measurement,
                }
            except BaseException as exc:
                if not _oom(exc):
                    raise
                pair["policies"][policy] = {
                    "status": "OOM",
                    "error_type": type(exc).__name__,
                }
                common = False
            finally:
                _cleanup_cuda()
        attempts.append(pair)
        if not common:
            continue

        none = pair["policies"]["none"]["measurement"]
        per_block = pair["policies"]["per_block"]["measurement"]
        none_reserved = int(none["peak_cuda_reserved_bytes"])
        checkpoint_reserved = int(per_block["peak_cuda_reserved_bytes"])
        pair["headroom_rule"] = _headroom_decision(
            none_reserved, usable_hbm_bytes
        )
        pair["memory_effect"] = {
            "peak_reserved_bytes_saved": none_reserved - checkpoint_reserved,
            "peak_reserved_fraction_saved": (
                (none_reserved - checkpoint_reserved) / none_reserved
                if none_reserved
                else 0.0
            ),
        }
        pair["performance_effect"] = {
            "step_time_ratio_per_block_over_none": (
                per_block["median_step_seconds"] / none["median_step_seconds"]
            ),
            "tokens_per_second_ratio_per_block_over_none": (
                per_block["median_tokens_per_second"]
                / none["median_tokens_per_second"]
            ),
        }
        pair["parity"] = _pair_parity(
            stage_path,
            sequence_length=context,
            dtype_name=dtype_name,
            seed=seed,
        )
        pair["dcp_compatibility"] = _dcp_roundtrip(
            stage_path,
            sequence_length=context,
            dtype_name=dtype_name,
            seed=seed,
        )
        pair["passed"] = bool(
            pair["parity"]["passed"]
            and pair["dcp_compatibility"]["passed"]
        )
        return {
            "status": "PASS" if pair["passed"] else "FAIL",
            "largest_common_feasible_sequence_length": context,
            "selected": pair,
            "attempts": attempts,
        }
    return {
        "status": "NO_COMMON_FEASIBLE_SETUP",
        "largest_common_feasible_sequence_length": None,
        "selected": None,
        "attempts": attempts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    probe = _device_probe()
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source_sha": _source_sha(),
        "scale143_source_sha": "30f85fdfb930bb38d45d116fee9c2e82f8241b56",
        "scale143_checkpoint_implementation": (
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper / "
            "CheckpointImpl.NO_REENTRANT"
        ),
        "paid_compute": False,
        "hardware": probe,
        "gpu_pass": False,
        "comparisons": {},
    }
    if not probe["cuda_available"]:
        result["status"] = "NOT_RUN_NO_GPU"
        result["reason"] = (
            "torch.cuda.is_available() is false; CPU evidence is not promoted "
            "to a GPU PASS"
        )
        return result

    identity_ok, identity_errors = _runtime_identity_ok(probe)
    result["cuda_runtime_identity"] = {
        "expected_torch": EXPECTED_TORCH,
        "expected_torch_cuda": EXPECTED_TORCH_CUDA,
        "passed": identity_ok,
        "errors": identity_errors,
    }
    if not identity_ok:
        result["status"] = "BLOCKED_CUDA_ENVIRONMENT_MISMATCH"
        result["reason"] = (
            "visible CUDA hardware is not running the exact accepted D08 "
            "CUDA purpose runtime"
        )
        return result

    usable = int(probe["free_hbm_bytes_preflight"])
    ten_stage = Path(args.ten_million_stage)
    hundred_stage = Path(args.hundred_million_stage)
    ten_context = min(
        1024,
        json.loads(ten_stage.read_text(encoding="utf-8"))["model"]["max_seq_len"],
    )
    result["comparisons"]["10m"] = _matched_comparison(
        ten_stage,
        contexts=[ten_context],
        measured_steps=args.measured_steps,
        dtype_name=args.dtype,
        seed=args.seed,
        usable_hbm_bytes=usable,
    )
    max_context = int(
        json.loads(hundred_stage.read_text(encoding="utf-8"))["model"][
            "max_seq_len"
        ]
    )
    contexts = [
        value
        for value in (4096, 3072, 2048, 1536, 1024, 768, 512, 256, 128)
        if value <= max_context
    ]
    result["comparisons"]["100m"] = _matched_comparison(
        hundred_stage,
        contexts=contexts,
        measured_steps=args.measured_steps,
        dtype_name=args.dtype,
        seed=args.seed + 100,
        usable_hbm_bytes=usable,
    )
    result["gpu_pass"] = all(
        row.get("status") == "PASS"
        for row in result["comparisons"].values()
    )
    result["status"] = (
        "PASS_GPU_MEASURED" if result["gpu_pass"] else "FAIL_GPU_MEASURED"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ten-million-stage", default="configs/stages/s3_10m.json"
    )
    parser.add_argument(
        "--hundred-million-stage",
        default="configs/stages/s4_100m_accelerator.candidate.json",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scale205-activation-checkpoint-gpu.json"),
    )
    args = parser.parse_args()
    if args.measured_steps < 1:
        parser.error("--measured-steps must be >= 1")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"NOT_RUN_NO_GPU", "PASS_GPU_MEASURED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
