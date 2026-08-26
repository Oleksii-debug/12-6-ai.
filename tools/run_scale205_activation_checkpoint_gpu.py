#!/usr/bin/env python3
"""Device-bound SCALE-205 activation-checkpoint qualification.

This is an evidence harness only. It reuses SCALE-143's maintained PyTorch
checkpoint wrapper through ``apply_activation_checkpointing`` and never implements
recomputation itself.

The parent process keeps OOM attempts isolated in fresh subprocesses. A CUDA PASS is
possible only with one visible CUDA device and an explicit free/authorized-compute
flag. Otherwise the tool writes a machine-readable NOT_RUN result and exits cleanly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from twelve_six.distributed.activation_checkpointing import apply_activation_checkpointing
from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
    verify_scale_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.loss import causal_lm_loss

SCHEMA = "12-6.scale205.activation-checkpoint-gpu.v1"
SOURCE_SCALE143_SHA = "30f85fdfb930bb38d45d116fee9c2e82f8241b56"
HEADROOM_LIMIT = 0.80
DEFAULT_10M = Path("configs/stages/s3_10m.json")
DEFAULT_100M = Path("configs/stages/s4_100m_accelerator.candidate.json")
BENCHMARK = Path("tools/benchmark_activation_checkpointing.py")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    for name in ("GITHUB_SHA", "SOURCE_SHA"):
        value = os.environ.get(name, "").strip().lower()
        if len(value) == 40 and all(ch in "0123456789abcdef" for ch in value):
            return value
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError):
        value = ""
    return value if len(value) == 40 else "0" * 40


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result.update(
            {
                "device_name": props.name,
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "total_hbm_bytes": int(total_bytes),
                "free_hbm_bytes_at_preflight": int(free_bytes),
                "usable_hbm_bytes": int(min(total_bytes, free_bytes)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
        try:
            smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().splitlines()
            result["nvidia_smi_rows"] = smi
        except (OSError, subprocess.CalledProcessError):
            result["nvidia_smi_rows"] = []
    return result


def decide_checkpoint_policy(
    *,
    uncheckpointed_status: str,
    peak_reserved_bytes: int | None,
    usable_hbm_bytes: int,
    threshold: float = HEADROOM_LIMIT,
) -> dict[str, Any]:
    """Apply the exact SCALE-143 decision rule to measured target-HBM evidence."""
    if usable_hbm_bytes <= 0:
        raise ValueError("usable_hbm_bytes must be positive")
    if uncheckpointed_status == "OOM":
        return {
            "decision": "per_block",
            "reason": "uncheckpointed_oom",
            "uncheckpointed_reserved_fraction_of_usable_hbm": None,
            "threshold": threshold,
        }
    if uncheckpointed_status != "PASS" or peak_reserved_bytes is None:
        return {
            "decision": "UNRESOLVED",
            "reason": "uncheckpointed_measurement_unavailable",
            "uncheckpointed_reserved_fraction_of_usable_hbm": None,
            "threshold": threshold,
        }
    fraction = peak_reserved_bytes / usable_hbm_bytes
    return {
        "decision": "none" if fraction <= threshold else "per_block",
        "reason": "measured_reserved_hbm_threshold",
        "uncheckpointed_reserved_fraction_of_usable_hbm": fraction,
        "threshold": threshold,
    }


def ordered_100m_setups(max_context: int) -> list[tuple[int, int]]:
    """Largest bounded setup first, prioritizing tokens/step then native context."""
    contexts = [item for item in (4096, 2048, 1024, 512, 256) if item <= max_context]
    candidates = [(batch, seq) for seq in contexts for batch in (4, 2, 1)]
    return sorted(candidates, key=lambda item: (item[0] * (item[1] - 1), item[1], item[0]), reverse=True)


def _run_json_command(command: list[str]) -> tuple[str, dict[str, Any] | None, str | None]:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        try:
            return "PASS", json.loads(completed.stdout), None
        except json.JSONDecodeError:
            return "ERROR", None, "child_did_not_emit_json"
    combined = (completed.stdout + "\n" + completed.stderr).lower()
    if "out of memory" in combined or "cuda error: out of memory" in combined:
        return "OOM", None, "cuda_out_of_memory"
    last = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
    return "ERROR", None, last[0][:240] if last else "child_failed"


def _benchmark(
    stage: Path,
    policy: str,
    *,
    batch: int,
    sequence: int,
    dtype: str,
    measured_steps: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--stage-config",
        str(stage),
        "--policy",
        policy,
        "--sequence-length",
        str(sequence),
        "--batch-size",
        str(batch),
        "--measured-steps",
        str(measured_steps),
        "--dtype",
        dtype,
        "--device",
        "cuda:0",
        "--seed",
        "205",
    ]
    status, payload, error = _run_json_command(command)
    return {"status": status, "error": error, "result": payload}


def _canonical_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _parity_child(stage_path: Path, batch: int, sequence: int, dtype_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for parity child")
    device = torch.device("cuda:0")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    torch.manual_seed(205)
    stage = load_stage_config(stage_path)
    if sequence > stage.model.max_seq_len:
        raise ValueError("sequence exceeds stage max_seq_len")

    # Keep the initial state on host memory so parity never requires two GPU models.
    initial = TwelveSixDecoder(stage.model, stage.init).to(dtype=dtype)
    initial_state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    del initial
    generator = torch.Generator(device="cpu").manual_seed(20_509)
    ids_cpu = torch.randint(0, stage.model.vocab_size, (batch, sequence), generator=generator)

    def run(policy: str):
        model = TwelveSixDecoder(stage.model, stage.init).to(dtype=dtype)
        model.load_state_dict(initial_state)
        plan = apply_activation_checkpointing(model, policy)  # maintained SCALE-143 wrapper
        model = model.to(device)
        ids = ids_cpu.to(device)
        logits = model(ids).logits
        loss = causal_lm_loss(logits, ids)
        loss.backward()
        logits_cpu = logits.detach().float().cpu()
        gradients = {
            _canonical_name(name): parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        outcome = {
            "loss": float(loss.detach().float().cpu()),
            "checkpointed_blocks": plan.checkpointed_blocks,
            "logits": logits_cpu,
            "gradients": gradients,
        }
        del ids, logits, loss, model
        gc.collect()
        torch.cuda.empty_cache()
        return outcome

    control = run("none")
    candidate = run("per_block")
    if set(control["gradients"]) != set(candidate["gradients"]):
        raise RuntimeError("gradient key mismatch")

    logit_delta = float((control["logits"] - candidate["logits"]).abs().max())
    gradient_delta = 0.0
    allclose = torch.allclose
    tolerance = {
        "fp32": {"rtol": 1e-6, "atol": 1e-7},
        "bf16": {"rtol": 2e-2, "atol": 2e-2},
        "fp16": {"rtol": 5e-3, "atol": 5e-3},
    }[dtype_name]
    gradients_close = True
    for name in control["gradients"]:
        left = control["gradients"][name]
        right = candidate["gradients"][name]
        gradient_delta = max(gradient_delta, float((left - right).abs().max()))
        gradients_close = gradients_close and bool(allclose(left, right, **tolerance))
    logits_close = bool(allclose(control["logits"], candidate["logits"], **tolerance))
    return {
        "stage": stage.stage,
        "parameter_count": stage.model.parameter_count(),
        "model_spec_sha256": stage.model.identity_sha256(),
        "dtype": dtype_name,
        "batch_size": batch,
        "sequence_length": sequence,
        "tolerance": tolerance,
        "none_loss": control["loss"],
        "per_block_loss": candidate["loss"],
        "max_abs_logit_delta": logit_delta,
        "max_abs_gradient_delta": gradient_delta,
        "logits_within_tolerance": logits_close,
        "gradients_within_tolerance": gradients_close,
        "passed": logits_close and gradients_close,
    }


def _dcp_child(stage_path: Path, dtype_name: str, output_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DCP child")
    import torch.distributed as dist

    device = torch.device("cuda:0")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    torch.manual_seed(205)
    stage = load_stage_config(stage_path)
    descriptor, init_file = tempfile.mkstemp(prefix="scale205-nccl-")
    os.close(descriptor)
    os.unlink(init_file)
    checkpoint = output_root / f"{stage.stage.lower()}-per-block-dcp"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        model = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
        plan_info = apply_activation_checkpointing(model, "per_block")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        generator = torch.Generator(device="cpu").manual_seed(205_001)
        ids = torch.randint(0, stage.model.vocab_size, (1, min(16, stage.model.max_seq_len)), generator=generator).to(device)
        loss = causal_lm_loss(model(ids).logits, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            reference = model(ids).logits.detach().float().cpu()

        plan = ParallelPlan(data_parallel=1, shard_model_state_across_data_parallel=False)
        identity = ScaleCheckpointIdentity(
            git_sha=_git_sha(),
            model_spec_sha256=stage.model.identity_sha256(),
            init_spec_sha256=_sha(f"scale205-init:{stage_path}"),
            tokenizer_config_sha256=_sha("scale205-compat-tokenizer-config"),
            tokenizer_vocab_sha256=_sha("scale205-compat-tokenizer-vocab"),
            data_manifest_sha256=_sha("scale205-compat-data"),
            packing_sha256=_sha("scale205-compat-packing"),
            training_config_sha256=_sha(f"scale205:{stage.stage}:{dtype_name}"),
            environment_lock_sha256=_sha("linux-x86_64-cuda-training"),
            seed=205,
            step=1,
            tokens_seen=int(ids.numel() - 1),
        )
        manifest = save_scale_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            plan=plan,
            identity=identity,
            metadata={
                "authority": "SCALE205_GPU_DCP_COMPATIBILITY_ONLY",
                "activation_checkpoint_policy": "per_block",
                "checkpoint_implementation": plan_info.implementation,
            },
        )
        verified = verify_scale_checkpoint(checkpoint)
        del model, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()

        model2 = TwelveSixDecoder(stage.model, stage.init).to(device=device, dtype=dtype)
        apply_activation_checkpointing(model2, "per_block")
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.0)
        loaded = load_scale_checkpoint(
            checkpoint,
            model=model2,
            optimizer=optimizer2,
            target_plan=plan,
            mode=ResumeMode.EXACT_TOPOLOGY,
            expected_identity_sha256=identity.sha256,
        )
        model2.eval()
        with torch.no_grad():
            restored = model2(ids).logits.detach().float().cpu()
        delta = float((reference - restored).abs().max())
        tolerance = {"fp32": 1e-7, "bf16": 2e-2, "fp16": 5e-3}[dtype_name]
        result = {
            "stage": stage.stage,
            "parameter_count": stage.model.parameter_count(),
            "model_spec_sha256": stage.model.identity_sha256(),
            "backend": "torch.distributed.checkpoint",
            "process_group_backend": "nccl",
            "activation_checkpoint_policy": "per_block",
            "checkpointed_blocks": plan_info.checkpointed_blocks,
            "identity_sha256": identity.sha256,
            "aggregate_checkpoint_sha256": manifest["aggregate_checkpoint_sha256"],
            "verified_aggregate_checkpoint_sha256": verified["aggregate_checkpoint_sha256"],
            "load_exact_topology": loaded.exact_topology,
            "post_reload_max_abs_logit_delta": delta,
            "post_reload_tolerance": tolerance,
            "passed": bool(loaded.exact_topology and delta <= tolerance),
        }
        shutil.rmtree(checkpoint)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(init_file):
            os.unlink(init_file)


def _child_command(mode: str, stage: Path, batch: int, sequence: int, dtype: str, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        __file__,
        mode,
        "--stage",
        str(stage),
        "--batch-size",
        str(batch),
        "--sequence-length",
        str(sequence),
        "--dtype",
        dtype,
        "--child-output-root",
        str(output),
    ]
    status, payload, error = _run_json_command(command)
    return {"status": status, "error": error, "result": payload}


def _scale_record(
    *,
    label: str,
    stage_path: Path,
    batch: int,
    sequence: int,
    dtype: str,
    usable_hbm: int,
    output_root: Path,
    measured_steps: int,
) -> dict[str, Any]:
    none = _benchmark(stage_path, "none", batch=batch, sequence=sequence, dtype=dtype, measured_steps=measured_steps)
    per_block = _benchmark(stage_path, "per_block", batch=batch, sequence=sequence, dtype=dtype, measured_steps=measured_steps)
    peak_reserved = None
    if none["status"] == "PASS" and none["result"] is not None:
        peak_reserved = int(none["result"]["peak_cuda_reserved_bytes"])
    decision = decide_checkpoint_policy(
        uncheckpointed_status=none["status"],
        peak_reserved_bytes=peak_reserved,
        usable_hbm_bytes=usable_hbm,
    )
    parity = _child_command("--parity-child", stage_path, batch, sequence, dtype, output_root)
    dcp = _child_command("--dcp-child", stage_path, batch, sequence, dtype, output_root)
    decision_supported = (
        decision["decision"] == "none" and none["status"] == "PASS"
    ) or (
        decision["decision"] == "per_block" and per_block["status"] == "PASS"
    )
    return {
        "label": label,
        "stage_config": str(stage_path),
        "setup": {"batch_size": batch, "sequence_length": sequence, "dtype": dtype},
        "none": none,
        "per_block": per_block,
        "measured_policy_decision": decision,
        "decision_supported_by_executable_path": decision_supported,
        "numerical_parity": parity,
        "dcp_compatibility": dcp,
    }


def _parent(args: argparse.Namespace) -> int:
    output = args.output
    hardware = _hardware()
    base: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": "SCALE-205-ACTIVATION-CHECKPOINT-GPU",
        "source_scale143_sha": SOURCE_SCALE143_SHA,
        "source_sha": _git_sha(),
        "headroom_policy": {
            "uncheckpointed_reserved_hbm_limit": HEADROOM_LIMIT,
            "above_limit": "per_block",
            "on_oom": "per_block",
            "every_other_block_is_default": False,
        },
        "hardware": hardware,
        "compute_boundary": {
            "paid_compute_authorized": False,
            "free_or_user_authorized_gpu_required": True,
            "authorization_flag_received": bool(args.authorized_free_gpu),
        },
        "scales": {},
    }
    if not hardware["cuda_available"] or hardware["cuda_device_count"] == 0:
        base["status"] = "NOT_RUN_NO_GPU"
        base["conclusion"] = "No CUDA measurements; CPU evidence is not promoted to GPU PASS."
        _write_json(output, base)
        print(json.dumps(base, sort_keys=True))
        return 0
    if not args.authorized_free_gpu:
        base["status"] = "NOT_RUN_NOT_AUTHORIZED"
        base["conclusion"] = "CUDA is visible but explicit free/authorized-compute permission is absent."
        _write_json(output, base)
        print(json.dumps(base, sort_keys=True))
        return 0
    if hardware["cuda_device_count"] != 1:
        base["status"] = "NOT_RUN_AMBIGUOUS_CUDA_VISIBILITY"
        base["conclusion"] = "Expose exactly one authorized target GPU with CUDA_VISIBLE_DEVICES."
        _write_json(output, base)
        print(json.dumps(base, sort_keys=True))
        return 0
    if args.dtype == "bf16" and not hardware.get("bf16_supported", False):
        base["status"] = "NOT_RUN_BF16_UNSUPPORTED"
        base["conclusion"] = "Requested native BF16 qualification is unsupported on the visible GPU."
        _write_json(output, base)
        print(json.dumps(base, sort_keys=True))
        return 0

    usable_hbm = int(hardware["usable_hbm_bytes"])
    output_root = output.parent / "scale205-transient"
    output_root.mkdir(parents=True, exist_ok=True)

    stage10 = load_stage_config(args.stage_10m)
    record10 = _scale_record(
        label="10M",
        stage_path=args.stage_10m,
        batch=1,
        sequence=stage10.model.max_seq_len,
        dtype=args.dtype,
        usable_hbm=usable_hbm,
        output_root=output_root,
        measured_steps=2,
    )
    base["scales"]["10M"] = record10

    stage100 = load_stage_config(args.stage_100m)
    selection_attempts: list[dict[str, Any]] = []
    selected: tuple[int, int] | None = None
    for batch, sequence in ordered_100m_setups(stage100.model.max_seq_len):
        probe = _benchmark(
            args.stage_100m,
            "per_block",
            batch=batch,
            sequence=sequence,
            dtype=args.dtype,
            measured_steps=1,
        )
        selection_attempts.append(
            {"batch_size": batch, "sequence_length": sequence, "status": probe["status"], "error": probe["error"]}
        )
        if probe["status"] == "PASS":
            selected = (batch, sequence)
            break
        if probe["status"] != "OOM":
            break

    if selected is None:
        base["scales"]["100M"] = {
            "status": "NOT_RUN_NO_FEASIBLE_100M_SETUP",
            "selection_attempts": selection_attempts,
        }
    else:
        batch, sequence = selected
        record100 = _scale_record(
            label="100M",
            stage_path=args.stage_100m,
            batch=batch,
            sequence=sequence,
            dtype=args.dtype,
            usable_hbm=usable_hbm,
            output_root=output_root,
            measured_steps=2,
        )
        record100["selection_attempts"] = selection_attempts
        record100["largest_feasible_definition"] = "maximum valid tokens/step among bounded candidates, tie-broken by longer context"
        base["scales"]["100M"] = record100

    shutil.rmtree(output_root, ignore_errors=True)
    scale_values = [value for value in base["scales"].values() if isinstance(value, dict)]
    gpu_pass = bool(scale_values) and all(
        value.get("decision_supported_by_executable_path") is True
        and value.get("numerical_parity", {}).get("status") == "PASS"
        and value.get("numerical_parity", {}).get("result", {}).get("passed") is True
        and value.get("dcp_compatibility", {}).get("status") == "PASS"
        and value.get("dcp_compatibility", {}).get("result", {}).get("passed") is True
        for value in scale_values
    ) and "100M" in base["scales"] and base["scales"]["100M"].get("label") == "100M"
    base["status"] = "PASS" if gpu_pass else "GPU_EXECUTED_WITH_BLOCKERS"
    base["conclusion"] = (
        "Measured target-HBM evidence supports the SCALE-143 80% decision rule."
        if gpu_pass
        else "CUDA executed, but one or more required SCALE-205 gates did not pass."
    )
    _write_json(output, base)
    print(json.dumps(base, sort_keys=True))
    return 0 if gpu_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-10m", type=Path, default=DEFAULT_10M)
    parser.add_argument("--stage-100m", type=Path, default=DEFAULT_100M)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--authorized-free-gpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/scale205/activation_checkpoint_gpu_qualification.json"),
    )
    parser.add_argument("--parity-child", action="store_true")
    parser.add_argument("--dcp-child", action="store_true")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--child-output-root", type=Path, default=Path(".scale205-child"))
    args = parser.parse_args()
    if args.parity_child:
        if args.stage is None:
            parser.error("--parity-child requires --stage")
        result = _parity_child(args.stage, args.batch_size, args.sequence_length, args.dtype)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.dcp_child:
        if args.stage is None:
            parser.error("--dcp-child requires --stage")
        result = _dcp_child(args.stage, args.dtype, args.child_output_root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 2
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
