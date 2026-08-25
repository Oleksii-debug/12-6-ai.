from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.s0_evidence_contract import validate_locked_environment_evidence

_GIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.is_file():
        return None
    fields = statm.read_text(encoding="utf-8").split()
    if len(fields) < 2:
        return None
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(_tensor_bytes(parameter) for parameter in model.parameters())


def _dtype_counts(tensors: list[torch.Tensor]) -> dict[str, int]:
    counts = Counter(str(tensor.dtype) for tensor in tensors)
    return dict(sorted(counts.items()))


def _optimizer_tensor_state(trainer: Trainer) -> tuple[int, dict[str, int]]:
    tensors = [
        value
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    return sum(_tensor_bytes(value) for value in tensors), _dtype_counts(tensors)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _weight_delta(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_probe(
    root: Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    stage_config: str,
    precision: str,
    device_name: str,
    seed: int,
    steps: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError("source SHA must be a full lowercase Git SHA")
    if steps <= 0 or batch_size <= 0 or sequence_length < 2:
        raise ValueError("steps/batch_size must be positive and sequence_length must be >= 2")

    root = root.resolve()
    stage_path = (root / stage_config).resolve()
    stage = load_stage_config(stage_path)
    if sequence_length > stage.model.max_seq_len:
        raise ValueError("probe sequence length exceeds ModelSpec max_seq_len")
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    device = torch.device(device_name)

    rss_before_model = _rss_bytes()
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    rss_after_model = _rss_bytes()
    before = _snapshot(model)
    snapshot_bytes = sum(_tensor_bytes(value) for value in before.values())

    config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        max_steps=steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision=precision,  # type: ignore[arg-type]
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    trainer = Trainer(model, config, device=device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    batches = [
        {
            "input_ids": torch.randint(
                0,
                stage.model.vocab_size,
                (batch_size, sequence_length),
                generator=generator,
                dtype=torch.long,
            )
        }
        for _ in range(steps)
    ]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler_scales: list[float] = []
    metrics: list[dict[str, float | int | bool | None]] = []
    _sync(device)
    wall_start = time.perf_counter()
    for batch in batches:
        item = trainer.train_microbatch(batch)
        if hasattr(trainer.scaler, "get_scale"):
            scaler_scales.append(float(trainer.scaler.get_scale()))
        metrics.append(
            {
                "micro_step": item.micro_step,
                "optimizer_step": item.optimizer_step,
                "loss": item.loss,
                "update_loss": item.update_loss,
                "grad_norm": item.grad_norm,
                "tokens": item.tokens,
                "optimizer_stepped": item.optimizer_stepped,
            }
        )
    _sync(device)
    wall_seconds = time.perf_counter() - wall_start

    rss_after_training_with_snapshot = _rss_bytes()
    delta = _weight_delta(model, before)
    del before
    gc.collect()
    rss_after_snapshot_release = _rss_bytes()

    optimizer_state_bytes, optimizer_state_dtypes = _optimizer_tensor_state(trainer)
    parameter_tensors = [parameter for parameter in model.parameters()]
    parameter_bytes = _parameter_bytes(model)

    model.eval()
    inference_input = batches[0]["input_ids"].to(device)
    with torch.no_grad():
        inference_output = model(inference_input)
    inference_logits = inference_output.logits

    trainer_state = trainer.state_dict()
    runtime = trainer.precision_runtime.to_dict()
    gpu_memory: dict[str, int] | None = None
    if device.type == "cuda":
        gpu_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    tokens = trainer.tokens_seen
    evidence: dict[str, Any] = {
        "schema_version": "12-6.precision-scale-probe.v1",
        "authority": "ENGINEERING_PRECISION_EVIDENCE_NOT_STAGE_PROMOTION",
        "identity": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "stage": stage.stage,
            "stage_config": stage_config,
            "stage_config_file_sha256": _sha256_file(stage_path),
            "modelspec_sha256": stage.model.identity_sha256(),
            "parameter_count": stage.expected_parameters,
            "environment": environment,
        },
        "requested": {
            "precision": precision,
            "device": str(device),
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        },
        "resolved_runtime": runtime,
        "execution": {
            "status": "PASS",
            "cuda_executed": device.type == "cuda",
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "loss_trajectory": metrics,
            "scaler_scale_trajectory": scaler_scales,
            "optimizer_steps": trainer.optimizer_step,
            "tokens": tokens,
            "wall_seconds": wall_seconds,
            "optimizer_steps_per_second": trainer.optimizer_step / wall_seconds,
            "tokens_per_second": tokens / wall_seconds,
            "weight_delta": delta,
        },
        "memory": {
            "model_parameter_bytes": parameter_bytes,
            "optimizer_tensor_state_bytes": optimizer_state_bytes,
            "accounted_model_plus_optimizer_bytes": parameter_bytes + optimizer_state_bytes,
            "measurement_snapshot_bytes": snapshot_bytes,
            "rss_before_model_bytes": rss_before_model,
            "rss_after_model_bytes": rss_after_model,
            "rss_after_training_with_snapshot_bytes": rss_after_training_with_snapshot,
            "rss_after_snapshot_release_bytes": rss_after_snapshot_release,
            "cuda": gpu_memory,
            "scope": "tensor-state plus process RSS; CPU activation peak is not claimed",
        },
        "dtype_semantics": {
            "parameter_dtypes": _dtype_counts(parameter_tensors),
            "optimizer_state_dtypes": optimizer_state_dtypes,
            "native_post_training_inference_parameter_dtype": str(parameter_tensors[0].dtype),
            "native_post_training_inference_logits_dtype": str(inference_logits.dtype),
        },
        "checkpoint_precision_identity": {
            "trainer_config_precision": trainer_state.config["precision"],
            "scaler_state_present": trainer_state.scaler is not None,
            "scaler_state_keys": sorted((trainer_state.scaler or {}).keys()),
        },
        "claims": {
            "stage_promoted": False,
            "cross_precision_bitwise_equality": False,
            "cuda_evidence": device.type == "cuda",
            "paid_compute_used": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a device-bound D02 precision scale probe")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--stage-config", default="configs/stages/s3_10m.json")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence = run_probe(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        stage_config=args.stage_config,
        precision=args.precision,
        device_name=args.device,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    execution = evidence["execution"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "precision": args.precision,
                "device": args.device,
                "parameter_count": evidence["identity"]["parameter_count"],
                "final_loss": execution["loss_trajectory"][-1]["loss"],
                "tokens_per_second": execution["tokens_per_second"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
