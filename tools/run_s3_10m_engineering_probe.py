"""Run the live byte-compatible S3 ~10M candidate through the integrated stack."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    prepare_checkpoint_load,
    verify_checkpoint,
)
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import TwelveSixDecoder, count_trainable_parameters
from twelve_six.s3_engineering import (
    S3_CURRENT_CANDIDATE_ID,
    S3_CURRENT_EXPECTED_PARAMETERS,
    S3_CURRENT_MODEL_SHA256,
    S3_CURRENT_SOURCE_PR,
    S3_CURRENT_SOURCE_SHA,
    S3_D11_CANDIDATE_ID,
    S3_D11_EXPECTED_PARAMETERS,
    S3_D11_MODEL_SHA256,
    S3_D11_SOURCE_PR,
    S3_D11_SOURCE_SHA,
    kv_cache_bytes,
    s3_current_model_spec,
    s3_d11_model_spec,
    s3_init_spec,
    s4_d11_model_spec,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

_SCHEMA = "12-6.s3-10m-engineering-evidence.v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_label(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _memory_kib() -> dict[str, int | None]:
    status = Path("/proc/self/status")
    values: dict[str, int | None] = {"rss_kib": None, "high_water_kib": None}
    if not status.exists():
        return values
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            values["rss_kib"] = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            values["high_water_kib"] = int(line.split()[1])
    return values


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return _tensor_tree_bytes(asdict(value))
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _model_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _small_tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _lock_identity(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "requirements/locks/linux-x86_64/runtime.lock.txt",
        "requirements/locks/linux-x86_64/dev.lock.txt",
    ):
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_identity(
    *,
    source_sha: str,
    training_config: Mapping[str, Any],
    step: int,
    tokens_seen: int,
    environment_lock_hash: str,
) -> CheckpointIdentity:
    tokenizer = ByteTokenizer()
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=s3_current_model_spec().to_dict(),
        parameter_count=S3_CURRENT_EXPECTED_PARAMETERS,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=_sha256_label("S3-controlled-synthetic-byte-mechanics-v1"),
        run_manifest_hash=_sha256_label("SCALE-03-S3-10M-engineering-probe-v2"),
        training_config=dict(training_config),
        seed=int(training_config["trainer"]["seed"]),
        precision=str(training_config["trainer"]["precision"]),
        step=step,
        tokens_seen=tokens_seen,
        optimizer={
            "name": "AdamW",
            "betas": list(training_config["trainer"]["betas"]),
            "eps": training_config["trainer"]["eps"],
            "weight_decay": training_config["trainer"]["weight_decay"],
        },
        scheduler=None,
        environment_lock_hash=environment_lock_hash,
    )


def _make_batch(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _training_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=args.optimizer_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=1.0,
        precision=args.precision,
        seed=args.seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _save_bound_checkpoint(
    *,
    checkpoint_dir: Path,
    model: TwelveSixDecoder,
    trainer: Trainer,
    source_sha: str,
    bound_training_config: Mapping[str, Any],
    environment_lock_hash: str,
) -> dict[str, Any]:
    identity = _checkpoint_identity(
        source_sha=source_sha,
        training_config=bound_training_config,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        environment_lock_hash=environment_lock_hash,
    )
    started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    elapsed = time.perf_counter() - started
    return {"manifest": manifest, "save_seconds": elapsed}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    if _git_head(repo_root) != args.source_sha:
        raise RuntimeError("source SHA does not match exact checkout")

    spec = s3_current_model_spec()
    init_spec = s3_init_spec()
    future_spec = s3_d11_model_spec()
    s4_spec = s4_d11_model_spec()
    if args.sequence_length > spec.max_seq_len:
        raise ValueError("sequence length exceeds current S3 context")
    if args.optimizer_steps < 1:
        raise ValueError("optimizer steps must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint cadence must be positive")

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    environment_lock_hash = _lock_identity(repo_root)
    tokenizer = ByteTokenizer()
    if tokenizer.vocab_size != spec.vocab_size:
        raise RuntimeError("current S3 candidate lost canonical byte-tokenizer compatibility")
    baseline_memory = _memory_kib()

    torch.manual_seed(args.seed)
    construct_started = time.perf_counter()
    model = TwelveSixDecoder(spec, init_spec)
    construct_seconds = time.perf_counter() - construct_started
    actual_parameters = count_trainable_parameters(model)
    if actual_parameters != S3_CURRENT_EXPECTED_PARAMETERS:
        raise RuntimeError("instantiated S3 parameter count differs from live algebra")
    model_memory = _memory_kib()
    model_parameter_bytes = _model_parameter_bytes(model)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA pilot requested but CUDA is unavailable")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 pilot requested but CUDA bf16 is unsupported")
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)

    forward_batch = _make_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=spec.vocab_size,
        seed=args.seed + 1,
    )
    forward_input = forward_batch["input_ids"].to(device)
    model.eval()
    forward_started = time.perf_counter()
    with torch.no_grad():
        forward_logits = model(forward_input).logits
        forward_checksum = float(forward_logits[0, -1, :32].float().sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    del forward_logits, forward_input, forward_batch

    trainer_config = _training_config(args)
    trainer = Trainer(model, trainer_config, device=device)
    bound_training_config: dict[str, Any] = {
        "schema": "12-6.s3-10m-training-mechanics.v2",
        "model_spec_sha256": S3_CURRENT_MODEL_SHA256,
        "init_spec_sha256": init_spec.identity_sha256(),
        "trainer": asdict(trainer_config),
        "data": {
            "kind": "controlled_synthetic_canonical_byte_vocabulary",
            "scope": "MECHANICS_ONLY_NOT_S3_CORPUS_FREEZE",
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
        },
    }
    before_update_sha = _small_tensor_sha256(model.final_norm.weight)
    training_started = time.perf_counter()
    last_metrics = None
    checkpoint_records: list[dict[str, Any]] = []

    checkpoint_root = args.checkpoint_root
    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_root is None:
        temp_context = tempfile.TemporaryDirectory(prefix="twelve-six-s3-10m-")
        checkpoint_root = Path(temp_context.name) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    micro_index = 0
    while trainer.optimizer_step < args.optimizer_steps:
        micro_index += 1
        current_batch = _make_batch(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocab_size=spec.vocab_size,
            seed=args.seed + 10_000 + micro_index,
        )
        last_metrics = trainer.train_microbatch(current_batch)
        if not last_metrics.optimizer_stepped:
            continue
        should_checkpoint = (
            last_metrics.optimizer_step % args.checkpoint_every == 0
            or last_metrics.optimizer_step == args.optimizer_steps
        )
        if should_checkpoint:
            checkpoint_dir = checkpoint_root / f"step-{last_metrics.optimizer_step:06d}"
            saved = _save_bound_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                trainer=trainer,
                source_sha=args.source_sha,
                bound_training_config=bound_training_config,
                environment_lock_hash=environment_lock_hash,
            )
            manifest = saved["manifest"]
            checkpoint_records.append(
                {
                    "step": last_metrics.optimizer_step,
                    "tokens_seen": trainer.tokens_seen,
                    "directory": str(checkpoint_dir),
                    "checkpoint_id": manifest["checkpoint_id"],
                    "payload_bytes": sum(item["bytes"] for item in manifest["files"].values()),
                    "save_seconds": saved["save_seconds"],
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    if last_metrics is None or not last_metrics.optimizer_stepped:
        raise RuntimeError("probe did not commit an optimizer step")
    if not math.isfinite(last_metrics.loss):
        raise RuntimeError("probe loss is non-finite")
    after_update_sha = _small_tensor_sha256(model.final_norm.weight)
    if before_update_sha == after_update_sha:
        raise RuntimeError("optimizer step did not change the probed parameter")

    optimizer_state_bytes = _tensor_tree_bytes(trainer.state_dict().optimizer)
    after_training_memory = _memory_kib()
    final_checkpoint = Path(checkpoint_records[-1]["directory"])
    verified_manifest = verify_checkpoint(final_checkpoint)

    snapshot_before = _memory_kib()
    snapshot_started = time.perf_counter()
    verified = prepare_checkpoint_load(final_checkpoint)
    snapshot_seconds = time.perf_counter() - snapshot_started
    snapshot_after = _memory_kib()
    snapshot_payload_bytes = sum(item["bytes"] for item in verified.manifest["files"].values())
    if snapshot_payload_bytes != checkpoint_records[-1]["payload_bytes"]:
        raise RuntimeError("verified snapshot payload size drifted")
    del verified
    gc.collect()

    restored_model = TwelveSixDecoder(spec, init_spec)
    restored_trainer = Trainer(restored_model, trainer_config, device=device)
    reload_started = time.perf_counter()
    load_result = load_trainer_checkpoint(
        final_checkpoint,
        model=restored_model,
        trainer=restored_trainer,
        expected_git_sha=args.source_sha,
        expected_model_spec_hash=S3_CURRENT_MODEL_SHA256,
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=verified_manifest["identity"]["dataset_manifest_hash"],
        expected_run_manifest_hash=verified_manifest["identity"]["run_manifest_hash"],
        expected_environment_lock_hash=environment_lock_hash,
        expected_seed=args.seed,
    )
    reload_seconds = time.perf_counter() - reload_started
    if restored_trainer.optimizer_step != trainer.optimizer_step:
        raise RuntimeError("restored optimizer step differs from saved trainer")
    if _small_tensor_sha256(restored_model.final_norm.weight) != after_update_sha:
        raise RuntimeError("restored model weights differ from saved model")
    after_reload_memory = _memory_kib()

    first_party_started = time.perf_counter()
    backend = load_first_party_backend(final_checkpoint)
    backend_diagnostics = backend.diagnostics()
    generation = generate(
        backend,
        "12-6 scale probe",
        GenerationConfig(max_new_tokens=2, sample=False, seed=args.seed),
    )
    inference_seconds = time.perf_counter() - first_party_started
    if len(generation.generated_token_ids) != 2:
        raise RuntimeError("first-party generation did not produce two tokens")
    if backend_diagnostics["parameter_count"] != S3_CURRENT_EXPECTED_PARAMETERS:
        raise RuntimeError("first-party backend reconstructed the wrong parameter count")
    if backend_diagnostics["model_spec_sha256"] != S3_CURRENT_MODEL_SHA256:
        raise RuntimeError("first-party backend reconstructed the wrong ModelSpec")

    cuda_peak_allocated_bytes = None
    if device.type == "cuda":
        cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))

    if temp_context is not None:
        temp_context.cleanup()

    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "source_sha": args.source_sha,
        "canonical_base": "random_init",
        "promotion_authority": False,
        "paid_compute": False,
        "candidate": {
            "id": S3_CURRENT_CANDIDATE_ID,
            "source_pr": S3_CURRENT_SOURCE_PR,
            "source_sha": S3_CURRENT_SOURCE_SHA,
            "selection": "CURRENT_EXECUTABLE_BYTE_COMPATIBLE_GEOMETRY",
            "model_spec_sha256": S3_CURRENT_MODEL_SHA256,
            "expected_parameters": S3_CURRENT_EXPECTED_PARAMETERS,
            "analytic_parameters": spec.parameter_count(),
            "instantiated_trainable_parameters": actual_parameters,
            "parameter_breakdown": spec.parameter_breakdown(),
            "geometry": spec.to_dict(),
        },
        "future_tokenizer_alternative": {
            "id": S3_D11_CANDIDATE_ID,
            "source_pr": S3_D11_SOURCE_PR,
            "source_sha": S3_D11_SOURCE_SHA,
            "model_spec_sha256": S3_D11_MODEL_SHA256,
            "parameters": S3_D11_EXPECTED_PARAMETERS,
            "geometry": future_spec.to_dict(),
            "current_byte_tokenizer_compatible": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "precision": args.precision,
            "cpu_threads": torch.get_num_threads(),
            "environment_lock_hash": environment_lock_hash,
        },
        "execution": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer_steps": args.optimizer_steps,
            "optimized_tokens": restored_trainer.tokens_seen,
            "construct_seconds": construct_seconds,
            "forward_no_grad_seconds": forward_seconds,
            "forward_checksum": forward_checksum,
            "train_backward_update_and_checkpoint_seconds": training_seconds,
            "last_loss": last_metrics.loss,
            "last_update_loss": last_metrics.update_loss,
            "last_grad_norm": last_metrics.grad_norm,
            "parameter_changed": before_update_sha != after_update_sha,
        },
        "memory": {
            "baseline": baseline_memory,
            "after_model_construction": model_memory,
            "after_training": after_training_memory,
            "before_verified_snapshot": snapshot_before,
            "holding_verified_snapshot": snapshot_after,
            "after_checkpoint_reload": after_reload_memory,
            "model_parameter_bytes": model_parameter_bytes,
            "optimizer_tensor_bytes": optimizer_state_bytes,
            "verified_snapshot_payload_bytes": snapshot_payload_bytes,
            "snapshot_is_full_payload_in_memory": True,
            "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
            "s3_bf16_native_gqa_kv_bytes_batch1_full_context": kv_cache_bytes(
                spec,
                batch_size=1,
                sequence_length=spec.max_seq_len,
                bytes_per_element=2,
            ),
            "s3_bf16_current_expanded_kv_bytes_batch1_full_context": (
                kv_cache_bytes(
                    spec,
                    batch_size=1,
                    sequence_length=spec.max_seq_len,
                    bytes_per_element=2,
                )
                * spec.n_heads
                // spec.n_kv_heads
            ),
        },
        "checkpoint": {
            "format": verified_manifest["format"],
            "format_version": verified_manifest["format_version"],
            "records": checkpoint_records,
            "verified_snapshot_seconds": snapshot_seconds,
            "reload_seconds": reload_seconds,
            "restored_checkpoint_id": load_result.manifest["checkpoint_id"],
            "restored_optimizer_step": restored_trainer.optimizer_step,
            "restored_tokens_seen": restored_trainer.tokens_seen,
            "scale_boundary": (
                "checkpoint-v1 verifies by retaining all serialized payload bytes in memory; "
                "acceptable to measure at S3 but not a final larger-scale design"
            ),
        },
        "inference": {
            "first_party_checkpoint_to_generation": "PASS_MECHANICS_ONLY",
            "generated_token_count": len(generation.generated_token_ids),
            "inference_seconds": inference_seconds,
            "backend_diagnostics": backend_diagnostics,
            "runtime_tokenizer": tokenizer.identity.version,
            "runtime_tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "runtime_tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "candidate_vocab": spec.vocab_size,
            "runtime_tokenizer_compatible": True,
            "s3_stage_tokenizer_frozen": False,
            "kv_cache": "NOT_ON_EXACT_GREEN_BASE__PR_138_CURRENT_HEAD_RED",
        },
        "architecture_runtime_boundary": {
            "gqa_parameter_savings_realized": True,
            "native_kv_cache_shape_algebra_valid": True,
            "training_attention_kv_is_currently_expanded_before_sdpa": True,
            "gqa_training_activation_memory_savings_realized": False,
            "note": "current attention repeats K/V to query-head count before SDPA",
        },
        "s4_readiness": {
            "owner": "SCALE-04_ACTIVE_CLAIM__HANDOFF_ONLY",
            "current_d11_candidate_parameters": s4_spec.parameter_count(),
            "model_spec_sha256": s4_spec.identity_sha256(),
            "geometry": s4_spec.to_dict(),
            "fp32_parameter_bytes": s4_spec.parameter_count() * 4,
            "bf16_kv_cache_bytes_batch1_full_context": kv_cache_bytes(
                s4_spec,
                batch_size=1,
                sequence_length=s4_spec.max_seq_len,
                bytes_per_element=2,
            ),
            "status": "HANDOFF_ALGEBRA_ONLY_NOT_INSTANTIATED",
        },
        "truth_boundary": {
            "training_data": "controlled synthetic mechanics only",
            "runtime_tokenizer_is_canonical_byte_v1": True,
            "s3_corpus_or_tokenizer_frozen": False,
            "capability_evidence": False,
            "gpu_execution": device.type == "cuda",
            "distributed_execution": False,
            "kv_cache_execution": False,
            "audit_pass": False,
            "stage_frozen": False,
        },
    }
    report["report_sha256"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0 or args.sequence_length < 2:
        raise ValueError("batch size must be positive and sequence length must be at least 2")
    report = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
