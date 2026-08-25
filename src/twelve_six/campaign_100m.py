"""Integrated qualification contracts for the first serious ~100M 12-6 campaign.

This module owns campaign sequencing and evidence algebra. It does not promote a stage,
approve corpus rights, freeze a tokenizer by itself, or authorize paid compute.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint.core import CheckpointIdentity, verify_checkpoint
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, count_trainable_parameters
from twelve_six.s3_engineering import (
    S3_D11_EXPECTED_PARAMETERS,
    S4_D11_EXPECTED_PARAMETERS,
    S4_D11_MODEL_SHA256,
    kv_cache_bytes,
    s4_d11_model_spec,
)
from twelve_six.training import Trainer, TrainerConfig

CAMPAIGN_SCHEMA = "12-6.campaign47-100m.v1"
S2_EVIDENCE_SCHEMA = "12-6.campaign47-s2-1m-evidence.v1"
S4_PREFLIGHT_SCHEMA = "12-6.campaign47-s4-100m-preflight.v1"
GPU_PILOT_SCHEMA = "12-6.campaign47-gpu-pilot.v1"
QUALIFICATION_SCHEMA = "12-6.campaign47-qualification.v1"

D11_SOURCE_PR = 67
D11_SOURCE_SHA = "2728b762cd998ca99403e84c6e4b33e6ae8ed29b"
S2_EXPECTED_PARAMETERS = 995_552
S2_MODEL_SHA256 = "c029915fb7b0c120c16ca1f30a3b335cdfdb8b252f82aa2fb34efda667fd6f3e"
INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"

_S2_MODEL_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "vocab_size": 2048,
    "max_seq_len": 512,
    "d_model": 128,
    "n_layers": 5,
    "n_heads": 4,
    "n_kv_heads": 1,
    "head_dim": 32,
    "d_ff": 272,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10_000.0,
    "rope_rotary_dim": 32,
    "attention_bias": True,
    "mlp_bias": True,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}

BUDGET_VARIANTS: dict[str, dict[str, Any]] = {
    "eur_2k": {
        "campaign_budget_eur": 2_000.0,
        "accelerator_compute_cap_eur": 1_600.0,
        "non_compute_reserve_eur": 400.0,
        "tokens_per_seed": 2_000_000_000,
        "seeds": [20260825],
    },
    "eur_10k": {
        "campaign_budget_eur": 10_000.0,
        "accelerator_compute_cap_eur": 8_000.0,
        "non_compute_reserve_eur": 2_000.0,
        "tokens_per_seed": 3_000_000_000,
        "seeds": [20260825, 20260826, 20260827],
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _seal(report: dict[str, Any]) -> dict[str, Any]:
    report["report_sha256"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    return report


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


def _require_exact_checkout(repo_root: Path, source_sha: str) -> None:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("source SHA does not match exact checkout")


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


def _rss_kib() -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


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


def s2_d11_model_spec() -> ModelSpec:
    """Return and identity-check the D11 ~1M representative geometry."""

    spec = ModelSpec.from_dict(_S2_MODEL_PAYLOAD)
    if spec.identity_sha256() != S2_MODEL_SHA256:
        raise RuntimeError("D11 S2 ModelSpec identity drifted")
    if spec.parameter_count() != S2_EXPECTED_PARAMETERS:
        raise RuntimeError("D11 S2 parameter count drifted")
    return spec


def _checked_init_spec() -> InitSpec:
    init_spec = InitSpec()
    if init_spec.identity_sha256() != INIT_SHA256:
        raise RuntimeError("campaign InitSpec identity drifted")
    return init_spec


def run_s2_probe(
    *,
    repo_root: Path,
    source_sha: str,
    sequence_length: int = 256,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Execute a real CPU fp32 Trainer update on the ~1M representative model."""

    _require_exact_checkout(repo_root, source_sha)
    spec = s2_d11_model_spec()
    if not 2 <= sequence_length <= spec.max_seq_len:
        raise ValueError("sequence_length must be within the S2 context")

    init_spec = _checked_init_spec()
    torch.manual_seed(seed)
    construct_started = time.perf_counter()
    model = TwelveSixDecoder(spec, init_spec)
    construct_seconds = time.perf_counter() - construct_started
    actual_parameters = count_trainable_parameters(model)
    if actual_parameters != S2_EXPECTED_PARAMETERS:
        raise RuntimeError("instantiated S2 parameter count drifted")

    input_ids = _make_batch(
        batch_size=1,
        sequence_length=sequence_length,
        vocab_size=spec.vocab_size,
        seed=seed + 1,
    )["input_ids"]
    config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=1,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    trainer = Trainer(model, config, device=torch.device("cpu"))
    before = model.final_norm.weight.detach().clone()
    train_started = time.perf_counter()
    metrics = trainer.train_microbatch({"input_ids": input_ids, "labels": input_ids.clone()})
    train_seconds = time.perf_counter() - train_started
    if not metrics.optimizer_stepped or trainer.optimizer_step != 1:
        raise RuntimeError("S2 probe failed to commit exactly one optimizer step")
    if not math.isfinite(metrics.loss) or metrics.grad_norm is None or not math.isfinite(metrics.grad_norm):
        raise RuntimeError("S2 probe produced non-finite training metrics")
    if torch.equal(before, model.final_norm.weight.detach()):
        raise RuntimeError("S2 probe did not change the probed parameter")

    optimized_tokens = trainer.tokens_seen
    report = {
        "schema": S2_EVIDENCE_SCHEMA,
        "source_sha": source_sha,
        "canonical_base": "random_init",
        "paid_compute": False,
        "promotion_authority": False,
        "candidate": {
            "source_pr": D11_SOURCE_PR,
            "source_sha": D11_SOURCE_SHA,
            "model_spec_sha256": spec.identity_sha256(),
            "analytic_parameters": spec.parameter_count(),
            "instantiated_trainable_parameters": actual_parameters,
            "geometry": spec.to_dict(),
        },
        "execution": {
            "device": "cpu",
            "precision": "fp32",
            "sequence_length": sequence_length,
            "optimizer_steps": trainer.optimizer_step,
            "optimized_tokens": optimized_tokens,
            "construct_seconds": construct_seconds,
            "train_step_seconds": train_seconds,
            "measured_optimized_tokens_per_second": (
                optimized_tokens / train_seconds if train_seconds > 0 else None
            ),
            "loss": metrics.loss,
            "grad_norm": metrics.grad_norm,
            "parameter_changed": True,
        },
        "truth_boundary": {
            "data": "deterministic synthetic full-vocabulary mechanics",
            "representative_of_gpu_throughput": False,
            "capability_evidence": False,
        },
    }
    return _seal(report)


def run_s4_construction_preflight(
    *,
    repo_root: Path,
    source_sha: str,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Materialize the exact ~100M model on CPU and measure construction/storage only."""

    _require_exact_checkout(repo_root, source_sha)
    spec = s4_d11_model_spec()
    init_spec = _checked_init_spec()
    before_rss = _rss_kib()
    torch.manual_seed(seed)
    started = time.perf_counter()
    model = TwelveSixDecoder(spec, init_spec)
    construct_seconds = time.perf_counter() - started
    after_rss = _rss_kib()
    actual_parameters = count_trainable_parameters(model)
    if actual_parameters != S4_D11_EXPECTED_PARAMETERS:
        raise RuntimeError("instantiated S4 parameter count drifted")
    parameter_bytes = _parameter_bytes(model)
    if parameter_bytes != S4_D11_EXPECTED_PARAMETERS * 4:
        raise RuntimeError("S4 fp32 parameter tensor bytes drifted")

    report = {
        "schema": S4_PREFLIGHT_SCHEMA,
        "source_sha": source_sha,
        "canonical_base": "random_init",
        "paid_compute": False,
        "promotion_authority": False,
        "candidate": {
            "source_pr": D11_SOURCE_PR,
            "source_sha": D11_SOURCE_SHA,
            "model_spec_sha256": S4_D11_MODEL_SHA256,
            "analytic_parameters": spec.parameter_count(),
            "instantiated_trainable_parameters": actual_parameters,
            "geometry": spec.to_dict(),
        },
        "construction": {
            "device": "cpu",
            "dtype": "fp32_parameters",
            "construct_seconds": construct_seconds,
            "rss_before_kib": before_rss,
            "rss_after_kib": after_rss,
            "rss_delta_kib": (
                max(0, after_rss - before_rss)
                if before_rss is not None and after_rss is not None
                else None
            ),
            "measured_parameter_tensor_bytes": parameter_bytes,
        },
        "memory_algebra_not_runtime_measurement": {
            "bf16_parameter_bytes": S4_D11_EXPECTED_PARAMETERS * 2,
            "fp32_parameter_bytes": S4_D11_EXPECTED_PARAMETERS * 4,
            "fp32_gradient_bytes": S4_D11_EXPECTED_PARAMETERS * 4,
            "two_fp32_adam_moment_bytes": S4_D11_EXPECTED_PARAMETERS * 8,
            "fp32_model_plus_gradient_plus_two_adam_moments_bytes": (
                S4_D11_EXPECTED_PARAMETERS * 16
            ),
            "bf16_kv_cache_bytes_batch1_full_4096_context": kv_cache_bytes(
                spec,
                batch_size=1,
                sequence_length=spec.max_seq_len,
                bytes_per_element=2,
            ),
            "activation_memory": "NOT_MEASURED_UNTIL_GPU_PILOT",
        },
        "truth_boundary": {
            "forward": False,
            "backward": False,
            "optimizer_state_materialized": False,
            "checkpoint_100m_measured": False,
            "gpu_memory_measured": False,
            "distributed_execution": False,
        },
    }
    del model
    gc.collect()
    return _seal(report)


def _pilot_checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    training_config: Mapping[str, Any],
    step: int,
    tokens_seen: int,
    environment_lock_hash: str,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=_sha256_label("CAMPAIGN-47-S4-engineering-tokenizer-interface-v1"),
        tokenizer_vocab_hash=_sha256_label("CAMPAIGN-47-S4-engineering-vocab-32768-v1"),
        dataset_manifest_hash=_sha256_label("CAMPAIGN-47-S4-synthetic-pilot-data-v1"),
        run_manifest_hash=_sha256_label("CAMPAIGN-47-S4-GPU-pilot-v1"),
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
        scheduler={"name": training_config["trainer"]["scheduler"]},
        environment_lock_hash=environment_lock_hash,
    )


def run_s4_gpu_pilot(
    *,
    repo_root: Path,
    source_sha: str,
    checkpoint_root: Path | None,
    provider_label: str,
    hardware_label: str,
    hourly_cost_eur: float,
    rate_evidence: str,
    compute_class: str = "local_free",
    paid_compute_authorized: bool = False,
    batch_size: int = 1,
    sequence_length: int = 2048,
    optimizer_steps: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Execute the exact ~100M candidate on CUDA and retain measured pilot evidence."""

    if compute_class not in {"local_free", "paid"}:
        raise ValueError("compute_class must be local_free or paid")
    if compute_class == "paid" and not paid_compute_authorized:
        raise PermissionError("paid GPU pilot requires explicit paid-compute authorization")
    if hourly_cost_eur <= 0:
        raise ValueError("hourly_cost_eur must be positive for budget projection")
    if batch_size <= 0 or optimizer_steps <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch size, optimizer steps and accumulation must be positive")

    _require_exact_checkout(repo_root, source_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("S4 GPU pilot requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S4 GPU pilot requires CUDA bf16 support; no silent downgrade")

    spec = s4_d11_model_spec()
    if not 2 <= sequence_length <= spec.max_seq_len:
        raise ValueError("sequence_length must be within the S4 context")
    init_spec = _checked_init_spec()
    environment_lock_hash = _lock_identity(repo_root)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(seed)
    construct_started = time.perf_counter()
    model = TwelveSixDecoder(spec, init_spec)
    trainer_config = TrainerConfig(
        learning_rate=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=optimizer_steps,
        warmup_steps=min(max(1, optimizer_steps // 100), optimizer_steps),
        scheduler="cosine",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=1.0,
        precision="bf16",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    trainer = Trainer(model, trainer_config, device=device)
    torch.cuda.synchronize(device)
    construct_seconds = time.perf_counter() - construct_started
    actual_parameters = count_trainable_parameters(model)
    if actual_parameters != S4_D11_EXPECTED_PARAMETERS:
        raise RuntimeError("instantiated S4 GPU pilot parameter count drifted")

    bound_training_config: dict[str, Any] = {
        "schema": "12-6.campaign47-s4-gpu-pilot-training.v1",
        "model_spec_sha256": S4_D11_MODEL_SHA256,
        "init_spec_sha256": init_spec.identity_sha256(),
        "trainer": asdict(trainer_config),
        "data": {
            "kind": "deterministic_synthetic_full_vocab",
            "tokenizer_status": "ENGINEERING_INTERFACE_NOT_FROZEN",
            "sequence_length": sequence_length,
            "batch_size": batch_size,
        },
    }

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="twelve-six-campaign47-s4-")
        checkpoint_root = Path(temporary.name) / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    first_loss: float | None = None
    last_metrics = None
    micro_index = 0
    run_started = time.perf_counter()
    while trainer.optimizer_step < optimizer_steps:
        micro_index += 1
        batch = _make_batch(
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=spec.vocab_size,
            seed=seed + 10_000 + micro_index,
        )
        last_metrics = trainer.train_microbatch(batch)
        if first_loss is None:
            first_loss = last_metrics.loss
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - run_started
    if last_metrics is None or not last_metrics.optimizer_stepped:
        raise RuntimeError("S4 GPU pilot did not commit its final optimizer step")
    if first_loss is None or not math.isfinite(first_loss):
        raise RuntimeError("S4 GPU pilot first loss is non-finite")
    if not math.isfinite(last_metrics.loss):
        raise RuntimeError("S4 GPU pilot final loss is non-finite")
    if last_metrics.grad_norm is None or not math.isfinite(last_metrics.grad_norm):
        raise RuntimeError("S4 GPU pilot gradient norm is non-finite")

    checkpoint_identity = _pilot_checkpoint_identity(
        source_sha=source_sha,
        spec=spec,
        training_config=bound_training_config,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        environment_lock_hash=environment_lock_hash,
    )
    save_started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        checkpoint_root,
        model=model,
        trainer=trainer,
        identity=checkpoint_identity,
    )
    verified_manifest = verify_checkpoint(checkpoint_root)
    checkpoint_save_seconds = time.perf_counter() - save_started
    total_training_and_checkpoint_seconds = time.perf_counter() - run_started
    checkpoint_payload_bytes = sum(
        item["bytes"] for item in verified_manifest["files"].values()
    )
    optimized_tokens = trainer.tokens_seen
    if optimized_tokens <= 0 or total_training_and_checkpoint_seconds <= 0:
        raise RuntimeError("S4 GPU pilot produced no usable throughput evidence")

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    device_name = torch.cuda.get_device_name(device)
    final_step = trainer.optimizer_step
    final_tokens = trainer.tokens_seen

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    restored_model = TwelveSixDecoder(spec, init_spec)
    restored_trainer = Trainer(restored_model, trainer_config, device=device)
    reload_started = time.perf_counter()
    load_result = load_trainer_checkpoint(
        checkpoint_root,
        model=restored_model,
        trainer=restored_trainer,
        expected_git_sha=source_sha,
        expected_model_spec_hash=S4_D11_MODEL_SHA256,
        expected_init_spec_hash=INIT_SHA256,
        expected_tokenizer_hash=checkpoint_identity.tokenizer_hash,
        expected_tokenizer_vocab_hash=checkpoint_identity.tokenizer_vocab_hash,
        expected_dataset_manifest_hash=checkpoint_identity.dataset_manifest_hash,
        expected_run_manifest_hash=checkpoint_identity.run_manifest_hash,
        expected_environment_lock_hash=environment_lock_hash,
        expected_seed=seed,
    )
    torch.cuda.synchronize(device)
    checkpoint_reload_seconds = time.perf_counter() - reload_started
    if restored_trainer.optimizer_step != final_step or restored_trainer.tokens_seen != final_tokens:
        raise RuntimeError("S4 GPU pilot checkpoint reload counter mismatch")

    measured_tps = optimized_tokens / total_training_and_checkpoint_seconds
    report = {
        "schema": GPU_PILOT_SCHEMA,
        "source_sha": source_sha,
        "canonical_base": "random_init",
        "promotion_authority": False,
        "compute_class": compute_class,
        "paid_compute_authorized": paid_compute_authorized,
        "provider_label": provider_label,
        "hardware_label": hardware_label,
        "hourly_cost_eur": hourly_cost_eur,
        "rate_evidence": rate_evidence,
        "candidate": {
            "model_spec_sha256": S4_D11_MODEL_SHA256,
            "analytic_parameters": S4_D11_EXPECTED_PARAMETERS,
            "instantiated_trainable_parameters": actual_parameters,
        },
        "runtime": {
            "device": str(device),
            "device_name": device_name,
            "precision": "bf16_autocast_fp32_parameters",
            "environment_lock_hash": environment_lock_hash,
        },
        "measurement": {
            "candidate_parameters": S4_D11_EXPECTED_PARAMETERS,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer_steps": final_step,
            "optimized_tokens": optimized_tokens,
            "construct_seconds": construct_seconds,
            "training_seconds_excluding_checkpoint": training_seconds,
            "checkpoint_save_and_verify_seconds": checkpoint_save_seconds,
            "checkpoint_reload_seconds": checkpoint_reload_seconds,
            "elapsed_training_and_checkpoint_seconds": total_training_and_checkpoint_seconds,
            "measured_end_to_end_optimized_tokens_per_second": measured_tps,
            "first_loss": first_loss,
            "last_loss": last_metrics.loss,
            "last_grad_norm": last_metrics.grad_norm,
            "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
            "checkpoint_payload_bytes": checkpoint_payload_bytes,
            "checkpoint_id": manifest["checkpoint_id"],
            "restored_checkpoint_id": load_result.manifest["checkpoint_id"],
        },
        "truth_boundary": {
            "100m_throughput_measured": True,
            "projection_requires_100m_pilot_recalibration": False,
            "distributed_execution": False,
            "training_data": "synthetic mechanics only",
            "tokenizer_frozen": False,
            "capability_evidence": False,
            "main_launch": False,
        },
    }

    del restored_trainer, restored_model
    gc.collect()
    torch.cuda.empty_cache()
    if temporary is not None:
        temporary.cleanup()
    return _seal(report)


def wrap_s3_gpu_pilot(
    s3_report: Mapping[str, Any],
    *,
    source_sha: str,
    provider_label: str,
    hardware_label: str,
    hourly_cost_eur: float,
    rate_evidence: str,
) -> dict[str, Any]:
    """Convert real S3 CUDA evidence into a preliminary throughput/cost authority."""

    if hourly_cost_eur <= 0:
        raise ValueError("hourly_cost_eur must be positive")
    if s3_report.get("schema") != "12-6.s3-10m-engineering-evidence.v1":
        raise ValueError("unexpected S3 evidence schema")
    if s3_report.get("source_sha") != source_sha:
        raise ValueError("S3 GPU evidence is not bound to the requested exact source SHA")
    truth = s3_report.get("truth_boundary")
    execution = s3_report.get("execution")
    runtime = s3_report.get("runtime")
    checkpoint = s3_report.get("checkpoint")
    if not isinstance(truth, Mapping) or truth.get("gpu_execution") is not True:
        raise ValueError("GPU pilot requires real CUDA execution")
    if not isinstance(runtime, Mapping) or not str(runtime.get("device", "")).startswith("cuda"):
        raise ValueError("GPU pilot runtime device is not CUDA")
    if not isinstance(execution, Mapping):
        raise ValueError("GPU pilot execution payload missing")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("records"):
        raise ValueError("GPU pilot requires measured checkpoint execution")
    if s3_report.get("candidate", {}).get("analytic_parameters") != S3_D11_EXPECTED_PARAMETERS:
        raise ValueError("GPU pilot must use the exact ~10M representative candidate")
    tokens = int(execution.get("optimized_tokens", 0))
    seconds = float(execution.get("train_backward_update_and_checkpoint_seconds", 0.0))
    if tokens <= 0 or seconds <= 0:
        raise ValueError("GPU pilot has no usable end-to-end throughput measurement")
    loss = float(execution.get("last_loss", math.nan))
    grad_norm = float(execution.get("last_grad_norm", math.nan))
    if not math.isfinite(loss) or not math.isfinite(grad_norm):
        raise ValueError("GPU pilot metrics are non-finite")

    measured_tps = tokens / seconds
    report = {
        "schema": GPU_PILOT_SCHEMA,
        "source_sha": source_sha,
        "canonical_base": "random_init",
        "paid_compute_authorized": False,
        "provider_label": provider_label,
        "hardware_label": hardware_label,
        "hourly_cost_eur": hourly_cost_eur,
        "rate_evidence": rate_evidence,
        "measurement": {
            "source_schema": s3_report["schema"],
            "candidate_parameters": S3_D11_EXPECTED_PARAMETERS,
            "optimized_tokens": tokens,
            "elapsed_training_and_checkpoint_seconds": seconds,
            "measured_end_to_end_optimized_tokens_per_second": measured_tps,
            "checkpoint_records": len(checkpoint["records"]),
            "loss": loss,
            "grad_norm": grad_norm,
        },
        "truth_boundary": {
            "100m_throughput_measured": False,
            "projection_requires_100m_pilot_recalibration": True,
            "distributed_execution": False,
            "provider_rate_is_operator_supplied": True,
        },
    }
    return _seal(report)


def project_budget(pilot: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    """Project token runtime/cost from measured pilot throughput; never invent throughput."""

    if pilot.get("schema") != GPU_PILOT_SCHEMA:
        raise ValueError("budget projection requires campaign GPU pilot evidence")
    if variant_name not in BUDGET_VARIANTS:
        raise ValueError(f"unknown budget variant: {variant_name}")
    measurement = pilot.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("GPU pilot measurement missing")
    tps = float(measurement.get("measured_end_to_end_optimized_tokens_per_second", 0.0))
    hourly_cost = float(pilot.get("hourly_cost_eur", 0.0))
    if tps <= 0 or hourly_cost <= 0:
        raise ValueError("measured throughput and hourly cost must be positive")

    variant = BUDGET_VARIANTS[variant_name]
    total_tokens = int(variant["tokens_per_seed"]) * len(variant["seeds"])
    projected_hours = total_tokens / tps / 3600.0
    projected_compute_cost = projected_hours * hourly_cost
    compute_cap = float(variant["accelerator_compute_cap_eur"])
    measured_100m = (
        pilot.get("truth_boundary", {}).get("100m_throughput_measured") is True
    )
    return {
        "variant": variant_name,
        "source_pilot_sha256": pilot.get("report_sha256"),
        "measured_pilot_tokens_per_second": tps,
        "hourly_cost_eur": hourly_cost,
        "tokens_per_seed": variant["tokens_per_seed"],
        "seeds": list(variant["seeds"]),
        "total_optimized_tokens": total_tokens,
        "projected_accelerator_hours": projected_hours,
        "projected_accelerator_cost_eur": projected_compute_cost,
        "accelerator_compute_cap_eur": compute_cap,
        "within_compute_cap": projected_compute_cost <= compute_cap,
        "campaign_budget_eur": variant["campaign_budget_eur"],
        "non_compute_reserve_eur": variant["non_compute_reserve_eur"],
        "projection_authority": (
            "100M_MEASURED_PILOT"
            if measured_100m
            else "10M_PRELIMINARY_EXTRAPOLATION_RECALIBRATION_REQUIRED"
        ),
    }


def _check_evidence_sha(payload: Mapping[str, Any], source_sha: str, schema: str) -> bool:
    return payload.get("schema") == schema and payload.get("source_sha") == source_sha


def qualify_main_launch(
    *,
    source_sha: str,
    variant_name: str,
    s2_evidence: Mapping[str, Any],
    s3_evidence: Mapping[str, Any],
    s4_preflight: Mapping[str, Any],
    gpu_pilot: Mapping[str, Any],
    tokenizer_freeze: Mapping[str, Any],
    corpus_freeze: Mapping[str, Any],
    paid_compute_authorized: bool,
) -> dict[str, Any]:
    """Evaluate the fail-closed main-launch interlock without launching anything."""

    if variant_name not in BUDGET_VARIANTS:
        raise ValueError(f"unknown budget variant: {variant_name}")
    variant = BUDGET_VARIANTS[variant_name]
    checks: dict[str, bool] = {}
    checks["s2_exact_head_real_update"] = (
        _check_evidence_sha(s2_evidence, source_sha, S2_EVIDENCE_SCHEMA)
        and s2_evidence.get("execution", {}).get("optimizer_steps") == 1
        and s2_evidence.get("execution", {}).get("parameter_changed") is True
    )
    checks["s3_exact_head_real_update_checkpoint"] = (
        _check_evidence_sha(
            s3_evidence,
            source_sha,
            "12-6.s3-10m-engineering-evidence.v1",
        )
        and s3_evidence.get("execution", {}).get("optimizer_steps", 0) >= 1
        and bool(s3_evidence.get("checkpoint", {}).get("records"))
    )
    checks["s4_exact_head_constructed"] = (
        _check_evidence_sha(s4_preflight, source_sha, S4_PREFLIGHT_SCHEMA)
        and s4_preflight.get("candidate", {}).get("instantiated_trainable_parameters")
        == S4_D11_EXPECTED_PARAMETERS
    )
    checks["gpu_pilot_measured"] = _check_evidence_sha(gpu_pilot, source_sha, GPU_PILOT_SCHEMA)

    checks["tokenizer_frozen_32k_bpe"] = (
        tokenizer_freeze.get("schema") == "12-6.tokenizer-freeze.v1"
        and tokenizer_freeze.get("status") == "FROZEN"
        and tokenizer_freeze.get("source_sha") == source_sha
        and tokenizer_freeze.get("algorithm") == "bytelevel_bpe"
        and tokenizer_freeze.get("vocab_size") == 32_768
        and tokenizer_freeze.get("round_trip_pass") is True
        and tokenizer_freeze.get("repeatability_pass") is True
        and tokenizer_freeze.get("heldout_fertility_measured") is True
        and bool(tokenizer_freeze.get("artifact_sha256"))
        and bool(tokenizer_freeze.get("ordered_vocab_sha256"))
    )
    required_train_tokens = int(variant["tokens_per_seed"])
    checks["corpus_frozen_and_eligible"] = (
        corpus_freeze.get("schema") == "12-6.corpus-freeze.v1"
        and corpus_freeze.get("status") == "FROZEN"
        and corpus_freeze.get("source_sha") == source_sha
        and int(corpus_freeze.get("eligible_train_tokens", 0)) >= required_train_tokens
        and corpus_freeze.get("rights_review_complete") is True
        and corpus_freeze.get("contamination_gate_pass") is True
        and corpus_freeze.get("reproducible_build_pass") is True
        and corpus_freeze.get("tokenizer_artifact_sha256")
        == tokenizer_freeze.get("artifact_sha256")
    )

    projection: dict[str, Any] | None = None
    if checks["gpu_pilot_measured"]:
        projection = project_budget(gpu_pilot, variant_name)
    checks["budget_projection_within_cap"] = bool(
        projection is not None and projection["within_compute_cap"]
    )
    checks["100m_pilot_recalibrated"] = (
        gpu_pilot.get("truth_boundary", {}).get("100m_throughput_measured") is True
        and gpu_pilot.get("truth_boundary", {}).get(
            "projection_requires_100m_pilot_recalibration"
        )
        is False
    )
    checks["paid_compute_explicitly_authorized"] = paid_compute_authorized

    technical_ready = all(
        value for key, value in checks.items() if key != "paid_compute_explicitly_authorized"
    )
    launch_ready = technical_ready and paid_compute_authorized
    report = {
        "schema": QUALIFICATION_SCHEMA,
        "source_sha": source_sha,
        "variant": variant_name,
        "checks": checks,
        "budget_projection": projection,
        "technical_ready": technical_ready,
        "paid_compute_authorized": paid_compute_authorized,
        "main_launch_ready": launch_ready,
        "action": (
            "PREPARED_FOR_EXPLICIT_PAYMENT_LAUNCH"
            if launch_ready
            else "BLOCKED_NO_PAYMENT_LAUNCH"
        ),
    }
    return _seal(report)
