#!/usr/bin/env python3
"""SCALE-202: one trustworthy real ~100M training-step qualification.

This is deliberately not a campaign launcher. It executes one DATA-25/s0-byte-v1
training transition only when an already-visible, authorized CUDA device has
conservative free-memory headroom. Otherwise it records the blocker and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    detect_git_sha,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.distributed.activation_checkpointing import apply_activation_checkpointing
from twelve_six.milestone100_first_learned import (
    EXPECTED_CORPUS_ID,
    RETAINED_CORPUS_MANIFEST,
    _build_corpus,
    _packed,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.single_gpu import SingleDeviceStepRunner, seed_before_model_init
from twelve_six.training.trainer import Trainer

WORKER_ID = "SCALE-202-100M-REAL-QUALIFICATION"
SCHEMA = "12-6.scale202-100m-real-qualification.v1"
STAGE_CONFIG = Path("configs/stages/s4_100m_accelerator.candidate.json")
EXPECTED_PARAMETER_COUNT = 99_897_600
EXPECTED_MODEL_SPEC_SHA256 = "6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170"
EXPECTED_INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
EXPECTED_TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
SEED = 202
SEQUENCE_LENGTH = 128
BATCH_SIZE = 1
# COMPUTE-99 retained the incumbent SCALE-04 first-order BF16 estimate at
# 4,215,510,401 bytes. Qualification requires 1.25x headroom before even trying.
PRIOR_ESTIMATED_BYTES = 4_215_510_401
HEADROOM_FACTOR = 1.25
MIN_FREE_BYTES = math.ceil(PRIOR_ESTIMATED_BYTES * HEADROOM_FACTOR)
EIGHT_GIB = 8 * 1024**3


class QualificationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return None
    return int(value * (1 if sys.platform == "darwin" else 1024))


def _native_bf16_supported(index: int) -> bool:
    try:
        with torch.cuda.device(index):
            try:
                return bool(torch.cuda.is_bf16_supported(including_emulation=False))
            except TypeError:
                return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def hardware_snapshot() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: list[dict[str, Any]] = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "capability": list(torch.cuda.get_device_capability(index)),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "native_bf16_supported": _native_bf16_supported(index),
            }
        )
    nvidia_smi = shutil.which("nvidia-smi")
    smi_summary: str | None = None
    if nvidia_smi:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        smi_summary = completed.stdout.strip() if completed.returncode == 0 else None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "visible_cuda_devices": count,
        "devices": devices,
        "nvidia_smi_present": nvidia_smi is not None,
        "nvidia_smi_summary": smi_summary,
        "process_peak_rss_bytes": _rss_bytes(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def contract_snapshot(repo: Path) -> dict[str, Any]:
    stage = load_stage_config(repo / STAGE_CONFIG)
    tok = ByteTokenizer()
    retained = _read_json(repo / RETAINED_CORPUS_MANIFEST)
    if stage.model.parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise QualificationError("100M parameter count drift")
    if stage.expected_parameters != EXPECTED_PARAMETER_COUNT:
        raise QualificationError("100M stage expected_parameters drift")
    if stage.model.identity_sha256() != EXPECTED_MODEL_SPEC_SHA256:
        raise QualificationError("100M ModelSpec identity drift")
    if stage.init.identity_sha256() != EXPECTED_INIT_SPEC_SHA256:
        raise QualificationError("100M InitSpec identity drift")
    if tok.identity.config_sha256 != EXPECTED_TOKENIZER_CONFIG_SHA256:
        raise QualificationError("tokenizer config identity drift")
    if tok.identity.vocab_sha256 != EXPECTED_TOKENIZER_VOCAB_SHA256:
        raise QualificationError("tokenizer vocab identity drift")
    if retained.get("corpus_identity_sha256") != EXPECTED_CORPUS_ID:
        raise QualificationError("DATA-25 corpus identity drift")
    if retained.get("train_validation_content_overlap") != 0:
        raise QualificationError("DATA-25 validation leakage")
    return {
        "stage_config": STAGE_CONFIG.as_posix(),
        "parameter_count": stage.model.parameter_count(),
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
        },
        "corpus_identity_sha256": retained["corpus_identity_sha256"],
        "train_validation_content_overlap": retained["train_validation_content_overlap"],
    }


def execution_decision(snapshot: dict[str, Any]) -> dict[str, Any]:
    devices = list(snapshot.get("devices") or [])
    if not snapshot.get("cuda_available") or not devices:
        return {
            "mode": "none",
            "status": "NOT_RUN_NO_GPU",
            "reason": "no CUDA hardware is visible; accelerator execution is not simulated",
        }
    bf16_devices = [item for item in devices if item.get("native_bf16_supported")]
    if not bf16_devices:
        return {
            "mode": "none",
            "status": "NOT_RUN_NO_NATIVE_BF16",
            "reason": "the current 100M stage prefers BF16 and no visible GPU proves native BF16",
        }
    adequate = [item for item in bf16_devices if int(item["free_bytes"]) >= MIN_FREE_BYTES]
    if adequate:
        selected = max(adequate, key=lambda item: int(item["free_bytes"]))
        return {
            "mode": "single_gpu",
            "status": "READY_SINGLE_GPU",
            "device_index": int(selected["index"]),
            "free_bytes": int(selected["free_bytes"]),
            "required_free_bytes": MIN_FREE_BYTES,
        }
    if len(bf16_devices) >= 2:
        return {
            "mode": "fsdp2_candidate",
            "status": "NOT_RUN_SINGLE_GPU_HEADROOM_FSDP2_CANDIDATE",
            "reason": (
                "multiple native-BF16 GPUs are visible but none individually clears the conservative "
                "single-GPU headroom gate; the accepted FSDP2+DCP path is the only allowed fallback"
            ),
            "aggregate_free_bytes": sum(int(item["free_bytes"]) for item in bf16_devices),
            "required_single_gpu_free_bytes": MIN_FREE_BYTES,
        }
    return {
        "mode": "none",
        "status": "NOT_RUN_INSUFFICIENT_SINGLE_GPU_HEADROOM",
        "reason": "the only eligible GPU does not clear the conservative measured/estimated headroom gate",
        "best_free_bytes": max(int(item["free_bytes"]) for item in bf16_devices),
        "required_free_bytes": MIN_FREE_BYTES,
    }


def _bootstrap_identity() -> str | None:
    path = os.environ.get("SCALE202_BOOTSTRAP_MANIFEST")
    if not path:
        return None
    manifest = _read_json(Path(path))
    value = manifest.get("identity_sha256")
    return value if isinstance(value, str) and len(value) == 64 else None


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().float().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _first_batch(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, split: str) -> dict[str, torch.Tensor]:
    example = next(_packed(corpus, manifest, tok, split, "uk"))
    return {
        "input_ids": torch.tensor([example.input_ids], dtype=torch.long),
        "labels": torch.tensor([example.labels], dtype=torch.long),
    }


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=1,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="bf16",
        seed=SEED,
        deterministic_algorithms=False,
        deterministic_warn_only=True,
    )


def _make_model(stage, device: torch.device, policy: str) -> TwelveSixDecoder:
    seed_before_model_init(SEED, device)
    model = TwelveSixDecoder(stage.model, stage.init)
    apply_activation_checkpointing(model, policy)
    return model.to(device)


def _qualification_run_manifest(source_sha: str, contract: dict[str, Any], policy: str) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "authority": "ONE_STEP_QUALIFICATION_ONLY_NOT_CAMPAIGN",
        "source_sha": source_sha,
        "model_spec_sha256": contract["model_spec_sha256"],
        "init_spec_sha256": contract["init_spec_sha256"],
        "parameter_count": contract["parameter_count"],
        "tokenizer": contract["tokenizer"],
        "corpus_identity_sha256": contract["corpus_identity_sha256"],
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "train_stratum": "uk",
        "validation_probe_stratum": "uk",
        "precision": "bf16",
        "activation_checkpointing": policy,
        "steps": 1,
        "paid_compute": False,
    }
    payload["identity_sha256"] = hash_json(payload)
    return payload


def _checkpoint_identity(
    *,
    source_sha: str,
    stage,
    tok: ByteTokenizer,
    corpus_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    config: TrainerConfig,
    trainer: Trainer,
    bootstrap_sha: str | None,
    policy: str,
) -> CheckpointIdentity:
    training_config = {
        "trainer": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": list(config.betas),
            "eps": config.eps,
            "max_steps": config.max_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clip_norm": config.gradient_clip_norm,
            "precision": config.precision,
            "seed": config.seed,
        },
        "init_spec_sha256": stage.init.identity_sha256(),
        "data": {
            "tokenizer_version": tok.identity.version,
            "packing_version": "document-isolated-v1",
            "packing_sequence_length": SEQUENCE_LENGTH,
            "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"],
            "train_stratum": "uk",
        },
        "activation_checkpointing": policy,
        "authority": "ONE_STEP_QUALIFICATION_ONLY_NOT_CAMPAIGN",
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256,
        tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=corpus_manifest["corpus_identity_sha256"],
        run_manifest_hash=run_manifest["identity_sha256"],
        training_config=training_config,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=bootstrap_sha,
    )


def run_single_gpu(repo: Path, output_dir: Path, device_index: int, contract: dict[str, Any], free_bytes: int) -> dict[str, Any]:
    if not _native_bf16_supported(device_index):
        raise QualificationError("selected device lost native BF16 support after preflight")
    device = torch.device("cuda", device_index)
    source_sha = detect_git_sha(repo)
    if not source_sha:
        raise QualificationError("exact git SHA is required")
    stage = load_stage_config(repo / STAGE_CONFIG)
    tok = ByteTokenizer()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_manifest = _build_corpus(repo, output_dir / "data-build")
    corpus = output_dir / "data-build" / "corpus-a"
    if corpus_manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise QualificationError("rebuilt DATA-25 identity mismatch")

    # SCALE-143 corrected the old unconditional checkpointing rule. For this
    # one-step qualification we use no checkpointing with ample headroom and a
    # conservative per-block policy on smaller but still prequalified devices.
    policy = "none" if free_bytes >= EIGHT_GIB else "per_block"
    run_manifest = _qualification_run_manifest(source_sha, contract, policy)
    (output_dir / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bootstrap_sha = _bootstrap_identity()

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_bytes()
    model = _make_model(stage, device, policy)
    config = _trainer_config()
    trainer = Trainer(model, config, device=device)
    runner = SingleDeviceStepRunner(trainer, synchronize_for_metrics=True)

    sentinel_name, sentinel_parameter = next(iter(model.named_parameters()))
    sentinel_before = sentinel_parameter.detach().cpu().clone()
    sentinel_before_sha = _tensor_sha256(sentinel_before)
    batch = _first_batch(corpus, corpus_manifest, tok, "train")
    step = runner.train_microbatch(batch)
    if not step.trainer.optimizer_stepped or step.trainer.optimizer_step != 1:
        raise QualificationError("optimizer did not execute exactly one update")
    if step.trainer.grad_norm is None or not math.isfinite(step.trainer.grad_norm):
        raise QualificationError("finite gradient proof missing")
    if step.trainer.grad_norm <= 0:
        raise QualificationError("gradient norm is zero")

    sentinel_after = dict(model.named_parameters())[sentinel_name].detach().cpu()
    update_l1 = float((sentinel_after.float() - sentinel_before.float()).abs().sum().item())
    changed_values = int(torch.ne(sentinel_after, sentinel_before).sum().item())
    if update_l1 <= 0.0 or changed_values <= 0:
        raise QualificationError("optimizer step did not move the sentinel weight tensor")
    sentinel_after_sha = _tensor_sha256(sentinel_after)

    identity = _checkpoint_identity(
        source_sha=source_sha,
        stage=stage,
        tok=tok,
        corpus_manifest=corpus_manifest,
        run_manifest=run_manifest,
        config=config,
        trainer=trainer,
        bootstrap_sha=bootstrap_sha,
        policy=policy,
    )
    checkpoint = output_dir / "checkpoint-step-0001"
    saved = save_trainer_checkpoint(checkpoint, model=model, trainer=trainer, identity=identity)
    verified = verify_checkpoint(checkpoint)

    # Fresh objects only: reload must not reuse the in-memory post-step model.
    del runner, trainer, model
    torch.cuda.empty_cache()
    fresh_model = _make_model(stage, device, policy)
    fresh_trainer = Trainer(fresh_model, config, device=device)
    load_result = load_trainer_checkpoint(
        checkpoint,
        model=fresh_model,
        trainer=fresh_trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_init_spec_hash=stage.init.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,
        expected_dataset_manifest_hash=corpus_manifest["corpus_identity_sha256"],
        expected_run_manifest_hash=run_manifest["identity_sha256"],
        expected_environment_lock_hash=bootstrap_sha,
        expected_seed=SEED,
    )
    if fresh_trainer.optimizer_step != 1:
        raise QualificationError("fresh checkpoint reload restored the wrong optimizer step")

    validation = _first_batch(corpus, corpus_manifest, tok, "validation")
    ids = validation["input_ids"].to(device)
    labels = validation["labels"].to(device)
    fresh_model.eval()
    sentinel_reload_before = _tensor_sha256(dict(fresh_model.named_parameters())[sentinel_name])
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast:
        logits = fresh_model(ids).logits
        validation_loss = causal_lm_loss(logits, labels)
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(validation_loss).item()):
        raise QualificationError("post-reload held-out probe is non-finite")
    logits_fingerprint = hashlib.sha256(
        logits[0, :2, :32].detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    sentinel_reload_after = _tensor_sha256(dict(fresh_model.named_parameters())[sentinel_name])
    if sentinel_reload_before != sentinel_reload_after:
        raise QualificationError("post-reload evaluation mutated the model sentinel")

    return {
        "status": "PASS_REAL_100M_ONE_STEP",
        "authority": "ONE_STEP_QUALIFICATION_ONLY_NOT_CAMPAIGN",
        "execution_mode": "single_gpu",
        "source_sha": source_sha,
        "device_index": device_index,
        "precision": "bf16",
        "activation_checkpointing": policy,
        "train": {
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "loss": step.trainer.loss,
            "grad_norm": step.trainer.grad_norm,
            "finite_gradient_check": True,
            "optimizer_stepped": step.trainer.optimizer_stepped,
            "optimizer_step": step.trainer.optimizer_step,
            "tokens": step.trainer.tokens,
            "tokens_per_second": step.tokens_per_second,
            "timing_authority": step.timing_authority,
        },
        "weights_moved": {
            "parameter": sentinel_name,
            "before_sha256": sentinel_before_sha,
            "after_sha256": sentinel_after_sha,
            "changed_values": changed_values,
            "update_l1": update_l1,
        },
        "memory": {
            "rss_before_model_bytes": rss_before,
            "rss_peak_after_step_bytes": step.process_rss_bytes,
            "cuda_allocated_after_step_bytes": step.cuda_memory_allocated_bytes,
            "cuda_reserved_after_step_bytes": step.cuda_memory_reserved_bytes,
            "cuda_peak_allocated_bytes": step.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": step.cuda_peak_reserved_bytes,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "checkpoint_id": saved.get("checkpoint_id"),
            "verified_checkpoint_id": verified.get("checkpoint_id"),
            "reload_checkpoint_id": load_result.manifest.get("checkpoint_id"),
            "fresh_object_reload": True,
            "restored_optimizer_step": fresh_trainer.optimizer_step,
            "restored_tokens_seen": fresh_trainer.tokens_seen,
        },
        "post_reload_probe": {
            "split": "validation",
            "stratum": "uk",
            "loss": float(validation_loss.detach().float().cpu().item()),
            "finite": True,
            "logits_sha256": logits_fingerprint,
            "non_mutation_sentinel_sha256": sentinel_reload_after,
        },
        "bootstrap_identity_sha256": bootstrap_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    work = args.work_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    contract = contract_snapshot(repo)
    hardware = hardware_snapshot()
    decision = execution_decision(hardware)
    base: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "contract": contract,
        "hardware_preflight": hardware,
        "decision": decision,
        "memory_gate": {
            "prior_estimated_bytes": PRIOR_ESTIMATED_BYTES,
            "headroom_factor": HEADROOM_FACTOR,
            "required_free_bytes": MIN_FREE_BYTES,
            "estimate_authority": "COMPUTE-99/SCALE-04 planning estimate; target VRAM remains unmeasured until a real CUDA run",
        },
        "universal_bootstrap_identity_sha256": _bootstrap_identity(),
        "paid_compute": False,
    }

    if decision["mode"] != "single_gpu":
        # The only current non-single fallback allowed by the mission is the
        # already-accepted FSDP2 path when multiple real GPUs exist. Do not fake
        # that execution in a one-process CPU/no-GPU environment.
        base["status"] = decision["status"]
        if decision["mode"] == "fsdp2_candidate":
            base["fsdp2"] = {
                "eligible_only_if_multiple_free_gpus_genuinely_present": True,
                "accepted_runtime_surface": "src/twelve_six/distributed/fsdp2_training.py",
                "accepted_checkpoint_surface": "src/twelve_six/distributed/dcp_checkpoint.py",
                "execution": "NOT_RUN_IN_THIS_PROCESS",
                "reason": "single-GPU headroom failed; an actual torchrun/NCCL invocation is required rather than simulating FSDP2",
            }
        output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": base["status"], "output": str(output)}, sort_keys=True))
        return 0

    try:
        result = run_single_gpu(
            repo,
            work,
            int(decision["device_index"]),
            contract,
            int(decision["free_bytes"]),
        )
        base.update(result)
    except torch.cuda.OutOfMemoryError as exc:
        base["status"] = "NOT_RUN_MEASURED_CUDA_OOM"
        base["blocker"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "process_peak_rss_bytes": _rss_bytes(),
            "retry_same_in_memory_state": False,
        }
    except Exception as exc:
        base["status"] = "FAILED_QUALIFICATION"
        base["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": base["status"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
