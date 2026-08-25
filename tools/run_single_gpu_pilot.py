from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    detect_git_sha,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.single_gpu import (
    SingleDeviceOOMError,
    SingleDeviceStepMetrics,
    SingleDeviceStepRunner,
    build_synthetic_lm_batch,
    greedy_inference_after_training,
    launch_environment,
    model_storage_dtypes,
    optimizer_state_dtypes,
    resolve_single_device,
    seed_before_model_init,
)
from twelve_six.training.trainer import Trainer


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_int(payload: dict[str, Any], key: str, minimum: int) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"pilot.{key} must be an integer >= {minimum}")
    return value


def _validate_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("single-GPU pilot config requires schema_version=1")
    if payload.get("run_kind") != "single_gpu_mechanics_pilot":
        raise ValueError("run_kind must be 'single_gpu_mechanics_pilot'")
    if payload.get("data_authority") != "CONTROLLED_SYNTHETIC_MECHANICS_ONLY":
        raise ValueError("pilot must remain explicitly non-corpus synthetic mechanics evidence")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization mapping is required")
    if authorization.get("provision_compute") is not False:
        raise ValueError("pilot runner must not provision compute")
    if authorization.get("preprovisioned_accelerator_only") is not True:
        raise ValueError("pilot runner is restricted to an already provisioned accelerator")

    pilot = payload.get("pilot")
    if not isinstance(pilot, dict):
        raise ValueError("pilot mapping is required")
    steps = _require_int(pilot, "steps", 2)
    resume_after = _require_int(pilot, "resume_after_step", 1)
    if resume_after >= steps:
        raise ValueError("pilot.resume_after_step must be < pilot.steps")
    _require_int(pilot, "microbatch_size", 1)
    _require_int(pilot, "sequence_length", 2)
    _require_int(pilot, "data_seed", 0)


def _trainer_config(payload: dict[str, Any], precision_override: str | None) -> TrainerConfig:
    raw = payload.get("trainer")
    if not isinstance(raw, dict):
        raise ValueError("trainer mapping is required")
    values = dict(raw)
    if precision_override is not None:
        values["precision"] = precision_override
    betas = values.get("betas")
    if isinstance(betas, list):
        values["betas"] = tuple(betas)
    return TrainerConfig(**values)


def _checkpoint_identity(
    *,
    git_sha: str,
    stage,
    config: TrainerConfig,
    trainer: Trainer,
    run_manifest_hash: str,
    fixture_hash: str,
    token_space_hash: str,
    training_identity: dict[str, Any],
) -> CheckpointIdentity:
    optimizer = {
        "name": "AdamW",
        "learning_rate": config.learning_rate,
        "betas": list(config.betas),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
    }
    scheduler = None
    if config.scheduler != "constant" or config.warmup_steps:
        scheduler = {
            "name": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
        }
    return CheckpointIdentity(
        git_sha=git_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=token_space_hash,
        tokenizer_vocab_hash=token_space_hash,
        dataset_manifest_hash=fixture_hash,
        run_manifest_hash=run_manifest_hash,
        training_config=training_identity,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_metric(path: Path, metric: SingleDeviceStepMetrics) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")


def _checkpoint_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _fresh_model_and_trainer(stage, config: TrainerConfig, device: torch.device):
    seed_before_model_init(config.seed, device)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device=device)
    return model, trainer


def _release_cuda_objects(*objects: Any, device: torch.device) -> None:
    del objects
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _validate_config(payload)
    stage_path = Path(str(payload["stage_config"]))
    stage = load_stage_config(stage_path)
    config = _trainer_config(payload, args.precision)
    pilot = payload["pilot"]

    device_request = args.device or str(payload.get("device", "cuda:0"))
    if args.allow_cpu_smoke and not torch.cuda.is_available():
        device_request = "cpu"
    device, device_runtime = resolve_single_device(
        device_request,
        allow_cpu_fallback=args.allow_cpu_smoke,
        require_single_visible_cuda=True,
    )

    sequence_length = int(pilot["sequence_length"])
    if sequence_length > stage.model.max_seq_len:
        raise ValueError(
            f"pilot sequence_length {sequence_length} exceeds stage max {stage.model.max_seq_len}"
        )
    if config.gradient_accumulation_steps != 1:
        raise ValueError("mechanics pilot currently requires gradient_accumulation_steps=1")
    if config.max_steps != int(pilot["steps"]):
        raise ValueError("trainer.max_steps must equal pilot.steps")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir or payload.get("output_root", "runs/train15-single-gpu"))
    output_dir = output_root / f"{stage.stage.lower()}-{config.precision}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "step_metrics.jsonl"

    git_sha = detect_git_sha()
    if git_sha is None:
        raise RuntimeError("pilot requires execution from an exact Git checkout")

    effective_manifest = {
        **payload,
        "device": str(device),
        "trainer": asdict(config),
        "git_sha": git_sha,
    }
    run_manifest_hash = hash_json(effective_manifest)
    fixture = {
        "kind": "uniform_random_integer_token_ids",
        "authority": "CONTROLLED_SYNTHETIC_MECHANICS_ONLY",
        "vocab_size": stage.model.vocab_size,
        "batch_size": int(pilot["microbatch_size"]),
        "sequence_length": sequence_length,
        "data_seed": int(pilot["data_seed"]),
        "generator": "torch.Generator(cpu)",
    }
    fixture_hash = hash_json(fixture)
    token_space = {
        "kind": "synthetic_integer_id_space_not_a_tokenizer",
        "vocab_size": stage.model.vocab_size,
    }
    token_space_hash = hash_json(token_space)
    training_identity = {
        "stage": stage.stage,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "fixture_sha256": fixture_hash,
        "data": {
            "split_identity": "train15_controlled_synthetic",
            "packing_sha256": hash_json(
                {"kind": "unpacked_fixed_length_lm_batch", "sequence_length": sequence_length}
            ),
            "packing_version": "train15-fixture-v1",
        },
        "trainer": asdict(config),
        "single_device": {
            "require_single_visible_cuda": True,
            "non_blocking_transfer": True,
            "synchronize_for_metrics": True,
        },
    }
    training_config_hash = hash_json(training_identity)
    _write_json(output_dir / "effective_run_manifest.json", effective_manifest)
    _write_json(output_dir / "fixture_identity.json", fixture)

    model, trainer = _fresh_model_and_trainer(stage, config, device)
    if model_storage_dtypes(model) != ["torch.float32"]:
        raise RuntimeError("AMP pilot requires fp32 model/master-weight storage")
    runner = SingleDeviceStepRunner(
        trainer,
        non_blocking_transfer=True,
        synchronize_for_metrics=True,
    )

    step_metrics: list[SingleDeviceStepMetrics] = []
    resume_after = int(pilot["resume_after_step"])
    checkpoint_dirs: list[Path] = []

    def run_one_step(step_index: int) -> None:
        batch = build_synthetic_lm_batch(
            vocab_size=stage.model.vocab_size,
            batch_size=int(pilot["microbatch_size"]),
            sequence_length=sequence_length,
            seed=int(pilot["data_seed"]) + step_index,
            pin_memory=device.type == "cuda",
        )
        measured = runner.train_microbatch(batch)
        step_metrics.append(measured)
        _append_metric(metrics_path, measured)

    for step_index in range(1, resume_after + 1):
        run_one_step(step_index)

    midpoint = output_dir / f"checkpoint-step-{trainer.optimizer_step:06d}"
    identity = _checkpoint_identity(
        git_sha=git_sha,
        stage=stage,
        config=config,
        trainer=trainer,
        run_manifest_hash=run_manifest_hash,
        fixture_hash=fixture_hash,
        token_space_hash=token_space_hash,
        training_identity=training_identity,
    )
    save_trainer_checkpoint(midpoint, model=model, trainer=trainer, identity=identity)
    checkpoint_dirs.append(midpoint)

    del runner, trainer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, trainer = _fresh_model_and_trainer(stage, config, device)
    load_trainer_checkpoint(
        midpoint,
        model=model,
        trainer=trainer,
        restore_rng=True,
        expected_git_sha=git_sha,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_init_spec_hash=stage.init.identity_sha256(),
        expected_tokenizer_hash=token_space_hash,
        expected_tokenizer_vocab_hash=token_space_hash,
        expected_dataset_manifest_hash=fixture_hash,
        expected_run_manifest_hash=run_manifest_hash,
        expected_training_config_hash=training_config_hash,
        expected_seed=config.seed,
    )
    runner = SingleDeviceStepRunner(
        trainer,
        non_blocking_transfer=True,
        synchronize_for_metrics=True,
    )
    for step_index in range(resume_after + 1, config.max_steps + 1):
        run_one_step(step_index)

    final_checkpoint = output_dir / f"checkpoint-step-{trainer.optimizer_step:06d}"
    final_identity = _checkpoint_identity(
        git_sha=git_sha,
        stage=stage,
        config=config,
        trainer=trainer,
        run_manifest_hash=run_manifest_hash,
        fixture_hash=fixture_hash,
        token_space_hash=token_space_hash,
        training_identity=training_identity,
    )
    save_trainer_checkpoint(
        final_checkpoint,
        model=model,
        trainer=trainer,
        identity=final_identity,
    )
    checkpoint_dirs.append(final_checkpoint)

    del runner, trainer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, trainer = _fresh_model_and_trainer(stage, config, device)
    load_trainer_checkpoint(
        final_checkpoint,
        model=model,
        trainer=trainer,
        restore_rng=True,
        expected_git_sha=git_sha,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_init_spec_hash=stage.init.identity_sha256(),
        expected_tokenizer_hash=token_space_hash,
        expected_tokenizer_vocab_hash=token_space_hash,
        expected_dataset_manifest_hash=fixture_hash,
        expected_run_manifest_hash=run_manifest_hash,
        expected_training_config_hash=training_config_hash,
        expected_seed=config.seed,
    )
    prompt = torch.arange(1, 9, dtype=torch.long).remainder(stage.model.vocab_size).unsqueeze(0)
    generated = greedy_inference_after_training(
        model,
        prompt,
        device=device,
        max_new_tokens=int(pilot.get("inference_new_tokens", 4)),
    )

    total_tokens = sum(item.trainer.tokens for item in step_metrics)
    total_train_seconds = sum(item.train_seconds for item in step_metrics)
    peak_allocated = max(
        (item.cuda_peak_allocated_bytes or 0 for item in step_metrics),
        default=0,
    )
    peak_reserved = max(
        (item.cuda_peak_reserved_bytes or 0 for item in step_metrics),
        default=0,
    )
    max_rss = max((item.process_rss_bytes or 0 for item in step_metrics), default=0)

    summary = {
        "schema_version": 1,
        "status": "PASS_MECHANICS_PILOT",
        "cuda_execution": device.type == "cuda",
        "cuda_evidence": "TESTED" if device.type == "cuda" else "NOT_TESTED_CPU_SMOKE_ONLY",
        "stage": stage.stage,
        "parameter_count": stage.model.parameter_count(),
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "git_sha": git_sha,
        "device": device_runtime.to_dict(),
        "launch_environment": launch_environment(),
        "precision_runtime": trainer.precision_runtime.to_dict(),
        "model_storage_dtypes": model_storage_dtypes(model),
        "optimizer_state_dtypes": optimizer_state_dtypes(trainer),
        "optimizer_steps": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "measured_tokens": total_tokens,
        "measured_train_seconds": total_train_seconds,
        "tokens_per_second": (
            total_tokens / total_train_seconds if total_train_seconds > 0 else None
        ),
        "cuda_peak_allocated_bytes": peak_allocated if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": peak_reserved if device.type == "cuda" else None,
        "process_peak_rss_bytes": max_rss or None,
        "checkpoint_bytes": {
            path.name: _checkpoint_bytes(path) for path in checkpoint_dirs
        },
        "resume_checkpoint_step": resume_after,
        "final_checkpoint_step": trainer.optimizer_step,
        "inference": {
            "mode": "greedy",
            "autocast": False,
            "dtype_semantics": "model_storage_dtype_fp32_no_autocast",
            "prompt_ids": prompt.tolist(),
            "generated_ids": generated.tolist(),
        },
        "truth_boundary": {
            "data": "controlled synthetic token IDs; not corpus/stage-quality evidence",
            "precision": "uses D02 resolved AMP policy; no cross-precision equality claim",
            "determinism": (
                "seeded before model construction; exact resume additionally restores Python, "
                "NumPy, Torch CPU and visible CUDA RNG state from D05 checkpoint"
            ),
            "oom": (
                "no blind in-memory retry; restore verified checkpoint into fresh objects and "
                "reduce microbatch/sequence length"
            ),
            "paid_compute": "runner provisions no compute and purchases no resources",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 12-6 TRAIN-15 single-GPU mechanics pilot")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runs/s3_10m.single_gpu_pilot.experimental.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="Override config device, e.g. cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Allow CPU mechanics smoke only; never counts as CUDA evidence",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = _load_json(args.config)
    try:
        summary = _run(payload, args)
    except SingleDeviceOOMError as exc:
        print(f"TRAIN-15 OOM: {exc}")
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
