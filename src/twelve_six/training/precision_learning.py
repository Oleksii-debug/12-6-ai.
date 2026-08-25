"""Paired CPU precision learning-curve evidence for the S1 ~100K engineering model.

This is experimental evidence, not stage promotion. It deliberately reuses the
controlled S0 fixture that the S1 numerical preflight used, while extending the
training horizon and observability. Evaluation and reload inference are FP32 so
cross-precision comparisons measure learned FP32 master weights rather than an
evaluation autocast policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import (
    batch_examples,
    collate_rows,
    iter_packed_examples,
    load_jsonl_records,
)
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .loss import causal_lm_loss
from .s0_evidence_contract import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    INIT_SPEC_SHA256,
    PACKING_CONFIG_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .trainer import Trainer

SCHEMA_VERSION = "12-6.train60-precision-learning.v1"
AUTHORITY = "ENGINEERING_PRECISION_LEARNING_EVIDENCE_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
S1_MODEL_SPEC_SHA256 = "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6"
S1_PARAMETER_COUNT = 107_856
S1_MODEL_VOCAB = 512
FIXTURE_TOKENIZER_VOCAB = 256
DEFAULT_TARGET_TOKENS = 100_000

TOLERANCES = {
    "final_train_bpb_abs": 0.08,
    "final_validation_bpb_abs": 0.08,
    "max_curve_validation_bpb_abs": 0.12,
    "gradient_norm_median_relative": 0.10,
    "gradient_norm_p95_relative": 0.25,
    "update_norm_median_relative": 0.10,
    "update_norm_p95_relative": 0.25,
    "same_precision_reload_logits_max_abs": 1e-6,
}


class PrecisionLearningError(RuntimeError):
    """Raised when the paired precision experiment cannot produce valid evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionLearningError(message)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_to_bpb(loss: float) -> float:
    """Convert natural-log cross entropy to bits per byte/token for byte-token data."""
    return float(loss) / math.log(2.0)


def _tensor_batches(
    root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], tuple[str, ...], int, int]:
    records = tuple(load_jsonl_records(root / f"data/s0/packaged/{split}.jsonl", split=split))
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    _require(bool(examples), f"controlled fixture split {split} produced no packed examples")

    batches: list[dict[str, torch.Tensor]] = []
    max_token_id = -1
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        max_token_id = max(max_token_id, int(input_ids.max().item()))
        batches.append({"input_ids": input_ids, "labels": labels})

    return (
        batches,
        tuple(record.record_id for record in records),
        sum(example.num_loss_tokens for example in examples),
        max_token_id,
    )


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, float | int]:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        tokens = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        _require(torch.isfinite(loss).item(), "FP32 evaluation produced non-finite loss")
        weighted_loss += float(loss.item()) * tokens
        token_count += tokens
    _require(token_count > 0, "evaluation split contains zero scoreable tokens")
    loss_value = weighted_loss / token_count
    return {"loss": loss_value, "bpb": loss_to_bpb(loss_value), "tokens": token_count}


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _update_metrics(
    model: TwelveSixDecoder,
    before: Mapping[str, torch.Tensor],
) -> tuple[float, float]:
    squared = 0.0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
    return math.sqrt(squared), max_abs


def _all_training_state_finite(model: TwelveSixDecoder, trainer: Trainer) -> bool:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            return False
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                return False
    return True


def _cpu_bf16_runtime_probe() -> dict[str, Any]:
    capability = None
    if hasattr(torch.backends, "cpu") and hasattr(torch.backends.cpu, "get_cpu_capability"):
        capability = str(torch.backends.cpu.get_cpu_capability())
    try:
        left = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        right = torch.eye(8, dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = left @ right
    except Exception as exc:
        raise PrecisionLearningError("CPU bf16 autocast execution is unavailable") from exc
    _require(output.dtype == torch.bfloat16, "CPU bf16 autocast did not produce bf16 matmul output")
    _require(torch.isfinite(output).all().item(), "CPU bf16 autocast smoke produced non-finite output")
    return {
        "status": "PASS",
        "output_dtype": str(output.dtype),
        "torch_cpu_capability": capability,
        "native_bf16_isa_claimed": False,
        "note": "runtime execution proven; private ISA probes are not treated as public hardware truth",
    }


def _fp16_cpu_failure_probe(stage_path: Path, *, seed: int) -> dict[str, str]:
    stage = load_stage_config(stage_path)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    try:
        Trainer(model, TrainerConfig(max_steps=1, precision="fp16", seed=seed), device="cpu")
    except ValueError as exc:
        _require("fp16 training requires a CUDA device" in str(exc), "unexpected fp16 CPU failure")
        return {"status": "FAIL_CLOSED_AS_DESIGNED", "exception": type(exc).__name__}
    raise PrecisionLearningError("fp16 CPU unexpectedly became trainable")


def _milestones(target_tokens: int) -> tuple[int, ...]:
    return tuple(sorted({0, target_tokens // 4, target_tokens // 2, 3 * target_tokens // 4, target_tokens}))


def _checkpoint_and_reload(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    config: TrainerConfig,
    stage: Any,
    validation_batches: list[dict[str, torch.Tensor]],
    precision: str,
    source_sha: str,
    seed: int,
    target_tokens: int,
    environment_lock_hash: str,
    paired_identity_sha256: str,
) -> dict[str, Any]:
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "paired_identity_sha256": paired_identity_sha256,
        "precision": precision,
        "source_sha": source_sha,
        "seed": seed,
        "requested_target_tokens": target_tokens,
        "actual_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "fixture_purpose": "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER",
    }
    run_manifest_hash = hash_json(run_identity)
    identity = CheckpointIdentity(
        git_sha=source_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=TOKENIZER_CONFIG_SHA256,
        tokenizer_vocab_hash=TOKENIZER_VOCAB_SHA256,
        dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        run_manifest_hash=run_manifest_hash,
        training_config={
            "experiment": "TRAIN-60-PRECISION-LEARNED",
            "paired_identity_sha256": paired_identity_sha256,
            "trainer": asdict(config),
            "fixture_compatibility_only": True,
        },
        seed=seed,
        precision=precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "learning_rate": config.learning_rate},
        scheduler=None,
        environment_lock_hash=environment_lock_hash,
    )

    model.eval()
    inference_input = validation_batches[0]["input_ids"]
    with torch.no_grad():
        pre_reload_logits = model(inference_input).logits.detach().clone()

    with TemporaryDirectory(prefix=f"train60-{precision}-") as directory:
        manifest = save_trainer_checkpoint(
            directory,
            model=model,
            trainer=trainer,
            identity=identity,
        )
        torch.manual_seed(seed)
        reloaded_model = TwelveSixDecoder(stage.model, stage.init)
        reloaded_trainer = Trainer(reloaded_model, config, device="cpu")
        load_trainer_checkpoint(
            directory,
            model=reloaded_model,
            trainer=reloaded_trainer,
            expected_git_sha=source_sha,
            expected_model_spec_hash=stage.model.identity_sha256(),
            expected_tokenizer_hash=TOKENIZER_CONFIG_SHA256,
            expected_tokenizer_vocab_hash=TOKENIZER_VOCAB_SHA256,
            expected_dataset_manifest_hash=DATASET_MANIFEST_SHA256,
            expected_run_manifest_hash=run_manifest_hash,
            expected_environment_lock_hash=environment_lock_hash,
            expected_seed=seed,
        )
        reloaded_model.eval()
        with torch.no_grad():
            post_reload_logits = reloaded_model(inference_input).logits.detach().clone()

    difference = (pre_reload_logits.float() - post_reload_logits.float()).abs()
    max_abs = float(difference.max().item())
    exact = bool(torch.equal(pre_reload_logits, post_reload_logits))
    _require(
        max_abs <= TOLERANCES["same_precision_reload_logits_max_abs"],
        f"{precision} checkpoint reload inference exceeded same-precision tolerance",
    )
    _require(reloaded_trainer.optimizer_step == trainer.optimizer_step, "reloaded optimizer step drift")
    _require(reloaded_trainer.tokens_seen == trainer.tokens_seen, "reloaded token counter drift")

    manifest_identity = manifest.get("identity")
    _require(isinstance(manifest_identity, Mapping), "checkpoint manifest identity missing")
    return {
        "run_manifest_sha256": run_manifest_hash,
        "manifest_identity_sha256": hash_json(manifest_identity),
        "precision": manifest_identity.get("precision"),
        "step": manifest_identity.get("step"),
        "tokens_seen": manifest_identity.get("tokens_seen"),
        "reload_optimizer_step": reloaded_trainer.optimizer_step,
        "reload_tokens_seen": reloaded_trainer.tokens_seen,
        "pre_reload_logits_dtype": str(pre_reload_logits.dtype),
        "post_reload_logits_dtype": str(post_reload_logits.dtype),
        "post_reload_logits_exact": exact,
        "post_reload_logits_max_abs": max_abs,
    }


def _run_profile(
    *,
    stage_path: Path,
    train_batches: list[dict[str, torch.Tensor]],
    validation_batches: list[dict[str, torch.Tensor]],
    precision: str,
    source_sha: str,
    seed: int,
    target_tokens: int,
    environment_lock_hash: str,
    paired_identity_sha256: str,
) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    config = TrainerConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        max_steps=10_000,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision=precision,  # type: ignore[arg-type]
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")

    curve: list[dict[str, Any]] = []
    milestones = _milestones(target_tokens)
    initial_train = _evaluate(model, train_batches)
    initial_validation = _evaluate(model, validation_batches)
    curve.append(
        {
            "requested_tokens": 0,
            "actual_tokens": 0,
            "optimizer_step": 0,
            "train": initial_train,
            "validation": initial_validation,
        }
    )
    next_milestone_index = 1
    step_records: list[dict[str, Any]] = []
    finite_state_checks = 0
    nonfinite_events: list[dict[str, Any]] = []
    batch_stream = cycle(train_batches)
    process_cpu_start = time.process_time()
    experiment_wall_start = time.perf_counter()
    trainer_step_wall_seconds = 0.0

    while trainer.tokens_seen < target_tokens:
        batch = next(batch_stream)
        before = _snapshot(model)
        step_start = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        step_wall = time.perf_counter() - step_start
        trainer_step_wall_seconds += step_wall
        update_norm, update_max_abs = _update_metrics(model, before)
        finite = _all_training_state_finite(model, trainer)
        finite_state_checks += 1
        if not finite:
            nonfinite_events.append(
                {"optimizer_step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen}
            )
            raise PrecisionLearningError(f"{precision} produced non-finite training state")
        _require(metrics.grad_norm is not None, "optimizer step missing gradient norm")
        step_records.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "batch_tokens": metrics.tokens,
                "training_loss": metrics.loss,
                "training_bpb": loss_to_bpb(metrics.loss),
                "gradient_norm": float(metrics.grad_norm),
                "update_norm": update_norm,
                "update_max_abs": update_max_abs,
                "trainer_step_wall_seconds": step_wall,
            }
        )

        if next_milestone_index < len(milestones):
            requested = milestones[next_milestone_index]
            if trainer.tokens_seen >= requested:
                curve.append(
                    {
                        "requested_tokens": requested,
                        "actual_tokens": trainer.tokens_seen,
                        "optimizer_step": trainer.optimizer_step,
                        "train": _evaluate(model, train_batches),
                        "validation": _evaluate(model, validation_batches),
                    }
                )
                next_milestone_index += 1

    experiment_wall_seconds = time.perf_counter() - experiment_wall_start
    process_cpu_seconds = time.process_time() - process_cpu_start
    _require(next_milestone_index == len(milestones), "learning curve missed a milestone")

    checkpoint = _checkpoint_and_reload(
        model=model,
        trainer=trainer,
        config=config,
        stage=stage,
        validation_batches=validation_batches,
        precision=precision,
        source_sha=source_sha,
        seed=seed,
        target_tokens=target_tokens,
        environment_lock_hash=environment_lock_hash,
        paired_identity_sha256=paired_identity_sha256,
    )
    return {
        "status": "PASS",
        "precision": precision,
        "runtime": trainer.precision_runtime.to_dict(),
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "requested_target_tokens": target_tokens,
        "curve": curve,
        "steps": step_records,
        "finite_state": {
            "all_finite": not nonfinite_events,
            "checks": finite_state_checks,
            "nonfinite_events": nonfinite_events,
        },
        "timing": {
            "trainer_step_wall_seconds": trainer_step_wall_seconds,
            "experiment_wall_seconds": experiment_wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "optimized_tokens_per_trainer_step_second": trainer.tokens_seen
            / trainer_step_wall_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "checkpoint": checkpoint,
    }


def _p95(values: list[float]) -> float:
    _require(bool(values), "cannot compute p95 of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _paired_relative_differences(
    fp32_steps: list[Mapping[str, Any]],
    bf16_steps: list[Mapping[str, Any]],
    field: str,
) -> list[float]:
    _require(len(fp32_steps) == len(bf16_steps), "precision profiles have different step counts")
    differences: list[float] = []
    for fp_item, bf_item in zip(fp32_steps, bf16_steps, strict=True):
        _require(fp_item["tokens_seen"] == bf_item["tokens_seen"], "paired token trace drift")
        left = float(fp_item[field])
        right = float(bf_item[field])
        differences.append(abs(left - right) / max(abs(left), 1e-12))
    return differences


def compare_precision_profiles(
    fp32: Mapping[str, Any],
    bf16: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare paired profiles with semantic tolerances, never bitwise cross-precision rules."""
    _require(fp32["optimized_tokens"] == bf16["optimized_tokens"], "optimized token budget drift")
    _require(fp32["optimizer_steps"] == bf16["optimizer_steps"], "optimizer step count drift")
    fp_curve = list(fp32["curve"])
    bf_curve = list(bf16["curve"])
    _require(len(fp_curve) == len(bf_curve), "learning curve checkpoint count drift")

    curve_validation_deltas: list[float] = []
    curve_train_deltas: list[float] = []
    for fp_item, bf_item in zip(fp_curve, bf_curve, strict=True):
        _require(fp_item["requested_tokens"] == bf_item["requested_tokens"], "curve milestone drift")
        _require(fp_item["actual_tokens"] == bf_item["actual_tokens"], "curve actual-token drift")
        curve_train_deltas.append(abs(float(fp_item["train"]["bpb"]) - float(bf_item["train"]["bpb"])))
        curve_validation_deltas.append(
            abs(float(fp_item["validation"]["bpb"]) - float(bf_item["validation"]["bpb"]))
        )

    gradient_differences = _paired_relative_differences(
        list(fp32["steps"]), list(bf16["steps"]), "gradient_norm"
    )
    update_differences = _paired_relative_differences(
        list(fp32["steps"]), list(bf16["steps"]), "update_norm"
    )
    metrics = {
        "final_train_bpb_abs": curve_train_deltas[-1],
        "final_validation_bpb_abs": curve_validation_deltas[-1],
        "max_curve_validation_bpb_abs": max(curve_validation_deltas),
        "gradient_norm_median_relative": statistics.median(gradient_differences),
        "gradient_norm_p95_relative": _p95(gradient_differences),
        "update_norm_median_relative": statistics.median(update_differences),
        "update_norm_p95_relative": _p95(update_differences),
    }
    checks = {name: metrics[name] <= limit for name, limit in TOLERANCES.items() if name in metrics}
    finite = bool(fp32["finite_state"]["all_finite"] and bf16["finite_state"]["all_finite"])
    reload_ok = (
        float(fp32["checkpoint"]["post_reload_logits_max_abs"])
        <= TOLERANCES["same_precision_reload_logits_max_abs"]
        and float(bf16["checkpoint"]["post_reload_logits_max_abs"])
        <= TOLERANCES["same_precision_reload_logits_max_abs"]
    )
    within_tolerance = finite and reload_ok and all(checks.values())
    return {
        "cross_precision_bitwise_equality_required": False,
        "tolerances": dict(TOLERANCES),
        "metrics": metrics,
        "checks": checks,
        "finite_state_ok": finite,
        "same_precision_reload_ok": reload_ok,
        "within_tolerance": within_tolerance,
    }


def run_precision_learning_experiment(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Execute paired fp32/bf16 CPU learning curves on identical S1 fixture traces."""
    _require(len(source_sha) == 40 and all(ch in "0123456789abcdef" for ch in source_sha), "invalid source SHA")
    _require(target_tokens >= 1_000, "target_tokens must be at least 1,000")
    _require(batch_size > 0, "batch_size must be positive")
    root = Path(root).resolve()
    stage_path = root / "configs/stages/s1_100k.json"
    stage = load_stage_config(stage_path)
    _require(stage.stage == "S1", "wrong stage config")
    _require(stage.model.identity_sha256() == S1_MODEL_SPEC_SHA256, "S1 ModelSpec drift")
    _require(stage.init.identity_sha256() == INIT_SPEC_SHA256, "S1 InitSpec drift")
    _require(stage.expected_parameters == S1_PARAMETER_COUNT, "S1 parameter count drift")
    _require(stage.model.vocab_size == S1_MODEL_VOCAB, "S1 model vocabulary drift")

    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == FIXTURE_TOKENIZER_VOCAB, "controlled tokenizer drift")
    train_batches, train_ids, train_tokens, train_max_id = _tensor_batches(
        root, split="train", tokenizer=tokenizer, batch_size=batch_size
    )
    validation_batches, validation_ids, validation_tokens, validation_max_id = _tensor_batches(
        root, split="validation", tokenizer=tokenizer, batch_size=batch_size
    )
    _require(not (set(train_ids) & set(validation_ids)), "controlled train/validation overlap")
    _require(max(train_max_id, validation_max_id) < stage.model.vocab_size, "fixture token outside model vocab")

    paired_identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S1",
        "stage_config_path": "configs/stages/s1_100k.json",
        "stage_config_file_sha256": _sha256_file(stage_path),
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "seed": seed,
        "batch_size": batch_size,
        "requested_target_tokens": target_tokens,
        "fixture": {
            "purpose": "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER",
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "train_scoreable_tokens_per_epoch": train_tokens,
            "validation_scoreable_tokens": validation_tokens,
        },
        "environment": environment,
    }
    paired_identity_sha256 = _canonical_hash(paired_identity)
    bf16_probe = _cpu_bf16_runtime_probe()
    fp16_probe = _fp16_cpu_failure_probe(stage_path, seed=seed)
    environment_lock_hash = str(environment["lock_profile_manifest_sha256"])

    profiles = {
        "fp32": _run_profile(
            stage_path=stage_path,
            train_batches=train_batches,
            validation_batches=validation_batches,
            precision="fp32",
            source_sha=source_sha,
            seed=seed,
            target_tokens=target_tokens,
            environment_lock_hash=environment_lock_hash,
            paired_identity_sha256=paired_identity_sha256,
        ),
        "bf16": _run_profile(
            stage_path=stage_path,
            train_batches=train_batches,
            validation_batches=validation_batches,
            precision="bf16",
            source_sha=source_sha,
            seed=seed,
            target_tokens=target_tokens,
            environment_lock_hash=environment_lock_hash,
            paired_identity_sha256=paired_identity_sha256,
        ),
    }
    comparison = compare_precision_profiles(profiles["fp32"], profiles["bf16"])
    recommendation = (
        "PROCEED_TO_NATIVE_CUDA_BF16_PILOT_WITH_FP32_REFERENCE"
        if comparison["within_tolerance"]
        else "KEEP_FP32_REFERENCE_AND_REQUIRE_DEVICE_BOUND_CUDA_BF16_NUMERICAL_PILOT"
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "paired_identity": paired_identity,
        "paired_identity_sha256": paired_identity_sha256,
        "capability_probes": {
            "cpu_bf16_autocast": bf16_probe,
            "cpu_fp16": fp16_probe,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_executed": False,
        },
        "profiles": profiles,
        "comparison": comparison,
        "recommendation": {
            "decision": recommendation,
            "cuda_speed_extrapolated_from_cpu": False,
            "cuda_pilot_requirements": [
                "prove requested CUDA device is visible",
                "prefer bf16 only when native CUDA bf16 capability probe passes",
                "repeat fp32 versus bf16 learning and finite-state guards on one fixed trace",
                "verify checkpoint reload before longer training",
                "measure CUDA wall time and memory independently of CPU results",
                "use fp16 plus GradScaler only as an explicit separately supported fallback",
            ],
        },
        "claims": {
            "cross_precision_bitwise_equality": False,
            "cpu_bf16_speed_generalized_to_gpu": False,
            "native_cpu_bf16_isa_proven": False,
            "cuda_precision_evidence": False,
            "paid_compute_used": False,
            "s1_stage_promoted": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence
