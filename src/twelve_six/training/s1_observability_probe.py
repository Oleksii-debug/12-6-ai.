"""Real S1 engineering-model observability probe.

This composes the incumbent D02 S1 mechanics path with TRAIN-29 observability.
It is LOCAL_FREE CPU engineering evidence, not S1 stage/capacity evidence. The
controlled S0 fixture remains compatibility-only and is not relabeled as S1 data.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .observability import TrainingObserver, paid_compute_decision_support
from .s0_evidence_contract import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    INIT_SPEC_SHA256,
    PACKING_CONFIG_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .s1_preflight import (
    FIXTURE_TOKENIZER_VOCAB,
    REPOSITORY,
    S1_MODEL_SPEC_SHA256,
    S1_MODEL_VOCAB,
    S1_PARAMETER_COUNT,
    _evaluate,
    _tensor_batches,
)
from .trainer import Trainer

SCHEMA_VERSION = "12-6.s1-training-observability-probe.v1"
AUTHORITY = "LOCAL_FREE_S1_ENGINEERING_OBSERVABILITY_NOT_STAGE_OR_CAPACITY_EVIDENCE"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class S1ObservabilityProbeError(ValueError):
    """Raised when S1 observability evidence cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S1ObservabilityProbeError(message)


def _finite_nonnegative(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number) and number >= 0.0, f"{field} must be finite and non-negative")
    return number


def _checkpoint_identity(
    *,
    source_sha: str,
    stage: Any,
    config: TrainerConfig,
    trainer: Trainer,
    run_identity_sha256: str,
    environment: Mapping[str, str],
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=TOKENIZER_CONFIG_SHA256,
        tokenizer_vocab_hash=TOKENIZER_VOCAB_SHA256,
        dataset_manifest_hash=DATASET_MANIFEST_SHA256,
        run_manifest_hash=run_identity_sha256,
        training_config=asdict(config),
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash=environment["lock_profile_manifest_sha256"],
    )


def run_s1_observability_probe(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    max_steps: int = 12,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Measure the current 107,856-parameter S1 engineering model on real Trainer seams."""
    _require(_GIT_SHA.fullmatch(source_sha) is not None, "source SHA must be full lowercase Git SHA")
    _require(max_steps > 0, "max_steps must be positive")
    _require(batch_size > 0, "batch_size must be positive")
    root = Path(root).resolve()
    stage_path = root / "configs/stages/s1_100k.json"
    stage = load_stage_config(stage_path)
    _require(stage.stage == "S1", "wrong stage config")
    _require(stage.model.identity_sha256() == S1_MODEL_SPEC_SHA256, "S1 ModelSpec drift")
    _require(stage.init.identity_sha256() == INIT_SPEC_SHA256, "S1 InitSpec drift")
    _require(stage.expected_parameters == S1_PARAMETER_COUNT, "S1 parameter-count drift")
    _require(stage.model.vocab_size == S1_MODEL_VOCAB, "S1 vocabulary-size drift")

    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == FIXTURE_TOKENIZER_VOCAB, "controlled byte tokenizer drift")
    train_batches, train_ids, train_tokens_per_epoch, train_max_id = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    validation_batches, validation_ids, validation_tokens, validation_max_id = _tensor_batches(
        root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    _require(not (set(train_ids) & set(validation_ids)), "controlled train/validation overlap")
    max_fixture_token_id = max(train_max_id, validation_max_id)
    _require(max_fixture_token_id < stage.model.vocab_size, "fixture token exceeds S1 vocabulary")

    config = TrainerConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    run_identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S1",
        "stage_config_path": "configs/stages/s1_100k.json",
        "stage_config_file_sha256": sha256_file(stage_path),
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "model_vocab_size": stage.model.vocab_size,
        "max_seq_len": stage.model.max_seq_len,
        "environment": dict(environment),
        "fixture": {
            "purpose": "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER",
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "max_emitted_token_id": max_fixture_token_id,
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "train_scoreable_tokens_per_epoch": train_tokens_per_epoch,
            "validation_scoreable_tokens": validation_tokens,
        },
        "training_config": asdict(config),
        "batch_size_examples": batch_size,
        "probe_contract": SCHEMA_VERSION,
    }
    observer = TrainingObserver(
        run_identity,
        device="cpu",
        max_step_samples=max(64, max_steps),
        gpu_sample_every_steps=1,
    )

    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")

    initial_train_loss, initial_train_eval_tokens = observer.measure_region(
        "evaluation",
        "initial_train_loss",
        lambda: _evaluate(model, train_batches),
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )
    initial_validation_loss, initial_validation_eval_tokens = observer.measure_region(
        "evaluation",
        "initial_validation_loss",
        lambda: _evaluate(model, validation_batches),
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )

    iterator = iter(islice(cycle(train_batches), max_steps))
    for _ in range(max_steps):
        batch, data_wait_seconds = observer.measure_next(iterator)
        observer.train_microbatch(
            trainer,
            batch,
            data_wait_seconds=data_wait_seconds,
        )

    final_train_loss, final_train_eval_tokens = observer.measure_region(
        "evaluation",
        "final_train_loss",
        lambda: _evaluate(model, train_batches),
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )
    validation_step_before = trainer.optimizer_step
    final_validation_loss, final_validation_eval_tokens = observer.measure_region(
        "evaluation",
        "final_validation_loss",
        lambda: _evaluate(model, validation_batches),
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )
    validation_step_after = trainer.optimizer_step
    _require(validation_step_before == validation_step_after, "evaluation mutated optimizer step")

    checkpoint_size_bytes = 0
    checkpoint_manifest: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="twelve-six-s1-observability-") as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_identity = _checkpoint_identity(
            source_sha=source_sha,
            stage=stage,
            config=config,
            trainer=trainer,
            run_identity_sha256=observer.run_identity_sha256,
            environment=environment,
        )
        checkpoint_manifest = observer.measure_region(
            "checkpoint",
            "save_trainer_checkpoint",
            lambda: save_trainer_checkpoint(
                checkpoint_dir,
                model=model,
                trainer=trainer,
                identity=checkpoint_identity,
            ),
            optimizer_step=trainer.optimizer_step,
            tokens_seen=trainer.tokens_seen,
        )
        observer.measure_region(
            "checkpoint",
            "verify_checkpoint",
            lambda: verify_checkpoint(checkpoint_dir),
            optimizer_step=trainer.optimizer_step,
            tokens_seen=trainer.tokens_seen,
        )
        checkpoint_size_bytes = sum(
            path.stat().st_size for path in checkpoint_dir.iterdir() if path.is_file()
        )

    summary = observer.summary()
    decision = paid_compute_decision_support(summary)
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": run_identity,
        "identity_sha256": observer.run_identity_sha256,
        "training": {
            "optimizer_steps": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "initial_train_eval_tokens": initial_train_eval_tokens,
            "final_train_eval_tokens": final_train_eval_tokens,
            "initial_validation_eval_tokens": initial_validation_eval_tokens,
            "final_validation_eval_tokens": final_validation_eval_tokens,
            "validation_optimizer_step_before_final_eval": validation_step_before,
            "validation_optimizer_step_after_final_eval": validation_step_after,
        },
        "checkpoint": {
            "size_bytes": checkpoint_size_bytes,
            "manifest_format": checkpoint_manifest.get("format") if checkpoint_manifest else None,
            "save_and_verify_measured_separately": True,
        },
        "telemetry": observer.export(),
        "paid_compute_decision_support": decision,
        "claims": {
            "s1_architecture_frozen": False,
            "s1_corpus_or_tokenizer_frozen": False,
            "s1_quality_or_capability_evidence": False,
            "paid_compute_authorized_or_used": False,
            "gpu_capacity_claim": False,
            "distributed_scaling_claim": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
        },
    }
    validate_s1_observability_probe(evidence)
    return evidence


def validate_s1_observability_probe(evidence: Mapping[str, Any]) -> None:
    """Fail closed on identity drift, missing telemetry, or paid-compute overclaim."""
    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong observability schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong observability authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity block missing")
    _require(identity.get("repository") == REPOSITORY, "repository identity mismatch")
    _require(identity.get("stage") == "S1", "stage identity mismatch")
    _require(identity.get("modelspec_sha256") == S1_MODEL_SPEC_SHA256, "S1 ModelSpec drift")
    _require(identity.get("initspec_sha256") == INIT_SPEC_SHA256, "S1 InitSpec drift")
    _require(identity.get("parameter_count") == S1_PARAMETER_COUNT, "S1 parameter-count drift")
    _require(evidence.get("identity_sha256") == hash_json(identity), "identity self-hash mismatch")

    training = evidence.get("training")
    _require(isinstance(training, Mapping), "training block missing")
    _require(
        isinstance(training.get("optimizer_steps"), int) and training["optimizer_steps"] > 0,
        "optimizer step count invalid",
    )
    _require(
        isinstance(training.get("optimized_tokens"), int) and training["optimized_tokens"] > 0,
        "optimized token count invalid",
    )
    for field in (
        "initial_train_loss",
        "final_train_loss",
        "initial_validation_loss",
        "final_validation_loss",
    ):
        _finite_nonnegative(training.get(field), field)
    _require(
        training.get("validation_optimizer_step_before_final_eval")
        == training.get("validation_optimizer_step_after_final_eval"),
        "held-out evaluation mutated optimizer state",
    )

    telemetry = evidence.get("telemetry")
    _require(isinstance(telemetry, Mapping), "telemetry block missing")
    _require(
        telemetry.get("run_identity_sha256") == evidence.get("identity_sha256"),
        "telemetry/run identity mismatch",
    )
    summary = telemetry.get("summary")
    _require(isinstance(summary, Mapping), "telemetry summary missing")
    counters = summary.get("counters")
    timing = summary.get("timing")
    throughput = summary.get("throughput")
    _require(isinstance(counters, Mapping), "telemetry counters missing")
    _require(isinstance(timing, Mapping), "telemetry timing missing")
    _require(isinstance(throughput, Mapping), "telemetry throughput missing")
    _require(
        counters.get("optimized_tokens") == training.get("optimized_tokens"),
        "telemetry token accounting mismatch",
    )
    _require(
        counters.get("observed_optimizer_steps") == training.get("optimizer_steps"),
        "telemetry optimizer-step accounting mismatch",
    )
    _finite_nonnegative(timing.get("step_seconds_total"), "step_seconds_total")
    _finite_nonnegative(timing.get("data_wait_seconds_total"), "data_wait_seconds_total")
    _finite_nonnegative(timing.get("checkpoint_seconds_total"), "checkpoint_seconds_total")
    _finite_nonnegative(timing.get("evaluation_seconds_total"), "evaluation_seconds_total")
    _finite_nonnegative(
        throughput.get("train_tokens_per_second"),
        "train_tokens_per_second",
    )
    phase = summary.get("phase_timing")
    _require(isinstance(phase, Mapping), "phase timing capability block missing")
    _require(
        phase.get("forward", {}).get("status") == "UNAVAILABLE_NOT_RECORDED",
        "current Trainer forward timing must not be fabricated",
    )
    _require(
        phase.get("backward", {}).get("status") == "UNAVAILABLE_NOT_RECORDED",
        "current Trainer backward timing must not be fabricated",
    )
    _require(
        phase.get("update", {}).get("status") == "UNAVAILABLE_NOT_RECORDED",
        "current Trainer update timing must not be fabricated",
    )

    checkpoint = evidence.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint block missing")
    _require(
        isinstance(checkpoint.get("size_bytes"), int) and checkpoint["size_bytes"] > 0,
        "checkpoint size missing",
    )
    _require(timing.get("checkpoint_count") == 2, "save and verify checkpoint timing required")
    _require(timing.get("evaluation_count") == 4, "four evaluation timing regions required")

    decision = evidence.get("paid_compute_decision_support")
    _require(isinstance(decision, Mapping), "paid-compute decision support missing")
    _require(
        decision.get("euro_2000_gate") == "BLOCKED_PENDING_TARGET_GPU_CALIBRATION",
        "CPU evidence must not open the €2k paid-compute gate",
    )
    _require(
        decision.get("euro_10000_gate")
        == "BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION",
        "CPU evidence must not open the €10k paid-compute gate",
    )
    _require(decision.get("telemetry_alone_authorizes_spend") is False, "spend overclaim")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims block missing")
    forbidden_true = (
        "s1_architecture_frozen",
        "s1_corpus_or_tokenizer_frozen",
        "s1_quality_or_capability_evidence",
        "paid_compute_authorized_or_used",
        "gpu_capacity_claim",
        "distributed_scaling_claim",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
    )
    for field in forbidden_true:
        _require(claims.get(field) is False, f"forbidden claim flag set: {field}")
