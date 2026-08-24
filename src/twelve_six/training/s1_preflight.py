"""Collision-safe S1 numerical preflight on the current engineering ModelSpec.

This module is intentionally not S1 stage evidence. It uses the controlled S0
fixture/tokenizer/packing only to exercise the current non-frozen S1 architecture
and D02 Trainer numerics under LOCAL_FREE CPU execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import batch_examples, collate_rows, iter_packed_examples, load_jsonl_records
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
from .trainer import StepMetrics, Trainer

SCHEMA_VERSION = "12-6.s1-numerical-preflight.v1"
AUTHORITY = "ENGINEERING_PREFLIGHT_NOT_STAGE_EVIDENCE"
REPOSITORY = "Oleksii-debug/12-6-ai."
S1_MODEL_SPEC_SHA256 = "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6"
S1_PARAMETER_COUNT = 107_856
S1_MODEL_VOCAB = 512
FIXTURE_TOKENIZER_VOCAB = 256
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class S1PreflightError(ValueError):
    """Raised when S1 engineering preflight evidence fails closed."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S1PreflightError(message)


def _finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} not numeric")
    number = float(value)
    _require(math.isfinite(number), f"{field} not finite")
    if nonnegative:
        _require(number >= 0.0, f"{field} negative")
    return number


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
) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        tokens = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        _require(torch.isfinite(loss).item(), "evaluation produced non-finite loss")
        weighted_loss += float(loss.item()) * tokens
        token_count += tokens
    _require(token_count > 0, "evaluation split contains zero scoreable tokens")
    return weighted_loss / token_count, token_count


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _weight_delta(
    model: TwelveSixDecoder,
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


def _run_precision_profile(
    stage_path: Path,
    train_batches: list[dict[str, torch.Tensor]],
    validation_batches: list[dict[str, torch.Tensor]],
    *,
    precision: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    config = TrainerConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        max_steps=max_steps,
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
    before = _snapshot(model)
    trainer = Trainer(model, config, device="cpu")
    initial_train_loss, train_eval_tokens = _evaluate(model, train_batches)
    initial_validation_loss, validation_eval_tokens = _evaluate(model, validation_batches)

    metrics: list[StepMetrics] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for batch in islice(cycle(train_batches), max_steps):
        metrics.append(trainer.train_microbatch(batch))
    wall_seconds = time.perf_counter() - wall_start
    process_cpu_seconds = time.process_time() - cpu_start

    before_validation_step = trainer.optimizer_step
    final_train_loss, final_train_eval_tokens = _evaluate(model, train_batches)
    final_validation_loss, final_validation_eval_tokens = _evaluate(model, validation_batches)
    after_validation_step = trainer.optimizer_step

    grad_norms = [item.grad_norm for item in metrics if item.grad_norm is not None]
    _require(bool(grad_norms), f"{precision} produced no optimizer-step gradient norms")
    delta = _weight_delta(model, before)
    return {
        "status": "PASS",
        "precision": precision,
        "seed": seed,
        "optimizer_steps": trainer.optimizer_step,
        "microbatches_consumed": len(metrics),
        "optimized_tokens": trainer.tokens_seen,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "initial_train_eval_tokens": train_eval_tokens,
        "final_train_eval_tokens": final_train_eval_tokens,
        "initial_validation_eval_tokens": validation_eval_tokens,
        "final_validation_eval_tokens": final_validation_eval_tokens,
        "gradient_norm_min": min(float(value) for value in grad_norms),
        "gradient_norm_max": max(float(value) for value in grad_norms),
        "weight_delta": delta,
        "validation_optimized_tokens": 0,
        "optimizer_step_before_final_validation": before_validation_step,
        "optimizer_step_after_final_validation": after_validation_step,
        "runtime": {
            "device": "cpu",
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def _fp16_cpu_failure_probe(stage_path: Path, *, seed: int) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    config = TrainerConfig(max_steps=1, precision="fp16", seed=seed)
    try:
        Trainer(model, config, device="cpu")
    except ValueError as exc:
        _require("fp16 training requires a CUDA device" in str(exc), "unexpected fp16 CPU error")
        return {"status": "FAIL_CLOSED_AS_DESIGNED", "exception": type(exc).__name__}
    raise S1PreflightError("fp16 CPU path unexpectedly became trainable")


def run_s1_numerical_preflight(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    max_steps: int = 6,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Exercise current S1 engineering architecture without declaring S1 readiness."""
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
    train_batches, train_ids, train_tokens, train_max_id = _tensor_batches(
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

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S1",
        "stage_config_path": "configs/stages/s1_100k.json",
        "stage_config_file_sha256": _sha256_file(stage_path),
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "model_vocab_size": stage.model.vocab_size,
        "max_seq_len": stage.model.max_seq_len,
        "environment": environment,
        "fixture": {
            "purpose": "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER",
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "max_emitted_token_id": max_fixture_token_id,
            "unused_model_vocab_rows": stage.model.vocab_size - tokenizer.vocab_size,
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "train_scoreable_tokens_per_epoch": train_tokens,
            "validation_scoreable_tokens": validation_tokens,
        },
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "seed_ordering": {
            "seed": seed,
            "seed_applied_before_model_construction": True,
            "trainer_reapplies_training_rng_seed": True,
        },
        "profiles": {
            "fp32": _run_precision_profile(
                stage_path,
                train_batches,
                validation_batches,
                precision="fp32",
                seed=seed,
                max_steps=max_steps,
            ),
            "bf16": _run_precision_profile(
                stage_path,
                train_batches,
                validation_batches,
                precision="bf16",
                seed=seed,
                max_steps=max_steps,
            ),
        },
        "fp16_cpu_probe": _fp16_cpu_failure_probe(stage_path, seed=seed),
        "claims": {
            "s1_architecture_frozen": False,
            "s1_corpus_or_tokenizer_frozen": False,
            "s1_quality_or_capability_evidence": False,
            "candidate_or_stable_promotion": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "paid_compute_authorized_or_used": False,
            "cross_hardware_bitwise_reproducibility": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_s1_numerical_preflight(evidence)
    return evidence


def validate_s1_numerical_preflight(evidence: Mapping[str, Any]) -> None:
    """Fail closed on drift, leakage, non-finite numerics, or authority overclaim."""
    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong S1 preflight schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong S1 preflight authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity block missing")
    _require(identity.get("repository") == REPOSITORY, "repository identity mismatch")
    source_sha = identity.get("source_sha")
    _require(isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None, "invalid source SHA")
    _require(identity.get("stage") == "S1", "stage identity mismatch")
    _require(identity.get("modelspec_sha256") == S1_MODEL_SPEC_SHA256, "S1 ModelSpec identity mismatch")
    _require(identity.get("initspec_sha256") == INIT_SPEC_SHA256, "S1 InitSpec identity mismatch")
    _require(identity.get("parameter_count") == S1_PARAMETER_COUNT, "S1 parameter count mismatch")
    _require(identity.get("model_vocab_size") == S1_MODEL_VOCAB, "S1 model vocab mismatch")
    _require(evidence.get("identity_sha256") == _canonical_hash(identity), "identity self-hash mismatch")

    fixture = identity.get("fixture")
    _require(isinstance(fixture, Mapping), "fixture identity missing")
    _require(
        fixture.get("purpose") == "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S1_CORPUS_OR_TOKENIZER",
        "fixture purpose overclaims S1 authority",
    )
    _require(fixture.get("tokenizer_vocab_size") == FIXTURE_TOKENIZER_VOCAB, "fixture vocab drift")
    max_emitted = fixture.get("max_emitted_token_id")
    _require(isinstance(max_emitted, int) and 0 <= max_emitted < S1_MODEL_VOCAB, "fixture token range invalid")
    _require(fixture.get("unused_model_vocab_rows") == 256, "S1 fixture unused-vocab accounting drift")
    train_ids = fixture.get("train_record_ids")
    validation_ids = fixture.get("validation_record_ids")
    _require(isinstance(train_ids, list) and isinstance(validation_ids, list), "fixture split IDs missing")
    _require(not (set(train_ids) & set(validation_ids)), "fixture train/validation overlap")

    profiles = evidence.get("profiles")
    _require(isinstance(profiles, Mapping), "precision profiles missing")
    for name in ("fp32", "bf16"):
        profile = profiles.get(name)
        _require(isinstance(profile, Mapping), f"{name} profile missing")
        _require(profile.get("status") == "PASS", f"{name} profile did not pass")
        steps = profile.get("optimizer_steps")
        _require(isinstance(steps, int) and steps > 0, f"{name} optimizer steps invalid")
        optimized = profile.get("optimized_tokens")
        _require(isinstance(optimized, int) and optimized > 0, f"{name} optimized tokens invalid")
        _require(profile.get("validation_optimized_tokens") == 0, f"{name} optimized validation data")
        _require(
            profile.get("optimizer_step_before_final_validation")
            == profile.get("optimizer_step_after_final_validation"),
            f"{name} validation mutated optimizer state",
        )
        for field in (
            "initial_train_loss",
            "final_train_loss",
            "initial_validation_loss",
            "final_validation_loss",
            "gradient_norm_min",
            "gradient_norm_max",
        ):
            _finite(profile.get(field), f"{name}.{field}", nonnegative=True)
        _require(
            _finite(profile.get("final_train_loss"), f"{name}.final_train_loss")
            <= 4.0 * _finite(profile.get("initial_train_loss"), f"{name}.initial_train_loss"),
            f"{name} train loss catastrophically diverged",
        )
        delta = profile.get("weight_delta")
        _require(isinstance(delta, Mapping), f"{name} weight delta missing")
        changed = delta.get("changed_parameter_elements")
        total = delta.get("trainable_parameter_elements")
        _require(isinstance(changed, int) and changed > 0, f"{name} changed no weights")
        _require(total == S1_PARAMETER_COUNT, f"{name} trainable parameter count drift")
        _finite(delta.get("l2"), f"{name}.weight_delta.l2", nonnegative=True)
        _finite(delta.get("max_abs"), f"{name}.weight_delta.max_abs", nonnegative=True)
        runtime = profile.get("runtime")
        _require(isinstance(runtime, Mapping) and runtime.get("device") == "cpu", f"{name} runtime drift")
        _finite(runtime.get("wall_seconds"), f"{name}.runtime.wall_seconds", nonnegative=True)
        _finite(runtime.get("process_cpu_seconds"), f"{name}.runtime.process_cpu_seconds", nonnegative=True)

    fp16 = evidence.get("fp16_cpu_probe")
    _require(isinstance(fp16, Mapping), "fp16 CPU probe missing")
    _require(fp16.get("status") == "FAIL_CLOSED_AS_DESIGNED", "fp16 CPU did not fail closed")
    ordering = evidence.get("seed_ordering")
    _require(isinstance(ordering, Mapping), "seed ordering missing")
    _require(ordering.get("seed_applied_before_model_construction") is True, "seed ordering not proven")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims block missing")
    for key in (
        "s1_architecture_frozen",
        "s1_corpus_or_tokenizer_frozen",
        "s1_quality_or_capability_evidence",
        "candidate_or_stable_promotion",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
        "cross_hardware_bitwise_reproducibility",
    ):
        _require(claims.get(key) is False, f"prohibited claim enabled: {key}")

    claimed_hash = evidence.get("evidence_sha256")
    _require(isinstance(claimed_hash, str), "evidence self-hash missing")
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    _require(_canonical_hash(unhashed) == claimed_hash, "evidence self-hash mismatch")
