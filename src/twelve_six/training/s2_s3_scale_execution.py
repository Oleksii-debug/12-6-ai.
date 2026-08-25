"""Execute canonical S2/S3 scale mechanics without declaring stage readiness."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Mapping
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

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
from .trainer import StepMetrics, Trainer

SCHEMA_VERSION = "12-6.s2-s3-scale-execution.v1"
RUN_CONFIG_SCHEMA = "12-6.s2-s3-scale-execution-config.v1"
AUTHORITY = "ENGINEERING_SCALE_EXECUTION_ONLY_NOT_STAGE_EVIDENCE"
REPOSITORY = "Oleksii-debug/12-6-ai."
RUN_CONFIG_PATH = "configs/runs/s2_s3.scale_execution.json"
FIXTURE_PURPOSE = (
    "CONTROLLED_S0_FIXTURE_COMPATIBILITY_ONLY_NOT_S2_S3_CORPUS_OR_TOKENIZER"
)
FIXTURE_TOKENIZER_VOCAB = 256
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_STAGE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "S2": {
        "stage_config_path": "configs/stages/s2_1m.json",
        "modelspec_sha256": (
            "2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566"
        ),
        "parameter_count": 1_066_112,
        "model_vocab_size": 2_048,
        "max_seq_len": 512,
        "optimizer_steps": 4,
        "batch_size": 2,
        "learning_rate": 1e-3,
    },
    "S3": {
        "stage_config_path": "configs/stages/s3_10m.json",
        "modelspec_sha256": (
            "3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a"
        ),
        "parameter_count": 10_059_840,
        "model_vocab_size": 8_192,
        "max_seq_len": 1_024,
        "optimizer_steps": 2,
        "batch_size": 1,
        "learning_rate": 1e-3,
    },
}


class ScaleExecutionError(ValueError):
    """Raised when scale execution evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScaleExecutionError(message)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, field: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} not numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{field} not finite")
    if nonnegative:
        _require(number >= 0.0, f"{field} negative")
    return number


def load_scale_execution_config(root: str | Path) -> dict[str, Any]:
    """Load the fixed S2/S3 LOCAL_FREE execution recipe and reject recipe drift."""
    root = Path(root).resolve()
    raw = json.loads((root / RUN_CONFIG_PATH).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "scale execution config must be an object")
    _require(raw.get("schema_version") == RUN_CONFIG_SCHEMA, "wrong run config schema")
    _require(raw.get("authority") == AUTHORITY, "wrong run config authority")
    _require(raw.get("device") == "cpu", "scale execution device drift")
    _require(raw.get("precision") == "fp32", "scale execution precision drift")
    _require(raw.get("sequence_length") == 128, "fixture sequence length drift")
    _require(raw.get("train_batch_limit") == 2, "train batch limit drift")
    _require(raw.get("validation_batch_limit") == 1, "validation batch limit drift")

    stages = raw.get("stages")
    _require(isinstance(stages, dict), "run config stages missing")
    _require(set(stages) == set(_STAGE_EXPECTATIONS), "run config stage set drift")
    for stage_name, expected in _STAGE_EXPECTATIONS.items():
        recipe = stages.get(stage_name)
        _require(isinstance(recipe, dict), f"{stage_name} recipe missing")
        for key in (
            "stage_config_path",
            "parameter_count",
            "model_vocab_size",
            "max_seq_len",
            "optimizer_steps",
            "batch_size",
            "learning_rate",
        ):
            _require(
                recipe.get(key) == expected[key],
                f"{stage_name} run config drift: {key}",
            )
    return raw


def _tensor_batches(
    root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
    limit: int,
) -> tuple[list[dict[str, torch.Tensor]], tuple[str, ...], int]:
    records = tuple(
        load_jsonl_records(root / f"data/s0/packaged/{split}.jsonl", split=split)
    )
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    _require(bool(examples), f"controlled fixture split {split} is empty")
    batches: list[dict[str, torch.Tensor]] = []
    max_token_id = -1
    for group in islice(
        batch_examples(examples, batch_size=batch_size, drop_last=False),
        limit,
    ):
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        max_token_id = max(max_token_id, int(input_ids.max().item()))
        batches.append({"input_ids": input_ids, "labels": labels})
    _require(bool(batches), f"controlled fixture split {split} produced no batches")
    return batches, tuple(record.record_id for record in records), max_token_id


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
    _require(token_count > 0, "evaluation contains zero scoreable tokens")
    return weighted_loss / token_count, token_count


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


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


def _model_parameter_bytes(model: TwelveSixDecoder) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def _optimizer_tensor_bytes(trainer: Trainer) -> int:
    total = 0
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def run_scale_execution(
    root: str | Path,
    *,
    stage_name: str,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
) -> dict[str, Any]:
    """Run one canonical scale stage using only LOCAL_FREE CPU mechanics evidence."""
    _require(stage_name in _STAGE_EXPECTATIONS, "stage must be S2 or S3")
    _require(
        _GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase Git SHA",
    )
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        "seed must be int",
    )

    root = Path(root).resolve()
    run_config = load_scale_execution_config(root)
    recipe = dict(run_config["stages"][stage_name])
    expected = _STAGE_EXPECTATIONS[stage_name]
    stage_path = root / expected["stage_config_path"]
    stage = load_stage_config(stage_path)
    _require(stage.stage == stage_name, "stage config identity mismatch")
    _require(
        stage.model.identity_sha256() == expected["modelspec_sha256"],
        f"{stage_name} ModelSpec drift",
    )
    _require(stage.init.identity_sha256() == INIT_SPEC_SHA256, "InitSpec drift")
    _require(
        stage.expected_parameters == expected["parameter_count"],
        f"{stage_name} parameter count drift",
    )
    _require(
        stage.model.vocab_size == expected["model_vocab_size"],
        f"{stage_name} vocabulary drift",
    )
    _require(
        stage.model.max_seq_len == expected["max_seq_len"],
        f"{stage_name} context drift",
    )
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )

    tokenizer = ByteTokenizer()
    _require(
        tokenizer.vocab_size == FIXTURE_TOKENIZER_VOCAB,
        "controlled byte tokenizer drift",
    )
    train_batches, train_ids, train_max_id = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=recipe["batch_size"],
        limit=run_config["train_batch_limit"],
    )
    validation_batches, validation_ids, validation_max_id = _tensor_batches(
        root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=recipe["batch_size"],
        limit=run_config["validation_batch_limit"],
    )
    _require(not (set(train_ids) & set(validation_ids)), "fixture split overlap")
    max_fixture_token_id = max(train_max_id, validation_max_id)
    _require(
        max_fixture_token_id < stage.model.vocab_size,
        "fixture token exceeds model vocabulary",
    )

    config = TrainerConfig(
        learning_rate=recipe["learning_rate"],
        weight_decay=0.0,
        max_steps=recipe["optimizer_steps"],
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    parameter_bytes = _model_parameter_bytes(model)
    before = _snapshot(model)
    snapshot_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in before.values()
    )
    trainer = Trainer(model, config, device="cpu")
    initial_train_loss, initial_train_eval_tokens = _evaluate(model, train_batches)
    initial_validation_loss, initial_validation_eval_tokens = _evaluate(
        model,
        validation_batches,
    )

    metrics: list[StepMetrics] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for batch in islice(cycle(train_batches), recipe["optimizer_steps"]):
        metrics.append(trainer.train_microbatch(batch))
    wall_seconds = time.perf_counter() - wall_start
    process_cpu_seconds = time.process_time() - cpu_start

    before_validation_step = trainer.optimizer_step
    final_train_loss, final_train_eval_tokens = _evaluate(model, train_batches)
    final_validation_loss, final_validation_eval_tokens = _evaluate(
        model,
        validation_batches,
    )
    after_validation_step = trainer.optimizer_step
    grad_norms = [item.grad_norm for item in metrics if item.grad_norm is not None]
    _require(bool(grad_norms), "no optimizer-step gradient norms observed")
    delta = _weight_delta(model, before)
    optimizer_bytes = _optimizer_tensor_bytes(trainer)
    observed_tensor_bytes = parameter_bytes + optimizer_bytes + snapshot_bytes
    tokens_per_second = trainer.tokens_seen / wall_seconds

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": stage_name,
        "stage_config_path": expected["stage_config_path"],
        "stage_config_file_sha256": _sha256_file(stage_path),
        "run_config_path": RUN_CONFIG_PATH,
        "run_config_file_sha256": _sha256_file(root / RUN_CONFIG_PATH),
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "model_vocab_size": stage.model.vocab_size,
        "max_seq_len": stage.model.max_seq_len,
        "environment": environment,
        "fixture": {
            "purpose": FIXTURE_PURPOSE,
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
        },
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "recipe": {
            "device": "cpu",
            "precision": "fp32",
            "sequence_length": run_config["sequence_length"],
            "train_batch_limit": run_config["train_batch_limit"],
            "validation_batch_limit": run_config["validation_batch_limit"],
            "batch_size": recipe["batch_size"],
            "optimizer_steps_requested": recipe["optimizer_steps"],
            "learning_rate": recipe["learning_rate"],
            "seed": seed,
        },
        "training": {
            "status": "PASS",
            "optimizer_steps": trainer.optimizer_step,
            "microbatches_consumed": len(metrics),
            "optimized_tokens": trainer.tokens_seen,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "initial_train_eval_tokens": initial_train_eval_tokens,
            "final_train_eval_tokens": final_train_eval_tokens,
            "initial_validation_eval_tokens": initial_validation_eval_tokens,
            "final_validation_eval_tokens": final_validation_eval_tokens,
            "gradient_norm_min": min(float(value) for value in grad_norms),
            "gradient_norm_max": max(float(value) for value in grad_norms),
            "weight_delta": delta,
            "validation_optimized_tokens": 0,
            "optimizer_step_before_final_validation": before_validation_step,
            "optimizer_step_after_final_validation": after_validation_step,
        },
        "resources": {
            "model_parameter_bytes": parameter_bytes,
            "optimizer_tensor_bytes_after_training": optimizer_bytes,
            "measurement_snapshot_bytes": snapshot_bytes,
            "observed_tensor_bytes_with_snapshot": observed_tensor_bytes,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "wall_seconds_training_only": wall_seconds,
            "process_cpu_seconds_training_only": process_cpu_seconds,
            "optimized_tokens_per_wall_second": tokens_per_second,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "claims": {
            "stage_architecture_frozen": False,
            "stage_corpus_or_tokenizer_frozen": False,
            "stage_quality_or_capability_evidence": False,
            "candidate_or_stable_promotion": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "paid_compute_authorized_or_used": False,
            "gpu_or_distributed_execution": False,
            "cross_hardware_bitwise_reproducibility": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_scale_execution_evidence(evidence, expected_stage=stage_name)
    return evidence


def validate_scale_execution_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
) -> None:
    """Fail closed on stage drift, numerical failure, or authority overclaim."""
    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong evidence schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong evidence authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity block missing")
    _require(identity.get("repository") == REPOSITORY, "repository mismatch")
    source_sha = identity.get("source_sha")
    _require(
        isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None,
        "invalid source SHA",
    )
    stage_name = identity.get("stage")
    _require(stage_name in _STAGE_EXPECTATIONS, "invalid stage identity")
    if expected_stage is not None:
        _require(stage_name == expected_stage, "unexpected stage evidence")
    expected = _STAGE_EXPECTATIONS[str(stage_name)]
    _require(
        identity.get("stage_config_path") == expected["stage_config_path"],
        "stage config path mismatch",
    )
    _require(
        identity.get("modelspec_sha256") == expected["modelspec_sha256"],
        "ModelSpec identity mismatch",
    )
    _require(identity.get("initspec_sha256") == INIT_SPEC_SHA256, "InitSpec mismatch")
    _require(
        identity.get("parameter_count") == expected["parameter_count"],
        "parameter count mismatch",
    )
    _require(
        identity.get("model_vocab_size") == expected["model_vocab_size"],
        "model vocab mismatch",
    )
    _require(
        identity.get("max_seq_len") == expected["max_seq_len"],
        "context mismatch",
    )
    for field in ("stage_config_file_sha256", "run_config_file_sha256"):
        value = identity.get(field)
        _require(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None,
            f"{field} invalid",
        )
    _require(identity.get("run_config_path") == RUN_CONFIG_PATH, "run config mismatch")
    _require(
        evidence.get("identity_sha256") == _canonical_hash(identity),
        "identity self-hash mismatch",
    )

    environment = identity.get("environment")
    _require(isinstance(environment, Mapping), "environment binding missing")
    _require(
        environment.get("profile_id") == "linux-x86_64",
        "environment profile drift",
    )
    _require(environment.get("python_version") == "3.11.16", "Python lock drift")
    environment_hash = environment.get("environment_evidence_sha256")
    _require(
        isinstance(environment_hash, str)
        and _SHA256.fullmatch(environment_hash) is not None,
        "environment evidence hash invalid",
    )

    fixture = identity.get("fixture")
    _require(isinstance(fixture, Mapping), "fixture block missing")
    _require(fixture.get("purpose") == FIXTURE_PURPOSE, "fixture authority overclaim")
    _require(
        fixture.get("tokenizer_vocab_size") == FIXTURE_TOKENIZER_VOCAB,
        "fixture vocab drift",
    )
    max_emitted = fixture.get("max_emitted_token_id")
    _require(
        isinstance(max_emitted, int)
        and not isinstance(max_emitted, bool)
        and 0 <= max_emitted < expected["model_vocab_size"],
        "fixture token range invalid",
    )
    _require(
        fixture.get("unused_model_vocab_rows")
        == expected["model_vocab_size"] - FIXTURE_TOKENIZER_VOCAB,
        "unused-vocab accounting drift",
    )
    train_ids = fixture.get("train_record_ids")
    validation_ids = fixture.get("validation_record_ids")
    _require(
        isinstance(train_ids, list) and isinstance(validation_ids, list),
        "fixture split IDs missing",
    )
    _require(not (set(train_ids) & set(validation_ids)), "fixture split overlap")

    recipe = evidence.get("recipe")
    _require(isinstance(recipe, Mapping), "recipe block missing")
    _require(recipe.get("device") == "cpu", "recipe device drift")
    _require(recipe.get("precision") == "fp32", "recipe precision drift")
    _require(recipe.get("sequence_length") == 128, "recipe sequence length drift")
    _require(recipe.get("train_batch_limit") == 2, "recipe train batch limit drift")
    _require(
        recipe.get("validation_batch_limit") == 1,
        "recipe validation batch limit drift",
    )
    _require(recipe.get("batch_size") == expected["batch_size"], "batch size drift")
    _require(
        recipe.get("optimizer_steps_requested") == expected["optimizer_steps"],
        "requested optimizer steps drift",
    )
    _require(
        recipe.get("learning_rate") == expected["learning_rate"],
        "learning rate drift",
    )

    training = evidence.get("training")
    _require(isinstance(training, Mapping), "training block missing")
    _require(training.get("status") == "PASS", "training did not pass")
    _require(
        training.get("optimizer_steps") == expected["optimizer_steps"],
        "optimizer step count mismatch",
    )
    _require(
        training.get("microbatches_consumed") == expected["optimizer_steps"],
        "microbatch count mismatch",
    )
    optimized_tokens = training.get("optimized_tokens")
    _require(
        isinstance(optimized_tokens, int)
        and not isinstance(optimized_tokens, bool)
        and optimized_tokens > 0,
        "optimized token count invalid",
    )
    _require(
        training.get("validation_optimized_tokens") == 0,
        "validation tokens were optimized",
    )
    _require(
        training.get("optimizer_step_before_final_validation")
        == training.get("optimizer_step_after_final_validation"),
        "validation mutated optimizer state",
    )
    for field in (
        "initial_train_loss",
        "final_train_loss",
        "initial_validation_loss",
        "final_validation_loss",
        "gradient_norm_min",
        "gradient_norm_max",
    ):
        _finite(training.get(field), f"training.{field}", nonnegative=True)
    _require(
        _finite(training.get("final_train_loss"), "training.final_train_loss")
        <= 4.0
        * _finite(training.get("initial_train_loss"), "training.initial_train_loss"),
        "train loss catastrophically diverged",
    )
    delta = training.get("weight_delta")
    _require(isinstance(delta, Mapping), "weight delta missing")
    changed = delta.get("changed_parameter_elements")
    _require(
        isinstance(changed, int) and not isinstance(changed, bool) and changed > 0,
        "no trainable parameter changed",
    )
    _require(
        delta.get("trainable_parameter_elements") == expected["parameter_count"],
        "weight delta parameter count mismatch",
    )
    _finite(delta.get("l2"), "weight_delta.l2", nonnegative=True)
    _finite(delta.get("max_abs"), "weight_delta.max_abs", nonnegative=True)

    resources = evidence.get("resources")
    _require(isinstance(resources, Mapping), "resources block missing")
    parameter_bytes = resources.get("model_parameter_bytes")
    _require(
        parameter_bytes == expected["parameter_count"] * 4,
        "fp32 model parameter bytes mismatch",
    )
    optimizer_bytes = resources.get("optimizer_tensor_bytes_after_training")
    _require(
        isinstance(optimizer_bytes, int)
        and not isinstance(optimizer_bytes, bool)
        and optimizer_bytes > parameter_bytes,
        "optimizer tensor bytes invalid",
    )
    _require(
        resources.get("measurement_snapshot_bytes") == parameter_bytes,
        "measurement snapshot bytes mismatch",
    )
    _require(
        resources.get("observed_tensor_bytes_with_snapshot")
        == parameter_bytes * 2 + optimizer_bytes,
        "observed tensor-byte accounting mismatch",
    )

    runtime = evidence.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime block missing")
    _require(runtime.get("device") == "cpu", "runtime device drift")
    _require(runtime.get("precision") == "fp32", "runtime precision drift")
    wall = _finite(
        runtime.get("wall_seconds_training_only"),
        "runtime.wall_seconds_training_only",
        nonnegative=True,
    )
    _require(wall > 0.0, "training wall time must be positive")
    _finite(
        runtime.get("process_cpu_seconds_training_only"),
        "runtime.process_cpu_seconds_training_only",
        nonnegative=True,
    )
    throughput = _finite(
        runtime.get("optimized_tokens_per_wall_second"),
        "runtime.optimized_tokens_per_wall_second",
        nonnegative=True,
    )
    _require(throughput > 0.0, "training throughput must be positive")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims block missing")
    for key in (
        "stage_architecture_frozen",
        "stage_corpus_or_tokenizer_frozen",
        "stage_quality_or_capability_evidence",
        "candidate_or_stable_promotion",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
        "gpu_or_distributed_execution",
        "cross_hardware_bitwise_reproducibility",
    ):
        _require(claims.get(key) is False, f"prohibited claim enabled: {key}")

    claimed_hash = evidence.get("evidence_sha256")
    _require(
        isinstance(claimed_hash, str) and _SHA256.fullmatch(claimed_hash) is not None,
        "evidence self-hash missing",
    )
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    _require(_canonical_hash(unhashed) == claimed_hash, "evidence self-hash mismatch")
