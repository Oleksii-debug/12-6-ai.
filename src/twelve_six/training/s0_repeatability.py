"""Fresh-run determinism and seed-causality evidence for canonical S0 training.

This module composes the existing D01/D02/D03/D04 contracts. It does not change
model, dataset, tokenizer, packing, checkpoint, evaluation, or promotion semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .s0_evidence import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    TRAIN_JSONL_SHA256,
    VALIDATION_JSONL_SHA256,
    _evaluate,
    _sha256_file,
    _tensor_batches,
    _weight_delta,
)
from .s0_evidence_contract import (
    INIT_SPEC_SHA256,
    MODEL_SPEC_SHA256,
    PARAMETER_COUNT,
    REPOSITORY,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .trainer import StepMetrics, Trainer

PROBE_SCHEMA_VERSION = "12-6.s0-determinism-probe.v1"
REPEATABILITY_SCHEMA_VERSION = "12-6.s0-repeatability-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
_SHA256_HEX = frozenset("0123456789abcdef")


class S0RepeatabilityError(ValueError):
    """Raised when determinism evidence fails closed."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0RepeatabilityError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _validate_source_sha(source_sha: Any) -> str:
    _require(
        isinstance(source_sha, str)
        and len(source_sha) == 40
        and set(source_sha) <= _SHA256_HEX,
        "source SHA must be a full lowercase Git SHA",
    )
    return source_sha


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite")
    if nonnegative:
        _require(number >= 0.0, f"{field} must be non-negative")
    return number


def _state_hash(value: Any) -> str:
    """Hash nested trainer/model state with tensor bytes and explicit type framing."""
    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            metadata = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            digest.update(b"tensor:")
            digest.update(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
            digest.update(b"\0")
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            ordered = sorted(
                item.items(),
                key=lambda pair: json.dumps(
                    pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ),
            )
            for key, child in ordered:
                feed(key)
                feed(child)
            digest.update(b"}")
            return
        if isinstance(item, tuple):
            digest.update(b"tuple[")
            for child in item:
                feed(child)
            digest.update(b"]")
            return
        if isinstance(item, list):
            digest.update(b"list[")
            for child in item:
                feed(child)
            digest.update(b"]")
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(b"\0")
            return
        raise TypeError(f"unsupported repeatability state type: {type(item).__name__}")

    feed(value)
    return digest.hexdigest()


def _stable_probe_payload(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": probe["identity"],
        "split_isolation": probe["split_isolation"],
        "training": probe["training"],
        "state_fingerprints": probe["state_fingerprints"],
        "step_trace": probe["step_trace"],
        "claims": probe["claims"],
    }


def run_s0_determinism_probe(
    root: str | Path,
    *,
    source_sha: str,
    seed: int,
    max_steps: int = 40,
    batch_size: int = 3,
) -> dict[str, Any]:
    """Execute one real canonical S0 run and emit timing-free deterministic evidence."""
    source_sha = _validate_source_sha(source_sha)
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    _require(0 <= seed < 2**63, "seed must be in [0, 2**63)")
    _require(
        isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0,
        "max_steps must be a positive integer",
    )
    _require(
        isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
        "batch_size must be a positive integer",
    )

    root = Path(root).resolve()
    stage_path = root / "configs/stages/s0_10k.json"
    dataset_path = root / "data/s0/packaged/manifest.json"
    train_path = root / "data/s0/packaged/train.jsonl"
    validation_path = root / "data/s0/packaged/validation.jsonl"

    _require(
        _sha256_file(dataset_path) == DATASET_MANIFEST_SHA256,
        "D03 dataset manifest SHA-256 mismatch",
    )
    _require(_sha256_file(train_path) == TRAIN_JSONL_SHA256, "D03 train split SHA-256 mismatch")
    _require(
        _sha256_file(validation_path) == VALIDATION_JSONL_SHA256,
        "D03 validation split SHA-256 mismatch",
    )
    dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    _require(
        dataset_manifest.get("dataset_identity_sha256") == DATASET_IDENTITY_SHA256,
        "D03 dataset semantic identity mismatch",
    )

    stage = load_stage_config(stage_path)
    tokenizer = ByteTokenizer()
    _require(stage.expected_parameters == PARAMETER_COUNT, "S0 parameter count mismatch")
    _require(stage.model.identity_sha256() == MODEL_SPEC_SHA256, "ModelSpec identity mismatch")
    _require(stage.init.identity_sha256() == INIT_SPEC_SHA256, "InitSpec identity mismatch")
    _require(tokenizer.identity.config_sha256 == TOKENIZER_CONFIG_SHA256, "tokenizer config mismatch")
    _require(tokenizer.identity.vocab_sha256 == TOKENIZER_VOCAB_SHA256, "tokenizer vocab mismatch")
    _require(PACKING_CONFIG_HASH == "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285", "packing identity mismatch")
    _require(stage.model.vocab_size == tokenizer.vocab_size, "model/tokenizer vocab mismatch")

    train_batches, train_ids, train_tokens_per_epoch = _tensor_batches(
        root, split="train", tokenizer=tokenizer, batch_size=batch_size
    )
    validation_batches, validation_ids, validation_tokens = _tensor_batches(
        root, split="validation", tokenizer=tokenizer, batch_size=batch_size
    )
    _require(not (set(train_ids) & set(validation_ids)), "train/validation record identity overlap")

    config = TrainerConfig(
        learning_rate=3e-2,
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

    # Seed causality: the declared seed is applied before scratch model construction.
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_model_sha256 = _state_hash(model.state_dict())
    initial_weights = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    trainer = Trainer(model, config, device="cpu")

    initial_train_loss, initial_train_eval_tokens = _evaluate(model, train_batches)
    initial_validation_loss, initial_validation_eval_tokens = _evaluate(
        model, validation_batches
    )
    validation_step_before_training = trainer.optimizer_step

    step_metrics: list[StepMetrics] = []
    run_result = trainer.run(
        islice(cycle(train_batches), max_steps),
        on_metrics=step_metrics.append,
    )

    final_train_loss, final_train_eval_tokens = _evaluate(model, train_batches)
    validation_step_before_final_eval = trainer.optimizer_step
    final_validation_loss, final_validation_eval_tokens = _evaluate(
        model, validation_batches
    )
    validation_step_after_final_eval = trainer.optimizer_step

    grad_norms = [metric.grad_norm for metric in step_metrics if metric.grad_norm is not None]
    _require(grad_norms and all(math.isfinite(value) for value in grad_norms), "non-finite gradient trace")
    _require(final_train_loss < initial_train_loss, "real S0 train loss did not decrease")
    _require(
        all(
            math.isfinite(value)
            for value in (
                initial_train_loss,
                final_train_loss,
                initial_validation_loss,
                final_validation_loss,
            )
        ),
        "non-finite measured S0 loss",
    )

    weight_delta = _weight_delta(model, initial_weights)
    _require(weight_delta["changed_parameter_elements"] > 0, "training changed no model weights")
    final_model_sha256 = _state_hash(model.state_dict())
    _require(final_model_sha256 != initial_model_sha256, "final model fingerprint equals initialization")

    trainer_state_sha256 = _state_hash(asdict(trainer.state_dict()))
    trace_payload = [asdict(metric) for metric in step_metrics]
    step_trace_sha256 = hashlib.sha256(
        json.dumps(
            trace_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S0",
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "train_jsonl_sha256": TRAIN_JSONL_SHA256,
        "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
        "seed": seed,
        "training_config": asdict(config),
        "batch_size_examples": batch_size,
    }
    probe: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "split_isolation": {
            "optimized_split": "train",
            "train_record_ids": list(train_ids),
            "validation_record_ids": list(validation_ids),
            "record_id_overlap": [],
            "train_tokens_per_full_epoch": train_tokens_per_epoch,
            "validation_scoreable_tokens": validation_tokens,
            "validation_optimizer_step_before_training": validation_step_before_training,
            "validation_optimizer_step_before_final_eval": validation_step_before_final_eval,
            "validation_optimizer_step_after_final_eval": validation_step_after_final_eval,
            "validation_optimized_tokens": 0,
        },
        "training": {
            "optimizer_steps": run_result.optimizer_steps_completed,
            "microbatches_consumed": run_result.microbatches_consumed,
            "optimized_tokens": run_result.tokens_consumed,
            "trainer_tokens_seen": trainer.tokens_seen,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "initial_train_eval_tokens": initial_train_eval_tokens,
            "final_train_eval_tokens": final_train_eval_tokens,
            "initial_validation_eval_tokens": initial_validation_eval_tokens,
            "final_validation_eval_tokens": final_validation_eval_tokens,
            "gradient_norm_min": min(grad_norms),
            "gradient_norm_max": max(grad_norms),
            "weight_delta": weight_delta,
        },
        "state_fingerprints": {
            "initial_model_sha256": initial_model_sha256,
            "final_model_sha256": final_model_sha256,
            "final_trainer_state_sha256": trainer_state_sha256,
        },
        "step_trace": {
            "entries": len(trace_payload),
            "sha256": step_trace_sha256,
        },
        "claims": {
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
        },
    }
    probe["stable_result_sha256"] = _canonical_hash(_stable_probe_payload(probe))
    probe["probe_sha256"] = _canonical_hash(probe)
    validate_determinism_probe(probe)
    return probe


def validate_determinism_probe(probe: Mapping[str, Any]) -> None:
    """Validate one timing-free deterministic S0 probe."""
    _require(probe.get("schema_version") == PROBE_SCHEMA_VERSION, "wrong probe schema")
    _require(probe.get("authority") == AUTHORITY, "wrong probe authority")

    identity = probe.get("identity")
    _require(isinstance(identity, Mapping), "probe identity missing")
    _require(identity.get("repository") == REPOSITORY, "probe repository mismatch")
    _validate_source_sha(identity.get("source_sha"))
    expected = {
        "stage": "S0",
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "train_jsonl_sha256": TRAIN_JSONL_SHA256,
        "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
    }
    for key, value in expected.items():
        _require(identity.get(key) == value, f"probe {key} mismatch")
    _require(
        isinstance(identity.get("seed"), int) and not isinstance(identity.get("seed"), bool),
        "probe seed invalid",
    )
    training_config = identity.get("training_config")
    _require(isinstance(training_config, Mapping), "probe training config missing")
    _require(training_config.get("seed") == identity.get("seed"), "probe seed/config mismatch")
    _require(training_config.get("deterministic_algorithms") is True, "determinism disabled")
    _require(training_config.get("deterministic_warn_only") is False, "determinism is warn-only")
    _require(training_config.get("precision") == "fp32", "probe precision mismatch")
    _require(
        probe.get("identity_sha256") == _canonical_hash(identity),
        "probe identity self-hash mismatch",
    )

    split = probe.get("split_isolation")
    _require(isinstance(split, Mapping), "probe split isolation missing")
    _require(split.get("optimized_split") == "train", "probe optimized non-train split")
    _require(split.get("record_id_overlap") == [], "probe train/validation overlap")
    _require(split.get("validation_optimized_tokens") == 0, "probe optimized validation tokens")
    _require(
        split.get("validation_optimizer_step_before_final_eval")
        == split.get("validation_optimizer_step_after_final_eval"),
        "probe held-out evaluation mutated optimizer step",
    )

    training = probe.get("training")
    _require(isinstance(training, Mapping), "probe training block missing")
    max_steps = training_config.get("max_steps")
    _require(training.get("optimizer_steps") == max_steps, "probe optimizer-step count mismatch")
    _require(training.get("microbatches_consumed") == max_steps, "probe microbatch count mismatch")
    _require(
        isinstance(training.get("optimized_tokens"), int) and training["optimized_tokens"] > 0,
        "probe optimized-token count invalid",
    )
    _require(
        training.get("optimized_tokens") == training.get("trainer_tokens_seen"),
        "probe optimized-token accounting mismatch",
    )
    initial_train = _finite_number(training.get("initial_train_loss"), "probe initial train loss")
    final_train = _finite_number(training.get("final_train_loss"), "probe final train loss")
    _require(final_train < initial_train, "probe train loss did not decrease")
    _finite_number(training.get("initial_validation_loss"), "probe initial validation loss")
    _finite_number(training.get("final_validation_loss"), "probe final validation loss")
    grad_min = _finite_number(training.get("gradient_norm_min"), "probe gradient norm min", nonnegative=True)
    grad_max = _finite_number(training.get("gradient_norm_max"), "probe gradient norm max", nonnegative=True)
    _require(grad_max >= grad_min, "probe gradient norm range inverted")
    delta = training.get("weight_delta")
    _require(isinstance(delta, Mapping), "probe weight delta missing")
    _require(
        delta.get("trainable_parameter_elements") == PARAMETER_COUNT,
        "probe trainable-parameter cardinality mismatch",
    )
    _require(
        isinstance(delta.get("changed_parameter_elements"), int)
        and delta["changed_parameter_elements"] > 0,
        "probe changed no trainable parameter",
    )

    fingerprints = probe.get("state_fingerprints")
    _require(isinstance(fingerprints, Mapping), "probe state fingerprints missing")
    for field in (
        "initial_model_sha256",
        "final_model_sha256",
        "final_trainer_state_sha256",
    ):
        _require(_is_sha256(fingerprints.get(field)), f"probe {field} invalid")
    _require(
        fingerprints.get("initial_model_sha256") != fingerprints.get("final_model_sha256"),
        "probe model fingerprint did not change",
    )

    trace = probe.get("step_trace")
    _require(isinstance(trace, Mapping), "probe step trace missing")
    _require(trace.get("entries") == max_steps, "probe step trace length mismatch")
    _require(_is_sha256(trace.get("sha256")), "probe step trace SHA invalid")

    claims = probe.get("claims")
    _require(isinstance(claims, Mapping), "probe claims missing")
    for field in (
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
    ):
        _require(claims.get(field) is False, f"forbidden probe flag set: {field}")

    _require(
        probe.get("stable_result_sha256") == _canonical_hash(_stable_probe_payload(probe)),
        "probe stable-result hash mismatch",
    )
    unhashed = dict(probe)
    claimed_probe_hash = unhashed.pop("probe_sha256", None)
    _require(_is_sha256(claimed_probe_hash), "probe self-hash missing")
    _require(_canonical_hash(unhashed) == claimed_probe_hash, "probe self-hash mismatch")


def _validate_repeatability_relationships(
    same_seed_a: Mapping[str, Any],
    same_seed_b: Mapping[str, Any],
    different_seed: Mapping[str, Any],
) -> str:
    for probe in (same_seed_a, same_seed_b, different_seed):
        validate_determinism_probe(probe)
    source_sha = same_seed_a["identity"]["source_sha"]
    _require(same_seed_b["identity"]["source_sha"] == source_sha, "same-seed source mismatch")
    _require(different_seed["identity"]["source_sha"] == source_sha, "different-seed source mismatch")

    same_seed = same_seed_a["identity"]["seed"]
    _require(same_seed_b["identity"]["seed"] == same_seed, "A/B seeds differ")
    _require(different_seed["identity"]["seed"] != same_seed, "different-seed probe reused same seed")
    _require(
        same_seed_a["stable_result_sha256"] == same_seed_b["stable_result_sha256"],
        "same-seed stable results differ",
    )
    _require(
        same_seed_a["state_fingerprints"] == same_seed_b["state_fingerprints"],
        "same-seed state fingerprints differ",
    )
    _require(
        same_seed_a["step_trace"] == same_seed_b["step_trace"],
        "same-seed optimizer trace differs",
    )

    same_fingerprints = same_seed_a["state_fingerprints"]
    different_fingerprints = different_seed["state_fingerprints"]
    _require(
        same_fingerprints["initial_model_sha256"]
        != different_fingerprints["initial_model_sha256"],
        "different seed did not change initial model fingerprint",
    )
    _require(
        same_fingerprints["final_model_sha256"] != different_fingerprints["final_model_sha256"],
        "different seed did not change final model fingerprint",
    )
    _require(
        same_seed_a["step_trace"]["sha256"] != different_seed["step_trace"]["sha256"],
        "different seed did not change training trace",
    )
    return source_sha


def build_s0_repeatability_evidence(
    same_seed_a: Mapping[str, Any],
    same_seed_b: Mapping[str, Any],
    different_seed: Mapping[str, Any],
    locked_environment_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind three standalone probe outputs into one downstream-consumable proof."""
    source_sha = _validate_repeatability_relationships(
        same_seed_a, same_seed_b, different_seed
    )
    environment_binding = validate_locked_environment_evidence(
        locked_environment_evidence, source_sha=source_sha
    )
    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S0",
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
        "same_seed": same_seed_a["identity"]["seed"],
        "different_seed": different_seed["identity"]["seed"],
        "environment": environment_binding,
    }
    evidence: dict[str, Any] = {
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "execution_contract": {
            "probe_invocation": "standalone_cli_process",
            "required_probe_count": 3,
            "same_seed_fresh_runs": 2,
            "different_seed_runs": 1,
            "timing_fields_excluded_from_probe": True,
        },
        "proof": {
            "same_seed_stable_result_sha256": same_seed_a["stable_result_sha256"],
            "same_seed_initial_model_sha256": same_seed_a["state_fingerprints"]["initial_model_sha256"],
            "same_seed_final_model_sha256": same_seed_a["state_fingerprints"]["final_model_sha256"],
            "same_seed_final_trainer_state_sha256": same_seed_a["state_fingerprints"]["final_trainer_state_sha256"],
            "same_seed_step_trace_sha256": same_seed_a["step_trace"]["sha256"],
            "different_seed_initial_model_sha256": different_seed["state_fingerprints"]["initial_model_sha256"],
            "different_seed_final_model_sha256": different_seed["state_fingerprints"]["final_model_sha256"],
            "different_seed_step_trace_sha256": different_seed["step_trace"]["sha256"],
            "same_seed_exact_equivalence": True,
            "different_seed_initialization_diverges": True,
            "different_seed_training_diverges": True,
            "validation_optimized_tokens": 0,
        },
        "probes": {
            "same_seed_a": dict(same_seed_a),
            "same_seed_b": dict(same_seed_b),
            "different_seed": dict(different_seed),
        },
        "locked_environment_evidence": dict(locked_environment_evidence),
        "claims": {
            "cross_hardware_bitwise_reproducibility": False,
            "gpu_reproducibility": False,
            "distributed_reproducibility": False,
            "candidate_or_stable_promotion": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_s0_repeatability_evidence(evidence)
    return evidence


def validate_s0_repeatability_evidence(evidence: Mapping[str, Any]) -> None:
    """Validate self-contained same-seed repeatability and seed-causality evidence."""
    _require(
        evidence.get("schema_version") == REPEATABILITY_SCHEMA_VERSION,
        "wrong repeatability schema",
    )
    _require(evidence.get("authority") == AUTHORITY, "wrong repeatability authority")

    probes = evidence.get("probes")
    _require(isinstance(probes, Mapping), "repeatability probes missing")
    same_seed_a = probes.get("same_seed_a")
    same_seed_b = probes.get("same_seed_b")
    different_seed = probes.get("different_seed")
    _require(isinstance(same_seed_a, Mapping), "same-seed A probe missing")
    _require(isinstance(same_seed_b, Mapping), "same-seed B probe missing")
    _require(isinstance(different_seed, Mapping), "different-seed probe missing")
    source_sha = _validate_repeatability_relationships(
        same_seed_a, same_seed_b, different_seed
    )

    locked_environment = evidence.get("locked_environment_evidence")
    _require(isinstance(locked_environment, Mapping), "locked environment evidence missing")
    environment_binding = validate_locked_environment_evidence(
        locked_environment, source_sha=source_sha
    )

    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "repeatability identity missing")
    expected_identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S0",
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_HASH,
        "same_seed": same_seed_a["identity"]["seed"],
        "different_seed": different_seed["identity"]["seed"],
        "environment": environment_binding,
    }
    _require(dict(identity) == expected_identity, "repeatability identity mismatch")
    _require(
        evidence.get("identity_sha256") == _canonical_hash(identity),
        "repeatability identity self-hash mismatch",
    )

    execution = evidence.get("execution_contract")
    _require(isinstance(execution, Mapping), "execution contract missing")
    _require(execution.get("probe_invocation") == "standalone_cli_process", "probe process contract mismatch")
    _require(execution.get("required_probe_count") == 3, "probe count contract mismatch")
    _require(execution.get("same_seed_fresh_runs") == 2, "same-seed run count mismatch")
    _require(execution.get("different_seed_runs") == 1, "different-seed run count mismatch")
    _require(execution.get("timing_fields_excluded_from_probe") is True, "timing exclusion not proven")

    proof = evidence.get("proof")
    _require(isinstance(proof, Mapping), "repeatability proof missing")
    required_proof = {
        "same_seed_stable_result_sha256": same_seed_a["stable_result_sha256"],
        "same_seed_initial_model_sha256": same_seed_a["state_fingerprints"]["initial_model_sha256"],
        "same_seed_final_model_sha256": same_seed_a["state_fingerprints"]["final_model_sha256"],
        "same_seed_final_trainer_state_sha256": same_seed_a["state_fingerprints"]["final_trainer_state_sha256"],
        "same_seed_step_trace_sha256": same_seed_a["step_trace"]["sha256"],
        "different_seed_initial_model_sha256": different_seed["state_fingerprints"]["initial_model_sha256"],
        "different_seed_final_model_sha256": different_seed["state_fingerprints"]["final_model_sha256"],
        "different_seed_step_trace_sha256": different_seed["step_trace"]["sha256"],
        "same_seed_exact_equivalence": True,
        "different_seed_initialization_diverges": True,
        "different_seed_training_diverges": True,
        "validation_optimized_tokens": 0,
    }
    _require(dict(proof) == required_proof, "repeatability proof summary mismatch")

    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "repeatability claims missing")
    for field in (
        "cross_hardware_bitwise_reproducibility",
        "gpu_reproducibility",
        "distributed_reproducibility",
        "candidate_or_stable_promotion",
    ):
        _require(claims.get(field) is False, f"forbidden repeatability claim set: {field}")

    unhashed = dict(evidence)
    claimed_hash = unhashed.pop("evidence_sha256", None)
    _require(_is_sha256(claimed_hash), "repeatability evidence self-hash missing")
    _require(_canonical_hash(unhashed) == claimed_hash, "repeatability evidence self-hash mismatch")
