"""TRAIN-53 fixed-268K batch-noise and effective-batch experiment.

This experiment composes two incumbents as real Git ancestry:

* TRAIN-29 ``TrainingObserver`` for bounded training/evaluation telemetry;
* DATA-21/22 rights-aware bounded external-source intake for real Ukrainian/English
  text bytes explicitly approved by that experimental registry for model training.

The DATA-21/22 sample is intentionally small and is not a canonical D03 external
snapshot or a representative broad pretraining corpus.  TRAIN-53 therefore treats
its batch recommendation as a provisional local starting point, not a theoretically
exact critical batch size and not a corpus/model promotion decision.
"""

from __future__ import annotations

import math
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.data.source_intake import (
    ELIGIBLE,
    load_candidate_registry,
    run_bounded_intake,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import TextRecord, batch_examples, collate_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

from .batch_noise import diagnose_gradient_noise
from .config import TrainerConfig
from .observability import TrainingObserver
from .s0_evidence_contract import (
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .s1_preflight import REPOSITORY, _evaluate
from .trainer import Trainer

SCHEMA_VERSION = "12-6.train53-batch-noise.v2"
AUTHORITY = (
    "LOCAL_FREE_FIXED_268K_REAL_BOUNDED_SOURCE_BATCH_NOISE_EVIDENCE_"
    "NOT_REPRESENTATIVE_CORPUS_OR_PAID_CAPACITY"
)
PARAMETER_COUNT = 267_912
BATCH_SIZE_EXAMPLES = 4
SEQUENCE_LENGTH = 64
BASE_LOSS_TOKENS = BATCH_SIZE_EXAMPLES * (SEQUENCE_LENGTH - 1)
ACCUMULATION_CANDIDATES = (1, 2, 4, 8)
TOTAL_MICROBATCHES = 256
CHECKPOINT_MICROBATCHES = (64, 128, 256)
DIAGNOSTIC_CHECKPOINT_MICROBATCHES = (0, 64, 128, 256)
DIAGNOSTIC_SAMPLES = 16
REAL_SOURCE_REGISTRY = Path("configs/data/external_source_candidates_ua_en_v1.json")
VALIDATION_SOURCE_ID = "en.standardebooks.manual"
VALIDATION_URL_SUFFIX = "/9-metadata.rst"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class BatchNoiseProbeError(ValueError):
    """Raised when TRAIN-53 evidence cannot satisfy its fixed-control contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchNoiseProbeError(message)


def fixed_268k_model_spec() -> ModelSpec:
    """Return the established RESEARCH41 267,912-parameter fixed-control member."""
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=72,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        head_dim=12,
        d_ff=192,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=12,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )
    if spec.parameter_count() != PARAMETER_COUNT:
        raise RuntimeError("fixed 268K model geometry drift")
    return spec


def _trainer_config(*, accumulation: int, max_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=accumulation,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _real_corpus_records(
    root: Path,
    intake_output: Path,
) -> tuple[list[TextRecord], list[TextRecord], dict[str, Any]]:
    """Acquire and split the exact DATA-21/22 bounded real-source sample.

    The split is predeclared by exact source identity: the pinned Standard Ebooks
    ``9-metadata.rst`` object is validation-only; the Rada object and pinned
    ``8-typography.rst`` object are training.  We require every eligible acquisition
    object to succeed so transient source failures cannot silently change the corpus.
    """
    registry_path = root / REAL_SOURCE_REGISTRY
    registry, sources = load_candidate_registry(registry_path)
    eligible_sources = [source for source in sources if source.eligibility_status == ELIGIBLE]
    _require(len(eligible_sources) == 2, "TRAIN-53 expects exactly two DATA-21/22 eligible sources")
    _require(
        all(source.rights.allows_model_training is True for source in eligible_sources),
        "all TRAIN-53 real sources must be explicitly approved for model training",
    )
    expected_attempts = sum(len(source.acquisition_urls) for source in eligible_sources)
    _require(expected_attempts == 3, "TRAIN-53 expects the three bounded DATA-21/22 objects")

    manifest = run_bounded_intake(registry, intake_output)
    _require(
        manifest.get("authority_boundary")
        == "REAL_BOUNDED_SAMPLE_NOT_CANONICAL_CORPUS_FREEZE_OR_SOURCE_SNAPSHOT_PROMOTION",
        "real-source intake authority boundary drift",
    )
    counts = manifest.get("record_counts")
    _require(isinstance(counts, Mapping), "real-source intake record counts missing")
    _require(counts.get("attempted") == expected_attempts, "real-source attempt count drift")
    _require(counts.get("accepted") == expected_attempts, "every eligible real-source object must be accepted")
    _require(counts.get("rejected") == 0, "real-source acquisition contained a rejection")
    _require(counts.get("exact_duplicates") == 0, "real-source acquisition unexpectedly deduplicated an object")

    accepted = [
        record
        for record in manifest.get("records", [])
        if isinstance(record, Mapping) and record.get("status") == "ACCEPTED"
    ]
    _require(len(accepted) == expected_attempts, "accepted real-source record list mismatch")
    _require(
        all(record.get("allows_model_training") is True for record in accepted),
        "accepted real-source record lacks model-training permission",
    )
    _require(
        {record.get("language") for record in accepted} == {"uk", "en"},
        "TRAIN-53 real-source sample must contain Ukrainian and English",
    )

    validation_meta = [
        record
        for record in accepted
        if record.get("source_id") == VALIDATION_SOURCE_ID
        and str(record.get("acquisition_url", "")).endswith(VALIDATION_URL_SUFFIX)
    ]
    _require(len(validation_meta) == 1, "predeclared real validation object missing or duplicated")
    validation_id = str(validation_meta[0]["id"])

    train_records: list[TextRecord] = []
    validation_records: list[TextRecord] = []
    accepted_summary: list[dict[str, Any]] = []
    for record in accepted:
        record_id = str(record["id"])
        text_path = intake_output / str(record["text_path"])
        _require(text_path.is_file(), f"accepted real-source text missing: {record_id}")
        text = text_path.read_text(encoding="utf-8").rstrip("\n")
        _require(bool(text), f"accepted real-source text empty: {record_id}")
        split = "validation" if record_id == validation_id else "train"
        target = validation_records if split == "validation" else train_records
        target.append(TextRecord(record_id=record_id, text=text, split=split))
        accepted_summary.append(
            {
                "record_id": record_id,
                "source_id": record["source_id"],
                "source_version": record["source_version"],
                "source_identity_sha256": record["source_identity_sha256"],
                "acquisition_url": record["acquisition_url"],
                "raw_sha256": record["raw_sha256"],
                "content_sha256": record["content_sha256"],
                "normalized_utf8_bytes": record["normalized_utf8_bytes"],
                "language": record["language"],
                "rights_status": record["rights_status"],
                "license_id": record["license_id"],
                "allows_model_training": record["allows_model_training"],
                "split": split,
            }
        )

    _require(len(train_records) == 2, "TRAIN-53 requires exactly two real training documents")
    _require(len(validation_records) == 1, "TRAIN-53 requires exactly one real validation document")
    _require(
        {record.record_id for record in train_records}.isdisjoint(
            record.record_id for record in validation_records
        ),
        "real train/validation document overlap",
    )

    split_identity_core = {
        "candidate_registry_identity_sha256": registry["registry_identity_sha256"],
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "accepted_records": accepted_summary,
        "split_policy": {
            "validation_source_id": VALIDATION_SOURCE_ID,
            "validation_url_suffix": VALIDATION_URL_SUFFIX,
            "document_isolation": True,
        },
    }
    dataset_identity_sha256 = hash_json(split_identity_core)
    return train_records, validation_records, {
        "registry_file_sha256": sha256_file(registry_path),
        "candidate_registry_identity_sha256": registry["registry_identity_sha256"],
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "dataset_identity_sha256": dataset_identity_sha256,
        "accepted_records": accepted_summary,
        "record_counts": dict(counts),
        "byte_counts": dict(manifest["byte_counts"]),
        "accepted_records_by_language": dict(manifest["accepted_records_by_language"]),
        "train_record_ids": [record.record_id for record in train_records],
        "validation_record_ids": [record.record_id for record in validation_records],
        "train_validation_record_overlap": [],
        "project_authored_synthetic_fixture": False,
        "real_external_source_bytes": True,
        "all_accepted_records_approved_for_model_training": True,
        "canonical_d03_external_snapshot_promoted": False,
        "representative_broad_pretraining_corpus": False,
        "source_scope_warning": (
            "DATA-21/22 supplies three rights-reviewed real external objects only. "
            "The Rada object is a bounded dated intake rather than an immutable D03 "
            "canonical snapshot, and validation is one pinned English document."
        ),
    }


def _tensor_batches_from_records(
    records: Sequence[TextRecord],
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
    sequence_length: int,
    full_only: bool,
) -> list[dict[str, torch.Tensor]]:
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=sequence_length,
        )
    )
    _require(bool(examples), f"{split} produced no packed examples")
    batches: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=full_only):
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        batch = {"input_ids": input_ids, "labels": labels}
        if full_only:
            valid_tokens = int(labels[:, 1:].ne(-100).sum().item())
            expected_tokens = batch_size * (sequence_length - 1)
            if input_ids.shape[0] != batch_size or valid_tokens != expected_tokens:
                continue
        batches.append(batch)
    _require(bool(batches), f"{split} produced no usable tensor batches")
    return batches


def _sample_indices(*, seed: int, population: int, count: int) -> list[int]:
    _require(population > 0, "sample population must be positive")
    rng = random.Random(seed)
    return [rng.randrange(population) for _ in range(count)]


def _diagnostic_at_checkpoint(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    checkpoint_microbatches: int,
    seed: int,
) -> dict[str, Any]:
    sample_seed = seed + 10_000 + checkpoint_microbatches * 97
    indices = _sample_indices(
        seed=sample_seed,
        population=len(train_batches),
        count=DIAGNOSTIC_SAMPLES,
    )
    sampled = [train_batches[index] for index in indices]
    diagnostic = diagnose_gradient_noise(
        model,
        trainer,
        sampled,
        effective_microbatch_counts=ACCUMULATION_CANDIDATES,
    )
    noise_scale = diagnostic["statistics"]["noise_scale_microbatches_proxy"]
    noise_scale_tokens = None
    if isinstance(noise_scale, (int, float)) and math.isfinite(float(noise_scale)):
        noise_scale_tokens = float(noise_scale) * BASE_LOSS_TOKENS
    return {
        "checkpoint_microbatches": checkpoint_microbatches,
        "checkpoint_optimized_tokens": trainer.tokens_seen,
        "checkpoint_optimizer_steps": trainer.optimizer_step,
        "sample_seed": sample_seed,
        "sample_indices_sha256": hash_json({"indices": indices}),
        "sampled_with_replacement": True,
        "independent_prng_draws": True,
        "base_microbatch_loss_tokens": BASE_LOSS_TOKENS,
        "local_noise_scale_loss_tokens_proxy": noise_scale_tokens,
        "diagnostic": diagnostic,
    }


def _candidate_run(
    *,
    source_sha: str,
    environment: Mapping[str, str],
    dataset_identity_sha256: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    validation_batches: list[dict[str, torch.Tensor]],
    trace_indices: Sequence[int],
    accumulation: int,
    seed: int,
    collect_diagnostics: bool,
) -> dict[str, Any]:
    _require(TOTAL_MICROBATCHES % accumulation == 0, "candidate does not divide fixed microbatch budget")
    max_steps = TOTAL_MICROBATCHES // accumulation
    config = _trainer_config(accumulation=accumulation, max_steps=max_steps, seed=seed)
    run_identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "experiment": SCHEMA_VERSION,
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "environment": dict(environment),
        "dataset_identity_sha256": dataset_identity_sha256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "batch_size_examples": BATCH_SIZE_EXAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "base_loss_tokens_per_microbatch": BASE_LOSS_TOKENS,
        "gradient_accumulation_steps": accumulation,
        "effective_loss_tokens_per_update": BASE_LOSS_TOKENS * accumulation,
        "fixed_microbatch_budget": TOTAL_MICROBATCHES,
        "training_trace_sha256": hash_json({"indices": list(trace_indices)}),
        "training_config": asdict(config),
        "seed": seed,
    }
    observer = TrainingObserver(
        run_identity,
        device="cpu",
        max_step_samples=TOTAL_MICROBATCHES,
        gpu_sample_every_steps=TOTAL_MICROBATCHES + 1,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")

    initial_validation_loss, validation_tokens = observer.measure_region(
        "evaluation",
        "validation_at_microbatch_0",
        lambda: _evaluate(model, validation_batches),
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )
    checkpoints: list[dict[str, Any]] = [
        {
            "microbatches_consumed": 0,
            "optimized_tokens": 0,
            "optimizer_steps": 0,
            "validation_loss": initial_validation_loss,
        }
    ]
    diagnostics: list[dict[str, Any]] = []
    if collect_diagnostics:
        diagnostics.append(
            _diagnostic_at_checkpoint(
                model=model,
                trainer=trainer,
                train_batches=train_batches,
                checkpoint_microbatches=0,
                seed=seed,
            )
        )

    last_metrics = None
    for microbatch_number, batch_index in enumerate(trace_indices, 1):
        last_metrics = observer.train_microbatch(
            trainer,
            train_batches[batch_index],
            data_wait_seconds=0.0,
        )
        if microbatch_number in CHECKPOINT_MICROBATCHES:
            trainer.assert_checkpoint_safe()
            validation_loss, checked_tokens = observer.measure_region(
                "evaluation",
                f"validation_at_microbatch_{microbatch_number}",
                lambda: _evaluate(model, validation_batches),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            )
            _require(checked_tokens == validation_tokens, "validation token count drift")
            checkpoints.append(
                {
                    "microbatches_consumed": microbatch_number,
                    "optimized_tokens": trainer.tokens_seen,
                    "optimizer_steps": trainer.optimizer_step,
                    "validation_loss": validation_loss,
                    "last_microbatch_loss": last_metrics.loss,
                    "last_update_loss": last_metrics.update_loss,
                    "last_preclip_gradient_norm": last_metrics.grad_norm,
                }
            )
            if collect_diagnostics:
                diagnostics.append(
                    _diagnostic_at_checkpoint(
                        model=model,
                        trainer=trainer,
                        train_batches=train_batches,
                        checkpoint_microbatches=microbatch_number,
                        seed=seed,
                    )
                )

    trainer.assert_checkpoint_safe()
    expected_tokens = TOTAL_MICROBATCHES * BASE_LOSS_TOKENS
    _require(trainer.tokens_seen == expected_tokens, "fixed token budget drift")
    _require(trainer.optimizer_step == max_steps, "optimizer-step count drift")
    summary = observer.summary()
    training_seconds = float(summary["timing"]["training_observed_seconds"])
    _require(training_seconds > 0.0, "training wall time must be positive")
    return {
        "gradient_accumulation_steps": accumulation,
        "effective_loss_tokens_per_update": BASE_LOSS_TOKENS * accumulation,
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "checkpoints": checkpoints,
        "diagnostics": diagnostics,
        "training_wall_seconds": training_seconds,
        "optimized_tokens_per_training_second": trainer.tokens_seen / training_seconds,
        "evaluation_wall_seconds": float(summary["timing"]["evaluation_seconds_total"]),
        "telemetry": observer.export(),
    }


def _recommend(candidates: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finals: list[dict[str, Any]] = []
    for candidate in candidates:
        final_checkpoint = candidate["checkpoints"][-1]
        finals.append(
            {
                "gradient_accumulation_steps": int(candidate["gradient_accumulation_steps"]),
                "effective_loss_tokens_per_update": int(candidate["effective_loss_tokens_per_update"]),
                "optimizer_steps": int(candidate["optimizer_steps"]),
                "final_validation_loss": float(final_checkpoint["validation_loss"]),
                "training_wall_seconds": float(candidate["training_wall_seconds"]),
                "optimized_tokens_per_training_second": float(
                    candidate["optimized_tokens_per_training_second"]
                ),
            }
        )
    best_loss = min(point["final_validation_loss"] for point in finals)
    quality_tolerance_relative = 0.005
    acceptable = [
        point
        for point in finals
        if point["final_validation_loss"] <= best_loss * (1.0 + quality_tolerance_relative)
    ]
    fastest_wall = min(point["training_wall_seconds"] for point in acceptable)
    speed_tolerance_relative = 0.05
    near_fastest = [
        point
        for point in acceptable
        if point["training_wall_seconds"] <= fastest_wall * (1.0 + speed_tolerance_relative)
    ]
    selected = min(near_fastest, key=lambda point: point["gradient_accumulation_steps"])

    noise_values: list[float] = []
    noise_token_values: list[float] = []
    for checkpoint in diagnostics:
        value = checkpoint["diagnostic"]["statistics"]["noise_scale_microbatches_proxy"]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            noise_values.append(float(value))
        token_value = checkpoint.get("local_noise_scale_loss_tokens_proxy")
        if isinstance(token_value, (int, float)) and math.isfinite(float(token_value)):
            noise_token_values.append(float(token_value))

    def median(values: Sequence[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    current = next(point for point in finals if point["gradient_accumulation_steps"] == 1)
    selected_accumulation = int(selected["gradient_accumulation_steps"])
    if selected_accumulation == 1:
        classification = "CURRENT_EFFECTIVE_BATCH_NOT_SHOWN_TOO_SMALL"
    else:
        classification = "MEASURED_LARGER_EFFECTIVE_BATCH_IS_PRACTICAL"
    if selected_accumulation == max(ACCUMULATION_CANDIDATES):
        upper_boundary = "MEASURED_GRID_EDGE_REACHED_DO_NOT_CALL_THIS_AN_OPTIMUM"
    else:
        upper_boundary = "MEASURED_GRID_CONTAINS_LARGER_BATCH_CONTROL"

    return {
        "selection_rule": (
            "retain candidates within 0.5% of best final held-out loss; choose the fastest; "
            "within 5% of that wall time choose the smaller accumulation to avoid gratuitous batching"
        ),
        "quality_tolerance_relative": quality_tolerance_relative,
        "speed_tolerance_relative": speed_tolerance_relative,
        "current_effective_loss_tokens_per_update": current["effective_loss_tokens_per_update"],
        "recommended_gradient_accumulation_steps": selected_accumulation,
        "recommended_effective_loss_tokens_per_update": int(selected["effective_loss_tokens_per_update"]),
        "classification": classification,
        "grid_boundary_interpretation": upper_boundary,
        "median_local_noise_scale_microbatches_proxy": median(noise_values),
        "median_local_noise_scale_loss_tokens_proxy": median(noise_token_values),
        "exact_critical_batch_size_claim": False,
        "recommended_for_100k_to_1m_campaigns": (
            "PROVISIONAL_STARTING_POINT_FROM_FIXED_268K_REAL_BOUNDED_SAMPLE_RECHECK_AT_SCALE"
        ),
        "do_not_extrapolate_beyond_measured_effective_loss_tokens_without_reprobe": max(
            point["effective_loss_tokens_per_update"] for point in finals
        ),
        "candidate_final_comparison": finals,
    }


def run_batch_noise_probe(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    """Execute the complete LOCAL_FREE fixed-268K real-source batch experiment."""
    _require(_GIT_SHA.fullmatch(source_sha) is not None, "source SHA must be full lowercase Git SHA")
    _require(torch_threads > 0, "torch_threads must be positive")
    root = Path(root).resolve()
    environment = validate_locked_environment_evidence(locked_environment_evidence, source_sha=source_sha)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "byte tokenizer vocabulary drift")
    with tempfile.TemporaryDirectory(prefix="train53-real-intake-") as intake_tmp:
        train_records, validation_records, real_data = _real_corpus_records(
            root,
            Path(intake_tmp),
        )
        train_batches = _tensor_batches_from_records(
            train_records,
            split="train",
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE_EXAMPLES,
            sequence_length=SEQUENCE_LENGTH,
            full_only=True,
        )
        validation_batches = _tensor_batches_from_records(
            validation_records,
            split="validation",
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE_EXAMPLES,
            sequence_length=SEQUENCE_LENGTH,
            full_only=False,
        )

        _require(len(train_batches) >= DIAGNOSTIC_SAMPLES, "real train corpus yields too few full microbatches")
        for batch in train_batches:
            tokens = int(batch["labels"][:, 1:].ne(-100).sum().item())
            _require(tokens == BASE_LOSS_TOKENS, "training microbatch token geometry drift")

        trace_indices = _sample_indices(
            seed=seed + 1_001,
            population=len(train_batches),
            count=TOTAL_MICROBATCHES,
        )
        spec = fixed_268k_model_spec()
        init_spec = InitSpec()
        candidates = [
            _candidate_run(
                source_sha=source_sha,
                environment=environment,
                dataset_identity_sha256=real_data["dataset_identity_sha256"],
                spec=spec,
                init_spec=init_spec,
                train_batches=train_batches,
                validation_batches=validation_batches,
                trace_indices=trace_indices,
                accumulation=accumulation,
                seed=seed,
                collect_diagnostics=accumulation == 1,
            )
            for accumulation in ACCUMULATION_CANDIDATES
        ]

    diagnostics = next(
        candidate["diagnostics"]
        for candidate in candidates
        if candidate["gradient_accumulation_steps"] == 1
    )
    recommendation = _recommend(candidates, diagnostics)
    real_data["materialized_full_train_microbatches"] = len(train_batches)
    real_data["materialized_validation_batches"] = len(validation_batches)

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "experiment": SCHEMA_VERSION,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "init_spec": init_spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "environment": dict(environment),
        "dataset_identity_sha256": real_data["dataset_identity_sha256"],
        "candidate_registry_identity_sha256": real_data[
            "candidate_registry_identity_sha256"
        ],
        "intake_manifest_sha256": real_data["intake_manifest_sha256"],
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "seed": seed,
        "batch_size_examples": BATCH_SIZE_EXAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "base_loss_tokens_per_microbatch": BASE_LOSS_TOKENS,
        "accumulation_candidates": list(ACCUMULATION_CANDIDATES),
        "fixed_microbatch_budget": TOTAL_MICROBATCHES,
        "fixed_optimized_loss_token_budget": TOTAL_MICROBATCHES * BASE_LOSS_TOKENS,
        "checkpoint_microbatches": list(CHECKPOINT_MICROBATCHES),
        "diagnostic_sample_count": DIAGNOSTIC_SAMPLES,
        "training_trace_sha256": hash_json({"indices": trace_indices}),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": hash_json(identity),
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "data": real_data,
        "controls": {
            "optimizer_recipe": asdict(
                _trainer_config(accumulation=1, max_steps=TOTAL_MICROBATCHES, seed=seed)
            ),
            "only_candidate_control_changed": "gradient_accumulation_steps/effective_batch",
            "same_initialization_seed": True,
            "same_training_microbatch_trace": True,
            "same_total_optimized_loss_tokens": True,
            "learning_rate_scaled_with_batch": False,
            "validation_split_held_out_from_training": True,
            "diagnostics_run_only_at_committed_optimizer_boundaries": True,
        },
        "gradient_diagnostics": diagnostics,
        "batch_candidates": candidates,
        "recommendation": recommendation,
        "assumptions_and_limits": {
            "gradient_samples_drawn_with_replacement": True,
            "sample_draws_use_independent_prng_draws": True,
            "underlying_training_windows_can_repeat": True,
            "gradient_noise_proxy_assumes_local_exchangeability": True,
            "iid_document_independence_established": False,
            "gradient_stationarity_across_training_established": False,
            "validation_representativeness_established": False,
            "cpu_wall_time_predicts_gpu_batch_efficiency": False,
            "fixed_lr_batch_comparison_retunes_learning_rate": False,
            "exact_critical_batch_size_claim": False,
            "optimizer_state_mutated_by_diagnostic_probes": False,
            "diagnostic_metrics_enter_model_or_checkpoint_fingerprint": False,
        },
        "claims": {
            "broad_corpus_batch_optimum": False,
            "theoretical_critical_batch_size": False,
            "target_gpu_throughput_claim": False,
            "paid_compute_authorized_or_used": False,
            "100k_to_1m_transfer_is_provisional": True,
            "canonical_external_corpus_freeze": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    validate_batch_noise_probe(report)
    return report


def validate_batch_noise_probe(report: Mapping[str, Any]) -> None:
    """Fail closed on data/control drift, state mutation, non-finite metrics, or overclaim."""
    _require(report.get("schema_version") == SCHEMA_VERSION, "wrong TRAIN-53 schema")
    _require(report.get("authority") == AUTHORITY, "wrong TRAIN-53 authority")
    identity = report.get("identity")
    _require(isinstance(identity, Mapping), "identity missing")
    _require(identity.get("parameter_count") == PARAMETER_COUNT, "fixed model parameter drift")
    _require(identity.get("batch_size_examples") == BATCH_SIZE_EXAMPLES, "microbatch example count drift")
    _require(identity.get("sequence_length") == SEQUENCE_LENGTH, "sequence length drift")
    _require(identity.get("base_loss_tokens_per_microbatch") == BASE_LOSS_TOKENS, "base token count drift")
    _require(identity.get("accumulation_candidates") == list(ACCUMULATION_CANDIDATES), "candidate grid drift")
    _require(identity.get("fixed_microbatch_budget") == TOTAL_MICROBATCHES, "microbatch budget drift")
    _require(report.get("identity_sha256") == hash_json(identity), "identity hash mismatch")

    data = report.get("data")
    _require(isinstance(data, Mapping), "data block missing")
    _require(data.get("train_validation_record_overlap") == [], "held-out split overlap")
    _require(data.get("project_authored_synthetic_fixture") is False, "synthetic fixture is forbidden")
    _require(data.get("real_external_source_bytes") is True, "real external corpus evidence missing")
    _require(
        data.get("all_accepted_records_approved_for_model_training") is True,
        "training-rights evidence missing",
    )
    _require(data.get("canonical_d03_external_snapshot_promoted") is False, "corpus promotion overclaim")
    _require(data.get("representative_broad_pretraining_corpus") is False, "corpus scope overclaim")
    _require(len(data.get("train_record_ids", [])) == 2, "real train document count drift")
    _require(len(data.get("validation_record_ids", [])) == 1, "real validation document count drift")
    _require(
        int(data.get("materialized_full_train_microbatches", 0)) >= DIAGNOSTIC_SAMPLES,
        "real corpus yielded too few microbatches",
    )

    diagnostics = report.get("gradient_diagnostics")
    _require(
        isinstance(diagnostics, list)
        and len(diagnostics) == len(DIAGNOSTIC_CHECKPOINT_MICROBATCHES),
        "diagnostic checkpoint grid drift",
    )
    observed_checkpoints = [int(item["checkpoint_microbatches"]) for item in diagnostics]
    _require(
        observed_checkpoints == list(DIAGNOSTIC_CHECKPOINT_MICROBATCHES),
        "diagnostic checkpoints out of order",
    )
    for checkpoint in diagnostics:
        diagnostic = checkpoint.get("diagnostic")
        _require(isinstance(diagnostic, Mapping), "diagnostic payload missing")
        state = diagnostic.get("state_preservation")
        _require(isinstance(state, Mapping), "state-preservation proof missing")
        for field in (
            "model_state_unchanged",
            "optimizer_scheduler_counters_unchanged",
            "parameter_gradients_restored",
            "python_torch_cuda_rng_restored",
            "model_train_eval_mode_restored",
        ):
            _require(state.get(field) is True, f"diagnostic state preservation failed: {field}")
        statistics = diagnostic.get("statistics")
        _require(isinstance(statistics, Mapping), "gradient statistics missing")
        _require(statistics.get("sample_count") == DIAGNOSTIC_SAMPLES, "diagnostic sample count drift")
        _require(statistics.get("exact_critical_batch_size_claim") is False, "critical-batch overclaim")
        for field in ("signal_norm", "signal_squared", "trace_covariance_unbiased"):
            value = float(statistics[field])
            _require(math.isfinite(value) and value >= 0.0, f"non-finite gradient statistic: {field}")

    candidates = report.get("batch_candidates")
    _require(
        isinstance(candidates, list) and len(candidates) == len(ACCUMULATION_CANDIDATES),
        "candidate count drift",
    )
    expected_tokens = TOTAL_MICROBATCHES * BASE_LOSS_TOKENS
    for candidate, expected_accumulation in zip(candidates, ACCUMULATION_CANDIDATES, strict=True):
        _require(candidate.get("gradient_accumulation_steps") == expected_accumulation, "candidate order drift")
        _require(candidate.get("optimized_tokens") == expected_tokens, "fixed token budget mismatch")
        _require(
            candidate.get("optimizer_steps") == TOTAL_MICROBATCHES // expected_accumulation,
            "optimizer-step count mismatch",
        )
        _require(
            candidate.get("effective_loss_tokens_per_update")
            == BASE_LOSS_TOKENS * expected_accumulation,
            "effective batch token count mismatch",
        )
        _require(float(candidate.get("training_wall_seconds", 0.0)) > 0.0, "training wall time missing")
        checkpoints = candidate.get("checkpoints")
        _require(isinstance(checkpoints, list) and len(checkpoints) == 4, "validation checkpoint grid drift")
        for point in checkpoints:
            loss = float(point["validation_loss"])
            _require(math.isfinite(loss) and loss > 0.0, "non-finite validation loss")
        telemetry = candidate.get("telemetry")
        _require(isinstance(telemetry, Mapping), "incumbent observability telemetry missing")
        summary = telemetry.get("summary")
        _require(isinstance(summary, Mapping), "observability summary missing")
        counters = summary.get("counters")
        _require(isinstance(counters, Mapping), "observability counters missing")
        _require(counters.get("optimized_tokens") == expected_tokens, "observer token accounting mismatch")

    recommendation = report.get("recommendation")
    _require(isinstance(recommendation, Mapping), "recommendation missing")
    _require(
        recommendation.get("exact_critical_batch_size_claim") is False,
        "recommendation overclaims exact critical batch",
    )
    _require(
        recommendation.get("recommended_gradient_accumulation_steps") in ACCUMULATION_CANDIDATES,
        "recommendation outside measured grid",
    )

    assumptions = report.get("assumptions_and_limits")
    _require(isinstance(assumptions, Mapping), "assumptions block missing")
    _require(assumptions.get("exact_critical_batch_size_claim") is False, "assumption boundary weakened")
    _require(
        assumptions.get("optimizer_state_mutated_by_diagnostic_probes") is False,
        "optimizer-state mutation overclaim",
    )
    claims = report.get("claims")
    _require(isinstance(claims, Mapping), "claims block missing")
    for field in (
        "broad_corpus_batch_optimum",
        "theoretical_critical_batch_size",
        "target_gpu_throughput_claim",
        "paid_compute_authorized_or_used",
        "canonical_external_corpus_freeze",
    ):
        _require(claims.get(field) is False, f"forbidden claim set: {field}")

    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    _require(supplied_hash == hash_json(unsigned), "report self-hash mismatch")
