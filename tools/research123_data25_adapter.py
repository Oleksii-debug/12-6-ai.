"""RECOVER-171 DATA-25 adapter for the frozen RESEARCH-123 T/N harness.

The original RESEARCH-123 orchestration remains unchanged. This module replaces
only its former DATA-21/22 intake adapter, aligns packing with MILESTONE-150, and
upgrades uncertainty propagation to paired nonparametric held-out bootstrap.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.packing import TextRecord, batch_examples, collate_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.loss import causal_lm_loss

SCHEMA_VERSION = "12-6.research123-data25-tn-scaling.v2"
RECOVERY_BRANCH = "recover171/research123-data25-tn-20260826"
EXPECTED_CORPUS_ID = m150.EXPECTED_CORPUS_ID
BATCH_SIZE = m150.BATCH
SEQUENCE_LENGTH = m150.SEQ
LOSS_TOKENS_PER_STEP = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
TARGET_TN_RATIOS = (1.0 / 32.0, 1.0 / 8.0, 1.0 / 2.0, 2.0)
TRAIN_BATCHES_BY_STRATUM = {"uk": 180, "en": 140, "code": 80}
VALIDATION_BATCHES_BY_STRATUM = {"uk": 64, "en": 33, "code": 32}
BOOTSTRAP_SAMPLES = 400
CURVE_BOOTSTRAP_SAMPLES = 300
UNIVERSAL_BOOTSTRAP_SEED = 171_123_150
STRATA = ("uk", "en", "code")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "percentile requires observations")
    index = round(fraction * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def _collect_records(
    corpus: Path,
    manifest: Mapping[str, Any],
    split: str,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    for stratum in STRATA:
        rows = m100._rows(corpus, dict(manifest), split, stratum)
        for row in rows:
            records.append(
                TextRecord(
                    record_id=f"{stratum}:{row['record_id']}",
                    text=str(row["text"]),
                    split=split,
                )
            )
    _require(bool(records), f"DATA-25 {split} records are empty")
    return records


def _real_corpus_records(
    root: Path,
    intake_output: Path,
) -> tuple[list[TextRecord], list[TextRecord], dict[str, Any]]:
    """Rebuild DATA-25 exactly and expose a deterministic bounded T/N view."""
    manifest = m100._build_corpus(root, intake_output)
    _require(manifest["corpus_identity_sha256"] == EXPECTED_CORPUS_ID, "DATA-25 identity drift")
    _require(manifest["train_validation_content_overlap"] == 0, "DATA-25 leakage detected")
    truth = manifest["truth_boundary"]
    _require(truth["contains_project_authored_data"] is True, "DATA-25 project-authored truth drift")
    _require(truth["contains_external_training_data"] is False, "DATA-25 external-data truth drift")

    corpus = intake_output / "corpus-a"
    train_records = _collect_records(corpus, manifest, "train")
    validation_records = _collect_records(corpus, manifest, "validation")
    retained_manifest = root / m100.RETAINED_CORPUS_MANIFEST
    subset_contract = {
        "schema": "12-6.research123-data25-bounded-view.v1",
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "train_batches_by_stratum": TRAIN_BATCHES_BY_STRATUM,
        "validation_batches_by_stratum": VALIDATION_BATCHES_BY_STRATUM,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "cross_document": False,
        "train_order": "MILESTONE-100 fixed 45/35/20 UK/EN/code mixture",
        "validation_order": list(STRATA),
    }
    return train_records, validation_records, {
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "dataset_identity_sha256": hash_json(subset_contract),
        "intake_manifest_sha256": sha256_file(retained_manifest),
        "retained_manifest_file": str(m100.RETAINED_CORPUS_MANIFEST),
        "retained_manifest_file_sha256": sha256_file(retained_manifest),
        "corpus_version": manifest["corpus_version"],
        "corpus_train_byte_tokens": manifest["by_split"]["train"]["byte_tokens"],
        "corpus_validation_byte_tokens": manifest["by_split"]["validation"]["byte_tokens"],
        "corpus_by_split_stratum": manifest["by_split_stratum"],
        "train_validation_content_overlap": manifest["train_validation_content_overlap"],
        "subset_contract": subset_contract,
        "contains_project_authored_data": True,
        "real_external_source_bytes": False,
        "representative_broad_pretraining_corpus": False,
        "representative_external_corpus_claim": False,
        "truth_boundary": truth,
        "source_scope_warning": (
            "DATA-25 is the strongest current common local UK/EN/code corpus identity, "
            "but its retained manifest states that it contains project-authored data and "
            "no external training data. Results are local scaling evidence only."
        ),
    }


def _stratum(record: TextRecord) -> str:
    prefix = record.record_id.split(":", 1)[0]
    _require(prefix in STRATA, f"record stratum prefix missing: {record.record_id}")
    return prefix


def _pack_stratum(
    records: Sequence[TextRecord],
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
    sequence_length: int,
    target_batches: int,
) -> list[dict[str, torch.Tensor]]:
    examples = iter_packed_examples(
        records,
        tokenizer,
        expected_split=split,
        sequence_length=sequence_length,
        cross_document=False,
    )
    result: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        if len(group) != batch_size:
            continue
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        valid = int(labels[:, 1:].ne(-100).sum().item())
        if valid != batch_size * (sequence_length - 1):
            continue
        result.append({"input_ids": input_ids, "labels": labels})
        if len(result) == target_batches:
            break
    _require(
        len(result) == target_batches,
        f"DATA-25 {split} stratum produced {len(result)} of {target_batches} required batches",
    )
    return result


def _tensor_batches_from_records(
    records: Sequence[TextRecord],
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
    sequence_length: int,
    full_only: bool,
) -> list[dict[str, torch.Tensor]]:
    """Create the preregistered bounded batch trace from DATA-25 records."""
    del full_only
    _require(batch_size == BATCH_SIZE, "RECOVER-171 batch-size drift")
    _require(sequence_length == SEQUENCE_LENGTH, "RECOVER-171 sequence-length drift")
    grouped = {name: [] for name in STRATA}
    for record in records:
        grouped[_stratum(record)].append(record)
    targets = TRAIN_BATCHES_BY_STRATUM if split == "train" else VALIDATION_BATCHES_BY_STRATUM
    packed = {
        stratum: _pack_stratum(
            grouped[stratum],
            split=split,
            tokenizer=tokenizer,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_batches=targets[stratum],
        )
        for stratum in STRATA
    }
    if split == "validation":
        return [batch for stratum in STRATA for batch in packed[stratum]]

    indices = {name: 0 for name in STRATA}
    result: list[dict[str, torch.Tensor]] = []
    total = sum(TRAIN_BATCHES_BY_STRATUM.values())
    for step in range(total):
        stratum = m100.MIXTURE[step % len(m100.MIXTURE)]
        result.append(packed[stratum][indices[stratum]])
        indices[stratum] += 1
    _require(indices == TRAIN_BATCHES_BY_STRATUM, f"mixture count drift: {indices}")
    return result


def install_batch_noise_probe_stub() -> None:
    """Satisfy the frozen harness import without importing the old TRAIN-53 stack."""
    name = "twelve_six.training.batch_noise_probe"
    module = types.ModuleType(name)
    module._real_corpus_records = _real_corpus_records
    module._tensor_batches_from_records = _tensor_batches_from_records
    sys.modules[name] = module


def _universal_bootstrap_samples(
    observations: Sequence[tuple[float, int]],
) -> list[float]:
    _require(bool(observations), "bootstrap requires observations")
    if len(observations) == 1:
        return [observations[0][0] / math.log(2.0)] * BOOTSTRAP_SAMPLES
    rng = random.Random(UNIVERSAL_BOOTSTRAP_SEED + len(observations) * 1_000_003)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(len(observations)) for _ in observations]
        total_tokens = sum(observations[index][1] for index in indices)
        weighted = sum(observations[index][0] * observations[index][1] for index in indices)
        samples.append((weighted / total_tokens) / math.log(2.0))
    return samples


@torch.no_grad()
def evaluate_nonmutating(
    experiment: Any,
    model: Any,
    trainer: Any,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Existing non-mutation evaluator plus paired universal bootstrap samples."""
    before_model = experiment._model_digest(model)
    before_trainer = experiment._state_digest(trainer.state_dict())
    before_mode = bool(model.training)
    before_tokens = trainer.tokens_seen
    before_step = trainer.optimizer_step

    observations: list[tuple[float, int]] = []
    weighted_loss = 0.0
    token_count = 0
    correct = 0
    model.eval()
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        loss = causal_lm_loss(logits, labels)
        _require(torch.isfinite(loss).item(), "evaluation produced non-finite loss")
        targets = labels[:, 1:]
        predictions = logits[:, :-1, :].argmax(dim=-1)
        valid = targets.ne(-100)
        tokens = int(valid.sum().item())
        _require(tokens > 0, "evaluation batch has zero scoreable tokens")
        batch_loss = float(loss.item())
        observations.append((batch_loss, tokens))
        weighted_loss += batch_loss * tokens
        token_count += tokens
        correct += int((predictions.eq(targets) & valid).sum().item())

    model.train(before_mode)
    after_model = experiment._model_digest(model)
    after_trainer = experiment._state_digest(trainer.state_dict())
    _require(before_model == after_model, "evaluation mutated model state")
    _require(before_trainer == after_trainer, "evaluation mutated Trainer/optimizer state")
    _require(before_tokens == trainer.tokens_seen, "evaluation mutated optimized-token ledger")
    _require(before_step == trainer.optimizer_step, "evaluation mutated optimizer-step ledger")
    _require(bool(model.training) == before_mode, "evaluation failed to restore model mode")

    loss_nats = weighted_loss / token_count
    samples = _universal_bootstrap_samples(observations)
    return {
        "loss_nats": loss_nats,
        "bpb": loss_nats / math.log(2.0),
        "bpb_bootstrap_95ci": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "bootstrap_bpb_samples": samples,
        "bootstrap_design": "paired_common_nonparametric_packed_batch_resampling",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "requested_legacy_bootstrap_seed": bootstrap_seed,
        "scoreable_tokens": token_count,
        "next_token_accuracy": correct / token_count,
        "evaluation_optimized_tokens": 0,
        "non_mutation_proof": {
            "model_sha256_before": before_model,
            "model_sha256_after": after_model,
            "trainer_sha256_before": before_trainer,
            "trainer_sha256_after": after_trainer,
            "tokens_seen_before": before_tokens,
            "tokens_seen_after": trainer.tokens_seen,
            "optimizer_step_before": before_step,
            "optimizer_step_after": trainer.optimizer_step,
            "pass": True,
        },
    }


def universal_fit_curve(experiment: Any, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[list[float]] = []
    observed: list[float] = []
    bootstrap_rows: list[Sequence[float]] = []
    for candidate in candidates:
        ln_n = math.log(float(candidate["parameter_count"]))
        for point in candidate["points"]:
            ln_r = math.log(float(point["realized_t_per_n"]))
            rows.append([1.0, ln_n, ln_r, ln_r * ln_r, ln_n * ln_r])
            observed.append(float(point["validation_eval"]["bpb"]))
            bootstrap_rows.append(point["validation_eval"]["bootstrap_bpb_samples"])

    x = torch.tensor(rows, dtype=torch.float64)
    y = torch.tensor(observed, dtype=torch.float64)
    coefficients = torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    fitted = x @ coefficients
    residuals = y - fitted

    bootstrap_coefficients: list[list[float]] = []
    for sample_index in range(CURVE_BOOTSTRAP_SAMPLES):
        sampled_y = torch.tensor(
            [float(samples[sample_index]) for samples in bootstrap_rows],
            dtype=torch.float64,
        )
        solution = torch.linalg.lstsq(x, sampled_y[:, None]).solution[:, 0]
        bootstrap_coefficients.append([float(value.item()) for value in solution])

    names = ["intercept", "ln_N", "ln_T_per_N", "ln_T_per_N_squared", "ln_N_x_ln_T_per_N"]
    report: dict[str, Any] = {}
    for index, name in enumerate(names):
        samples = [row[index] for row in bootstrap_coefficients]
        report[name] = {
            "estimate": float(coefficients[index].item()),
            "paired_nonparametric_bootstrap_95ci": [
                _percentile(samples, 0.025),
                _percentile(samples, 0.975),
            ],
        }
    return {
        "form": "validation_BPB = b0 + bN*ln(N) + bR*ln(T/N) + bR2*ln(T/N)^2 + bNR*ln(N)*ln(T/N)",
        "coefficients": report,
        "observations": len(observed),
        "residual_rmse_bpb": float(torch.sqrt(torch.mean(residuals * residuals)).item()),
        "bootstrap": {
            "kind": "universal paired nonparametric packed-batch bootstrap",
            "point_samples": BOOTSTRAP_SAMPLES,
            "curve_samples": CURVE_BOOTSTRAP_SAMPLES,
            "shared_resample_indices_across_checkpoints_and_scales": True,
        },
        "uncertainty_scope": (
            "Intervals propagate held-out packed-batch sampling uncertainty across the full matrix. "
            "They do not include training-seed, corpus-authorship, or external-domain uncertainty."
        ),
        "universal_scaling_law_claim": False,
        "extrapolation_authority": "NONE_OUTSIDE_OBSERVED_95K_TO_1.04M_AND_TN_GRID",
    }


def recommend_data25(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    for candidate in candidates:
        points = list(candidate["points"])
        best_index = min(range(len(points)), key=lambda index: points[index]["validation_eval"]["bpb"])
        best = points[best_index]
        counts = [0] * len(points)
        for sample_index in range(BOOTSTRAP_SAMPLES):
            winner = min(
                range(len(points)),
                key=lambda index: points[index]["validation_eval"]["bootstrap_bpb_samples"][sample_index],
            )
            counts[winner] += 1
        frequencies = {
            str(points[index]["realized_t_per_n"]): counts[index] / BOOTSTRAP_SAMPLES
            for index in range(len(points))
        }
        reversal_after_best = any(
            point["validation_eval"]["bpb"] > best["validation_eval"]["bpb"]
            and point["training_eval"]["bpb"] < best["training_eval"]["bpb"]
            for point in points[best_index + 1 :]
        )
        per_model[str(candidate["label"])] = {
            "parameter_count": candidate["parameter_count"],
            "recommended_observed_checkpoint_tokens": best["optimized_tokens"],
            "recommended_observed_t_per_n": best["realized_t_per_n"],
            "heldout_bpb": best["validation_eval"]["bpb"],
            "training_bpb": best["training_eval"]["bpb"],
            "generalization_gap_bpb": best["generalization_gap_bpb"],
            "checkpoint": best["checkpoint"],
            "status": (
                "GRID_EDGE_STILL_BEST_NO_SATURATION_CLAIM"
                if best_index == len(points) - 1
                else "OBSERVED_INTERIOR_OR_EARLY_BEST"
            ),
            "overfit_reversal_after_best": reversal_after_best,
            "bootstrap_best_t_per_n_frequency": frequencies,
            "data_scope": "DATA-25 bounded common project-authored view only",
        }
    return {
        "per_model": per_model,
        "selection_basis": "lowest held-out BPB on the preregistered common T/N grid",
        "uncertainty_basis": "paired nonparametric held-out packed-batch bootstrap",
        "ten_million_transfer": "ABSENT_NOT_SUPPORTED_BY_RECOVER171",
    }


def run_experiment_data25(
    experiment: Any,
    *,
    source_sha: str,
    output_dir: Path,
    torch_threads: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    experiment._require(
        len(source_sha) == 40 and all(ch in "0123456789abcdef" for ch in source_sha),
        "source SHA must be exact 40-hex",
    )
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_hash = experiment._lock_bundle_hash(root)

    tokenizer = ByteTokenizer()
    train_records, validation_records, data = _real_corpus_records(root, output_dir / "data25-intake")
    train_batches = _tensor_batches_from_records(
        train_records,
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=True,
    )
    validation_batches = _tensor_batches_from_records(
        validation_records,
        split="validation",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        full_only=False,
    )
    data["training_trace_sha256"] = experiment._batch_trace_digest(train_batches)
    data["validation_trace_sha256"] = experiment._batch_trace_digest(validation_batches)
    data["training_unique_scoreable_tokens"] = len(train_batches) * LOSS_TOKENS_PER_STEP
    data["validation_scoreable_tokens"] = len(validation_batches) * LOSS_TOKENS_PER_STEP
    data["fixed_recipe"] = {
        "tokenizer": tokenizer.identity.version,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "packing": "incumbent iter_packed_examples; cross_document=False",
        "stream_recycling": "deterministic cyclic common batch trace",
        "training_mixture": list(m100.MIXTURE),
        "evaluation_subset_identity_sha256": data["dataset_identity_sha256"],
    }

    train_prompt = train_records[0].text[:24]
    validation_prompt = validation_records[0].text[:24]
    candidates: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for label, spec in experiment.model_family():
        candidate = experiment._run_candidate(
            root=root,
            output_dir=output_dir,
            source_sha=source_sha,
            label=label,
            spec=spec,
            train_batches=train_batches,
            validation_batches=validation_batches,
            tokenizer=tokenizer,
            data=data,
            train_prompt=train_prompt,
            validation_prompt=validation_prompt,
            lock_hash=lock_hash,
        )
        candidates.append(candidate)

    resume_proof = experiment._prove_fresh_process_resume(
        root=root,
        output_dir=output_dir,
        source_sha=source_sha,
        candidate=candidates[0],
        data=data,
    )
    curve = universal_fit_curve(experiment, candidates)
    recommendation = recommend_data25(candidates)

    selected_candidate: Mapping[str, Any] | None = None
    selected_point: Mapping[str, Any] | None = None
    for candidate in candidates:
        for point in candidate["points"]:
            if selected_point is None or point["validation_eval"]["bpb"] < selected_point["validation_eval"]["bpb"]:
                selected_candidate = candidate
                selected_point = point
    experiment._require(selected_candidate is not None and selected_point is not None, "selection failed")

    all_train_decrease = all(
        candidate["points"][-1]["training_eval"]["bpb"] < candidate["baseline"]["training_eval"]["bpb"]
        for candidate in candidates
    )
    all_val_improve = all(
        min(point["validation_eval"]["bpb"] for point in candidate["points"])
        < candidate["baseline"]["validation_eval"]["bpb"]
        for candidate in candidates
    )
    experiment._require(all_train_decrease, "one or more models failed train-BPB decrease proof")
    experiment._require(all_val_improve, "one or more models failed held-out-BPB improvement proof")

    shared_ladder_specs = {
        label: {
            "parameter_count": m150.model_spec(label).parameter_count(),
            "model_spec_sha256": m150.model_spec(label).identity_sha256(),
        }
        for label in ("100k", "500k", "1m")
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "authority": "LOCAL_FREE_DATA25_BOUNDED_TN_EVIDENCE_NOT_UNIVERSAL_SCALING_LAW",
        "source": {
            "repository": experiment.REPOSITORY,
            "git_sha": source_sha,
            "branch_expected": RECOVERY_BRANCH,
            "frozen_research123_origin_sha": "7327f2d101d8e8fd6aac452fcf1c7c06ce74c556",
            "milestone150_parent_sha": "5838cd16869dcfcf762368d8673eddf52d51b7e3",
        },
        "constraints": {
            "paid_compute": False,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "instruction_following_claim": False,
            "alignment_claim": False,
            "production_readiness_claim": False,
            "intelligence_claim": False,
            "evaluation_tokens_in_optimized_token_accounting": 0,
        },
        "component_selection": {
            "model": "frozen RESEARCH-123 / RESEARCH-41 95K/268K/468K/1.04M family",
            "tokenizer": tokenizer.identity.version,
            "corpus": "DATA-25 corpus V0.1 retained identity",
            "packing": "MILESTONE-150 seq128 document-isolated packing",
            "trainer_optimizer": "D02 Trainer + AdamW fp32 constant 3e-4 betas 0.9/0.95 clip 1.0",
            "checkpoint_resume": "D05 save/load checkpoint exact fresh-process resume",
            "heldout_evaluation": "common DATA-25 bounded validation view; non-mutation proof",
            "first_party_inference": "TwelveSixDecoder.generate greedy",
        },
        "data": data,
        "tokenizer": {
            "version": tokenizer.identity.version,
            "config_sha256": tokenizer.identity.config_sha256,
            "vocab_sha256": tokenizer.identity.vocab_sha256,
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": dict(tokenizer.identity.special_tokens),
        },
        "common_t_per_n_design": {
            "requested_ratios": list(TARGET_TN_RATIOS),
            "log_spacing_factor": 4.0,
            "loss_tokens_per_full_optimizer_step": LOSS_TOKENS_PER_STEP,
            "evaluation_excluded": True,
        },
        "milestone150_compatibility": {
            "corpus_identity_sha256": EXPECTED_CORPUS_ID,
            "packing_sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "shared_ladder_model_specs": shared_ladder_specs,
            "research_only_additional_268k": 267_912,
            "common_evaluation_subset_identity_sha256": data["dataset_identity_sha256"],
        },
        "candidates": candidates,
        "fresh_process_resume": resume_proof,
        "descriptive_curve": curve,
        "recommendation": recommendation,
        "selected_learned_base_checkpoint": {
            "label": selected_candidate["label"],
            "parameter_count": selected_candidate["parameter_count"],
            "optimized_tokens": selected_point["optimized_tokens"],
            "realized_t_per_n": selected_point["realized_t_per_n"],
            "training_bpb": selected_point["training_eval"]["bpb"],
            "heldout_bpb": selected_point["validation_eval"]["bpb"],
            "checkpoint": selected_point["checkpoint"],
            "generation_before": selected_candidate["baseline"]["generation"],
            "generation_after": selected_point["generation"],
        },
        "proof_summary": {
            "random_initialization": True,
            "exact_parameter_counts": [candidate["parameter_count"] for candidate in candidates],
            "data25_common_corpus_identity": True,
            "project_authored_corpus_truth_preserved": True,
            "representative_external_corpus_claim": False,
            "versioned_tokenizer": True,
            "train_bpb_decreased_all_models": all_train_decrease,
            "heldout_bpb_improved_somewhere_all_models": all_val_improve,
            "multiple_checkpoints_per_model": all(len(candidate["points"]) == 4 for candidate in candidates),
            "fresh_process_resume_exact": resume_proof["status"] == "PASS_EXACT",
            "evaluation_non_mutation": True,
            "universal_paired_bootstrap": True,
            "generation_before_after": True,
            "retained_exact_checkpoint": True,
        },
        "research123_status": {
            "status": "PASS_IF_THIS_REPORT_VALIDATES",
            "scope": "descriptive T/N evidence for the exact DATA-25 bounded view and fixed small-model family only",
            "universal_scaling_law": False,
        },
        "ten_million_status": "ABSENT_NO_RECOVER171_10M_RUN",
        "machine_manifest": experiment._machine_manifest(source_sha, lock_hash),
        "total_experiment_wall_seconds": time.perf_counter() - run_start,
        "reproduction_command": (
            f"python tools/run_research123_real_tn_scaling.py --source-sha {source_sha} "
            f"--output-dir {output_dir} --torch-threads {torch_threads}"
        ),
    }
    report["report_sha256"] = hash_json(report)
    return report


def validate_report_data25(experiment: Any, report: Mapping[str, Any]) -> None:
    experiment._require(report.get("schema_version") == SCHEMA_VERSION, "report schema drift")
    experiment._require(report["data"]["corpus_identity_sha256"] == EXPECTED_CORPUS_ID, "corpus identity drift")
    experiment._require(report["data"]["real_external_source_bytes"] is False, "external-data truth drift")
    experiment._require(report["data"]["contains_project_authored_data"] is True, "project-authored truth drift")
    experiment._require(report["data"]["train_validation_content_overlap"] == 0, "data leakage")
    proof = report["proof_summary"]
    for field in (
        "random_initialization",
        "data25_common_corpus_identity",
        "project_authored_corpus_truth_preserved",
        "versioned_tokenizer",
        "train_bpb_decreased_all_models",
        "heldout_bpb_improved_somewhere_all_models",
        "multiple_checkpoints_per_model",
        "fresh_process_resume_exact",
        "evaluation_non_mutation",
        "universal_paired_bootstrap",
        "generation_before_after",
        "retained_exact_checkpoint",
    ):
        experiment._require(proof.get(field) is True, f"proof field failed: {field}")
    experiment._require(proof["representative_external_corpus_claim"] is False, "unsupported corpus claim")
    experiment._require(len(report["candidates"]) == 4, "expected four model sizes")
    experiment._require(
        [candidate["parameter_count"] for candidate in report["candidates"]]
        == [95_568, 267_912, 467_808, 1_037_696],
        "fixed model family drift",
    )
    for candidate in report["candidates"]:
        experiment._require(len(candidate["points"]) == 4, "candidate missing T/N points")
        for point in candidate["points"]:
            for key in ("training_eval", "validation_eval"):
                evaluation = point[key]
                experiment._require(evaluation["evaluation_optimized_tokens"] == 0, "evaluation entered token ledger")
                experiment._require(evaluation["non_mutation_proof"]["pass"] is True, "evaluation mutated state")
                experiment._require(len(evaluation["bootstrap_bpb_samples"]) == BOOTSTRAP_SAMPLES, "bootstrap sample drift")
    experiment._require(report["ten_million_status"] == "ABSENT_NO_RECOVER171_10M_RUN", "10M scope drift")
    recorded_hash = report.get("report_sha256")
    experiment._require(isinstance(recorded_hash, str) and len(recorded_hash) == 64, "report hash missing")
    core = dict(report)
    core.pop("report_sha256", None)
    experiment._require(hash_json(core) == recorded_hash, "report self-hash mismatch")


def configure_experiment(experiment: Any) -> None:
    """Apply the recovery contract to the frozen orchestration module."""
    experiment.SCHEMA_VERSION = SCHEMA_VERSION
    experiment.BATCH_SIZE = BATCH_SIZE
    experiment.SEQUENCE_LENGTH = SEQUENCE_LENGTH
    experiment.LOSS_TOKENS_PER_STEP = LOSS_TOKENS_PER_STEP
    experiment.TARGET_TN_RATIOS = TARGET_TN_RATIOS
    experiment.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES
    experiment.CURVE_BOOTSTRAP_SAMPLES = CURVE_BOOTSTRAP_SAMPLES
    experiment._real_corpus_records = _real_corpus_records
    experiment._tensor_batches_from_records = _tensor_batches_from_records

    def evaluation(model: Any, trainer: Any, batches: Sequence[Mapping[str, torch.Tensor]], *, bootstrap_seed: int):
        return evaluate_nonmutating(
            experiment,
            model,
            trainer,
            batches,
            bootstrap_seed=bootstrap_seed,
        )

    experiment._evaluate_nonmutating = evaluation
    experiment._fit_curve = lambda candidates: universal_fit_curve(experiment, candidates)
    experiment._recommend = recommend_data25
    experiment.run_experiment = lambda *, source_sha, output_dir, torch_threads: run_experiment_data25(
        experiment,
        source_sha=source_sha,
        output_dir=output_dir,
        torch_threads=torch_threads,
    )
    experiment.validate_report = lambda report: validate_report_data25(experiment, report)


def decision_ledger(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "source": report["source"],
        "corpus_identity_sha256": report["data"]["corpus_identity_sha256"],
        "evaluation_subset_identity_sha256": report["data"]["dataset_identity_sha256"],
        "models": [
            {
                "label": candidate["label"],
                "parameters": candidate["parameter_count"],
                "points": [
                    {
                        "T_over_N": point["realized_t_per_n"],
                        "optimized_tokens": point["optimized_tokens"],
                        "train_bpb": point["training_eval"]["bpb"],
                        "heldout_bpb": point["validation_eval"]["bpb"],
                        "gap_bpb": point["generalization_gap_bpb"],
                        "recycle_factor": point["memorization"]["corpus_recycle_factor"],
                    }
                    for point in candidate["points"]
                ],
                "overfit_reversals": candidate["overfit_reversals"],
            }
            for candidate in report["candidates"]
        ],
        "resume": report["fresh_process_resume"],
        "recommendation": report["recommendation"],
        "selected": report["selected_learned_base_checkpoint"],
        "bootstrap": report["descriptive_curve"]["bootstrap"],
        "ten_million_status": report["ten_million_status"],
        "machine": report["machine_manifest"],
        "report_sha256": report["report_sha256"],
    }
