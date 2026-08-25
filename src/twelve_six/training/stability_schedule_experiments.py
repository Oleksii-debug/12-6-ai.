"""TRAIN-43/45 controlled warmup and gradient-clipping experiments.

This module deliberately reuses the incumbent TRAIN-33 optimization harness so model,
initialization, packing, tokenization, AdamW execution, scheduler semantics, update
measurement, and finite-gradient checks stay identical. Only warmup or clipping is
varied at a time.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from twelve_six.tokenization import ByteTokenizer

from .optimization_experiments import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    FIXTURE_PURPOSE,
    PACKING_CONFIG_SHA256,
    REPOSITORY,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    OptimizationRecipe,
    _batch_trace_sha256,
    _canonical_hash,
    _run_recipe,
    _sha256_file,
    _stage_identity,
    _tensor_batches,
)
from .s0_evidence_contract import validate_locked_environment_evidence

PLAN_SCHEMA_VERSION = "12-6.train43-45-stability-plan.v1"
EVIDENCE_SCHEMA_VERSION = "12-6.train43-45-stability-evidence.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_STABILITY_EVIDENCE_PROVISIONAL"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_STAGE_NAMES = ("S1", "S2")


class StabilityExperimentError(ValueError):
    """Raised when TRAIN-43/45 evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StabilityExperimentError(message)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def _load_plan(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "plan root must be an object")
    _require(raw.get("schema_version") == PLAN_SCHEMA_VERSION, "wrong plan schema")
    incumbent = raw.get("incumbent")
    _require(isinstance(incumbent, dict), "incumbent recipe missing")
    _require(incumbent.get("optimizer") == "adamw", "only AdamW is authorized")
    _require(float(incumbent.get("learning_rate")) == 3e-4, "incumbent LR drift")
    _require(float(incumbent.get("weight_decay")) == 0.1, "weight decay drift")
    _require(incumbent.get("betas") == [0.9, 0.95], "AdamW beta drift")
    _require(float(incumbent.get("eps")) == 1e-8, "AdamW eps drift")
    _require(incumbent.get("decay_embeddings") is False, "embedding-decay policy drift")
    _require(incumbent.get("precision") == "fp32", "precision drift")

    warmup = raw.get("warmup")
    _require(isinstance(warmup, dict), "warmup plan missing")
    _require(warmup.get("scheduler") == "linear_warmup", "warmup scheduler drift")
    _require(warmup.get("gradient_clip_norm") is None, "warmup sweep must be unclipped")
    fractions = warmup.get("fractions")
    _require(
        isinstance(fractions, list)
        and 2 <= len(fractions) <= 6
        and fractions[0] == 0.0
        and all(isinstance(value, (int, float)) for value in fractions),
        "invalid warmup fraction set",
    )
    _require(
        all(0.0 <= float(value) < 1.0 for value in fractions),
        "warmup fractions must be in [0,1)",
    )
    _require(len(set(float(value) for value in fractions)) == len(fractions), "duplicate warmup")
    _require(
        isinstance(warmup.get("early_window_steps"), int)
        and int(warmup["early_window_steps"]) > 0,
        "early_window_steps invalid",
    )

    clipping = raw.get("clipping")
    _require(isinstance(clipping, dict), "clipping plan missing")
    thresholds = clipping.get("thresholds")
    _require(isinstance(thresholds, list) and 2 <= len(thresholds) <= 6, "invalid clip set")
    _require(thresholds.count(None) == 1, "clip sweep must contain exactly one no-clip control")
    for threshold in thresholds:
        if threshold is not None:
            _require(
                isinstance(threshold, (int, float)) and float(threshold) > 0.0,
                "clip threshold must be positive",
            )

    stages = raw.get("stages")
    _require(isinstance(stages, list) and len(stages) == 2, "plan must cover two stages")
    _require(tuple(item.get("stage") for item in stages) == _STAGE_NAMES, "stage order drift")
    for item in stages:
        for key in ("execution_steps", "schedule_horizon_steps", "batch_size", "sequence_length"):
            _require(
                isinstance(item.get(key), int) and int(item[key]) > 0,
                f"{item.get('stage')}.{key} invalid",
            )
        _require(
            int(item["execution_steps"]) < int(item["schedule_horizon_steps"]),
            f"{item['stage']}: scheduler horizon must exceed shortened execution",
        )
    seed = raw.get("seed")
    _require(isinstance(seed, int) and seed >= 0, "seed invalid")
    return {"raw": raw, "file_sha256": _sha256_file(path)}


def _recipe(
    name: str,
    incumbent: Mapping[str, Any],
    *,
    warmup_fraction: float,
    gradient_clip_norm: float | None,
) -> OptimizationRecipe:
    return OptimizationRecipe(
        name=name,
        optimizer="adamw",
        learning_rate=float(incumbent["learning_rate"]),
        weight_decay=float(incumbent["weight_decay"]),
        betas=(float(incumbent["betas"][0]), float(incumbent["betas"][1])),
        eps=float(incumbent["eps"]),
        scheduler="linear_warmup",
        warmup_fraction=float(warmup_fraction),
        gradient_clip_norm=gradient_clip_norm,
        decay_embeddings=bool(incumbent["decay_embeddings"]),
        precision="fp32",
    )


def _batch_scoreable_tokens(batch: Mapping[str, Tensor]) -> int:
    labels = batch["labels"]
    return int(labels[:, 1:].ne(-100).sum().item())


def _augment_result(
    result: Mapping[str, Any],
    *,
    train_batches: Sequence[Mapping[str, Tensor]],
    execution_steps: int,
    early_window_steps: int,
    gradient_clip_norm: float | None,
) -> dict[str, Any]:
    copied = dict(result)
    progression = [dict(item) for item in result["progression"]]
    batch_tokens = [
        _batch_scoreable_tokens(batch)
        for batch in islice(cycle(train_batches), execution_steps)
    ]
    cumulative = 0
    reference = float(result["summary"]["initial_validation_loss"])
    recovery_tokens: int | None = None
    first_loss = float(progression[0]["loss"]) if progression else None
    recovery_needed = first_loss is not None and first_loss > reference
    clip_factors: list[float] = []
    for item, tokens in zip(progression, batch_tokens, strict=False):
        cumulative += tokens
        grad_norm = item.get("grad_norm")
        if gradient_clip_norm is None or grad_norm is None:
            factor = 1.0
        else:
            factor = min(1.0, gradient_clip_norm / (float(grad_norm) + 1e-6))
        item["scoreable_tokens"] = tokens
        item["cumulative_training_tokens"] = cumulative
        item["post_clip_factor"] = factor
        clip_factors.append(factor)
        if recovery_needed and recovery_tokens is None and float(item["loss"]) <= reference:
            recovery_tokens = cumulative
    if not recovery_needed:
        recovery_tokens = 0

    early = progression[: min(early_window_steps, len(progression))]
    early_losses = [float(item["loss"]) for item in early]
    grad_norms = [
        float(item["grad_norm"])
        for item in progression
        if item.get("grad_norm") is not None
    ]
    updates = [float(item["relative_update_l2"]) for item in progression]
    spike_abs = max(0.0, max(early_losses) - reference) if early_losses else None
    spike_ratio = (
        max(0.0, max(early_losses) / reference - 1.0)
        if early_losses and reference > 0.0
        else None
    )
    summary = dict(result["summary"])
    summary.update(
        {
            "initialization_loss_reference": reference,
            "initialization_recovery_needed": recovery_needed,
            "initialization_recovery_tokens": recovery_tokens,
            "early_window_steps": len(early),
            "early_loss_max": max(early_losses) if early_losses else None,
            "early_loss_spike_absolute": spike_abs,
            "early_loss_spike_ratio": spike_ratio,
            "gradient_norm_p95": _percentile(grad_norms, 0.95),
            "relative_update_l2_p95": _percentile(updates, 0.95),
            "clip_frequency": (
                sum(factor < 1.0 for factor in clip_factors) / len(clip_factors)
                if clip_factors
                else None
            ),
            "post_clip_factor_min": min(clip_factors) if clip_factors else None,
            "post_clip_factor_median": _percentile(clip_factors, 0.50),
            "finite_gradient_guard": "Trainer rejects non-finite gradients before clip_grad_norm_",
        }
    )
    copied["summary"] = summary
    copied["progression"] = progression
    return copied


def _select_warmup(results: Mapping[str, Mapping[str, Any]]) -> str:
    passed = [
        (name, result)
        for name, result in results.items()
        if result["summary"]["status"] == "PASS"
        and result["summary"]["final_validation_loss"] is not None
    ]
    _require(bool(passed), "no warmup candidate completed")
    best_validation = min(float(result["summary"]["final_validation_loss"]) for _, result in passed)
    eligible = [
        (name, result)
        for name, result in passed
        if float(result["summary"]["final_validation_loss"]) <= best_validation * 1.01
    ]

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, int]:
        summary = item[1]["summary"]
        recovery = summary["initialization_recovery_tokens"]
        return (
            float(summary["early_loss_spike_ratio"] or 0.0),
            float(summary["gradient_norm_p95"] or math.inf),
            float(recovery if recovery is not None else math.inf),
            int(item[1]["recipe"]["warmup_steps"]),
        )

    return min(eligible, key=key)[0]


def _select_clip(results: Mapping[str, Mapping[str, Any]]) -> str:
    passed = [
        (name, result)
        for name, result in results.items()
        if result["summary"]["status"] == "PASS"
        and result["summary"]["final_validation_loss"] is not None
    ]
    _require(bool(passed), "no clipping candidate completed")
    best_validation = min(float(result["summary"]["final_validation_loss"]) for _, result in passed)
    eligible = [
        (name, result)
        for name, result in passed
        if float(result["summary"]["final_validation_loss"]) <= best_validation * 1.01
    ]
    occasional = [
        item
        for item in eligible
        if item[1]["recipe"]["gradient_clip_norm"] is not None
        and float(item[1]["summary"]["clip_frequency"] or 0.0) <= 0.10
    ]
    if occasional:
        return min(
            occasional,
            key=lambda item: float(item[1]["recipe"]["gradient_clip_norm"]),
        )[0]
    no_clip = [item for item in eligible if item[1]["recipe"]["gradient_clip_norm"] is None]
    if no_clip:
        return no_clip[0][0]
    return min(
        eligible,
        key=lambda item: float(item[1]["summary"]["clip_frequency"] or 1.0),
    )[0]


def _warmup_rule(stage_outputs: Mapping[str, Any]) -> dict[str, Any]:
    selections = {
        stage: float(output["warmup"]["results"][output["warmup"]["selected"]]["recipe"]["warmup_fraction"])
        for stage, output in stage_outputs.items()
    }
    unique = set(selections.values())
    if len(unique) == 1:
        fraction = next(iter(unique))
        return {"kind": "shared_fraction", "warmup_fraction": fraction, "by_stage": selections}
    return {"kind": "scale_aware_fraction", "by_stage": selections}


def _clip_rule(stage_outputs: Mapping[str, Any]) -> dict[str, Any]:
    selections = {
        stage: output["clipping"]["results"][output["clipping"]["selected"]]["recipe"]["gradient_clip_norm"]
        for stage, output in stage_outputs.items()
    }
    unique = set(selections.values())
    if len(unique) == 1:
        return {"kind": "shared_threshold", "gradient_clip_norm": next(iter(unique)), "by_stage": selections}
    return {"kind": "scale_aware_threshold", "by_stage": selections}


def run_stability_schedule_experiments(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    plan_path: str | Path = "configs/runs/train43_45_stability.experimental.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute warmup first, then clipping with the selected warmup per stage."""
    _require(_GIT_SHA.fullmatch(source_sha) is not None, "source SHA must be full lowercase SHA")
    root_path = Path(root).resolve()
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = root_path / plan_file
    plan = _load_plan(plan_file)
    raw = plan["raw"]
    seed = int(raw["seed"])
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "controlled tokenizer drift")

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "plan_path": str(plan_file.relative_to(root_path)),
        "plan_file_sha256": plan["file_sha256"],
        "environment": environment,
        "fixture": {
            "purpose": FIXTURE_PURPOSE,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        "finite_gradient_guard": "incumbent Trainer._normalize_gradients_and_norm before clip_grad_norm_",
    }

    stages: dict[str, Any] = {}
    incumbent = raw["incumbent"]
    early_window = int(raw["warmup"]["early_window_steps"])
    for stage_plan in raw["stages"]:
        stage_name = stage_plan["stage"]
        stage_path, stage_identity = _stage_identity(root_path, stage_name)
        train_batches, train_ids, train_tokens, train_max_id = _tensor_batches(
            root_path,
            split="train",
            tokenizer=tokenizer,
            batch_size=int(stage_plan["batch_size"]),
            sequence_length=int(stage_plan["sequence_length"]),
        )
        validation_batches, validation_ids, validation_tokens, validation_max_id = _tensor_batches(
            root_path,
            split="validation",
            tokenizer=tokenizer,
            batch_size=int(stage_plan["batch_size"]),
            sequence_length=int(stage_plan["sequence_length"]),
        )
        _require(not (set(train_ids) & set(validation_ids)), f"{stage_name}: split overlap")
        _require(
            max(train_max_id, validation_max_id) < stage_identity["model_vocab_size"],
            f"{stage_name}: token exceeds model vocabulary",
        )
        execution_steps = int(stage_plan["execution_steps"])
        schedule_horizon = int(stage_plan["schedule_horizon_steps"])

        warmup_results: dict[str, Any] = {}
        for fraction_raw in raw["warmup"]["fractions"]:
            fraction = float(fraction_raw)
            name = f"warmup_{fraction:g}"
            recipe = _recipe(
                name,
                incumbent,
                warmup_fraction=fraction,
                gradient_clip_norm=None,
            )
            result = _run_recipe(
                stage_path,
                recipe=recipe,
                execution_steps=execution_steps,
                schedule_horizon_steps=schedule_horizon,
                train_batches=train_batches,
                validation_batches=validation_batches,
                seed=seed,
            )
            warmup_results[name] = _augment_result(
                result,
                train_batches=train_batches,
                execution_steps=execution_steps,
                early_window_steps=early_window,
                gradient_clip_norm=None,
            )
        _require(
            len({result["initial_model_sha256"] for result in warmup_results.values()}) == 1,
            f"{stage_name}: warmup candidates do not share initialization",
        )
        _require(
            len({result["batch_trace_sha256"] for result in warmup_results.values()}) == 1,
            f"{stage_name}: warmup candidates do not share batch trace",
        )
        selected_warmup = _select_warmup(warmup_results)
        selected_fraction = float(warmup_results[selected_warmup]["recipe"]["warmup_fraction"])

        clip_results: dict[str, Any] = {}
        for threshold_raw in raw["clipping"]["thresholds"]:
            threshold = None if threshold_raw is None else float(threshold_raw)
            suffix = "none" if threshold is None else f"{threshold:g}"
            name = f"clip_{suffix}"
            recipe = _recipe(
                name,
                incumbent,
                warmup_fraction=selected_fraction,
                gradient_clip_norm=threshold,
            )
            result = _run_recipe(
                stage_path,
                recipe=recipe,
                execution_steps=execution_steps,
                schedule_horizon_steps=schedule_horizon,
                train_batches=train_batches,
                validation_batches=validation_batches,
                seed=seed,
            )
            clip_results[name] = _augment_result(
                result,
                train_batches=train_batches,
                execution_steps=execution_steps,
                early_window_steps=early_window,
                gradient_clip_norm=threshold,
            )
        expected_trace = _batch_trace_sha256(train_batches, execution_steps)
        _require(
            all(result["batch_trace_sha256"] == expected_trace for result in clip_results.values()),
            f"{stage_name}: clipping batch trace drift",
        )
        _require(
            len({result["initial_model_sha256"] for result in clip_results.values()}) == 1,
            f"{stage_name}: clipping candidates do not share initialization",
        )
        _require(
            next(iter(warmup_results.values()))["initial_model_sha256"]
            == next(iter(clip_results.values()))["initial_model_sha256"],
            f"{stage_name}: phase initialization drift",
        )
        selected_clip = _select_clip(clip_results)
        stages[stage_name] = {
            "identity": stage_identity,
            "experiment": {
                "execution_steps": execution_steps,
                "schedule_horizon_steps": schedule_horizon,
                "scheduler_horizon_independent_of_execution": schedule_horizon > execution_steps,
                "batch_size": int(stage_plan["batch_size"]),
                "sequence_length": int(stage_plan["sequence_length"]),
                "train_scoreable_tokens_per_epoch": train_tokens,
                "validation_scoreable_tokens": validation_tokens,
                "train_record_ids": list(train_ids),
                "validation_record_ids": list(validation_ids),
                "shared_batch_trace_sha256": expected_trace,
            },
            "warmup": {
                "selected": selected_warmup,
                "selection_rule": raw["warmup"]["selection_rule"],
                "results": warmup_results,
            },
            "clipping": {
                "selected": selected_clip,
                "selection_rule": raw["clipping"]["selection_rule"],
                "results": clip_results,
            },
        }

    provisional = {
        "warmup": _warmup_rule(stages),
        "gradient_clipping": _clip_rule(stages),
    }
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "seed": seed,
        "incumbent": incumbent,
        "stages": stages,
        "provisional_rules": provisional,
        "claims": {
            "quality_or_capability_evidence": False,
            "hyperparameters_finalized": False,
            "paid_compute_authorized_or_used": False,
            "optimizer_betas_or_weight_decay_retuned": False,
            "finite_gradient_failures_hidden_by_clipping": False,
            "scheduler_horizon_coupled_to_short_experiment": False,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_stability_evidence(evidence)
    summary = {
        "schema_version": evidence["schema_version"],
        "authority": evidence["authority"],
        "source_sha": source_sha,
        "evidence_sha256": evidence["evidence_sha256"],
        "incumbent": incumbent,
        "provisional_rules": provisional,
        "stages": {
            stage: {
                "parameter_count": output["identity"]["parameter_count"],
                "execution_steps": output["experiment"]["execution_steps"],
                "schedule_horizon_steps": output["experiment"]["schedule_horizon_steps"],
                "warmup_selected": output["warmup"]["selected"],
                "warmup": {
                    name: result["summary"] for name, result in output["warmup"]["results"].items()
                },
                "clip_selected": output["clipping"]["selected"],
                "clipping": {
                    name: result["summary"] for name, result in output["clipping"]["results"].items()
                },
            }
            for stage, output in stages.items()
        },
    }
    return evidence, summary


def validate_stability_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed on identity drift, missing measurements, or overclaim."""
    _require(evidence.get("schema_version") == EVIDENCE_SCHEMA_VERSION, "wrong evidence schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity missing")
    _require(identity.get("repository") == REPOSITORY, "repository mismatch")
    source_sha = identity.get("source_sha")
    _require(isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None, "bad SHA")
    _require(evidence.get("identity_sha256") == _canonical_hash(identity), "identity hash mismatch")
    stages = evidence.get("stages")
    _require(isinstance(stages, Mapping) and tuple(stages) == _STAGE_NAMES, "stage evidence missing")
    for stage_name, output in stages.items():
        experiment = output["experiment"]
        _require(
            experiment["schedule_horizon_steps"] > experiment["execution_steps"],
            f"{stage_name}: schedule horizon coupled to experiment",
        )
        for phase in ("warmup", "clipping"):
            results = output[phase]["results"]
            _require(bool(results), f"{stage_name}/{phase}: results missing")
            for name, result in results.items():
                summary = result["summary"]
                _require(summary["status"] in {"PASS", "FAIL"}, f"{stage_name}/{name}: bad status")
                _require("early_loss_spike_ratio" in summary, f"{stage_name}/{name}: spike missing")
                _require("gradient_norm_p95" in summary, f"{stage_name}/{name}: grad p95 missing")
                _require("relative_update_l2_p95" in summary, f"{stage_name}/{name}: update missing")
                _require("clip_frequency" in summary, f"{stage_name}/{name}: clip frequency missing")
                _require("initialization_recovery_tokens" in summary, f"{stage_name}/{name}: recovery missing")
                _require(
                    summary.get("finite_gradient_guard")
                    == "Trainer rejects non-finite gradients before clip_grad_norm_",
                    f"{stage_name}/{name}: finite-gradient guard missing",
                )
    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims missing")
    _require(not any(bool(value) for value in claims.values()), "evidence overclaims authority")
    supplied = evidence.get("evidence_sha256")
    without = dict(evidence)
    without.pop("evidence_sha256", None)
    _require(supplied == _canonical_hash(without), "evidence self-hash mismatch")
