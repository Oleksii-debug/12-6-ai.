"""TRAIN-127 evidence-based gradient clipping transfer to the accepted ~10M model.

This experiment composes existing incumbents rather than changing model or Trainer
semantics.  TRAIN-54 supplies side-effect-free layer-health windows, TRAIN-53
supplies the rights-reviewed bounded real-source corpus path, and the incumbent
Trainer remains the only optimizer/update implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

from .batch_noise_probe import _real_corpus_records, _tensor_batches_from_records
from .config import TrainerConfig
from .layer_health import capture_layer_health_window
from .observability import TrainingObserver
from .s1_preflight import _evaluate
from .trainer import NonFiniteTrainingError, Trainer

SCHEMA_VERSION = "12-6.train127-clip-10m-transfer.v1"
PLAN_SCHEMA_VERSION = "12-6.train127-clip-plan.v1"
AUTHORITY = "LOCAL_FREE_10M_CLIPPING_TRANSFER_PROVISIONAL_NOT_LONG_HORIZON_POLICY"
REPOSITORY = "Oleksii-debug/12-6-ai."
STAGE_CONFIG = Path("configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json")
EXPECTED_PARAMETERS = 10_000_640
EXPECTED_MODEL_IDENTITY = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_INIT_IDENTITY = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
DIAGNOSTIC_STEPS = 8
CONTROLLED_STEPS = 12
EVAL_EVERY_STEPS = 3
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128
MAX_VALIDATION_BATCHES = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
EPS = 1e-8
WARMUP_STEPS = 0
SEED = 1515


class Clip10MTransferError(RuntimeError):
    """Raised when TRAIN-127 cannot preserve its controlled evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Clip10MTransferError(message)


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values), "percentile requires observations")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _round_up_two_significant(value: float) -> float:
    _require(math.isfinite(value) and value > 0.0, "clip basis must be finite and positive")
    exponent = math.floor(math.log10(value)) - 1
    quantum = 10.0**exponent
    return float(math.ceil(value / quantum - 1e-12) * quantum)


def derive_clip_thresholds(global_norms: Sequence[float]) -> list[float | None]:
    """Preregister no-clip plus p90/p95-derived occasional-safety thresholds."""
    _require(len(global_norms) >= 4, "diagnostic window too small to justify clipping thresholds")
    p90 = _round_up_two_significant(_percentile(global_norms, 0.90))
    p95 = _round_up_two_significant(_percentile(global_norms, 0.95))
    result: list[float | None] = [None]
    for value in (p90, p95):
        if value not in result:
            result.append(value)
    return result


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().double()
        _require(bool(torch.isfinite(grad).all().item()), "non-finite gradient observed by TRAIN-127")
        squared += float(grad.square().sum().item())
    return math.sqrt(squared)


def _layer_gradients(model: TwelveSixDecoder) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for index, block in enumerate(model.blocks):
        attn_norm = tuple(block.attn_norm.parameters())
        mlp_norm = tuple(block.mlp_norm.parameters())
        result.append(
            {
                "layer": index,
                "attention": _gradient_norm(tuple(block.attn.parameters())),
                "mlp": _gradient_norm(tuple(block.mlp.parameters())),
                "norm": _gradient_norm(attn_norm + mlp_norm),
                "combined_block": _gradient_norm(tuple(block.parameters())),
            }
        )
    return result


def _diagnostic_layer_gradients(window: Mapping[str, Any]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for layer in window["layers"]:
        gradients = layer["gradient_norms"]
        attention = float(gradients["attention"])
        mlp = float(gradients["mlp"])
        norm = float(gradients["norm"])
        result.append(
            {
                "layer": int(layer["layer"]),
                "attention": attention,
                "mlp": mlp,
                "norm": norm,
                "combined_block": math.sqrt(attention * attention + mlp * mlp + norm * norm),
            }
        )
    return result


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _update_ratios(model: TwelveSixDecoder, before: Mapping[str, Tensor]) -> dict[str, Any]:
    global_before = 0.0
    global_delta = 0.0
    layer_before = [0.0 for _ in model.blocks]
    layer_delta = [0.0 for _ in model.blocks]
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        old = before[name].double()
        new = parameter.detach().double()
        before_sq = float(old.square().sum().item())
        delta_sq = float((new - old).square().sum().item())
        global_before += before_sq
        global_delta += delta_sq
        if name.startswith("blocks."):
            pieces = name.split(".")
            if len(pieces) > 1 and pieces[1].isdigit():
                index = int(pieces[1])
                if 0 <= index < len(layer_before):
                    layer_before[index] += before_sq
                    layer_delta[index] += delta_sq
    global_ratio = math.sqrt(global_delta) / max(math.sqrt(global_before), 1e-30)
    return {
        "global_relative_update_l2": global_ratio,
        "per_layer_relative_update_l2": [
            math.sqrt(delta) / max(math.sqrt(base), 1e-30)
            for base, delta in zip(layer_before, layer_delta, strict=True)
        ],
    }


def _evaluate_bpb(model: TwelveSixDecoder, validation_batches: Sequence[Mapping[str, Tensor]]) -> dict[str, Any]:
    before = _state_digest(model)
    loss, tokens = _evaluate(model, list(validation_batches))
    after = _state_digest(model)
    _require(before == after, "held-out evaluation mutated model state")
    value = float(loss)
    _require(math.isfinite(value), "held-out evaluation loss is non-finite")
    return {
        "loss_nats_per_byte_token": value,
        "bpb": value / math.log(2.0),
        "scoreable_tokens": int(tokens),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutating": True,
    }


def _load_accepted_spec(root: Path) -> tuple[ModelSpec, InitSpec, dict[str, Any]]:
    path = root / STAGE_CONFIG
    raw = json.loads(path.read_text(encoding="utf-8"))
    spec = ModelSpec(**raw["model"])
    init_spec = InitSpec(**raw["init"])
    _require(spec.parameter_count() == EXPECTED_PARAMETERS, "accepted 10M parameter count drift")
    _require(spec.identity_sha256() == EXPECTED_MODEL_IDENTITY, "accepted 10M model identity drift")
    _require(init_spec.identity_sha256() == EXPECTED_INIT_IDENTITY, "accepted 10M init identity drift")
    _require(raw.get("canonical_base") == "random_init", "10M base must remain random_init")
    return spec, init_spec, {"path": str(STAGE_CONFIG), "file_sha256": sha256_file(path), "raw": raw}


def _trainer_config(*, gradient_clip_norm: float | None, max_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
        max_steps=max_steps,
        warmup_steps=WARMUP_STEPS,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=gradient_clip_norm,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _model(spec: ModelSpec, init_spec: InitSpec, seed: int) -> TwelveSixDecoder:
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    _require(spec.parameter_count() == sum(p.numel() for p in model.parameters()), "runtime parameter count mismatch")
    return model


def _profile(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "profile requires observations")
    return {
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _layer_distribution(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require(bool(windows), "layer distribution requires windows")
    count = len(windows[0]["preclip_layer_gradients"])
    output: list[dict[str, Any]] = []
    for index in range(count):
        item: dict[str, Any] = {"layer": index}
        for key in ("attention", "mlp", "norm", "combined_block"):
            item[key] = _profile(
                [float(window["preclip_layer_gradients"][index][key]) for window in windows]
            )
        output.append(item)
    return output


def _loss_spikes(step_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    spike_steps: list[int] = []
    previous: float | None = None
    for item in step_records:
        current = float(item["loss"])
        if previous is not None and previous > 0.0:
            ratio = current / previous - 1.0
            ratios.append(ratio)
            if ratio > 0.25:
                spike_steps.append(int(item["step"]))
        previous = current
    return {
        "definition": "step loss > previous step loss by 25%",
        "count": len(spike_steps),
        "steps": spike_steps,
        "maximum_step_over_step_increase": max(ratios) if ratios else 0.0,
    }


def _run_trajectory(
    *,
    label: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    train_batches: Sequence[Mapping[str, Tensor]],
    validation_batches: Sequence[Mapping[str, Tensor]],
    gradient_clip_norm: float | None,
    steps: int,
    seed: int,
    source_sha: str,
) -> dict[str, Any]:
    model = _model(spec, init_spec, seed)
    config = _trainer_config(gradient_clip_norm=gradient_clip_norm, max_steps=steps, seed=seed)
    trainer = Trainer(model, config, device="cpu")
    initial_model_sha256 = _state_digest(model)
    observer = TrainingObserver(
        {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "experiment": SCHEMA_VERSION,
            "label": label,
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "training_config": asdict(config),
        },
        device="cpu",
        max_step_samples=steps + 1,
        gpu_sample_every_steps=steps + 1,
    )
    evaluations: list[dict[str, Any]] = []
    initial_eval = _evaluate_bpb(model, validation_batches)
    evaluations.append({"step": 0, "optimized_tokens": 0, "training_seconds": 0.0, **initial_eval})
    records: list[dict[str, Any]] = []
    training_seconds = 0.0
    finite_failures: list[str] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    for step in range(1, steps + 1):
        batch = train_batches[(step - 1) % len(train_batches)]
        diagnostic = capture_layer_health_window(
            model,
            batch,
            label=f"{label}.pre_step_{step}",
            optimizer_step=trainer.optimizer_step,
            tokens_seen=trainer.tokens_seen,
            gradient_clip_norm=gradient_clip_norm,
        )
        diagnostic_global = float(diagnostic["global_gradient_norm"])
        diagnostic_layers = _diagnostic_layer_gradients(diagnostic)
        before = _snapshot_parameters(model)
        clip_measurement: dict[str, Any] = {}

        def measured_clip(parameters, max_norm, *args, **kwargs):
            params = tuple(parameters)
            clip_measurement["preclip_global_gradient_norm"] = _gradient_norm(params)
            clip_measurement["preclip_layer_gradients"] = _layer_gradients(model)
            result = original_clip(params, max_norm, *args, **kwargs)
            clip_measurement["postclip_global_gradient_norm"] = _gradient_norm(params)
            clip_measurement["postclip_layer_gradients"] = _layer_gradients(model)
            return result

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_ = measured_clip
        started = time.perf_counter()
        try:
            metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=0.0)
        except NonFiniteTrainingError as exc:
            finite_failures.append(str(exc))
            raise
        finally:
            training_seconds += time.perf_counter() - started
            torch.nn.utils.clip_grad_norm_ = original_clip
        trainer.assert_checkpoint_safe()
        _require(metrics.optimizer_stepped, "TRAIN-127 requires one optimizer update per microbatch")
        _require(metrics.grad_norm is not None, "Trainer did not report pre-clip gradient norm")
        actual_pre = float(metrics.grad_norm)
        tolerance = max(1e-6, abs(actual_pre) * 2e-4)
        _require(abs(actual_pre - diagnostic_global) <= tolerance, "side-effect-free diagnostic gradient drift")

        if gradient_clip_norm is None:
            actual_post = actual_pre
            pre_layers = diagnostic_layers
            post_layers = diagnostic_layers
        else:
            _require(bool(clip_measurement), "clip wrapper did not observe incumbent clipping")
            measured_pre = float(clip_measurement["preclip_global_gradient_norm"])
            _require(abs(measured_pre - actual_pre) <= tolerance, "Trainer grad norm and measured clip input disagree")
            actual_post = float(clip_measurement["postclip_global_gradient_norm"])
            pre_layers = clip_measurement["preclip_layer_gradients"]
            post_layers = clip_measurement["postclip_layer_gradients"]
        clip_engaged = gradient_clip_norm is not None and actual_pre > float(gradient_clip_norm)
        update = _update_ratios(model, before)
        _require(all(torch.isfinite(p).all().item() for p in model.parameters()), "non-finite parameter state after update")
        record = {
            "step": step,
            "optimized_tokens": trainer.tokens_seen,
            "loss": float(metrics.loss),
            "update_loss": float(metrics.update_loss) if metrics.update_loss is not None else None,
            "learning_rate": float(metrics.learning_rate),
            "preclip_global_gradient_norm": actual_pre,
            "postclip_global_gradient_norm": actual_post,
            "preclip_layer_gradients": pre_layers,
            "postclip_layer_gradients": post_layers,
            "clip_engaged": clip_engaged,
            "clip_factor_measured": actual_post / max(actual_pre, 1e-30),
            "update_ratios": update,
            "depth_health": diagnostic["depth_health"],
        }
        records.append(record)
        if step % EVAL_EVERY_STEPS == 0 or step == steps:
            evaluations.append(
                {
                    "step": step,
                    "optimized_tokens": trainer.tokens_seen,
                    "training_seconds": training_seconds,
                    **_evaluate_bpb(model, validation_batches),
                }
            )

    final_model_sha256 = _state_digest(model)
    global_norms = [float(item["preclip_global_gradient_norm"]) for item in records]
    post_norms = [float(item["postclip_global_gradient_norm"]) for item in records]
    updates = [float(item["update_ratios"]["global_relative_update_l2"]) for item in records]
    depth_warning_steps = [
        int(item["step"])
        for item in records
        if item["depth_health"]["status"] != "HEALTHY_NO_SEVERE_DEPTH_TREND"
    ]
    target_bpb = float(evaluations[0]["bpb"]) * 0.99
    quality_point = next((item for item in evaluations[1:] if float(item["bpb"]) <= target_bpb), None)
    summary = {
        "status": "PASS" if not finite_failures else "FAIL_NONFINITE",
        "initial_bpb": float(evaluations[0]["bpb"]),
        "final_bpb": float(evaluations[-1]["bpb"]),
        "bpb_delta": float(evaluations[-1]["bpb"]) - float(evaluations[0]["bpb"]),
        "preclip_global_gradient_norm_distribution": _profile(global_norms),
        "postclip_global_gradient_norm_distribution": _profile(post_norms),
        "preclip_per_layer_gradient_distributions": _layer_distribution(records),
        "clip_frequency": sum(bool(item["clip_engaged"]) for item in records) / len(records),
        "loss_spikes": _loss_spikes(records),
        "global_relative_update_l2_distribution": _profile(updates),
        "depth_warning_steps": depth_warning_steps,
        "finite_state_failures": finite_failures,
        "time_to_quality": None if quality_point is None else {
            "definition": "first held-out BPB at least 1% below initialization BPB",
            "step": int(quality_point["step"]),
            "optimized_tokens": int(quality_point["optimized_tokens"]),
            "training_seconds": float(quality_point["training_seconds"]),
            "bpb": float(quality_point["bpb"]),
        },
        "training_wall_seconds": training_seconds,
        "optimized_tokens_per_training_second": trainer.tokens_seen / max(training_seconds, 1e-30),
    }
    return {
        "label": label,
        "gradient_clip_norm": gradient_clip_norm,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "training_config": asdict(config),
        "steps": records,
        "evaluations": evaluations,
        "observer_summary": observer.summary(),
        "summary": summary,
    }


def select_weakest_intervention(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [item for item in candidates if item["summary"]["status"] == "PASS"]
    _require(bool(passed), "all clipping candidates failed")
    best_bpb = min(float(item["summary"]["final_bpb"]) for item in passed)
    quality = [item for item in passed if float(item["summary"]["final_bpb"]) <= best_bpb * 1.01]
    min_spikes = min(int(item["summary"]["loss_spikes"]["count"]) for item in quality)
    min_depth_warnings = min(len(item["summary"]["depth_warning_steps"]) for item in quality)
    stable = [
        item for item in quality
        if int(item["summary"]["loss_spikes"]["count"]) <= min_spikes + 1
        and len(item["summary"]["depth_warning_steps"]) <= min_depth_warnings + 1
    ]
    _require(bool(stable), "no quality-equivalent stability-preserving clipping candidate")

    def intervention_key(item: Mapping[str, Any]) -> tuple[int, float, float]:
        threshold = item["gradient_clip_norm"]
        if threshold is None:
            return (0, 0.0, float(item["summary"]["clip_frequency"]))
        return (1, -float(threshold), float(item["summary"]["clip_frequency"]))

    selected = min(stable, key=intervention_key)
    return {
        "selected_label": selected["label"],
        "selected_gradient_clip_norm": selected["gradient_clip_norm"],
        "rule": (
            "finite candidates within 1% of best final held-out BPB; reject materially worse "
            "loss-spike/depth-warning behavior; then prefer no clipping, otherwise the highest "
            "threshold (weakest intervention)"
        ),
        "best_final_bpb": best_bpb,
        "quality_equivalent_labels": [item["label"] for item in quality],
        "stability_preserving_labels": [item["label"] for item in stable],
    }


def run_clip_10m_transfer(
    root: Path,
    *,
    source_sha: str,
    locked_environment_evidence: Path,
    output: Path,
    preregistration_output: Path,
    seed: int = SEED,
    torch_threads: int = 2,
) -> dict[str, Any]:
    _require(len(source_sha) == 40 and all(char in "0123456789abcdef" for char in source_sha), "source_sha must be exact git SHA")
    _require(locked_environment_evidence.is_file(), "locked environment evidence missing")
    torch.set_num_threads(torch_threads)
    spec, init_spec, stage = _load_accepted_spec(root)
    tokenizer = ByteTokenizer()
    intake_dir = output.parent / "train127-real-source-intake"
    train_records, validation_records, data = _real_corpus_records(root, intake_dir)
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
        full_only=True,
    )[:MAX_VALIDATION_BATCHES]
    _require(len(train_batches) >= DIAGNOSTIC_STEPS, "real corpus yielded too few training batches")
    _require(bool(validation_batches), "real held-out object yielded no validation batches")

    diagnostic = _run_trajectory(
        label="unclipped_diagnostic",
        spec=spec,
        init_spec=init_spec,
        train_batches=train_batches,
        validation_batches=validation_batches,
        gradient_clip_norm=None,
        steps=DIAGNOSTIC_STEPS,
        seed=seed,
        source_sha=source_sha,
    )
    observed = [float(item["preclip_global_gradient_norm"]) for item in diagnostic["steps"]]
    thresholds = derive_clip_thresholds(observed)
    plan_core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_sha": source_sha,
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "dataset_identity_sha256": data["dataset_identity_sha256"],
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "diagnostic_global_norm_distribution": _profile(observed),
        "threshold_derivation": (
            "no-clip control plus diagnostic p90/p95 global norms rounded upward to two "
            "significant digits; duplicates removed. These target occasional <=10%/<=5% "
            "engagement under the observed short-window distribution, not a universal bound."
        ),
        "thresholds": thresholds,
        "controlled_steps": CONTROLLED_STEPS,
        "identical_controls": {
            "seed": seed,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "eps": EPS,
            "warmup_steps": WARMUP_STEPS,
            "scheduler": "constant",
            "precision": "fp32",
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "data_order": "same deterministic cyclic materialized DATA-21/22 train batches",
        },
    }
    plan = {**plan_core, "plan_sha256": hash_json(plan_core)}
    preregistration_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    candidates: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        label = "clip_none" if threshold is None else f"clip_{threshold:g}"
        candidate = _run_trajectory(
            label=label,
            spec=spec,
            init_spec=init_spec,
            train_batches=train_batches,
            validation_batches=validation_batches,
            gradient_clip_norm=threshold,
            steps=CONTROLLED_STEPS,
            seed=seed,
            source_sha=source_sha,
        )
        candidates.append(candidate)
    _require(len({item["initial_model_sha256"] for item in candidates}) == 1, "controlled candidates do not share initialization")
    _require(len({hash_json(item["training_config"] | {"gradient_clip_norm": None}) for item in candidates}) == 1, "controlled candidate config drift beyond clipping")
    selection = select_weakest_intervention(candidates)

    report_core = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "parameter_count": spec.parameter_count(),
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "random_initialization": True,
            "foreign_pretrained_weights": False,
            "stage_config": stage,
        },
        "incumbent_evidence": {
            "TRAIN-45": {
                "head": "4a398c1b476e35b72091e77f17de33da8a3f742d",
                "exact_head_workflow_status": "FAILURE",
                "consumed_as": "preregistered clipping design and occasional-clipping criterion only",
                "numerical_results_promoted": False,
            },
            "TRAIN-54": {
                "head": "14533bcfb512986d814d368e4639ad5b88f66e64",
                "exact_head_workflow_status": "FAILURE",
                "consumed_as": "exact side-effect-free layer-health implementation blob",
                "numerical_results_promoted": False,
            },
            "TRAIN-120": {
                "status": "NOT_DISCOVERED_BY_EXACT_LIVE_REPO_SEARCH",
                "consumed": False,
                "fabricated": False,
            },
        },
        "data": {
            **data,
            "materialized_train_batches": len(train_batches),
            "materialized_validation_batches_used": len(validation_batches),
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
        },
        "locked_environment_evidence_sha256": sha256_file(locked_environment_evidence),
        "diagnostic": diagnostic,
        "preregistration": plan,
        "preregistration_file_sha256": sha256_file(preregistration_output),
        "candidates": candidates,
        "selection": selection,
        "truth_boundary": {
            "paid_compute_used": False,
            "cuda_used": False,
            "real_external_source_bytes": True,
            "representative_broad_pretraining_corpus": False,
            "long_horizon_10m_clipping_policy_proven": False,
            "remaining_conclusion": "UNPROVEN_LONG_HORIZON_AND_TARGET_ACCELERATOR_TRANSFER",
        },
        "machine_manifest": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    }
    report = {**report_core, "report_sha256": hash_json(report_core)}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
