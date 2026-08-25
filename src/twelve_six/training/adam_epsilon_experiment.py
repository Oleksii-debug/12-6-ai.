"""Controlled AdamW epsilon ablation on the fixed ~500K 12-6 research model.

This extends the incumbent optimization experiment surface rather than changing
Trainer semantics. The controlled S0 fixture is mechanics/stability input only.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
from collections.abc import Mapping, Sequence
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .optimization_experiments import (
    _batch_trace_sha256,
    _evaluate,
    _model_fingerprint,
    _model_parameter_bytes,
    _optimizer_state_tensor_bytes,
    _parameter_l2,
    _snapshot,
    _stable_model,
    _tensor_batches,
    _update_metrics,
)
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

PLAN_SCHEMA = "12-6.adam-epsilon-plan.v1"
EVIDENCE_SCHEMA = "12-6.adam-epsilon-evidence.v1"
AUTHORITY = "LOCAL_FREE_ADAM_EPSILON_EVIDENCE_PROVISIONAL"
REPOSITORY = "Oleksii-debug/12-6-ai."
FIXTURE_PURPOSE = "CONTROLLED_S0_FIXTURE_ONLY_NOT_REPRESENTATIVE_500K_CORPUS"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_EPSILONS = (1e-8, 1e-6, 1e-4)
_EXPECTED_PRECISIONS = ("fp32", "bf16")
_EXPECTED_PARAMETER_COUNT = 467_808


class AdamEpsilonExperimentError(ValueError):
    """Raised when the epsilon ablation contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdamEpsilonExperimentError(message)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_500k_model_spec() -> ModelSpec:
    """Return the exact 467,808-parameter RESEARCH41 family member."""
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        head_dim=16,
        d_ff=256,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=16,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )
    _require(
        spec.parameter_count() == _EXPECTED_PARAMETER_COUNT,
        f"fixed ~500K model drift: {spec.parameter_count()} != {_EXPECTED_PARAMETER_COUNT}",
    )
    return spec


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "plan root must be an object")
    _require(raw.get("schema_version") == PLAN_SCHEMA, "wrong epsilon plan schema")
    model = raw.get("model")
    controls = raw.get("controls")
    thresholds = raw.get("materiality_thresholds")
    _require(isinstance(model, Mapping), "model block missing")
    _require(isinstance(controls, Mapping), "controls block missing")
    _require(isinstance(thresholds, Mapping), "materiality thresholds missing")
    _require(model.get("source_pr") == 162, "fixed model must remain bound to RESEARCH41 PR 162")
    _require(model.get("parameter_count") == _EXPECTED_PARAMETER_COUNT, "parameter-count drift")
    expected_geometry = {
        "vocab_size": 256,
        "max_seq_len": 256,
        "d_model": 96,
        "n_layers": 4,
        "n_heads": 6,
        "n_kv_heads": 6,
        "head_dim": 16,
        "d_ff": 256,
    }
    _require(
        all(model.get(name) == value for name, value in expected_geometry.items()),
        "fixed ~500K geometry drift",
    )
    _require(tuple(float(v) for v in raw.get("epsilon_values", ())) == _EXPECTED_EPSILONS, "epsilon set drift")
    _require(tuple(raw.get("precisions", ())) == _EXPECTED_PRECISIONS, "precision set drift")
    _require(float(controls.get("learning_rate")) == 3e-4, "learning rate must stay fixed")
    _require(tuple(float(v) for v in controls.get("betas", ())) == (0.9, 0.95), "betas must stay fixed")
    _require(float(controls.get("weight_decay")) == 0.0, "weight decay must stay fixed")
    _require(controls.get("scheduler") == "constant", "scheduler must stay constant")
    _require(int(controls.get("warmup_steps")) == 0, "warmup must stay zero")
    _require(float(controls.get("gradient_clip_norm")) == 1.0, "clip norm must stay fixed")
    _require(int(controls.get("gradient_accumulation_steps")) == 1, "accumulation must stay fixed")
    _require(int(controls.get("execution_steps")) > 0, "execution_steps must be positive")
    _require(
        int(controls.get("schedule_horizon_steps")) > int(controls.get("execution_steps")),
        "schedule horizon must remain independent from shortened execution length",
    )
    _require(int(controls.get("sequence_length")) <= 256, "sequence length exceeds fixed model context")
    for name, value in thresholds.items():
        _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"threshold {name} must be numeric")
        _require(math.isfinite(float(value)) and float(value) >= 0.0, f"threshold {name} invalid")
    return {"raw": raw, "path": str(plan_path), "file_sha256": _file_sha256(plan_path)}


def cpu_bf16_autocast_probe() -> dict[str, Any]:
    """Probe the exact runtime path used by Trainer for CPU bf16 autocast."""
    try:
        left = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 17.0
        right = torch.arange(64, dtype=torch.float32).reshape(8, 8).t() / 19.0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = left @ right
        supported = result.dtype == torch.bfloat16 and bool(torch.isfinite(result).all().item())
        return {
            "supported": supported,
            "result_dtype": str(result.dtype),
            "finite": bool(torch.isfinite(result).all().item()),
            "error_type": None,
        }
    except (RuntimeError, TypeError, ValueError) as exc:
        return {
            "supported": False,
            "result_dtype": None,
            "finite": False,
            "error_type": type(exc).__name__,
        }


def _optimizer_state_finite(trainer: Trainer) -> bool:
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _second_moment_stats(trainer: Trainer, *, epsilon: float, beta2: float) -> dict[str, Any]:
    values: list[Tensor] = []
    steps: set[int] = set()
    for state in trainer.optimizer.state.values():
        exp_avg_sq = state.get("exp_avg_sq")
        step_value = state.get("step")
        if not isinstance(exp_avg_sq, Tensor):
            continue
        if isinstance(step_value, Tensor):
            step = int(step_value.detach().cpu().item())
        else:
            step = int(step_value)
        steps.add(step)
        correction = 1.0 - beta2**step
        corrected = torch.sqrt(exp_avg_sq.detach().float().cpu() / correction)
        values.append(corrected.reshape(-1))
    if not values:
        return {
            "state_elements": 0,
            "optimizer_steps": [],
            "sqrt_v_min": None,
            "sqrt_v_p05": None,
            "sqrt_v_median": None,
            "sqrt_v_max": None,
            "fraction_sqrt_v_le_epsilon": None,
            "fraction_sqrt_v_le_10epsilon": None,
        }
    merged = torch.cat(values)
    return {
        "state_elements": int(merged.numel()),
        "optimizer_steps": sorted(steps),
        "sqrt_v_min": float(merged.min().item()),
        "sqrt_v_p05": float(torch.quantile(merged, 0.05).item()),
        "sqrt_v_median": float(torch.quantile(merged, 0.5).item()),
        "sqrt_v_max": float(merged.max().item()),
        "fraction_sqrt_v_le_epsilon": float(merged.le(epsilon).float().mean().item()),
        "fraction_sqrt_v_le_10epsilon": float(merged.le(10.0 * epsilon).float().mean().item()),
    }


def _relative_difference(value: float, baseline: float) -> float:
    denominator = abs(baseline)
    if denominator == 0.0:
        return 0.0 if value == baseline else math.inf
    return abs(value - baseline) / denominator


def _run_candidate(
    *,
    spec: ModelSpec,
    init: Any,
    train_batches: list[dict[str, Tensor]],
    validation_batches: list[dict[str, Tensor]],
    controls: Mapping[str, Any],
    epsilon: float,
    precision: str,
) -> dict[str, Any]:
    seed = int(controls["seed"])
    config = TrainerConfig(
        learning_rate=float(controls["learning_rate"]),
        weight_decay=float(controls["weight_decay"]),
        betas=tuple(float(v) for v in controls["betas"]),
        eps=epsilon,
        max_steps=int(controls["schedule_horizon_steps"]),
        warmup_steps=int(controls["warmup_steps"]),
        scheduler=str(controls["scheduler"]),
        gradient_accumulation_steps=int(controls["gradient_accumulation_steps"]),
        gradient_clip_norm=float(controls["gradient_clip_norm"]),
        precision=precision,
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    initial_model_sha256 = _model_fingerprint(model)
    initial_validation_loss = _evaluate(model, validation_batches)
    trainer = Trainer(model, config, device="cpu")
    progression: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    execution_steps = int(controls["execution_steps"])
    clip = float(controls["gradient_clip_norm"])
    for step_index, batch in enumerate(islice(cycle(train_batches), execution_steps), start=1):
        before = _snapshot(model)
        parameter_l2_before = _parameter_l2(model)
        try:
            metrics = trainer.train_microbatch(batch)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failure = {"step": step_index, "exception_type": type(exc).__name__}
            break
        update = _update_metrics(model, before, parameter_l2_before=parameter_l2_before)
        finite_model = _stable_model(model)
        finite_optimizer = _optimizer_state_finite(trainer)
        progression.append(
            {
                "step": step_index,
                "loss": metrics.loss,
                "update_loss": metrics.update_loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": metrics.grad_norm,
                "clip_would_activate": metrics.grad_norm is not None and metrics.grad_norm > clip,
                "model_parameters_finite": finite_model,
                "optimizer_state_finite": finite_optimizer,
                "optimizer_state_tensor_bytes": _optimizer_state_tensor_bytes(trainer),
                **update,
            }
        )
        if not finite_model or not finite_optimizer:
            failure = {
                "step": step_index,
                "exception_type": "NonFiniteModelOrOptimizerState",
            }
            break
    final_validation_loss = _evaluate(model, validation_batches) if progression else None
    losses = [float(item["loss"]) for item in progression]
    gradients = [float(item["grad_norm"]) for item in progression if item["grad_norm"] is not None]
    relative_updates = [float(item["relative_update_l2"]) for item in progression]
    status = "PASS" if failure is None and len(progression) == execution_steps else "FAIL"
    summary = {
        "status": status,
        "steps_requested": execution_steps,
        "steps_completed": len(progression),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "training_loss_first": losses[0] if losses else None,
        "training_loss_last": losses[-1] if losses else None,
        "gradient_norm_min": min(gradients) if gradients else None,
        "gradient_norm_max": max(gradients) if gradients else None,
        "gradient_norm_median": statistics.median(gradients) if gradients else None,
        "clip_activation_count": sum(bool(item["clip_would_activate"]) for item in progression),
        "relative_update_l2_median": statistics.median(relative_updates) if relative_updates else None,
        "relative_update_l2_max": max(relative_updates) if relative_updates else None,
        "model_parameter_bytes": _model_parameter_bytes(model),
        "optimizer_state_tensor_bytes_final": _optimizer_state_tensor_bytes(trainer),
        "all_model_states_finite": all(bool(item["model_parameters_finite"]) for item in progression),
        "all_optimizer_states_finite": all(bool(item["optimizer_state_finite"]) for item in progression),
        "second_moment": _second_moment_stats(
            trainer,
            epsilon=epsilon,
            beta2=float(controls["betas"][1]),
        ),
    }
    return {
        "epsilon": epsilon,
        "precision": precision,
        "trainer_config": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": list(config.betas),
            "eps": config.eps,
            "max_steps": config.max_steps,
            "warmup_steps": config.warmup_steps,
            "scheduler": config.scheduler,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clip_norm": config.gradient_clip_norm,
            "precision": config.precision,
            "seed": config.seed,
        },
        "initial_model_sha256": initial_model_sha256,
        "batch_trace_sha256": _batch_trace_sha256(train_batches, execution_steps),
        "summary": summary,
        "failure": failure,
        "progression": progression,
    }


def _compare_to_baseline(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]
    finite_state_changed = (
        cand_summary["status"] != base_summary["status"]
        or cand_summary["all_model_states_finite"] != base_summary["all_model_states_finite"]
        or cand_summary["all_optimizer_states_finite"] != base_summary["all_optimizer_states_finite"]
    )
    if cand_summary["final_validation_loss"] is None or cand_summary["training_loss_last"] is None:
        return {
            "material": True,
            "finite_state_changed": finite_state_changed,
            "reason": "candidate_incomplete_or_failed",
        }
    metrics = {
        "final_validation_loss_relative": _relative_difference(
            float(cand_summary["final_validation_loss"]),
            float(base_summary["final_validation_loss"]),
        ),
        "training_loss_last_relative": _relative_difference(
            float(cand_summary["training_loss_last"]),
            float(base_summary["training_loss_last"]),
        ),
        "median_relative_update_l2_relative": _relative_difference(
            float(cand_summary["relative_update_l2_median"]),
            float(base_summary["relative_update_l2_median"]),
        ),
        "gradient_norm_median_relative": _relative_difference(
            float(cand_summary["gradient_norm_median"]),
            float(base_summary["gradient_norm_median"]),
        ),
    }
    baseline_losses = [float(item["loss"]) for item in baseline["progression"]]
    candidate_losses = [float(item["loss"]) for item in candidate["progression"]]
    metrics["max_step_loss_relative"] = max(
        _relative_difference(value, reference)
        for value, reference in zip(candidate_losses, baseline_losses, strict=True)
    )
    exceeded = {
        name: metrics[name] > float(thresholds[name])
        for name in metrics
    }
    material = finite_state_changed or any(exceeded.values())
    return {
        "material": material,
        "finite_state_changed": finite_state_changed,
        "metrics": metrics,
        "thresholds_exceeded": exceeded,
        "reason": "material_threshold_exceeded" if material else "within_precommitted_materiality_thresholds",
    }


def _derive_decision(
    precision_outputs: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    bf16_supported = bool(precision_outputs["bf16"]["supported"])
    baseline_failures = [
        precision
        for precision, output in precision_outputs.items()
        if output["supported"] and output["results"]["1e-08"]["summary"]["status"] != "PASS"
    ]
    any_alternative_failure = any(
        result["summary"]["status"] != "PASS"
        for output in precision_outputs.values()
        if output["supported"]
        for epsilon, result in output["results"].items()
        if epsilon != "1e-08"
    )
    any_material = any(
        comparison["material"]
        for precision in comparisons.values()
        for comparison in precision.values()
    )
    if baseline_failures:
        rule = "BASELINE_UNSTABLE_REOPEN_NUMERICAL_STACK_BEFORE_EPSILON_SELECTION"
        stop_retuning = False
    elif not bf16_supported:
        rule = "KEEP_1E8_PROVISIONALLY_BF16_RUNTIME_NOT_AVAILABLE_FOR_FULL_DECISION"
        stop_retuning = False
    elif any_alternative_failure:
        rule = "KEEP_1E8_ALTERNATIVE_EPSILON_SHOWED_STABILITY_RISK"
        stop_retuning = True
    elif any_material:
        rule = "KEEP_1E8_REOPEN_EPSILON_ONLY_IF_MATERIAL_EFFECT_REPEATS_ON_REPRESENTATIVE_DATA"
        stop_retuning = False
    else:
        rule = "STOP_RETUNING_EPS_KEEP_1E8_FOR_CURRENT_FP32_BF16_PLAN"
        stop_retuning = True
    return {
        "rule": rule,
        "baseline_epsilon": 1e-8,
        "bf16_supported": bf16_supported,
        "baseline_failures": baseline_failures,
        "any_alternative_failure": any_alternative_failure,
        "any_material_effect": any_material,
        "stop_future_epsilon_retuning_for_current_plan": stop_retuning,
        "reopen_conditions": [
            "optimizer family changes",
            "loss or gradient scale changes materially",
            "parameter/state precision changes from current fp32-storage autocast plan",
            "representative-corpus evidence contradicts this controlled result",
        ],
    }


def _summary_view(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": evidence["schema_version"],
        "authority": evidence["authority"],
        "source_sha": evidence["identity"]["source_sha"],
        "model": evidence["model"],
        "precision_support": {
            name: output["supported"] for name, output in evidence["precisions"].items()
        },
        "results": {
            precision: {
                epsilon: result["summary"]
                for epsilon, result in output["results"].items()
            }
            for precision, output in evidence["precisions"].items()
        },
        "comparisons": evidence["comparisons"],
        "decision": evidence["decision"],
        "evidence_sha256": evidence["evidence_sha256"],
    }


def run_adam_epsilon_experiment(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    plan_path: str | Path = "configs/runs/adam_epsilon_500k.experimental.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(_GIT_SHA.fullmatch(source_sha) is not None, "source SHA must be a full lowercase Git SHA")
    root = Path(root).resolve()
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    plan = load_plan(plan_file)
    raw = plan["raw"]
    controls = raw["controls"]
    environment = validate_locked_environment_evidence(locked_environment_evidence, source_sha=source_sha)
    spec = fixed_500k_model_spec()
    init = load_stage_config(root / "configs/stages/s1_100k.json").init
    _require(init.identity_sha256() == INIT_SPEC_SHA256, "InitSpec drift")
    tokenizer = ByteTokenizer()
    train_batches, train_ids, train_tokens, train_max_id = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=int(controls["batch_size"]),
        sequence_length=int(controls["sequence_length"]),
    )
    validation_batches, validation_ids, validation_tokens, validation_max_id = _tensor_batches(
        root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=int(controls["batch_size"]),
        sequence_length=int(controls["sequence_length"]),
    )
    _require(not (set(train_ids) & set(validation_ids)), "train/validation record overlap")
    _require(max(train_max_id, validation_max_id) < spec.vocab_size, "fixture token exceeds model vocabulary")
    bf16_probe = cpu_bf16_autocast_probe()
    precision_outputs: dict[str, Any] = {}
    for precision in _EXPECTED_PRECISIONS:
        supported = precision == "fp32" or bool(bf16_probe["supported"])
        results: dict[str, Any] = {}
        if supported:
            for epsilon in _EXPECTED_EPSILONS:
                results[format(epsilon, ".0e")] = _run_candidate(
                    spec=spec,
                    init=init,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    controls=controls,
                    epsilon=epsilon,
                    precision=precision,
                )
        precision_outputs[precision] = {
            "supported": supported,
            "runtime_probe": bf16_probe if precision == "bf16" else {"supported": True},
            "results": results,
        }
    all_results = [
        result
        for output in precision_outputs.values()
        if output["supported"]
        for result in output["results"].values()
    ]
    _require(bool(all_results), "no epsilon candidates executed")
    initial_hashes = {result["initial_model_sha256"] for result in all_results}
    batch_hashes = {result["batch_trace_sha256"] for result in all_results}
    _require(len(initial_hashes) == 1, "epsilon candidates did not share initialization")
    _require(len(batch_hashes) == 1, "epsilon candidates did not share batch trace")
    comparisons: dict[str, Any] = {}
    thresholds = raw["materiality_thresholds"]
    for precision, output in precision_outputs.items():
        if not output["supported"]:
            comparisons[precision] = {}
            continue
        baseline = output["results"]["1e-08"]
        comparisons[precision] = {
            epsilon: _compare_to_baseline(baseline, result, thresholds)
            for epsilon, result in output["results"].items()
            if epsilon != "1e-08"
        }
    decision = _derive_decision(precision_outputs, comparisons)
    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "plan_path": str(plan_file.relative_to(root)),
        "plan_file_sha256": plan["file_sha256"],
        "environment": environment,
        "fixture": {
            "purpose": FIXTURE_PURPOSE,
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
    }
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "model": {
            "family": raw["model"]["family"],
            "source_pr": raw["model"]["source_pr"],
            "parameter_count": spec.parameter_count(),
            "modelspec_sha256": spec.identity_sha256(),
            "initspec_sha256": init.identity_sha256(),
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "geometry": {
                "d_model": spec.d_model,
                "n_layers": spec.n_layers,
                "n_heads": spec.n_heads,
                "n_kv_heads": spec.n_kv_heads,
                "head_dim": spec.head_dim,
                "d_ff": spec.d_ff,
            },
        },
        "controls": raw["controls"],
        "epsilon_values": list(_EXPECTED_EPSILONS),
        "validation_precision": "fp32",
        "precisions": precision_outputs,
        "materiality_thresholds": thresholds,
        "comparisons": comparisons,
        "decision": decision,
        "claims": {
            "epsilon_only_optimizer_variable": True,
            "shared_initialization_and_batch_trace": True,
            "model_and_optimizer_state_storage_fp32": True,
            "representative_corpus_quality_evidence": False,
            "hyperparameter_finalization_beyond_epsilon": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_adam_epsilon_evidence(evidence)
    return evidence, _summary_view(evidence)


def validate_adam_epsilon_evidence(evidence: Mapping[str, Any]) -> None:
    _require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "wrong evidence schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong evidence authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity missing")
    _require(identity.get("repository") == REPOSITORY, "repository mismatch")
    source_sha = identity.get("source_sha")
    _require(isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None, "invalid source SHA")
    _require(evidence.get("identity_sha256") == _canonical_hash(identity), "identity hash mismatch")
    model = evidence.get("model")
    _require(isinstance(model, Mapping), "model identity missing")
    _require(model.get("parameter_count") == _EXPECTED_PARAMETER_COUNT, "wrong fixed model size")
    _require(model.get("source_pr") == 162, "wrong fixed model family source")
    _require(model.get("initspec_sha256") == INIT_SPEC_SHA256, "InitSpec mismatch")
    controls = evidence.get("controls")
    _require(isinstance(controls, Mapping), "controls missing")
    _require(float(controls.get("learning_rate")) == 3e-4, "learning rate drift")
    _require(tuple(float(v) for v in controls.get("betas", ())) == (0.9, 0.95), "beta drift")
    _require(float(controls.get("weight_decay")) == 0.0, "weight decay drift")
    _require(int(controls.get("schedule_horizon_steps")) > int(controls.get("execution_steps")), "scheduler horizon collapsed to experiment length")
    precisions = evidence.get("precisions")
    _require(isinstance(precisions, Mapping) and set(precisions) == set(_EXPECTED_PRECISIONS), "precision evidence missing")
    for precision, output in precisions.items():
        _require(isinstance(output, Mapping), f"{precision}: output missing")
        supported = bool(output.get("supported"))
        results = output.get("results")
        _require(isinstance(results, Mapping), f"{precision}: results missing")
        if not supported:
            _require(precision == "bf16" and not results, "only unsupported bf16 may be skipped")
            continue
        _require(set(results) == {"1e-08", "1e-06", "1e-04"}, f"{precision}: epsilon matrix incomplete")
        baseline = results["1e-08"]
        _require(baseline["summary"]["status"] == "PASS", f"{precision}: baseline epsilon failed")
        for epsilon, result in results.items():
            _require(result.get("precision") == precision, f"{precision}/{epsilon}: precision mismatch")
            config = result.get("trainer_config")
            _require(isinstance(config, Mapping), f"{precision}/{epsilon}: trainer config missing")
            _require(float(config.get("learning_rate")) == 3e-4, f"{precision}/{epsilon}: learning rate changed")
            _require(tuple(float(v) for v in config.get("betas", ())) == (0.9, 0.95), f"{precision}/{epsilon}: betas changed")
            _require(float(config.get("weight_decay")) == 0.0, f"{precision}/{epsilon}: weight decay changed")
            summary = result.get("summary")
            _require(isinstance(summary, Mapping), f"{precision}/{epsilon}: summary missing")
            _require(summary.get("status") in {"PASS", "FAIL"}, f"{precision}/{epsilon}: bad status")
            _require(isinstance(result.get("progression"), list), f"{precision}/{epsilon}: progression missing")
            if summary.get("status") == "PASS":
                _require(summary.get("steps_completed") == summary.get("steps_requested"), f"{precision}/{epsilon}: incomplete PASS")
                _require(float(summary.get("relative_update_l2_median", 0.0)) > 0.0, f"{precision}/{epsilon}: update magnitude missing")
                _require(math.isfinite(float(summary["gradient_norm_median"])), f"{precision}/{epsilon}: gradient norm invalid")
                _require(summary.get("all_model_states_finite") is True, f"{precision}/{epsilon}: model nonfinite")
                _require(summary.get("all_optimizer_states_finite") is True, f"{precision}/{epsilon}: optimizer nonfinite")
    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims missing")
    _require(claims.get("epsilon_only_optimizer_variable") is True, "epsilon isolation claim missing")
    _require(claims.get("shared_initialization_and_batch_trace") is True, "shared-trace claim missing")
    _require(claims.get("paid_compute_authorized_or_used") is False, "paid compute overclaim")
    _require(claims.get("representative_corpus_quality_evidence") is False, "fixture overclaim")
    decision = evidence.get("decision")
    comparisons = evidence.get("comparisons")
    _require(isinstance(decision, Mapping) and isinstance(comparisons, Mapping), "decision/comparisons missing")
    _require(decision == _derive_decision(precisions, comparisons), "decision does not match executed evidence")
    supplied_hash = evidence.get("evidence_sha256")
    without_hash = dict(evidence)
    without_hash.pop("evidence_sha256", None)
    _require(supplied_hash == _canonical_hash(without_hash), "evidence self-hash mismatch")
