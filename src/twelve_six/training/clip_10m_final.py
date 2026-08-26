"""TRAIN-194 final 10M gradient-clipping policy experiment.

This experiment composes TRAIN-127's exact 10M trajectory machinery with a longer
unclipped diagnostic, three paired seeds, raw gradient distributions, and a narrow
quantile-derived clipping decision.  It deliberately does not change Trainer,
optimizer, learning rate, betas, tokenizer, corpus acquisition, or model semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json
from twelve_six.tokenization import ByteTokenizer

from .clip_10m_transfer import (
    AUTHORITY as TRAIN127_AUTHORITY,
    BATCH_SIZE,
    BETAS,
    EPS,
    LEARNING_RATE,
    MAX_VALIDATION_BATCHES,
    SEQUENCE_LENGTH,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    _load_accepted_spec,
    _profile,
    _real_corpus_records,
    _round_up_two_significant,
    _run_trajectory,
    _tensor_batches_from_records,
)

SCHEMA_VERSION = "12-6.train194-10m-clipping-final.v1"
PLAN_SCHEMA_VERSION = "12-6.train194-10m-clipping-plan.v1"
AUTHORITY = "LOCAL_FREE_10M_CLIPPING_POLICY_FINAL_FOR_TESTED_STAGE_ONLY"
DIAGNOSTIC_STEPS = 32
CONTROLLED_STEPS = 32
SEEDS = (1515, 1516, 1517)
DIAGNOSTIC_SEED = SEEDS[0]
QUALITY_MEAN_BPB_TOLERANCE = 0.01
QUALITY_MAX_SEED_BPB_TOLERANCE = 0.02
MIN_CLIP_FREQUENCY = 0.02
MAX_CLIP_FREQUENCY = 0.25
MAX_SEED_CLIP_FREQUENCY = 0.35
MIN_P95_TAIL_REDUCTION = 0.02
MIN_MAX_TAIL_REDUCTION = 0.10
MAX_UPDATE_RATIO_P95_INCREASE = 0.05
LOSS_SPIKE_RELAXATION_PER_SEED = 1
DEPTH_WARNING_RELAXATION_PER_SEED = 1


class Clip10MFinalError(RuntimeError):
    """Raised when TRAIN-194 cannot preserve its controlled experiment contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Clip10MFinalError(message)


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "mean requires observations")
    return sum(float(value) for value in values) / len(values)


def _raw_diagnostic_distribution(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    steps = diagnostic["steps"]
    _require(len(steps) == DIAGNOSTIC_STEPS, "diagnostic step count drift")
    layer_count = len(steps[0]["preclip_layer_gradients"])
    layers: list[dict[str, Any]] = []
    for layer in range(layer_count):
        item: dict[str, Any] = {"layer": layer}
        for key in ("attention", "mlp", "norm", "combined_block"):
            values = [
                float(step["preclip_layer_gradients"][layer][key])
                for step in steps
            ]
            item[key] = {"raw": values, "profile": _profile(values)}
        layers.append(item)
    global_values = [float(step["preclip_global_gradient_norm"]) for step in steps]
    return {
        "global_preclip": {"raw": global_values, "profile": _profile(global_values)},
        "per_layer_preclip": layers,
    }


def _derive_threshold_plan(global_norms: Sequence[float]) -> list[dict[str, Any]]:
    profile = _profile(global_norms)
    p95 = _round_up_two_significant(float(profile["p95"]))
    p90 = _round_up_two_significant(float(profile["p90"]))
    thresholds: list[dict[str, Any]] = [
        {"label": "unclipped", "gradient_clip_norm": None, "basis": "control"},
        {"label": "clip_p95", "gradient_clip_norm": p95, "basis": "diagnostic_p95"},
    ]
    if p90 != p95:
        thresholds.append(
            {"label": "clip_p90", "gradient_clip_norm": p90, "basis": "diagnostic_p90"}
        )
    _require(len(thresholds) >= 2, "diagnostic did not yield a clipped candidate")
    return thresholds


def _run_group_summary(label: str, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(runs), "candidate group requires runs")
    global_pre = [
        float(step["preclip_global_gradient_norm"])
        for run in runs
        for step in run["steps"]
    ]
    global_post = [
        float(step["postclip_global_gradient_norm"])
        for run in runs
        for step in run["steps"]
    ]
    post_pre = [
        float(step["clip_factor_measured"])
        for run in runs
        for step in run["steps"]
    ]
    update_ratios = [
        float(step["update_ratios"]["global_relative_update_l2"])
        for run in runs
        for step in run["steps"]
    ]
    max_layer_pre = [
        max(float(layer["combined_block"]) for layer in step["preclip_layer_gradients"])
        for run in runs
        for step in run["steps"]
    ]
    max_layer_post = [
        max(float(layer["combined_block"]) for layer in step["postclip_layer_gradients"])
        for run in runs
        for step in run["steps"]
    ]
    final_bpbs = [float(run["summary"]["final_bpb"]) for run in runs]
    clip_frequencies = [float(run["summary"]["clip_frequency"]) for run in runs]
    spike_counts = [int(run["summary"]["loss_spikes"]["count"]) for run in runs]
    spike_maxima = [
        float(run["summary"]["loss_spikes"]["maximum_step_over_step_increase"])
        for run in runs
    ]
    depth_warning_counts = [len(run["summary"]["depth_warning_steps"]) for run in runs]
    numerical_failures = [
        failure
        for run in runs
        for failure in run["summary"]["finite_state_failures"]
    ]
    return {
        "label": label,
        "seeds": [int(run["seed"]) for run in runs],
        "final_bpb_by_seed": final_bpbs,
        "mean_final_bpb": _mean(final_bpbs),
        "global_preclip_norm": {"raw": global_pre, "profile": _profile(global_pre)},
        "global_postclip_norm": {"raw": global_post, "profile": _profile(global_post)},
        "post_pre_norm_ratio": {"raw": post_pre, "profile": _profile(post_pre)},
        "update_weight_ratio": {"raw": update_ratios, "profile": _profile(update_ratios)},
        "max_per_layer_preclip_norm": {"raw": max_layer_pre, "profile": _profile(max_layer_pre)},
        "max_per_layer_postclip_norm": {"raw": max_layer_post, "profile": _profile(max_layer_post)},
        "clip_frequency_by_seed": clip_frequencies,
        "mean_clip_frequency": _mean(clip_frequencies),
        "loss_spike_count_by_seed": spike_counts,
        "loss_spike_maximum_by_seed": spike_maxima,
        "depth_warning_count_by_seed": depth_warning_counts,
        "numerical_failures": numerical_failures,
    }


def _relative_reduction(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0
    return (reference - candidate) / reference


def _decision(
    candidate_specs: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = summaries["unclipped"]
    baseline_bpb = float(baseline["mean_final_bpb"])
    baseline_p95 = float(baseline["global_preclip_norm"]["profile"]["p95"])
    baseline_max = float(baseline["global_preclip_norm"]["profile"]["max"])
    baseline_update_p95 = float(baseline["update_weight_ratio"]["profile"]["p95"])
    baseline_spikes = sum(int(value) for value in baseline["loss_spike_count_by_seed"])
    baseline_depth = sum(int(value) for value in baseline["depth_warning_count_by_seed"])
    baseline_by_seed = [float(value) for value in baseline["final_bpb_by_seed"]]

    evaluated: list[dict[str, Any]] = []
    for spec in candidate_specs:
        if spec["gradient_clip_norm"] is None:
            continue
        label = str(spec["label"])
        summary = summaries[label]
        final_by_seed = [float(value) for value in summary["final_bpb_by_seed"]]
        _require(len(final_by_seed) == len(baseline_by_seed), "paired seed count drift")
        paired_bpb_delta = [
            candidate - control
            for candidate, control in zip(final_by_seed, baseline_by_seed, strict=True)
        ]
        p95_tail_reduction = _relative_reduction(
            baseline_p95,
            float(summary["global_postclip_norm"]["profile"]["p95"]),
        )
        max_tail_reduction = _relative_reduction(
            baseline_max,
            float(summary["global_postclip_norm"]["profile"]["max"]),
        )
        update_p95 = float(summary["update_weight_ratio"]["profile"]["p95"])
        update_increase = (
            (update_p95 - baseline_update_p95) / baseline_update_p95
            if baseline_update_p95 > 0.0
            else 0.0
        )
        total_spikes = sum(int(value) for value in summary["loss_spike_count_by_seed"])
        total_depth = sum(int(value) for value in summary["depth_warning_count_by_seed"])
        clip_frequencies = [float(value) for value in summary["clip_frequency_by_seed"]]

        finite = not summary["numerical_failures"]
        quality = (
            float(summary["mean_final_bpb"]) <= baseline_bpb + QUALITY_MEAN_BPB_TOLERANCE
            and max(paired_bpb_delta) <= QUALITY_MAX_SEED_BPB_TOLERANCE
        )
        occasional = (
            MIN_CLIP_FREQUENCY <= float(summary["mean_clip_frequency"]) <= MAX_CLIP_FREQUENCY
            and max(clip_frequencies) <= MAX_SEED_CLIP_FREQUENCY
        )
        material_tail = (
            p95_tail_reduction >= MIN_P95_TAIL_REDUCTION
            or max_tail_reduction >= MIN_MAX_TAIL_REDUCTION
        )
        spikes_preserved = total_spikes <= baseline_spikes + LOSS_SPIKE_RELAXATION_PER_SEED * len(SEEDS)
        depth_preserved = total_depth <= baseline_depth + DEPTH_WARNING_RELAXATION_PER_SEED * len(SEEDS)
        updates_preserved = update_increase <= MAX_UPDATE_RATIO_P95_INCREASE
        stability = material_tail and spikes_preserved and depth_preserved and updates_preserved
        eligible = finite and quality and occasional and stability
        evaluated.append(
            {
                "label": label,
                "gradient_clip_norm": float(spec["gradient_clip_norm"]),
                "paired_final_bpb_delta": paired_bpb_delta,
                "mean_final_bpb_delta": float(summary["mean_final_bpb"]) - baseline_bpb,
                "p95_tail_reduction": p95_tail_reduction,
                "max_tail_reduction": max_tail_reduction,
                "update_ratio_p95_relative_increase": update_increase,
                "total_loss_spikes": total_spikes,
                "baseline_total_loss_spikes": baseline_spikes,
                "total_depth_warnings": total_depth,
                "baseline_total_depth_warnings": baseline_depth,
                "mean_clip_frequency": float(summary["mean_clip_frequency"]),
                "max_seed_clip_frequency": max(clip_frequencies),
                "finite": finite,
                "quality_safeguard": quality,
                "occasional_clipping": occasional,
                "material_stability_improvement": stability,
                "eligible": eligible,
            }
        )

    eligible = [item for item in evaluated if item["eligible"]]
    if not eligible:
        return {
            "verdict": "NO_USABLE_CLIPPING_THRESHOLD",
            "selected_label": None,
            "selected_gradient_clip_norm": None,
            "candidate_decisions": evaluated,
            "rule": (
                "Require zero numerical failures; mean final held-out BPB <= unclipped +0.01; "
                "every paired seed <= unclipped +0.02; mean clip frequency 2%-25% and no seed "
                ">35%; >=2% pooled p95 post-clip tail reduction or >=10% pooled max tail "
                "reduction; no material loss-spike/depth-warning increase; update/weight p95 "
                "increase <=5%. Among eligible thresholds choose the highest threshold, then "
                "lower clip frequency."
            ),
        }
    selected = max(
        eligible,
        key=lambda item: (
            float(item["gradient_clip_norm"]),
            -float(item["mean_clip_frequency"]),
        ),
    )
    return {
        "verdict": "SELECT_CLIPPING_POLICY",
        "selected_label": selected["label"],
        "selected_gradient_clip_norm": selected["gradient_clip_norm"],
        "candidate_decisions": evaluated,
        "rule": (
            "Require zero numerical failures; mean final held-out BPB <= unclipped +0.01; "
            "every paired seed <= unclipped +0.02; mean clip frequency 2%-25% and no seed "
            ">35%; >=2% pooled p95 post-clip tail reduction or >=10% pooled max tail "
            "reduction; no material loss-spike/depth-warning increase; update/weight p95 "
            "increase <=5%. Among eligible thresholds choose the highest threshold, then "
            "lower clip frequency."
        ),
    }


def run_clip_10m_final(
    root: Path,
    *,
    source_sha: str,
    locked_environment_evidence: Path,
    preregistration_output: Path,
    output: Path,
    torch_threads: int = 2,
) -> dict[str, Any]:
    import torch

    _require(len(source_sha) == 40 and all(char in "0123456789abcdef" for char in source_sha), "source_sha must be exact git SHA")
    _require(locked_environment_evidence.is_file(), "locked environment evidence missing")
    torch.set_num_threads(torch_threads)

    spec, init_spec, stage = _load_accepted_spec(root)
    tokenizer = ByteTokenizer()
    intake_dir = output.parent / "train194-real-source-intake"
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
    _require(len(train_batches) >= DIAGNOSTIC_STEPS, "real corpus yielded too few diagnostic batches")
    _require(bool(validation_batches), "real held-out object yielded no validation batches")

    diagnostic = _run_trajectory(
        label="train194_unclipped_diagnostic",
        spec=spec,
        init_spec=init_spec,
        train_batches=train_batches,
        validation_batches=validation_batches,
        gradient_clip_norm=None,
        steps=DIAGNOSTIC_STEPS,
        seed=DIAGNOSTIC_SEED,
        source_sha=source_sha,
    )
    diagnostic["seed"] = DIAGNOSTIC_SEED
    raw_diagnostic = _raw_diagnostic_distribution(diagnostic)
    global_norms = raw_diagnostic["global_preclip"]["raw"]
    candidates = _derive_threshold_plan(global_norms)

    plan_core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_sha": source_sha,
        "parameter_count": spec.parameter_count(),
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "dataset_identity_sha256": data["dataset_identity_sha256"],
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "diagnostic_optimized_tokens": DIAGNOSTIC_STEPS * (SEQUENCE_LENGTH - 1) * BATCH_SIZE,
        "diagnostic_distribution": raw_diagnostic,
        "candidate_thresholds": candidates,
        "candidate_seeds": list(SEEDS),
        "controlled_steps_per_seed": CONTROLLED_STEPS,
        "controlled_optimized_tokens_per_seed": CONTROLLED_STEPS * (SEQUENCE_LENGTH - 1) * BATCH_SIZE,
        "fixed_controls": {
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "scheduler": "constant",
            "precision": "fp32",
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "data_order": "identical deterministic cyclic materialized DATA-21/22 batches within each paired seed",
            "only_varied_factor": "gradient_clip_norm",
        },
        "decision_constants": {
            "quality_mean_bpb_tolerance": QUALITY_MEAN_BPB_TOLERANCE,
            "quality_max_seed_bpb_tolerance": QUALITY_MAX_SEED_BPB_TOLERANCE,
            "min_clip_frequency": MIN_CLIP_FREQUENCY,
            "max_clip_frequency": MAX_CLIP_FREQUENCY,
            "max_seed_clip_frequency": MAX_SEED_CLIP_FREQUENCY,
            "min_p95_tail_reduction": MIN_P95_TAIL_REDUCTION,
            "min_max_tail_reduction": MIN_MAX_TAIL_REDUCTION,
            "max_update_ratio_p95_increase": MAX_UPDATE_RATIO_P95_INCREASE,
            "loss_spike_relaxation_per_seed": LOSS_SPIKE_RELAXATION_PER_SEED,
            "depth_warning_relaxation_per_seed": DEPTH_WARNING_RELAXATION_PER_SEED,
        },
        "nonfinite_contract": (
            "Trainer._normalize_gradients_and_norm checks every gradient tensor for finiteness "
            "and raises NonFiniteTrainingError before torch.nn.utils.clip_grad_norm_ is called."
        ),
    }
    plan = {**plan_core, "plan_sha256": hash_json(plan_core)}
    preregistration_output.parent.mkdir(parents=True, exist_ok=True)
    preregistration_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    grouped_runs: dict[str, list[dict[str, Any]]] = {str(item["label"]): [] for item in candidates}
    for seed in SEEDS:
        initial_hashes: set[str] = set()
        for candidate in candidates:
            run = _run_trajectory(
                label=f"{candidate['label']}_seed_{seed}",
                spec=spec,
                init_spec=init_spec,
                train_batches=train_batches,
                validation_batches=validation_batches,
                gradient_clip_norm=candidate["gradient_clip_norm"],
                steps=CONTROLLED_STEPS,
                seed=seed,
                source_sha=source_sha,
            )
            run["seed"] = seed
            run["candidate_label"] = candidate["label"]
            initial_hashes.add(str(run["initial_model_sha256"]))
            grouped_runs[str(candidate["label"])].append(run)
        _require(len(initial_hashes) == 1, f"paired seed {seed} candidates do not share initialization")

    summaries = {
        label: _run_group_summary(label, runs)
        for label, runs in grouped_runs.items()
    }
    decision = _decision(candidates, summaries)
    selected_label = decision["selected_label"]
    selected_summary = None if selected_label is None else summaries[str(selected_label)]

    report_core = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "composed_train127_authority": TRAIN127_AUTHORITY,
        "identity": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "parameter_count": spec.parameter_count(),
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "random_initialization": True,
            "stage_config": stage,
        },
        "fixed_optimizer": {
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "scheduler": "constant",
        },
        "data": data,
        "preregistration": plan,
        "diagnostic": diagnostic,
        "candidate_runs": grouped_runs,
        "candidate_summaries": summaries,
        "decision": decision,
        "selected_policy_summary": selected_summary,
        "truth_boundary": {
            "local_free_only": True,
            "paid_compute": False,
            "lr_retuned": False,
            "betas_retuned": False,
            "trainer_semantics_changed": False,
            "nonfinite_can_be_hidden_by_clipping": False,
            "representative_broad_pretraining_corpus": False,
            "policy_scope": "tested 10,000,640-parameter 8Q/2KV stage and this bounded real-source corpus only",
        },
    }
    report = {**report_core, "report_sha256": hash_json(report_core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report