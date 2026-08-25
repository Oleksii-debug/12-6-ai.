"""MODEL-09 iso-parameter depth-vs-width experiment around 500K parameters.

This is deliberately a thin successor to ``fixed_token_research`` from
RESEARCH06/MODEL08.  It reuses that runner's exact loss-mask token accounting,
control identity, data trace, checkpoint/resume format, evaluator and Trainer.
Only the predeclared ~500K ModelSpecs and layer telemetry are new here.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from . import fixed_token_research as ft
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .tokenization import ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.model09-depth-width-500k.v1"
COLLECTION_SCHEMA = "12-6.model09-depth-width-500k-collection.v1"
AUTHORITY = "LOCAL_FREE_MODEL09_500K_ARCHITECTURE_RESEARCH_NOT_PROMOTION"
FAMILY = "depth_width_500k"
DEFAULT_BUDGETS = (16_384, 65_536)

_GEOMETRIES: tuple[tuple[str, dict[str, int]], ...] = (
    ("shallow_wide", {"d_model":136,"n_layers":2,"n_heads":4,"n_kv_heads":4,"head_dim":34,"d_ff":384}),
    ("mid_shallow", {"d_model":112,"n_layers":3,"n_heads":4,"n_kv_heads":4,"head_dim":28,"d_ff":320}),
    ("balanced", {"d_model":96,"n_layers":4,"n_heads":4,"n_kv_heads":4,"head_dim":24,"d_ff":280}),
    ("deep_narrow", {"d_model":80,"n_layers":6,"n_heads":4,"n_kv_heads":4,"head_dim":20,"d_ff":224}),
    ("very_deep_narrow", {"d_model":72,"n_layers":8,"n_heads":4,"n_kv_heads":4,"head_dim":18,"d_ff":184}),
)
_EXPECTED: Mapping[str, tuple[int, str]] = {
    "shallow_wide": (496_808, "4ef570a2737363e6773fbac5491797a97bee3c490e0dc90c9612141157163a09"),
    "mid_shallow": (502_544, "078741421f86ed366d5ea495dc00d2dac8133548ff7b255f5ef9573291fa3ffc"),
    "balanced": (495_456, "eac28a6dbd474591fda57e6cdeaff7fbd9d11594ae9e50750822ee3fe3fbc9e5"),
    "deep_narrow": (497_680, "1e583c2609d8347e01c7497df18224b8ac71197ccf7c3d8a402e1844d24b4c12"),
    "very_deep_narrow": (503_496, "0c03d075a2330ecd3c8f8f1a60cd6edef495b829834a236b04d37dd8eb726845"),
}


def candidate_specs() -> dict[str, ModelSpec]:
    specs = {name: ft._model_spec(geometry) for name, geometry in _GEOMETRIES}
    for name, spec in specs.items():
        expected_count, expected_identity = _EXPECTED[name]
        if spec.parameter_count() != expected_count:
            raise RuntimeError(f"MODEL09 parameter-count drift for {name}")
        if spec.identity_sha256() != expected_identity:
            raise RuntimeError(f"MODEL09 ModelSpec identity drift for {name}")
        allocation = spec.parameter_breakdown()
        if not spec.tie_word_embeddings or allocation["lm_head_extra"] != 0:
            raise RuntimeError("MODEL09 requires exactly tied embedding/output weights")
        if allocation["token_embedding"] != spec.vocab_size * spec.d_model:
            raise RuntimeError("MODEL09 tied embedding accounting drift")
    return specs


def config_payload() -> dict[str, Any]:
    init = InitSpec()
    return {
        "schema": "12-6.model09-depth-width-500k-config.v1",
        "authority": "PREDECLARED_EXPERIMENT_CONFIG_NOT_CANONICAL_ARCHITECTURE",
        "parent": "RESEARCH06_MODEL08_EXACT_FIXED_TOKEN_RUNNER",
        "fixed_controls": {
            "token_budgets": list(DEFAULT_BUDGETS),
            "tokenizer": ft.BYTE_TOKENIZER_VERSION,
            "vocab_size": 256,
            "max_seq_len": 256,
            "batch_size": ft.DEFAULT_BATCH_SIZE,
            "sequence_length": ft.DEFAULT_SEQUENCE_LENGTH,
            "seed": ft.DEFAULT_SEED,
            "trace_protocol": ft.TRACE_PROTOCOL,
            "research41_parent_packing": ft.RESEARCH41_PACKING_ID,
            "init_spec": init.to_dict(),
            "init_identity_sha256": init.identity_sha256(),
            "optimizer": {
                "name": "AdamW", "learning_rate": 3e-4, "betas": [0.9, 0.95],
                "eps": 1e-8, "weight_decay": 0.0, "gradient_clip_norm": 1.0,
                "scheduler": "constant", "warmup_steps": 0, "precision": "fp32",
            },
        },
        "selection_rule": "lowest final held-out validation loss; exact tie -> lower median optimizer-step wall time",
        "candidates": [
            {
                "candidate_id": name,
                "parameters": spec.parameter_count(),
                "model_identity_sha256": spec.identity_sha256(),
                "model_spec": spec.to_dict(),
                "parameter_allocation": spec.parameter_breakdown(),
            }
            for name, spec in candidate_specs().items()
        ],
    }


def _layer_snapshot(model: TwelveSixDecoder) -> dict[int, tuple[list[torch.Tensor], float]]:
    result: dict[int, tuple[list[torch.Tensor], float]] = {}
    for index, block in enumerate(model.blocks):
        values: list[torch.Tensor] = []
        norm_sq = 0.0
        for parameter in block.parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach().cpu().clone()
            values.append(value)
            norm_sq += float(torch.sum(value.float() * value.float()).item())
        result[index] = (values, math.sqrt(norm_sq))
    return result


def _layer_update_ratios(
    model: TwelveSixDecoder,
    before: Mapping[int, tuple[list[torch.Tensor], float]],
) -> dict[int, float]:
    output: dict[int, float] = {}
    for index, block in enumerate(model.blocks):
        prior, weight_norm = before[index]
        delta_sq = 0.0
        for parameter, old in zip(
            (p for p in block.parameters() if p.requires_grad), prior, strict=True
        ):
            delta = parameter.detach().cpu().float() - old.float()
            delta_sq += float(torch.sum(delta * delta).item())
        output[index] = math.sqrt(delta_sq) / weight_norm if weight_norm > 0 else math.inf
    return output


class LayerTelemetry:
    """Capture training-only activation, gradient and update distributions per block."""

    def __init__(self, model: TwelveSixDecoder, history: list[dict[str, Any]]) -> None:
        self.model = model
        self.history = history
        self.capture = False
        self.activation: dict[int, tuple[float, float]] = {}
        self.raw_grad_sq: dict[int, float] = {}
        self.handles: list[Any] = []
        for index, block in enumerate(model.blocks):
            self.handles.append(block.register_forward_hook(self._forward_hook(index)))
            for parameter in block.parameters():
                if parameter.requires_grad:
                    self.handles.append(parameter.register_hook(self._grad_hook(index)))

    def _forward_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.capture or not isinstance(output, torch.Tensor):
                return
            value = output.detach().float()
            self.activation[index] = (
                float(torch.sqrt(torch.mean(value * value)).item()),
                float(value.abs().max().item()),
            )
        return hook

    def _grad_hook(self, index: int):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            if self.capture:
                value = gradient.detach().float()
                self.raw_grad_sq[index] = self.raw_grad_sq.get(index, 0.0) + float(
                    torch.sum(value * value).item()
                )
            return gradient
        return hook

    def begin(self) -> dict[int, tuple[list[torch.Tensor], float]]:
        self.activation.clear()
        self.raw_grad_sq.clear()
        self.capture = True
        return _layer_snapshot(self.model)

    def finish(
        self,
        *,
        optimizer_step: int,
        optimized_tokens: int,
        valid_tokens: int,
        before: Mapping[int, tuple[list[torch.Tensor], float]],
    ) -> list[dict[str, Any]]:
        self.capture = False
        ratios = _layer_update_ratios(self.model, before)
        step_rows: list[dict[str, Any]] = []
        for index in range(len(self.model.blocks)):
            if index not in self.activation or index not in self.raw_grad_sq:
                raise RuntimeError(f"MODEL09 missing layer telemetry for block {index}")
            activation_rms, activation_max_abs = self.activation[index]
            # Trainer backpropagates loss * valid_tokens, then divides gradients by
            # the pending valid-token count before clipping. grad hooks therefore
            # see the summed gradient and must be normalized by valid_tokens here.
            normalized_grad_norm = math.sqrt(self.raw_grad_sq[index]) / valid_tokens
            row = {
                "optimizer_step": optimizer_step,
                "optimized_tokens": optimized_tokens,
                "layer": index,
                "activation_rms": activation_rms,
                "activation_max_abs": activation_max_abs,
                "pre_clip_grad_norm": normalized_grad_norm,
                "update_to_weight_ratio": ratios[index],
            }
            if not all(math.isfinite(float(row[key])) for key in (
                "activation_rms", "activation_max_abs", "pre_clip_grad_norm",
                "update_to_weight_ratio",
            )):
                raise RuntimeError("MODEL09 non-finite layer telemetry")
            self.history.append(row)
            step_rows.append(row)
        return step_rows

    def close(self) -> None:
        self.capture = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _layer_summary(history: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        items = [item for item in history if int(item["layer"]) == layer]
        if not items:
            raise RuntimeError(f"MODEL09 empty telemetry history for layer {layer}")
        rows.append({
            "layer": layer,
            "samples": len(items),
            "activation_rms": ft._summary([float(item["activation_rms"]) for item in items]),
            "activation_max_abs": ft._summary([float(item["activation_max_abs"]) for item in items]),
            "pre_clip_grad_norm": ft._summary([float(item["pre_clip_grad_norm"]) for item in items]),
            "update_to_weight_ratio": ft._summary([float(item["update_to_weight_ratio"]) for item in items]),
        })
    return rows


def run_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    candidate_id: str,
    output_path: Path,
    checkpoint_dir: Path,
    token_budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    batch_size: int = ft.DEFAULT_BATCH_SIZE,
    sequence_length: int = ft.DEFAULT_SEQUENCE_LENGTH,
    seed: int = ft.DEFAULT_SEED,
    torch_threads: int = ft.DEFAULT_THREADS,
) -> dict[str, Any]:
    token_budgets = ft._validate_budgets(token_budgets)
    if ft._git_head(repo_root) != source_sha:
        raise RuntimeError("MODEL09 exact-checkout mismatch")
    specs = candidate_specs()
    if candidate_id not in specs:
        raise ValueError(f"unknown MODEL09 candidate {candidate_id!r}")
    if batch_size <= 0 or sequence_length < 2 or sequence_length > 256 or torch_threads <= 0:
        raise ValueError("invalid execution geometry")

    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    capacity = batch_size * (sequence_length - 1)
    max_steps = ft._steps_for_budgets(token_budgets, capacity)
    controls = ft._controls(
        repo_root,
        source_sha=source_sha,
        token_budgets=token_budgets,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
        threads=torch_threads,
        max_steps=max_steps,
    )
    controls_sha256 = ft._canonical_hash(controls)
    tokenizer = ByteTokenizer()
    train_records = ft._read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    validation_records = ft._read_jsonl(repo_root / "data/s0/packaged/validation.jsonl")
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    train_stream = ft._byte_stream(train_records, tokenizer)
    spec = specs[candidate_id]
    init_spec = InitSpec()
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, ft._trainer_config(max_steps=max_steps, seed=seed), device="cpu")
    probe = ft._make_batch(
        train_stream, step=0, batch_size=batch_size, sequence_length=sequence_length
    )
    initial_activation = ft._activation_probe(model, probe)
    initial_model_hash = ft._model_state_hash(model)
    initial_counters = (trainer.tokens_seen, trainer.optimizer_step)
    initial_validation_loss, validation_tokens = ft._validation_loss(
        model, validation_records, tokenizer
    )
    if (
        (trainer.tokens_seen, trainer.optimizer_step) != initial_counters
        or ft._model_state_hash(model) != initial_model_hash
    ):
        raise RuntimeError("MODEL09 initial evaluation mutated training state")

    trace_events: list[dict[str, Any]] = []
    training_curve: list[dict[str, Any]] = []
    layer_history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    resume_events: list[dict[str, Any]] = []
    telemetry = LayerTelemetry(model, layer_history)
    optimization_wall = 0.0
    experiment_started = time.perf_counter()
    previous_budget = 0

    for budget_index, budget in enumerate(token_budgets):
        expected_segment_steps = math.ceil((budget - previous_budget) / capacity)
        segment_steps = 0
        while trainer.tokens_seen < budget:
            remaining = budget - trainer.tokens_seen
            valid_tokens = min(capacity, remaining)
            step_before = trainer.optimizer_step
            tokens_before = trainer.tokens_seen
            raw = ft._make_batch(
                train_stream,
                step=step_before,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            batch = ft._aligned_batch(raw, valid_tokens)
            global_before, global_weight_norm = ft._parameter_snapshot(model)
            layer_before = telemetry.begin()
            started = time.perf_counter()
            metrics = trainer.train_microbatch(batch)
            step_seconds = time.perf_counter() - started
            optimization_wall += step_seconds
            global_delta_l2, global_update_ratio, _max_abs, _changed = ft._update_stats(
                model, global_before, global_weight_norm
            )
            del global_before
            if (
                not metrics.optimizer_stepped
                or metrics.tokens != valid_tokens
                or trainer.tokens_seen != tokens_before + valid_tokens
                or trainer.tokens_seen > budget
                or trainer.optimizer_step != step_before + 1
            ):
                raise RuntimeError("MODEL09 strict token/update accounting drift")
            if metrics.grad_norm is None or not math.isfinite(metrics.grad_norm):
                raise RuntimeError("MODEL09 missing/non-finite gradient norm")
            layer_rows = telemetry.finish(
                optimizer_step=trainer.optimizer_step,
                optimized_tokens=trainer.tokens_seen,
                valid_tokens=valid_tokens,
                before=layer_before,
            )
            training_curve.append({
                "optimizer_step": trainer.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "valid_causal_tokens": valid_tokens,
                "train_loss": metrics.loss,
                "pre_clip_grad_norm": float(metrics.grad_norm),
                "global_update_l2": global_delta_l2,
                "global_update_to_weight_ratio": global_update_ratio,
                "step_seconds": step_seconds,
                "layer_rows": layer_rows,
            })
            trace_events.append({
                "optimizer_step": trainer.optimizer_step,
                "budget_index": budget_index,
                "budget": budget,
                "valid_causal_tokens": valid_tokens,
                "cumulative_optimized_tokens": trainer.tokens_seen,
                "input_sha256": ft._tensor_sha256(raw),
                "target_sha256": ft._tensor_sha256(batch["target_ids"]),
                "loss_mask_sha256": ft._tensor_sha256(batch["loss_mask"]),
            })
            segment_steps += 1

        if trainer.tokens_seen != budget or segment_steps != expected_segment_steps:
            raise RuntimeError("MODEL09 failed exact token-budget landing")
        model_hash = ft._model_state_hash(model)
        trainer_hash = ft._trainer_state_hash(trainer)
        counters = (trainer.tokens_seen, trainer.optimizer_step)
        eval_started = time.perf_counter()
        validation_loss, checked_tokens = ft._validation_loss(
            model, validation_records, tokenizer
        )
        evaluation_wall = time.perf_counter() - eval_started
        if (
            checked_tokens != validation_tokens
            or (trainer.tokens_seen, trainer.optimizer_step) != counters
            or ft._model_state_hash(model) != model_hash
            or ft._trainer_state_hash(trainer) != trainer_hash
        ):
            raise RuntimeError("MODEL09 evaluation leaked into optimized state")
        checkpoint_path = checkpoint_dir / f"{FAMILY}-{candidate_id}-tokens-{budget}.pt"
        checkpoint_evidence = ft._save_checkpoint(
            checkpoint_path,
            controls_sha256=controls_sha256,
            source_sha=source_sha,
            family=FAMILY,
            candidate_id=candidate_id,
            model=model,
            trainer=trainer,
            completed_budget_index=budget_index,
            token_budgets=token_budgets,
            trace_events=trace_events,
        )
        checkpoints.append({
            "requested_token_budget": budget,
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
            "validation_loss": validation_loss,
            "validation_bpb": validation_loss / math.log(2.0),
            "validation_tokens": checked_tokens,
            "evaluation_optimized_tokens": 0,
            "compute_proxy": 6 * spec.parameter_count() * trainer.tokens_seen,
            "optimization_wall_seconds": optimization_wall,
            "evaluation_wall_seconds": evaluation_wall,
            "checkpoint": checkpoint_evidence,
        })
        if budget_index == 0 and len(token_budgets) > 1:
            expected_model_hash = ft._model_state_hash(model)
            expected_trainer_hash = ft._trainer_state_hash(trainer)
            telemetry.close()
            del telemetry, trainer, model
            gc.collect()
            model, trainer, trace_events, resume = ft._fresh_resume(
                checkpoint_path,
                expected_controls_sha256=controls_sha256,
                expected_source_sha=source_sha,
                expected_family=FAMILY,
                expected_candidate_id=candidate_id,
                spec=spec,
                init_spec=init_spec,
                max_steps=max_steps,
                seed=seed,
                expected_token_budgets=token_budgets,
            )
            if (
                resume["model_state_sha256"] != expected_model_hash
                or resume["trainer_state_sha256"] != expected_trainer_hash
            ):
                raise RuntimeError("MODEL09 fresh-resume hash mismatch")
            resume_events.append(resume)
            telemetry = LayerTelemetry(model, layer_history)
        previous_budget = budget

    telemetry.close()
    experiment_wall = time.perf_counter() - experiment_started
    if trainer.tokens_seen != token_budgets[-1] or trainer.optimizer_step != max_steps:
        raise RuntimeError("MODEL09 final token accounting drift")
    if len(token_budgets) > 1 and not resume_events:
        raise RuntimeError("MODEL09 required fresh-object resume not exercised")
    final = checkpoints[-1]
    step_times = [float(row["step_seconds"]) for row in training_curve]
    grad_norms = [float(row["pre_clip_grad_norm"]) for row in training_curve]
    update_ratios = [float(row["global_update_to_weight_ratio"]) for row in training_curve]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "parent_runner": "RESEARCH06_MODEL08_FIXED_VALID_TOKEN_V1",
        "family": FAMILY,
        "candidate_id": candidate_id,
        "controls": controls,
        "controls_sha256": controls_sha256,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "parameter_allocation": spec.parameter_breakdown(),
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_bpb": initial_validation_loss / math.log(2.0),
        "initial_activation_scale": initial_activation,
        "validation_tokens": validation_tokens,
        "checkpoints": checkpoints,
        "training_curve": training_curve,
        "layer_summary": _layer_summary(layer_history, spec.n_layers),
        "trace_sha256": ft._canonical_hash(trace_events),
        "trace_steps": len(trace_events),
        "resume_exercised": bool(resume_events),
        "resume_events": resume_events,
        "gradient_norm": ft._summary(grad_norms),
        "clip_frequency": sum(value > 1.0 for value in grad_norms) / len(grad_norms),
        "update_ratio": ft._summary(update_ratios),
        "step_time_seconds": ft._summary(step_times),
        "optimization_wall_seconds": optimization_wall,
        "experiment_wall_seconds": experiment_wall,
        "optimized_tokens_per_optimization_second": trainer.tokens_seen / optimization_wall,
        "memory": {
            "process_rss_hwm_mib": ft._rss_hwm_mib(),
            "model_parameter_tensor_bytes": ft._parameter_tensor_bytes(model),
            "optimizer_tensor_bytes": ft._optimizer_tensor_bytes(trainer),
        },
        "final_validation_improvement": initial_validation_loss - float(final["validation_loss"]),
        "truth_boundary": [
            "Held-out validation, never train loss, is the architecture selection signal.",
            "Only masked valid causal targets increment optimized tokens; evaluation contributes zero optimized tokens.",
            "The tiny project-authored S0 EN/UK fixture is repeatedly cycled, so this is controlled local evidence only.",
            "No candidate is added or modified after validation is observed.",
            "No paid compute, stage promotion, architecture freeze, or capability claim is authorized.",
        ],
        "runtime": {"device": "cpu", "paid_compute": False},
    }
    report["report_sha256"] = ft._canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _validate_candidate(report: dict[str, Any], expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("MODEL09 candidate report schema mismatch")
    material = dict(report)
    claimed = material.pop("report_sha256", None)
    if claimed != ft._canonical_hash(material):
        raise ValueError("MODEL09 candidate report self-hash mismatch")
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise ValueError("MODEL09 source SHA mismatch")
    budgets = tuple(int(value) for value in report["controls"]["token_budgets"])
    if [int(point["optimized_tokens"]) for point in report["checkpoints"]] != list(budgets):
        raise ValueError("MODEL09 token-budget overshoot/drift")
    if any(int(point["evaluation_optimized_tokens"]) != 0 for point in report["checkpoints"]):
        raise ValueError("MODEL09 evaluation tokens counted as optimized")
    if len(budgets) > 1 and not report.get("resume_exercised"):
        raise ValueError("MODEL09 fresh-object resume missing")
    spec = candidate_specs()[str(report["candidate_id"])]
    if report["model_identity_sha256"] != spec.identity_sha256():
        raise ValueError("MODEL09 model identity drift")
    if len(report["layer_summary"]) != spec.n_layers:
        raise ValueError("MODEL09 layer telemetry incomplete")


def collect_reports(
    *, input_paths: list[Path], output_path: Path, expected_source_sha: str
) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    for report in reports:
        _validate_candidate(report, expected_source_sha)
    expected = set(candidate_specs())
    observed = {str(report["candidate_id"]) for report in reports}
    if observed != expected:
        raise ValueError(f"MODEL09 candidate set mismatch: {observed!r} != {expected!r}")
    controls = {str(report["controls_sha256"]) for report in reports}
    traces = {str(report["trace_sha256"]) for report in reports}
    if len(controls) != 1:
        raise ValueError("MODEL09 fixed-control identity drift")
    if len(traces) != 1:
        raise ValueError("MODEL09 exact data/token trace drift")
    rows: list[dict[str, Any]] = []
    for report in reports:
        final = report["checkpoints"][-1]
        step_median = report["step_time_seconds"]["median"]
        rows.append({
            "candidate_id": report["candidate_id"],
            "parameters": int(report["parameters"]),
            "model_spec": report["model_spec"],
            "model_identity_sha256": report["model_identity_sha256"],
            "validation_loss": float(final["validation_loss"]),
            "validation_bpb": float(final["validation_bpb"]),
            "validation_improvement": float(report["final_validation_improvement"]),
            "compute_proxy": int(final["compute_proxy"]),
            "optimization_wall_seconds": float(report["optimization_wall_seconds"]),
            "median_step_seconds": float(step_median),
            "tokens_per_second": float(report["optimized_tokens_per_optimization_second"]),
            "rss_hwm_mib": float(report["memory"]["process_rss_hwm_mib"]),
            "model_parameter_tensor_bytes": int(report["memory"]["model_parameter_tensor_bytes"]),
            "optimizer_tensor_bytes": int(report["memory"]["optimizer_tensor_bytes"]),
            "gradient_norm": report["gradient_norm"],
            "clip_frequency": float(report["clip_frequency"]),
            "update_ratio": report["update_ratio"],
            "layer_summary": report["layer_summary"],
        })
    ordered = sorted(rows, key=lambda row: (row["validation_loss"], row["median_step_seconds"]))
    recommendation = ordered[0]
    collection: dict[str, Any] = {
        "schema": COLLECTION_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": expected_source_sha,
        "family": FAMILY,
        "fixed_token_budget": int(reports[0]["controls"]["token_budgets"][-1]),
        "controls_sha256": next(iter(controls)),
        "exact_trace_sha256": next(iter(traces)),
        "candidate_parameter_span": {
            "min": min(row["parameters"] for row in rows),
            "max": max(row["parameters"] for row in rows),
            "absolute_span": max(row["parameters"] for row in rows) - min(row["parameters"] for row in rows),
        },
        "rows": sorted(rows, key=lambda row: int(row["model_spec"]["n_layers"])),
        "ranking": [row["candidate_id"] for row in ordered],
        "selection_rule": "lowest final held-out validation loss; exact tie -> lower median optimizer-step wall time",
        "recommended_500k_geometry": {
            "candidate_id": recommendation["candidate_id"],
            "parameters": recommendation["parameters"],
            "model_spec": recommendation["model_spec"],
            "validation_loss": recommendation["validation_loss"],
            "validation_bpb": recommendation["validation_bpb"],
        },
        "truth_boundary": [
            "This recommendation is local to the frozen tiny S0 fixture and exact 65,536-token budget.",
            "The candidate set and selection rule were predeclared before held-out results.",
            "No paid compute or promotion authority is implied.",
        ],
    }
    collection["report_sha256"] = ft._canonical_hash(collection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return collection


def validate(path: Path, expected_source_sha: str | None = None) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") == SCHEMA:
        _validate_candidate(report, expected_source_sha)
        return
    if report.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("MODEL09 report schema mismatch")
    material = dict(report)
    claimed = material.pop("report_sha256", None)
    if claimed != ft._canonical_hash(material):
        raise ValueError("MODEL09 collection self-hash mismatch")
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise ValueError("MODEL09 collection source SHA mismatch")
    if {str(row["candidate_id"]) for row in report["rows"]} != set(candidate_specs()):
        raise ValueError("MODEL09 collection candidate set drift")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-candidate")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--candidate-id", choices=tuple(candidate_specs()), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-dir", type=Path, required=True)
    run.add_argument("--token-budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    run.add_argument("--torch-threads", type=int, default=ft.DEFAULT_THREADS)
    collect = sub.add_parser("collect")
    collect.add_argument("--expected-source-sha", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("inputs", type=Path, nargs="+")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    config = sub.add_parser("write-config")
    config.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run-candidate":
        run_candidate(
            repo_root=args.repo_root.resolve(), source_sha=args.source_sha,
            candidate_id=args.candidate_id, output_path=args.output,
            checkpoint_dir=args.checkpoint_dir, token_budgets=tuple(args.token_budgets),
            torch_threads=args.torch_threads,
        )
        return 0
    if args.command == "collect":
        collect_reports(
            input_paths=args.inputs, output_path=args.output,
            expected_source_sha=args.expected_source_sha,
        )
        return 0
    if args.command == "validate":
        validate(args.path, args.expected_source_sha)
        return 0
    payload = config_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
