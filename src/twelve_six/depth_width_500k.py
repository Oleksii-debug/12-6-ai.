"""MODEL-09 iso-parameter depth-vs-width experiment around 500K parameters.

The scientific execution loop remains the RESEARCH06 fixed-valid-token runner.
This module follows the live MODEL08 wrapper pattern: it temporarily exposes a
predeclared ~500K candidate family and instruments the canonical Trainer in the
same process.  Token accounting, held-out evaluation, checkpoint/resume, data
trace, initialization and optimizer semantics therefore remain owned by the
incumbent runner rather than being reimplemented here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, ClassVar, Mapping

import torch

from . import fixed_token_research as research
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .training import Trainer as CanonicalTrainer

SCHEMA = "12-6.model09-depth-width-500k-collection.v1"
AUTHORITY = "LOCAL_FREE_MODEL09_500K_ARCHITECTURE_RESEARCH_NOT_PROMOTION"
FAMILY = "depth_width_500k"
DEFAULT_BUDGETS = (16_384, 65_536)

_GEOMETRIES: tuple[tuple[str, dict[str, int]], ...] = (
    (
        "shallow_wide",
        {"d_model":136,"n_layers":2,"n_heads":4,"n_kv_heads":4,"head_dim":34,"d_ff":384},
    ),
    (
        "mid_shallow",
        {"d_model":112,"n_layers":3,"n_heads":4,"n_kv_heads":4,"head_dim":28,"d_ff":320},
    ),
    (
        "balanced",
        {"d_model":96,"n_layers":4,"n_heads":4,"n_kv_heads":4,"head_dim":24,"d_ff":280},
    ),
    (
        "deep_narrow",
        {"d_model":80,"n_layers":6,"n_heads":4,"n_kv_heads":4,"head_dim":20,"d_ff":224},
    ),
    (
        "very_deep_narrow",
        {"d_model":72,"n_layers":8,"n_heads":4,"n_kv_heads":4,"head_dim":18,"d_ff":184},
    ),
)
_EXPECTED: Mapping[str, tuple[int, str]] = {
    "shallow_wide": (496_808, "4ef570a2737363e6773fbac5491797a97bee3c490e0dc90c9612141157163a09"),
    "mid_shallow": (502_544, "078741421f86ed366d5ea495dc00d2dac8133548ff7b255f5ef9573291fa3ffc"),
    "balanced": (495_456, "eac28a6dbd474591fda57e6cdeaff7fbd9d11594ae9e50750822ee3fe3fbc9e5"),
    "deep_narrow": (497_680, "1e583c2609d8347e01c7497df18224b8ac71197ccf7c3d8a402e1844d24b4c12"),
    "very_deep_narrow": (503_496, "0c03d075a2330ecd3c8f8f1a60cd6edef495b829834a236b04d37dd8eb726845"),
}


def candidate_specs() -> dict[str, ModelSpec]:
    """Return the immutable pre-validation ~500K candidate family."""
    specs = {name: research._model_spec(geometry) for name, geometry in _GEOMETRIES}
    for name, spec in specs.items():
        count, identity = _EXPECTED[name]
        if spec.parameter_count() != count or spec.identity_sha256() != identity:
            raise RuntimeError(f"MODEL09 ModelSpec drift for {name}")
        allocation = spec.parameter_breakdown()
        if not spec.tie_word_embeddings or allocation["lm_head_extra"] != 0:
            raise RuntimeError("MODEL09 requires tied token embedding/output weights")
        if allocation["token_embedding"] != 256 * spec.d_model:
            raise RuntimeError("MODEL09 tied embedding parameter accounting drift")
    return specs


def config_payload() -> dict[str, Any]:
    init = InitSpec()
    return {
        "schema": "12-6.model09-depth-width-500k-config.v1",
        "authority": "PREDECLARED_EXPERIMENT_CONFIG_NOT_CANONICAL_ARCHITECTURE",
        "parent": "RESEARCH06_MODEL08_EXACT_FIXED_TOKEN_RUNNER",
        "fixed_controls": {
            "token_budgets": list(DEFAULT_BUDGETS),
            "tokenizer": research.BYTE_TOKENIZER_VERSION,
            "vocab_size": 256,
            "max_seq_len": 256,
            "batch_size": research.DEFAULT_BATCH_SIZE,
            "sequence_length": research.DEFAULT_SEQUENCE_LENGTH,
            "seed": research.DEFAULT_SEED,
            "trace_protocol": research.TRACE_PROTOCOL,
            "research41_parent_packing": research.RESEARCH41_PACKING_ID,
            "init_spec": init.to_dict(),
            "init_identity_sha256": init.identity_sha256(),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "scheduler": "constant",
                "warmup_steps": 0,
                "precision": "fp32",
            },
        },
        "selection_rule": (
            "lowest final held-out validation loss; exact tie -> lower median "
            "optimizer-step wall time"
        ),
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


def _block_snapshot(
    model: TwelveSixDecoder,
) -> dict[int, tuple[list[torch.Tensor], float]]:
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


def _block_update_ratios(
    model: TwelveSixDecoder,
    before: Mapping[int, tuple[list[torch.Tensor], float]],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for index, block in enumerate(model.blocks):
        prior, weight_norm = before[index]
        delta_sq = 0.0
        parameters = (parameter for parameter in block.parameters() if parameter.requires_grad)
        for parameter, old in zip(parameters, prior, strict=True):
            delta = parameter.detach().cpu().float() - old.float()
            delta_sq += float(torch.sum(delta * delta).item())
        result[index] = math.sqrt(delta_sq) / weight_norm if weight_norm > 0 else math.inf
    return result


class RecordingTrainer(CanonicalTrainer):
    """Canonical Trainer plus training-only block telemetry for MODEL09."""

    observations: ClassVar[list[dict[str, Any]]] = []

    def train_microbatch(self, batch: Any):  # type: ignore[override]
        if self.config.gradient_accumulation_steps != 1:
            raise RuntimeError("MODEL09 layer telemetry requires frozen grad_accum=1")
        block_before = _block_snapshot(self.model)
        activations: dict[int, tuple[float, float]] = {}
        raw_grad_sq: dict[int, float] = {}
        handles: list[Any] = []

        def forward_hook(index: int):
            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError("MODEL09 expected tensor TransformerBlock output")
                value = output.detach().float()
                activations[index] = (
                    float(torch.sqrt(torch.mean(value * value)).item()),
                    float(value.abs().max().item()),
                )
            return capture

        def grad_hook(index: int):
            def capture(gradient: torch.Tensor) -> torch.Tensor:
                value = gradient.detach().float()
                raw_grad_sq[index] = raw_grad_sq.get(index, 0.0) + float(
                    torch.sum(value * value).item()
                )
                return gradient
            return capture

        for index, block in enumerate(self.model.blocks):
            handles.append(block.register_forward_hook(forward_hook(index)))
            for parameter in block.parameters():
                if parameter.requires_grad:
                    handles.append(parameter.register_hook(grad_hook(index)))
        started = time.perf_counter()
        try:
            metrics = super().train_microbatch(batch)
        finally:
            elapsed = time.perf_counter() - started
            for handle in handles:
                handle.remove()
        if not metrics.optimizer_stepped or metrics.tokens <= 0:
            raise RuntimeError("MODEL09 expected one committed optimizer update per microbatch")
        ratios = _block_update_ratios(self.model, block_before)
        layers: list[dict[str, Any]] = []
        for index in range(len(self.model.blocks)):
            if index not in activations or index not in raw_grad_sq:
                raise RuntimeError(f"MODEL09 missing block telemetry for layer {index}")
            activation_rms, activation_max_abs = activations[index]
            # Canonical Trainer backpropagates loss * valid_tokens and then divides
            # parameter gradients by the pending valid-token count before clipping.
            # Hooks observe the summed gradient, so normalize by metrics.tokens.
            grad_norm = math.sqrt(raw_grad_sq[index]) / int(metrics.tokens)
            row = {
                "layer": index,
                "activation_rms": activation_rms,
                "activation_max_abs": activation_max_abs,
                "pre_clip_grad_norm": grad_norm,
                "update_to_weight_ratio": ratios[index],
            }
            if not all(math.isfinite(float(row[key])) for key in (
                "activation_rms",
                "activation_max_abs",
                "pre_clip_grad_norm",
                "update_to_weight_ratio",
            )):
                raise RuntimeError("MODEL09 non-finite block telemetry")
            layers.append(row)
        self.observations.append(
            {
                "optimizer_step": int(metrics.optimizer_step),
                "optimized_tokens": int(self.tokens_seen),
                "valid_causal_tokens": int(metrics.tokens),
                "train_loss": float(metrics.loss),
                "update_loss": (
                    None if metrics.update_loss is None else float(metrics.update_loss)
                ),
                "pre_clip_grad_norm": (
                    None if metrics.grad_norm is None else float(metrics.grad_norm)
                ),
                "step_seconds": elapsed,
                "layers": layers,
            }
        )
        return metrics


def _layer_summary(
    observations: list[dict[str, Any]], n_layers: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in range(n_layers):
        rows = [
            layer_row
            for observation in observations
            for layer_row in observation["layers"]
            if int(layer_row["layer"]) == layer
        ]
        if not rows:
            raise RuntimeError(f"MODEL09 empty telemetry for layer {layer}")
        result.append(
            {
                "layer": layer,
                "samples": len(rows),
                "activation_rms": research._summary(
                    [float(row["activation_rms"]) for row in rows]
                ),
                "activation_max_abs": research._summary(
                    [float(row["activation_max_abs"]) for row in rows]
                ),
                "pre_clip_grad_norm": research._summary(
                    [float(row["pre_clip_grad_norm"]) for row in rows]
                ),
                "update_to_weight_ratio": research._summary(
                    [float(row["update_to_weight_ratio"]) for row in rows]
                ),
            }
        )
    return result


def run_model09_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    candidate_id: str,
    output_path: Path,
    checkpoint_dir: Path,
    token_budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    batch_size: int = research.DEFAULT_BATCH_SIZE,
    sequence_length: int = research.DEFAULT_SEQUENCE_LENGTH,
    seed: int = research.DEFAULT_SEED,
    torch_threads: int = research.DEFAULT_THREADS,
) -> dict[str, Any]:
    budgets = research._validate_budgets(token_budgets)
    specs = candidate_specs()
    if candidate_id not in specs:
        raise ValueError(f"unknown MODEL09 candidate {candidate_id!r}")
    RecordingTrainer.observations = []
    original_trainer = research.Trainer
    original_candidate_specs = research.candidate_specs

    def patched_candidate_specs(family: str) -> dict[str, ModelSpec]:
        if family == FAMILY:
            return specs
        return original_candidate_specs(family)

    research.Trainer = RecordingTrainer
    research.candidate_specs = patched_candidate_specs
    try:
        report = research.run_candidate(
            repo_root=repo_root,
            source_sha=source_sha,
            family=FAMILY,
            candidate_id=candidate_id,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            token_budgets=budgets,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
            torch_threads=torch_threads,
            exercise_resume=True,
        )
    finally:
        research.candidate_specs = original_candidate_specs
        research.Trainer = original_trainer

    observations = list(RecordingTrainer.observations)
    if not observations:
        raise RuntimeError("MODEL09 captured no optimizer-step telemetry")
    if int(observations[-1]["optimized_tokens"]) != budgets[-1]:
        raise RuntimeError("MODEL09 training telemetry token ledger drift")
    if sum(int(row["valid_causal_tokens"]) for row in observations) != budgets[-1]:
        raise RuntimeError("MODEL09 valid-causal-token telemetry sum drift")
    expected_steps = int(report["trace_steps"])
    if len(observations) != expected_steps:
        raise RuntimeError("MODEL09 telemetry/trace step-count drift")
    step_times = [float(row["step_seconds"]) for row in observations]
    report["model09_telemetry"] = {
        "definition": (
            "Training-only per-optimizer-step loss, wall time, block activation RMS/max, "
            "normalized pre-clip block gradient norm, and block update/weight ratio. "
            "Train loss is diagnostic and never substitutes for held-out validation."
        ),
        "all_steps": observations,
        "step_time_seconds": research._summary(step_times),
        "layer_summary": _layer_summary(observations, specs[candidate_id].n_layers),
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = research._canonical_hash(report)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    research._validate_candidate(report, expected_source_sha=source_sha)
    return report


def _validate_model09_candidate(
    report: dict[str, Any], *, expected_source_sha: str | None = None
) -> None:
    research._validate_candidate(report, expected_source_sha=expected_source_sha)
    if report.get("family") != FAMILY:
        raise ValueError("MODEL09 family mismatch")
    candidate = str(report["candidate_id"])
    specs = candidate_specs()
    if candidate not in specs:
        raise ValueError("MODEL09 unknown candidate")
    spec = specs[candidate]
    if report.get("model_identity_sha256") != spec.identity_sha256():
        raise ValueError("MODEL09 ModelSpec identity mismatch")
    telemetry = report.get("model09_telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("MODEL09 telemetry missing")
    if len(telemetry.get("layer_summary", [])) != spec.n_layers:
        raise ValueError("MODEL09 layer telemetry incomplete")
    if sum(
        int(row["valid_causal_tokens"]) for row in telemetry.get("all_steps", [])
    ) != int(report["controls"]["token_budgets"][-1]):
        raise ValueError("MODEL09 telemetry optimized-token drift")


def collect_reports(
    *, input_paths: list[Path], output_path: Path, expected_source_sha: str
) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    for report in reports:
        _validate_model09_candidate(report, expected_source_sha=expected_source_sha)
    if {str(report["candidate_id"]) for report in reports} != set(candidate_specs()):
        raise ValueError("MODEL09 candidate set mismatch")
    controls = {str(report["controls_sha256"]) for report in reports}
    traces = {str(report["trace_sha256"]) for report in reports}
    if len(controls) != 1:
        raise ValueError("MODEL09 fixed-control identity drift")
    if len(traces) != 1:
        raise ValueError("MODEL09 exact data/token trace drift")

    rows: list[dict[str, Any]] = []
    for report in reports:
        final = report["checkpoints"][-1]
        telemetry = report["model09_telemetry"]
        rows.append(
            {
                "candidate_id": report["candidate_id"],
                "parameters": int(report["parameters"]),
                "model_identity_sha256": report["model_identity_sha256"],
                "model_spec": report["model_spec"],
                "parameter_allocation": report["parameter_allocation"],
                "validation_loss": float(final["validation_loss"]),
                "validation_bpb": float(final["validation_bpb"]),
                "validation_improvement": float(report["final_validation_improvement"]),
                "compute_proxy": int(final["compute_proxy"]),
                "optimization_wall_seconds": float(report["optimization_wall_seconds"]),
                "experiment_wall_seconds": float(report["experiment_wall_seconds"]),
                "median_step_seconds": float(telemetry["step_time_seconds"]["median"]),
                "tokens_per_second": float(report["optimized_tokens_per_optimization_second"]),
                "rss_hwm_mib": float(report["memory"]["process_rss_hwm_mib"]),
                "model_parameter_tensor_bytes": int(
                    report["memory"]["model_parameter_tensor_bytes"]
                ),
                "optimizer_tensor_bytes": int(report["memory"]["optimizer_tensor_bytes"]),
                "gradient_norm": report["gradient_norm"],
                "clip_frequency": float(report["clip_frequency"]),
                "update_ratio": report["update_ratio"],
                "layer_summary": telemetry["layer_summary"],
            }
        )
    ordered = sorted(
        rows,
        key=lambda row: (float(row["validation_loss"]), float(row["median_step_seconds"])),
    )
    winner = ordered[0]
    counts = [int(row["parameters"]) for row in rows]
    collection: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": expected_source_sha,
        "family": FAMILY,
        "fixed_token_budget": int(reports[0]["controls"]["token_budgets"][-1]),
        "controls_sha256": next(iter(controls)),
        "exact_trace_sha256": next(iter(traces)),
        "candidate_parameter_span": {
            "min": min(counts),
            "max": max(counts),
            "absolute_span": max(counts) - min(counts),
        },
        "rows": sorted(rows, key=lambda row: int(row["model_spec"]["n_layers"])),
        "ranking": [str(row["candidate_id"]) for row in ordered],
        "selection_rule": (
            "lowest final held-out validation loss; exact tie -> lower median "
            "optimizer-step wall time"
        ),
        "recommended_500k_geometry": {
            "candidate_id": winner["candidate_id"],
            "parameters": winner["parameters"],
            "model_spec": winner["model_spec"],
            "validation_loss": winner["validation_loss"],
            "validation_bpb": winner["validation_bpb"],
        },
        "truth_boundary": [
            "Recommendation is local to the frozen tiny S0 fixture and exact 65,536-token budget.",
            "Candidate set and selection rule were committed before held-out results.",
            "Train loss is retained as diagnostics, never as generalization evidence.",
            "No paid compute, stage promotion, architecture freeze, or capability claim is implied.",
        ],
    }
    collection["report_sha256"] = research._canonical_hash(collection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return collection


def validate(path: Path, expected_source_sha: str | None = None) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") == research.SCHEMA:
        _validate_model09_candidate(report, expected_source_sha=expected_source_sha)
        return
    if report.get("schema") != SCHEMA:
        raise ValueError("MODEL09 collection schema mismatch")
    material = dict(report)
    claimed = material.pop("report_sha256", None)
    if claimed != research._canonical_hash(material):
        raise ValueError("MODEL09 collection self-hash mismatch")
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise ValueError("MODEL09 collection source SHA mismatch")
    if {str(row["candidate_id"]) for row in report.get("rows", [])} != set(
        candidate_specs()
    ):
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
    run.add_argument("--torch-threads", type=int, default=research.DEFAULT_THREADS)
    collect = sub.add_parser("collect")
    collect.add_argument("--expected-source-sha", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("inputs", type=Path, nargs="+")
    verify = sub.add_parser("validate")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-source-sha")
    config = sub.add_parser("write-config")
    config.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run-candidate":
        run_model09_candidate(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            candidate_id=args.candidate_id,
            output_path=args.output,
            checkpoint_dir=args.checkpoint_dir,
            token_budgets=tuple(args.token_budgets),
            torch_threads=args.torch_threads,
        )
        return 0
    if args.command == "collect":
        collect_reports(
            input_paths=args.inputs,
            output_path=args.output,
            expected_source_sha=args.expected_source_sha,
        )
        return 0
    if args.command == "validate":
        validate(args.path, args.expected_source_sha)
        return 0
    payload = config_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
