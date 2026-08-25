"""PERF-59 LOCAL_FREE torch.compile correctness and step-time benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.training.compile_backend import (
    CompileTrainingConfig,
    break_even_step_count,
    build_training_backend,
    compilation_runtime_audit,
    explain_model_graph,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer

SCHEMA = "12-6.perf59-torch-compile.v1"
AUTHORITY = "LOCAL_FREE_COMPILE_EVIDENCE_NOT_GLOBAL_DEFAULT_OR_PAID_COMPUTE_AUTHORIZATION"
FP32_ATOL = 2e-5
FP32_RTOL = 2e-5


@dataclass(frozen=True, slots=True)
class ScaleCase:
    label: str
    spec: ModelSpec
    init: InitSpec
    parity_required: bool


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cases(repo_root: Path) -> tuple[ScaleCase, ...]:
    controls = controlled_specs()
    s3 = load_stage_config(repo_root / "configs/stages/s3_10m.json")
    return (
        ScaleCase("controlled_468k", controls[2], InitSpec(), True),
        ScaleCase("controlled_1m", controls[3], InitSpec(), True),
        ScaleCase("stage_s3_10m", s3.model, s3.init, False),
    )


def _trainer_config(*, max_steps: int, accumulation: int = 1) -> TrainerConfig:
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
        seed=1337,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _batch(spec: ModelSpec, *, step: int, batch_size: int, sequence_length: int) -> Tensor:
    base = torch.arange(batch_size * sequence_length, dtype=torch.long).reshape(
        batch_size,
        sequence_length,
    )
    return (base * 31 + step * 17 + 7).remainder(spec.vocab_size)


def _fresh_model(case: ScaleCase, state: dict[str, Tensor] | None = None) -> TwelveSixDecoder:
    model = TwelveSixDecoder(case.spec, case.init)
    if state is not None:
        model.load_state_dict(state)
    return model


def _reset_dynamo() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def _timed_training(
    case: ScaleCase,
    *,
    compiled: bool,
    batch_size: int,
    sequence_length: int,
    steady_steps: int,
) -> dict[str, Any]:
    total_steps = steady_steps + 1
    torch.manual_seed(1337)
    model = _fresh_model(case)
    config = _trainer_config(max_steps=total_steps)
    if compiled:
        _reset_dynamo()
        trainer = build_training_backend(
            model,
            config,
            compile_config=CompileTrainingConfig(
                enabled=True,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                mode=None,
            ),
            device="cpu",
        )
    else:
        trainer = Trainer(model, config, device="cpu")

    durations: list[float] = []
    losses: list[float] = []
    for step in range(total_steps):
        batch = _batch(
            case.spec,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        durations.append(time.perf_counter() - started)
        losses.append(metrics.loss)

    first = durations[0]
    steady = durations[1:]
    return {
        "first_step_seconds": first,
        "steady_step_seconds": steady,
        "steady_median_seconds": statistics.median(steady),
        "steady_mean_seconds": statistics.fmean(steady),
        "last_loss": losses[-1],
        "optimizer_steps": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }


def _tensor_error(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        return {
            "pass": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs": math.inf,
            "max_scaled_error": math.inf,
        }
    absolute = (reference - candidate).abs()
    tolerance = FP32_ATOL + FP32_RTOL * reference.abs()
    scaled = absolute / tolerance
    max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
    max_scaled = float(scaled.max().item()) if scaled.numel() else 0.0
    return {
        "pass": bool(max_scaled <= 1.0),
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs": max_abs,
        "max_scaled_error": max_scaled,
    }


def _named_tensor_error(
    reference: Iterable[tuple[str, Tensor]],
    candidate: Iterable[tuple[str, Tensor]],
) -> dict[str, Any]:
    reference_map = dict(reference)
    candidate_map = dict(candidate)
    if reference_map.keys() != candidate_map.keys():
        return {
            "pass": False,
            "key_match": False,
            "max_abs": math.inf,
            "max_scaled_error": math.inf,
            "worst_key": None,
        }

    worst_key: str | None = None
    max_abs = 0.0
    max_scaled = 0.0
    passed = True
    for key in reference_map:
        result = _tensor_error(reference_map[key], candidate_map[key])
        passed = passed and bool(result["pass"])
        if float(result["max_scaled_error"]) >= max_scaled:
            max_scaled = float(result["max_scaled_error"])
            max_abs = float(result["max_abs"])
            worst_key = key
    return {
        "pass": passed,
        "key_match": True,
        "max_abs": max_abs,
        "max_scaled_error": max_scaled,
        "worst_key": worst_key,
    }


def _optimizer_tensors(trainer: Trainer) -> list[tuple[str, Tensor]]:
    tensors: list[tuple[str, Tensor]] = []
    state = trainer.optimizer.state_dict()["state"]
    for parameter_id in sorted(state):
        parameter_state = state[parameter_id]
        for key in sorted(parameter_state):
            value = parameter_state[key]
            if isinstance(value, Tensor):
                tensors.append((f"{parameter_id}.{key}", value))
    return tensors


def _parity(
    case: ScaleCase,
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    torch.manual_seed(2026)
    base = _fresh_model(case)
    initial_state = copy.deepcopy(base.state_dict())
    eager_model = _fresh_model(case, initial_state)
    compiled_model = _fresh_model(case, initial_state)
    config = _trainer_config(max_steps=1, accumulation=2)

    eager = Trainer(eager_model, config, device="cpu")
    _reset_dynamo()
    compiled = build_training_backend(
        compiled_model,
        config,
        compile_config=CompileTrainingConfig(
            enabled=True,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            mode=None,
        ),
        device="cpu",
    )

    batch0 = _batch(
        case.spec,
        step=101,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    batch1 = _batch(
        case.spec,
        step=102,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    eager_logits = eager.model(batch0).logits
    compiled_logits = compiled.model(batch0).logits
    logits_error = _tensor_error(eager_logits, compiled_logits)

    eager_first = eager.train_microbatch({"input_ids": batch0})
    compiled_first = compiled.train_microbatch({"input_ids": batch0})
    gradient_error = _named_tensor_error(
        (
            (name, parameter.grad)
            for name, parameter in eager.model.named_parameters()
            if parameter.grad is not None
        ),
        (
            (name, parameter.grad)
            for name, parameter in compiled.model.named_parameters()
            if parameter.grad is not None
        ),
    )
    loss_abs = abs(eager_first.loss - compiled_first.loss)
    loss_tolerance = FP32_ATOL + FP32_RTOL * abs(eager_first.loss)
    loss_pass = loss_abs <= loss_tolerance

    eager_second = eager.train_microbatch({"input_ids": batch1})
    compiled_second = compiled.train_microbatch({"input_ids": batch1})
    parameter_error = _named_tensor_error(
        eager.model.named_parameters(),
        compiled.model.named_parameters(),
    )
    optimizer_error = _named_tensor_error(
        _optimizer_tensors(eager),
        _optimizer_tensors(compiled),
    )
    update_loss_abs = abs(
        float(eager_second.update_loss or 0.0) - float(compiled_second.update_loss or 0.0)
    )
    update_loss_tolerance = FP32_ATOL + FP32_RTOL * abs(float(eager_second.update_loss or 0.0))
    grad_norm_abs = abs(
        float(eager_second.grad_norm or 0.0) - float(compiled_second.grad_norm or 0.0)
    )
    grad_norm_tolerance = FP32_ATOL + FP32_RTOL * abs(float(eager_second.grad_norm or 0.0))

    all_pass = all(
        (
            bool(logits_error["pass"]),
            loss_pass,
            bool(gradient_error["pass"]),
            update_loss_abs <= update_loss_tolerance,
            grad_norm_abs <= grad_norm_tolerance,
            bool(parameter_error["pass"]),
            bool(optimizer_error["pass"]),
        )
    )
    return {
        "tolerance": {"atol": FP32_ATOL, "rtol": FP32_RTOL, "dtype": "fp32"},
        "forward_logits": logits_error,
        "first_microbatch_loss": {
            "pass": loss_pass,
            "eager": eager_first.loss,
            "compiled": compiled_first.loss,
            "abs_error": loss_abs,
        },
        "accumulated_gradients": gradient_error,
        "optimizer_boundary_update_loss": {
            "pass": update_loss_abs <= update_loss_tolerance,
            "eager": eager_second.update_loss,
            "compiled": compiled_second.update_loss,
            "abs_error": update_loss_abs,
        },
        "optimizer_boundary_grad_norm": {
            "pass": grad_norm_abs <= grad_norm_tolerance,
            "eager": eager_second.grad_norm,
            "compiled": compiled_second.grad_norm,
            "abs_error": grad_norm_abs,
        },
        "updated_parameters": parameter_error,
        "optimizer_tensor_state": optimizer_error,
        "all_pass": all_pass,
    }


def run(
    *,
    repo_root: Path,
    output_path: Path,
    expected_source_sha: str | None,
    torch_threads: int,
    batch_size: int,
    sequence_length: int,
    steady_steps: int,
) -> dict[str, Any]:
    if torch_threads <= 0 or batch_size <= 0 or steady_steps <= 0:
        raise ValueError("thread, batch, and steady-step counts must be positive")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    source_sha = _git_head(repo_root)
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise RuntimeError(
            f"exact-checkout mismatch: expected {expected_source_sha}, observed {source_sha}"
        )

    case_reports: list[dict[str, Any]] = []
    parity_pass = True
    for case in _cases(repo_root):
        if sequence_length > case.spec.max_seq_len:
            raise ValueError(f"{case.label}: sequence_length exceeds model max_seq_len")

        torch.manual_seed(1337)
        graph_model = _fresh_model(case)
        graph_batch = _batch(
            case.spec,
            step=0,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        graph_diagnostics = explain_model_graph(graph_model, graph_batch).to_dict()
        _reset_dynamo()

        eager = _timed_training(
            case,
            compiled=False,
            batch_size=batch_size,
            sequence_length=sequence_length,
            steady_steps=steady_steps,
        )
        compiled = _timed_training(
            case,
            compiled=True,
            batch_size=batch_size,
            sequence_length=sequence_length,
            steady_steps=steady_steps,
        )
        eager_steady = float(eager["steady_median_seconds"])
        compiled_steady = float(compiled["steady_median_seconds"])
        speedup = eager_steady / compiled_steady
        break_even = break_even_step_count(
            eager_first_seconds=float(eager["first_step_seconds"]),
            eager_steady_seconds=eager_steady,
            compiled_first_seconds=float(compiled["first_step_seconds"]),
            compiled_steady_seconds=compiled_steady,
        )
        parity = None
        if case.parity_required:
            parity = _parity(
                case,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            parity_pass = parity_pass and bool(parity["all_pass"])

        case_reports.append(
            {
                "label": case.label,
                "parameters": case.spec.parameter_count(),
                "vocab_size": case.spec.vocab_size,
                "max_seq_len": case.spec.max_seq_len,
                "graph_diagnostics": graph_diagnostics,
                "eager": eager,
                "compiled": compiled,
                "steady_state_speedup_eager_over_compiled": speedup,
                "break_even_optimizer_steps": break_even,
                "compile_front_load_seconds_vs_eager_first": max(
                    0.0,
                    float(compiled["first_step_seconds"]) - float(eager["first_step_seconds"]),
                ),
                "parity": parity,
            }
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "runtime": compilation_runtime_audit(),
        "compile_config": CompileTrainingConfig(
            enabled=True,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            mode=None,
        ).to_dict(),
        "benchmark_controls": {
            "device": "cpu",
            "paid_compute": False,
            "torch_threads": torch_threads,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "steady_steps": steady_steps,
            "trainer": asdict(_trainer_config(max_steps=steady_steps + 1)),
        },
        "cases": case_reports,
        "parity_required_scales": ["controlled_468k", "controlled_1m"],
        "strict_fp32_parity_pass": parity_pass,
        "decision": {
            "enable_compile_by_default": False,
            "reason": (
                "A single LOCAL_FREE CPU environment cannot justify a hardware-global default; "
                "the compiled backend remains explicit opt-in even where this report measures a win."
            ),
            "per_case_opt_in_useful": {
                case["label"]: bool(
                    float(case["steady_state_speedup_eager_over_compiled"]) > 1.0
                    and case["break_even_optimizer_steps"] is not None
                )
                for case in case_reports
            },
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("perf59-torch-compile-report.json"))
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--steady-steps", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run(
        repo_root=args.repo_root.resolve(),
        output_path=args.output,
        expected_source_sha=args.expected_source_sha,
        torch_threads=args.torch_threads,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        steady_steps=args.steady_steps,
    )
    print(
        json.dumps(
            {
                "runtime": report["runtime"],
                "strict_fp32_parity_pass": report["strict_fp32_parity_pass"],
                "decision": report["decision"],
                "cases": [
                    {
                        "label": case["label"],
                        "parameters": case["parameters"],
                        "graph_break_count": case["graph_diagnostics"]["graph_break_count"],
                        "first_eager": case["eager"]["first_step_seconds"],
                        "first_compiled": case["compiled"]["first_step_seconds"],
                        "steady_eager": case["eager"]["steady_median_seconds"],
                        "steady_compiled": case["compiled"]["steady_median_seconds"],
                        "speedup": case["steady_state_speedup_eager_over_compiled"],
                        "break_even_steps": case["break_even_optimizer_steps"],
                        "parity_pass": (
                            None if case["parity"] is None else case["parity"]["all_pass"]
                        ),
                    }
                    for case in report["cases"]
                ],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if report["strict_fp32_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
