"""Controlled initialization and deep-scale stability probes.

This module is deliberately an engineering preflight. It never mutates canonical
stage configs or grants stage/promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from twelve_six.model import InitSpec, StageConfig, TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss

REPORT_SCHEMA = "12-6.init-stability-matrix.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_INIT_PREFLIGHT_NOT_STAGE_EVIDENCE"
CandidateKind = Literal[
    "stage_default",
    "unscaled_residual_control",
    "s1_width_reference_control",
]
_CANDIDATES = {
    "stage_default",
    "unscaled_residual_control",
    "s1_width_reference_control",
}


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    batch_size: int = 1
    sequence_length: int = 32
    steps: int = 2
    seeds: tuple[int, ...] = (1337, 1338, 1339)
    data_seed: int = 424242
    width_reference: int = 48

    def __post_init__(self) -> None:
        for name in ("batch_size", "sequence_length", "width_reference"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        for seed in (*self.seeds, self.data_seed):
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("all seeds must be non-negative integers")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_stats(tensor: Tensor) -> dict[str, float | bool]:
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    if value.numel() == 0:
        return {
            "finite": finite,
            "mean": 0.0,
            "std": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
        }
    return {
        "finite": finite,
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "rms": float(value.square().mean().sqrt().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _global_grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().double()
        total += grad.square().sum()
    return float(total.sqrt().item())


def _module_grad_norm(module: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().double()
        total += grad.square().sum()
    return float(total.sqrt().item())


def _candidate_init(stage: StageConfig, kind: CandidateKind, width_reference: int) -> InitSpec:
    if kind not in _CANDIDATES:
        raise ValueError(f"unsupported initialization candidate: {kind!r}")
    if kind == "stage_default":
        return stage.init
    if kind == "unscaled_residual_control":
        return InitSpec(
            schema_version=stage.init.schema_version,
            family=stage.init.family,
            std=stage.init.std,
            residual_branch_scale="none",
        )
    width_scale = math.sqrt(width_reference / stage.model.d_model)
    return InitSpec(
        schema_version=stage.init.schema_version,
        family=stage.init.family,
        std=stage.init.std * width_scale,
        residual_branch_scale=stage.init.residual_branch_scale,
    )


def _activation_hooks(
    model: TwelveSixDecoder,
) -> tuple[dict[str, dict[str, float | bool]], list[Any]]:
    stats: dict[str, dict[str, float | bool]] = {}
    handles: list[Any] = []

    def record(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                stats[name] = _tensor_stats(output)
        return hook

    handles.append(model.token_embedding.register_forward_hook(record("token_embedding")))
    for index, block in enumerate(model.blocks):
        handles.append(block.attn.register_forward_hook(record(f"blocks.{index}.attn_branch")))
        handles.append(block.mlp.register_forward_hook(record(f"blocks.{index}.mlp_branch")))
        handles.append(block.register_forward_hook(record(f"blocks.{index}.residual_stream")))
    handles.append(model.final_norm.register_forward_hook(record("final_norm")))
    return stats, handles


def _fixed_batches(
    *,
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    count: int,
    data_seed: int,
) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    return [
        torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            generator=generator,
            dtype=torch.long,
        )
        for _ in range(count)
    ]


def run_seed_probe(
    *,
    stage: StageConfig,
    candidate: CandidateKind,
    probe: ProbeSpec,
    seed: int,
) -> dict[str, Any]:
    if probe.sequence_length > stage.model.max_seq_len:
        raise ValueError("probe sequence exceeds ModelSpec context")

    init_spec = _candidate_init(stage, candidate, probe.width_reference)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, init_spec=init_spec)
    model.train()

    trainer_config = TrainerConfig(
        max_steps=max(1, probe.steps),
        seed=seed,
        precision="fp32",
        deterministic_algorithms=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=trainer_config.betas,
        eps=trainer_config.eps,
        weight_decay=trainer_config.weight_decay,
    )
    batches = _fixed_batches(
        vocab_size=stage.model.vocab_size,
        batch_size=probe.batch_size,
        sequence_length=probe.sequence_length,
        count=max(1, probe.steps),
        data_seed=probe.data_seed,
    )

    activation_stats, handles = _activation_hooks(model)
    losses: list[float] = []
    grad_norms: list[float] = []
    clip_factors: list[float] = []
    initial_layer_grad_norms: dict[str, float] = {}
    initial_logits: dict[str, float | bool] | None = None

    iterations = max(1, probe.steps)
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        batch = batches[step if probe.steps else 0]
        output = model(batch)
        loss = causal_lm_loss(output.logits, batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite loss during initialization stability probe")
        loss.backward()

        grad_norm = _global_grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite global gradient norm")
        losses.append(float(loss.detach().item()))
        grad_norms.append(grad_norm)

        if step == 0:
            initial_logits = _tensor_stats(output.logits)
            initial_layer_grad_norms = {
                f"blocks.{index}": _module_grad_norm(block)
                for index, block in enumerate(model.blocks)
            }

        clip_norm = trainer_config.gradient_clip_norm
        if clip_norm is None:
            clip_factors.append(1.0)
        else:
            clip_factors.append(min(1.0, clip_norm / max(grad_norm, 1e-30)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        if probe.steps:
            optimizer.step()

        if step == 0:
            for handle in handles:
                handle.remove()

    if initial_logits is None:
        raise RuntimeError("initial logits were not captured")

    uniform_ce = math.log(stage.model.vocab_size)
    return {
        "seed": seed,
        "candidate": candidate,
        "init_spec": init_spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "uniform_cross_entropy": uniform_ce,
        "initial_loss": losses[0],
        "initial_loss_excess_over_uniform": losses[0] - uniform_ce,
        "initial_logits": initial_logits,
        "initial_activations": activation_stats,
        "initial_global_grad_norm": grad_norms[0],
        "initial_layer_grad_norms": initial_layer_grad_norms,
        "losses": losses[: probe.steps] if probe.steps else losses[:1],
        "pre_clip_grad_norms": grad_norms[: probe.steps] if probe.steps else grad_norms[:1],
        "clip_factors": clip_factors[: probe.steps] if probe.steps else clip_factors[:1],
        "all_finite": all(
            math.isfinite(value)
            for value in (
                *losses,
                *grad_norms,
                *(float(item["rms"]) for item in activation_stats.values()),
            )
        ),
    }


def _aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(values: list[float]) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(tensor.mean().item()),
            "pstdev": float(tensor.std(unbiased=False).item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }

    initial_losses = [float(item["initial_loss"]) for item in seed_results]
    grad_norms = [float(item["initial_global_grad_norm"]) for item in seed_results]
    logit_std = [float(item["initial_logits"]["std"]) for item in seed_results]
    final_block_key = max(
        (
            key
            for key in seed_results[0]["initial_activations"]
            if key.endswith(".residual_stream")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    final_hidden_rms = [
        float(item["initial_activations"][final_block_key]["rms"])
        for item in seed_results
    ]
    clip_values = [
        factor
        for item in seed_results
        for factor in item["clip_factors"]
    ]
    return {
        "initial_loss": summary(initial_losses),
        "initial_global_grad_norm": summary(grad_norms),
        "initial_logit_std": summary(logit_std),
        "final_block_hidden_rms": summary(final_hidden_rms),
        "clip_fraction": (
            sum(float(value) < 1.0 for value in clip_values) / len(clip_values)
            if clip_values
            else 0.0
        ),
        "all_finite": all(bool(item["all_finite"]) for item in seed_results),
    }


def run_stage_matrix(
    *,
    stage_config_path: str | Path,
    candidate: CandidateKind,
    probe: ProbeSpec,
    source_sha: str | None = None,
) -> dict[str, Any]:
    if source_sha is not None:
        valid_lengths = {40, 64}
        hex_chars = frozenset("0123456789abcdef")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) not in valid_lengths
            or source_sha != source_sha.lower()
            or any(char not in hex_chars for char in source_sha)
        ):
            raise ValueError("source_sha must be exact lowercase 40- or 64-hex")
    stage_path = Path(stage_config_path)
    stage = load_stage_config(stage_path)
    seed_results = [
        run_seed_probe(stage=stage, candidate=candidate, probe=probe, seed=seed)
        for seed in probe.seeds
    ]
    init_ids = {item["init_identity_sha256"] for item in seed_results}
    if len(init_ids) != 1:
        raise RuntimeError("same candidate produced multiple InitSpec identities")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "stage": stage.stage,
        "stage_config_path": stage_path.as_posix(),
        "expected_parameters": stage.expected_parameters,
        "observed_parameters": stage.model.parameter_count(),
        "model_identity_sha256": stage.model.identity_sha256(),
        "canonical_stage_init_identity_sha256": stage.init.identity_sha256(),
        "candidate": candidate,
        "candidate_init_identity_sha256": next(iter(init_ids)),
        "probe": probe.to_dict(),
        "source_sha": source_sha,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "precision": "fp32",
        },
        "seed_results": seed_results,
        "aggregate": _aggregate_seed_results(seed_results),
        "truth_boundary": {
            "canonical_init_changed": False,
            "stage_promotion_granted": False,
            "paid_compute_used": False,
            "quality_or_capability_evidence": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    payload = dict(report)
    expected_hash = payload.pop("report_sha256", None)
    if expected_hash != canonical_json_sha256(payload):
        raise ValueError("init stability report SHA-256 mismatch")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unexpected init stability report schema")
    if report.get("authority") != AUTHORITY:
        raise ValueError("unexpected init stability report authority")
    truth = report.get("truth_boundary")
    if not isinstance(truth, dict):
        raise ValueError("missing truth boundary")
    if truth != {
        "canonical_init_changed": False,
        "stage_promotion_granted": False,
        "paid_compute_used": False,
        "quality_or_capability_evidence": False,
    }:
        raise ValueError("init stability truth boundary was weakened")
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or aggregate.get("all_finite") is not True:
        raise ValueError("init stability report contains non-finite probe state")
    if report.get("observed_parameters") != report.get("expected_parameters"):
        raise ValueError("stage parameter count mismatch in init stability report")


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    validate_report(report)
    Path(path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
