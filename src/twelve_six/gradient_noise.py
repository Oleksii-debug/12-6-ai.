"""Controlled non-mutating gradient-stochasticity diagnostics for 12-6 AI."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    controlled_specs,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig, causal_lm_loss

SCHEMA = "12-6.gradient-stochasticity.v1"
CONFIG_SCHEMA = "12-6.gradient-stochasticity-config.v1"
AUTHORITY = "LOCAL_FREE_OPTIMIZATION_DIAGNOSTIC_NOT_STAGE_OR_UNIVERSAL_NOISE_SCALE"
NOISE_PROXY_NAME = "gradient_variance_trace_over_squared_mean_gradient_norm"
NOISE_PROXY_FORMULA = "E[||g_i-g_bar||^2] / max(||g_bar||^2, epsilon)"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        digest.update(b"tensor\0" + str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(digest, (type(key).__name__, repr(key)))
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _hash_value(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())
    else:
        raise TypeError(f"unsupported fingerprint type: {type(value)!r}")


def state_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def model_state_fingerprint(model: torch.nn.Module) -> str:
    return state_fingerprint(dict(model.state_dict()))


def optimizer_state_fingerprint(trainer: Trainer) -> str:
    return state_fingerprint(trainer.optimizer.state_dict())


def gradient_state_fingerprint(model: torch.nn.Module) -> str:
    return state_fingerprint(
        {
            name: None if parameter.grad is None else parameter.grad
            for name, parameter in model.named_parameters()
        }
    )


def _rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_fingerprint(state: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {
        "python": repr(state["python"]),
        "cpu": state["cpu"],
    }
    if "cuda" in state:
        payload["cuda"] = state["cuda"]
    return state_fingerprint(payload)


def _guard(trainer: Trainer) -> dict[str, Any]:
    trainer_state = trainer.state_dict()
    return {
        "model": model_state_fingerprint(trainer.model),
        "optimizer": optimizer_state_fingerprint(trainer),
        "trainer": state_fingerprint(asdict(trainer_state)),
        "grads": gradient_state_fingerprint(trainer.model),
        "mode": trainer.model.training,
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(variance, 0.0))


def _proxy(mean_squared_norm: float, squared_mean_norm: float) -> dict[str, float]:
    variance_trace = max(mean_squared_norm - squared_mean_norm, 0.0)
    return {
        "mean_squared_norm": mean_squared_norm,
        "squared_mean_gradient_norm": squared_mean_norm,
        "variance_trace": variance_trace,
        "noise_to_signal_trace_ratio": variance_trace / max(squared_mean_norm, 1e-30),
    }


def _squared_norm(grads: Sequence[Tensor]) -> float:
    return math.fsum(float(torch.sum(grad.float().square()).item()) for grad in grads)


def estimate_gradient_stochasticity(
    trainer: Trainer,
    batches: Sequence[Mapping[str, Tensor]],
    *,
    clip_norm: float | None,
    virtual_batch_multipliers: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    """Estimate microbatch gradient variation without mutating Trainer state."""
    repeats = len(batches)
    multipliers = tuple(virtual_batch_multipliers)
    if repeats < 2:
        raise ValueError("at least two repeated microbatches are required")
    if not multipliers or multipliers[0] != 1 or multipliers != tuple(sorted(set(multipliers))):
        raise ValueError("virtual batch multipliers must start at 1 and be strictly increasing")
    if any(multiplier <= 0 or repeats % multiplier for multiplier in multipliers):
        raise ValueError("every virtual batch multiplier must divide repeat count")

    named = [(name, p) for name, p in trainer.model.named_parameters() if p.requires_grad]
    parameters = tuple(parameter for _, parameter in named)
    blocks: dict[str, list[int]] = {}
    for index, (name, _) in enumerate(named):
        parts = name.split(".", 2)
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            blocks.setdefault(f"block_{int(parts[1]):02d}", []).append(index)

    before = _guard(trainer)
    rng_before = _rng_state()
    rng_sha = _rng_fingerprint(rng_before)
    sums = [torch.zeros_like(parameter) for parameter in parameters]
    group_sums = {
        multiplier: [torch.zeros_like(parameter) for parameter in parameters]
        for multiplier in multipliers
        if multiplier > 1
    }
    group_squared: dict[int, list[float]] = {multiplier: [] for multiplier in multipliers}
    norms: list[float] = []
    squared_norms: list[float] = []
    losses: list[float] = []
    block_norms = {block: [] for block in blocks}
    block_squared = {block: 0.0 for block in blocks}
    block_sums = {
        block: [torch.zeros_like(parameters[index]) for index in indices]
        for block, indices in blocks.items()
    }
    would_clip = 0

    try:
        for repeat_index, batch in enumerate(batches):
            input_ids = batch["input_ids"].to(trainer.device)
            loss = causal_lm_loss(trainer.model(input_ids).logits, input_ids)
            grads = torch.autograd.grad(loss, parameters, allow_unused=False)
            if not torch.isfinite(loss).item() or any(
                not torch.isfinite(grad).all().item() for grad in grads
            ):
                raise FloatingPointError("non-finite gradient diagnostic")

            squared = _squared_norm(grads)
            norm = math.sqrt(max(squared, 0.0))
            losses.append(float(loss.detach().float().item()))
            squared_norms.append(squared)
            norms.append(norm)
            if clip_norm is not None and norm > clip_norm:
                would_clip += 1
            for index, grad in enumerate(grads):
                sums[index].add_(grad.detach())

            group_squared[1].append(squared)
            for multiplier, accumulators in group_sums.items():
                for index, grad in enumerate(grads):
                    accumulators[index].add_(grad.detach())
                if (repeat_index + 1) % multiplier == 0:
                    group_squared[multiplier].append(
                        math.fsum(
                            float(torch.sum((value.float() / multiplier).square()).item())
                            for value in accumulators
                        )
                    )
                    for value in accumulators:
                        value.zero_()

            for block, indices in blocks.items():
                squared = math.fsum(
                    float(torch.sum(grads[index].float().square()).item()) for index in indices
                )
                block_squared[block] += squared
                block_norms[block].append(math.sqrt(max(squared, 0.0)))
                for local, index in enumerate(indices):
                    block_sums[block][local].add_(grads[index].detach())

        squared_mean = math.fsum(
            float(torch.sum((value.float() / repeats).square()).item()) for value in sums
        )
        mean_norm, std_norm = _mean_std(norms)
        mean_loss, std_loss = _mean_std(losses)
        per_block = []
        for block, values in sorted(block_norms.items()):
            block_mean_squared = math.fsum(
                float(torch.sum((value.float() / repeats).square()).item())
                for value in block_sums[block]
            )
            mean_block_norm, std_block_norm = _mean_std(values)
            per_block.append(
                {
                    "block": block,
                    "mean_grad_norm": mean_block_norm,
                    "std_grad_norm": std_block_norm,
                    "mean_gradient_norm": math.sqrt(max(block_mean_squared, 0.0)),
                    **_proxy(block_squared[block] / repeats, block_mean_squared),
                }
            )
        virtual_batch = []
        for multiplier in multipliers:
            values = group_squared[multiplier]
            virtual_batch.append(
                {
                    "microbatches_per_virtual_batch": multiplier,
                    "independent_groups_observed": len(values),
                    "effective_examples": int(batches[0]["input_ids"].shape[0]) * multiplier,
                    **_proxy(math.fsum(values) / len(values), squared_mean),
                }
            )
        result = {
            "repeat_microbatches": repeats,
            "loss": {"mean": mean_loss, "std": std_loss},
            "global": {
                "mean_grad_norm": mean_norm,
                "std_grad_norm": std_norm,
                "min_grad_norm": min(norms),
                "max_grad_norm": max(norms),
                "mean_gradient_norm": math.sqrt(max(squared_mean, 0.0)),
                **_proxy(math.fsum(squared_norms) / repeats, squared_mean),
            },
            "per_block": per_block,
            "clip_threshold": clip_norm,
            "probe_would_clip_frequency": None if clip_norm is None else would_clip / repeats,
            "virtual_batch_noise_proxy": virtual_batch,
            "noise_proxy": {
                "name": NOISE_PROXY_NAME,
                "formula": NOISE_PROXY_FORMULA,
                "universal_gradient_noise_scale_claimed": False,
            },
        }
    finally:
        _restore_rng(rng_before)

    after = _guard(trainer)
    if before != after or _rng_fingerprint(_rng_state()) != rng_sha:
        raise RuntimeError("gradient diagnostic mutated model/Trainer/gradient/RNG state")
    result["non_mutation"] = {
        "verified": True,
        "model_state_sha256": before["model"],
        "optimizer_state_sha256": before["optimizer"],
        "trainer_state_sha256": before["trainer"],
        "gradient_state_sha256": before["grads"],
        "rng_state_sha256": rng_sha,
    }
    return result


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("gradient stochasticity config schema mismatch")
    checkpoints = config.get("training", {}).get("checkpoints")
    probe = config.get("probe", {})
    if not isinstance(checkpoints, list) or checkpoints != sorted(set(checkpoints)):
        raise ValueError("checkpoints must be a strictly increasing list")
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("checkpoints must start at zero")
    repeats = probe.get("repeat_microbatches")
    multipliers = probe.get("virtual_batch_multipliers")
    if not isinstance(repeats, int) or repeats < 2 or not isinstance(multipliers, list):
        raise ValueError("invalid repeat microbatch configuration")
    if any(not isinstance(value, int) or value <= 0 or repeats % value for value in multipliers):
        raise ValueError("virtual batch multipliers must divide repeat count")
    return config


def _update_ratio(model: torch.nn.Module, before: Sequence[Tensor]) -> dict[str, float]:
    delta_squared = 0.0
    weight_squared = 0.0
    for parameter, prior in zip(model.parameters(), before, strict=True):
        delta = parameter.detach().float() - prior.float()
        delta_squared += float(torch.sum(delta.square()).item())
        weight_squared += float(torch.sum(prior.float().square()).item())
    update_l2 = math.sqrt(delta_squared)
    weight_l2 = math.sqrt(weight_squared)
    return {
        "update_l2": update_l2,
        "weight_l2_before_update": weight_l2,
        "update_to_weight_ratio": update_l2 / max(weight_l2, 1e-30),
    }


def _probe_batches(
    stream: bytes,
    *,
    checkpoint_index: int,
    batch_size: int,
    sequence_length: int,
    repeats: int,
    offset: int,
) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    steps = [offset + checkpoint_index * repeats + index for index in range(repeats)]
    batches = [
        {
            "input_ids": _make_batch(
                stream,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        }
        for step in steps
    ]
    trace = {
        "packing_id": PACKING_ID,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }
    trace["trace_sha256"] = _canonical_hash(trace)
    return batches, trace


def _decision_support(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_scale = []
    for run in runs:
        trained = [point for point in run["checkpoints"] if point["optimizer_step"] > 0]
        clip_steps = sum(point["interval_clipping"]["steps"] for point in trained)
        clipped = sum(point["interval_clipping"]["clipped_steps"] for point in trained)
        by_scale.append(
            {
                "parameters": run["parameters"],
                "mean_trained_noise_proxy": math.fsum(
                    point["probe"]["global"]["noise_to_signal_trace_ratio"] for point in trained
                )
                / len(trained),
                "mean_update_to_weight_ratio": math.fsum(
                    point["update"]["update_to_weight_ratio"] for point in trained
                )
                / len(trained),
                "training_clip_frequency": clipped / clip_steps,
            }
        )
    return {
        "by_scale": by_scale,
        "batch_size_use": (
            "Use the measured proxy and empirical 1x/2x/4x virtual-batch reduction; "
            "no universal batch-size threshold is asserted."
        ),
        "clipping_use": (
            "Use actual Trainer pre-clip norms and clip frequency; qualify clipping and LR jointly."
        ),
        "learning_rate_use": (
            "Compare update/weight ratios at the identical LR; growth with scale justifies testing "
            "a lower LR, while shrinkage alone does not authorize a higher LR."
        ),
        "universal_gradient_noise_scale_claimed": False,
    }


def run_gradient_noise_matrix(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_sha) or _git_head(repo_root) != source_sha:
        raise RuntimeError("exact source identity mismatch")
    config = load_experiment_config(config_path)
    runtime = config["runtime"]
    training = config["training"]
    probe_config = config["probe"]
    if runtime.get("device") != "cpu" or runtime.get("paid_compute") is not False:
        raise ValueError("RESEARCH-20 requires LOCAL_FREE CPU execution")

    torch_threads = int(runtime["torch_threads"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    checkpoints = tuple(int(value) for value in training["checkpoints"])
    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    repeats = int(probe_config["repeat_microbatches"])
    multipliers = tuple(int(value) for value in probe_config["virtual_batch_multipliers"])
    probe_offset = int(probe_config["batch_trace_offset"])
    trainer_config: TrainerConfig = _trainer_config(max_steps=checkpoints[-1], seed=1337)

    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    overlap = sorted(
        {str(record["id"]) for record in train_records}
        & {str(record["id"]) for record in validation_records}
    )
    if overlap:
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    stream = _byte_stream(train_records, tokenizer)
    init_spec = InitSpec()

    runs = []
    for spec in controlled_specs():
        random.seed(trainer_config.seed)
        torch.manual_seed(trainer_config.seed)
        model = TwelveSixDecoder(spec, init_spec)
        trainer = Trainer(model, trainer_config, device="cpu")
        points = []
        clip_count = 0
        interval_steps = 0
        latest_update = None
        latest_metrics = None
        started = time.perf_counter()
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            while trainer.optimizer_step < checkpoint:
                measure_update = trainer.optimizer_step + 1 == checkpoint
                before = (
                    [parameter.detach().clone() for parameter in model.parameters()]
                    if measure_update
                    else None
                )
                batch = _make_batch(
                    stream,
                    step=trainer.optimizer_step,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
                latest_metrics = trainer.train_microbatch({"input_ids": batch})
                if not latest_metrics.optimizer_stepped or latest_metrics.grad_norm is None:
                    raise RuntimeError("matrix requires one optimizer update per microbatch")
                if before is not None:
                    latest_update = _update_ratio(model, before)
                interval_steps += 1
                if latest_metrics.grad_norm > float(trainer_config.gradient_clip_norm):
                    clip_count += 1

            batches, trace = _probe_batches(
                stream,
                checkpoint_index=checkpoint_index,
                batch_size=batch_size,
                sequence_length=sequence_length,
                repeats=repeats,
                offset=probe_offset,
            )
            points.append(
                {
                    "optimizer_step": trainer.optimizer_step,
                    "tokens_seen": trainer.tokens_seen,
                    "model_state_sha256": model_state_fingerprint(model),
                    "optimizer_state_sha256": optimizer_state_fingerprint(trainer),
                    "last_train_loss": None if latest_metrics is None else latest_metrics.loss,
                    "last_pre_clip_grad_norm": (
                        None if latest_metrics is None else latest_metrics.grad_norm
                    ),
                    "update": None if checkpoint == 0 else latest_update,
                    "interval_clipping": {
                        "threshold": trainer_config.gradient_clip_norm,
                        "steps": interval_steps,
                        "clipped_steps": clip_count,
                        "frequency": clip_count / interval_steps if interval_steps else 0.0,
                    },
                    "probe_batch_trace": trace,
                    "probe": estimate_gradient_stochasticity(
                        trainer,
                        batches,
                        clip_norm=trainer_config.gradient_clip_norm,
                        virtual_batch_multipliers=multipliers,
                    ),
                }
            )
            clip_count = 0
            interval_steps = 0
        runs.append(
            {
                "parameters": spec.parameter_count(),
                "model_spec": spec.to_dict(),
                "model_identity_sha256": spec.identity_sha256(),
                "parameter_breakdown": spec.parameter_breakdown(),
                "wall_seconds": time.perf_counter() - started,
                "checkpoints": points,
            }
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optimizer_payload = asdict(trainer_config)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "experiment_config": {
            "path": str(config_path.relative_to(repo_root)),
            "sha256": _file_sha256(config_path),
            "semantic_sha256": _canonical_hash(config),
            "payload": config,
        },
        "identities": {
            "init_spec": init_spec.to_dict(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "optimizer": "torch.optim.AdamW",
            "trainer_config": optimizer_payload,
            "trainer_config_sha256": _canonical_hash(optimizer_payload),
            "packing_id": PACKING_ID,
        },
        "data": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
            "manifest_sha256": _file_sha256(manifest_path),
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "train_validation_record_overlap": overlap,
            "repeated_fixture": True,
        },
        "model_runs": runs,
        "decision_support": _decision_support(runs),
        "truth_boundary": {
            "universal_gradient_noise_scale_claimed": False,
            "noise_proxy_name": NOISE_PROXY_NAME,
            "noise_proxy_formula": NOISE_PROXY_FORMULA,
            "representative_large_corpus_claimed": False,
            "stage_freeze": False,
            "optimizer_freeze": False,
            "paid_compute_authority": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: Mapping[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("gradient stochasticity report schema/authority mismatch")
    source = report.get("source", {})
    if not _HEX40.fullmatch(str(source.get("git_sha", ""))):
        raise ValueError("invalid source SHA")
    if expected_source_sha is not None and source["git_sha"] != expected_source_sha:
        raise ValueError("source SHA mismatch")
    stored_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if stored_hash != _canonical_hash(unhashed):
        raise ValueError("report self-hash mismatch")
    if report.get("truth_boundary", {}).get("universal_gradient_noise_scale_claimed") is not False:
        raise ValueError("universal gradient noise scale claims are forbidden")
    expected = [spec.parameter_count() for spec in controlled_specs()]
    runs = report.get("model_runs", [])
    if [run.get("parameters") for run in runs] != expected:
        raise ValueError("controlled model family drifted")
    for run in runs:
        for point in run["checkpoints"]:
            if point["probe"]["non_mutation"].get("verified") is not True:
                raise ValueError("diagnostic probe did not prove non-mutation")
            ratio = point["probe"]["global"]["noise_to_signal_trace_ratio"]
            if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0:
                raise ValueError("invalid noise proxy")


__all__ = [
    "AUTHORITY",
    "CONFIG_SCHEMA",
    "NOISE_PROXY_FORMULA",
    "NOISE_PROXY_NAME",
    "SCHEMA",
    "estimate_gradient_stochasticity",
    "gradient_state_fingerprint",
    "load_experiment_config",
    "model_state_fingerprint",
    "optimizer_state_fingerprint",
    "run_gradient_noise_matrix",
    "state_fingerprint",
    "validate_report",
]
