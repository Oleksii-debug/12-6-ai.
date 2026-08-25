"""Optimizer-preserving gradient signal/variance diagnostics for batch-size research.

This module is deliberately a diagnostic extension of TRAIN-29 observability, not a
second training logger.  It computes local gradient statistics at a frozen parameter
state and restores all mutable training-side state before returning.

The reported ``noise_scale_microbatches_proxy`` is trace(covariance) / ||mean(g)||^2.
It is an empirical signal/variance proxy in units of the supplied base microbatch.
It is not presented as an exact critical batch size: that interpretation would need
stronger iid/stationarity assumptions than the project corpus provides.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from .loss import causal_lm_loss, causal_pair_loss
from .trainer import Trainer

SCHEMA_VERSION = "12-6.batch-noise-diagnostic.v1"


def _state_hash(value: Any) -> str:
    """Hash nested Python/Tensor state without depending on torch.save metadata."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(b"\0")
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            raw = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
            return
        if item is None:
            digest.update(b"none\0")
            return
        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _model_state_hash(model: nn.Module) -> str:
    return _state_hash(model.state_dict())


def _trainer_state_hash(trainer: Trainer) -> str:
    return _state_hash(asdict(trainer.state_dict()))


def _gradient_vector(model: nn.Module) -> Tensor:
    pieces: list[Tensor] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            pieces.append(parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    if not pieces:
        raise ValueError("model has no trainable parameters")
    return torch.cat(pieces)


def _batch_loss_and_tokens(model: nn.Module, batch: Mapping[str, Tensor], device: torch.device) -> tuple[Tensor, int]:
    if "input_ids" not in batch:
        raise KeyError("gradient probe batch must contain input_ids")
    input_ids = batch["input_ids"].to(device)
    if "labels" in batch and "target_ids" in batch:
        raise ValueError("gradient probe batch must not contain both labels and target_ids")
    logits = model(input_ids).logits
    if "target_ids" in batch:
        target_ids = batch["target_ids"].to(device)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        loss = causal_pair_loss(logits, target_ids, loss_mask=loss_mask)
        valid = target_ids.ne(-100)
        if loss_mask is not None:
            valid = valid & loss_mask.bool()
        tokens = int(valid.sum().item())
    else:
        labels = batch.get("labels", batch["input_ids"]).to(device)
        loss = causal_lm_loss(logits, labels)
        tokens = int(labels[:, 1:].ne(-100).sum().item())
    if tokens <= 0:
        raise ValueError("gradient probe batch has zero valid loss tokens")
    if not torch.isfinite(loss).item():
        raise FloatingPointError("gradient probe produced non-finite loss")
    return loss, tokens


def gradient_statistics(
    gradients: Sequence[Tensor],
    *,
    effective_microbatch_counts: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Compute signal/covariance proxies and empirical accumulated-gradient behavior.

    ``gradients`` must contain mean gradients for equal-semantics base microbatches at
    one fixed parameter state.  Grouped effective batches are formed from contiguous
    samples in the supplied deterministic sample order.
    """
    if len(gradients) < 2:
        raise ValueError("at least two gradient samples are required")
    vector_length = int(gradients[0].numel())
    if vector_length <= 0:
        raise ValueError("gradient vectors must be non-empty")
    normalized: list[Tensor] = []
    for gradient in gradients:
        flat = gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if int(flat.numel()) != vector_length:
            raise ValueError("all gradient samples must have equal length")
        if not torch.isfinite(flat).all().item():
            raise ValueError("gradient samples must be finite")
        normalized.append(flat)

    mean_gradient = torch.zeros(vector_length, dtype=torch.float64)
    for gradient in normalized:
        mean_gradient.add_(gradient)
    mean_gradient.div_(len(normalized))
    signal_squared = float(torch.dot(mean_gradient, mean_gradient).item())
    signal_norm = math.sqrt(max(signal_squared, 0.0))

    centered_sum = 0.0
    second_moment_sum = 0.0
    for gradient in normalized:
        second_moment_sum += float(torch.dot(gradient, gradient).item())
        delta = gradient - mean_gradient
        centered_sum += float(torch.dot(delta, delta).item())
    trace_covariance = centered_sum / (len(normalized) - 1)
    mean_gradient_second_moment = second_moment_sum / len(normalized)
    if signal_squared > 0.0:
        noise_scale_proxy: float | None = trace_covariance / signal_squared
        snr_proxy: float | None = signal_squared / trace_covariance if trace_covariance > 0.0 else math.inf
    else:
        noise_scale_proxy = None
        snr_proxy = None

    effective: list[dict[str, Any]] = []
    for raw_count in effective_microbatch_counts:
        count = int(raw_count)
        if count <= 0:
            raise ValueError("effective microbatch counts must be positive")
        groups = [
            normalized[start : start + count]
            for start in range(0, len(normalized), count)
            if len(normalized[start : start + count]) == count
        ]
        if not groups:
            continue
        group_means: list[Tensor] = []
        relative_deviations: list[float] = []
        cosines: list[float] = []
        for group in groups:
            group_mean = torch.zeros(vector_length, dtype=torch.float64)
            for gradient in group:
                group_mean.add_(gradient)
            group_mean.div_(count)
            group_means.append(group_mean)
            delta_norm = float(torch.linalg.vector_norm(group_mean - mean_gradient).item())
            relative_deviations.append(delta_norm / signal_norm if signal_norm > 0.0 else math.inf)
            group_norm = float(torch.linalg.vector_norm(group_mean).item())
            if group_norm > 0.0 and signal_norm > 0.0:
                cosine = float(torch.dot(group_mean, mean_gradient).item()) / (group_norm * signal_norm)
                cosines.append(max(-1.0, min(1.0, cosine)))

        empirical_group_trace_covariance: float | None = None
        if len(group_means) >= 2:
            group_center = torch.zeros(vector_length, dtype=torch.float64)
            for group_mean in group_means:
                group_center.add_(group_mean)
            group_center.div_(len(group_means))
            group_variance_sum = 0.0
            for group_mean in group_means:
                delta = group_mean - group_center
                group_variance_sum += float(torch.dot(delta, delta).item())
            empirical_group_trace_covariance = group_variance_sum / (len(group_means) - 1)

        effective.append(
            {
                "effective_microbatches": count,
                "groups": len(groups),
                "predicted_trace_covariance_under_iid": trace_covariance / count,
                "predicted_signal_to_noise_ratio": (
                    signal_squared / (trace_covariance / count)
                    if trace_covariance > 0.0
                    else math.inf
                ),
                "empirical_group_trace_covariance": empirical_group_trace_covariance,
                "mean_relative_deviation_from_all_sample_mean": (
                    sum(relative_deviations) / len(relative_deviations)
                ),
                "mean_cosine_to_all_sample_mean": (
                    sum(cosines) / len(cosines) if cosines else None
                ),
            }
        )

    return {
        "sample_count": len(normalized),
        "gradient_elements": vector_length,
        "signal_norm": signal_norm,
        "signal_squared": signal_squared,
        "mean_gradient_second_moment": mean_gradient_second_moment,
        "trace_covariance_unbiased": trace_covariance,
        "noise_scale_microbatches_proxy": noise_scale_proxy,
        "signal_to_noise_ratio_proxy": snr_proxy,
        "effective_batch_proxies": effective,
        "estimator_interpretation": (
            "trace(covariance)/||mean_gradient||^2 is a local empirical noise-scale proxy "
            "in units of the supplied base microbatch"
        ),
        "exact_critical_batch_size_claim": False,
    }


def diagnose_gradient_noise(
    model: nn.Module,
    trainer: Trainer,
    sampled_batches: Sequence[Mapping[str, Tensor]],
    *,
    effective_microbatch_counts: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Measure local gradient statistics while proving training state is unchanged."""
    if len(sampled_batches) < 2:
        raise ValueError("at least two sampled microbatches are required")
    trainer.assert_checkpoint_safe()
    device = trainer.device
    model_hash_before = _model_state_hash(model)
    trainer_hash_before = _trainer_state_hash(trainer)
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    saved_gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]

    gradients: list[Tensor] = []
    losses: list[float] = []
    token_counts: list[int] = []
    started = time.perf_counter()
    try:
        model.train()
        for batch in sampled_batches:
            model.zero_grad(set_to_none=True)
            loss, tokens = _batch_loss_and_tokens(model, batch, device)
            loss.backward()
            vector = _gradient_vector(model)
            if not torch.isfinite(vector).all().item():
                raise FloatingPointError("gradient probe produced non-finite gradient")
            gradients.append(vector)
            losses.append(float(loss.detach().cpu().item()))
            token_counts.append(tokens)
        statistics = gradient_statistics(
            gradients,
            effective_microbatch_counts=effective_microbatch_counts,
        )
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, saved_gradient in zip(model.parameters(), saved_gradients, strict=True):
            parameter.grad = None if saved_gradient is None else saved_gradient.to(parameter.device)
        model.train(was_training)
        random.setstate(python_rng_before)
        torch.set_rng_state(torch_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
    elapsed = time.perf_counter() - started

    model_hash_after = _model_state_hash(model)
    trainer_hash_after = _trainer_state_hash(trainer)
    if model_hash_after != model_hash_before:
        raise RuntimeError("diagnostic probe mutated model parameters/buffers")
    if trainer_hash_after != trainer_hash_before:
        raise RuntimeError("diagnostic probe mutated trainer/optimizer/scheduler state")
    rng_restored = torch.equal(torch.get_rng_state(), torch_rng_before) and random.getstate() == python_rng_before
    if cuda_rng_before is not None:
        current_cuda = torch.cuda.get_rng_state_all()
        rng_restored = rng_restored and len(current_cuda) == len(cuda_rng_before) and all(
            torch.equal(left, right) for left, right in zip(current_cuda, cuda_rng_before, strict=True)
        )
    if not rng_restored:
        raise RuntimeError("diagnostic probe failed to restore RNG state")

    gradient_bytes = int(statistics["gradient_elements"]) * 8
    return {
        "schema_version": SCHEMA_VERSION,
        "statistics": statistics,
        "sample_loss_mean": sum(losses) / len(losses),
        "sample_loss_min": min(losses),
        "sample_loss_max": max(losses),
        "sample_valid_loss_tokens_min": min(token_counts),
        "sample_valid_loss_tokens_max": max(token_counts),
        "sample_valid_loss_tokens_mean": sum(token_counts) / len(token_counts),
        "probe_wall_seconds": elapsed,
        "state_preservation": {
            "checkpoint_safe_before_probe": True,
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": model_hash_after,
            "model_state_unchanged": True,
            "trainer_state_sha256_before": trainer_hash_before,
            "trainer_state_sha256_after": trainer_hash_after,
            "optimizer_scheduler_counters_unchanged": True,
            "parameter_gradients_restored": True,
            "python_torch_cuda_rng_restored": True,
            "model_train_eval_mode_restored": True,
        },
        "memory_overhead": {
            "full_model_duplicate_retained": False,
            "gradient_vector_dtype": "float64_cpu",
            "bytes_per_gradient_vector": gradient_bytes,
            "retained_gradient_sample_bytes": gradient_bytes * len(gradients),
            "note": "gradient samples are released when the diagnostic result is returned",
        },
        "assumptions": {
            "gradients_are_local_to_one_fixed_parameter_state": True,
            "base_microbatch_draws_treated_as_exchangeable_for_proxy": True,
            "iid_data_assumption_proven": False,
            "stationarity_beyond_checkpoint_proven": False,
            "exact_critical_batch_size_claim": False,
        },
    }
