"""Training observability with bounded relative-update diagnostics.

The public :class:`TrainingObserver` extends the incumbent TRAIN-29 observer rather
than introducing a second logging path.  Update telemetry is sampled around the
actual ``Optimizer.step`` call through PyTorch optimizer hooks.  The temporary
pre-update parameter snapshot exists only for a sampled optimizer step and is
released immediately after the post-step reduction (or on a failed transition).

Update diagnostics are telemetry only.  They are absent from TrainerState,
checkpoint payloads, model identities, and run-identity hashing.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from . import _observability_v1 as _v1
from ._observability_v1 import (
    MemorySnapshot,
    PhaseTimings,
    RankMetadata,
    RegionObservation,
    StepObservation,
    aggregate_rank_summaries,
    gather_distributed_summary,
    paid_compute_decision_support,
)

SCHEMA_VERSION = "12-6.training-observability.v2"


@dataclass(frozen=True, slots=True)
class UpdateMagnitude:
    """L2-relative magnitude for one parameter population before an update."""

    parameter_tensors: int
    parameter_elements: int
    parameter_norm: float
    update_norm: float
    update_weight_ratio: float | None
    max_update_magnitude: float


@dataclass(frozen=True, slots=True)
class UpdateObservation:
    """One sampled committed optimizer update."""

    optimizer_step: int
    global_metrics: UpdateMagnitude
    per_block: dict[str, UpdateMagnitude]
    temporary_snapshot_bytes: int
    largest_parameter_bytes: int
    probe_seconds: float


def _block_name(parameter_name: str) -> str | None:
    parts = parameter_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
        return f"blocks.{int(parts[1])}"
    return None


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "parameter_tensors": 0,
        "parameter_elements": 0,
        "parameter_sq": 0.0,
        "update_sq": 0.0,
        "max_update_magnitude": 0.0,
    }


def _accumulate(
    accumulator: dict[str, float | int],
    *,
    parameter_elements: int,
    parameter_sq: float,
    update_sq: float,
    max_update_magnitude: float,
) -> None:
    accumulator["parameter_tensors"] = int(accumulator["parameter_tensors"]) + 1
    accumulator["parameter_elements"] = int(accumulator["parameter_elements"]) + int(
        parameter_elements
    )
    accumulator["parameter_sq"] = float(accumulator["parameter_sq"]) + float(parameter_sq)
    accumulator["update_sq"] = float(accumulator["update_sq"]) + float(update_sq)
    accumulator["max_update_magnitude"] = max(
        float(accumulator["max_update_magnitude"]),
        float(max_update_magnitude),
    )


def _finish(accumulator: Mapping[str, float | int]) -> UpdateMagnitude:
    parameter_norm = math.sqrt(max(float(accumulator["parameter_sq"]), 0.0))
    update_norm = math.sqrt(max(float(accumulator["update_sq"]), 0.0))
    return UpdateMagnitude(
        parameter_tensors=int(accumulator["parameter_tensors"]),
        parameter_elements=int(accumulator["parameter_elements"]),
        parameter_norm=parameter_norm,
        update_norm=update_norm,
        update_weight_ratio=_ratio(update_norm, parameter_norm),
        max_update_magnitude=float(accumulator["max_update_magnitude"]),
    )


def summarize_parameter_update(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
) -> tuple[UpdateMagnitude, dict[str, UpdateMagnitude]]:
    """Numerically summarize one explicit pre/post parameter mapping.

    This pure helper is used by the optimizer-hook path and gives tests a small,
    deterministic numerical oracle.  Parameter names are deduplicated by the caller;
    ``nn.Module.named_parameters()`` already does this for tied weights by default.
    """
    if set(before) != set(after):
        raise ValueError("before/after parameter names must match exactly")
    global_accumulator = _empty_accumulator()
    block_accumulators: dict[str, dict[str, float | int]] = {}
    for name in sorted(before):
        prior = before[name]
        current = after[name]
        if prior.shape != current.shape:
            raise ValueError(f"parameter shape changed across update: {name}")
        if prior.device != current.device:
            raise ValueError(f"parameter device changed across update: {name}")
        prior_float = prior.detach().float()
        current_float = current.detach().float()
        delta = current_float - prior_float
        parameter_sq = float(torch.sum(prior_float * prior_float).item())
        update_sq = float(torch.sum(delta * delta).item())
        maximum = (
            float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
        )
        values = {
            "parameter_elements": prior.numel(),
            "parameter_sq": parameter_sq,
            "update_sq": update_sq,
            "max_update_magnitude": maximum,
        }
        _accumulate(global_accumulator, **values)
        block = _block_name(name)
        if block is not None:
            _accumulate(block_accumulators.setdefault(block, _empty_accumulator()), **values)
    return _finish(global_accumulator), {
        block: _finish(values) for block, values in sorted(block_accumulators.items())
    }


class TrainingObserver(_v1.TrainingObserver):
    """Incumbent observer plus bounded optimizer-update magnitude telemetry."""

    def __init__(
        self,
        run_identity: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        rank: RankMetadata | None = None,
        max_step_samples: int = 4096,
        gpu_sample_every_steps: int = 10,
        synchronize_cuda_steps: bool = False,
        enable_update_magnitude: bool = True,
        update_sample_every_steps: int = 1,
        max_update_samples: int = 256,
        max_update_alerts: int = 32,
    ) -> None:
        super().__init__(
            run_identity,
            device=device,
            rank=rank,
            max_step_samples=max_step_samples,
            gpu_sample_every_steps=gpu_sample_every_steps,
            synchronize_cuda_steps=synchronize_cuda_steps,
        )
        if update_sample_every_steps <= 0:
            raise ValueError("update_sample_every_steps must be positive")
        if max_update_samples <= 0:
            raise ValueError("max_update_samples must be positive")
        if max_update_alerts <= 0:
            raise ValueError("max_update_alerts must be positive")
        self.enable_update_magnitude = bool(enable_update_magnitude)
        self.update_sample_every_steps = int(update_sample_every_steps)
        self.max_update_samples = int(max_update_samples)
        self.max_update_alerts = int(max_update_alerts)

        self._update_samples: list[UpdateObservation] = []
        self._update_sample_stride = 1
        self._update_probe_samples = 0
        self._update_probe_seconds = 0.0
        self._update_snapshot_peak_bytes = 0
        self._largest_parameter_bytes = 0
        self._model_trainable_parameter_bytes = 0
        self._pending_update_snapshot: dict[str, Tensor] | None = None
        self._pending_update_step: int | None = None
        self._pending_snapshot_bytes = 0
        self._pending_probe_started: float | None = None
        self._parameter_refs: tuple[tuple[str, nn.Parameter], ...] = ()
        self._attached_optimizer: Optimizer | None = None
        self._attached_trainer_ref: weakref.ReferenceType[Any] | None = None
        self._optimizer_hook_handles: list[Any] = []

    @property
    def update_samples(self) -> tuple[UpdateObservation, ...]:
        return tuple(self._update_samples)

    def _sync_update_probe_if_requested(self) -> None:
        if self.synchronize_cuda_steps:
            self._synchronize_cuda()

    def _ensure_update_probe(self, trainer: Any) -> None:
        if not self.enable_update_magnitude:
            return
        optimizer = getattr(trainer, "optimizer", None)
        model = getattr(trainer, "model", None)
        if not isinstance(optimizer, Optimizer) or not isinstance(model, nn.Module):
            raise TypeError("update magnitude telemetry requires Trainer-like model and optimizer")
        if self._attached_optimizer is optimizer:
            return
        if self._attached_optimizer is not None:
            raise RuntimeError("one TrainingObserver cannot attach to multiple optimizers")

        refs = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not refs:
            raise ValueError("update magnitude telemetry requires trainable parameters")
        parameter_ids = [id(parameter) for _, parameter in refs]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("named_parameters unexpectedly returned duplicate parameters")
        self._parameter_refs = refs
        self._model_trainable_parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for _, parameter in refs
        )
        self._largest_parameter_bytes = max(
            parameter.numel() * parameter.element_size() for _, parameter in refs
        )
        self._attached_optimizer = optimizer
        self._attached_trainer_ref = weakref.ref(trainer)
        observer_ref = weakref.ref(self)

        def pre_hook(opt: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            observer = observer_ref()
            if observer is not None:
                observer._optimizer_step_pre_hook(opt)

        def post_hook(opt: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            observer = observer_ref()
            if observer is not None:
                observer._optimizer_step_post_hook(opt)

        self._optimizer_hook_handles = [
            optimizer.register_step_pre_hook(pre_hook),
            optimizer.register_step_post_hook(post_hook),
        ]

    def detach_update_probe(self) -> None:
        self._discard_pending_update_snapshot()
        for handle in self._optimizer_hook_handles:
            handle.remove()
        self._optimizer_hook_handles.clear()
        self._attached_optimizer = None
        self._attached_trainer_ref = None
        self._parameter_refs = ()

    def _should_sample_update(self, optimizer_step: int) -> bool:
        cadence = self.update_sample_every_steps * self._update_sample_stride
        return optimizer_step > 0 and optimizer_step % cadence == 0

    def _optimizer_step_pre_hook(self, optimizer: Optimizer) -> None:
        if optimizer is not self._attached_optimizer:
            raise RuntimeError("update probe optimizer identity drift")
        if self._pending_update_snapshot is not None:
            raise RuntimeError("update probe already holds a pending snapshot")
        trainer = self._attached_trainer_ref() if self._attached_trainer_ref is not None else None
        if trainer is None:
            return
        optimizer_step = int(getattr(trainer, "optimizer_step")) + 1
        if not self._should_sample_update(optimizer_step):
            return

        self._sync_update_probe_if_requested()
        started = time.perf_counter()
        snapshot: dict[str, Tensor] = {}
        snapshot_bytes = 0
        for name, parameter in self._parameter_refs:
            copied = parameter.detach().clone(memory_format=torch.preserve_format)
            snapshot[name] = copied
            snapshot_bytes += copied.numel() * copied.element_size()
        self._sync_update_probe_if_requested()
        self._pending_update_snapshot = snapshot
        self._pending_update_step = optimizer_step
        self._pending_snapshot_bytes = snapshot_bytes
        self._pending_probe_started = started
        self._update_snapshot_peak_bytes = max(self._update_snapshot_peak_bytes, snapshot_bytes)

    def _optimizer_step_post_hook(self, optimizer: Optimizer) -> None:
        if optimizer is not self._attached_optimizer:
            raise RuntimeError("update probe optimizer identity drift")
        if self._pending_update_snapshot is None:
            return
        snapshot = self._pending_update_snapshot
        optimizer_step = self._pending_update_step
        snapshot_bytes = self._pending_snapshot_bytes
        started = self._pending_probe_started
        try:
            self._sync_update_probe_if_requested()
            after = {name: parameter.detach() for name, parameter in self._parameter_refs}
            global_metrics, per_block = summarize_parameter_update(snapshot, after)
            self._sync_update_probe_if_requested()
            probe_seconds = time.perf_counter() - (started if started is not None else time.perf_counter())
            if optimizer_step is None:
                raise RuntimeError("update probe lost pending optimizer step")
            observation = UpdateObservation(
                optimizer_step=optimizer_step,
                global_metrics=global_metrics,
                per_block=per_block,
                temporary_snapshot_bytes=snapshot_bytes,
                largest_parameter_bytes=self._largest_parameter_bytes,
                probe_seconds=probe_seconds,
            )
            self._update_probe_samples += 1
            self._update_probe_seconds += probe_seconds
            self._retain_update_sample(observation)
        finally:
            self._discard_pending_update_snapshot()

    def _discard_pending_update_snapshot(self) -> None:
        self._pending_update_snapshot = None
        self._pending_update_step = None
        self._pending_snapshot_bytes = 0
        self._pending_probe_started = None

    def _retain_update_sample(self, observation: UpdateObservation) -> None:
        self._update_samples.append(observation)
        if len(self._update_samples) <= self.max_update_samples:
            return
        self._update_samples = self._update_samples[::2]
        self._update_sample_stride *= 2

    def train_microbatch(
        self,
        trainer: Any,
        batch: Mapping[str, torch.Tensor],
        *,
        data_wait_seconds: float,
        phases: PhaseTimings | None = None,
    ):
        self._ensure_update_probe(trainer)
        try:
            return super().train_microbatch(
                trainer,
                batch,
                data_wait_seconds=data_wait_seconds,
                phases=phases,
            )
        except Exception:
            self._discard_pending_update_snapshot()
            raise

    def _update_ratio_summary(self) -> dict[str, Any]:
        ratios = [
            sample.global_metrics.update_weight_ratio
            for sample in self._update_samples
            if sample.global_metrics.update_weight_ratio is not None
        ]
        block_maxima: dict[str, dict[str, float | int]] = {}
        alerts: list[dict[str, Any]] = []
        for sample in self._update_samples:
            block_ratios = [
                metrics.update_weight_ratio
                for metrics in sample.per_block.values()
                if metrics.update_weight_ratio is not None
            ]
            positive_block_ratios = [value for value in block_ratios if value is not None and value > 0.0]
            median_block_ratio = (
                statistics.median(positive_block_ratios) if positive_block_ratios else None
            )
            global_ratio = sample.global_metrics.update_weight_ratio
            if global_ratio is not None and global_ratio >= 0.10 and len(alerts) < self.max_update_alerts:
                alerts.append(
                    {
                        "optimizer_step": sample.optimizer_step,
                        "scope": "global",
                        "update_weight_ratio": global_ratio,
                        "reason": "GLOBAL_RATIO_GE_0_10",
                    }
                )
            for block, metrics in sample.per_block.items():
                ratio = metrics.update_weight_ratio
                if ratio is None:
                    continue
                previous = block_maxima.get(block)
                if previous is None or ratio > float(previous["update_weight_ratio"]):
                    block_maxima[block] = {
                        "optimizer_step": sample.optimizer_step,
                        "update_weight_ratio": ratio,
                        "max_update_magnitude": metrics.max_update_magnitude,
                    }
                relative_outlier = (
                    median_block_ratio is not None
                    and median_block_ratio > 0.0
                    and ratio >= 8.0 * median_block_ratio
                )
                absolute_outlier = ratio >= 0.10
                if (relative_outlier or absolute_outlier) and len(alerts) < self.max_update_alerts:
                    reason = []
                    if absolute_outlier:
                        reason.append("BLOCK_RATIO_GE_0_10")
                    if relative_outlier:
                        reason.append("BLOCK_RATIO_GE_8X_STEP_MEDIAN")
                    alerts.append(
                        {
                            "optimizer_step": sample.optimizer_step,
                            "scope": block,
                            "update_weight_ratio": ratio,
                            "step_block_median_ratio": median_block_ratio,
                            "reason": "+".join(reason),
                        }
                    )

        step_seconds = float(getattr(self, "_step_seconds", 0.0))
        conservative_extra = (
            self._update_snapshot_peak_bytes + self._largest_parameter_bytes
            if self._update_snapshot_peak_bytes
            else 0
        )
        return {
            "enabled": self.enable_update_magnitude,
            "status": (
                "MEASURED" if self._update_probe_samples else
                "ENABLED_NOT_YET_SAMPLED" if self.enable_update_magnitude else
                "DISABLED"
            ),
            "sample_every_optimizer_steps_configured": self.update_sample_every_steps,
            "retention_stride": self._update_sample_stride,
            "effective_future_sample_every_optimizer_steps": (
                self.update_sample_every_steps * self._update_sample_stride
            ),
            "probe_samples_total": self._update_probe_samples,
            "retained_samples": len(self._update_samples),
            "retained_samples_limit": self.max_update_samples,
            "global_update_weight_ratio_min": min(ratios) if ratios else None,
            "global_update_weight_ratio_median": statistics.median(ratios) if ratios else None,
            "global_update_weight_ratio_max": max(ratios) if ratios else None,
            "per_block_maxima": block_maxima,
            "pathology_candidates": {
                "status": "CANDIDATES" if alerts else "NO_CANDIDATES",
                "criteria": {
                    "absolute_update_weight_ratio_ge": 0.10,
                    "per_block_vs_step_median_multiplier_ge": 8.0,
                    "interpretation": "HEURISTIC_ALERT_NOT_AUTOMATIC_TRAINING_FAILURE",
                },
                "records": alerts,
                "record_limit": self.max_update_alerts,
            },
            "overhead": {
                "model_trainable_parameter_storage_bytes": self._model_trainable_parameter_bytes,
                "temporary_snapshot_peak_bytes": self._update_snapshot_peak_bytes,
                "largest_single_parameter_bytes": self._largest_parameter_bytes,
                "conservative_peak_extra_tensor_bytes_upper_bound": conservative_extra,
                "snapshot_lifetime": "SAMPLED_OPTIMIZER_STEP_ONLY",
                "snapshot_device": str(self.device),
                "probe_seconds_total": self._update_probe_seconds,
                "probe_fraction_of_observed_step_seconds": (
                    self._update_probe_seconds / step_seconds if step_seconds > 0.0 else None
                ),
                "timing_mode": (
                    "CUDA_SYNCHRONIZED_WALL"
                    if self.device.type == "cuda" and self.synchronize_cuda_steps
                    else "HOST_WALL_INCLUDES_CPU_WORK_AND_DEVICE_ENQUEUE"
                ),
                "allocator_or_reduction_workspace_in_upper_bound": False,
            },
            "semantics": {
                "parameter_norm_reference": "PRE_UPDATE_WEIGHTS",
                "update": "POST_UPDATE_MINUS_PRE_UPDATE",
                "update_norm": "GLOBAL_L2_OVER_UNIQUE_TRAINABLE_PARAMETERS",
                "per_block_grouping": "PARAMETER_NAME_PREFIX_blocks.INDEX",
                "tied_parameter_double_counted": False,
                "telemetry_is_training_or_checkpoint_state": False,
            },
        }

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["schema_version"] = SCHEMA_VERSION
        summary["update_magnitude"] = self._update_ratio_summary()
        return summary

    def export(self) -> dict[str, Any]:
        payload = super().export()
        payload["schema_version"] = SCHEMA_VERSION
        payload["update_samples"] = [asdict(item) for item in self._update_samples]
        payload["summary"] = self.summary()
        return payload

    def write_jsonl(self, path: str | Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "record_type": "run_identity",
                "run_identity": self.run_identity,
                "run_identity_sha256": self.run_identity_sha256,
            }
        ]
        records.extend(
            {
                "record_type": "step",
                "run_identity_sha256": self.run_identity_sha256,
                **asdict(item),
            }
            for item in self.step_samples
        )
        records.extend(
            {
                "record_type": "update_magnitude",
                "run_identity_sha256": self.run_identity_sha256,
                **asdict(item),
            }
            for item in self._update_samples
        )
        records.extend(
            {
                "record_type": "region",
                "run_identity_sha256": self.run_identity_sha256,
                **asdict(item),
            }
            for item in self.regions
        )
        records.append(
            {
                "record_type": "summary",
                "run_identity_sha256": self.run_identity_sha256,
                "summary": self.summary(),
            }
        )
        Path(path).write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )


__all__ = [
    "MemorySnapshot",
    "PhaseTimings",
    "RankMetadata",
    "RegionObservation",
    "SCHEMA_VERSION",
    "StepObservation",
    "TrainingObserver",
    "UpdateMagnitude",
    "UpdateObservation",
    "aggregate_rank_summaries",
    "gather_distributed_summary",
    "paid_compute_decision_support",
    "summarize_parameter_update",
]
