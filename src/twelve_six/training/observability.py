"""Public training observability surface.

TRAIN-52 layers corrected overhead accounting and streaming extrema on the bounded
update-magnitude observer while retaining one public TrainingObserver API. Private
versioned modules are compatibility snapshots of the incumbent implementation, not
parallel loggers.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace
from typing import Any

from torch.optim import Optimizer

from . import _observability_v2 as _v2
from ._observability_v2 import *  # noqa: F403 - preserve the incumbent public surface

SCHEMA_VERSION = _v2.SCHEMA_VERSION


class TrainingObserver(_v2.TrainingObserver):
    """TRAIN-52 observer with bounded detail plus all-probed streaming extrema."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending_snapshot_seconds = 0.0
        self._largest_parameter_float32_bytes = 0
        self._stream_global_ratio_min: float | None = None
        self._stream_global_ratio_min_step: int | None = None
        self._stream_global_ratio_max: float | None = None
        self._stream_global_ratio_max_step: int | None = None
        self._stream_global_max_update_magnitude = 0.0
        self._stream_global_max_update_magnitude_step: int | None = None
        self._stream_block_extrema: dict[str, dict[str, float | int]] = {}
        self._stream_update_alerts: list[dict[str, Any]] = []

    def _ensure_update_probe(self, trainer: Any) -> None:
        already_attached = self._attached_optimizer is not None
        super()._ensure_update_probe(trainer)
        if not already_attached and self._parameter_refs:
            self._largest_parameter_float32_bytes = max(
                parameter.numel() * 4 for _, parameter in self._parameter_refs
            )

    def _optimizer_step_pre_hook(self, optimizer: Optimizer) -> None:
        started = time.perf_counter()
        super()._optimizer_step_pre_hook(optimizer)
        if self._pending_update_snapshot is not None:
            self._pending_snapshot_seconds = time.perf_counter() - started

    def _optimizer_step_post_hook(self, optimizer: Optimizer) -> None:
        if self._pending_update_snapshot is None:
            return super()._optimizer_step_post_hook(optimizer)
        pending_step = self._pending_update_step
        snapshot_seconds = self._pending_snapshot_seconds
        previous_total = self._update_probe_seconds
        post_started = time.perf_counter()
        super()._optimizer_step_post_hook(optimizer)
        post_seconds = time.perf_counter() - post_started
        incorrectly_spanned_seconds = self._update_probe_seconds - previous_total
        corrected_probe_seconds = snapshot_seconds + post_seconds
        self._update_probe_seconds += corrected_probe_seconds - incorrectly_spanned_seconds
        if pending_step is not None:
            for index, sample in enumerate(self._update_samples):
                if sample.optimizer_step == pending_step:
                    self._update_samples[index] = replace(
                        sample,
                        probe_seconds=corrected_probe_seconds,
                    )
                    break

    def _discard_pending_update_snapshot(self) -> None:
        super()._discard_pending_update_snapshot()
        self._pending_snapshot_seconds = 0.0

    def _append_stream_alert(self, record: dict[str, Any]) -> None:
        if len(self._stream_update_alerts) < self.max_update_alerts:
            self._stream_update_alerts.append(record)

    def _observe_update_extrema(self, observation: _v2.UpdateObservation) -> None:
        step = observation.optimizer_step
        global_metrics = observation.global_metrics
        global_ratio = global_metrics.update_weight_ratio
        if global_ratio is not None:
            if self._stream_global_ratio_min is None or global_ratio < self._stream_global_ratio_min:
                self._stream_global_ratio_min = global_ratio
                self._stream_global_ratio_min_step = step
            if self._stream_global_ratio_max is None or global_ratio > self._stream_global_ratio_max:
                self._stream_global_ratio_max = global_ratio
                self._stream_global_ratio_max_step = step
            if global_ratio >= 0.10:
                self._append_stream_alert(
                    {
                        "optimizer_step": step,
                        "scope": "global",
                        "update_weight_ratio": global_ratio,
                        "reason": "GLOBAL_RATIO_GE_0_10",
                    }
                )
        if global_metrics.max_update_magnitude > self._stream_global_max_update_magnitude:
            self._stream_global_max_update_magnitude = global_metrics.max_update_magnitude
            self._stream_global_max_update_magnitude_step = step

        positive_block_ratios = [
            metrics.update_weight_ratio
            for metrics in observation.per_block.values()
            if metrics.update_weight_ratio is not None and metrics.update_weight_ratio > 0.0
        ]
        median_block_ratio = (
            statistics.median(positive_block_ratios) if positive_block_ratios else None
        )
        for block, metrics in observation.per_block.items():
            ratio = metrics.update_weight_ratio
            extrema = self._stream_block_extrema.setdefault(
                block,
                {
                    "max_update_weight_ratio": -1.0,
                    "max_update_weight_ratio_optimizer_step": 0,
                    "max_update_magnitude": 0.0,
                    "max_update_magnitude_optimizer_step": 0,
                },
            )
            if ratio is not None and ratio > float(extrema["max_update_weight_ratio"]):
                extrema["max_update_weight_ratio"] = ratio
                extrema["max_update_weight_ratio_optimizer_step"] = step
            if metrics.max_update_magnitude > float(extrema["max_update_magnitude"]):
                extrema["max_update_magnitude"] = metrics.max_update_magnitude
                extrema["max_update_magnitude_optimizer_step"] = step

            if ratio is None:
                continue
            absolute_outlier = ratio >= 0.10
            relative_outlier = (
                median_block_ratio is not None
                and median_block_ratio > 0.0
                and ratio >= 8.0 * median_block_ratio
            )
            if absolute_outlier or relative_outlier:
                reasons: list[str] = []
                if absolute_outlier:
                    reasons.append("BLOCK_RATIO_GE_0_10")
                if relative_outlier:
                    reasons.append("BLOCK_RATIO_GE_8X_STEP_MEDIAN")
                self._append_stream_alert(
                    {
                        "optimizer_step": step,
                        "scope": block,
                        "update_weight_ratio": ratio,
                        "step_block_median_ratio": median_block_ratio,
                        "reason": "+".join(reasons),
                    }
                )

    def _retain_update_sample(self, observation: _v2.UpdateObservation) -> None:
        self._observe_update_extrema(observation)
        super()._retain_update_sample(observation)

    def _update_ratio_summary(self) -> dict[str, Any]:
        summary = super()._update_ratio_summary()
        summary["global_update_weight_ratio_min"] = self._stream_global_ratio_min
        summary["global_update_weight_ratio_min_optimizer_step"] = self._stream_global_ratio_min_step
        summary["global_update_weight_ratio_max"] = self._stream_global_ratio_max
        summary["global_update_weight_ratio_max_optimizer_step"] = self._stream_global_ratio_max_step
        summary["global_update_weight_ratio_median_scope"] = "RETAINED_DETAIL_SAMPLES_ONLY"
        summary["global_max_update_magnitude"] = self._stream_global_max_update_magnitude
        summary["global_max_update_magnitude_optimizer_step"] = (
            self._stream_global_max_update_magnitude_step
        )
        summary["per_block_extrema_all_probed_updates"] = {
            block: dict(values) for block, values in sorted(self._stream_block_extrema.items())
        }
        summary["extrema_coverage"] = "ALL_PROBED_UPDATES_BEFORE_DETAIL_RETENTION_THINNING"
        pathology = summary["pathology_candidates"]
        pathology["status"] = "CANDIDATES" if self._stream_update_alerts else "NO_CANDIDATES"
        pathology["records"] = list(self._stream_update_alerts)
        pathology["coverage"] = "ALL_PROBED_UPDATES_BEFORE_DETAIL_RETENTION_THINNING"

        overhead = summary["overhead"]
        snapshot = int(overhead["temporary_snapshot_peak_bytes"])
        overhead["conservative_peak_extra_tensor_bytes_upper_bound"] = (
            snapshot + 4 * self._largest_parameter_float32_bytes if snapshot else 0
        )
        overhead["workspace_bound_semantics"] = (
            "SNAPSHOT_PLUS_FOUR_LARGEST_PARAMETER_FP32_BUFFERS; "
            "EXCLUDES_FRAMEWORK_ALLOCATOR_METADATA"
        )
        overhead["probe_seconds_semantics"] = (
            "PRE_STEP_SNAPSHOT_COPY_PLUS_POST_STEP_REDUCTION_ONLY; OPTIMIZER_STEP_EXCLUDED"
        )
        return summary


__all__ = list(_v2.__all__)
