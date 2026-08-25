"""Public training observability surface.

TRAIN-52 layers the corrected overhead accounting on the bounded update-magnitude
observer while retaining one public TrainingObserver API.  Private versioned modules
are compatibility snapshots of the incumbent implementation, not parallel loggers.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from torch.optim import Optimizer

from . import _observability_v2 as _v2
from ._observability_v2 import *  # noqa: F403 - preserve the incumbent public surface

SCHEMA_VERSION = _v2.SCHEMA_VERSION


class TrainingObserver(_v2.TrainingObserver):
    """TRAIN-52 observer with probe-only timing and conservative workspace bounds."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending_snapshot_seconds = 0.0
        self._largest_parameter_float32_bytes = 0

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

    def _update_ratio_summary(self) -> dict[str, Any]:
        summary = super()._update_ratio_summary()
        overhead = summary["overhead"]
        snapshot = int(overhead["temporary_snapshot_peak_bytes"])
        overhead["conservative_peak_extra_tensor_bytes_upper_bound"] = (
            snapshot + 3 * self._largest_parameter_float32_bytes if snapshot else 0
        )
        overhead["workspace_bound_semantics"] = (
            "SNAPSHOT_PLUS_THREE_LARGEST_PARAMETER_FP32_BUFFERS; "
            "EXCLUDES_FRAMEWORK_ALLOCATOR_METADATA"
        )
        overhead["probe_seconds_semantics"] = (
            "PRE_STEP_SNAPSHOT_COPY_PLUS_POST_STEP_REDUCTION_ONLY; OPTIMIZER_STEP_EXCLUDED"
        )
        return summary


__all__ = list(_v2.__all__)
