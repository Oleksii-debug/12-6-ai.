"""Low-overhead structured observability for 12-6 training.

Telemetry is deliberately separated from deterministic training/checkpoint state.
The caller supplies an exact, timing-free run identity; this module hashes that
identity once and records non-deterministic observations beside it.

The current D02 Trainer exposes a trustworthy whole-microbatch seam but no public
forward/backward/update timing seams. ``TrainingObserver.train_microbatch`` therefore
measures the whole transition without copying Trainer semantics. Backends that own
cleaner seams may supply ``PhaseTimings`` when recording a step.
"""

from __future__ import annotations

import copy
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import torch

from twelve_six.checkpoint import hash_json

from .trainer import StepMetrics

SCHEMA_VERSION = "12-6.training-observability.v1"
_T = TypeVar("_T")
_NONDETERMINISTIC_IDENTITY_KEYS = (
    "duration",
    "elapsed",
    "memory_peak",
    "seconds",
    "throughput",
    "timestamp",
    "tokens_per_second",
    "utilization_percent",
    "wall_time",
)


@dataclass(frozen=True, slots=True)
class RankMetadata:
    rank: int
    local_rank: int
    world_size: int
    distributed_initialized: bool
    backend: str | None

    @classmethod
    def detect(cls) -> RankMetadata:
        """Read rank/world metadata without initializing a process group."""
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            rank = int(distributed.get_rank())
            world_size = int(distributed.get_world_size())
            local_rank = _environment_int("LOCAL_RANK", default=0)
            return cls(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                distributed_initialized=True,
                backend=str(distributed.get_backend()),
            )

        world_size = _environment_int("WORLD_SIZE", default=1)
        rank = _environment_int("RANK", default=0)
        local_rank = _environment_int("LOCAL_RANK", default=0)
        if world_size <= 0:
            raise ValueError("WORLD_SIZE must be positive")
        if rank >= world_size:
            raise ValueError(f"RANK={rank} must be < WORLD_SIZE={world_size}")
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed_initialized=False,
            backend=None,
        )


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    """Optional backend-supplied phase timings for a single training transition."""

    forward_seconds: float | None = None
    backward_seconds: float | None = None
    update_seconds: float | None = None

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value is not None:
                _require_nonnegative_finite(value, name)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    process_rss_peak_bytes: int | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None


@dataclass(frozen=True, slots=True)
class StepObservation:
    micro_step: int
    optimizer_step: int
    optimizer_stepped: bool
    tokens: int
    loss: float
    update_loss: float | None
    learning_rate: float
    grad_norm: float | None
    data_wait_seconds: float
    step_seconds: float
    train_tokens_per_second: float | None
    compute_tokens_per_second: float | None
    forward_seconds: float | None
    backward_seconds: float | None
    update_seconds: float | None
    memory: MemorySnapshot
    gpu_utilization_percent: float | None


@dataclass(frozen=True, slots=True)
class RegionObservation:
    kind: str
    operation: str
    seconds: float
    optimizer_step: int | None
    tokens_seen: int | None
    status: str


def _environment_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_nonnegative_finite(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _assert_timing_free_identity(value: Any, *, path: str = "identity") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if any(token in normalized for token in _NONDETERMINISTIC_IDENTITY_KEYS):
                raise ValueError(
                    f"non-deterministic telemetry field {path}.{key} must not enter run identity"
                )
            _assert_timing_free_identity(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_timing_free_identity(child, path=f"{path}[{index}]")


def _process_rss_peak_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if peak < 0:
        return None
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def _cuda_device(device: torch.device) -> torch.device | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _memory_snapshot(device: torch.device) -> MemorySnapshot:
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return MemorySnapshot(
            process_rss_peak_bytes=_process_rss_peak_bytes(),
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
        )
    return MemorySnapshot(
        process_rss_peak_bytes=_process_rss_peak_bytes(),
        cuda_allocated_bytes=int(torch.cuda.memory_allocated(cuda_device)),
        cuda_reserved_bytes=int(torch.cuda.memory_reserved(cuda_device)),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(cuda_device)),
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(cuda_device)),
    )


def _sample_gpu_utilization(device: torch.device) -> tuple[float | None, str]:
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return None, "NOT_CUDA"
    utilization = getattr(torch.cuda, "utilization", None)
    if not callable(utilization):
        return None, "UNAVAILABLE_API"
    try:
        value = float(utilization(cuda_device))
    except Exception:  # noqa: BLE001 - optional provider failures must not stop telemetry
        return None, "UNAVAILABLE_RUNTIME"
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        return None, "UNAVAILABLE_INVALID_SAMPLE"
    return value, "AVAILABLE_TORCH_CUDA_UTILIZATION"


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


class TrainingObserver:
    """Bounded in-memory telemetry collector for one exact training run identity."""

    def __init__(
        self,
        run_identity: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        rank: RankMetadata | None = None,
        max_step_samples: int = 4096,
        gpu_sample_every_steps: int = 10,
        synchronize_cuda_steps: bool = False,
    ) -> None:
        if not isinstance(run_identity, Mapping) or not run_identity:
            raise ValueError("run_identity must be a non-empty mapping")
        if max_step_samples <= 0:
            raise ValueError("max_step_samples must be positive")
        if gpu_sample_every_steps <= 0:
            raise ValueError("gpu_sample_every_steps must be positive")
        _assert_timing_free_identity(run_identity)
        self._run_identity = copy.deepcopy(dict(run_identity))
        self.run_identity_sha256 = hash_json(self._run_identity)
        self.device = torch.device(device)
        self.rank = rank or RankMetadata.detect()
        self.max_step_samples = int(max_step_samples)
        self.gpu_sample_every_steps = int(gpu_sample_every_steps)
        self.synchronize_cuda_steps = bool(synchronize_cuda_steps)
        self.started_at_utc = datetime.now(UTC).isoformat()

        self._step_samples: list[StepObservation] = []
        self._regions: list[RegionObservation] = []
        self._sample_stride = 1
        self._observed_steps = 0
        self._optimizer_steps = 0
        self._tokens = 0
        self._data_wait_seconds = 0.0
        self._step_seconds = 0.0
        self._phase_totals = {"forward": 0.0, "backward": 0.0, "update": 0.0}
        self._phase_counts = {"forward": 0, "backward": 0, "update": 0}
        self._loss_first: float | None = None
        self._loss_final: float | None = None
        self._loss_min: float | None = None
        self._loss_max: float | None = None
        self._lr_first: float | None = None
        self._lr_final: float | None = None
        self._grad_norm_min: float | None = None
        self._grad_norm_max: float | None = None
        self._memory_peak = MemorySnapshot(None, None, None, None, None)
        self._gpu_samples: list[float] = []
        self._gpu_utilization_status = "NOT_SAMPLED"

    @property
    def run_identity(self) -> dict[str, Any]:
        return copy.deepcopy(self._run_identity)

    @property
    def step_samples(self) -> tuple[StepObservation, ...]:
        return tuple(self._step_samples)

    @property
    def regions(self) -> tuple[RegionObservation, ...]:
        return tuple(self._regions)

    def _synchronize_cuda(self) -> None:
        cuda_device = _cuda_device(self.device)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)

    def measure_next(self, iterator: Any) -> tuple[Any, float]:
        """Measure blocking data-fetch latency without owning dataset semantics."""
        started = time.perf_counter()
        item = next(iterator)
        return item, time.perf_counter() - started

    def train_microbatch(
        self,
        trainer: Any,
        batch: Mapping[str, torch.Tensor],
        *,
        data_wait_seconds: float,
        phases: PhaseTimings | None = None,
    ) -> StepMetrics:
        """Measure the existing Trainer transition without duplicating its semantics."""
        _require_nonnegative_finite(data_wait_seconds, "data_wait_seconds")
        if self.synchronize_cuda_steps:
            self._synchronize_cuda()
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        if self.synchronize_cuda_steps:
            self._synchronize_cuda()
        step_seconds = time.perf_counter() - started
        if not isinstance(metrics, StepMetrics):
            raise TypeError("trainer.train_microbatch() must return StepMetrics")
        self.record_step(
            metrics,
            data_wait_seconds=data_wait_seconds,
            step_seconds=step_seconds,
            phases=phases,
        )
        return metrics

    def record_step(
        self,
        metrics: StepMetrics,
        *,
        data_wait_seconds: float,
        step_seconds: float,
        phases: PhaseTimings | None = None,
    ) -> StepObservation:
        data_wait = _require_nonnegative_finite(data_wait_seconds, "data_wait_seconds")
        step = _require_nonnegative_finite(step_seconds, "step_seconds")
        if metrics.tokens < 0:
            raise ValueError("StepMetrics.tokens must be non-negative")
        if phases is not None:
            phases.validate()
        else:
            phases = PhaseTimings()

        self._observed_steps += 1
        self._optimizer_steps += int(metrics.optimizer_stepped)
        self._tokens += int(metrics.tokens)
        self._data_wait_seconds += data_wait
        self._step_seconds += step
        self._update_scalar_metrics(metrics)
        self._update_phase_totals(phases)

        memory = _memory_snapshot(self.device)
        self._update_memory_peak(memory)
        gpu_utilization: float | None = None
        if self._observed_steps % self.gpu_sample_every_steps == 0:
            gpu_utilization, status = _sample_gpu_utilization(self.device)
            self._gpu_utilization_status = status
            if gpu_utilization is not None:
                self._gpu_samples.append(gpu_utilization)

        train_elapsed = data_wait + step
        observation = StepObservation(
            micro_step=metrics.micro_step,
            optimizer_step=metrics.optimizer_step,
            optimizer_stepped=metrics.optimizer_stepped,
            tokens=metrics.tokens,
            loss=metrics.loss,
            update_loss=metrics.update_loss,
            learning_rate=metrics.learning_rate,
            grad_norm=metrics.grad_norm,
            data_wait_seconds=data_wait,
            step_seconds=step,
            train_tokens_per_second=_safe_ratio(float(metrics.tokens), train_elapsed),
            compute_tokens_per_second=_safe_ratio(float(metrics.tokens), step),
            forward_seconds=phases.forward_seconds,
            backward_seconds=phases.backward_seconds,
            update_seconds=phases.update_seconds,
            memory=memory,
            gpu_utilization_percent=gpu_utilization,
        )
        self._retain_step_sample(observation)
        return observation

    def _update_scalar_metrics(self, metrics: StepMetrics) -> None:
        loss = float(metrics.loss)
        if not math.isfinite(loss):
            raise ValueError("StepMetrics.loss must be finite")
        lr = float(metrics.learning_rate)
        if not math.isfinite(lr):
            raise ValueError("StepMetrics.learning_rate must be finite")
        if self._loss_first is None:
            self._loss_first = loss
            self._lr_first = lr
        self._loss_final = loss
        self._lr_final = lr
        self._loss_min = loss if self._loss_min is None else min(self._loss_min, loss)
        self._loss_max = loss if self._loss_max is None else max(self._loss_max, loss)
        if metrics.grad_norm is not None:
            grad_norm = float(metrics.grad_norm)
            if not math.isfinite(grad_norm) or grad_norm < 0.0:
                raise ValueError("StepMetrics.grad_norm must be finite and non-negative")
            self._grad_norm_min = (
                grad_norm if self._grad_norm_min is None else min(self._grad_norm_min, grad_norm)
            )
            self._grad_norm_max = (
                grad_norm if self._grad_norm_max is None else max(self._grad_norm_max, grad_norm)
            )

    def _update_phase_totals(self, phases: PhaseTimings) -> None:
        values = {
            "forward": phases.forward_seconds,
            "backward": phases.backward_seconds,
            "update": phases.update_seconds,
        }
        for name, value in values.items():
            if value is not None:
                self._phase_totals[name] += float(value)
                self._phase_counts[name] += 1

    def _update_memory_peak(self, snapshot: MemorySnapshot) -> None:
        def maximum(left: int | None, right: int | None) -> int | None:
            if left is None:
                return right
            if right is None:
                return left
            return max(left, right)

        self._memory_peak = MemorySnapshot(
            process_rss_peak_bytes=maximum(
                self._memory_peak.process_rss_peak_bytes,
                snapshot.process_rss_peak_bytes,
            ),
            cuda_allocated_bytes=maximum(
                self._memory_peak.cuda_allocated_bytes,
                snapshot.cuda_allocated_bytes,
            ),
            cuda_reserved_bytes=maximum(
                self._memory_peak.cuda_reserved_bytes,
                snapshot.cuda_reserved_bytes,
            ),
            cuda_peak_allocated_bytes=maximum(
                self._memory_peak.cuda_peak_allocated_bytes,
                snapshot.cuda_peak_allocated_bytes,
            ),
            cuda_peak_reserved_bytes=maximum(
                self._memory_peak.cuda_peak_reserved_bytes,
                snapshot.cuda_peak_reserved_bytes,
            ),
        )

    def _retain_step_sample(self, observation: StepObservation) -> None:
        if self._observed_steps % self._sample_stride != 0:
            return
        self._step_samples.append(observation)
        if len(self._step_samples) <= self.max_step_samples:
            return
        self._step_samples = self._step_samples[::2]
        self._sample_stride *= 2

    def measure_region(
        self,
        kind: str,
        operation: str,
        fn: Callable[[], _T],
        *,
        optimizer_step: int | None = None,
        tokens_seen: int | None = None,
        synchronize_cuda: bool = True,
    ) -> _T:
        """Measure checkpoint/evaluation work outside the hot training-step path."""
        if kind not in {"checkpoint", "evaluation"}:
            raise ValueError("kind must be 'checkpoint' or 'evaluation'")
        if not operation.strip():
            raise ValueError("operation must be non-empty")
        if synchronize_cuda:
            self._synchronize_cuda()
        started = time.perf_counter()
        status = "PASS"
        try:
            return fn()
        except Exception:
            status = "ERROR"
            raise
        finally:
            if synchronize_cuda:
                self._synchronize_cuda()
            elapsed = time.perf_counter() - started
            self._regions.append(
                RegionObservation(
                    kind=kind,
                    operation=operation,
                    seconds=elapsed,
                    optimizer_step=optimizer_step,
                    tokens_seen=tokens_seen,
                    status=status,
                )
            )

    def summary(self) -> dict[str, Any]:
        step_values = [item.step_seconds for item in self._step_samples]
        data_values = [item.data_wait_seconds for item in self._step_samples]
        checkpoint_regions = [item for item in self._regions if item.kind == "checkpoint"]
        evaluation_regions = [item for item in self._regions if item.kind == "evaluation"]
        checkpoint_seconds = sum(item.seconds for item in checkpoint_regions)
        evaluation_seconds = sum(item.seconds for item in evaluation_regions)
        training_observed_seconds = self._step_seconds + self._data_wait_seconds
        total_observed_seconds = (
            training_observed_seconds + checkpoint_seconds + evaluation_seconds
        )
        bottleneck = _classify_bottleneck(
            data_wait_seconds=self._data_wait_seconds,
            step_seconds=self._step_seconds,
            checkpoint_seconds=checkpoint_seconds,
            evaluation_seconds=evaluation_seconds,
            gpu_samples=self._gpu_samples,
            device_type=self.device.type,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_identity_sha256": self.run_identity_sha256,
            "rank": asdict(self.rank),
            "device": {
                "type": self.device.type,
                "value": str(self.device),
                "step_timing_mode": (
                    "CUDA_SYNCHRONIZED_WALL"
                    if self.device.type == "cuda" and self.synchronize_cuda_steps
                    else "CUDA_HOST_ENQUEUE_WALL"
                    if self.device.type == "cuda"
                    else "CPU_WALL"
                ),
            },
            "counters": {
                "observed_microbatches": self._observed_steps,
                "observed_optimizer_steps": self._optimizer_steps,
                "optimized_tokens": self._tokens,
                "retained_step_samples": len(self._step_samples),
                "retained_step_stride": self._sample_stride,
            },
            "throughput": {
                "train_tokens_per_second": _safe_ratio(
                    float(self._tokens), training_observed_seconds
                ),
                "compute_tokens_per_second": _safe_ratio(float(self._tokens), self._step_seconds),
            },
            "timing": {
                "training_observed_seconds": training_observed_seconds,
                "step_seconds_total": self._step_seconds,
                "step_seconds_p50": statistics.median(step_values) if step_values else None,
                "step_seconds_p95": _percentile(step_values, 0.95),
                "data_wait_seconds_total": self._data_wait_seconds,
                "data_wait_seconds_p50": statistics.median(data_values) if data_values else None,
                "data_wait_seconds_p95": _percentile(data_values, 0.95),
                "checkpoint_seconds_total": checkpoint_seconds,
                "checkpoint_count": len(checkpoint_regions),
                "evaluation_seconds_total": evaluation_seconds,
                "evaluation_count": len(evaluation_regions),
                "total_observed_seconds": total_observed_seconds,
            },
            "phase_timing": {
                "forward": _phase_summary(
                    self._phase_totals["forward"], self._phase_counts["forward"]
                ),
                "backward": _phase_summary(
                    self._phase_totals["backward"], self._phase_counts["backward"]
                ),
                "update": _phase_summary(
                    self._phase_totals["update"], self._phase_counts["update"]
                ),
                "current_d02_trainer_contract": (
                    "WHOLE_MICROBATCH_ONLY_NO_PUBLIC_FORWARD_BACKWARD_UPDATE_TIMING_SEAMS"
                ),
            },
            "optimization": {
                "loss_first": self._loss_first,
                "loss_final": self._loss_final,
                "loss_min": self._loss_min,
                "loss_max": self._loss_max,
                "learning_rate_first": self._lr_first,
                "learning_rate_final": self._lr_final,
                "gradient_norm_min": self._grad_norm_min,
                "gradient_norm_max": self._grad_norm_max,
            },
            "memory_peak": asdict(self._memory_peak),
            "gpu_utilization": {
                "status": self._gpu_utilization_status,
                "samples": len(self._gpu_samples),
                "mean_percent": (
                    statistics.fmean(self._gpu_samples) if self._gpu_samples else None
                ),
                "min_percent": min(self._gpu_samples) if self._gpu_samples else None,
                "max_percent": max(self._gpu_samples) if self._gpu_samples else None,
            },
            "bottleneck": bottleneck,
            "determinism_boundary": {
                "run_identity_hash_contains_telemetry": False,
                "timing_or_resource_metrics_are_training_state": False,
                "telemetry_collection_initializes_distributed_process_group": False,
            },
        }

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "authority": "OBSERVABILITY_EVIDENCE_NOT_TRAINING_STATE_OR_STAGE_PROMOTION",
            "run_identity": self.run_identity,
            "run_identity_sha256": self.run_identity_sha256,
            "started_at_utc": self.started_at_utc,
            "step_samples": [asdict(item) for item in self._step_samples],
            "regions": [asdict(item) for item in self._regions],
            "summary": self.summary(),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.export(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_jsonl(self, path: str | Path) -> None:
        """Write bounded structured records after the run; no hot-loop file I/O."""
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
            for item in self._step_samples
        )
        records.extend(
            {
                "record_type": "region",
                "run_identity_sha256": self.run_identity_sha256,
                **asdict(item),
            }
            for item in self._regions
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


def _phase_summary(total_seconds: float, count: int) -> dict[str, Any]:
    return {
        "status": "MEASURED" if count else "UNAVAILABLE_NOT_RECORDED",
        "samples": count,
        "seconds_total": total_seconds if count else None,
        "seconds_mean": total_seconds / count if count else None,
    }


def _classify_bottleneck(
    *,
    data_wait_seconds: float,
    step_seconds: float,
    checkpoint_seconds: float,
    evaluation_seconds: float,
    gpu_samples: Sequence[float],
    device_type: str,
) -> dict[str, Any]:
    training_seconds = data_wait_seconds + step_seconds
    total_seconds = training_seconds + checkpoint_seconds + evaluation_seconds
    if training_seconds <= 0.0:
        return {
            "classification": "INSUFFICIENT_OBSERVATION",
            "basis": "no measured training time",
        }

    data_share = data_wait_seconds / training_seconds
    step_share = step_seconds / training_seconds
    checkpoint_share = checkpoint_seconds / total_seconds if total_seconds > 0.0 else 0.0
    evaluation_share = evaluation_seconds / total_seconds if total_seconds > 0.0 else 0.0
    gpu_mean = statistics.fmean(gpu_samples) if gpu_samples else None

    if checkpoint_share >= 0.15:
        classification = "CHECKPOINT_BOUND"
        basis = "checkpoint time is at least 15% of observed train/eval/checkpoint wall"
    elif data_share >= 0.20:
        classification = "DATA_BOUND"
        basis = "blocking data wait is at least 20% of observed training wall"
    elif device_type == "cuda" and gpu_mean is not None and gpu_mean >= 70.0:
        classification = "COMPUTE_BOUND_GPU_SATURATED"
        basis = "data wait is low and sampled GPU utilization averages at least 70%"
    elif step_share >= 0.80:
        classification = "COMPUTE_BOUND_OR_RUNTIME_BOUND"
        basis = "training-step work is at least 80% of observed training wall"
    else:
        classification = "MIXED_OR_INCONCLUSIVE"
        basis = "no configured bottleneck threshold dominates"

    return {
        "classification": classification,
        "basis": basis,
        "heuristic_not_capacity_claim": True,
        "data_wait_fraction_of_training": data_share,
        "step_fraction_of_training": step_share,
        "checkpoint_fraction_of_observed_total": checkpoint_share,
        "evaluation_fraction_of_observed_total": evaluation_share,
        "gpu_utilization_mean_percent": gpu_mean,
    }


def aggregate_rank_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate end-of-run summaries without per-step distributed collectives."""
    if not summaries:
        raise ValueError("at least one rank summary is required")
    identity_hashes = {str(item.get("run_identity_sha256")) for item in summaries}
    if len(identity_hashes) != 1:
        raise ValueError("all rank summaries must have the same run identity SHA-256")

    ranks: list[int] = []
    declared_world_sizes: set[int] = set()
    elapsed: list[float] = []
    tokens: list[int] = []
    data_wait: list[float] = []
    step_time: list[float] = []
    for summary in summaries:
        rank = summary.get("rank")
        counters = summary.get("counters")
        timing = summary.get("timing")
        if not isinstance(rank, Mapping) or not isinstance(counters, Mapping):
            raise TypeError("rank summary missing rank/counters mapping")
        if not isinstance(timing, Mapping):
            raise TypeError("rank summary missing timing mapping")
        ranks.append(int(rank["rank"]))
        declared_world_sizes.add(int(rank["world_size"]))
        tokens.append(int(counters["optimized_tokens"]))
        elapsed.append(float(timing["training_observed_seconds"]))
        data_wait.append(float(timing["data_wait_seconds_total"]))
        step_time.append(float(timing["step_seconds_total"]))

    if len(set(ranks)) != len(ranks):
        raise ValueError("rank summaries contain duplicate rank IDs")
    if len(declared_world_sizes) != 1:
        raise ValueError("rank summaries disagree on world size")
    world_size = next(iter(declared_world_sizes))
    if len(summaries) != world_size:
        raise ValueError(
            f"distributed aggregate requires all ranks: got {len(summaries)}, expected {world_size}"
        )
    if set(ranks) != set(range(world_size)):
        raise ValueError("distributed aggregate requires contiguous ranks 0..world_size-1")

    critical_path_seconds = max(elapsed)
    minimum_elapsed = min(elapsed)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_identity_sha256": next(iter(identity_hashes)),
        "world_size": world_size,
        "optimized_tokens_sum": sum(tokens),
        "critical_path_training_seconds": critical_path_seconds,
        "global_train_tokens_per_second": _safe_ratio(float(sum(tokens)), critical_path_seconds),
        "rank_training_seconds_min": minimum_elapsed,
        "rank_training_seconds_max": critical_path_seconds,
        "rank_time_skew_ratio": (
            critical_path_seconds / minimum_elapsed if minimum_elapsed > 0.0 else None
        ),
        "rank_data_wait_seconds_max": max(data_wait),
        "rank_step_seconds_max": max(step_time),
        "aggregation_semantics": (
            "SUM_TOKENS_OVER_MAX_RANK_TRAINING_WALL_NO_PER_STEP_COLLECTIVES"
        ),
    }


def gather_distributed_summary(local_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Gather one compact summary per initialized rank, then aggregate locally."""
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return aggregate_rank_summaries([local_summary])
    world_size = int(distributed.get_world_size())
    gathered: list[Any] = [None] * world_size
    distributed.all_gather_object(gathered, dict(local_summary))
    if not all(isinstance(item, Mapping) for item in gathered):
        raise RuntimeError("distributed observability gather returned a non-mapping summary")
    return aggregate_rank_summaries(gathered)


def paid_compute_decision_support(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return conservative evidence gates; never authorize spend from telemetry alone."""
    rank = summary.get("rank")
    device = summary.get("device")
    throughput = summary.get("throughput")
    if not isinstance(rank, Mapping) or not isinstance(device, Mapping):
        raise TypeError("summary missing rank/device metadata")
    if not isinstance(throughput, Mapping):
        raise TypeError("summary missing throughput metadata")

    device_type = str(device.get("type"))
    world_size = int(rank.get("world_size", 1))
    measured_tps = throughput.get("train_tokens_per_second")
    common = {
        "measured_train_tokens_per_second": measured_tps,
        "bottleneck": summary.get("bottleneck"),
        "cost_projection_formula": (
            "projected_cost_eur = target_training_tokens / measured_global_tokens_per_second "
            "/ 3600 * measured_or_quoted_eur_per_gpu_hour * gpu_count"
        ),
        "telemetry_alone_authorizes_spend": False,
    }
    if device_type != "cuda":
        return {
            **common,
            "euro_2000_gate": "BLOCKED_PENDING_TARGET_GPU_CALIBRATION",
            "euro_10000_gate": "BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION",
            "reason": "CPU observability cannot support a paid GPU capacity/cost claim",
        }
    if world_size == 1:
        return {
            **common,
            "euro_2000_gate": "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE",
            "euro_10000_gate": "BLOCKED_PENDING_MULTI_GPU_SCALING_EVIDENCE",
            "reason": "single-GPU throughput can price a bounded run but not distributed scaling",
        }
    return {
        **common,
        "euro_2000_gate": "REQUIRES_TOKEN_BUDGET_AND_PROVIDER_PRICE",
        "euro_10000_gate": "REQUIRES_TOKEN_BUDGET_PROVIDER_PRICE_AND_STABILITY_GATE",
        "reason": "target-hardware telemetry exists; cost and scientific gates remain external",
    }