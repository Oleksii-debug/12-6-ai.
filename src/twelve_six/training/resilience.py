"""Failure/recovery orchestration for interruption-safe long-running training.

This module deliberately sits above D02 Trainer and D05 checkpoint primitives.
It never repairs a poisoned Trainer in place and it does not implement elastic
multi-rank recovery. The recovery authority is always a newly verified durable
checkpoint applied to fresh model/Trainer objects by the caller.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import CheckpointError, hash_json, verify_checkpoint

RECOVERY_SCHEMA_VERSION = "12-6.training-recovery.v1"
RECOVERY_AUTHORITY = "LOCAL_PROCESS_RECOVERY_POLICY_NOT_DISTRIBUTED_ELASTICITY"
_CHECKPOINT_DIR = re.compile(r"^step-(\d{12})-attempt-(\d{6})$")
_SHA_HEX = frozenset("0123456789abcdef")


class RecoveryError(RuntimeError):
    """Base recovery orchestration failure."""


class RecoveryStateError(RecoveryError):
    """Raised when the recovery journal or transition is invalid."""


class RetryBudgetExceeded(RecoveryError):
    """Raised before starting an attempt beyond the configured restart budget."""


class RunPhase(StrEnum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    PREEMPTED = "PREEMPTED"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FailureClass(StrEnum):
    PREEMPTION = "PREEMPTION"
    PROCESS_LOSS = "PROCESS_LOSS"
    CHECKPOINT_IO = "CHECKPOINT_IO"
    TRAINER_POISONED = "TRAINER_POISONED"
    RANK_LOSS = "RANK_LOSS"
    USER_SHUTDOWN = "USER_SHUTDOWN"


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    checkpoint_every_steps: int
    retain_last: int = 3
    max_restarts: int = 3
    max_preemptions: int = 8
    target_checkpoint_overhead_fraction: float = 0.05
    max_recovery_window_seconds: float = 900.0
    require_exact_topology_resume: bool = True

    def validate(self) -> None:
        for field, value, minimum in (
            ("checkpoint_every_steps", self.checkpoint_every_steps, 1),
            ("retain_last", self.retain_last, 2),
            ("max_restarts", self.max_restarts, 0),
            ("max_preemptions", self.max_preemptions, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        fraction = self.target_checkpoint_overhead_fraction
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise ValueError("target_checkpoint_overhead_fraction must be finite in (0, 1)")
        window = self.max_recovery_window_seconds
        if (
            not isinstance(window, (int, float))
            or isinstance(window, bool)
            or not math.isfinite(float(window))
            or float(window) <= 0.0
        ):
            raise ValueError("max_recovery_window_seconds must be finite and positive")
        if not isinstance(self.require_exact_topology_resume, bool):
            raise TypeError("require_exact_topology_resume must be boolean")


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    directory: str
    checkpoint_id: str
    optimizer_step: int
    tokens_seen: int
    attempt: int
    bytes: int


@dataclass(frozen=True, slots=True)
class CadenceRecommendation:
    checkpoint_seconds: float
    optimizer_step_seconds: float
    target_overhead_fraction: float
    max_recovery_window_seconds: float
    minimum_interval_steps_for_overhead: int
    maximum_interval_steps_for_recovery: int
    recommended_interval_steps: int
    predicted_checkpoint_overhead_fraction: float
    predicted_max_recompute_seconds: float
    constraints_satisfied: bool


@dataclass(frozen=True, slots=True)
class AttemptResult:
    status: str
    optimizer_step: int
    tokens_seen: int
    checkpoint_id: str | None
    stop_reason: str | None


class StopLatch:
    """Signal-safe cooperative stop request observed only at committed step boundaries."""

    def __init__(self) -> None:
        self._reason: str | None = None
        self._failure_class = FailureClass.USER_SHUTDOWN

    @property
    def requested(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def failure_class(self) -> FailureClass:
        return self._failure_class

    def request(
        self,
        reason: str,
        *,
        failure_class: FailureClass = FailureClass.USER_SHUTDOWN,
    ) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason must be non-empty text")
        if not isinstance(failure_class, FailureClass):
            raise TypeError("failure_class must be FailureClass")
        if failure_class not in {FailureClass.PREEMPTION, FailureClass.USER_SHUTDOWN}:
            raise ValueError("stop latch supports only preemption or user shutdown")
        if self._reason is None:
            self._reason = reason.strip()
            self._failure_class = failure_class


@contextmanager
def install_preemption_handlers(latch: StopLatch):
    """Translate SIGTERM/SIGINT into a cooperative safe-boundary stop request."""

    handled = [signal.SIGTERM, signal.SIGINT]
    previous = {item: signal.getsignal(item) for item in handled}

    def handle(signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        latch.request(f"signal:{name}", failure_class=FailureClass.PREEMPTION)

    try:
        for item in handled:
            signal.signal(item, handle)
        yield latch
    finally:
        for item, old_handler in previous.items():
            signal.signal(item, old_handler)


def recommend_checkpoint_interval(
    *,
    optimizer_step_seconds: float,
    checkpoint_seconds: float,
    target_overhead_fraction: float = 0.05,
    max_recovery_window_seconds: float = 900.0,
) -> CadenceRecommendation:
    """Choose a cadence from measured step/checkpoint time and explicit risk bounds."""

    values = {
        "optimizer_step_seconds": optimizer_step_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "target_overhead_fraction": target_overhead_fraction,
        "max_recovery_window_seconds": max_recovery_window_seconds,
    }
    for name, value in values.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < target_overhead_fraction < 1.0:
        raise ValueError("target_overhead_fraction must be in (0, 1)")

    step_seconds = float(optimizer_step_seconds)
    save_seconds = float(checkpoint_seconds)
    target = float(target_overhead_fraction)
    recovery_window = float(max_recovery_window_seconds)

    minimum = max(1, math.ceil(save_seconds * (1.0 - target) / (target * step_seconds)))
    maximum = max(1, math.floor(recovery_window / step_seconds))
    satisfied = minimum <= maximum
    recommended = minimum
    overhead = save_seconds / (recommended * step_seconds + save_seconds)
    recompute = recommended * step_seconds
    return CadenceRecommendation(
        checkpoint_seconds=save_seconds,
        optimizer_step_seconds=step_seconds,
        target_overhead_fraction=target,
        max_recovery_window_seconds=recovery_window,
        minimum_interval_steps_for_overhead=minimum,
        maximum_interval_steps_for_recovery=maximum,
        recommended_interval_steps=recommended,
        predicted_checkpoint_overhead_fraction=overhead,
        predicted_max_recompute_seconds=recompute,
        constraints_satisfied=satisfied,
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _require_sha(value: Any, *, field: str, lengths: set[int]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or value != value.lower()
        or any(char not in _SHA_HEX for char in value)
    ):
        expected = "/".join(str(item) for item in sorted(lengths))
        raise RecoveryStateError(f"{field} must be exact lowercase {expected}-hex")
    return value


def _state_digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("state_sha256", None)
    return hash_json(body)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = _canonical_json(payload)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryStateError(f"{path} must contain a JSON object")
    return value


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


class RecoveryStore:
    """Single-supervisor recovery state plus verified last-known-good checkpoint discovery."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_manifest: Mapping[str, Any],
        policy: RecoveryPolicy,
    ) -> None:
        policy.validate()
        self.root = Path(root).resolve()
        self.checkpoints_dir = self.root / "checkpoints"
        self.attempts_dir = self.root / "attempts"
        self.failures_dir = self.root / "failures"
        self.state_path = self.root / "recovery-state.json"
        self.run_manifest_path = self.root / "run-manifest.json"
        self.terminal_path = self.root / "terminal.json"
        self.policy = policy
        self.run_manifest = dict(run_manifest)
        self.run_manifest_sha256 = hash_json(self.run_manifest)

        run_id = self.run_manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RecoveryStateError("run manifest requires non-empty run_id")
        self.run_id = run_id
        candidate = self.run_manifest.get("candidate")
        if not isinstance(candidate, Mapping):
            raise RecoveryStateError("run manifest candidate must be a mapping")
        self.source_sha = _require_sha(
            candidate.get("git_sha"), field="candidate.git_sha", lengths={40, 64}
        )
        recovery = self.run_manifest.get("recovery")
        if not isinstance(recovery, Mapping):
            raise RecoveryStateError("run manifest recovery must be a mapping")
        topology = recovery.get("topology")
        if not isinstance(topology, Mapping) or not topology:
            raise RecoveryStateError("run manifest recovery.topology must be non-empty")
        self.topology_sha256 = hash_json(topology)
        world_size = topology.get("world_size")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size <= 0
        ):
            raise RecoveryStateError("recovery.topology.world_size must be positive integer")
        self.world_size = world_size

        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.attempts_dir.mkdir(exist_ok=True)
        self.failures_dir.mkdir(exist_ok=True)
        if self.run_manifest_path.exists():
            persisted_manifest = _load_json_object(self.run_manifest_path)
            if hash_json(persisted_manifest) != self.run_manifest_sha256:
                raise RecoveryStateError("persisted run manifest does not match launch identity")
        else:
            _atomic_write_json(self.run_manifest_path, self.run_manifest)

    def _base_state(self) -> dict[str, Any]:
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "authority": RECOVERY_AUTHORITY,
            "run_id": self.run_id,
            "run_manifest_sha256": self.run_manifest_sha256,
            "source_sha": self.source_sha,
            "topology_sha256": self.topology_sha256,
            "world_size": self.world_size,
            "policy": asdict(self.policy),
            "phase": RunPhase.PREPARED.value,
            "attempt": 0,
            "restarts_used": 0,
            "preemptions_seen": 0,
            "journal_reconstructed": False,
            "last_known_good": None,
            "checkpoint_count": 0,
            "invalid_checkpoint_directories": [],
            "failure_count": 0,
        }

    def _validate_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("schema_version") != RECOVERY_SCHEMA_VERSION:
            raise RecoveryStateError("recovery state schema mismatch")
        if state.get("authority") != RECOVERY_AUTHORITY:
            raise RecoveryStateError("recovery state authority mismatch")
        checks = {
            "run_id": self.run_id,
            "run_manifest_sha256": self.run_manifest_sha256,
            "source_sha": self.source_sha,
            "topology_sha256": self.topology_sha256,
            "world_size": self.world_size,
            "policy": asdict(self.policy),
        }
        for field, expected in checks.items():
            if state.get(field) != expected:
                raise RecoveryStateError(f"recovery state {field} mismatch")
        try:
            RunPhase(state.get("phase"))
        except ValueError as exc:
            raise RecoveryStateError("recovery state has invalid phase") from exc
        if state.get("state_sha256") != _state_digest(state):
            raise RecoveryStateError("recovery state self-hash mismatch")
        return dict(state)

    def _persist(self, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(state)
        payload["state_sha256"] = _state_digest(payload)
        _atomic_write_json(self.state_path, payload)
        return payload

    def _marker_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_manifest_sha256": self.run_manifest_sha256,
            "source_sha": self.source_sha,
            "topology_sha256": self.topology_sha256,
            **dict(payload),
        }
        body["state_sha256"] = _state_digest(body)
        return body

    def _write_terminal(self, phase: RunPhase) -> None:
        if phase not in {RunPhase.COMPLETED, RunPhase.FAILED}:
            raise ValueError("terminal marker requires COMPLETED or FAILED")
        payload = self._marker_payload({"phase": phase.value})
        if self.terminal_path.exists():
            existing = _load_json_object(self.terminal_path)
            if existing != payload:
                raise RecoveryStateError("terminal marker conflicts with requested terminal phase")
            return
        _atomic_write_json(self.terminal_path, payload)

    def _read_terminal(self) -> RunPhase | None:
        if not self.terminal_path.exists():
            return None
        marker = _load_json_object(self.terminal_path)
        if marker.get("state_sha256") != _state_digest(marker):
            raise RecoveryStateError("terminal marker self-hash mismatch")
        for field, expected in (
            ("schema_version", RECOVERY_SCHEMA_VERSION),
            ("run_id", self.run_id),
            ("run_manifest_sha256", self.run_manifest_sha256),
            ("source_sha", self.source_sha),
            ("topology_sha256", self.topology_sha256),
        ):
            if marker.get(field) != expected:
                raise RecoveryStateError(f"terminal marker {field} mismatch")
        try:
            phase = RunPhase(marker.get("phase"))
        except ValueError as exc:
            raise RecoveryStateError("terminal marker phase invalid") from exc
        if phase not in {RunPhase.COMPLETED, RunPhase.FAILED}:
            raise RecoveryStateError("terminal marker must be COMPLETED or FAILED")
        return phase

    def _read_markers(self, directory: Path, prefix: str) -> list[dict[str, Any]]:
        markers: list[dict[str, Any]] = []
        for path in sorted(directory.glob(f"{prefix}-*.json")):
            marker = _load_json_object(path)
            if marker.get("state_sha256") != _state_digest(marker):
                raise RecoveryStateError(f"marker self-hash mismatch: {path.name}")
            for field, expected in (
                ("schema_version", RECOVERY_SCHEMA_VERSION),
                ("run_id", self.run_id),
                ("run_manifest_sha256", self.run_manifest_sha256),
                ("source_sha", self.source_sha),
                ("topology_sha256", self.topology_sha256),
            ):
                if marker.get(field) != expected:
                    raise RecoveryStateError(f"marker {field} mismatch: {path.name}")
            markers.append(marker)
        return markers

    def discover_checkpoints(self) -> tuple[list[CheckpointRecord], list[str]]:
        valid: list[CheckpointRecord] = []
        invalid: list[str] = []
        for path in sorted(self.checkpoints_dir.iterdir()):
            if not path.is_dir():
                continue
            match = _CHECKPOINT_DIR.fullmatch(path.name)
            if match is None:
                continue
            step_from_name = int(match.group(1))
            attempt_from_name = int(match.group(2))
            try:
                manifest = verify_checkpoint(path)
                identity = manifest.get("identity")
                if not isinstance(identity, Mapping):
                    raise RecoveryStateError("checkpoint identity is missing")
                if identity.get("git_sha") != self.source_sha:
                    raise RecoveryStateError("checkpoint source SHA mismatch")
                if identity.get("run_manifest_hash") != self.run_manifest_sha256:
                    raise RecoveryStateError("checkpoint run manifest mismatch")
                training_config = identity.get("training_config")
                if not isinstance(training_config, Mapping):
                    raise RecoveryStateError("checkpoint training config missing")
                if training_config.get("run_id") != self.run_id:
                    raise RecoveryStateError("checkpoint run id mismatch")
                step = identity.get("step")
                tokens_seen = identity.get("tokens_seen")
                if step != step_from_name:
                    raise RecoveryStateError("checkpoint directory step mismatch")
                if (
                    not isinstance(tokens_seen, int)
                    or isinstance(tokens_seen, bool)
                    or tokens_seen < 0
                ):
                    raise RecoveryStateError("checkpoint tokens_seen invalid")
                checkpoint_id = _require_sha(
                    manifest.get("checkpoint_id"), field="checkpoint_id", lengths={64}
                )
                valid.append(
                    CheckpointRecord(
                        directory=path.name,
                        checkpoint_id=checkpoint_id,
                        optimizer_step=step,
                        tokens_seen=tokens_seen,
                        attempt=attempt_from_name,
                        bytes=_directory_bytes(path),
                    )
                )
            except (CheckpointError, OSError, ValueError, RecoveryError):
                invalid.append(path.name)
        valid.sort(key=lambda item: (item.optimizer_step, item.attempt, item.checkpoint_id))
        return valid, invalid

    def _prune_verified_retention(self) -> None:
        checkpoints, _invalid = self.discover_checkpoints()
        for record in checkpoints[:-self.policy.retain_last]:
            shutil.rmtree(self.checkpoints_dir / record.directory)

    def _rebuild_state(self, *, journal_reconstructed: bool) -> dict[str, Any]:
        attempts = self._read_markers(self.attempts_dir, "attempt")
        failures = self._read_markers(self.failures_dir, "failure")
        checkpoints, invalid = self.discover_checkpoints()
        preemptions = sum(
            marker.get("failure_class") == FailureClass.PREEMPTION.value
            for marker in failures
        )
        restarts_used = max(0, len(attempts) - 1)
        terminal = self._read_terminal()
        if terminal is not None:
            phase = terminal
        elif restarts_used > self.policy.max_restarts or preemptions > self.policy.max_preemptions:
            phase = RunPhase.FAILED
        else:
            phase = RunPhase.RECOVERING if attempts else RunPhase.PREPARED
        state = self._base_state()
        state.update(
            {
                "phase": phase.value,
                "attempt": len(attempts),
                "restarts_used": restarts_used,
                "preemptions_seen": preemptions,
                "journal_reconstructed": journal_reconstructed,
                "last_known_good": asdict(checkpoints[-1]) if checkpoints else None,
                "checkpoint_count": len(checkpoints),
                "invalid_checkpoint_directories": invalid,
                "failure_count": len(failures),
            }
        )
        return self._persist(state)

    def open(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._rebuild_state(journal_reconstructed=False)
        try:
            state = self._validate_state(_load_json_object(self.state_path))
        except (OSError, json.JSONDecodeError, RecoveryStateError):
            return self._rebuild_state(journal_reconstructed=True)

        attempts = self._read_markers(self.attempts_dir, "attempt")
        failures = self._read_markers(self.failures_dir, "failure")
        checkpoints, invalid = self.discover_checkpoints()
        terminal = self._read_terminal()
        latest = asdict(checkpoints[-1]) if checkpoints else None
        attempt_count = len(attempts)
        restart_count = max(0, attempt_count - 1)
        preemptions = sum(
            marker.get("failure_class") == FailureClass.PREEMPTION.value
            for marker in failures
        )
        failure_count = len(failures)
        phase = RunPhase(state["phase"])
        if terminal is not None:
            phase = terminal
        elif failure_count > state.get("failure_count", 0):
            latest_failure = failures[-1]
            latest_class = latest_failure.get("failure_class")
            if latest_class == FailureClass.PREEMPTION.value:
                phase = RunPhase.PREEMPTED
            elif latest_class == FailureClass.USER_SHUTDOWN.value:
                phase = RunPhase.PAUSED
            else:
                phase = RunPhase.RECOVERING
        elif attempt_count > state.get("attempt", 0):
            phase = RunPhase.RECOVERING
        if restart_count > self.policy.max_restarts or preemptions > self.policy.max_preemptions:
            phase = RunPhase.FAILED
            self._write_terminal(RunPhase.FAILED)

        reconciled = {
            "phase": phase.value,
            "attempt": attempt_count,
            "restarts_used": restart_count,
            "preemptions_seen": preemptions,
            "last_known_good": latest,
            "checkpoint_count": len(checkpoints),
            "invalid_checkpoint_directories": invalid,
            "failure_count": failure_count,
        }
        if any(state.get(key) != value for key, value in reconciled.items()):
            state.update(reconciled)
            state = self._persist(state)
        return state

    def last_known_good(self) -> CheckpointRecord | None:
        checkpoints, _invalid = self.discover_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def checkpoint_path(self, *, optimizer_step: int, attempt: int) -> Path:
        if optimizer_step < 0 or attempt <= 0:
            raise ValueError("optimizer_step must be non-negative and attempt must be positive")
        return self.checkpoints_dir / f"step-{optimizer_step:012d}-attempt-{attempt:06d}"

    def begin_attempt(self) -> dict[str, Any]:
        state = self.open()
        phase = RunPhase(state["phase"])
        if phase in {RunPhase.COMPLETED, RunPhase.FAILED}:
            raise RecoveryStateError(f"cannot start attempt from terminal phase {phase.value}")
        if phase in {RunPhase.RUNNING, RunPhase.CHECKPOINTING}:
            raise RecoveryStateError(
                "previous attempt is still RUNNING/CHECKPOINTING; supervisor must record "
                "its termination before restart"
            )

        attempts = self._read_markers(self.attempts_dir, "attempt")
        next_attempt = len(attempts) + 1
        restarts_used = max(0, next_attempt - 1)
        if restarts_used > self.policy.max_restarts:
            state["phase"] = RunPhase.FAILED.value
            self._write_terminal(RunPhase.FAILED)
            self._persist(state)
            raise RetryBudgetExceeded(
                f"restart budget exhausted: {restarts_used}>{self.policy.max_restarts}"
            )
        last_good = self.last_known_good()
        marker = self._marker_payload(
            {
                "attempt": next_attempt,
                "resume_checkpoint_id": None if last_good is None else last_good.checkpoint_id,
                "resume_optimizer_step": 0 if last_good is None else last_good.optimizer_step,
            }
        )
        marker_path = self.attempts_dir / f"attempt-{next_attempt:06d}.json"
        _atomic_write_json(marker_path, marker)
        state = self.open()
        state.update(
            {
                "phase": RunPhase.RUNNING.value,
                "attempt": next_attempt,
                "restarts_used": restarts_used,
            }
        )
        return self._persist(state)

    def record_failure(
        self,
        failure_class: FailureClass,
        *,
        optimizer_step: int,
        detail_code: str,
    ) -> dict[str, Any]:
        if not isinstance(failure_class, FailureClass):
            raise TypeError("failure_class must be FailureClass")
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
        ):
            raise ValueError("optimizer_step must be non-negative integer")
        if not isinstance(detail_code, str) or not detail_code.strip():
            raise ValueError("detail_code must be non-empty text")
        state = self.open()
        if RunPhase(state["phase"]) in {RunPhase.COMPLETED, RunPhase.FAILED}:
            raise RecoveryStateError("cannot record failure after terminal state")

        failures = self._read_markers(self.failures_dir, "failure")
        sequence = len(failures) + 1
        marker = self._marker_payload(
            {
                "sequence": sequence,
                "attempt": state["attempt"],
                "failure_class": failure_class.value,
                "optimizer_step": optimizer_step,
                "detail_code": detail_code.strip(),
            }
        )
        marker_path = self.failures_dir / f"failure-{sequence:06d}.json"
        _atomic_write_json(marker_path, marker)
        state = self.open()
        preemptions = state["preemptions_seen"]
        if RunPhase(state["phase"]) is RunPhase.FAILED:
            return state
        if failure_class is FailureClass.PREEMPTION:
            phase = RunPhase.PREEMPTED
        elif failure_class is FailureClass.USER_SHUTDOWN:
            phase = RunPhase.PAUSED
        else:
            phase = RunPhase.RECOVERING
        state.update(
            {
                "phase": phase.value,
                "preemptions_seen": preemptions,
                "failure_count": sequence,
            }
        )
        return self._persist(state)

    def commit_checkpoint(
        self,
        trainer: Any,
        save: Callable[[Path], Mapping[str, Any]],
    ) -> CheckpointRecord:
        trainer.assert_checkpoint_safe()
        state = self.open()
        if RunPhase(state["phase"]) is not RunPhase.RUNNING:
            raise RecoveryStateError("checkpoint commit requires RUNNING phase")
        attempt = state["attempt"]
        step = trainer.optimizer_step
        destination = self.checkpoint_path(optimizer_step=step, attempt=attempt)
        if destination.exists():
            raise RecoveryStateError(f"checkpoint destination already exists: {destination.name}")
        state["phase"] = RunPhase.CHECKPOINTING.value
        self._persist(state)

        save(destination)
        manifest = verify_checkpoint(destination)
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise RecoveryStateError("saved checkpoint identity missing")
        if identity.get("step") != step or identity.get("tokens_seen") != trainer.tokens_seen:
            raise RecoveryStateError("saved checkpoint counters do not match trainer")
        if identity.get("git_sha") != self.source_sha:
            raise RecoveryStateError("saved checkpoint source SHA mismatch")
        if identity.get("run_manifest_hash") != self.run_manifest_sha256:
            raise RecoveryStateError("saved checkpoint run-manifest hash mismatch")

        record = CheckpointRecord(
            directory=destination.name,
            checkpoint_id=_require_sha(
                manifest.get("checkpoint_id"), field="checkpoint_id", lengths={64}
            ),
            optimizer_step=step,
            tokens_seen=trainer.tokens_seen,
            attempt=attempt,
            bytes=_directory_bytes(destination),
        )
        state = self.open()
        state.update(
            {
                "phase": RunPhase.RUNNING.value,
                "last_known_good": asdict(record),
            }
        )
        self._persist(state)
        self._prune_verified_retention()
        self.open()
        return record

    def mark_preempted(self, *, optimizer_step: int, reason_code: str) -> dict[str, Any]:
        return self.record_failure(
            FailureClass.PREEMPTION,
            optimizer_step=optimizer_step,
            detail_code=reason_code,
        )

    def mark_clean_shutdown(self, *, optimizer_step: int, reason_code: str) -> dict[str, Any]:
        return self.record_failure(
            FailureClass.USER_SHUTDOWN,
            optimizer_step=optimizer_step,
            detail_code=reason_code,
        )

    def mark_completed(self) -> dict[str, Any]:
        state = self.open()
        if RunPhase(state["phase"]) is not RunPhase.RUNNING:
            raise RecoveryStateError("completion requires RUNNING phase")
        state["phase"] = RunPhase.COMPLETED.value
        self._write_terminal(RunPhase.COMPLETED)
        return self._persist(state)


def run_resilient_training(
    trainer: Any,
    batches: Sequence[Mapping[str, Any]],
    store: RecoveryStore,
    *,
    save_checkpoint: Callable[[Path], Mapping[str, Any]],
    stop_latch: StopLatch | None = None,
    after_metrics: Callable[[Any], None] | None = None,
) -> AttemptResult:
    """Run one attempt, checkpointing only committed optimizer state.

    Callers must construct a fresh Trainer and restore the selected verified
    checkpoint before invoking this function after any poisoned/ambiguous attempt.
    """

    if not batches:
        raise ValueError("batches must be non-empty")
    store.begin_attempt()
    trainer.assert_checkpoint_safe()
    last_good = store.last_known_good()

    while trainer.optimizer_step < trainer.config.max_steps:
        batch_index = trainer.micro_step % len(batches)
        metrics = trainer.train_microbatch(batches[batch_index])
        if after_metrics is not None:
            after_metrics(metrics)
        if not metrics.optimizer_stepped:
            continue

        due = metrics.optimizer_step % store.policy.checkpoint_every_steps == 0
        if due:
            last_good = store.commit_checkpoint(trainer, save_checkpoint)

        if stop_latch is not None and stop_latch.requested:
            if last_good is None or last_good.optimizer_step != trainer.optimizer_step:
                last_good = store.commit_checkpoint(trainer, save_checkpoint)
            reason = stop_latch.reason or "cooperative-stop"
            failure_class = stop_latch.failure_class
            if failure_class is FailureClass.PREEMPTION:
                phase = RunPhase.PREEMPTED
                store.mark_preempted(
                    optimizer_step=trainer.optimizer_step,
                    reason_code=reason,
                )
            elif failure_class is FailureClass.USER_SHUTDOWN:
                phase = RunPhase.PAUSED
                store.mark_clean_shutdown(
                    optimizer_step=trainer.optimizer_step,
                    reason_code=reason,
                )
            else:
                raise RecoveryStateError("unsupported cooperative stop class")
            return AttemptResult(
                status=phase.value,
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
                checkpoint_id=last_good.checkpoint_id,
                stop_reason=reason,
            )

    if last_good is None or last_good.optimizer_step != trainer.optimizer_step:
        last_good = store.commit_checkpoint(trainer, save_checkpoint)
    store.mark_completed()
    return AttemptResult(
        status=RunPhase.COMPLETED.value,
        optimizer_step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        checkpoint_id=last_good.checkpoint_id,
        stop_reason=None,
    )
