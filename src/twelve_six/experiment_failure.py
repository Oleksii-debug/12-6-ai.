"""Machine-readable experimental workflow failure classification.

This module is intentionally stdlib-only so it can be imported before the ML runtime
is fully available. It does not replace Trainer; training wrappers can feed
StepMetrics into :class:`RunFailureTracker` through the existing ``on_metrics`` hook.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "12-6.experiment-failure-report.v1"
TAXONOMY_VERSION = "ci161.v1"
_MAX_CODES = 16
_MAX_SUMMARY = 240
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.:-]+")
_MISSING_MODULE = re.compile(r"no module named\s+['\"]?([A-Za-z0-9_.-]+)", re.IGNORECASE)


class FailureClass(str, Enum):
    BOOTSTRAP_DEPENDENCY_MISSING = "BOOTSTRAP_DEPENDENCY_MISSING"
    LOCK_PROFILE_STALE = "LOCK_PROFILE_STALE"
    STATIC_CHECK_FAILED = "STATIC_CHECK_FAILED"
    FOCUSED_TEST_FAILED = "FOCUSED_TEST_FAILED"
    EXPERIMENT_NOT_STARTED = "EXPERIMENT_NOT_STARTED"
    RESOURCE_OOM = "RESOURCE_OOM"
    TIMEOUT = "TIMEOUT"
    NONFINITE_TRAINING = "NONFINITE_TRAINING"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"
    SCIENTIFIC_REJECTION = "SCIENTIFIC_REJECTION"
    SUCCESS = "SUCCESS"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DATA_INPUT_FAILURE = "DATA_INPUT_FAILURE"
    CANCELLED = "CANCELLED"
    UNCLASSIFIED_FAILURE = "UNCLASSIFIED_FAILURE"


class FailurePhase(str, Enum):
    BOOTSTRAP = "bootstrap"
    LOCK_VALIDATION = "lock_validation"
    STATIC_CHECK = "static_check"
    FOCUSED_TEST = "focused_test"
    PREPARE = "prepare"
    TRAINING = "training"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"
    SCIENTIFIC_GATE = "scientific_gate"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class FailureSignal:
    phase: FailurePhase
    return_code: int | None = None
    optimizer_steps_completed: int = 0
    missing_dependency: str | None = None
    lock_profile_stale: bool = False
    timed_out: bool = False
    resource_oom: bool = False
    nonfinite_training: bool = False
    checkpoint_failure: bool = False
    evaluation_failure: bool = False
    scientific_rejection: bool = False
    configuration_error: bool = False
    data_input_failure: bool = False
    cancelled: bool = False
    success: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_steps_completed < 0:
            raise ValueError("optimizer_steps_completed must be >= 0")
        if self.success and self.return_code not in (None, 0):
            raise ValueError("success signal cannot carry a non-zero return_code")

    @property
    def experiment_started(self) -> bool:
        return self.optimizer_steps_completed > 0


@dataclass(slots=True)
class RunFailureTracker:
    """Track optimizer progress without changing Trainer semantics.

    Pass ``tracker.observe_metrics`` as (or compose it into) Trainer.run(on_metrics=...).
    For resumed runs, ``start_optimizer_step`` is the loaded step and the tracker records
    optimizer steps completed by this process, not lifetime checkpoint steps.
    """

    start_optimizer_step: int = 0
    optimizer_steps_completed: int = 0
    _highest_optimizer_step: int = field(init=False)

    def __post_init__(self) -> None:
        if self.start_optimizer_step < 0:
            raise ValueError("start_optimizer_step must be >= 0")
        self._highest_optimizer_step = self.start_optimizer_step

    @property
    def experiment_started(self) -> bool:
        return self.optimizer_steps_completed > 0

    def observe_metrics(self, metrics: Any) -> None:
        if not bool(getattr(metrics, "optimizer_stepped", False)):
            return
        current = int(getattr(metrics, "optimizer_step"))
        if current < self.start_optimizer_step:
            raise ValueError("optimizer_step regressed below the process start step")
        self._highest_optimizer_step = max(self._highest_optimizer_step, current)
        self.optimizer_steps_completed = self._highest_optimizer_step - self.start_optimizer_step

    def write_start_marker(self, path: str | Path) -> None:
        payload = {
            "schema": "12-6.experiment-start-marker.v1",
            "experiment_started": self.experiment_started,
            "start_optimizer_step": self.start_optimizer_step,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "highest_optimizer_step": self._highest_optimizer_step,
        }
        _atomic_json(Path(path), payload)


def classify(signal: FailureSignal) -> FailureClass:
    """Return the most specific supported failure class using stable precedence."""
    if signal.success:
        return FailureClass.SUCCESS
    if signal.missing_dependency:
        return FailureClass.BOOTSTRAP_DEPENDENCY_MISSING
    if signal.lock_profile_stale:
        return FailureClass.LOCK_PROFILE_STALE
    if signal.cancelled:
        return FailureClass.CANCELLED
    if signal.timed_out:
        return FailureClass.TIMEOUT
    if signal.resource_oom:
        return FailureClass.RESOURCE_OOM
    if signal.nonfinite_training:
        return FailureClass.NONFINITE_TRAINING
    if signal.checkpoint_failure or signal.phase is FailurePhase.CHECKPOINT:
        return FailureClass.CHECKPOINT_FAILURE
    if signal.evaluation_failure or signal.phase is FailurePhase.EVALUATION:
        return FailureClass.EVALUATION_FAILURE
    if signal.scientific_rejection or signal.phase is FailurePhase.SCIENTIFIC_GATE:
        return FailureClass.SCIENTIFIC_REJECTION
    if signal.configuration_error:
        return FailureClass.CONFIGURATION_ERROR
    if signal.data_input_failure:
        return FailureClass.DATA_INPUT_FAILURE
    if signal.phase is FailurePhase.STATIC_CHECK:
        return FailureClass.STATIC_CHECK_FAILED
    if signal.phase is FailurePhase.FOCUSED_TEST:
        return FailureClass.FOCUSED_TEST_FAILED
    if not signal.experiment_started:
        return FailureClass.EXPERIMENT_NOT_STARTED
    return FailureClass.UNCLASSIFIED_FAILURE


def safe_name(value: str | None, *, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    normalized = _SAFE_NAME.sub("_", value.strip())[:96]
    return normalized or fallback


def detect_process_markers(
    *,
    phase: FailurePhase,
    stdout_tail: bytes = b"",
    stderr_tail: bytes = b"",
) -> dict[str, Any]:
    """Extract bounded diagnostic markers without returning raw process text."""
    text = (stdout_tail + b"\n" + stderr_tail).decode("utf-8", errors="replace")
    lower = text.lower()
    missing = None
    match = _MISSING_MODULE.search(text)
    if match:
        missing = safe_name(match.group(1))
    oom = any(
        marker in lower
        for marker in (
            "cuda out of memory",
            "out of memory",
            "cannot allocate memory",
            "memoryerror",
        )
    )
    nonfinite = phase is FailurePhase.TRAINING and any(
        marker in lower
        for marker in (
            "non-finite loss",
            "non-finite gradient",
            "nonfinitetrainingerror",
        )
    )
    codes: list[str] = []
    if missing:
        codes.append(f"PYTHON_MODULE_MISSING:{missing}")
    if oom:
        codes.append("RESOURCE_OOM_MARKER")
    if nonfinite:
        codes.append("NONFINITE_TRAINING_MARKER")
    return {
        "missing_dependency": missing,
        "resource_oom": oom,
        "nonfinite_training": nonfinite,
        "diagnostic_codes": codes[:_MAX_CODES],
    }


def read_start_marker(path: str | Path | None) -> int:
    if path is None:
        return 0
    marker = Path(path)
    if not marker.is_file():
        return 0
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    value = payload.get("optimizer_steps_completed", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def make_signal_for_process(
    *,
    phase: FailurePhase,
    return_code: int,
    optimizer_steps_completed: int = 0,
    stdout_tail: bytes = b"",
    stderr_tail: bytes = b"",
    timed_out: bool = False,
    lock_profile_stale: bool = False,
) -> tuple[FailureSignal, tuple[str, ...]]:
    markers = detect_process_markers(
        phase=phase,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )
    success = return_code == 0 and not timed_out
    signal = FailureSignal(
        phase=phase,
        return_code=return_code,
        optimizer_steps_completed=optimizer_steps_completed,
        missing_dependency=markers["missing_dependency"],
        lock_profile_stale=lock_profile_stale or (phase is FailurePhase.LOCK_VALIDATION and not success),
        timed_out=timed_out,
        resource_oom=bool(markers["resource_oom"]) or return_code in (137, -9),
        nonfinite_training=bool(markers["nonfinite_training"]),
        checkpoint_failure=phase is FailurePhase.CHECKPOINT and not success,
        evaluation_failure=phase is FailurePhase.EVALUATION and not success,
        scientific_rejection=phase is FailurePhase.SCIENTIFIC_GATE and not success,
        cancelled=return_code in (130, -2),
        success=success,
    )
    return signal, tuple(markers["diagnostic_codes"])


def stream_digest(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build_report(
    signal: FailureSignal,
    *,
    diagnostic_codes: Iterable[str] = (),
    diagnostic_summary: str = "",
    source_sha: str | None = None,
    workflow: str | None = None,
    run_id: str | int | None = None,
    command_sha256: str | None = None,
    executable: str | None = None,
    stdout_digest: Mapping[str, Any] | None = None,
    stderr_digest: Mapping[str, Any] | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    failure_class = classify(signal)
    codes = [safe_name(str(code), fallback="UNKNOWN") for code in diagnostic_codes][:_MAX_CODES]
    summary = " ".join(str(diagnostic_summary).split())[:_MAX_SUMMARY]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "taxonomy_version": TAXONOMY_VERSION,
        "historical": bool(historical),
        "outcome": "success" if failure_class is FailureClass.SUCCESS else "failure",
        "failure_class": failure_class.value,
        "phase": signal.phase.value,
        "experiment_started": signal.experiment_started,
        "optimizer_steps_completed": signal.optimizer_steps_completed,
        "return_code": signal.return_code,
        "source": {
            "git_sha": source_sha,
            "workflow": workflow,
            "run_id": None if run_id is None else str(run_id),
        },
        "diagnostics": {
            "codes": codes,
            "summary": summary,
            "command_sha256": command_sha256,
            "executable": safe_name(executable) if executable else None,
            "stdout": dict(stdout_digest or {}),
            "stderr": dict(stderr_digest or {}),
            "raw_output_retained": False,
            "environment_retained": False,
        },
    }
    report["report_sha256"] = _report_hash(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected failure-report schema")
    try:
        FailureClass(str(report["failure_class"]))
        FailurePhase(str(report["phase"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid taxonomy value") from exc
    started = report.get("experiment_started")
    steps = report.get("optimizer_steps_completed")
    if not isinstance(started, bool):
        raise ValueError("experiment_started must be bool")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("optimizer_steps_completed must be non-negative int")
    if started != (steps > 0):
        raise ValueError("experiment_started must match optimizer_steps_completed > 0")
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be an object")
    if diagnostics.get("raw_output_retained") is not False:
        raise ValueError("raw output retention is forbidden")
    if diagnostics.get("environment_retained") is not False:
        raise ValueError("environment retention is forbidden")
    expected = _report_hash(report)
    if report.get("report_sha256") != expected:
        raise ValueError("failure-report hash mismatch")


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    _atomic_json(Path(path), report)


def _report_hash(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
