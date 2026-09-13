"""Durable run/attempt facade for the learned-20M bounded pilot.

The already-qualified optimizer-boundary implementation lives byte-for-byte in
``bounded_pilot_core``.  This facade composes the incumbent TRAIN39
``RecoveryStore`` so a FRESH_START execution cannot be replayed from stale zero
state after process loss.  No second recovery/checkpoint framework is defined
here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import bounded_pilot_core as _core
from .resilience import FailureClass, RecoveryStateError, RecoveryStore, RunPhase

BoundedPilotAuthorizationError = _core.BoundedPilotAuthorizationError
BoundedPilotRecoveryRequiredError = _core.BoundedPilotRecoveryRequiredError

_RUN_BINDING_SCHEMA = "12-6.bounded-pilot-run-attempt-binding.v1"


@dataclass(frozen=True, slots=True)
class BoundedPilotStepReceipt:
    """One optimizer receipt additionally rooted in the durable run attempt."""

    optimizer_step_before: int
    optimizer_step_after: int
    actual_nonignored_targets: int
    exposure_identity_sha256: str
    consumed_loss_positions_after: int
    packet_sha256: str
    modelspec_sha256: str
    initspec_sha256: str
    run_id: str
    attempt: int
    run_manifest_sha256: str
    attempt_state_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "actual_nonignored_targets": self.actual_nonignored_targets,
            "exposure_identity_sha256": self.exposure_identity_sha256,
            "consumed_loss_positions_after": self.consumed_loss_positions_after,
            "packet_sha256": self.packet_sha256,
            "modelspec_sha256": self.modelspec_sha256,
            "initspec_sha256": self.initspec_sha256,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "run_manifest_sha256": self.run_manifest_sha256,
            "attempt_state_sha256": self.attempt_state_sha256,
        }


_TestRecoveryStoreFactory = Callable[
    ["BoundedPilotStepRunner"], tuple[RecoveryStore, str, str]
]
_TEST_RECOVERY_STORE_FACTORY: _TestRecoveryStoreFactory | None = None


def _set_test_recovery_store_factory(
    factory: _TestRecoveryStoreFactory | None,
) -> _TestRecoveryStoreFactory | None:
    """Install a test-only dependency factory; production must pass authority explicitly."""
    global _TEST_RECOVERY_STORE_FACTORY
    previous = _TEST_RECOVERY_STORE_FACTORY
    _TEST_RECOVERY_STORE_FACTORY = factory
    return previous


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: {label} is not canonical sha256"
        )
    return value


def _require_run_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: {label} must be non-empty canonical text"
        )
    return value


class BoundedPilotStepRunner(_core.BoundedPilotStepRunner):
    """Core bounded pilot plus a durable TRAIN39 run/attempt replay barrier."""

    def __init__(
        self,
        *args: Any,
        recovery_store: RecoveryStore | None = None,
        expected_run_manifest_sha256: str | None = None,
        expected_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if recovery_store is None:
            factory = _TEST_RECOVERY_STORE_FACTORY
            if factory is None:
                self.close()
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: durable TRAIN39 run/attempt authority is required"
                )
            recovery_store, expected_run_manifest_sha256, expected_run_id = factory(self)

        if not isinstance(recovery_store, RecoveryStore):
            self.close()
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: canonical TRAIN39 RecoveryStore required"
            )
        self.recovery_store = recovery_store
        self._expected_run_manifest_sha256 = _require_sha256(
            expected_run_manifest_sha256,
            label="external run-manifest root",
        )
        self._expected_run_id = _require_run_id(
            expected_run_id,
            label="external run_id",
        )
        self._attempt_started = False
        self._attempt = 0
        self._attempt_state_sha256 = ""
        try:
            self._validate_durable_run_binding()
        except Exception:
            self.close()
            raise

    def _bounded_start_projection(self) -> dict[str, Any]:
        state = self.replay_guard.state_dict()
        state_identity = _require_sha256(
            state.get("state_identity_sha256"),
            label="initial D04 exposure-state root",
        )
        return {
            "schema_version": _RUN_BINDING_SCHEMA,
            "mode": "FRESH_START",
            "packet_sha256": self._packet_sha256,
            "modelspec_sha256": self._modelspec_sha256,
            "initspec_sha256": self._initspec_sha256,
            "ledger_identity_sha256": self.replay_guard.ledger_identity_sha256,
            "loss_bearing_manifest_identity_sha256": self._manifest_root,
            "exposure_plan_identity_sha256": self.expected_plan_identity_sha256,
            "initial_exposure_state_identity_sha256": state_identity,
            "initial_optimizer_step": self.trainer.optimizer_step,
            "initial_nonignored_target_count": self.trainer.tokens_seen,
            "authorized_unique_loss_positions": self.replay_guard.authorized_budget,
        }

    def _validate_durable_run_binding(self) -> None:
        if self.recovery_store.run_id != self._expected_run_id:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run_id differs from external authority"
            )
        if self.recovery_store.run_manifest_sha256 != self._expected_run_manifest_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run manifest differs from external authority"
            )
        manifest = self.recovery_store.run_manifest
        bounded = manifest.get("bounded_pilot")
        expected = self._bounded_start_projection()
        if not isinstance(bounded, Mapping) or dict(bounded) != expected:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run manifest does not bind exact bounded-pilot start state"
            )

    def _ensure_durable_attempt(self) -> None:
        if self._attempt_started:
            try:
                state = self.recovery_store.open()
            except (OSError, ValueError, RecoveryStateError) as exc:
                self._poison(f"durable attempt state cannot be verified: {exc}")
                self._raise_recovery_required()
            if (
                state.get("attempt") != self._attempt
                or state.get("phase") != RunPhase.RUNNING.value
            ):
                self._poison("durable attempt is no longer uniquely RUNNING")
                self._raise_recovery_required()
            return

        try:
            state = self.recovery_store.open()
        except (OSError, ValueError, RecoveryStateError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: durable recovery journal rejected: {exc}"
            ) from exc
        if (
            state.get("phase") != RunPhase.PREPARED.value
            or state.get("attempt") != 0
            or state.get("failure_count") != 0
            or state.get("checkpoint_count") != 0
        ):
            raise BoundedPilotRecoveryRequiredError(
                "BLOCKED_RECOVERY_REQUIRED: FRESH_START run already has durable attempt/history; "
                "verified D05 RESUME authority is required before any further optimizer effect"
            )
        try:
            started = self.recovery_store.begin_attempt()
        except (OSError, ValueError, RecoveryStateError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: durable attempt start rejected: {exc}"
            ) from exc
        if started.get("attempt") != 1 or started.get("phase") != RunPhase.RUNNING.value:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: canonical first durable attempt was not established"
            )
        self._attempt_state_sha256 = _require_sha256(
            started.get("state_sha256"),
            label="durable attempt state root",
        )
        self._attempt = 1
        self._attempt_started = True

    def _record_durable_poison(self) -> None:
        try:
            self.recovery_store.record_failure(
                FailureClass.TRAINER_POISONED,
                optimizer_step=self.trainer.optimizer_step,
                detail_code="BOUNDED_PILOT_AMBIGUOUS_OR_POST_COMMIT_FAILURE",
            )
        except (OSError, ValueError, RecoveryStateError):
            # The immutable attempt-start marker is already durable.  If failure
            # journaling itself fails, a fresh process still sees nonzero attempt
            # history and must refuse FRESH_START rather than replaying exposure.
            pass

    def train_authorized_microbatch(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, BoundedPilotStepReceipt]:
        self._ensure_durable_attempt()
        try:
            metrics, core_receipt = super().train_authorized_microbatch(*args, **kwargs)
        except BoundedPilotRecoveryRequiredError:
            self._record_durable_poison()
            raise

        if self.trainer.optimizer_step >= self.trainer.config.max_steps:
            try:
                self.recovery_store.mark_completed()
            except (OSError, ValueError, RecoveryStateError) as exc:
                self._poison(f"durable terminal marker failed after optimizer commit: {exc}")
                self._record_durable_poison()
                self._raise_recovery_required()

        payload = core_receipt.as_dict()
        return metrics, BoundedPilotStepReceipt(
            optimizer_step_before=payload["optimizer_step_before"],
            optimizer_step_after=payload["optimizer_step_after"],
            actual_nonignored_targets=payload["actual_nonignored_targets"],
            exposure_identity_sha256=payload["exposure_identity_sha256"],
            consumed_loss_positions_after=payload["consumed_loss_positions_after"],
            packet_sha256=payload["packet_sha256"],
            modelspec_sha256=payload["modelspec_sha256"],
            initspec_sha256=payload["initspec_sha256"],
            run_id=self._expected_run_id,
            attempt=self._attempt,
            run_manifest_sha256=self._expected_run_manifest_sha256,
            attempt_state_sha256=self._attempt_state_sha256,
        )


__all__ = [
    "BoundedPilotAuthorizationError",
    "BoundedPilotRecoveryRequiredError",
    "BoundedPilotStepReceipt",
    "BoundedPilotStepRunner",
]
