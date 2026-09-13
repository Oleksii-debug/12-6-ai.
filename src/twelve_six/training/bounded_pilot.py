"""Durable, recipe-bound facade for the learned-20M bounded pilot.

The optimizer-boundary implementation that was independently driven green lives
byte-for-byte in :mod:`bounded_pilot_core`.  This facade composes two existing
canonical authorities instead of defining replacements:

* TRAIN39 ``RecoveryStore`` for durable run/attempt state; and
* LEARN-345 ``learned20m_recipe`` for the frozen training recipe.

No object in this module authorizes training by itself.  The caller must supply
independently rooted packet, recipe-session, policy, run-manifest and D04/D10
inputs before an optimizer effect can occur.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from twelve_six.learned20m_recipe import (
    SESSION_SCHEMA,
    RecipeValidationError,
    identity_sha256,
    validate_policy,
)

from . import bounded_pilot_core as _core
from .resilience import FailureClass, RecoveryStateError, RecoveryStore, RunPhase

BoundedPilotAuthorizationError = _core.BoundedPilotAuthorizationError
BoundedPilotRecoveryRequiredError = _core.BoundedPilotRecoveryRequiredError

_RUN_BINDING_SCHEMA = "12-6.bounded-pilot-run-attempt-binding.v1"
_SESSION_KEYS = frozenset(
    {
        "schema",
        "status",
        "policy_identity_sha256",
        "bindings_identity_sha256",
        "trusted_authorities_identity_sha256",
        "session_identity_sha256",
        "qualified_runtime_unique_loss_positions",
        "training_recipe_status",
        "training_authorized",
        "compute_authorized",
        "authorized_optimized_targets",
        "optimizer_updates_executed",
    }
)


@dataclass(frozen=True, slots=True)
class BoundedPilotStepReceipt:
    """One optimizer receipt rooted in run-attempt and frozen recipe authority."""

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
    training_config_sha256: str
    training_recipe_policy_identity_sha256: str

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
            "training_config_sha256": self.training_config_sha256,
            "training_recipe_policy_identity_sha256": (
                self.training_recipe_policy_identity_sha256
            ),
        }


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


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: {label} must be a positive integer"
        )
    return value


def build_bounded_start_projection(
    *,
    packet_sha256: str,
    modelspec_sha256: str,
    initspec_sha256: str,
    ledger_identity_sha256: str,
    loss_bearing_manifest_identity_sha256: str,
    exposure_plan_identity_sha256: str,
    exposure_state_identity_sha256: str,
    authorized_unique_loss_positions: int,
    training_config_sha256: str,
    training_recipe_policy_identity_sha256: str,
) -> dict[str, Any]:
    """Build the exact immutable FRESH_START projection stored by TRAIN39."""

    return {
        "schema_version": _RUN_BINDING_SCHEMA,
        "mode": "FRESH_START",
        "packet_sha256": _require_sha256(packet_sha256, label="packet root"),
        "modelspec_sha256": _require_sha256(modelspec_sha256, label="ModelSpec root"),
        "initspec_sha256": _require_sha256(initspec_sha256, label="InitSpec root"),
        "ledger_identity_sha256": _require_sha256(
            ledger_identity_sha256,
            label="D04 ledger root",
        ),
        "loss_bearing_manifest_identity_sha256": _require_sha256(
            loss_bearing_manifest_identity_sha256,
            label="loss-bearing manifest root",
        ),
        "exposure_plan_identity_sha256": _require_sha256(
            exposure_plan_identity_sha256,
            label="exposure-plan root",
        ),
        "initial_exposure_state_identity_sha256": _require_sha256(
            exposure_state_identity_sha256,
            label="initial D04 exposure-state root",
        ),
        "initial_optimizer_step": 0,
        "initial_nonignored_target_count": 0,
        "authorized_unique_loss_positions": _positive_int(
            authorized_unique_loss_positions,
            label="authorized unique-loss positions",
        ),
        "training_config_sha256": _require_sha256(
            training_config_sha256,
            label="LEARN-345 qualified-session root",
        ),
        "training_recipe_policy_identity_sha256": _require_sha256(
            training_recipe_policy_identity_sha256,
            label="LEARN-345 policy root",
        ),
    }


class BoundedPilotStepRunner(_core.BoundedPilotStepRunner):
    """Core bounded pilot plus durable attempt and frozen LEARN-345 recipe gates."""

    def __init__(
        self,
        *args: Any,
        recovery_store: RecoveryStore,
        expected_run_manifest_sha256: str,
        expected_run_id: str,
        training_recipe_policy: Mapping[str, Any],
        training_recipe_session: Mapping[str, Any],
        expected_training_config_sha256: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

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
        self._expected_training_config_sha256 = _require_sha256(
            expected_training_config_sha256,
            label="external LEARN-345 qualified-session root",
        )
        self._attempt_started = False
        self._attempt = 0
        self._attempt_state_sha256 = ""
        try:
            self._training_recipe_policy = self._validate_training_recipe_authority(
                training_recipe_policy,
                training_recipe_session,
            )
            self._training_recipe = self._training_recipe_policy["recipe"]
            self._training_recipe_policy_identity_sha256 = _require_sha256(
                self._training_recipe_policy["policy_identity_sha256"],
                label="LEARN-345 policy root",
            )
            self._require_live_recipe_matches_authority()
            self._validate_durable_run_binding()
        except Exception:
            self.close()
            raise

    def _validate_training_recipe_authority(
        self,
        policy: Mapping[str, Any],
        session: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            validated_policy = validate_policy(dict(policy))
        except (RecipeValidationError, TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: canonical LEARN-345 policy rejected: {exc}"
            ) from exc

        if not isinstance(session, Mapping) or set(session) != set(_SESSION_KEYS):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 qualified session is not closed-world"
            )
        value = dict(session)
        if value.get("schema") != SESSION_SCHEMA or value.get("status") != "QUALIFIED_RECIPE_ONLY":
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 qualified recipe session required"
            )
        if value.get("training_recipe_status") != "QUALIFIED":
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 recipe session is not qualified"
            )
        for key in ("training_authorized", "compute_authorized"):
            if value.get(key) is not False:
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: LEARN-345 session {key} widened"
                )
        for key in ("authorized_optimized_targets", "optimizer_updates_executed"):
            if value.get(key) != 0 or isinstance(value.get(key), bool):
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: LEARN-345 session {key} must remain zero"
                )
        for key in (
            "policy_identity_sha256",
            "bindings_identity_sha256",
            "trusted_authorities_identity_sha256",
            "session_identity_sha256",
        ):
            _require_sha256(value.get(key), label=f"LEARN-345 session {key}")

        session_core = {
            key: item for key, item in value.items() if key != "session_identity_sha256"
        }
        observed_session_root = identity_sha256(session_core)
        if value["session_identity_sha256"] != observed_session_root:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 qualified-session identity drift"
            )
        packet_recipe = self.packet.get("recipe")
        if not isinstance(packet_recipe, Mapping):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: packet recipe section missing"
            )
        packet_config_root = _require_sha256(
            packet_recipe.get("training_config_sha256"),
            label="packet LEARN-345 qualified-session root",
        )
        if (
            observed_session_root != self._expected_training_config_sha256
            or packet_config_root != self._expected_training_config_sha256
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 qualified-session root differs from packet/external authority"
            )
        if value["policy_identity_sha256"] != validated_policy["policy_identity_sha256"]:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 session points to a different policy"
            )
        qualified_budget = _positive_int(
            value.get("qualified_runtime_unique_loss_positions"),
            label="qualified runtime unique-loss positions",
        )
        if (
            qualified_budget != packet_recipe.get("target_unique_loss_positions")
            or qualified_budget != self.replay_guard.authorized_budget
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 qualified budget differs from packet/D04"
            )
        return validated_policy

    def _require_live_recipe_matches_authority(self) -> None:
        recipe = self._training_recipe
        config = self.trainer.config
        actual = {
            "optimizer": self.trainer.optimizer.__class__.__name__,
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "scheduler": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "precision": config.precision,
        }
        expected = {
            key: recipe[key]
            for key in (
                "optimizer",
                "learning_rate",
                "betas",
                "eps",
                "weight_decay",
                "gradient_clip_norm",
                "scheduler",
                "warmup_steps",
                "gradient_accumulation_steps",
                "precision",
            )
        }
        if identity_sha256(actual) != identity_sha256(expected):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live TrainerConfig differs from frozen LEARN-345 recipe"
            )
        seed_vector = recipe.get("seed_vector")
        if (
            not isinstance(seed_vector, Mapping)
            or set(seed_vector) != {"model_init", "data_order", "dataloader"}
            or any(value != config.seed for value in seed_vector.values())
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live Trainer seed differs from frozen LEARN-345 seed vector"
            )
        if config.deterministic_algorithms is not True or config.deterministic_warn_only is not False:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: deterministic Trainer semantics differ from LEARN-345 runtime"
            )
        model_max_seq_len = getattr(getattr(self.trainer.model, "spec", None), "max_seq_len", None)
        if model_max_seq_len != recipe.get("sequence_length"):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live model sequence length differs from frozen LEARN-345 recipe"
            )
        batches = self.exposure_plan.get("batches")
        if not isinstance(batches, list) or config.max_steps != len(batches):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: Trainer max_steps differs from exact D04 exposure plan"
            )
        planned_targets = 0
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: D04 exposure-plan batch malformed"
                )
            planned_targets += _positive_int(
                batch.get("actual_nonignored_targets"),
                label="D04 planned nonignored targets",
            )
        if planned_targets != self.replay_guard.authorized_budget:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan target sum differs from LEARN-345 qualified budget"
            )

    def _require_live_batch_recipe_semantics(self, batch: Mapping[str, Any]) -> None:
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: LEARN-345 batch input_ids must be rank-2"
            )
        if input_ids.shape[0] != self._training_recipe["micro_batch_size"]:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live micro-batch size differs from frozen LEARN-345 recipe"
            )
        if input_ids.shape[1] > self._training_recipe["sequence_length"]:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live sequence exceeds frozen LEARN-345 sequence length"
            )

    def _bounded_start_projection(self) -> dict[str, Any]:
        state = self.replay_guard.state_dict()
        return build_bounded_start_projection(
            packet_sha256=self._packet_sha256,
            modelspec_sha256=self._modelspec_sha256,
            initspec_sha256=self._initspec_sha256,
            ledger_identity_sha256=self.replay_guard.ledger_identity_sha256,
            loss_bearing_manifest_identity_sha256=self._manifest_root,
            exposure_plan_identity_sha256=self.expected_plan_identity_sha256,
            exposure_state_identity_sha256=_require_sha256(
                state.get("state_identity_sha256"),
                label="initial D04 exposure-state root",
            ),
            authorized_unique_loss_positions=self.replay_guard.authorized_budget,
            training_config_sha256=self._expected_training_config_sha256,
            training_recipe_policy_identity_sha256=(
                self._training_recipe_policy_identity_sha256
            ),
        )

    def _validate_durable_run_binding(self) -> None:
        if self.recovery_store.run_id != self._expected_run_id:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run_id differs from external authority"
            )
        if self.recovery_store.run_manifest_sha256 != self._expected_run_manifest_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run manifest differs from external authority"
            )
        bounded = self.recovery_store.run_manifest.get("bounded_pilot")
        if not isinstance(bounded, Mapping) or dict(bounded) != self._bounded_start_projection():
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: durable run manifest does not bind exact bounded-pilot start state"
            )

    def _require_live_execution_chain(self, *, expected_inflight_targets: int = 0) -> None:
        super()._require_live_execution_chain(
            expected_inflight_targets=expected_inflight_targets,
        )
        self._require_live_recipe_matches_authority()

    def _preflight_handoff(
        self,
        *,
        batch: Mapping[str, Tensor],
        batch_index: int,
        expected_identity: str,
        actual_targets: int,
        expected_inflight_targets: int = 0,
    ) -> str:
        self._require_live_batch_recipe_semantics(batch)
        return super()._preflight_handoff(
            batch=batch,
            batch_index=batch_index,
            expected_identity=expected_identity,
            actual_targets=actual_targets,
            expected_inflight_targets=expected_inflight_targets,
        )

    def _ensure_durable_attempt(self) -> None:
        if self._attempt_started:
            try:
                state = self.recovery_store.open()
            except (OSError, ValueError, RecoveryStateError) as exc:
                self._poison(f"durable attempt state cannot be verified: {exc}")
                self._raise_recovery_required()
            if state.get("attempt") != self._attempt or state.get("phase") != RunPhase.RUNNING.value:
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
            # The immutable attempt-start marker is already durable. If failure
            # journaling fails, a fresh process still sees nonzero attempt history
            # and must refuse FRESH_START rather than replaying exposure.
            pass

    def train_authorized_microbatch(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, BoundedPilotStepReceipt]:
        self._require_live_recipe_matches_authority()
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
            training_config_sha256=self._expected_training_config_sha256,
            training_recipe_policy_identity_sha256=(
                self._training_recipe_policy_identity_sha256
            ),
        )


__all__ = [
    "BoundedPilotAuthorizationError",
    "BoundedPilotRecoveryRequiredError",
    "BoundedPilotStepReceipt",
    "BoundedPilotStepRunner",
    "build_bounded_start_projection",
]
