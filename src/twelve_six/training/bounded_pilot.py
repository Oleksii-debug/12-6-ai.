"""Fail-closed execution gate for the learned-20M bounded LOCAL_FREE pilot.

The gate consumes canonical readiness/run-packet and D04 ordered-exposure
contracts. It does not authorize training itself. Its job is to prove that the
*actual* model/Trainer about to update match the authorized packet and to consume
the exact next D04 exposure identity immediately before ``optimizer.step()``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from twelve_six.data.deterministic_exposure_order import (
    authorize_ordered_batch,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256

from .single_gpu import SingleDeviceStepMetrics, SingleDeviceStepRunner
from .trainer import Trainer

Batch = Mapping[str, Tensor]


class BoundedPilotAuthorizationError(RuntimeError):
    """Raised when a learned-target optimizer update is not exactly authorized."""


class BoundedPilotRecoveryRequiredError(BoundedPilotAuthorizationError):
    """Raised once execution may have crossed a learned-state commit boundary."""


@dataclass(frozen=True, slots=True)
class BoundedPilotStepReceipt:
    """Text-free evidence for one exact authorized optimizer transition."""

    optimizer_step_before: int
    optimizer_step_after: int
    actual_nonignored_targets: int
    exposure_identity_sha256: str
    consumed_loss_positions_after: int
    packet_sha256: str
    modelspec_sha256: str
    initspec_sha256: str

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
        }


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoundedPilotAuthorizationError(f"BLOCKED_PRE_STEP_1: {label} is not canonical sha256")
    return value


def _actual_nonignored_targets(batch: Batch) -> int:
    if "input_ids" not in batch:
        raise BoundedPilotAuthorizationError("batch is missing input_ids")
    if "labels" in batch and "target_ids" in batch:
        raise BoundedPilotAuthorizationError("batch must not contain both labels and target_ids")
    aligned = "target_ids" in batch
    targets = batch.get("target_ids", batch.get("labels", batch["input_ids"]))
    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise BoundedPilotAuthorizationError("training targets must be a rank-2 tensor")
    if aligned:
        valid = targets.ne(-100)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            if not isinstance(loss_mask, Tensor) or loss_mask.shape != targets.shape:
                raise BoundedPilotAuthorizationError("loss_mask must match target_ids")
            valid = valid & loss_mask.bool()
        return int(valid.sum().item())
    return 0 if targets.shape[1] < 2 else int(targets[:, 1:].ne(-100).sum().item())


def _require_local_free_packet(
    binding: PortableRunBinding,
    *,
    expected_packet_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected_root = _require_sha256(
        expected_packet_sha256,
        label="external packet root",
    )
    if (
        not binding.binding_ready
        or not binding.readiness_ready
        or not binding.overlay_contract_valid
        or not binding.packet_contract_valid
        or binding.packet is None
        or binding.packet_sha256 is None
    ):
        blockers = ",".join(binding.blockers) if binding.blockers else "binding_not_ready"
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: portable run binding is not runnable: {blockers}"
        )

    binding_root = _require_sha256(binding.packet_sha256, label="binding packet root")
    packet = copy.deepcopy(binding.packet)
    observed_root = canonical_sha256(packet)
    if observed_root != binding_root:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet content differs from binding packet root"
        )
    if observed_root != expected_root:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet root differs from external authority"
        )

    resource = packet.get("resource")
    truth = packet.get("truth_boundary")
    identities = packet.get("identities")
    recipe = packet.get("recipe")
    if not all(isinstance(value, dict) for value in (resource, truth, identities, recipe)):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: packet sections malformed")
    assert isinstance(resource, dict) and isinstance(truth, dict)
    assert isinstance(identities, dict) and isinstance(recipe, dict)
    if resource.get("resource_class") != "LOCAL_FREE":
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: LOCAL_FREE required")
    if resource.get("maximum_cost_usd") != 0 or resource.get("materially_paid") is not False:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: paid compute forbidden")
    if truth.get("final_test_payload_accessed") is not False:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: final test must stay sealed")
    if identities.get("canonical_base") != "random_init":
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: Base must be random_init")
    target = recipe.get("target_unique_loss_positions")
    maximum = recipe.get("maximum_total_exposures")
    available = recipe.get("available_unique_loss_positions")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (target, maximum, available)
    ):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimized-target authority must be positive"
        )
    if maximum != target or recipe.get("max_exposures_per_unique_position") != 1:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: exposure budget not exact")
    return packet, observed_root


def _actual_model_identities(trainer: Trainer) -> tuple[str, str]:
    model = trainer.model
    spec = getattr(model, "spec", None)
    init_spec = getattr(model, "init_spec", None)
    spec_identity = getattr(spec, "identity_sha256", None)
    init_identity = getattr(init_spec, "identity_sha256", None)
    if not callable(spec_identity) or not callable(init_identity):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: actual model does not expose ModelSpec/InitSpec identities"
        )
    model_sha = spec_identity()
    init_sha = init_identity()
    if not isinstance(model_sha, str) or not isinstance(init_sha, str):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: model identities malformed")
    parameter_count = getattr(spec, "parameter_count", None)
    if not callable(parameter_count):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: ModelSpec lacks parameter_count")
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count() != actual_parameters:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual parameter count drift")
    return model_sha, init_sha


def _actual_overlay_projection(trainer: Trainer) -> dict[str, Any]:
    return {
        "optimizer": trainer.optimizer.__class__.__name__,
        "scheduler": trainer.config.scheduler,
        "precision": trainer.config.precision,
    }


def _require_optimizer_parameter_coverage(trainer: Trainer) -> None:
    model_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    optimizer_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimizer contains duplicate trainable parameters"
        )
    if set(optimizer_ids) != {id(parameter) for parameter in model_parameters}:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimizer parameter coverage differs from model"
        )


def _require_live_optimizer_matches_config(trainer: Trainer) -> None:
    _require_optimizer_parameter_coverage(trainer)
    if not trainer.optimizer.param_groups:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: optimizer has no param groups")
    expected = {
        "lr": trainer.config.learning_rate,
        "betas": trainer.config.betas,
        "eps": trainer.config.eps,
        "weight_decay": trainer.config.weight_decay,
    }
    for group_index, group in enumerate(trainer.optimizer.param_groups):
        for key, expected_value in expected.items():
            if key not in group:
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: optimizer group {group_index} lacks {key}"
                )
            actual_value = group[key]
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: live optimizer hyperparameters differ "
                    f"from TrainerConfig ({key})"
                )


def _require_actual_execution_matches_packet(
    trainer: Trainer,
    packet: Mapping[str, Any],
) -> tuple[str, str]:
    identities = packet["identities"]
    recipe = packet["recipe"]
    model_sha, init_sha = _actual_model_identities(trainer)
    if identities.get("modelspec_sha256") != model_sha:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual ModelSpec differs")
    if identities.get("initspec_sha256") != init_sha:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual InitSpec differs")

    expected_overlay = recipe.get("optimizer_scheduler_precision")
    actual_overlay = _actual_overlay_projection(trainer)
    if not isinstance(expected_overlay, dict) or set(expected_overlay) != set(actual_overlay):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet lacks canonical optimizer/scheduler/precision overlay"
        )
    if expected_overlay != actual_overlay:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: actual optimizer/scheduler/precision differs from packet"
        )
    if recipe.get("seed") != trainer.config.seed:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: Trainer seed differs")
    if trainer.config.gradient_accumulation_steps != 1:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: bounded pilot requires one microbatch per optimizer step"
        )
    _require_live_optimizer_matches_config(trainer)
    return model_sha, init_sha


class BoundedPilotStepRunner:
    """Consume D04 authority at the final pre-optimizer-step boundary."""

    def __init__(
        self,
        runner: SingleDeviceStepRunner,
        *,
        binding: PortableRunBinding,
        expected_packet_sha256: str,
        replay_guard: IdentitySafeExposureReplayGuard,
        exposure_plan: Mapping[str, Any],
        expected_plan_identity_sha256: str,
    ) -> None:
        self.runner = runner
        self.trainer: Trainer = runner.trainer
        self.binding = binding
        self.packet, self._packet_sha256 = _require_local_free_packet(
            binding,
            expected_packet_sha256=expected_packet_sha256,
        )
        self.replay_guard = replay_guard
        self.exposure_plan = copy.deepcopy(exposure_plan)
        self.expected_plan_identity_sha256 = _require_sha256(
            expected_plan_identity_sha256,
            label="external D04 plan root",
        )
        if self.exposure_plan.get("plan_identity_sha256") != self.expected_plan_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan differs from external handoff"
            )
        self._pending: tuple[int, str, int] | None = None
        self._authorized_identity: str | None = None
        self._poisoned_reason: str | None = None
        self._modelspec_sha256, self._initspec_sha256 = _require_actual_execution_matches_packet(
            self.trainer,
            self.packet,
        )

        identities = self.packet["identities"]
        recipe = self.packet["recipe"]
        if identities.get("unique_loss_ledger_sha256") != replay_guard.ledger_identity_sha256:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: D04 ledger differs")
        if recipe.get("maximum_total_exposures") != replay_guard.authorized_budget:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: D04 budget differs")
        if replay_guard.trainer_state_binding.get("optimizer_step") != self.trainer.optimizer_step:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: replay optimizer state stale")
        if (
            replay_guard.trainer_state_binding.get("trainer_nonignored_target_count")
            != replay_guard.consumed_loss_positions
        ):
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: replay target counter stale")
        if replay_guard.consumed_loss_positions != self.trainer.tokens_seen:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: Trainer target counter differs from D04 replay state"
            )

        register = getattr(self.trainer.optimizer, "register_step_pre_hook", None)
        if register is None:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: no optimizer pre-hook")
        self._hook_handle = register(self._authorize_immediately_before_optimizer_step)

    @property
    def poisoned(self) -> bool:
        return self._poisoned_reason is not None

    def close(self) -> None:
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def _raise_recovery_required(self) -> None:
        reason = self._poisoned_reason or "learned-state commit status is ambiguous"
        raise BoundedPilotRecoveryRequiredError(
            "BLOCKED_RECOVERY_REQUIRED: fresh verified recovery required; " + reason
        )

    def _preflight_handoff(
        self,
        *,
        batch_index: int,
        expected_identity: str,
        actual_targets: int,
    ) -> None:
        observed = ordered_next_exposure_identity(
            self.replay_guard,
            self.exposure_plan,
            batch_index=batch_index,
            expected_plan_identity_sha256=self.expected_plan_identity_sha256,
        )
        if observed != expected_identity:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: next exposure identity differs from external handoff"
            )
        plan_batch = self.exposure_plan.get("batches", [])[batch_index]
        if plan_batch.get("actual_nonignored_targets") != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 cardinality differs from Trainer batch"
            )

    def _authorize_immediately_before_optimizer_step(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if self.poisoned:
            self._raise_recovery_required()
        if self._pending is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer step lacks D04 handoff"
            )
        _require_actual_execution_matches_packet(self.trainer, self.packet)
        batch_index, expected_identity, actual_targets = self._pending
        self._preflight_handoff(
            batch_index=batch_index,
            expected_identity=expected_identity,
            actual_targets=actual_targets,
        )
        self._authorized_identity = authorize_ordered_batch(
            self.replay_guard,
            self.exposure_plan,
            batch_index=batch_index,
            expected_plan_identity_sha256=self.expected_plan_identity_sha256,
            expected_ordered_next_exposure_identity_sha256=expected_identity,
        )

    def _bind_live_post_step_state(self) -> None:
        if self.trainer.tokens_seen != self.replay_guard.consumed_loss_positions:
            self._poison("Trainer/D04 target counters diverged after optimizer transition")
            self._raise_recovery_required()
        state = dict(self.replay_guard.trainer_state_binding)
        state["optimizer_step"] = self.trainer.optimizer_step
        state["trainer_nonignored_target_count"] = self.trainer.tokens_seen
        try:
            self.replay_guard.bind_checkpoint_state(state)
        except Exception as exc:
            self._poison(f"post-step D04 state binding failed: {exc}")
            self._raise_recovery_required()

    def train_authorized_microbatch(
        self,
        batch: Batch,
        *,
        batch_index: int,
        expected_next_exposure_identity_sha256: str,
    ) -> tuple[SingleDeviceStepMetrics, BoundedPilotStepReceipt]:
        if self.poisoned:
            self._raise_recovery_required()
        if self._pending is not None:
            raise BoundedPilotAuthorizationError("prior authorization handoff still pending")
        if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
            raise BoundedPilotAuthorizationError("batch_index must be non-negative integer")
        batches = self.exposure_plan.get("batches")
        if not isinstance(batches, list) or batch_index >= len(batches):
            raise BoundedPilotAuthorizationError("batch_index outside exposure plan")
        actual_targets = _actual_nonignored_targets(batch)
        planned_targets = batches[batch_index].get("actual_nonignored_targets")
        if actual_targets <= 0 or planned_targets != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: loss-bearing cardinality is not exact"
            )
        _require_actual_execution_matches_packet(self.trainer, self.packet)
        self._preflight_handoff(
            batch_index=batch_index,
            expected_identity=expected_next_exposure_identity_sha256,
            actual_targets=actual_targets,
        )

        before_step = self.trainer.optimizer_step
        self._pending = (batch_index, expected_next_exposure_identity_sha256, actual_targets)
        self._authorized_identity = None
        try:
            metrics = self.runner.train_microbatch(batch)
        except Exception as exc:
            if self._authorized_identity is None:
                reason = "Trainer execution failed after entering the transition boundary"
            else:
                reason = "Trainer execution failed after D04 authorization was consumed"
            self._poison(f"{reason}: {exc}")
            self._raise_recovery_required()
        finally:
            self._pending = None

        if not metrics.trainer.optimizer_stepped or self._authorized_identity is None:
            self._poison("Trainer returned without one consumed optimizer authorization")
            self._raise_recovery_required()
        if self.trainer.optimizer_step != before_step + 1:
            self._poison("optimizer step did not advance exactly once")
            self._raise_recovery_required()

        self._bind_live_post_step_state()
        receipt = BoundedPilotStepReceipt(
            optimizer_step_before=before_step,
            optimizer_step_after=self.trainer.optimizer_step,
            actual_nonignored_targets=actual_targets,
            exposure_identity_sha256=self._authorized_identity,
            consumed_loss_positions_after=self.replay_guard.consumed_loss_positions,
            packet_sha256=self._packet_sha256,
            modelspec_sha256=self._modelspec_sha256,
            initspec_sha256=self._initspec_sha256,
        )
        return metrics, receipt
