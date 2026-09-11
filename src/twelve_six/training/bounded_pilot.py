"""Fail-closed execution gate for the learned-20M bounded LOCAL_FREE pilot.

This module deliberately does not own readiness, corpus construction, recipe
selection, checkpoint formats, or evaluation.  It consumes the canonical
portable run binding plus D04 ordered-exposure authority and inserts the final
scientific authorization check immediately before ``optimizer.step()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from twelve_six.data.deterministic_exposure_order import (
    authorize_ordered_batch,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.portable_run_binding import PortableRunBinding

from .single_gpu import SingleDeviceStepMetrics, SingleDeviceStepRunner
from .trainer import Trainer

Batch = Mapping[str, Tensor]


class BoundedPilotAuthorizationError(RuntimeError):
    """Raised when a learned-target optimizer update is not exactly authorized."""


@dataclass(frozen=True, slots=True)
class BoundedPilotStepReceipt:
    """Text-free execution evidence for one optimizer transition."""

    optimizer_step_before: int
    optimizer_step_after: int
    actual_nonignored_targets: int
    exposure_identity_sha256: str
    consumed_loss_positions_after: int
    packet_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "actual_nonignored_targets": self.actual_nonignored_targets,
            "exposure_identity_sha256": self.exposure_identity_sha256,
            "consumed_loss_positions_after": self.consumed_loss_positions_after,
            "packet_sha256": self.packet_sha256,
        }


def _actual_nonignored_targets(batch: Batch) -> int:
    """Mirror Trainer token accounting without exposing payload text."""
    if "input_ids" not in batch:
        raise BoundedPilotAuthorizationError("batch is missing input_ids")
    if "labels" in batch and "target_ids" in batch:
        raise BoundedPilotAuthorizationError(
            "batch must not contain both labels and target_ids"
        )

    aligned_targets = "target_ids" in batch
    targets = batch.get("target_ids", batch.get("labels", batch["input_ids"]))
    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise BoundedPilotAuthorizationError("training targets must be a rank-2 tensor")

    if aligned_targets:
        valid = targets.ne(-100)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            if not isinstance(loss_mask, Tensor) or loss_mask.shape != targets.shape:
                raise BoundedPilotAuthorizationError(
                    "loss_mask must be a tensor matching target_ids"
                )
            valid = valid & loss_mask.bool()
        return int(valid.sum().item())

    if targets.shape[1] < 2:
        return 0
    return int(targets[:, 1:].ne(-100).sum().item())


def _require_local_free_packet(binding: PortableRunBinding) -> dict[str, Any]:
    if not binding.binding_ready or binding.packet is None or binding.packet_sha256 is None:
        blockers = ",".join(binding.blockers) if binding.blockers else "binding_not_ready"
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: portable run binding is not runnable: {blockers}"
        )

    packet = binding.packet
    resource = packet.get("resource")
    truth = packet.get("truth_boundary")
    identities = packet.get("identities")
    recipe = packet.get("recipe")
    if not all(isinstance(value, dict) for value in (resource, truth, identities, recipe)):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: runnable packet sections are malformed"
        )

    assert isinstance(resource, dict)
    assert isinstance(truth, dict)
    assert isinstance(identities, dict)
    assert isinstance(recipe, dict)
    if resource.get("resource_class") != "LOCAL_FREE":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: this runtime accepts LOCAL_FREE only"
        )
    if resource.get("maximum_cost_usd") != 0 or resource.get("materially_paid") is not False:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: materially paid compute is forbidden"
        )
    if truth.get("final_test_payload_accessed") is not False:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: final-test payload must remain sealed"
        )
    if identities.get("canonical_base") != "random_init":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: canonical Base must remain random_init"
        )

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
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: replay-free exposure budget is not exact"
        )
    return packet


class BoundedPilotStepRunner:
    """Authorize exact ordered exposure at the optimizer's final pre-step boundary.

    The current LEARN-345 recipe uses gradient_accumulation_steps=1.  This V6
    consumer rejects any other accumulation contract rather than authorizing a
    semantically different transition.  A later recipe change therefore requires
    an explicit audited runtime revision.
    """

    def __init__(
        self,
        runner: SingleDeviceStepRunner,
        *,
        binding: PortableRunBinding,
        replay_guard: IdentitySafeExposureReplayGuard,
        exposure_plan: Mapping[str, Any],
        expected_plan_identity_sha256: str,
    ) -> None:
        self.runner = runner
        self.trainer: Trainer = runner.trainer
        self.binding = binding
        self.packet = _require_local_free_packet(binding)
        self.replay_guard = replay_guard
        self.exposure_plan = exposure_plan
        self.expected_plan_identity_sha256 = expected_plan_identity_sha256
        self._pending: tuple[int, str, int] | None = None
        self._authorized_identity: str | None = None

        if self.trainer.config.gradient_accumulation_steps != 1:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: V6 requires gradient_accumulation_steps=1"
            )
        identities = self.packet["identities"]
        recipe = self.packet["recipe"]
        if identities.get("unique_loss_ledger_sha256") != replay_guard.ledger_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: packet and D04 ledger identities differ"
            )
        if recipe.get("maximum_total_exposures") != replay_guard.authorized_budget:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: packet and D04 authorized budgets differ"
            )
        if replay_guard.trainer_state_binding.get("optimizer_step") != self.trainer.optimizer_step:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: replay guard optimizer state is stale"
            )
        if (
            replay_guard.trainer_state_binding.get("trainer_nonignored_target_count")
            != replay_guard.consumed_loss_positions
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: replay guard target counter is incoherent"
            )

        register = getattr(self.trainer.optimizer, "register_step_pre_hook", None)
        if register is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer lacks a step pre-hook boundary"
            )
        self._hook_handle = register(self._authorize_immediately_before_optimizer_step)

    def close(self) -> None:
        """Remove the optimizer hook; safe to call more than once."""
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

    def _authorize_immediately_before_optimizer_step(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if self._pending is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer step attempted without D04 exposure handoff"
            )
        batch_index, expected_identity, actual_targets = self._pending
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
        observed = authorize_ordered_batch(
            self.replay_guard,
            self.exposure_plan,
            batch_index=batch_index,
            expected_plan_identity_sha256=self.expected_plan_identity_sha256,
            expected_ordered_next_exposure_identity_sha256=expected_identity,
        )
        plan_batch = self.exposure_plan.get("batches", [])[batch_index]
        if plan_batch.get("actual_nonignored_targets") != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan cardinality differs from Trainer batch"
            )
        self._authorized_identity = observed

    def train_authorized_microbatch(
        self,
        batch: Batch,
        *,
        batch_index: int,
        expected_next_exposure_identity_sha256: str,
    ) -> tuple[SingleDeviceStepMetrics, BoundedPilotStepReceipt]:
        """Run exactly one authorized optimizer transition on learned target data."""
        if self._pending is not None:
            raise BoundedPilotAuthorizationError("a prior authorization handoff is still pending")
        if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
            raise BoundedPilotAuthorizationError("batch_index must be a non-negative integer")
        batches = self.exposure_plan.get("batches")
        if not isinstance(batches, list) or batch_index >= len(batches):
            raise BoundedPilotAuthorizationError("batch_index is outside the exposure plan")

        actual_targets = _actual_nonignored_targets(batch)
        if actual_targets <= 0:
            raise BoundedPilotAuthorizationError("learned-target batch has no loss-bearing targets")
        plan_targets = batches[batch_index].get("actual_nonignored_targets")
        if plan_targets != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan cardinality differs from Trainer batch"
            )

        before_step = self.trainer.optimizer_step
        guard_snapshot = self.replay_guard.state_dict()
        guard_binding = dict(self.replay_guard.trainer_state_binding)
        self._pending = (batch_index, expected_next_exposure_identity_sha256, actual_targets)
        self._authorized_identity = None
        try:
            metrics = self.runner.train_microbatch(batch)
        except Exception:
            if self._authorized_identity is not None:
                self.replay_guard.load_state_dict(
                    guard_snapshot,
                    expected_trainer_state_binding=guard_binding,
                )
            raise
        finally:
            self._pending = None

        if not metrics.trainer.optimizer_stepped or self._authorized_identity is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: Trainer transition did not consume pre-step authorization"
            )
        if self.trainer.optimizer_step != before_step + 1:
            raise BoundedPilotAuthorizationError(
                "optimizer step counter did not advance exactly once"
            )

        receipt = BoundedPilotStepReceipt(
            optimizer_step_before=before_step,
            optimizer_step_after=self.trainer.optimizer_step,
            actual_nonignored_targets=actual_targets,
            exposure_identity_sha256=self._authorized_identity,
            consumed_loss_positions_after=self.replay_guard.consumed_loss_positions,
            packet_sha256=self.binding.packet_sha256,
        )
        return metrics, receipt
