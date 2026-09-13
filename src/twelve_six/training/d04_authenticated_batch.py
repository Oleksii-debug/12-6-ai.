"""Authenticated D04 post-pack content -> bounded-pilot Batch composition.

This module closes the final producer seam between independently rooted D04
post-pack token content and the tensor Batch consumed by the bounded learned-20M
runtime.  It does not authorize training.  The caller must provide independent
expected materialization/ledger/plan/content-manifest roots; the source
reconstructs the canonical content manifest from the supplied post-pack token
IDs before it can produce a tensor batch.

The canonical runner subclass keeps exposure mutation at the optimizer pre-hook
and delegates that mutation to ``authorize_ordered_live_batch``.  Caller-authored
token tensors are deliberately not accepted by its public training entry point.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from twelve_six.data.deterministic_exposure_order import authorize_ordered_live_batch
from twelve_six.data.loss_bearing_content_binding_v1 import (
    build_loss_bearing_content_manifest,
    verify_live_loss_bearing_batch,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError

from . import bounded_pilot_core as _core
from .bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    BoundedPilotStepReceipt,
    BoundedPilotStepRunner,
)


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


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: {label} must be a non-negative integer"
        )
    return value


def _pack_tokens(materialization: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    packing = materialization.get("packing")
    if not isinstance(packing, Mapping):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D04 materialization packing is missing"
        )
    raw_packs = packing.get("packs")
    if not isinstance(raw_packs, Sequence) or isinstance(raw_packs, (str, bytes)):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D04 materialization packs are malformed"
        )
    result: dict[str, tuple[int, ...]] = {}
    for index, raw_pack in enumerate(raw_packs):
        if not isinstance(raw_pack, Mapping):
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: D04 pack[{index}] is malformed"
            )
        pack_id = raw_pack.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id or pack_id in result:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: D04 pack[{index}] identity is invalid"
            )
        raw_tokens = raw_pack.get("token_ids")
        if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes)):
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: D04 pack[{index}] token_ids are missing"
            )
        tokens: list[int] = []
        for token_index, token in enumerate(raw_tokens):
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: D04 token_ids must be non-negative integers "
                    f"(pack={index}, token={token_index})"
                )
            tokens.append(token)
        token_count = _nonnegative_int(
            raw_pack.get("token_count"),
            label=f"D04 pack[{index}] token_count",
        )
        if token_count != len(tokens):
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: D04 pack[{index}] token_count drift"
            )
        result[pack_id] = tuple(tokens)
    return result


class AuthenticatedD04BatchSource:
    """Deterministically derive verified tensor batches from authenticated D04 bytes."""

    def __init__(
        self,
        *,
        materialization: Mapping[str, Any],
        ledger: Mapping[str, Any],
        exposure_plan: Mapping[str, Any],
        loss_bearing_content_manifest: Mapping[str, Any],
        expected_materialization_identity_sha256: str,
        expected_ledger_identity_sha256: str,
        expected_plan_identity_sha256: str,
        expected_loss_bearing_manifest_identity_sha256: str,
    ) -> None:
        self.materialization_identity_sha256 = _require_sha256(
            expected_materialization_identity_sha256,
            label="external D04 materialization root",
        )
        self.ledger_identity_sha256 = _require_sha256(
            expected_ledger_identity_sha256,
            label="external D04 ledger root",
        )
        self.plan_identity_sha256 = _require_sha256(
            expected_plan_identity_sha256,
            label="external D04 exposure-plan root",
        )
        self.manifest_identity_sha256 = _require_sha256(
            expected_loss_bearing_manifest_identity_sha256,
            label="external D04 content-manifest root",
        )

        self._materialization = copy.deepcopy(dict(materialization))
        self._ledger = copy.deepcopy(dict(ledger))
        self._plan = copy.deepcopy(dict(exposure_plan))
        self._manifest = copy.deepcopy(dict(loss_bearing_content_manifest))

        try:
            rebuilt_manifest = build_loss_bearing_content_manifest(
                self._materialization,
                self._ledger,
                self._plan,
                expected_materialization_identity_sha256=(
                    self.materialization_identity_sha256
                ),
                expected_ledger_identity_sha256=self.ledger_identity_sha256,
                expected_plan_identity_sha256=self.plan_identity_sha256,
            )
        except (LedgerError, TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: authenticated D04 batch source rejected: {exc}"
            ) from exc

        if rebuilt_manifest.get("manifest_identity_sha256") != self.manifest_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 materialization/ledger/plan do not rebuild "
                "the externally authorized content manifest"
            )
        if rebuilt_manifest != self._manifest:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: supplied D04 content manifest differs from "
                "canonical reconstruction"
            )
        if self._materialization.get("materialization_identity_sha256") != (
            self.materialization_identity_sha256
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 materialization root differs from external authority"
            )
        if self._ledger.get("ledger_identity_sha256") != self.ledger_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 ledger root differs from external authority"
            )
        if self._plan.get("plan_identity_sha256") != self.plan_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan root differs from external authority"
            )

        self._tokens_by_pack = _pack_tokens(self._materialization)
        raw_batches = self._manifest.get("batches")
        if not isinstance(raw_batches, Sequence) or isinstance(raw_batches, (str, bytes)):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 content-manifest batches are malformed"
            )
        self._batch_count = len(raw_batches)

    @property
    def batch_count(self) -> int:
        return self._batch_count

    def build_batch(
        self,
        *,
        batch_index: int,
        device: torch.device | str,
    ) -> dict[str, Tensor]:
        """Return the exact aligned predictor/target tensor batch for one plan index."""
        index = _nonnegative_int(batch_index, label="D04 batch_index")
        raw_batches = self._manifest["batches"]
        if index >= len(raw_batches):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 batch_index is outside authenticated manifest"
            )
        manifest_batch = raw_batches[index]
        if not isinstance(manifest_batch, Mapping):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 content-manifest batch is malformed"
            )
        if manifest_batch.get("global_batch_index") != index:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 content-manifest batch order is stale"
            )
        raw_claims = manifest_batch.get("claims")
        if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 content-manifest claims are malformed"
            )
        if not raw_claims:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 authenticated batch has no loss-bearing claims"
            )

        input_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        row_width: int | None = None
        for claim_index, raw_claim in enumerate(raw_claims):
            if not isinstance(raw_claim, Mapping):
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: D04 claim[{claim_index}] is malformed"
                )
            pack_id = raw_claim.get("pack_id")
            if not isinstance(pack_id, str) or pack_id not in self._tokens_by_pack:
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: D04 claim[{claim_index}] pack is unavailable"
                )
            start = _nonnegative_int(
                raw_claim.get("pack_target_start"),
                label=f"D04 claim[{claim_index}] pack_target_start",
            )
            end = _nonnegative_int(
                raw_claim.get("pack_target_end"),
                label=f"D04 claim[{claim_index}] pack_target_end",
            )
            tokens = self._tokens_by_pack[pack_id]
            if start < 1 or start >= end or end > len(tokens):
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: D04 claim[{claim_index}] token slice is invalid"
                )
            inputs = list(tokens[start - 1 : end - 1])
            targets = list(tokens[start:end])
            if len(inputs) != len(targets) or not targets:
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: D04 claim[{claim_index}] geometry is invalid"
                )
            expected_targets = _nonnegative_int(
                raw_claim.get("target_count"),
                label=f"D04 claim[{claim_index}] target_count",
            )
            if expected_targets != len(targets):
                raise BoundedPilotAuthorizationError(
                    f"BLOCKED_PRE_STEP_1: D04 claim[{claim_index}] target count drift"
                )
            if row_width is None:
                row_width = len(targets)
            elif row_width != len(targets):
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: authenticated D04 claim rows have unequal widths"
                )
            input_rows.append(inputs)
            target_rows.append(targets)

        batch: dict[str, Tensor] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
            "target_ids": torch.tensor(target_rows, dtype=torch.long, device=device),
        }
        try:
            verify_live_loss_bearing_batch(
                self._manifest,
                expected_manifest_identity_sha256=self.manifest_identity_sha256,
                batch_index=index,
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                loss_mask=None,
                shifted=False,
            )
        except (LedgerError, TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: derived D04 tensor batch failed canonical verifier: {exc}"
            ) from exc
        return batch


class AuthenticatedD04BoundedPilotStepRunner(BoundedPilotStepRunner):
    """Canonical bounded runner whose training Batch can only come from D04 content."""

    def __init__(
        self,
        *args: Any,
        batch_source: AuthenticatedD04BatchSource,
        **kwargs: Any,
    ) -> None:
        if type(batch_source) is not AuthenticatedD04BatchSource:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: exact canonical AuthenticatedD04BatchSource required"
            )
        self.batch_source = batch_source
        super().__init__(*args, **kwargs)
        try:
            if batch_source.ledger_identity_sha256 != self.replay_guard.ledger_identity_sha256:
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: authenticated batch source ledger differs from runtime"
                )
            if batch_source.plan_identity_sha256 != self.expected_plan_identity_sha256:
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: authenticated batch source plan differs from runtime"
                )
            if batch_source.manifest_identity_sha256 != self._manifest_root:
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: authenticated batch source manifest differs from runtime"
                )
            if batch_source.batch_count != len(self.exposure_plan.get("batches", ())):
                raise BoundedPilotAuthorizationError(
                    "BLOCKED_PRE_STEP_1: authenticated batch source length differs from plan"
                )
        except Exception:
            self.close()
            raise

    def _authorize_immediately_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        """Consume D04 only at the optimizer pre-hook via current canonical helper."""
        if self.poisoned:
            self._raise_recovery_required()
        if self._pending is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer step lacks D04 handoff"
            )
        if optimizer is not self._optimizer_object or self.trainer.optimizer is not self._optimizer_object:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer object differs from authorized hook target"
            )
        pending = self._pending
        self._preflight_handoff(
            batch=pending.batch,
            batch_index=pending.batch_index,
            expected_identity=pending.expected_identity,
            actual_targets=pending.actual_targets,
            expected_inflight_targets=pending.actual_targets,
        )
        input_ids, targets, loss_mask, shifted = _core._batch_projection(pending.batch)
        try:
            self._authorized_identity = authorize_ordered_live_batch(
                self.replay_guard,
                self.exposure_plan,
                batch_index=pending.batch_index,
                expected_plan_identity_sha256=self.expected_plan_identity_sha256,
                expected_ordered_next_exposure_identity_sha256=pending.expected_identity,
                input_ids=input_ids,
                target_ids=targets,
                loss_mask=loss_mask,
                shifted=shifted,
            )
        except (LedgerError, TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: canonical ordered live D04 authorization rejected: {exc}"
            ) from exc

    def train_authorized_microbatch(self, *args: Any, **kwargs: Any):
        """Reject caller-authored token tensors on the canonical authenticated path."""
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: caller-authored Batch is forbidden on authenticated D04 runtime; "
            "use train_authenticated_batch"
        )

    def train_authenticated_batch(
        self,
        *,
        batch_index: int,
        expected_next_exposure_identity_sha256: str,
    ) -> tuple[Any, BoundedPilotStepReceipt]:
        """Derive the exact D04 Batch and execute one bounded optimizer transition."""
        if batch_index != self.replay_guard.claim_sequence:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: authenticated D04 batch_index is not the next exposure"
            )
        try:
            device = next(self.trainer.model.parameters()).device
        except StopIteration as exc:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: Trainer model has no parameters"
            ) from exc
        batch = self.batch_source.build_batch(batch_index=batch_index, device=device)
        return super().train_authorized_microbatch(
            batch,
            batch_index=batch_index,
            expected_next_exposure_identity_sha256=expected_next_exposure_identity_sha256,
        )


__all__ = [
    "AuthenticatedD04BatchSource",
    "AuthenticatedD04BoundedPilotStepRunner",
]
