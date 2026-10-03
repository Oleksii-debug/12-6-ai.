from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.unique_loss_ledger_v2 import ExposureReplayGuard, LedgerError

_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "position_policy",
        "materialization_identity_sha256",
        "stage_bindings",
        "tokenizer",
        "packing_identity_sha256",
        "complete_one_pass",
        "eligible_causal_targets_before_packing",
        "one_pass_unique_nonignored_causal_loss_positions",
        "eligible_targets_not_packed",
        "by_language",
        "by_modality",
        "by_family",
        "segments",
        "padding_loss_positions",
        "cross_document_loss_positions",
        "source_bytes_relabelled_as_loss_positions",
        "ledger_identity_sha256",
    }
)
_CLAIM_KEYS = frozenset(
    {"segment_identity_sha256", "offset_start", "offset_end"}
)
_NEXT_EXPOSURE_SCHEMA = "12-6.next-exposure-identity.v1"
_CONTENT_MANIFEST_SCHEMA = "12-6.d04-loss-bearing-content-manifest.v2"
_CONTENT_MANIFEST_STATE_KEY = "loss_bearing_manifest_identity_sha256"


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
    return value


def require_expected_ledger_identity(
    ledger: Mapping[str, Any], *, expected_ledger_identity_sha256: str
) -> None:
    """Fail closed unless ledger bytes match the externally bound ledger identity."""
    if not isinstance(ledger, Mapping):
        raise LedgerError("ledger must be an object")
    if set(ledger) != _LEDGER_KEYS:
        raise LedgerError("ledger fields do not match the V2 identity-safe schema")

    expected = _require_sha256(
        expected_ledger_identity_sha256, "expected_ledger_identity_sha256"
    )
    observed = _require_sha256(
        ledger.get("ledger_identity_sha256"), "ledger_identity_sha256"
    )
    if observed != expected:
        raise LedgerError("ledger identity does not match expected stage authority")

    payload = deepcopy(dict(ledger))
    payload.pop("ledger_identity_sha256")
    if _canonical_sha256(payload) != observed:
        raise LedgerError("ledger self-hash mismatch")


def _validate_content_manifest_root(
    manifest: Mapping[str, Any], *, expected_manifest_identity_sha256: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(manifest, Mapping):
        raise LedgerError("loss-bearing content manifest must be an object")
    copied = deepcopy(dict(manifest))
    if copied.get("schema_version") != _CONTENT_MANIFEST_SCHEMA:
        raise LedgerError("unsupported loss-bearing content manifest schema")
    expected = _require_sha256(
        expected_manifest_identity_sha256,
        "expected_loss_bearing_manifest_identity_sha256",
    )
    observed = _require_sha256(
        copied.get("manifest_identity_sha256"),
        "loss_bearing_content_manifest.manifest_identity_sha256",
    )
    unhashed = deepcopy(copied)
    unhashed.pop("manifest_identity_sha256", None)
    if _canonical_sha256(unhashed) != observed:
        raise LedgerError("loss-bearing content manifest self-hash mismatch")
    if observed != expected:
        raise LedgerError("loss-bearing content manifest differs from external authority")
    if copied.get("training_authorized_by_this_manifest") is not False:
        raise LedgerError("loss-bearing content manifest may not self-authorize training")
    if copied.get("live_input_binding") != "canonical_predictor_context_and_targets_v2":
        raise LedgerError("loss-bearing content manifest lacks predictor/context authority")
    batches = copied.get("batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise LedgerError("loss-bearing content manifest batches must be a sequence")
    return copied, observed


class IdentitySafeExposureReplayGuard(ExposureReplayGuard):
    """Exposure guard with external ledger, state and optional content authority."""

    def __init__(
        self,
        ledger: Mapping[str, Any],
        *,
        expected_ledger_identity_sha256: str,
        authorized_budget: int,
        trainer_state_binding: Mapping[str, Any],
        loss_bearing_content_manifest: Mapping[str, Any] | None = None,
        expected_loss_bearing_manifest_identity_sha256: str | None = None,
    ) -> None:
        require_expected_ledger_identity(
            ledger,
            expected_ledger_identity_sha256=expected_ledger_identity_sha256,
        )
        if (loss_bearing_content_manifest is None) != (
            expected_loss_bearing_manifest_identity_sha256 is None
        ):
            raise LedgerError(
                "loss-bearing content manifest and external manifest identity "
                "must be supplied together"
            )
        self._loss_bearing_content_manifest: dict[str, Any] | None = None
        self.loss_bearing_manifest_identity_sha256: str | None = None
        if loss_bearing_content_manifest is not None:
            assert expected_loss_bearing_manifest_identity_sha256 is not None
            (
                self._loss_bearing_content_manifest,
                self.loss_bearing_manifest_identity_sha256,
            ) = _validate_content_manifest_root(
                loss_bearing_content_manifest,
                expected_manifest_identity_sha256=(
                    expected_loss_bearing_manifest_identity_sha256
                ),
            )
        super().__init__(
            ledger,
            expected_ledger_identity_sha256=expected_ledger_identity_sha256,
            authorized_budget=authorized_budget,
            trainer_state_binding=trainer_state_binding,
        )

    @property
    def content_authority_configured(self) -> bool:
        return self._loss_bearing_content_manifest is not None

    def _normalized_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
    ) -> tuple[list[dict[str, Any]], int]:
        actual = _require_nonnegative_int(
            actual_nonignored_targets, "actual_nonignored_targets"
        )
        normalized_claims: list[dict[str, Any]] = []
        claimed_count = 0
        tentative = {key: list(value) for key, value in self._claims.items()}
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise LedgerError(f"claims[{index}] must be an object")
            if set(claim) != _CLAIM_KEYS:
                raise LedgerError(
                    f"claims[{index}] must contain exactly segment_identity_sha256, "
                    "offset_start and offset_end"
                )
            segment_id = _require_sha256(
                claim.get("segment_identity_sha256"),
                f"claims[{index}].segment_identity_sha256",
            )
            if segment_id not in self._segments:
                raise LedgerError("claim references unknown ledger segment")
            start = _require_nonnegative_int(
                claim.get("offset_start"), f"claims[{index}].offset_start"
            )
            end = _require_nonnegative_int(
                claim.get("offset_end"), f"claims[{index}].offset_end"
            )
            if start >= end or end > self._segments[segment_id]:
                raise LedgerError("claim interval is outside ledger segment")
            tentative[segment_id] = self._insert_interval(
                tentative[segment_id], start, end
            )
            normalized_claims.append(
                {
                    "segment_identity_sha256": segment_id,
                    "offset_start": start,
                    "offset_end": end,
                }
            )
            claimed_count += end - start

        if claimed_count != actual:
            raise LedgerError(
                "claimed loss-position count does not match actual nonignored target count"
            )
        if self.consumed_loss_positions + claimed_count > self.authorized_budget:
            raise LedgerError("batch would exceed authorized exposure budget")
        return normalized_claims, actual

    def _manifest_claims_for_batch(self, batch_index: int) -> list[dict[str, Any]]:
        if self._loss_bearing_content_manifest is None:
            raise LedgerError("loss-bearing content authority is not configured")
        index = _require_nonnegative_int(batch_index, "batch_index")
        batches = self._loss_bearing_content_manifest.get("batches")
        if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
            raise LedgerError("loss-bearing content manifest batches must be a sequence")
        if index >= len(batches):
            raise LedgerError("batch_index is outside loss-bearing content manifest")
        batch = batches[index]
        if not isinstance(batch, Mapping):
            raise LedgerError("loss-bearing content batch must be an object")
        if batch.get("global_batch_index") != index:
            raise LedgerError("loss-bearing content batch index is non-canonical")
        raw_claims = batch.get("claims")
        if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
            raise LedgerError("loss-bearing content batch claims must be a sequence")
        normalized: list[dict[str, Any]] = []
        for claim_index, claim in enumerate(raw_claims):
            if not isinstance(claim, Mapping):
                raise LedgerError(
                    f"loss-bearing content claims[{claim_index}] must be an object"
                )
            normalized.append(
                {
                    "segment_identity_sha256": _require_sha256(
                        claim.get("segment_identity_sha256"),
                        f"loss-bearing content claims[{claim_index}]"
                        ".segment_identity_sha256",
                    ),
                    "offset_start": _require_nonnegative_int(
                        claim.get("offset_start"),
                        f"loss-bearing content claims[{claim_index}].offset_start",
                    ),
                    "offset_end": _require_nonnegative_int(
                        claim.get("offset_end"),
                        f"loss-bearing content claims[{claim_index}].offset_end",
                    ),
                }
            )
        return normalized

    def next_exposure_identity(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
    ) -> str:
        """Return an order-sensitive identity for the exact next authorized batch."""
        normalized_claims, actual = self._normalized_claims(
            claims,
            actual_nonignored_targets=actual_nonignored_targets,
        )
        state_identity = _require_sha256(
            self.state_dict()["state_identity_sha256"], "state_identity_sha256"
        )
        payload: dict[str, Any] = {
            "schema_version": _NEXT_EXPOSURE_SCHEMA,
            "ledger_identity_sha256": self.ledger_identity_sha256,
            "materialization_identity_sha256": self.materialization_identity_sha256,
            "packing_identity_sha256": self.packing_identity_sha256,
            "current_exposure_state_identity_sha256": state_identity,
            "next_claim_sequence": self.claim_sequence + 1,
            "ordered_claims": normalized_claims,
            "actual_nonignored_targets": actual,
        }
        if self.loss_bearing_manifest_identity_sha256 is not None:
            payload[_CONTENT_MANIFEST_STATE_KEY] = (
                self.loss_bearing_manifest_identity_sha256
            )
        return _canonical_sha256(payload)

    def authorize_batch(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
    ) -> None:
        if self.content_authority_configured:
            raise LedgerError(
                "content-aware authorization is required while loss-bearing "
                "content authority is configured"
            )
        super().authorize_batch(
            claims,
            actual_nonignored_targets=actual_nonignored_targets,
        )

    def authorize_batch_with_identity(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
        expected_next_exposure_identity_sha256: str,
    ) -> str:
        """Authorize a legacy count-only batch only when no content authority exists."""
        if self.content_authority_configured:
            raise LedgerError(
                "content-aware authorization is required while loss-bearing "
                "content authority is configured"
            )
        expected = _require_sha256(
            expected_next_exposure_identity_sha256,
            "expected_next_exposure_identity_sha256",
        )
        observed = self.next_exposure_identity(
            claims,
            actual_nonignored_targets=actual_nonignored_targets,
        )
        if observed != expected:
            raise LedgerError("next exposure identity does not match expected handoff")
        super().authorize_batch(
            claims,
            actual_nonignored_targets=actual_nonignored_targets,
        )
        return observed

    def authorize_live_batch_with_identity(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
        expected_next_exposure_identity_sha256: str,
        batch_index: int,
        input_ids: Any,
        target_ids: Any,
        loss_mask: Any | None = None,
        shifted: bool = False,
        ignore_index: int = -100,
    ) -> str:
        """Authenticate exact live predictor/target content before exposure mutation."""
        if self._loss_bearing_content_manifest is None:
            raise LedgerError("loss-bearing content authority is not configured")
        manifest_identity = self.loss_bearing_manifest_identity_sha256
        assert manifest_identity is not None
        expected = _require_sha256(
            expected_next_exposure_identity_sha256,
            "expected_next_exposure_identity_sha256",
        )
        normalized_claims, actual = self._normalized_claims(
            claims,
            actual_nonignored_targets=actual_nonignored_targets,
        )
        observed = self.next_exposure_identity(
            normalized_claims,
            actual_nonignored_targets=actual,
        )
        if observed != expected:
            raise LedgerError("next exposure identity does not match expected handoff")
        manifest_claims = self._manifest_claims_for_batch(batch_index)
        if normalized_claims != manifest_claims:
            raise LedgerError(
                "submitted claims differ from canonical loss-bearing content manifest"
            )

        from twelve_six.data.loss_bearing_content_binding_v1 import (
            verify_live_loss_bearing_batch,
        )

        verify_live_loss_bearing_batch(
            self._loss_bearing_content_manifest,
            expected_manifest_identity_sha256=manifest_identity,
            batch_index=batch_index,
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            shifted=shifted,
            ignore_index=ignore_index,
        )
        super().authorize_batch(
            normalized_claims,
            actual_nonignored_targets=actual,
        )
        return observed

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        manifest_identity = self.loss_bearing_manifest_identity_sha256
        if manifest_identity is None:
            return state
        hardened = deepcopy(state)
        hardened.pop("state_identity_sha256", None)
        hardened[_CONTENT_MANIFEST_STATE_KEY] = manifest_identity
        hardened["state_identity_sha256"] = _canonical_sha256(hardened)
        return hardened

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_state_identity_sha256: str,
        expected_trainer_state_binding: Mapping[str, Any],
    ) -> None:
        manifest_identity = self.loss_bearing_manifest_identity_sha256
        if manifest_identity is None:
            if isinstance(state, Mapping) and _CONTENT_MANIFEST_STATE_KEY in state:
                raise LedgerError(
                    "content-bound exposure state requires matching content authority"
                )
            super().load_state_dict(
                state,
                expected_state_identity_sha256=expected_state_identity_sha256,
                expected_trainer_state_binding=expected_trainer_state_binding,
            )
            return

        if not isinstance(state, Mapping):
            raise LedgerError("exposure state must be an object")
        expected_outer = _require_sha256(
            expected_state_identity_sha256, "expected_state_identity_sha256"
        )
        outer = deepcopy(dict(state))
        observed_outer = _require_sha256(
            outer.pop("state_identity_sha256", None), "state_identity_sha256"
        )
        if _canonical_sha256(outer) != observed_outer:
            raise LedgerError("exposure state self-hash mismatch")
        if observed_outer != expected_outer:
            raise LedgerError("resume exposure state identity mismatch")
        saved_manifest_identity = _require_sha256(
            outer.pop(_CONTENT_MANIFEST_STATE_KEY, None),
            _CONTENT_MANIFEST_STATE_KEY,
        )
        if saved_manifest_identity != manifest_identity:
            raise LedgerError("resume loss-bearing content manifest identity mismatch")

        base_state = outer
        base_identity = _canonical_sha256(base_state)
        base_state["state_identity_sha256"] = base_identity
        super().load_state_dict(
            base_state,
            expected_state_identity_sha256=base_identity,
            expected_trainer_state_binding=expected_trainer_state_binding,
        )
