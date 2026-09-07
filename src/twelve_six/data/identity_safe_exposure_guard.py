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
    """Fail closed unless ledger bytes match the externally bound ledger identity.

    ExposureReplayGuard V2 binds resume state to the ledger identity string. This
    preflight additionally proves that the string is the hash of the exact ledger
    object and that it equals the identity supplied by the stage/checkpoint
    authority. Unknown ledger fields are rejected so a self-rehashed future schema
    cannot silently alter current replay semantics.
    """
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


class IdentitySafeExposureReplayGuard(ExposureReplayGuard):
    """ExposureReplayGuard with external ledger and next-exposure identity guards."""

    def __init__(
        self,
        ledger: Mapping[str, Any],
        *,
        expected_ledger_identity_sha256: str,
        authorized_budget: int,
        trainer_state_binding: Mapping[str, Any],
    ) -> None:
        require_expected_ledger_identity(
            ledger,
            expected_ledger_identity_sha256=expected_ledger_identity_sha256,
        )
        super().__init__(
            ledger,
            authorized_budget=authorized_budget,
            trainer_state_binding=trainer_state_binding,
        )

    def next_exposure_identity(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
    ) -> str:
        """Return an order-sensitive identity for the exact next authorized batch.

        The identity binds the current self-hashed exposure state, ledger/corpus/
        packing identities, next claim sequence, ordered claim intervals and the
        trainer-observed target cardinality. It is stable across a fresh-process
        resume of the same state and changes if a worker/order/batch substitution
        is attempted.
        """
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

        state_identity = _require_sha256(
            self.state_dict()["state_identity_sha256"], "state_identity_sha256"
        )
        return _canonical_sha256(
            {
                "schema_version": _NEXT_EXPOSURE_SCHEMA,
                "ledger_identity_sha256": self.ledger_identity_sha256,
                "materialization_identity_sha256": self.materialization_identity_sha256,
                "packing_identity_sha256": self.packing_identity_sha256,
                "current_exposure_state_identity_sha256": state_identity,
                "next_claim_sequence": self.claim_sequence + 1,
                "ordered_claims": normalized_claims,
                "actual_nonignored_targets": actual,
            }
        )

    def authorize_batch_with_identity(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
        expected_next_exposure_identity_sha256: str,
    ) -> str:
        """Authorize only the externally expected exact next exposure identity."""
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
