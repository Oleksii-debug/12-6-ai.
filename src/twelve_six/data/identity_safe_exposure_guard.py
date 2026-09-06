from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
    """ExposureReplayGuard with mandatory external ledger-identity preflight."""

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
