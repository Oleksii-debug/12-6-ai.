from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from twelve_six.data import _unique_loss_ledger_v2_core as _core

# Preserve the canonical module surface byte-for-byte from the verified V2 core,
# including existing private helpers that downstream repository code may import.
for _name in dir(_core):
    if not _name.startswith("__") and _name != "ExposureReplayGuard":
        globals()[_name] = getattr(_core, _name)

LEDGER_KEYS = frozenset(
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


class ExposureReplayGuard(_core.ExposureReplayGuard):
    """Replay guard rooted in externally authorized exact identities.

    The V2 core retains the deterministic ledger builder and replay accounting.
    This canonical facade adds the two missing trust roots: construction must
    match an independently expected exact ledger identity, and resume must match
    an independently expected exact exposure-state identity. A candidate cannot
    widen capacity or relocate consumed positions merely by re-sealing itself.
    """

    def __init__(
        self,
        ledger: Mapping[str, Any],
        *,
        expected_ledger_identity_sha256: str,
        authorized_budget: int,
        trainer_state_binding: Mapping[str, Any],
    ) -> None:
        if not isinstance(ledger, Mapping):
            raise LedgerError("ledger must be an object")
        if set(ledger) != LEDGER_KEYS:
            raise LedgerError("ledger fields do not match the V2 guard schema")
        if ledger.get("schema_version") != LEDGER_SCHEMA:
            raise LedgerError("guard requires a V2 ledger")

        expected_identity = _require_sha256(
            expected_ledger_identity_sha256, "expected_ledger_identity_sha256"
        )
        payload = deepcopy(dict(ledger))
        observed_identity = _require_sha256(
            payload.pop("ledger_identity_sha256", None), "ledger_identity_sha256"
        )
        if _sha256_obj(payload) != observed_identity:
            raise LedgerError("ledger self-hash mismatch")
        if observed_identity != expected_identity:
            raise LedgerError("ledger identity does not match expected authority")

        super().__init__(
            ledger,
            authorized_budget=authorized_budget,
            trainer_state_binding=trainer_state_binding,
        )

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_state_identity_sha256: str,
        expected_trainer_state_binding: Mapping[str, Any],
    ) -> None:
        if not isinstance(state, Mapping):
            raise LedgerError("exposure state must be an object")
        if set(state) != EXPOSURE_STATE_KEYS:
            raise LedgerError("exposure state fields do not match the V2 schema")

        expected_identity = _require_sha256(
            expected_state_identity_sha256, "expected_state_identity_sha256"
        )
        payload = deepcopy(dict(state))
        observed_identity = _require_sha256(
            payload.pop("state_identity_sha256", None), "state_identity_sha256"
        )
        if _sha256_obj(payload) != observed_identity:
            raise LedgerError("exposure state self-hash mismatch")
        if observed_identity != expected_identity:
            raise LedgerError("resume exposure state identity mismatch")

        super().load_state_dict(
            state,
            expected_trainer_state_binding=expected_trainer_state_binding,
        )
