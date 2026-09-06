from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.identity_safe_exposure_guard import (
    IdentitySafeExposureReplayGuard,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _materialization() -> dict:
    value = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "stage_bindings": {
            "normalization": _sha("normalization"),
            "evaluation_reservations": _sha("reservations"),
            "dedup": _sha("dedup"),
            "split": _sha("split"),
            "packing": _sha("packing-stage"),
        },
        "tokenizer": {
            "name": "s0-byte-v1",
            "identity_sha256": _sha("tokenizer"),
            "source_bytes_are_loss_positions": False,
        },
        "documents": [
            {
                "document_id": "doc",
                "language": "uk",
                "modality": "text",
                "family_id": "family.uk",
                "normalized_payload_sha256": _sha("payload"),
                "source_bytes": 4,
                "token_count": 4,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 4]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 4,
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": 4,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    value["materialization_identity_sha256"] = _identity(
        value, "materialization_identity_sha256"
    )
    return value


def _binding() -> dict:
    return {
        "checkpoint_generation": "g000",
        "checkpoint_manifest_sha256": _sha("checkpoint-g000"),
        "optimizer_step": 0,
        "trainer_nonignored_target_count": 0,
    }


def _guard(ledger: dict) -> IdentitySafeExposureReplayGuard:
    return IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=3,
        trainer_state_binding=_binding(),
    )


def test_identity_safe_guard_accepts_exact_built_ledger() -> None:
    ledger = build_ledger(_materialization())
    guard = _guard(ledger)
    assert guard.ledger_identity_sha256 == ledger["ledger_identity_sha256"]
    assert guard.one_pass_maximum == 3


def test_identity_safe_guard_rejects_tampered_bytes_with_stale_identity() -> None:
    ledger = build_ledger(_materialization())
    tampered = deepcopy(ledger)
    tampered["segments"][0]["loss_position_count"] = 30
    with pytest.raises(LedgerError, match="self-hash mismatch"):
        _guard(tampered)


def test_identity_safe_guard_rejects_rehashed_substitution_against_stage_identity() -> None:
    ledger = build_ledger(_materialization())
    expected = ledger["ledger_identity_sha256"]
    tampered = deepcopy(ledger)
    tampered["segments"][0]["loss_position_count"] = 30
    tampered["ledger_identity_sha256"] = _identity(
        tampered, "ledger_identity_sha256"
    )
    with pytest.raises(LedgerError, match="does not match expected stage authority"):
        IdentitySafeExposureReplayGuard(
            tampered,
            expected_ledger_identity_sha256=expected,
            authorized_budget=3,
            trainer_state_binding=_binding(),
        )


def test_identity_safe_guard_rejects_self_hashed_unknown_ledger_field() -> None:
    ledger = build_ledger(_materialization())
    tampered = deepcopy(ledger)
    tampered["future_replay_semantics"] = {"replay_allowed": True}
    tampered["ledger_identity_sha256"] = _identity(
        tampered, "ledger_identity_sha256"
    )
    with pytest.raises(LedgerError, match="fields do not match"):
        IdentitySafeExposureReplayGuard(
            tampered,
            expected_ledger_identity_sha256=tampered["ledger_identity_sha256"],
            authorized_budget=3,
            trainer_state_binding=_binding(),
        )
