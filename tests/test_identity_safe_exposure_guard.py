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


def _claim(ledger: dict, start: int, end: int) -> dict:
    return {
        "segment_identity_sha256": ledger["segments"][0]["segment_identity_sha256"],
        "offset_start": start,
        "offset_end": end,
    }


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


def test_next_exposure_identity_is_order_sensitive() -> None:
    ledger = build_ledger(_materialization())
    guard = _guard(ledger)
    left = guard.next_exposure_identity(
        [_claim(ledger, 0, 1), _claim(ledger, 1, 3)],
        actual_nonignored_targets=3,
    )
    right = guard.next_exposure_identity(
        [_claim(ledger, 1, 3), _claim(ledger, 0, 1)],
        actual_nonignored_targets=3,
    )
    assert left != right
    assert guard.consumed_loss_positions == 0
    assert guard.claim_sequence == 0


def test_next_exposure_identity_is_stable_after_fresh_resume() -> None:
    ledger = build_ledger(_materialization())
    guard = _guard(ledger)
    first = [_claim(ledger, 0, 1)]
    first_identity = guard.next_exposure_identity(
        first,
        actual_nonignored_targets=1,
    )
    guard.authorize_batch_with_identity(
        first,
        actual_nonignored_targets=1,
        expected_next_exposure_identity_sha256=first_identity,
    )
    checkpoint_binding = {
        "checkpoint_generation": "g001",
        "checkpoint_manifest_sha256": _sha("checkpoint-g001"),
        "optimizer_step": 1,
        "trainer_nonignored_target_count": 1,
    }
    guard.bind_checkpoint_state(checkpoint_binding)
    state = guard.state_dict()

    expected_next = guard.next_exposure_identity(
        [_claim(ledger, 1, 3)],
        actual_nonignored_targets=2,
    )
    resumed = _guard(ledger)
    resumed.load_state_dict(
        state,
        expected_trainer_state_binding=checkpoint_binding,
    )
    assert resumed.next_exposure_identity(
        [_claim(ledger, 1, 3)],
        actual_nonignored_targets=2,
    ) == expected_next


def test_wrong_next_exposure_identity_rejects_without_mutation() -> None:
    ledger = build_ledger(_materialization())
    guard = _guard(ledger)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="next exposure identity"):
        guard.authorize_batch_with_identity(
            [_claim(ledger, 0, 1)],
            actual_nonignored_targets=1,
            expected_next_exposure_identity_sha256=_sha("wrong-next-exposure"),
        )
    assert guard.state_dict() == before


def test_next_exposure_identity_rejects_replay_before_identity_is_issued() -> None:
    ledger = build_ledger(_materialization())
    guard = _guard(ledger)
    claim = [_claim(ledger, 0, 1)]
    identity = guard.next_exposure_identity(claim, actual_nonignored_targets=1)
    guard.authorize_batch_with_identity(
        claim,
        actual_nonignored_targets=1,
        expected_next_exposure_identity_sha256=identity,
    )
    with pytest.raises(LedgerError, match="replay/overlapping"):
        guard.next_exposure_identity(claim, actual_nonignored_targets=1)
