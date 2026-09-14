from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.deterministic_exposure_order import build_deterministic_exposure_plan
from twelve_six.data.identity_safe_exposure_guard import (
    IdentitySafeExposureReplayGuard,
)
from twelve_six.data.loss_bearing_content_binding_v1 import (
    build_loss_bearing_content_manifest,
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
                    "token_ids": [10, 11, 12, 13],
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


def _content_authority() -> tuple[dict, dict, dict, dict]:
    materialization = _materialization()
    ledger = build_ledger(materialization)
    claim = _claim(ledger, 0, 3)
    plan = build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [claim],
                "actual_nonignored_targets": 3,
            }
        ],
        num_workers=1,
        batches_per_shard=1,
        shard_count=1,
    )
    manifest = build_loss_bearing_content_manifest(
        materialization,
        ledger,
        plan,
        expected_materialization_identity_sha256=materialization[
            "materialization_identity_sha256"
        ],
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    return materialization, ledger, plan, manifest


def _content_guard(
    ledger: dict,
    manifest: dict,
    *,
    expected_manifest_identity_sha256: str | None = None,
) -> IdentitySafeExposureReplayGuard:
    return IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=3,
        trainer_state_binding=_binding(),
        loss_bearing_content_manifest=manifest,
        expected_loss_bearing_manifest_identity_sha256=(
            expected_manifest_identity_sha256 or manifest["manifest_identity_sha256"]
        ),
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
        expected_state_identity_sha256=state["state_identity_sha256"],
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


def test_content_bound_guard_authorizes_exact_aligned_live_batch() -> None:
    _, ledger, _, manifest = _content_authority()
    guard = _content_guard(ledger, manifest)
    claims = [_claim(ledger, 0, 3)]
    identity = guard.next_exposure_identity(claims, actual_nonignored_targets=3)
    observed = guard.authorize_live_batch_with_identity(
        claims,
        actual_nonignored_targets=3,
        expected_next_exposure_identity_sha256=identity,
        batch_index=0,
        input_ids=[[10, 11, 12]],
        target_ids=[[11, 12, 13]],
        loss_mask=[[1, 1, 1]],
    )
    assert observed == identity
    assert guard.consumed_loss_positions == 3
    assert guard.claim_sequence == 1
    assert guard.state_dict()["loss_bearing_manifest_identity_sha256"] == manifest[
        "manifest_identity_sha256"
    ]


def test_content_bound_guard_authorizes_exact_shifted_live_batch() -> None:
    _, ledger, _, manifest = _content_authority()
    guard = _content_guard(ledger, manifest)
    claims = [_claim(ledger, 0, 3)]
    identity = guard.next_exposure_identity(claims, actual_nonignored_targets=3)
    guard.authorize_live_batch_with_identity(
        claims,
        actual_nonignored_targets=3,
        expected_next_exposure_identity_sha256=identity,
        batch_index=0,
        input_ids=[[10, 11, 12, 13]],
        target_ids=[[10, 11, 12, 13]],
        shifted=True,
    )
    assert guard.consumed_loss_positions == 3


@pytest.mark.parametrize(
    ("input_ids", "target_ids", "message"),
    [
        ([[10, 99, 12]], [[11, 12, 13]], "predictor/context"),
        ([[10, 11, 12]], [[11, 99, 13]], "target content"),
    ],
)
def test_content_mutation_rejects_without_guard_state_mutation(
    input_ids: list[list[int]],
    target_ids: list[list[int]],
    message: str,
) -> None:
    _, ledger, _, manifest = _content_authority()
    guard = _content_guard(ledger, manifest)
    claims = [_claim(ledger, 0, 3)]
    identity = guard.next_exposure_identity(claims, actual_nonignored_targets=3)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match=message):
        guard.authorize_live_batch_with_identity(
            claims,
            actual_nonignored_targets=3,
            expected_next_exposure_identity_sha256=identity,
            batch_index=0,
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=[[1, 1, 1]],
        )
    assert guard.state_dict() == before


def test_equal_count_claim_relocation_rejects_before_mutation() -> None:
    _, ledger, _, manifest = _content_authority()
    guard = _content_guard(ledger, manifest)
    relocated = [_claim(ledger, 0, 1), _claim(ledger, 1, 3)]
    identity = guard.next_exposure_identity(relocated, actual_nonignored_targets=3)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="submitted claims differ"):
        guard.authorize_live_batch_with_identity(
            relocated,
            actual_nonignored_targets=3,
            expected_next_exposure_identity_sha256=identity,
            batch_index=0,
            input_ids=[[10, 11, 12]],
            target_ids=[[11, 12, 13]],
            loss_mask=[[1, 1, 1]],
        )
    assert guard.state_dict() == before


def test_contentless_authorization_rejects_when_content_authority_is_configured() -> None:
    _, ledger, _, manifest = _content_authority()
    guard = _content_guard(ledger, manifest)
    claims = [_claim(ledger, 0, 3)]
    identity = guard.next_exposure_identity(claims, actual_nonignored_targets=3)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="content-aware authorization"):
        guard.authorize_batch_with_identity(
            claims,
            actual_nonignored_targets=3,
            expected_next_exposure_identity_sha256=identity,
        )
    assert guard.state_dict() == before


def test_resealed_manifest_rejects_against_external_manifest_root() -> None:
    _, ledger, _, manifest = _content_authority()
    tampered = deepcopy(manifest)
    tampered["future_content_semantics"] = "resealed"
    tampered["manifest_identity_sha256"] = _identity(
        tampered, "manifest_identity_sha256"
    )
    with pytest.raises(LedgerError, match="differs from external authority"):
        _content_guard(
            ledger,
            tampered,
            expected_manifest_identity_sha256=manifest["manifest_identity_sha256"],
        )


def test_resume_rejects_different_loss_bearing_manifest_root_before_mutation() -> None:
    _, ledger, _, manifest = _content_authority()
    source = _content_guard(ledger, manifest)
    claims = [_claim(ledger, 0, 3)]
    identity = source.next_exposure_identity(claims, actual_nonignored_targets=3)
    source.authorize_live_batch_with_identity(
        claims,
        actual_nonignored_targets=3,
        expected_next_exposure_identity_sha256=identity,
        batch_index=0,
        input_ids=[[10, 11, 12]],
        target_ids=[[11, 12, 13]],
        loss_mask=[[1, 1, 1]],
    )
    checkpoint_binding = {
        "checkpoint_generation": "g001",
        "checkpoint_manifest_sha256": _sha("checkpoint-g001"),
        "optimizer_step": 1,
        "trainer_nonignored_target_count": 3,
    }
    source.bind_checkpoint_state(checkpoint_binding)
    state = source.state_dict()

    other_manifest = deepcopy(manifest)
    other_manifest["future_content_semantics"] = "other-authority"
    other_manifest["manifest_identity_sha256"] = _identity(
        other_manifest, "manifest_identity_sha256"
    )
    resumed = _content_guard(ledger, other_manifest)
    before = resumed.state_dict()
    with pytest.raises(LedgerError, match="content manifest identity mismatch"):
        resumed.load_state_dict(
            state,
            expected_state_identity_sha256=state["state_identity_sha256"],
            expected_trainer_state_binding=checkpoint_binding,
        )
    assert resumed.state_dict() == before
