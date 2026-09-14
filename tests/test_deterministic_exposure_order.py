from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.deterministic_exposure_order import (
    authorize_ordered_batch,
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _guard() -> tuple[dict, IdentitySafeExposureReplayGuard]:
    materialization = {
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
                "source_bytes": 5,
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 5,
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    ledger = build_ledger(materialization)
    guard = IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=4,
        trainer_state_binding={
            "checkpoint_generation": "g000",
            "checkpoint_manifest_sha256": _sha("checkpoint"),
            "optimizer_step": 0,
            "trainer_nonignored_target_count": 0,
        },
    )
    return ledger, guard


def _claim(ledger: dict, start: int, end: int) -> dict:
    return {
        "segment_identity_sha256": ledger["segments"][0]["segment_identity_sha256"],
        "offset_start": start,
        "offset_end": end,
    }


def _plan(ledger: dict) -> dict:
    return build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [_claim(ledger, 0, 1)],
                "actual_nonignored_targets": 1,
            },
            {
                "global_batch_index": 1,
                "shard_index": 0,
                "worker_id": 1,
                "claims": [_claim(ledger, 1, 2)],
                "actual_nonignored_targets": 1,
            },
            {
                "global_batch_index": 2,
                "shard_index": 1,
                "worker_id": 0,
                "claims": [_claim(ledger, 2, 4)],
                "actual_nonignored_targets": 2,
            },
        ],
        num_workers=2,
        batches_per_shard=2,
        shard_count=2,
    )


def test_plan_is_deterministic_and_order_sensitive() -> None:
    ledger, _ = _guard()
    left = _plan(ledger)
    right = _plan(ledger)
    assert left == right
    tampered = deepcopy(left)
    tampered["batches"][0]["claims"], tampered["batches"][1]["claims"] = (
        tampered["batches"][1]["claims"],
        tampered["batches"][0]["claims"],
    )
    assert left["plan_identity_sha256"] != _identity(tampered, "plan_identity_sha256")


def test_plan_rejects_nondeterministic_worker_assignment() -> None:
    ledger, _ = _guard()
    batches = _plan(ledger)["batches"]
    batches[1]["worker_id"] = 0
    with pytest.raises(LedgerError, match="worker assignment"):
        build_deterministic_exposure_plan(
            batches, num_workers=2, batches_per_shard=2, shard_count=2
        )


def test_plan_rejects_nondeterministic_shard_assignment() -> None:
    ledger, _ = _guard()
    batches = _plan(ledger)["batches"]
    batches[2]["shard_index"] = 0
    with pytest.raises(LedgerError, match="shard assignment"):
        build_deterministic_exposure_plan(
            batches, num_workers=2, batches_per_shard=2, shard_count=2
        )


def test_plan_rejects_zero_target_batch_at_construction() -> None:
    with pytest.raises(LedgerError, match="actual_nonignored_targets must be positive"):
        build_deterministic_exposure_plan(
            [
                {
                    "global_batch_index": 0,
                    "shard_index": 0,
                    "worker_id": 0,
                    "claims": [],
                    "actual_nonignored_targets": 0,
                }
            ],
            num_workers=1,
            batches_per_shard=1,
            shard_count=1,
        )


def test_runtime_rejects_rehashed_nondeterministic_plan() -> None:
    ledger, guard = _guard()
    plan = _plan(ledger)
    tampered = deepcopy(plan)
    tampered["batches"][0]["worker_id"] = 1
    tampered["plan_identity_sha256"] = _identity(tampered, "plan_identity_sha256")
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="worker assignment"):
        ordered_next_exposure_identity(
            guard,
            tampered,
            batch_index=0,
            expected_plan_identity_sha256=tampered["plan_identity_sha256"],
        )
    assert guard.state_dict() == before


def test_runtime_rejects_rehashed_valid_plan_substitution() -> None:
    ledger, guard = _guard()
    plan = _plan(ledger)
    expected_plan_identity = plan["plan_identity_sha256"]

    substitute = deepcopy(plan)
    substitute["batches"][0]["claims"] = [_claim(ledger, 0, 2)]
    substitute["batches"][0]["actual_nonignored_targets"] = 2
    substitute["batches"][1]["claims"] = [_claim(ledger, 2, 3)]
    substitute["plan_identity_sha256"] = _identity(
        substitute, "plan_identity_sha256"
    )

    before = guard.state_dict()
    with pytest.raises(LedgerError, match="plan identity does not match expected handoff"):
        ordered_next_exposure_identity(
            guard,
            substitute,
            batch_index=0,
            expected_plan_identity_sha256=expected_plan_identity,
        )
    assert guard.state_dict() == before


def test_authorize_ordered_batch_is_resume_safe_and_sequence_strict() -> None:
    ledger, guard = _guard()
    plan = _plan(ledger)
    plan_identity = plan["plan_identity_sha256"]
    first = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan_identity,
    )
    authorize_ordered_batch(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan_identity,
        expected_ordered_next_exposure_identity_sha256=first,
    )
    with pytest.raises(LedgerError, match="next exposure sequence"):
        ordered_next_exposure_identity(
            guard,
            plan,
            batch_index=2,
            expected_plan_identity_sha256=plan_identity,
        )
    second = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=1,
        expected_plan_identity_sha256=plan_identity,
    )
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="does not match expected handoff"):
        authorize_ordered_batch(
            guard,
            plan,
            batch_index=1,
            expected_plan_identity_sha256=plan_identity,
            expected_ordered_next_exposure_identity_sha256=_sha("wrong"),
        )
    assert guard.state_dict() == before
    authorize_ordered_batch(
        guard,
        plan,
        batch_index=1,
        expected_plan_identity_sha256=plan_identity,
        expected_ordered_next_exposure_identity_sha256=second,
    )
    assert guard.claim_sequence == 2
    assert guard.consumed_loss_positions == 2
