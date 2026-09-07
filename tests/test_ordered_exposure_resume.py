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
from twelve_six.data.ordered_exposure_resume import (
    load_ordered_resume_state,
    ordered_resume_state_dict,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _ledger() -> dict:
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
    return build_ledger(materialization)


def _binding(generation: str, optimizer_step: int, target_count: int) -> dict:
    return {
        "checkpoint_generation": generation,
        "checkpoint_manifest_sha256": _sha(f"checkpoint-{generation}"),
        "optimizer_step": optimizer_step,
        "trainer_nonignored_target_count": target_count,
    }


def _guard(ledger: dict, binding: dict | None = None) -> IdentitySafeExposureReplayGuard:
    return IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=4,
        trainer_state_binding=binding or _binding("g000", 0, 0),
    )


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


def _checkpoint_after_first(ledger: dict, plan: dict) -> tuple[dict, dict]:
    guard = _guard(ledger)
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
    checkpoint = _binding("g001", 1, 1)
    guard.bind_checkpoint_state(checkpoint)
    state = ordered_resume_state_dict(
        guard,
        plan,
        expected_plan_identity_sha256=plan_identity,
    )
    return checkpoint, state


def test_ordered_resume_roundtrip_binds_exact_next_exposure() -> None:
    ledger = _ledger()
    plan = _plan(ledger)
    checkpoint, state = _checkpoint_after_first(ledger, plan)
    resumed = _guard(ledger)
    load_ordered_resume_state(
        resumed,
        plan,
        state,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
        expected_trainer_state_binding=checkpoint,
    )
    assert resumed.claim_sequence == 1
    assert resumed.consumed_loss_positions == 1
    observed_next = ordered_next_exposure_identity(
        resumed,
        plan,
        batch_index=1,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    assert observed_next == state["next_ordered_exposure_identity_sha256"]


def test_ordered_resume_rejects_rehashed_plan_substitution_without_mutation() -> None:
    ledger = _ledger()
    plan = _plan(ledger)
    checkpoint, state = _checkpoint_after_first(ledger, plan)
    substitute = deepcopy(plan)
    substitute["batches"][1]["claims"] = [_claim(ledger, 1, 3)]
    substitute["batches"][1]["actual_nonignored_targets"] = 2
    substitute["batches"][2]["claims"] = [_claim(ledger, 3, 4)]
    substitute["batches"][2]["actual_nonignored_targets"] = 1
    substitute["plan_identity_sha256"] = _identity(
        substitute, "plan_identity_sha256"
    )

    resumed = _guard(ledger)
    before = resumed.state_dict()
    with pytest.raises(LedgerError, match="plan identity does not match expected handoff"):
        load_ordered_resume_state(
            resumed,
            substitute,
            state,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_trainer_state_binding=checkpoint,
        )
    assert resumed.state_dict() == before


def test_ordered_resume_rejects_consumed_claims_not_matching_plan_prefix() -> None:
    ledger = _ledger()
    plan = _plan(ledger)
    checkpoint, state = _checkpoint_after_first(ledger, plan)
    tampered = deepcopy(state)
    segment_id = ledger["segments"][0]["segment_identity_sha256"]
    guard_state = tampered["guard_state"]
    guard_state["claims"] = {segment_id: [[1, 2]]}
    guard_state["state_identity_sha256"] = _identity(
        guard_state, "state_identity_sha256"
    )
    tampered["ordered_resume_identity_sha256"] = _identity(
        tampered, "ordered_resume_identity_sha256"
    )

    resumed = _guard(ledger)
    before = resumed.state_dict()
    with pytest.raises(LedgerError, match="claims do not match deterministic plan prefix"):
        load_ordered_resume_state(
            resumed,
            plan,
            tampered,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_trainer_state_binding=checkpoint,
        )
    assert resumed.state_dict() == before


def test_ordered_resume_restores_guard_after_next_identity_rejection() -> None:
    ledger = _ledger()
    plan = _plan(ledger)
    checkpoint, state = _checkpoint_after_first(ledger, plan)
    tampered = deepcopy(state)
    tampered["next_ordered_exposure_identity_sha256"] = _sha("wrong-next")
    tampered["ordered_resume_identity_sha256"] = _identity(
        tampered, "ordered_resume_identity_sha256"
    )

    resumed = _guard(ledger)
    before = resumed.state_dict()
    with pytest.raises(LedgerError, match="next exposure identity does not match saved handoff"):
        load_ordered_resume_state(
            resumed,
            plan,
            tampered,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_trainer_state_binding=checkpoint,
        )
    assert resumed.state_dict() == before
