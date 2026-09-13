from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.deterministic_exposure_order import (
    authorize_ordered_batch,
    authorize_ordered_live_batch,
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.loss_bearing_content_binding_v1 import (
    build_loss_bearing_content_manifest,
)
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
                    "token_ids": [10, 11, 12, 13, 14],
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
    value["materialization_identity_sha256"] = _identity(
        value, "materialization_identity_sha256"
    )
    return value


def _binding(generation: str = "g000", step: int = 0, targets: int = 0) -> dict:
    return {
        "checkpoint_generation": generation,
        "checkpoint_manifest_sha256": _sha(f"checkpoint-{generation}"),
        "optimizer_step": step,
        "trainer_nonignored_target_count": targets,
    }


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


def _authority() -> tuple[dict, dict, dict, dict]:
    materialization = _materialization()
    ledger = build_ledger(materialization)
    plan = _plan(ledger)
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


def _guard(ledger: dict, manifest: dict) -> IdentitySafeExposureReplayGuard:
    return IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=4,
        trainer_state_binding=_binding(),
        loss_bearing_content_manifest=manifest,
        expected_loss_bearing_manifest_identity_sha256=manifest[
            "manifest_identity_sha256"
        ],
    )


def _live(batch_index: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    if batch_index == 0:
        return [[10]], [[11]], [[1]]
    if batch_index == 1:
        return [[11]], [[12]], [[1]]
    if batch_index == 2:
        return [[12, 13]], [[13, 14]], [[1, 1]]
    raise AssertionError("unexpected test batch")


def _ordered_identity(guard: IdentitySafeExposureReplayGuard, plan: dict, index: int) -> str:
    return ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=index,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )


def _authorize(guard: IdentitySafeExposureReplayGuard, plan: dict, index: int) -> str:
    input_ids, target_ids, loss_mask = _live(index)
    return authorize_ordered_live_batch(
        guard,
        plan,
        batch_index=index,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
        expected_ordered_next_exposure_identity_sha256=_ordered_identity(
            guard, plan, index
        ),
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )


def test_ordered_live_authority_consumes_multi_batch_plan_exactly_once() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    for index in range(3):
        observed = _authorize(guard, plan, index)
        assert len(observed) == 64
        assert guard.claim_sequence == index + 1
    assert guard.consumed_loss_positions == 4


def test_ordered_live_authority_rejects_later_batch_before_mutation() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    before = guard.state_dict()
    input_ids, target_ids, loss_mask = _live(1)
    with pytest.raises(LedgerError, match="next exposure sequence"):
        authorize_ordered_live_batch(
            guard,
            plan,
            batch_index=1,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_ordered_next_exposure_identity_sha256=_sha("unused"),
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )
    assert guard.state_dict() == before


def test_ordered_live_authority_rejects_wrong_plan_root_without_mutation() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    before = guard.state_dict()
    input_ids, target_ids, loss_mask = _live(0)
    with pytest.raises(LedgerError, match="plan identity does not match expected handoff"):
        authorize_ordered_live_batch(
            guard,
            plan,
            batch_index=0,
            expected_plan_identity_sha256=_sha("wrong-plan"),
            expected_ordered_next_exposure_identity_sha256=_sha("unused"),
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )
    assert guard.state_dict() == before


def test_ordered_live_authority_rejects_wrong_ordered_root_without_mutation() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    before = guard.state_dict()
    input_ids, target_ids, loss_mask = _live(0)
    with pytest.raises(LedgerError, match="ordered next exposure identity"):
        authorize_ordered_live_batch(
            guard,
            plan,
            batch_index=0,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_ordered_next_exposure_identity_sha256=_sha("wrong-ordered"),
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )
    assert guard.state_dict() == before


def test_ordered_live_authority_rejects_manifest_claim_mismatch_without_mutation() -> None:
    _, ledger, plan, manifest = _authority()
    mismatched = deepcopy(manifest)
    mismatched["batches"][0]["claims"][0]["offset_end"] = 2
    mismatched["manifest_identity_sha256"] = _identity(
        mismatched, "manifest_identity_sha256"
    )
    guard = _guard(ledger, mismatched)
    before = guard.state_dict()
    input_ids, target_ids, loss_mask = _live(0)
    with pytest.raises(LedgerError, match="submitted claims differ"):
        authorize_ordered_live_batch(
            guard,
            plan,
            batch_index=0,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_ordered_next_exposure_identity_sha256=_ordered_identity(
                guard, plan, 0
            ),
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
        )
    assert guard.state_dict() == before


@pytest.mark.parametrize(
    ("input_ids", "target_ids", "message"),
    [
        ([[99]], [[11]], "predictor/context"),
        ([[10]], [[99]], "target content"),
    ],
)
def test_ordered_live_authority_rejects_content_mutation_without_state_change(
    input_ids: list[list[int]], target_ids: list[list[int]], message: str
) -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match=message):
        authorize_ordered_live_batch(
            guard,
            plan,
            batch_index=0,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_ordered_next_exposure_identity_sha256=_ordered_identity(
                guard, plan, 0
            ),
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=[[1]],
        )
    assert guard.state_dict() == before


def test_content_bound_resume_authorizes_only_saved_next_ordered_live_batch() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    _authorize(guard, plan, 0)
    checkpoint = _binding("g001", 1, 1)
    guard.bind_checkpoint_state(checkpoint)
    state = ordered_resume_state_dict(
        guard,
        plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )

    resumed = _guard(ledger, manifest)
    load_ordered_resume_state(
        resumed,
        plan,
        state,
        expected_ordered_resume_identity_sha256=state[
            "ordered_resume_identity_sha256"
        ],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
        expected_trainer_state_binding=checkpoint,
    )
    assert resumed.claim_sequence == 1
    assert state["next_ordered_exposure_identity_sha256"] == _ordered_identity(
        resumed, plan, 1
    )
    _authorize(resumed, plan, 1)
    assert resumed.claim_sequence == 2
    assert resumed.consumed_loss_positions == 2


def test_legacy_ordered_helper_stays_closed_under_content_authority() -> None:
    _, ledger, plan, manifest = _authority()
    guard = _guard(ledger, manifest)
    before = guard.state_dict()
    with pytest.raises(LedgerError, match="content-aware authorization"):
        authorize_ordered_batch(
            guard,
            plan,
            batch_index=0,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_ordered_next_exposure_identity_sha256=_ordered_identity(
                guard, plan, 0
            ),
        )
    assert guard.state_dict() == before
