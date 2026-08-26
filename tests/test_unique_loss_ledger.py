from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from twelve_six.data.unique_loss_ledger import (
    ExposureAccountingError,
    ExposureBudgetGuard,
    build_unique_loss_ledger,
    ledger_identity,
    validate_unique_loss_ledger,
)
from twelve_six.training.exposure import (
    assert_guarded_checkpoint_safe,
    count_batch_optimized_targets,
    restore_guarded_exposure_state,
    train_microbatch_with_exposure,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEDGER_ID = "9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0"


def _inputs():
    registry = json.loads((ROOT / "data/registry/real_snapshots.v1.json").read_text())
    reservations = json.loads(
        (ROOT / "configs/data/data294_reserved_eval_ranges_v1.json").read_text()
    )
    return registry, reservations


def _ledger():
    registry, reservations = _inputs()
    return build_unique_loss_ledger(registry, reservations)


def _first_claim(ledger, start: int, end: int):
    segment = ledger["documents"][0]["segments"][0]
    return {
        "segment_identity_sha256": segment["segment_identity_sha256"],
        "loss_position_start": start,
        "loss_position_end_exclusive": end,
    }


def test_data229_one_pass_accounting_is_exact_and_deterministic():
    ledger = _ledger()
    validate_unique_loss_ledger(ledger)
    assert ledger["ledger_identity_sha256"] == EXPECTED_LEDGER_ID
    assert ledger["ledger_identity_sha256"] == ledger_identity(ledger)
    assert ledger["normalized_bytes"] == 173_358
    assert ledger["reserved_eval_bytes"] == 0
    assert ledger["eligible_bytes"] == 173_358
    assert ledger["one_pass_max_unique_optimized_targets"] == 173_355
    assert ledger["padding_optimized_targets"] == 0
    assert ledger["cross_document_optimized_targets"] == 0
    assert ledger["by_language"] == [
        {
            "key": "en",
            "normalized_bytes": 84_793,
            "reserved_eval_bytes": 0,
            "eligible_bytes": 84_793,
            "unique_optimized_targets": 84_791,
            "excluded_causal_targets_due_to_reservation": 0,
            "document_count": 2,
        },
        {
            "key": "uk",
            "normalized_bytes": 88_565,
            "reserved_eval_bytes": 0,
            "eligible_bytes": 88_565,
            "unique_optimized_targets": 88_564,
            "excluded_causal_targets_due_to_reservation": 0,
            "document_count": 1,
        },
    ]
    assert ledger["by_modality"][0]["key"] == "text"
    assert ledger["by_modality"][0]["unique_optimized_targets"] == 173_355
    families = {row["key"]: row for row in ledger["by_family"]}
    assert families["en.standardebooks.manual"]["unique_optimized_targets"] == 84_791
    assert families["ua.rada.open-data.laws-texts"]["unique_optimized_targets"] == 88_564


def test_reserved_eval_bytes_split_document_and_remove_boundary_targets():
    registry, reservations = _inputs()
    changed = copy.deepcopy(reservations)
    changed["sources"][0]["reserved_eval_byte_ranges"] = [[10, 20]]
    ledger = build_unique_loss_ledger(registry, changed)
    document = ledger["documents"][0]
    assert document["reserved_eval_bytes"] == 10
    assert document["unique_optimized_targets"] == 47_990
    assert document["excluded_causal_targets_due_to_reservation"] == 11
    assert [(row["byte_start"], row["byte_end"]) for row in document["segments"]] == [
        (0, 10),
        (20, 48_002),
    ]
    assert ledger["one_pass_max_unique_optimized_targets"] == 173_344


def test_reservations_fail_closed_on_overlap_or_inventory_drift():
    registry, reservations = _inputs()
    overlap = copy.deepcopy(reservations)
    overlap["sources"][0]["reserved_eval_byte_ranges"] = [[10, 20], [19, 30]]
    with pytest.raises(ExposureAccountingError, match="overlap"):
        build_unique_loss_ledger(registry, overlap)

    missing = copy.deepcopy(reservations)
    missing["sources"].pop()
    with pytest.raises(ExposureAccountingError, match="exactly cover training sources"):
        build_unique_loss_ledger(registry, missing)


def test_guard_rejects_replay_budget_overrun_and_mismatched_batch_count_atomically():
    ledger = _ledger()
    guard = ExposureBudgetGuard(ledger, 10)
    assert guard.authorize_batch(
        [_first_claim(ledger, 1, 6)], actual_optimized_targets=5
    ) == 5
    state_before = guard.state_dict()

    with pytest.raises(ExposureAccountingError, match="replayed"):
        guard.authorize_batch(
            [_first_claim(ledger, 3, 4)], actual_optimized_targets=1
        )
    assert guard.state_dict() == state_before

    with pytest.raises(ExposureAccountingError, match="!= Trainer"):
        guard.authorize_batch(
            [_first_claim(ledger, 6, 8)], actual_optimized_targets=1
        )
    assert guard.state_dict() == state_before

    guard.authorize_batch([_first_claim(ledger, 6, 11)], actual_optimized_targets=5)
    with pytest.raises(ExposureAccountingError, match="budget exceeded"):
        guard.authorize_batch([_first_claim(ledger, 11, 12)], actual_optimized_targets=1)


def test_guard_state_roundtrip_binds_ledger_and_budget():
    ledger = _ledger()
    guard = ExposureBudgetGuard(ledger, 20)
    guard.authorize_batch([_first_claim(ledger, 1, 8)], actual_optimized_targets=7)
    state = guard.state_dict()
    restored = ExposureBudgetGuard(ledger, 20, state=state)
    assert restored.state_dict() == state

    bad = copy.deepcopy(state)
    bad["authorized_budget"] = 19
    with pytest.raises(ExposureAccountingError, match="self-identity"):
        ExposureBudgetGuard(ledger, 20, state=bad)


def test_trainer_bridge_counts_padding_as_zero_and_keeps_guard_in_lockstep():
    ledger = _ledger()
    guard = ExposureBudgetGuard(ledger, 3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
        "target_ids": torch.tensor([[2, 3, 4, -100, -100]]),
        "loss_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    assert count_batch_optimized_targets(batch) == 3

    class FakeTrainer:
        def __init__(self):
            self.tokens_seen = 0
            self.safe = False

        def train_microbatch(self, current_batch):
            tokens = count_batch_optimized_targets(current_batch)
            self.tokens_seen += tokens
            return SimpleNamespace(tokens=tokens)

        def assert_checkpoint_safe(self):
            self.safe = True

    trainer = FakeTrainer()
    metrics = train_microbatch_with_exposure(
        trainer,
        guard,
        batch,
        [_first_claim(ledger, 1, 4)],
    )
    assert metrics.tokens == 3
    assert trainer.tokens_seen == guard.consumed_targets == 3
    assert_guarded_checkpoint_safe(trainer, guard)
    assert trainer.safe is True

    state = guard.state_dict()
    restored = ExposureBudgetGuard(ledger, 3)
    restore_guarded_exposure_state(trainer, restored, state)
    assert restored.consumed_targets == 3
