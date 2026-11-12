from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_ordered_live_exposure_authority import _authority, _identity, _live
from test_plan3_section5_run_control import _identity as checkpoint_identity
from test_plan3_section7_fresh_process_resume import StochasticDecoder

from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.checkpoint.transaction import checkpoint_transaction
from twelve_six.data.unique_loss_ledger_v2 import LedgerError
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.config import TrainingRecipe
from twelve_six.training.exposure_accounting import SCHEMA, OrderedExposureRuntime
from twelve_six.training.run_control import RunControlError, RunController, RunPhase


def _fixture():
    _, ledger, plan, content = _authority()
    runtime = OrderedExposureRuntime(
        ledger=ledger,
        plan=plan,
        content_manifest=content,
        expected_ledger_sha256=ledger["ledger_identity_sha256"],
        expected_plan_sha256=plan["plan_identity_sha256"],
        expected_content_sha256=content["manifest_identity_sha256"],
        unique_target_budget=4,
    )
    identity = replace(
        checkpoint_identity(),
        model_spec={"kind": "exposure-accounting-fixture"},
        parameter_count=272,
        tokenizer_hash=content["tokenizer_identity_sha256"],
        dataset_manifest_hash=hash_json({"ledger": ledger, "content": content}),
    )
    return runtime, identity


def _trainer(*, accumulation=1):
    return Trainer(
        StochasticDecoder(),
        TrainerConfig(max_steps=3 // accumulation, gradient_accumulation_steps=accumulation),
    )


def _batch(index):
    inputs, targets, mask = _live(index)
    return {
        "input_ids": torch.tensor(inputs),
        "target_ids": torch.tensor(targets),
        "loss_mask": torch.tensor(mask),
    }


def _source(index):
    return (_batch(position) for position in range(index, 3))


def _path(controller, receipt):
    return (
        controller.namespace
        / "checkpoints"
        / f"{receipt.checkpoint_epoch:08d}-{receipt.committed_step:08d}"
    )


def _cursor(controller, receipt):
    manifest = verify_checkpoint(_path(controller, receipt))
    assert manifest["checkpoint_id"] == receipt.checkpoint_id
    transaction = checkpoint_transaction(manifest)
    assert transaction is not None
    return transaction.exposure_cursor


def test_provider_migration_resumes_exact_next_exposure_and_commits_audit(
    tmp_path: Path,
) -> None:
    runtime, identity = _fixture()
    first = RunController(tmp_path / "first", identity.run_manifest_hash)
    ready = runtime.prepare(first, _trainer(), identity)
    opened = []

    def first_provider(index):
        opened.append(index)
        return _source(index)

    paused = runtime.run(
        first,
        _trainer(),
        identity,
        first_provider,
        expected_receipt_sha256=ready.receipt_sha256,
        control=lambda _: RunPhase.PAUSED,
    )
    assert opened == [0]
    saved = _cursor(first, paused)
    assert saved["schema"] == SCHEMA
    assert saved["progress"]["unique_targets_seen"] == 1
    assert saved["ordered_state"]["next_batch_index"] == 1
    assert saved["ordered_state"]["next_ordered_exposure_identity_sha256"]

    moved = RunController(tmp_path / "first", identity.run_manifest_hash)
    replacement, _ = _fixture()
    visited = []

    def second_provider(index):
        opened.append(index)
        for position in range(index, 3):
            visited.append(position)
            yield _batch(position)

    complete = replacement.run(
        moved,
        _trainer(),
        identity,
        second_provider,
        expected_receipt_sha256=paused.receipt_sha256,
    )
    assert opened == [0, 1]
    assert visited == [1, 2]
    assert (complete.phase, complete.committed_step, complete.tokens_seen) == (
        RunPhase.COMPLETED,
        3,
        4,
    )
    final = _cursor(moved, complete)
    assert final["progress"]["micro_step"] == 3
    assert (
        final["progress"]["unique_targets_seen"] == final["progress"]["unique_target_budget"] == 4
    )
    ordered = final["ordered_state"]
    assert ordered["complete"] is True
    assert ordered["next_ordered_exposure_identity_sha256"] is None
    guard = ordered["guard_state"]
    assert guard["claim_sequence"] == 3
    assert guard["consumed_loss_positions"] == 4
    assert (
        sum(end - start for intervals in guard["claims"].values() for start, end in intervals) == 4
    )
    assert first.status() == complete
    with pytest.raises(RunControlError, match="stale run receipt"):
        runtime.run(
            first,
            _trainer(),
            identity,
            first_provider,
            expected_receipt_sha256=paused.receipt_sha256,
        )
    assert opened == [0, 1]


def test_retry_discards_partial_accumulation_and_rejects_live_content_before_update(
    tmp_path: Path,
) -> None:
    runtime, identity = _fixture()
    controller = RunController(tmp_path, identity.run_manifest_hash)
    ready = runtime.prepare(controller, _trainer(accumulation=3), identity)
    broken = _trainer(accumulation=3)

    def bad_provider(index):
        assert index == 0
        yield _batch(0)
        repeated = _batch(0)
        yield repeated

    with pytest.raises(LedgerError, match="live"):
        runtime.run(
            controller,
            broken,
            identity,
            bad_provider,
            expected_receipt_sha256=ready.receipt_sha256,
        )
    failed = controller.status()
    assert failed.phase is RunPhase.FAILED
    assert (failed.committed_step, failed.tokens_seen, broken.micro_step) == (0, 0, 1)
    assert _cursor(controller, failed)["ordered_state"]["next_batch_index"] == 0
    opened = []

    def repaired(index):
        opened.append(index)
        return _source(index)

    result = runtime.run(
        controller,
        _trainer(accumulation=3),
        identity,
        repaired,
        expected_receipt_sha256=failed.receipt_sha256,
        recover=True,
    )
    assert opened == [0]
    assert (result.committed_step, result.tokens_seen) == (1, 4)
    assert _cursor(controller, result)["ordered_state"]["complete"] is True


def test_retry_adopts_already_published_accounting_before_opening_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, identity = _fixture()
    controller = RunController(tmp_path, identity.run_manifest_hash)
    ready = runtime.prepare(controller, _trainer(), identity)
    original = RunController._save_checkpoint

    def interrupted(path, trainer, snapshot):
        original(path, trainer, snapshot)
        raise OSError("interrupted after publication")

    with monkeypatch.context() as scoped:
        scoped.setattr(RunController, "_save_checkpoint", staticmethod(interrupted))
        with pytest.raises(OSError, match="after publication"):
            runtime.run(
                controller,
                _trainer(),
                identity,
                _source,
                expected_receipt_sha256=ready.receipt_sha256,
            )
    failed = controller.status()
    assert (failed.committed_step, failed.pending_step) == (0, 1)
    opened = []

    def migrated(index):
        opened.append(index)
        return _source(index)

    result = runtime.run(
        controller,
        _trainer(),
        identity,
        migrated,
        expected_receipt_sha256=failed.receipt_sha256,
        recover=True,
    )
    assert opened == [1]
    assert result.tokens_seen == 4
    assert _cursor(controller, result)["ordered_state"]["next_batch_index"] == 3


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [("count", "consumed count"), ("next", "next exposure identity")],
)
def test_self_rehashed_cursor_still_requires_exact_ordered_prefix(
    tmp_path: Path,
    mutation: str,
    diagnostic: str,
) -> None:
    runtime, identity = _fixture()
    original = RunController(tmp_path / "original", identity.run_manifest_hash)
    ready = runtime.prepare(original, _trainer(), identity)
    forged = _cursor(original, ready)
    ordered = forged["ordered_state"]
    if mutation == "count":
        guard = ordered["guard_state"]
        guard["consumed_loss_positions"] = 1
        guard["state_identity_sha256"] = _identity(guard, "state_identity_sha256")
    else:
        ordered["next_ordered_exposure_identity_sha256"] = "f" * 64
    ordered["ordered_resume_identity_sha256"] = _identity(ordered, "ordered_resume_identity_sha256")
    controller = RunController(tmp_path / "forged", identity.run_manifest_hash)
    saved = controller.prepare(_trainer(), identity, exposure_cursor=forged)

    def unexpected(_index):
        pytest.fail("unverified cursor opened the source")

    with pytest.raises(LedgerError, match=diagnostic):
        runtime.run(
            controller,
            _trainer(),
            identity,
            unexpected,
            expected_receipt_sha256=saved.receipt_sha256,
        )
    assert controller.status().committed_step == 0


def test_exposure_content_packing_and_tokenizer_must_match_authorities(tmp_path: Path) -> None:
    _, ledger, plan, content = _authority()
    wrong = {**content, "packing_identity_sha256": "e" * 64}
    wrong["manifest_identity_sha256"] = _identity(wrong, "manifest_identity_sha256")
    with pytest.raises(LedgerError, match="does not match exposure ledger"):
        OrderedExposureRuntime(
            ledger=ledger,
            plan=plan,
            content_manifest=wrong,
            expected_ledger_sha256=ledger["ledger_identity_sha256"],
            expected_plan_sha256=plan["plan_identity_sha256"],
            expected_content_sha256=wrong["manifest_identity_sha256"],
            unique_target_budget=4,
        )
    runtime, identity = _fixture()
    controller = RunController(tmp_path, identity.run_manifest_hash)
    with pytest.raises(LedgerError, match="tokenizer differs"):
        runtime.prepare(controller, _trainer(), replace(identity, tokenizer_hash="e" * 64))
    assert not controller.namespace.exists()


def test_bound_training_recipe_and_run_data_must_match_unique_exposure_plan(tmp_path: Path) -> None:
    runtime, identity = _fixture()
    trainer = _trainer()
    recipe = TrainingRecipe(
        trainer=trainer.config,
        microbatch_size=1,
        sequence_length=2,
        target_unique_loss_positions=4,
        maximum_total_exposures=4,
        max_exposures_per_unique_position=1,
    )
    wrong = replace(recipe, target_unique_loss_positions=3, maximum_total_exposures=3)
    wrong_identity = replace(identity, training_config={"training": wrong.run_training_fields()})
    controller = RunController(tmp_path, identity.run_manifest_hash)
    with pytest.raises(LedgerError, match="target budget differs"):
        runtime.prepare(controller, trainer, wrong_identity)
    mismatched_pin = replace(
        identity,
        training_config={
            "training": recipe.run_training_fields(),
            "data": {"exposure_plan_identity_sha256": "b" * 64},
        },
    )
    with pytest.raises(LedgerError, match="exposure_plan_identity_sha256 differs"):
        runtime.prepare(controller, trainer, mismatched_pin)
    assert not controller.namespace.exists()
    good = replace(identity, training_config={"training": recipe.run_training_fields()})
    ready = runtime.prepare(controller, trainer, good)
    result = runtime.run(
        controller,
        _trainer(),
        good,
        _source,
        expected_receipt_sha256=ready.receipt_sha256,
    )
    assert result.tokens_seen == 4
