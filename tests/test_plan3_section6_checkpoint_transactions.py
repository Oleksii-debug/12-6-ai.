from __future__ import annotations

import json
import multiprocessing
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_ordered_exposure_resume import _checkpoint_after_first, _guard, _ledger, _plan
from test_plan3_section5_run_control import RUN_SHA, TinyDecoder, _identity, _run, _trainer

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    hash_json,
    load_trainer_checkpoint,
    plan_checkpoint_retention,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.selection import CheckpointCandidate
from twelve_six.checkpoint.transaction import (
    TRANSACTION_KEY,
    bind_checkpoint_transaction,
    checkpoint_transaction,
)
from twelve_six.data.ordered_exposure_resume import load_ordered_resume_state
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.run_control import RunControlError, RunController, RunPhase


def _path(controller, epoch, step):
    return controller.namespace / "checkpoints" / f"{epoch:08d}-{step:08d}"


def test_native_checkpoint_contains_exact_d04_cursor_and_full_trainer_state(tmp_path: Path) -> None:
    ledger = _ledger()
    plan = _plan(ledger)
    guard_binding, cursor = _checkpoint_after_first(ledger, plan)
    config = TrainerConfig(max_steps=3, scheduler="linear_warmup", warmup_steps=2)
    trainer = Trainer(TinyDecoder(), config)
    trainer.train_microbatch({"input_ids": torch.tensor([[0, 1]])})
    identity = replace(_identity(), step=1, tokens_seen=1)
    bound = bind_checkpoint_transaction(
        identity,
        generation=1,
        parent_generation=0,
        parent_checkpoint_id="b" * 64,
        exposure_cursor=cursor,
    )
    path = tmp_path / "transaction"
    saved = save_trainer_checkpoint(path, model=trainer.model, trainer=trainer, identity=bound)
    fresh = Trainer(TinyDecoder(), config)
    loaded = load_trainer_checkpoint(
        path,
        model=fresh.model,
        trainer=fresh,
        expected_checkpoint_id=saved["checkpoint_id"],
        expected_run_manifest_hash=RUN_SHA,
        expected_step=1,
        expected_tokens_seen=1,
    )
    transaction = checkpoint_transaction(loaded.manifest)
    assert transaction is not None
    assert transaction.exposure_cursor == cursor
    assert transaction.exposure_cursor_sha256 == hash_json(cursor)
    assert transaction.training_config_sha256 == hash_json(identity.training_config)
    assert (
        transaction.generation,
        transaction.parent_generation,
        transaction.parent_checkpoint_id,
    ) == (
        1,
        0,
        "b" * 64,
    )
    assert loaded.trainer_state["optimizer"]["state"]
    assert loaded.trainer_state["scheduler"] == trainer.scheduler.state_dict()
    assert loaded.rng_state
    assert fresh.optimizer_step == 1
    assert loaded.manifest["identity"]["dataset_manifest_hash"] == identity.dataset_manifest_hash
    for left, right in zip(fresh.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    guard = _guard(ledger)
    load_ordered_resume_state(
        guard,
        plan,
        transaction.exposure_cursor,
        expected_ordered_resume_identity_sha256=cursor["ordered_resume_identity_sha256"],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
        expected_trainer_state_binding=guard_binding,
    )
    assert (guard.claim_sequence, guard.consumed_loss_positions) == (1, 1)


def test_periodic_generation_is_independent_but_cannot_exceed_optimizer_steps(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    trainer.train_microbatch({"input_ids": torch.tensor([[0, 1]])})
    trainer.train_microbatch({"input_ids": torch.tensor([[1, 2]])})
    cursor = {"schema": "test.stream.v1", "next_batch_index": 2}
    identity = replace(_identity(), step=1, tokens_seen=2)
    bound = bind_checkpoint_transaction(
        identity,
        generation=0,
        parent_generation=None,
        parent_checkpoint_id=None,
        exposure_cursor=cursor,
    )
    saved = save_trainer_checkpoint(
        tmp_path / "periodic",
        model=trainer.model,
        trainer=trainer,
        identity=bound,
    )
    assert checkpoint_transaction(saved).generation == 0

    config = dict(bound.training_config)
    config[TRANSACTION_KEY] = {
        **config[TRANSACTION_KEY],
        "generation": 2,
        "parent_generation": 1,
        "parent_checkpoint_id": "b" * 64,
    }
    with pytest.raises(CheckpointIntegrityError, match="transaction generation"):
        save_trainer_checkpoint(
            tmp_path / "future",
            model=trainer.model,
            trainer=trainer,
            identity=replace(identity, training_config=config),
        )
    assert not (tmp_path / "future").exists()


@pytest.mark.parametrize("mutation", ["parent", "cursor", "metadata-less"])
def test_pending_adoption_requires_the_reserved_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    prepared = controller.prepare(_trainer(), _identity())
    save = RunController._save_checkpoint

    def publish_wrong_transaction(path, trainer, identity):
        config = dict(identity.training_config)
        data = config.pop(TRANSACTION_KEY)
        legacy = replace(identity, training_config=config)
        if mutation == "metadata-less":
            wrong = legacy
        else:
            cursor = dict(data["exposure_cursor"])
            parent = data["parent_checkpoint_id"]
            if mutation == "cursor":
                cursor["next_microbatch_index"] += 1
            else:
                parent = "b" * 64
            wrong = bind_checkpoint_transaction(
                legacy,
                generation=data["generation"],
                parent_generation=data["parent_generation"],
                parent_checkpoint_id=parent,
                exposure_cursor=cursor,
            )
        save(path, trainer, wrong)
        raise OSError("interrupted writer after publication")

    with monkeypatch.context() as scoped:
        scoped.setattr(RunController, "_save_checkpoint", staticmethod(publish_wrong_transaction))
        with pytest.raises(OSError, match="interrupted writer"):
            _run(controller, _trainer(), prepared)
    failed = controller.status()
    fresh = _trainer()
    assert failed.pending_cursor_sha256 is not None
    with pytest.raises(RunControlError, match="reserved"):
        _run(controller, fresh, failed, recover=True)
    assert fresh.optimizer_step == 0
    assert controller.status() == failed


def test_retention_protects_authoritative_parent_when_newer_checkpoint_is_visible(
    tmp_path: Path,
) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    prepared = controller.prepare(_trainer(), _identity())
    completed = _run(controller, _trainer(), prepared)
    candidates = [
        CheckpointCandidate.from_manifest(verify_checkpoint(_path(controller, epoch, step)))
        for epoch, step in [(0, 0), (1, 1), (1, 2)]
    ]
    initial, middle, final = candidates
    assert len({item.lineage_key for item in candidates}) == 1
    assert len({item.training_config_hash for item in candidates}) == 3
    assert final.checkpoint_id == completed.checkpoint_id
    assert final.parent_checkpoint_id == middle.checkpoint_id
    with pytest.raises(CheckpointCompatibilityError, match="requires an authoritative checkpoint"):
        plan_checkpoint_retention(candidates, keep_recent=1)
    pending = plan_checkpoint_retention(
        candidates,
        keep_recent=1,
        authoritative_checkpoint_id=middle.checkpoint_id,
    )
    assert pending.keep == (initial.checkpoint_id, middle.checkpoint_id, final.checkpoint_id)
    assert pending.delete == ()
    committed = plan_checkpoint_retention(
        candidates,
        keep_recent=1,
        authoritative_checkpoint_id=final.checkpoint_id,
    )
    assert committed.keep == (middle.checkpoint_id, final.checkpoint_id)
    assert committed.delete == (initial.checkpoint_id,)
    with pytest.raises(CheckpointCompatibilityError, match="required rollback parent is missing"):
        plan_checkpoint_retention(
            [initial, final],
            keep_recent=1,
            authoritative_checkpoint_id=final.checkpoint_id,
        )
    wrong_generation = replace(
        middle,
        generation=0,
        parent_generation=None,
        parent_checkpoint_id=None,
    )
    with pytest.raises(CheckpointCompatibilityError, match="rollback parent generation disagrees"):
        plan_checkpoint_retention(
            [initial, wrong_generation, final],
            keep_recent=1,
            authoritative_checkpoint_id=final.checkpoint_id,
        )


def test_existing_s5_journal_and_checkpoint_remain_resumable(tmp_path: Path) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    initial = _path(controller, 0, 0)
    initial.parent.mkdir(parents=True)
    trainer = _trainer()
    legacy = save_trainer_checkpoint(
        initial,
        model=trainer.model,
        trainer=trainer,
        identity=_identity(),
    )
    prepared = controller.prepare(trainer, _identity())
    assert prepared.checkpoint_id == legacy["checkpoint_id"]
    with sqlite3.connect(controller.namespace / "control.sqlite3") as connection:
        row = connection.execute("SELECT payload FROM events WHERE generation = 0").fetchone()
        payload = json.loads(row[0])
        payload.pop("exposure_cursor_schema")
        payload.pop("pending_cursor_sha256")
        connection.execute(
            "UPDATE events SET payload = ?, digest = ? WHERE generation = 0",
            (json.dumps(payload, sort_keys=True), hash_json(payload)),
        )
    old = controller.status()
    assert old.exposure_cursor_schema is None
    paused = _run(controller, _trainer(), old, control=lambda _: RunPhase.PAUSED)
    manifest = verify_checkpoint(_path(controller, 1, 1))
    transaction = checkpoint_transaction(manifest)
    assert transaction.parent_checkpoint_id == old.checkpoint_id
    assert transaction.exposure_cursor["next_microbatch_index"] == 2
    candidates = [CheckpointCandidate.from_manifest(record) for record in (legacy, manifest)]
    plan = plan_checkpoint_retention(
        candidates,
        keep_recent=1,
        authoritative_checkpoint_id=paused.checkpoint_id,
    )
    assert plan.keep == (old.checkpoint_id, paused.checkpoint_id)
    assert plan.delete == ()


_EXTERNAL_POSITIONS = ("stream-start", "row-29", "shard-b-entry", "row-64", "stream-end")


def _external_cursor(position):
    return {"schema_version": "test.opaque-exposure.v1", "resume_at": _EXTERNAL_POSITIONS[position]}


def _run_external(controller, receipt, *, pause=False):
    index = None
    opened = []

    def restore(cursor):
        nonlocal index
        index = _EXTERNAL_POSITIONS.index(cursor["resume_at"])

    def read(_trainer):
        return _external_cursor(index)

    def batches(_current):
        nonlocal index
        assert index is not None
        opened.append(_EXTERNAL_POSITIONS[index])
        while index < 4:
            tokens = torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]])
            index += 1
            yield {"input_ids": tokens}

    result = controller.run(
        _trainer(),
        _identity(),
        batches,
        expected_receipt_sha256=receipt.receipt_sha256,
        control=(lambda _: RunPhase.PAUSED) if pause else None,
        read_exposure_cursor=read,
        restore_exposure_cursor=restore,
    )
    return result, opened


def _pause_external_in_fresh_process(root, receipt):
    torch.set_num_threads(1)
    controller = RunController(root, RUN_SHA)
    paused, opened = _run_external(controller, receipt, pause=True)
    assert paused.phase is RunPhase.PAUSED
    assert opened == ["stream-start"]


def test_fresh_batch_factory_uses_exact_verified_external_cursor(tmp_path: Path) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    prepared = controller.prepare(_trainer(), _identity(), exposure_cursor=_external_cursor(0))
    process = multiprocessing.get_context("spawn").Process(
        target=_pause_external_in_fresh_process,
        args=(tmp_path, prepared),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("checkpoint writer did not finish")
    assert process.exitcode == 0
    paused = controller.status()
    assert (paused.committed_step, paused.tokens_seen) == (1, 4)
    persisted = checkpoint_transaction(verify_checkpoint(_path(controller, 1, 1)))
    assert persisted.exposure_cursor["resume_at"] == "shard-b-entry"
    with pytest.raises(RunControlError, match="cursor provider differs"):
        _run(controller, _trainer(), paused)
    complete, opened = _run_external(RunController(tmp_path, RUN_SHA), paused)
    assert opened == ["shard-b-entry"]
    assert complete.phase is RunPhase.COMPLETED
    final = checkpoint_transaction(verify_checkpoint(_path(controller, 2, 2)))
    assert final.exposure_cursor["resume_at"] == "stream-end"
