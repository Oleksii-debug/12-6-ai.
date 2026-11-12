from __future__ import annotations

import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint, save_trainer_checkpoint
from twelve_six.training import Trainer, TrainerConfig, run_control
from twelve_six.training.config import TrainingRecipe
from twelve_six.training.run_control import RunControlError, RunController, RunPhase

RUN_SHA = "a" * 64


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.head = nn.Linear(4, 4)

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))


def _trainer() -> Trainer:
    return Trainer(TinyDecoder(), TrainerConfig(max_steps=2, gradient_accumulation_steps=2))


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="1" * 40,
        model_spec={"kind": "tiny-test"},
        parameter_count=36,
        tokenizer_hash="2" * 64,
        tokenizer_vocab_hash="3" * 64,
        dataset_manifest_hash="4" * 64,
        run_manifest_hash=RUN_SHA,
        training_config={"gradient_accumulation_steps": 2, "max_steps": 2},
        seed=11,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "adamw"},
        scheduler=None,
    )


def _batches(receipt):
    for index in range(receipt.committed_step * 2, 4):
        yield {"input_ids": torch.tensor([[index % 4, (index + 1) % 4, (index + 2) % 4]])}


def _run(controller, trainer, receipt, **options):
    return controller.run(
        trainer,
        _identity(),
        _batches,
        expected_receipt_sha256=receipt.receipt_sha256,
        **options,
    )


def _stop_at_first_step(boundary):
    return RunPhase.STOPPED if boundary.committed_step == 1 else None


def _worker(root, receipt, start, result, crash):
    torch.set_num_threads(1)
    controller = RunController(root, RUN_SHA)
    control = None
    if crash in {"published", "published_stop"}:
        original = RunController._save_checkpoint

        def exit_after_save(path, trainer, identity):
            original(path, trainer, identity)
            os._exit(17)

        RunController._save_checkpoint = staticmethod(exit_after_save)
    if crash == "published_stop":
        control = _stop_at_first_step
    if crash == "during_control":

        def control(_boundary):
            os._exit(18)

    start.wait(timeout=30)
    try:
        result.put(_run(controller, _trainer(), receipt, control=control).phase.value)
    except RunControlError as exc:
        result.put(str(exc))


def _join(processes):
    for process in processes:
        process.join(timeout=30)
    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.kill()
        process.join()
    assert not remaining, "writer process did not finish"


def test_pause_restart_replays_exact_checkpoint_and_rejects_stale_writer(tmp_path: Path) -> None:
    initial = _trainer()
    uninterrupted = RunController(tmp_path / "baseline", RUN_SHA)
    uninterrupted_start = uninterrupted.prepare(initial, _identity())
    controller = RunController(tmp_path / "paused", RUN_SHA)
    start = controller.prepare(initial, _identity())
    assert start.phase is RunPhase.PREPARED
    assert start.control_epoch == 0
    assert controller.status() == start
    assert start.checkpoint_id == uninterrupted_start.checkpoint_id
    direct_trainer = _trainer()
    direct = _run(uninterrupted, direct_trainer, uninterrupted_start)
    paused = _run(controller, _trainer(), start, control=lambda _: RunPhase.PAUSED)
    assert (paused.phase, paused.control_epoch, paused.committed_step, paused.tokens_seen) == (
        RunPhase.PAUSED,
        1,
        1,
        4,
    )
    stale_trainer = _trainer()
    with pytest.raises(RunControlError, match="stale run receipt"):
        _run(controller, stale_trainer, start)
    assert stale_trainer.optimizer_step == 0
    restarted = RunController(tmp_path / "paused", RUN_SHA)
    fresh = _trainer()
    final = _run(
        restarted,
        fresh,
        paused,
        control=lambda boundary: RunPhase.PAUSED if boundary.committed_step == 1 else None,
    )
    assert restarted.status() == final
    assert (final.phase, final.control_epoch, final.committed_step) == (RunPhase.COMPLETED, 2, 2)
    assert final.checkpoint_id == direct.checkpoint_id
    for left, right in zip(
        fresh.model.parameters(), direct_trainer.model.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_stop_is_terminal_and_failed_input_restarts_at_committed_step(tmp_path: Path) -> None:
    controller = RunController(tmp_path / "stopped", RUN_SHA)
    start = controller.prepare(_trainer(), _identity())
    stopped = _run(controller, _trainer(), start, control=lambda _: RunPhase.STOPPED)
    assert (stopped.phase, stopped.committed_step) == (RunPhase.STOPPED, 1)
    with pytest.raises(RunControlError, match="phase"):
        _run(controller, _trainer(), stopped)
    with pytest.raises(RunControlError, match="phase"):
        _run(controller, _trainer(), stopped, recover=True)
    assert controller.status() == stopped

    controller = RunController(tmp_path / "failed", RUN_SHA)
    start = controller.prepare(_trainer(), _identity())
    with pytest.raises(RunControlError, match="input ended"):
        controller.run(
            _trainer(),
            _identity(),
            lambda receipt: list(_batches(receipt))[:3],
            expected_receipt_sha256=start.receipt_sha256,
        )
    failed = controller.status()
    assert (failed.phase, failed.committed_step, failed.pending_step) == (RunPhase.FAILED, 1, 2)
    final = _run(controller, _trainer(), failed, recover=True)
    assert (final.phase, final.committed_step, final.control_epoch) == (RunPhase.COMPLETED, 2, 2)


def test_rejects_different_checkpoint_before_mutating_fresh_trainer(tmp_path: Path) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    trainer = _trainer()
    start = controller.prepare(trainer, _identity())
    checkpoint = controller.namespace / "checkpoints" / "00000000-00000000"
    checkpoint.rename(checkpoint.with_name("original"))
    save_trainer_checkpoint(
        checkpoint,
        model=trainer.model,
        trainer=trainer,
        identity=replace(_identity(), seed=19),
    )
    fresh = _trainer()
    original = fresh.model.embedding.weight.detach().clone()
    with pytest.raises(RuntimeError, match="checkpoint ID differs"):
        _run(controller, fresh, start)
    torch.testing.assert_close(fresh.model.embedding.weight, original, atol=0, rtol=0)
    assert controller.status() == start


def test_prepared_receipt_requires_durable_ancestor_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RunController(tmp_path / "outer" / "inner", RUN_SHA)
    flushed = []
    actual_fsync = run_control.fsync_parent_directory

    def record_parent_flush(path):
        actual_fsync(path)
        flushed.append(Path(path).name)

    def fail_before_checkpoint_directory_is_durable(path):
        if Path(path).name == "checkpoints":
            raise OSError("checkpoint directory parent flush failed")
        record_parent_flush(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            run_control,
            "fsync_parent_directory",
            fail_before_checkpoint_directory_is_durable,
        )
        with pytest.raises(OSError, match="parent flush"):
            controller.prepare(_trainer(), _identity())
    with pytest.raises(RunControlError, match="has not been prepared"):
        controller.status()
    assert flushed[-3:] == ["outer", "inner", RUN_SHA]
    monkeypatch.setattr(run_control, "fsync_parent_directory", record_parent_flush)
    prepared = controller.prepare(_trainer(), _identity())
    assert flushed[-3:] == ["inner", RUN_SHA, "checkpoints"]
    assert prepared == controller.status()


def test_run_identity_rejects_trainer_that_differs_from_bound_recipe(tmp_path: Path) -> None:
    trainer = _trainer()
    recipe = TrainingRecipe(
        trainer=trainer.config,
        microbatch_size=1,
        sequence_length=3,
        target_unique_loss_positions=8,
        maximum_total_exposures=8,
        max_exposures_per_unique_position=1,
    )
    identity = replace(_identity(), training_config={"training": recipe.run_training_fields()})
    controller = RunController(tmp_path, RUN_SHA)
    wrong = Trainer(TinyDecoder(), replace(trainer.config, max_steps=1))
    with pytest.raises(RunControlError, match="differs from bound training recipe"):
        controller.prepare(wrong, identity)
    assert not controller.namespace.exists()
    prepared = controller.prepare(trainer, identity)
    with pytest.raises(RunControlError, match="differs from bound training recipe"):
        controller.run(wrong, identity, _batches, expected_receipt_sha256=prepared.receipt_sha256)
    assert controller.status() == prepared
    assert (
        controller.run(
            _trainer(),
            identity,
            _batches,
            expected_receipt_sha256=prepared.receipt_sha256,
        ).phase
        is RunPhase.COMPLETED
    )


def test_recovery_does_not_promote_checkpoint_without_trainer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    start = controller.prepare(_trainer(), _identity())

    def publish_without_trainer(path, trainer, identity):
        save_checkpoint(path, model=trainer.model, identity=identity)
        raise OSError("writer interrupted")

    with monkeypatch.context() as scoped:
        scoped.setattr(RunController, "_save_checkpoint", staticmethod(publish_without_trainer))
        with pytest.raises(OSError, match="writer interrupted"):
            _run(controller, _trainer(), start)
    failed = controller.status()
    assert (failed.phase, failed.committed_step, failed.pending_step) == (RunPhase.FAILED, 0, 1)
    fresh = _trainer()
    original = fresh.model.embedding.weight.detach().clone()
    with pytest.raises(RuntimeError, match="trainer state"):
        _run(controller, fresh, failed, recover=True)
    torch.testing.assert_close(fresh.model.embedding.weight, original, atol=0, rtol=0)
    assert controller.status() == failed


def test_two_processes_with_same_receipt_have_one_writer(tmp_path: Path) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    receipt = controller.prepare(_trainer(), _identity())
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_worker, args=(tmp_path, receipt, start, result, None))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    _join(processes)
    assert [process.exitcode for process in processes] == [0, 0]
    assert {result.get(timeout=5) for _ in processes} == {"COMPLETED", "stale run receipt"}
    assert controller.status().committed_step == 2
    assert controller.status().control_epoch == 1


def test_process_exit_after_publication_adopts_step_without_repeating_optimizer(
    tmp_path: Path,
) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    receipt = controller.prepare(_trainer(), _identity())
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    process = context.Process(target=_worker, args=(tmp_path, receipt, start, result, "published"))
    process.start()
    start.set()
    _join([process])
    assert process.exitcode == 17
    interrupted = controller.status()
    assert (interrupted.phase, interrupted.committed_step, interrupted.pending_step) == (
        RunPhase.RUNNING,
        0,
        1,
    )
    resumed_from = []

    def resume_batches(current):
        resumed_from.append(current.committed_step)
        return _batches(current)

    final = controller.run(
        _trainer(),
        _identity(),
        resume_batches,
        recover=True,
        expected_receipt_sha256=interrupted.receipt_sha256,
    )
    assert resumed_from == [1]
    assert (final.phase, final.committed_step, final.control_epoch) == (RunPhase.COMPLETED, 2, 2)


@pytest.mark.parametrize(("mode", "exitcode"), [("published_stop", 17), ("during_control", 18)])
def test_recovery_replays_interrupted_control_before_loading_next_batches(
    tmp_path: Path,
    mode: str,
    exitcode: int,
) -> None:
    controller = RunController(tmp_path, RUN_SHA)
    receipt = controller.prepare(_trainer(), _identity())
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result = context.Queue()
    process = context.Process(target=_worker, args=(tmp_path, receipt, start, result, mode))
    process.start()
    start.set()
    _join([process])
    assert process.exitcode == exitcode
    interrupted = controller.status()
    assert interrupted.phase is RunPhase.RUNNING
    with pytest.raises(RunControlError, match="requires the interrupted control policy"):
        _run(controller, _trainer(), interrupted, recover=True)
    assert controller.status() == interrupted

    def unexpected_batches(_receipt):
        pytest.fail("recovery consumed input before handling the prior stop decision")

    if mode == "during_control":

        def failed_replay(_boundary):
            raise RuntimeError("replayed control failed")

        with pytest.raises(RuntimeError, match="replayed control failed"):
            controller.run(
                _trainer(),
                _identity(),
                unexpected_batches,
                recover=True,
                control=failed_replay,
                expected_receipt_sha256=interrupted.receipt_sha256,
            )
        interrupted = controller.status()
        assert interrupted.phase is RunPhase.FAILED
        assert interrupted.committed_step == 1

    stopped = controller.run(
        _trainer(),
        _identity(),
        unexpected_batches,
        recover=True,
        control=_stop_at_first_step,
        expected_receipt_sha256=interrupted.receipt_sha256,
    )
    assert (stopped.phase, stopped.committed_step, stopped.control_epoch) == (
        RunPhase.STOPPED,
        1,
        1,
    )
    assert controller.status() == stopped
