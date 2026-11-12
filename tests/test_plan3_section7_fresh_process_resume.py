from __future__ import annotations

import json
import multiprocessing
import os
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ordered_live_exposure_authority import (
    _authority,
    _authorize,
    _guard,
    _live,
    _ordered_identity,
)
from test_plan3_section5_run_control import _identity
from torch import nn

from twelve_six.checkpoint import CheckpointError, hash_json, verify_checkpoint
from twelve_six.checkpoint.transaction import checkpoint_transaction
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.run_control import RunController, RunPhase


class StochasticDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids):
        scale = 0.75 + 0.1 * random.random() + 0.1 * np.random.random()
        return self.head(self.dropout(self.embedding(input_ids)) * scale)


class RecordingTrainer(Trainer):
    def __init__(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        super().__init__(
            StochasticDecoder(),
            TrainerConfig(
                max_steps=3,
                scheduler="cosine",
                warmup_steps=1,
                seed=921,
                deterministic_algorithms=True,
                deterministic_warn_only=False,
            ),
        )
        self.trajectory = []

    def train_microbatch(self, batch):
        metrics = super().train_microbatch(batch)
        self.trajectory.append(asdict(metrics))
        return metrics


def _fixture(path: Path):
    _, ledger, plan, content = _authority()
    data = {"ledger": ledger, "plan": plan, "content": content}
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    sha = hash_json(data)
    return replace(
        _identity(),
        model_spec={"kind": "stochastic-resume-fixture"},
        parameter_count=272,
        dataset_manifest_hash=sha,
        run_manifest_hash=hash_json({"dataset": sha, "seed": 921}),
        training_config={"max_steps": 3, "seed": 921, "scheduler": "cosine"},
        scheduler={"name": "cosine", "warmup_steps": 1},
        seed=921,
    )


class VerifiedFixtureStream:
    def __init__(self, source: Path, expected_sha: str) -> None:
        self.source = source
        self.expected_sha = expected_sha
        self.optimized = []

    def _open(self) -> None:
        data = json.loads(self.source.read_text(encoding="utf-8"))
        if hash_json(data) != self.expected_sha:
            raise ValueError(f"sealed exposure source hash differs at {self.source}")
        self.plan = data["plan"]
        self.guard = _guard(data["ledger"], data["content"])

    def cursor(self, trainer=None):
        index = self.guard.claim_sequence
        if trainer is not None:
            assert (trainer.micro_step, trainer.tokens_seen) == (
                index,
                self.guard.consumed_loss_positions,
            )
        return {
            "schema": "test.d04-verified-prefix.v1",
            "dataset_sha256": self.expected_sha,
            "next_batch_index": index,
            "unique_targets_seen": self.guard.consumed_loss_positions,
            "next_exposure_sha256": (
                _ordered_identity(self.guard, self.plan, index)
                if index < len(self.plan["batches"])
                else None
            ),
        }

    def initial(self):
        self._open()
        return self.cursor()

    def restore(self, expected):
        self._open()
        index = expected["next_batch_index"]
        for position in range(index):
            _authorize(self.guard, self.plan, position)
        if self.cursor() != expected:
            raise ValueError(f"next exposure cursor differs from sealed source at {self.source}")

    def batches(self, _receipt):
        for index in range(self.guard.claim_sequence, len(self.plan["batches"])):
            observed = _authorize(self.guard, self.plan, index)
            self.optimized.append((index, observed))
            inputs, targets, mask = _live(index)
            yield {
                "input_ids": torch.tensor(inputs),
                "target_ids": torch.tensor(targets),
                "loss_mask": torch.tensor(mask),
            }


def _prepare(root, source, identity):
    controller = RunController(root, identity.run_manifest_hash)
    initial = VerifiedFixtureStream(source, identity.dataset_manifest_hash).initial()
    receipt = controller.prepare(RecordingTrainer(101), identity, exposure_cursor=initial)
    return controller, receipt


def _run(controller, source, identity, receipt, trainer, *, mode="complete"):
    stream = VerifiedFixtureStream(source, identity.dataset_manifest_hash)
    result = controller.run(
        trainer,
        identity,
        stream.batches,
        expected_receipt_sha256=receipt.receipt_sha256,
        read_exposure_cursor=stream.cursor,
        restore_exposure_cursor=stream.restore,
        recover=mode == "recover",
        control=(lambda _: RunPhase.PAUSED) if mode == "pause" else None,
    )
    return result, stream.optimized, trainer.trajectory


def _worker(root, source, identity, receipt, mode, output):
    torch.set_num_threads(1)
    if mode == "interrupt":
        save = RunController._save_checkpoint

        def publish_then_exit(path, trainer, checkpoint_identity):
            save(path, trainer, checkpoint_identity)
            os._exit(19)

        RunController._save_checkpoint = staticmethod(publish_then_exit)
    controller = RunController(root, identity.run_manifest_hash)
    result = _run(controller, source, identity, receipt, RecordingTrainer(917), mode=mode)
    output.put((os.getpid(), result))


def _spawn(root, source, identity, receipt, *, mode):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_worker, args=(root, source, identity, receipt, mode, output))
    process.start()
    process.join(timeout=45)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("isolated recovery process did not finish")
    if mode == "interrupt":
        assert process.exitcode == 19
        return None
    assert process.exitcode == 0
    pid, result = output.get(timeout=5)
    assert pid != os.getpid()
    return result


def _checkpoint_path(controller, *, epoch, step):
    return controller.namespace / "checkpoints" / f"{epoch:08d}-{step:08d}"


@pytest.mark.parametrize("interruption", ["pause", "interrupt"])
def test_relocated_fresh_process_replays_exact_exposure_and_trajectory(
    tmp_path: Path,
    interruption: str,
) -> None:
    source = tmp_path / "source.json"
    identity = _fixture(source)
    baseline_root = tmp_path / "baseline"
    baseline, initial = _prepare(baseline_root, source, identity)
    final, baseline_exposures, baseline_metrics = _spawn(
        baseline_root,
        source,
        identity,
        initial,
        mode="complete",
    )

    stopped_root = tmp_path / "stopped"
    stopped, start = _prepare(stopped_root, source, identity)
    assert start.checkpoint_id == initial.checkpoint_id
    first = _spawn(stopped_root, source, identity, start, mode=interruption)
    saved = stopped.status()
    assert saved.committed_step == (1 if interruption == "pause" else 0)
    step1 = verify_checkpoint(_checkpoint_path(stopped, epoch=1, step=1))
    base_step1 = verify_checkpoint(_checkpoint_path(baseline, epoch=1, step=1))
    assert step1["checkpoint_id"] == base_step1["checkpoint_id"]
    assert (
        checkpoint_transaction(step1).exposure_cursor["next_exposure_sha256"]
        == (baseline_exposures[1][1])
    )

    moved_root = tmp_path / "relocated"
    moved = RunController(moved_root, identity.run_manifest_hash)
    shutil.copytree(stopped.namespace, moved.namespace)
    moved_source = tmp_path / "relocated-source.json"
    shutil.copyfile(source, moved_source)
    assert moved.status() == saved
    resumed, exposures, metrics = _spawn(
        moved_root,
        moved_source,
        identity,
        saved,
        mode="complete" if first is not None else "recover",
    )
    assert exposures == baseline_exposures[1:]
    assert metrics == baseline_metrics[1:]
    assert (resumed.phase, resumed.committed_step, resumed.tokens_seen) == (
        RunPhase.COMPLETED,
        3,
        4,
    )
    assert resumed.checkpoint_id == final.checkpoint_id
    assert stopped.status() == saved


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_external_dependency_failure_preserves_committed_progress(
    tmp_path: Path,
    damage: str,
) -> None:
    source = tmp_path / "source.json"
    identity = _fixture(source)
    controller, prepared = _prepare(tmp_path / "state", source, identity)
    paused, _, _ = _run(
        controller,
        source,
        identity,
        prepared,
        RecordingTrainer(917),
        mode="pause",
    )
    original = source.read_bytes()
    if damage == "missing":
        source.unlink()
        error = FileNotFoundError
    else:
        source.write_text("{}", encoding="utf-8")
        error = ValueError
    fresh = RecordingTrainer(18)
    with pytest.raises(error) as failure:
        _run(controller, source, identity, paused, fresh)
    detail = "\n".join(failure.value.__notes__)
    assert paused.checkpoint_id in detail
    assert "Restore the source data" in detail
    failed = controller.status()
    assert failed.phase is RunPhase.FAILED
    assert failed.checkpoint_id == paused.checkpoint_id
    assert failed.committed_step == fresh.optimizer_step == 1
    assert fresh.trajectory == []

    source.write_bytes(original)
    final, exposures, _ = _run(
        controller,
        source,
        identity,
        failed,
        RecordingTrainer(19),
        mode="recover",
    )
    assert final.phase is RunPhase.COMPLETED
    assert [index for index, _ in exposures] == [1, 2]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_checkpoint_dependency_failure_preserves_trainer_and_authoritative_receipt(
    tmp_path: Path,
    damage: str,
) -> None:
    source = tmp_path / "source.json"
    identity = _fixture(source)
    controller, prepared = _prepare(tmp_path / "state", source, identity)
    paused, _, _ = _run(
        controller,
        source,
        identity,
        prepared,
        RecordingTrainer(917),
        mode="pause",
    )
    path = _checkpoint_path(controller, epoch=1, step=1)
    if damage == "missing":
        path.rename(path.with_name("unavailable"))
    else:
        (path / "weights.safetensors").write_bytes(b"corrupt")
    fresh = RecordingTrainer(19)
    original = fresh.model.embedding.weight.detach().clone()
    with pytest.raises(CheckpointError) as failure:
        _run(controller, source, identity, paused, fresh)
    detail = "\n".join(failure.value.__notes__)
    assert paused.checkpoint_id in detail
    assert str(path) in detail
    assert "Restore the exact verified checkpoint" in detail
    assert fresh.optimizer_step == 0
    assert fresh.trajectory == []
    torch.testing.assert_close(fresh.model.embedding.weight, original, atol=0, rtol=0)
    assert controller.status() == paused
