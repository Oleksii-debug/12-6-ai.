from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.optim import AdamW

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
    verify_scale_checkpoint,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs)


def _identity() -> ScaleCheckpointIdentity:
    return ScaleCheckpointIdentity(
        git_sha="1" * 40,
        model_spec_sha256="2" * 64,
        init_spec_sha256="3" * 64,
        tokenizer_config_sha256="4" * 64,
        tokenizer_vocab_sha256="5" * 64,
        data_manifest_sha256="6" * 64,
        packing_sha256="7" * 64,
        training_config_sha256="8" * 64,
        environment_lock_sha256="9" * 64,
        seed=123,
        step=2,
        tokens_seen=16,
    )


def _trained() -> tuple[_TinyModel, AdamW]:
    torch.manual_seed(7)
    model = _TinyModel()
    optimizer = AdamW(model.parameters(), lr=0.01)
    inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    targets = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 20
    for _ in range(2):
        optimizer.zero_grad()
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
    return model, optimizer


def _fresh() -> tuple[_TinyModel, AdamW]:
    torch.manual_seed(99)
    model = _TinyModel()
    return model, AdamW(model.parameters(), lr=0.01)


def _fingerprint(model: nn.Module, optimizer: AdamW) -> tuple[list, list[int]]:
    weights = [parameter.detach().cpu().tolist() for parameter in model.parameters()]
    steps = sorted(int(item["step"].item()) for item in optimizer.state_dict()["state"].values())
    return weights, steps


def _init_process_group(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _save_worker(rank: int, world_size: int, init_file: str, checkpoint: str, queue) -> None:
    _init_process_group(rank, world_size, init_file)
    try:
        model, optimizer = _trained()
        manifest = save_scale_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            plan=ParallelPlan(data_parallel=world_size),
            identity=_identity(),
            metadata={"proof": "local-free-gloo"},
            rank_state={"rng_seed": 700 + rank, "sampler_cursor": 2},
        )
        weights, steps = _fingerprint(model, optimizer)
        queue.put((rank, manifest["aggregate_checkpoint_sha256"], weights, steps))
    finally:
        dist.destroy_process_group()


def _load_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint: str,
    mode: str,
    queue,
) -> None:
    _init_process_group(rank, world_size, init_file)
    try:
        model, optimizer = _fresh()
        restored: dict = {}

        def restore_rank_state(value) -> None:
            restored.update(value)

        callback = restore_rank_state if mode == ResumeMode.EXACT_TOPOLOGY.value else None
        result = load_scale_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            target_plan=ParallelPlan(data_parallel=world_size),
            mode=ResumeMode(mode),
            expected_identity_sha256=_identity().sha256,
            restore_rank_state=callback,
        )
        weights, steps = _fingerprint(model, optimizer)
        queue.put(
            (
                rank,
                result.exact_topology,
                result.exact_trajectory_claim_allowed,
                result.rank_state,
                restored,
                weights,
                steps,
            )
        )
    finally:
        dist.destroy_process_group()


def _run_group(worker, world_size: int, checkpoint: Path, *args):
    descriptor, init_file = tempfile.mkstemp(prefix="d18-dcp-gloo-")
    os.close(descriptor)
    os.unlink(init_file)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(rank, world_size, init_file, str(checkpoint), *args, queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
    exit_codes = [process.exitcode for process in processes]
    assert exit_codes == [0] * world_size
    output = [queue.get(timeout=5) for _ in processes]
    if os.path.exists(init_file):
        os.unlink(init_file)
    return output


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_real_dcp_exact_resume_reshard_and_integrity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scale-checkpoint"
    saved = _run_group(_save_worker, 2, checkpoint)
    manifest = verify_scale_checkpoint(checkpoint)
    assert len({item[1] for item in saved}) == 1
    assert manifest["identity"]["model_spec_sha256"] == "2" * 64
    assert manifest["identity"]["init_spec_sha256"] == "3" * 64
    assert manifest["metadata"] == {"proof": "local-free-gloo"}
    assert any(row["path"].endswith(".distcp") for row in manifest["artifacts"])

    reference_model, reference_optimizer = _trained()
    expected_weights, expected_steps = _fingerprint(reference_model, reference_optimizer)

    exact = _run_group(_load_worker, 2, checkpoint, ResumeMode.EXACT_TOPOLOGY.value)
    for row in exact:
        rank = row[0]
        assert row[1] is True
        assert row[2] is True
        assert row[3] == {"rng_seed": 700 + rank, "sampler_cursor": 2}
        assert row[4] == row[3]
        assert row[5] == expected_weights
        assert row[6] == expected_steps == [2, 2]

    resharded = _run_group(_load_worker, 1, checkpoint, ResumeMode.RESHARD.value)[0]
    assert resharded[1] is False
    assert resharded[2] is False
    assert resharded[3] is None
    assert resharded[5] == expected_weights
    assert resharded[6] == [2, 2]

    shard = next(checkpoint.glob("*.distcp"))
    with shard.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        verify_scale_checkpoint(checkpoint)
