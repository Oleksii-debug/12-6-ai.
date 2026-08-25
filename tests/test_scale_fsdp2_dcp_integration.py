from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
    verify_scale_checkpoint,
)
from twelve_six.distributed.fsdp2_training import FSDP2Trainer, apply_fsdp2
from twelve_six.distributed.runtime import build_torch_mesh_spec
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import build_optimizer

_WORLD_SIZE = 2
_SEED = 4242
_SEQUENCE_LENGTH = 16


def _build_stack(stage_path: str, rank: int):
    torch.manual_seed(_SEED)
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init)
    plan = ParallelPlan(
        data_parallel=_WORLD_SIZE,
        shard_model_state_across_data_parallel=True,
    )
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=_WORLD_SIZE)
    full_mesh = mesh_spec.create_device_mesh("cpu")
    model = apply_fsdp2(
        model,
        **mesh_spec.fsdp2_kwargs(full_mesh, reshard_after_forward=True),
    )
    config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=2,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=_SEED,
        deterministic_algorithms=True,
    )
    optimizer = build_optimizer(model, config)
    trainer = FSDP2Trainer(model, config, device="cpu", optimizer=optimizer)
    assert stage.model.parameter_count() == 107_856
    assert rank == dist.get_rank()
    return stage, model, optimizer, trainer, plan


def _batch(index: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(9000 + index)
    return {
        "input_ids": torch.randint(
            0,
            512,
            (2, _SEQUENCE_LENGTH),
            generator=generator,
            dtype=torch.long,
        )
    }


def _snapshot(model: TwelveSixDecoder) -> tuple[torch.Tensor, ...]:
    result: list[torch.Tensor] = []
    for parameter in model.parameters():
        to_local = getattr(parameter, "to_local", None)
        local = to_local() if callable(to_local) else parameter
        result.append(local.detach().cpu().clone())
    return tuple(result)


def _assert_snapshot_equal(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.shape == right.shape
        assert torch.equal(left, right)


def _identity(stage, trainer: FSDP2Trainer) -> ScaleCheckpointIdentity:
    source_sha = os.environ.get("GITHUB_SHA", "a" * 40)
    if len(source_sha) not in {40, 64}:
        source_sha = "a" * 40
    return ScaleCheckpointIdentity(
        git_sha=source_sha.lower(),
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256="1" * 64,
        tokenizer_config_sha256="2" * 64,
        tokenizer_vocab_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        packing_sha256="5" * 64,
        training_config_sha256="6" * 64,
        environment_lock_sha256="7" * 64,
        seed=_SEED,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )


def _worker(
    rank: int,
    init_file: str,
    stage_path: str,
    checkpoint_path: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        stage, model, optimizer, trainer, plan = _build_stack(stage_path, rank)
        first = trainer.train_microbatch(_batch(0))
        assert first.optimizer_stepped and first.optimizer_step == 1
        assert first.grad_norm is not None and first.grad_norm > 0
        checkpoint_snapshot = _snapshot(model)

        identity = _identity(stage, trainer)
        manifest = save_scale_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            plan=plan,
            identity=identity,
            metadata={
                "authority": "LOCAL_FREE_FSDP2_DCP_INTEGRATION_NOT_GPU_OR_PROMOTION",
                "stage": "S1",
                "parameter_count": stage.model.parameter_count(),
            },
            rank_state={
                "micro_step": trainer.micro_step,
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
            },
        )
        assert manifest["identity_sha256"] == identity.sha256

        second_control = trainer.train_microbatch(_batch(1))
        assert second_control.optimizer_stepped and second_control.optimizer_step == 2
        control_final = _snapshot(model)
        control_loss = second_control.loss

        del trainer, optimizer, model
        gc.collect()
        dist.barrier()

        stage2, model2, optimizer2, trainer2, plan2 = _build_stack(stage_path, rank)
        assert stage2.model.identity_sha256() == stage.model.identity_sha256()

        def restore_rank_state(value) -> None:
            trainer2.micro_step = int(value["micro_step"])
            trainer2.optimizer_step = int(value["optimizer_step"])
            trainer2.tokens_seen = int(value["tokens_seen"])

        loaded = load_scale_checkpoint(
            checkpoint_path,
            model=model2,
            optimizer=optimizer2,
            target_plan=plan2,
            mode=ResumeMode.EXACT_TOPOLOGY,
            expected_identity_sha256=identity.sha256,
            restore_rank_state=restore_rank_state,
        )
        assert loaded.exact_topology is True
        assert loaded.exact_trajectory_claim_allowed is True
        assert loaded.identity_sha256 == identity.sha256
        assert trainer2.optimizer_step == 1
        _assert_snapshot_equal(_snapshot(model2), checkpoint_snapshot)

        second_resumed = trainer2.train_microbatch(_batch(1))
        assert second_resumed.optimizer_stepped and second_resumed.optimizer_step == 2
        assert second_resumed.loss == pytest.approx(control_loss, rel=0.0, abs=0.0)
        _assert_snapshot_equal(_snapshot(model2), control_final)

        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "parameter_count": stage.model.parameter_count(),
                    "model_spec_sha256": stage.model.identity_sha256(),
                    "checkpoint_sha256": loaded.aggregate_checkpoint_sha256,
                    "restored_optimizer_step": 1,
                    "final_optimizer_step": second_resumed.optimizer_step,
                    "exact_checkpoint_shard_match": True,
                    "exact_final_shard_match": True,
                    "control_loss": control_loss,
                    "resumed_loss": second_resumed.loss,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_integration(tmp_path: Path) -> list[dict]:
    descriptor, init_file = tempfile.mkstemp(prefix="scale-fsdp2-dcp-")
    os.close(descriptor)
    os.unlink(init_file)
    checkpoint = tmp_path / "fsdp2-dcp-checkpoint"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    stage_path = str((Path(__file__).parents[1] / "configs/stages/s1_100k.json").resolve())
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, init_file, stage_path, str(checkpoint), str(result_dir)),
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(120)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert [process.exitcode for process in processes] == [0, 0]
    if os.path.exists(init_file):
        os.unlink(init_file)
    manifest = verify_scale_checkpoint(checkpoint)
    assert manifest["metadata"]["parameter_count"] == 107_856
    return [
        json.loads((result_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(_WORLD_SIZE)
    ]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch distributed Gloo unavailable",
)
def test_canonical_fsdp2_dcp_exact_resume_and_continue(tmp_path: Path) -> None:
    rows = _run_integration(tmp_path)
    assert [row["rank"] for row in rows] == [0, 1]
    assert {row["parameter_count"] for row in rows} == {107_856}
    assert len({row["model_spec_sha256"] for row in rows}) == 1
    assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert all(row["exact_checkpoint_shard_match"] for row in rows)
    assert all(row["exact_final_shard_match"] for row in rows)
    assert all(row["final_optimizer_step"] == 2 for row in rows)
