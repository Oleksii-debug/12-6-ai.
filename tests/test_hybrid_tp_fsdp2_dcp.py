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
from twelve_six.distributed.hybrid_tp_fsdp2 import (
    HybridTPFSDP2Trainer,
    apply_hybrid_tp_fsdp2,
)
from twelve_six.distributed.rank_layout import RankLayout
from twelve_six.distributed.tensor_parallel import TensorParallelPlan
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import build_optimizer

_WORLD_SIZE = 4
_DP = 2
_TP = 2
_SEED = 424242
_SEQUENCE_LENGTH = 8


def _planning_spec_1b() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32768,
        max_seq_len=4096,
        d_model=2048,
        n_layers=18,
        n_heads=32,
        n_kv_heads=8,
        head_dim=64,
        d_ff=6720,
        rope_rotary_dim=64,
    )


def test_1b_candidate_maps_to_dp2_tp8_without_changing_model_identity() -> None:
    spec = _planning_spec_1b()
    assert spec.parameter_count() == 999_106_560
    assert spec.identity_sha256() == (
        "cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261"
    )
    plan = ParallelPlan(
        data_parallel=2,
        tensor_parallel=8,
        shard_model_state_across_data_parallel=True,
    )
    plan.validate()
    tp = TensorParallelPlan.from_model_spec(spec, plan.tensor_parallel)
    assert plan.world_size == 16
    assert plan.model_state_shard_factor == 16
    assert tp.model_spec_sha256 == spec.identity_sha256()
    assert tp.rank_geometry(7).local_query_heads == 4
    assert tp.rank_geometry(7).local_kv_heads == 1
    assert tp.rank_geometry(7).local_d_ff == 840


def _batch(dp_rank: int, index: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(9100 + 100 * index + dp_rank)
    return {
        "input_ids": torch.randint(
            0,
            512,
            (1, _SEQUENCE_LENGTH),
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


def _build_stack(stage_path: str):
    torch.manual_seed(_SEED)
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init)
    plan = ParallelPlan(
        data_parallel=_DP,
        tensor_parallel=_TP,
        shard_model_state_across_data_parallel=True,
    )
    model, binding = apply_hybrid_tp_fsdp2(model, plan, device_type="cpu")
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
    trainer = HybridTPFSDP2Trainer(
        model,
        config,
        device="cpu",
        optimizer=optimizer,
        data_parallel_group=binding.data_parallel_group,
        data_parallel_degree=_DP,
    )
    assert stage.model.parameter_count() == 107_856
    return stage, model, optimizer, trainer, plan, binding


def _identity(stage, trainer: HybridTPFSDP2Trainer) -> ScaleCheckpointIdentity:
    source_sha = os.environ.get("GITHUB_SHA", "a" * 40)
    if len(source_sha) not in {40, 64}:
        source_sha = "a" * 40
    return ScaleCheckpointIdentity(
        git_sha=source_sha.lower(),
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256=stage.init.identity_sha256(),
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
        layout = RankLayout(
            ParallelPlan(
                data_parallel=_DP,
                tensor_parallel=_TP,
                shard_model_state_across_data_parallel=True,
            )
        )
        coordinate = layout.coordinate(rank)
        stage, model, optimizer, trainer, plan, binding = _build_stack(stage_path)
        assert binding.world_size == _WORLD_SIZE
        assert binding.parallel_plan.data_parallel == _DP
        assert binding.parallel_plan.tensor_parallel == _TP
        assert binding.tensor_parallel_plan.tp_degree == _TP
        assert binding.tensor_parallel_plan.model_spec_sha256 == stage.model.identity_sha256()

        first = trainer.train_microbatch(_batch(coordinate.dp, 0))
        assert first.optimizer_stepped and first.optimizer_step == 1
        assert first.grad_norm is not None and first.grad_norm > 0
        expected_dp_tokens = _DP * (_SEQUENCE_LENGTH - 1)
        assert trainer.last_data_parallel_global_tokens == expected_dp_tokens
        assert expected_dp_tokens < _WORLD_SIZE * (_SEQUENCE_LENGTH - 1)
        checkpoint_snapshot = _snapshot(model)

        identity = _identity(stage, trainer)
        manifest = save_scale_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            plan=plan,
            identity=identity,
            metadata={
                "authority": "LOCAL_FREE_DP2_TP2_FSDP2_DCP_NOT_GPU_OR_PROMOTION",
                "stage": "S1_MECHANICS_ANALOGUE",
                "parameter_count": stage.model.parameter_count(),
                "tensor_parallel_plan_sha256": binding.tensor_parallel_plan.identity_sha256,
                "tensor_parallel_layout_sha256": (
                    binding.tensor_parallel_plan.checkpoint_layout_sha256
                ),
                "data_parallel_degree": _DP,
                "tensor_parallel_degree": _TP,
            },
            rank_state={
                "dp_rank": coordinate.dp,
                "tp_rank": coordinate.tp,
                "micro_step": trainer.micro_step,
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
            },
        )
        assert manifest["identity_sha256"] == identity.sha256

        second_control = trainer.train_microbatch(_batch(coordinate.dp, 1))
        assert second_control.optimizer_stepped and second_control.optimizer_step == 2
        control_final = _snapshot(model)
        control_loss = second_control.loss

        del trainer, optimizer, model, binding
        gc.collect()
        dist.barrier()

        stage2, model2, optimizer2, trainer2, plan2, binding2 = _build_stack(stage_path)
        assert stage2.model.identity_sha256() == stage.model.identity_sha256()

        def restore_rank_state(value) -> None:
            assert int(value["dp_rank"]) == coordinate.dp
            assert int(value["tp_rank"]) == coordinate.tp
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
        assert trainer2.optimizer_step == 1
        _assert_snapshot_equal(_snapshot(model2), checkpoint_snapshot)

        second_resumed = trainer2.train_microbatch(_batch(coordinate.dp, 1))
        assert second_resumed.optimizer_stepped and second_resumed.optimizer_step == 2
        assert second_resumed.loss == pytest.approx(control_loss, rel=0.0, abs=0.0)
        _assert_snapshot_equal(_snapshot(model2), control_final)

        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "dp_rank": coordinate.dp,
                    "tp_rank": coordinate.tp,
                    "model_spec_sha256": stage.model.identity_sha256(),
                    "tp_plan_sha256": binding2.tensor_parallel_plan.identity_sha256,
                    "checkpoint_sha256": loaded.aggregate_checkpoint_sha256,
                    "dp_global_tokens": trainer2.last_data_parallel_global_tokens,
                    "final_optimizer_step": second_resumed.optimizer_step,
                    "exact_checkpoint_local_shard_match": True,
                    "exact_final_local_shard_match": True,
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
    descriptor, init_file = tempfile.mkstemp(prefix="scale-tp-fsdp2-dcp-")
    os.close(descriptor)
    os.unlink(init_file)
    checkpoint = tmp_path / "tp-fsdp2-dcp-checkpoint"
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
        process.join(180)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    if os.path.exists(init_file):
        os.unlink(init_file)
    manifest = verify_scale_checkpoint(checkpoint)
    assert manifest["save_topology"]["world_size"] == _WORLD_SIZE
    assert manifest["save_topology"]["parallel_plan"]["data_parallel"] == _DP
    assert manifest["save_topology"]["parallel_plan"]["tensor_parallel"] == _TP
    return [
        json.loads((result_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(_WORLD_SIZE)
    ]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch distributed Gloo unavailable",
)
def test_real_dp2_tp2_fsdp2_dcp_exact_resume_and_continue(tmp_path: Path) -> None:
    rows = _run_integration(tmp_path)
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert {(row["dp_rank"], row["tp_rank"]) for row in rows} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert len({row["model_spec_sha256"] for row in rows}) == 1
    assert len({row["tp_plan_sha256"] for row in rows}) == 1
    assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert {row["dp_global_tokens"] for row in rows} == {_DP * (_SEQUENCE_LENGTH - 1)}
    assert all(row["final_optimizer_step"] == 2 for row in rows)
    assert all(row["exact_checkpoint_local_shard_match"] for row in rows)
    assert all(row["exact_final_local_shard_match"] for row in rows)
