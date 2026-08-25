from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from twelve_six.distributed.async_dcp_checkpoint import begin_async_scale_checkpoint
from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    COMMITTED,
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
_SEED = 14542
_SEQUENCE_LENGTH = 12


def _build_stack(stage_path: str):
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
    return stage, model, optimizer, trainer, plan


def _batch(stage, index: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(145_000 + index)
    return {
        "input_ids": torch.randint(
            0,
            stage.model.vocab_size,
            (1, _SEQUENCE_LENGTH),
            generator=generator,
            dtype=torch.long,
        )
    }


def _hash_tree(value: Any, digest: Any) -> None:
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        rows = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        for key, item in rows:
            digest.update(type(key).__name__.encode())
            digest.update(repr(key).encode())
            _hash_tree(item, digest)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _hash_tree(item, digest)
        return
    digest.update(type(value).__name__.encode())
    digest.update(repr(value).encode())


def _state_digest(model: Any, optimizer: Any) -> str:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optim_state = get_state_dict(model, optimizer)
    digest = hashlib.sha256()
    _hash_tree({"model": model_state, "optimizer": optim_state}, digest)
    return digest.hexdigest()


def _trainer_state(trainer: FSDP2Trainer) -> dict[str, int]:
    return {
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }


def _identity(stage, trainer: FSDP2Trainer) -> ScaleCheckpointIdentity:
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


def _destroy_stack(*objects: Any) -> None:
    del objects
    gc.collect()


def _load_boundary(stage_path: str, checkpoint: str, identity_sha: str):
    stage, model, optimizer, trainer, plan = _build_stack(stage_path)

    def restore_rank_state(value: Mapping[str, Any]) -> None:
        trainer.micro_step = int(value["micro_step"])
        trainer.optimizer_step = int(value["optimizer_step"])
        trainer.tokens_seen = int(value["tokens_seen"])

    loaded = load_scale_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        target_plan=plan,
        mode=ResumeMode.EXACT_TOPOLOGY,
        expected_identity_sha256=identity_sha,
        restore_rank_state=restore_rank_state,
    )
    assert loaded.exact_trajectory_claim_allowed is True
    return stage, model, optimizer, trainer, plan


def _worker(
    rank: int,
    init_file: str,
    stage_path: str,
    root: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    root_path = Path(root)
    sync_path = root_path / "sync"
    async_path = root_path / "async"
    failed_path = root_path / "failed-async"
    try:
        stage, model, optimizer, trainer, plan = _build_stack(stage_path)
        assert stage.model.parameter_count() == 107_856
        first = trainer.train_microbatch(_batch(stage, 0))
        assert first.optimizer_stepped and first.optimizer_step == 1
        boundary_digest = _state_digest(model, optimizer)
        boundary_trainer = _trainer_state(trainer)
        identity = _identity(stage, trainer)
        save_scale_checkpoint(
            sync_path,
            model=model,
            optimizer=optimizer,
            plan=plan,
            identity=identity,
            metadata={"worker": "CHECKPOINT-145", "mode": "sync"},
            rank_state=boundary_trainer,
        )
        second = trainer.train_microbatch(_batch(stage, 1))
        sync_final_digest = _state_digest(model, optimizer)
        sync_final_loss = second.loss
        _destroy_stack(trainer, optimizer, model)
        dist.barrier()

        stage_a, model_a, optimizer_a, trainer_a, plan_a = _build_stack(stage_path)
        first_a = trainer_a.train_microbatch(_batch(stage_a, 0))
        assert first_a.optimizer_stepped and first_a.optimizer_step == 1
        assert _state_digest(model_a, optimizer_a) == boundary_digest
        assert _trainer_state(trainer_a) == boundary_trainer
        async_identity = _identity(stage_a, trainer_a)
        assert async_identity.sha256 == identity.sha256

        pending = begin_async_scale_checkpoint(
            async_path,
            model=model_a,
            optimizer=optimizer_a,
            plan=plan_a,
            identity=async_identity,
            metadata={"worker": "CHECKPOINT-145", "mode": "async"},
            rank_state=_trainer_state(trainer_a),
        )
        assert pending.requires_wait_before_exit is True
        assert not async_path.exists()
        assert not (pending.staging / COMMITTED).exists()

        # Standard async_save must have staged a training-safe copy before return.
        second_a = trainer_a.train_microbatch(_batch(stage_a, 1))
        async_live_final_digest = _state_digest(model_a, optimizer_a)
        assert second_a.loss == pytest.approx(sync_final_loss, rel=0.0, abs=0.0)
        assert async_live_final_digest == sync_final_digest

        async_manifest = pending.close()
        assert pending.requires_wait_before_exit is False
        assert pending.published is True
        assert async_path.is_dir()
        assert (async_path / COMMITTED).is_file()
        assert async_manifest["identity_sha256"] == identity.sha256
        _destroy_stack(trainer_a, optimizer_a, model_a)
        dist.barrier()

        # Fresh objects from the synchronous control checkpoint.
        stage_s, model_s, optimizer_s, trainer_s, _ = _load_boundary(
            stage_path, str(sync_path), identity.sha256
        )
        assert _trainer_state(trainer_s) == boundary_trainer
        assert _state_digest(model_s, optimizer_s) == boundary_digest
        resumed_s = trainer_s.train_microbatch(_batch(stage_s, 1))
        resumed_sync_digest = _state_digest(model_s, optimizer_s)
        assert resumed_s.loss == pytest.approx(sync_final_loss, rel=0.0, abs=0.0)
        assert resumed_sync_digest == sync_final_digest
        _destroy_stack(trainer_s, optimizer_s, model_s)
        dist.barrier()

        # Fresh objects from the async checkpoint must be byte-exact at the logical state level.
        stage_b, model_b, optimizer_b, trainer_b, plan_b = _load_boundary(
            stage_path, str(async_path), identity.sha256
        )
        assert _trainer_state(trainer_b) == boundary_trainer
        assert _state_digest(model_b, optimizer_b) == boundary_digest
        resumed_a = trainer_b.train_microbatch(_batch(stage_b, 1))
        resumed_async_digest = _state_digest(model_b, optimizer_b)
        assert resumed_a.loss == pytest.approx(sync_final_loss, rel=0.0, abs=0.0)
        assert resumed_async_digest == resumed_sync_digest == sync_final_digest

        # Inject an async I/O failure before publication. The hidden generation may remain,
        # but it must never become a visible committed checkpoint and prior checkpoints survive.
        real_async_save = dcp.async_save

        class _InjectedFailure:
            def result(self) -> Any:
                raise OSError("CHECKPOINT-145 injected async storage failure")

        def injected_async_save(*args: Any, **kwargs: Any) -> _InjectedFailure:
            del args, kwargs
            return _InjectedFailure()

        dcp.async_save = injected_async_save
        try:
            failed_identity = ScaleCheckpointIdentity(
                **{
                    **identity.__dict__,
                    "step": trainer_b.optimizer_step,
                    "tokens_seen": trainer_b.tokens_seen,
                }
            )
            failed = begin_async_scale_checkpoint(
                failed_path,
                model=model_b,
                optimizer=optimizer_b,
                plan=plan_b,
                identity=failed_identity,
                metadata={"worker": "CHECKPOINT-145", "mode": "injected-failure"},
                rank_state=_trainer_state(trainer_b),
            )
            assert not failed_path.exists()
            assert not (failed.staging / COMMITTED).exists()
            try:
                failed.wait()
            except RuntimeError as exc:
                assert "failed before publication" in str(exc)
                assert "injected async storage failure" in str(exc)
            else:
                raise AssertionError("injected async failure unexpectedly published")
            assert not failed_path.exists()
            assert not (failed.staging / COMMITTED).exists()
        finally:
            dcp.async_save = real_async_save

        dist.barrier()
        if rank == 0:
            verify_scale_checkpoint(sync_path)
            verify_scale_checkpoint(async_path)
        dist.barrier()

        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "boundary_state_digest": boundary_digest,
                    "sync_final_state_digest": sync_final_digest,
                    "async_final_state_digest": resumed_async_digest,
                    "trainer_state_exact": True,
                    "async_training_overlap_state_exact": True,
                    "safe_exit_wait_required_and_drained": True,
                    "incomplete_async_not_advertised": True,
                    "previous_checkpoint_recoverable": True,
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


def _run(tmp_path: Path) -> list[dict[str, Any]]:
    descriptor, init_file = tempfile.mkstemp(prefix="checkpoint145-")
    os.close(descriptor)
    os.unlink(init_file)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    root = tmp_path / "checkpoints"
    root.mkdir()
    stage_path = str((Path(__file__).parents[1] / "configs/stages/s1_100k.json").resolve())
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, init_file, stage_path, str(root), str(result_dir)),
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
    assert [process.exitcode for process in processes] == [0, 0]
    if os.path.exists(init_file):
        os.unlink(init_file)
    return [
        json.loads((result_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(_WORLD_SIZE)
    ]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch distributed Gloo unavailable",
)
def test_async_dcp_preserves_d18_truth_and_exact_continuation(tmp_path: Path) -> None:
    rows = _run(tmp_path)
    assert [row["rank"] for row in rows] == [0, 1]
    assert all(row["trainer_state_exact"] for row in rows)
    assert all(row["async_training_overlap_state_exact"] for row in rows)
    assert all(row["safe_exit_wait_required_and_drained"] for row in rows)
    assert all(row["incomplete_async_not_advertised"] for row in rows)
    assert all(row["previous_checkpoint_recoverable"] for row in rows)
    assert all(
        row["sync_final_state_digest"] == row["async_final_state_digest"] for row in rows
    )
