#!/usr/bin/env python3
"""Measure synchronous D18 save against maintained DCP async_save on FSDP2.

Run under ``torchrun --standalone --nproc-per-node=2``. The tool writes metrics only;
checkpoint payloads stay local to the runner and are not uploaded as artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from twelve_six.distributed.async_dcp_checkpoint import begin_async_scale_checkpoint
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

_SEED = 145_145
_SEQUENCE_LENGTH = 12


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class _PeakRssSampler:
    def __init__(self, baseline: int, interval: float = 0.002) -> None:
        self.baseline = baseline
        self.interval = interval
        self.peak = baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_bytes())
            self._stop.wait(self.interval)
        self.peak = max(self.peak, _rss_bytes())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return max(0, self.peak - self.baseline)


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


def _build_stack(stage_path: str, world_size: int):
    torch.manual_seed(_SEED)
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init)
    plan = ParallelPlan(
        data_parallel=world_size,
        shard_model_state_across_data_parallel=True,
    )
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=world_size)
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
    return stage, model, optimizer, trainer, plan, config


def _batch(stage, rank: int) -> dict[str, torch.Tensor]:
    # Identical batch on both ranks is intentional: the benchmark isolates checkpoint cost,
    # not data-parallel statistical behavior.
    del rank
    generator = torch.Generator(device="cpu").manual_seed(1_451_450)
    return {
        "input_ids": torch.randint(
            0,
            stage.model.vocab_size,
            (1, _SEQUENCE_LENGTH),
            generator=generator,
            dtype=torch.long,
        )
    }


def _trainer_state(trainer: FSDP2Trainer) -> dict[str, int]:
    return {
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }


def _identity(stage, trainer: FSDP2Trainer, config: TrainerConfig) -> ScaleCheckpointIdentity:
    source_sha = os.environ.get("GITHUB_SHA", "a" * 40).lower()
    if len(source_sha) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in source_sha):
        source_sha = "a" * 40
    return ScaleCheckpointIdentity(
        git_sha=source_sha,
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256=stage.init.identity_sha256(),
        tokenizer_config_sha256=_label_sha("checkpoint145-byte-tokenizer-config"),
        tokenizer_vocab_sha256=_label_sha("checkpoint145-byte-tokenizer-vocab"),
        data_manifest_sha256=_label_sha("checkpoint145-synthetic-benchmark-batch"),
        packing_sha256=_label_sha("checkpoint145-sequence-length-12"),
        training_config_sha256=_label_sha(json.dumps(asdict(config), sort_keys=True)),
        environment_lock_sha256=_label_sha("requirements/locks/linux-x86_64/runtime.lock.txt"),
        seed=_SEED,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _critical_max(rows: list[dict[str, Any]], key: str) -> float | int:
    return max(row[key] for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sync", "async"), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("CHECKPOINT-145 benchmark requires exactly two ranks")

    work_dir = Path(args.work_dir).resolve()
    if rank == 0:
        work_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    checkpoint = work_dir / f"checkpoint-{args.mode}"

    try:
        stage, model, optimizer, trainer, plan, config = _build_stack(args.stage, world_size)
        train_result = trainer.train_microbatch(_batch(stage, rank))
        if not train_result.optimizer_stepped or train_result.optimizer_step != 1:
            raise RuntimeError("benchmark failed to materialize one real optimizer step")
        boundary_digest = _state_digest(model, optimizer)
        trainer_boundary = _trainer_state(trainer)
        identity = _identity(stage, trainer, config)
        gc.collect()
        baseline_rss = _rss_bytes()
        sampler = _PeakRssSampler(baseline_rss)
        sampler.start()
        started = time.perf_counter()

        if args.mode == "sync":
            save_scale_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                plan=plan,
                identity=identity,
                metadata={
                    "worker": "CHECKPOINT-145",
                    "mode": "sync",
                    "authority": "LOCAL_FREE_CPU_GLOO_NOT_GPU_OR_PROMOTION",
                    "parameter_count": stage.model.parameter_count(),
                },
                rank_state=trainer_boundary,
            )
            returned = time.perf_counter()
            completed = returned
        else:
            pending = begin_async_scale_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                plan=plan,
                identity=identity,
                metadata={
                    "worker": "CHECKPOINT-145",
                    "mode": "async",
                    "authority": "LOCAL_FREE_CPU_GLOO_NOT_GPU_OR_PROMOTION",
                    "parameter_count": stage.model.parameter_count(),
                },
                rank_state=trainer_boundary,
            )
            returned = time.perf_counter()
            if checkpoint.exists():
                raise RuntimeError("async checkpoint became visible before explicit wait")
            if not pending.requires_wait_before_exit:
                raise RuntimeError("async handle did not require shutdown drain")
            pending.close()
            completed = time.perf_counter()
            if pending.requires_wait_before_exit:
                raise RuntimeError("async checkpoint still requires wait after close")

        peak_extra_rss = sampler.stop()
        local_save = {
            "rank": rank,
            "foreground_stall_seconds": returned - started,
            "total_checkpoint_completion_seconds": completed - started,
            "background_tail_seconds": completed - returned,
            "baseline_rss_bytes": baseline_rss,
            "peak_extra_rss_bytes": peak_extra_rss,
            "boundary_state_digest": boundary_digest,
            "trainer_boundary": trainer_boundary,
        }
        save_rows: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(save_rows, local_save)
        save_rows_typed = [row for row in save_rows if row is not None]

        dist.barrier()
        verify_seconds: float | None = None
        storage_bytes: int | None = None
        if rank == 0:
            storage_bytes = _tree_bytes(checkpoint)
            verify_started = time.perf_counter()
            manifest = verify_scale_checkpoint(checkpoint)
            verify_seconds = time.perf_counter() - verify_started
            if manifest["identity_sha256"] != identity.sha256:
                raise RuntimeError("verified checkpoint identity differs from save identity")
        dist.barrier()

        # Release the live training objects before measuring fresh-object construction + load.
        del trainer
        del optimizer
        del model
        gc.collect()
        dist.barrier()

        fresh_started = time.perf_counter()
        stage2, model2, optimizer2, trainer2, plan2, _ = _build_stack(args.stage, world_size)

        def restore_rank_state(value: Mapping[str, Any]) -> None:
            trainer2.micro_step = int(value["micro_step"])
            trainer2.optimizer_step = int(value["optimizer_step"])
            trainer2.tokens_seen = int(value["tokens_seen"])

        load_started = time.perf_counter()
        loaded = load_scale_checkpoint(
            checkpoint,
            model=model2,
            optimizer=optimizer2,
            target_plan=plan2,
            mode=ResumeMode.EXACT_TOPOLOGY,
            expected_identity_sha256=identity.sha256,
            restore_rank_state=restore_rank_state,
        )
        load_finished = time.perf_counter()
        loaded_digest = _state_digest(model2, optimizer2)
        loaded_trainer = _trainer_state(trainer2)
        if loaded_digest != boundary_digest:
            raise RuntimeError("fresh-loaded model/optimizer state differs from checkpoint boundary")
        if loaded_trainer != trainer_boundary:
            raise RuntimeError("fresh-loaded Trainer state differs from checkpoint boundary")
        if not loaded.exact_trajectory_claim_allowed:
            raise RuntimeError("exact-topology load did not restore exact trajectory control state")
        if stage2.model.identity_sha256() != stage.model.identity_sha256():
            raise RuntimeError("fresh model identity changed")

        local_load = {
            "rank": rank,
            "fresh_object_load_seconds": load_finished - fresh_started,
            "load_only_seconds": load_finished - load_started,
            "loaded_state_digest": loaded_digest,
            "loaded_trainer": loaded_trainer,
            "exact_trajectory_claim_allowed": loaded.exact_trajectory_claim_allowed,
        }
        load_rows: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(load_rows, local_load)
        load_rows_typed = [row for row in load_rows if row is not None]

        if rank == 0:
            report = {
                "schema": "12-6.checkpoint145-async-dcp-benchmark.v1",
                "worker": "CHECKPOINT-145-ASYNC-DCP",
                "mode": args.mode,
                "authority": "LOCAL_FREE_CPU_GLOO_NOT_GPU_OR_PROMOTION",
                "git_sha": os.environ.get("GITHUB_SHA"),
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "torch_cuda_version": torch.version.cuda,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "stage": stage.stage,
                "stage_path": str(Path(args.stage)),
                "parameter_count": stage.model.parameter_count(),
                "model_spec_sha256": stage.model.identity_sha256(),
                "init_spec_sha256": stage.init.identity_sha256(),
                "optimizer_state_materialized_by_real_step": True,
                "sequence_length": _SEQUENCE_LENGTH,
                "foreground_training_stall_seconds": _critical_max(
                    save_rows_typed, "foreground_stall_seconds"
                ),
                "total_checkpoint_completion_seconds": _critical_max(
                    save_rows_typed, "total_checkpoint_completion_seconds"
                ),
                "background_tail_seconds": _critical_max(
                    save_rows_typed, "background_tail_seconds"
                ),
                "peak_extra_rss_bytes_max_rank": _critical_max(
                    save_rows_typed, "peak_extra_rss_bytes"
                ),
                "storage_bytes": storage_bytes,
                "verify_seconds_rank0": verify_seconds,
                "fresh_object_load_seconds_max_rank": _critical_max(
                    load_rows_typed, "fresh_object_load_seconds"
                ),
                "load_only_seconds_max_rank": _critical_max(load_rows_typed, "load_only_seconds"),
                "state_exact_after_fresh_load": all(
                    row["loaded_state_digest"] == save_rows_typed[index]["boundary_state_digest"]
                    for index, row in enumerate(load_rows_typed)
                ),
                "trainer_state_exact_after_fresh_load": all(
                    row["loaded_trainer"] == save_rows_typed[index]["trainer_boundary"]
                    for index, row in enumerate(load_rows_typed)
                ),
                "per_rank_save": save_rows_typed,
                "per_rank_load": load_rows_typed,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(report, sort_keys=True))
        dist.barrier()
    finally:
        if rank == 0 and checkpoint.exists():
            shutil.rmtree(checkpoint)
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
