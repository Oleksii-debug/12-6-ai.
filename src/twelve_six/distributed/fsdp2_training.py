"""Real PyTorch FSDP2 execution for the canonical 12-6 model stack.

This module is intentionally not imported from ``twelve_six.distributed`` so the
single-device S0 import path stays lazy and unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.runtime import build_torch_mesh_spec
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer, build_optimizer


@dataclass(frozen=True, slots=True)
class FSDP2RankExecution:
    rank: int
    world_size: int
    backend: str
    device_type: str
    model_spec_sha256: str
    parameter_count: int
    sample_indices: tuple[int, ...]
    local_tokens: int
    global_tokens: int
    local_loss: float
    reduced_loss: float
    grad_norm: float
    optimizer_step: int
    global_parameter_update_l1: float
    parameters_are_dtensor: bool
    error_recovery_exercised: bool
    process_group_destroyed: bool


@dataclass(frozen=True, slots=True)
class FSDP2ExecutionResult:
    world_size: int
    backend: str
    device_type: str
    model_spec_sha256: str
    parameter_count: int
    ranks_seen: tuple[int, ...]
    sampler_indices: tuple[tuple[int, ...], ...]
    global_tokens: int
    reduced_loss: float
    grad_norm_min: float
    grad_norm_max: float
    optimizer_steps: tuple[int, ...]
    global_parameter_update_l1: float
    parameters_are_dtensor: bool
    error_recovery_exercised: bool
    clean_shutdown: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FSDP2Trainer(Trainer):
    """D12 adapter that makes D02 gradient accounting safe for DTensor gradients."""

    def _normalize_gradients_and_norm(self, token_count: int):
        if token_count <= 0:
            raise RuntimeError("optimizer update requires at least one valid target token")
        import torch
        import torch.distributed as dist

        global_tokens = token_count
        dp_world_size = 1
        if dist.is_available() and dist.is_initialized():
            token_tensor = torch.tensor(token_count, dtype=torch.int64, device=self.device)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            global_tokens = int(token_tensor.item())
            dp_world_size = dist.get_world_size()
        if global_tokens <= 0:
            raise RuntimeError("distributed optimizer update requires positive global tokens")

        found = False
        gradient_scale = dp_world_size / global_tokens
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            parameter.grad.mul_(gradient_scale)

        if not found:
            return torch.zeros((), device=self.device)

        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=math.inf,
            error_if_nonfinite=True,
        )
        full_tensor = getattr(norm, "full_tensor", None)
        if callable(full_tensor):
            norm = full_tensor()
        if not isinstance(norm, torch.Tensor):
            norm = torch.as_tensor(norm, device=self.device)
        return norm.to(device=self.device)

    def train_microbatch(self, batch):
        try:
            return super().train_microbatch(batch)
        except Exception:
            reset = getattr(self.model, "reset_iter_state", None)
            if callable(reset):
                reset()
            raise


def apply_fsdp2(
    model: TwelveSixDecoder,
    mesh: Any,
    *,
    reshard_after_forward: bool = True,
) -> TwelveSixDecoder:
    """Apply FSDP2 bottom-up, grouping shared embedding/head weights explicitly."""

    from torch.distributed.fsdp import FSDPModule, fully_shard

    kwargs = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
    }
    if model.spec.tie_word_embeddings:
        if model.token_embedding.weight is not model.lm_head.weight:
            raise RuntimeError("ModelSpec requires tied embeddings but modules do not share weight")
        fully_shard([model.token_embedding, model.lm_head], **kwargs)
    for block in model.blocks:
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)
    if not isinstance(model, FSDPModule):
        raise TypeError("fully_shard did not convert the root model to FSDPModule")
    return model


def _local_parameter_snapshot(model: TwelveSixDecoder) -> tuple[Any, ...]:
    snapshots = []
    for parameter in model.parameters():
        local = getattr(parameter, "to_local", None)
        tensor = local() if callable(local) else parameter
        snapshots.append(tensor.detach().clone())
    return tuple(snapshots)


def _local_parameter_update_l1(
    model: TwelveSixDecoder,
    before: tuple[Any, ...],
) -> float:
    import torch

    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    parameters = tuple(model.parameters())
    if len(parameters) != len(before):
        raise RuntimeError("parameter cardinality changed across optimizer step")
    for parameter, old in zip(parameters, before, strict=True):
        local = getattr(parameter, "to_local", None)
        current = local() if callable(local) else parameter
        total += (current.detach().to(torch.float64) - old.to(torch.float64)).abs().sum()
    return float(total.item())


def _exercise_fsdp_error_recovery(model: TwelveSixDecoder, device: Any) -> bool:
    import torch

    reset = getattr(model, "reset_iter_state", None)
    if not callable(reset):
        raise TypeError("current FSDPModule.reset_iter_state API is unavailable")
    invalid = torch.zeros(
        (1, model.spec.max_seq_len + 1),
        dtype=torch.long,
        device=device,
    )
    try:
        model(invalid)
    except ValueError as exc:
        if "exceeds max_seq_len" not in str(exc):
            raise
        reset()
    else:
        raise RuntimeError("expected over-context forward to fail")

    valid = torch.zeros((1, 2), dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(valid)
    if output.logits.shape != (1, 2, model.spec.vocab_size):
        raise RuntimeError("FSDP2 model did not recover after reset_iter_state")
    return True


def run_initialized_fsdp2_rank(
    *,
    stage_config_path: str | Path,
    backend: str,
    device_type: str,
    local_rank: int,
    samples_per_rank: int = 2,
    sequence_length: int = 16,
    seed: int = 1337,
    exercise_error_recovery: bool = True,
) -> FSDP2RankExecution:
    """Run one real FSDP2 optimizer step in an already initialized process group."""

    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    if not dist.is_initialized():
        raise RuntimeError("process group must be initialized before FSDP2 execution")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if samples_per_rank < 1:
        raise ValueError("samples_per_rank must be >= 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device_type must be 'cpu' or 'cuda'")

    stage = load_stage_config(stage_config_path)
    spec = stage.model
    if sequence_length > spec.max_seq_len:
        raise ValueError("sequence_length exceeds ModelSpec.max_seq_len")

    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TwelveSixDecoder(spec, stage.init)

    plan = ParallelPlan(
        data_parallel=world_size,
        shard_model_state_across_data_parallel=True,
    )
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=world_size)
    full_mesh = mesh_spec.create_device_mesh(device_type)
    fsdp_kwargs = mesh_spec.fsdp2_kwargs(
        full_mesh,
        reshard_after_forward=device_type == "cuda",
    )
    model = apply_fsdp2(model, **fsdp_kwargs)

    parameters_are_dtensor = all(isinstance(parameter, DTensor) for parameter in model.parameters())
    if not parameters_are_dtensor:
        raise RuntimeError("fully_shard did not expose DTensor parameters outside computation")

    error_recovery_exercised = False
    if exercise_error_recovery:
        error_recovery_exercised = _exercise_fsdp_error_recovery(model, device)

    config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=1,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    optimizer = build_optimizer(model, config)
    trainer = FSDP2Trainer(
        model,
        config,
        device=device,
        optimizer=optimizer,
    )

    sample_count = world_size * samples_per_rank
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    input_ids = torch.randint(
        0,
        spec.vocab_size,
        (sample_count, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    dataset = TensorDataset(input_ids)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    sample_indices = tuple(int(index) for index in sampler)
    loader = DataLoader(
        dataset,
        batch_size=samples_per_rank,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    batch = next(iter(loader))[0]

    before = _local_parameter_snapshot(model)
    metrics = trainer.train_microbatch({"input_ids": batch})
    if not metrics.optimizer_stepped or metrics.optimizer_step != 1:
        raise RuntimeError("FSDP2 Trainer did not commit exactly one optimizer step")
    if metrics.grad_norm is None or not math.isfinite(metrics.grad_norm) or metrics.grad_norm <= 0:
        raise RuntimeError("FSDP2 backward did not produce a finite non-zero gradient norm")

    local_update = _local_parameter_update_l1(model, before)
    update_tensor = torch.tensor(local_update, dtype=torch.float64, device=device)
    dist.all_reduce(update_tensor, op=dist.ReduceOp.SUM)
    global_parameter_update_l1 = float(update_tensor.item())
    if not math.isfinite(global_parameter_update_l1) or global_parameter_update_l1 <= 0:
        raise RuntimeError("FSDP2 optimizer step did not change sharded parameters")

    loss_tensor = torch.tensor(metrics.loss, dtype=torch.float64, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    reduced_loss = float((loss_tensor / world_size).item())

    token_tensor = torch.tensor(metrics.tokens, dtype=torch.int64, device=device)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    global_tokens = int(token_tensor.item())

    return FSDP2RankExecution(
        rank=rank,
        world_size=world_size,
        backend=backend,
        device_type=device_type,
        model_spec_sha256=spec.identity_sha256(),
        parameter_count=spec.parameter_count(),
        sample_indices=sample_indices,
        local_tokens=metrics.tokens,
        global_tokens=global_tokens,
        local_loss=metrics.loss,
        reduced_loss=reduced_loss,
        grad_norm=metrics.grad_norm,
        optimizer_step=metrics.optimizer_step,
        global_parameter_update_l1=global_parameter_update_l1,
        parameters_are_dtensor=parameters_are_dtensor,
        error_recovery_exercised=error_recovery_exercised,
        process_group_destroyed=False,
    )


def _local_cpu_worker(
    rank: int,
    world_size: int,
    init_file: str,
    stage_config_path: str,
    samples_per_rank: int,
    sequence_length: int,
    seed: int,
    exercise_error_recovery: bool,
    inject_failure_rank: int | None,
    output: Any,
) -> None:
    import torch.distributed as dist

    record: FSDP2RankExecution | None = None
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        record = run_initialized_fsdp2_rank(
            stage_config_path=stage_config_path,
            backend="gloo",
            device_type="cpu",
            local_rank=rank,
            samples_per_rank=samples_per_rank,
            sequence_length=sequence_length,
            seed=seed,
            exercise_error_recovery=exercise_error_recovery,
        )
        dist.barrier()
        if inject_failure_rank == rank:
            raise RuntimeError(f"injected post-step failure on rank {rank}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if record is not None:
        output.put(
            FSDP2RankExecution(
                **{
                    **asdict(record),
                    "process_group_destroyed": True,
                }
            )
        )


def run_local_cpu_fsdp2(
    stage_config_path: str | Path,
    *,
    world_size: int = 2,
    samples_per_rank: int = 2,
    sequence_length: int = 16,
    seed: int = 1337,
    timeout_seconds: float = 90.0,
    exercise_error_recovery: bool = True,
    inject_failure_rank: int | None = None,
) -> FSDP2ExecutionResult:
    """Spawn bounded LOCAL_FREE CPU/Gloo workers and execute the real 12-6 model."""

    if world_size < 2 or world_size > 4:
        raise ValueError("LOCAL_FREE FSDP2 CPU execution requires 2 <= world_size <= 4")
    if inject_failure_rank is not None and not 0 <= inject_failure_rank < world_size:
        raise ValueError("inject_failure_rank must name a local rank")

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch Gloo backend is unavailable")

    stage_path = str(Path(stage_config_path).resolve())
    context = mp.get_context("spawn")
    output = context.Queue()
    with tempfile.TemporaryDirectory(prefix="twelve-six-fsdp2-") as directory:
        init_file = str(Path(directory) / "store")
        processes = [
            context.Process(
                target=_local_cpu_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    stage_path,
                    samples_per_rank,
                    sequence_length,
                    seed,
                    exercise_error_recovery,
                    inject_failure_rank,
                    output,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout_seconds)
        stuck = [process for process in processes if process.is_alive()]
        if stuck:
            for process in stuck:
                process.terminate()
            for process in stuck:
                process.join(5)
            raise RuntimeError("LOCAL_FREE FSDP2 CPU execution timed out")
        failures = [
            (process.pid, process.exitcode)
            for process in processes
            if process.exitcode != 0
        ]
        if failures:
            raise RuntimeError(f"LOCAL_FREE FSDP2 child failures: {failures}")

        records: list[FSDP2RankExecution] = []
        for _ in range(world_size):
            try:
                records.append(output.get(timeout=10))
            except Empty as exc:
                raise RuntimeError("LOCAL_FREE FSDP2 execution lost a rank result") from exc

    records.sort(key=lambda item: item.rank)
    ranks_seen = tuple(record.rank for record in records)
    if ranks_seen != tuple(range(world_size)):
        raise RuntimeError("FSDP2 rank accounting is incomplete")
    identities = {record.model_spec_sha256 for record in records}
    counts = {record.parameter_count for record in records}
    reduced_losses = {record.reduced_loss for record in records}
    global_tokens = {record.global_tokens for record in records}
    global_updates = {record.global_parameter_update_l1 for record in records}
    if len(identities) != 1 or len(counts) != 1:
        raise RuntimeError("ModelSpec identity differs across FSDP2 ranks")
    if len(reduced_losses) != 1 or len(global_tokens) != 1 or len(global_updates) != 1:
        raise RuntimeError("reduced FSDP2 metrics differ across ranks")

    sampler_indices = tuple(record.sample_indices for record in records)
    flattened = tuple(index for indices in sampler_indices for index in indices)
    expected = tuple(range(world_size * samples_per_rank))
    if tuple(sorted(flattened)) != expected or len(set(flattened)) != len(flattened):
        raise RuntimeError(
            "DistributedSampler did not partition the synthetic dataset exactly once"
        )

    return FSDP2ExecutionResult(
        world_size=world_size,
        backend="gloo",
        device_type="cpu",
        model_spec_sha256=identities.pop(),
        parameter_count=counts.pop(),
        ranks_seen=ranks_seen,
        sampler_indices=sampler_indices,
        global_tokens=global_tokens.pop(),
        reduced_loss=reduced_losses.pop(),
        grad_norm_min=min(record.grad_norm for record in records),
        grad_norm_max=max(record.grad_norm for record in records),
        optimizer_steps=tuple(record.optimizer_step for record in records),
        global_parameter_update_l1=global_updates.pop(),
        parameters_are_dtensor=all(record.parameters_are_dtensor for record in records),
        error_recovery_exercised=all(record.error_recovery_exercised for record in records),
        clean_shutdown=all(record.process_group_destroyed for record in records),
    )


def run_torchrun_fsdp2(
    stage_config_path: str | Path,
    *,
    backend: str,
    device_type: str,
    samples_per_rank: int = 2,
    sequence_length: int = 16,
    seed: int = 1337,
    exercise_error_recovery: bool = True,
) -> FSDP2RankExecution:
    """Execute one rank under torchrun; intended GPU extension uses NCCL + CUDA."""

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA torchrun requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
    if dist.is_initialized():
        raise RuntimeError("torchrun entrypoint requires an uninitialized default process group")

    dist.init_process_group(backend=backend)
    try:
        record = run_initialized_fsdp2_rank(
            stage_config_path=stage_config_path,
            backend=backend,
            device_type=device_type,
            local_rank=local_rank,
            samples_per_rank=samples_per_rank,
            sequence_length=sequence_length,
            seed=seed,
            exercise_error_recovery=exercise_error_recovery,
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return FSDP2RankExecution(
        **{
            **asdict(record),
            "process_group_destroyed": True,
        }
    )


def build_execution_evidence(
    result: FSDP2ExecutionResult,
    *,
    source_sha: str,
) -> dict[str, Any]:
    """Bind LOCAL_FREE CPU/Gloo execution metrics to exact source and runtime identity."""

    import torch
    import torch.distributed as dist

    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be a full lowercase 40-character Git SHA")
    payload: dict[str, Any] = {
        "schema_version": "12-6.fsdp2-model-execution.v1",
        "authority": "LOCAL_FREE_CPU_GLOO_ENGINEERING_EVIDENCE_ONLY",
        "source_sha": source_sha,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "nccl_available": bool(dist.is_available() and dist.is_nccl_available()),
        },
        "execution": result.to_dict(),
        "claims": {
            "real_process_group": True,
            "real_device_mesh": True,
            "real_fully_shard": True,
            "real_forward_backward": True,
            "real_optimizer_update": True,
            "distributed_sampler": True,
            "loss_reduction": True,
            "gpu_nccl_executed": False,
            "multi_node_executed": False,
            "promotion_authority": False,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def write_execution_evidence(
    result: FSDP2ExecutionResult,
    *,
    source_sha: str,
    output_path: str | Path,
) -> dict[str, Any]:
    evidence = build_execution_evidence(result, source_sha=source_sha)
    Path(output_path).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence
