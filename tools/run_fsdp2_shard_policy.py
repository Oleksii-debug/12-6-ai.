#!/usr/bin/env python3
"""LOCAL_FREE SCALE-144 FSDP2 reshard-policy comparison.

Runs real 2-rank CPU/Gloo evidence only. The 10M case performs two optimizer
steps with a DCP save/destroy/reload/exact continuation between them. The ~100M
case is deliberately forward/materialization-only to bound memory and runtime on
free CPU hardware; it is not training-performance evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
)
from twelve_six.distributed.fsdp2_policy import (
    FSDP2ReshardPolicy,
    apply_fsdp2_reshard_policy,
    fsdp2_reshard_policy_spec,
)
from twelve_six.distributed.fsdp2_training import FSDP2Trainer
from twelve_six.distributed.runtime import build_torch_mesh_spec
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import build_optimizer

WORLD_SIZE = 2
SEED = 144_2026
TEN_M_SEQUENCE = 8
HUNDRED_M_SEQUENCE = 2
POLICIES = tuple(FSDP2ReshardPolicy)


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


class _PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.002) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = _current_rss_bytes()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())

    def __enter__(self) -> "_PeakRssSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=1.0)


def _local_tensor(tensor: Any) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    value = to_local() if callable(to_local) else tensor
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor-like state, got {type(value).__name__}")
    return value


def _tensor_bytes(tensor: Any) -> int:
    value = _local_tensor(tensor)
    return value.numel() * value.element_size()


def _parameter_bytes(model: TwelveSixDecoder) -> int:
    return sum(_tensor_bytes(parameter) for parameter in model.parameters())


def _gradient_bytes(model: TwelveSixDecoder) -> int:
    return sum(
        _tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    seen: set[int] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if isinstance(value, torch.Tensor):
            local = _local_tensor(value)
            key = id(local)
            if key not in seen:
                seen.add(key)
                total += local.numel() * local.element_size()
            return
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    for state in optimizer.state.values():
        visit(state)
    return total


def _snapshot_sha256(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        local = _local_tensor(parameter).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(str(tuple(local.shape)).encode("ascii"))
        digest.update(local.numpy().tobytes())
    return digest.hexdigest()


def _claim_module_bytes(modules: list[torch.nn.Module], claimed: set[int]) -> int:
    total = 0
    for module in modules:
        for parameter in module.parameters():
            key = id(parameter)
            if key in claimed:
                continue
            claimed.add(key)
            total += parameter.numel() * parameter.element_size()
    return total


def _group_bytes(model: TwelveSixDecoder) -> dict[str, int]:
    claimed: set[int] = set()
    result: dict[str, int] = {}
    if model.spec.tie_word_embeddings:
        result["tied_embedding_head"] = _claim_module_bytes(
            [model.token_embedding, model.lm_head], claimed
        )
    for index, block in enumerate(model.blocks):
        result[f"block_{index}"] = _claim_module_bytes([block], claimed)
    result["root"] = _claim_module_bytes([model], claimed)
    expected = model.spec.parameter_count() * next(model.parameters()).element_size()
    actual = sum(result.values())
    if actual != expected:
        raise RuntimeError(f"FSDP group byte accounting mismatch: {actual} != {expected}")
    return result


def _communication_proxy(
    group_bytes: dict[str, int], policy: FSDP2ReshardPolicy
) -> dict[str, int | str]:
    spec = fsdp2_reshard_policy_spec(policy)
    total = sum(group_bytes.values())
    backward = 0
    for name, value in group_bytes.items():
        should_reshard = (
            spec.root_reshard_after_forward
            if name == "root"
            else spec.non_root_reshard_after_forward
        )
        if should_reshard:
            backward += value
    reduce_scatter = total
    scheduled = total + backward + reduce_scatter
    per_rank_transfer_proxy = scheduled * (WORLD_SIZE - 1) // WORLD_SIZE
    return {
        "kind": "algorithm_independent_collective_tensor_payload_proxy_not_wire_bytes",
        "forward_all_gather_global_payload_bytes": total,
        "backward_all_gather_global_payload_bytes": backward,
        "reduce_scatter_global_payload_bytes": reduce_scatter,
        "scheduled_global_payload_bytes": scheduled,
        "per_rank_transfer_proxy_bytes": per_rank_transfer_proxy,
    }


def _identity(stage: Any, trainer: FSDP2Trainer, policy: FSDP2ReshardPolicy) -> ScaleCheckpointIdentity:
    source_sha = os.environ.get("GITHUB_SHA", "a" * 40).lower()
    if len(source_sha) not in {40, 64}:
        source_sha = "a" * 40
    policy_sha = hashlib.sha256(f"SCALE-144:{policy.value}".encode("utf-8")).hexdigest()
    return ScaleCheckpointIdentity(
        git_sha=source_sha,
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256=stage.init.identity_sha256(),
        tokenizer_config_sha256="2" * 64,
        tokenizer_vocab_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        packing_sha256="5" * 64,
        training_config_sha256=policy_sha,
        environment_lock_sha256="7" * 64,
        seed=SEED,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )


def _configure_worker_threads() -> None:
    cpu_count = os.cpu_count() or 2
    torch.set_num_threads(max(1, cpu_count // WORLD_SIZE))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_stack(stage_path: str, policy: FSDP2ReshardPolicy):
    torch.manual_seed(SEED)
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init)
    tied_before = (
        not stage.model.tie_word_embeddings
        or model.token_embedding.weight is model.lm_head.weight
    )
    groups = _group_bytes(model)
    plan = ParallelPlan(data_parallel=WORLD_SIZE, shard_model_state_across_data_parallel=True)
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=WORLD_SIZE)
    full_mesh = mesh_spec.create_device_mesh("cpu")
    model = apply_fsdp2_reshard_policy(
        model,
        mesh_spec.fsdp2_data_parallel_mesh(full_mesh),
        policy=policy,
    )
    tied_after = (
        not stage.model.tie_word_embeddings
        or model.token_embedding.weight is model.lm_head.weight
    )
    if not tied_before or not tied_after:
        raise RuntimeError("tied embedding/head alias was not preserved by FSDP2")
    return stage, model, plan, groups, tied_before, tied_after


def _batch(stage: Any, index: int, sequence_length: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1000 + index)
    return {
        "input_ids": torch.randint(
            0,
            stage.model.vocab_size,
            (1, sequence_length),
            generator=generator,
            dtype=torch.long,
        )
    }


def _ten_m_worker(
    rank: int,
    init_file: str,
    stage_path: str,
    policy_value: str,
    checkpoint_path: str,
    result_path: str,
) -> None:
    _configure_worker_threads()
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        policy = FSDP2ReshardPolicy(policy_value)
        stage, model, plan, groups, tied_before, tied_after = _build_stack(stage_path, policy)
        config = TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=2,
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=SEED,
            deterministic_algorithms=True,
        )
        optimizer = build_optimizer(model, config)
        trainer = FSDP2Trainer(model, config, device="cpu", optimizer=optimizer)
        captured: dict[str, int] = {}

        def capture_gradients(_optimizer, _args, _kwargs):
            captured["gradient_state_bytes"] = _gradient_bytes(model)

        handle = optimizer.register_step_pre_hook(capture_gradients)
        parameter_state_bytes = _parameter_bytes(model)
        rss_before = _current_rss_bytes()
        dist.barrier()
        with _PeakRssSampler() as sampler:
            started = time.perf_counter()
            first = trainer.train_microbatch(_batch(stage, 0, TEN_M_SEQUENCE))
            dist.barrier()
            step_seconds = time.perf_counter() - started
        handle.remove()
        if not first.optimizer_stepped or first.optimizer_step != 1:
            raise RuntimeError("10M FSDP2 policy run did not complete optimizer step 1")
        if first.grad_norm is None or not math.isfinite(first.grad_norm):
            raise RuntimeError("10M FSDP2 policy run produced invalid grad norm")
        gradient_state_bytes = captured.get("gradient_state_bytes", 0)
        if gradient_state_bytes <= 0:
            raise RuntimeError("optimizer pre-hook did not observe sharded gradients")
        optimizer_state_bytes = _optimizer_state_bytes(optimizer)
        checkpoint_snapshot = _snapshot_sha256(model)
        identity = _identity(stage, trainer, policy)
        manifest = save_scale_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            plan=plan,
            identity=identity,
            metadata={
                "authority": "SCALE_144_LOCAL_FREE_CPU_GLOO_ONLY",
                "policy": policy.value,
                "stage": "S3_10M",
                "parameter_count": stage.model.parameter_count(),
            },
            rank_state={
                "micro_step": trainer.micro_step,
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
            },
        )
        control = trainer.train_microbatch(_batch(stage, 1, TEN_M_SEQUENCE))
        control_snapshot = _snapshot_sha256(model)
        control_loss = control.loss
        del trainer, optimizer, model
        gc.collect()
        dist.barrier()

        stage2, model2, plan2, _, _, tied_after_reload_build = _build_stack(stage_path, policy)
        optimizer2 = build_optimizer(model2, config)
        trainer2 = FSDP2Trainer(model2, config, device="cpu", optimizer=optimizer2)

        def restore_rank_state(value: Any) -> None:
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
        checkpoint_exact = _snapshot_sha256(model2) == checkpoint_snapshot
        resumed = trainer2.train_microbatch(_batch(stage2, 1, TEN_M_SEQUENCE))
        final_exact = _snapshot_sha256(model2) == control_snapshot
        loss_exact = resumed.loss == control_loss
        if not (
            loaded.exact_topology
            and loaded.exact_trajectory_claim_allowed
            and checkpoint_exact
            and final_exact
            and loss_exact
            and trainer2.optimizer_step == 2
        ):
            raise RuntimeError("DCP exact continuation invariant failed for policy " + policy.value)

        row = {
            "rank": rank,
            "world_size": WORLD_SIZE,
            "backend": "gloo",
            "device_type": "cpu",
            "policy": policy.value,
            "parameter_count": stage.model.parameter_count(),
            "model_spec_sha256": stage.model.identity_sha256(),
            "parameter_state_bytes": parameter_state_bytes,
            "gradient_state_bytes": gradient_state_bytes,
            "optimizer_state_bytes_after_step": optimizer_state_bytes,
            "persistent_train_state_bytes_after_step": parameter_state_bytes
            + gradient_state_bytes
            + optimizer_state_bytes,
            "rss_before_step_bytes": rss_before,
            "peak_rss_step_bytes": sampler.peak_bytes,
            "peak_rss_step_delta_bytes": max(0, sampler.peak_bytes - rss_before),
            "step_seconds": step_seconds,
            "tokens_in_step": first.tokens,
            "tokens_per_second": first.tokens / step_seconds,
            "loss_step_1": first.loss,
            "grad_norm_step_1": first.grad_norm,
            "group_parameter_bytes": groups,
            "communication_proxy": _communication_proxy(groups, policy),
            "tied_alias_before_shard": tied_before,
            "tied_alias_after_shard": tied_after,
            "tied_alias_after_reload_build": tied_after_reload_build,
            "checkpoint_identity_sha256": identity.sha256,
            "checkpoint_aggregate_sha256": loaded.aggregate_checkpoint_sha256,
            "checkpoint_manifest_identity_sha256": manifest["identity_sha256"],
            "dcp_exact_topology": loaded.exact_topology,
            "dcp_exact_trajectory_claim_allowed": loaded.exact_trajectory_claim_allowed,
            "checkpoint_local_shard_exact_after_reload": checkpoint_exact,
            "continued_local_shard_exact": final_exact,
            "continued_loss_exact": loss_exact,
            "control_loss_step_2": control_loss,
            "resumed_loss_step_2": resumed.loss,
            "final_optimizer_step": trainer2.optimizer_step,
            "final_local_shard_sha256": control_snapshot,
        }
        Path(result_path).write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hundred_m_worker(
    rank: int,
    init_file: str,
    stage_path: str,
    policy_value: str,
    result_path: str,
) -> None:
    _configure_worker_threads()
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        policy = FSDP2ReshardPolicy(policy_value)
        stage, model, _plan, groups, tied_before, tied_after = _build_stack(stage_path, policy)
        parameter_state_bytes_before_forward = _parameter_bytes(model)
        rss_before = _current_rss_bytes()
        batch = _batch(stage, 0, HUNDRED_M_SEQUENCE)
        dist.barrier()
        with _PeakRssSampler() as sampler:
            started = time.perf_counter()
            with torch.no_grad():
                output = model(batch["input_ids"])
            dist.barrier()
            forward_seconds = time.perf_counter() - started
        if output.logits.shape != (1, HUNDRED_M_SEQUENCE, stage.model.vocab_size):
            raise RuntimeError("100M forward produced unexpected logits shape")
        registered_parameter_bytes_after_forward = _parameter_bytes(model)
        tied_after_forward = (
            not stage.model.tie_word_embeddings
            or model.token_embedding.weight is model.lm_head.weight
        )
        if not tied_after_forward:
            raise RuntimeError("100M forward broke tied embedding/head alias")
        row = {
            "rank": rank,
            "world_size": WORLD_SIZE,
            "backend": "gloo",
            "device_type": "cpu",
            "policy": policy.value,
            "boundary": "materialized_forward_only_not_training_evidence",
            "parameter_count": stage.model.parameter_count(),
            "model_spec_sha256": stage.model.identity_sha256(),
            "parameter_state_bytes_before_forward": parameter_state_bytes_before_forward,
            "registered_parameter_bytes_after_forward": registered_parameter_bytes_after_forward,
            "rss_before_forward_bytes": rss_before,
            "peak_rss_forward_bytes": sampler.peak_bytes,
            "peak_rss_forward_delta_bytes": max(0, sampler.peak_bytes - rss_before),
            "forward_seconds": forward_seconds,
            "group_parameter_bytes": groups,
            "communication_proxy_if_training": _communication_proxy(groups, policy),
            "tied_alias_before_shard": tied_before,
            "tied_alias_after_shard": tied_after,
            "tied_alias_after_forward": tied_after_forward,
        }
        Path(result_path).write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_rank(worker: Any, args: tuple[Any, ...], result_dir: Path, prefix: str) -> list[dict[str, Any]]:
    descriptor, init_file = tempfile.mkstemp(prefix=f"scale144-{prefix}-")
    os.close(descriptor)
    os.unlink(init_file)
    context = mp.get_context("spawn")
    result_paths = [result_dir / f"{prefix}-rank-{rank}.json" for rank in range(WORLD_SIZE)]
    processes = [
        context.Process(target=worker, args=(rank, init_file, *args, str(result_paths[rank])))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(900)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)
    exitcodes = [process.exitcode for process in processes]
    if exitcodes != [0, 0]:
        raise RuntimeError(f"{prefix} workers failed with exit codes {exitcodes}")
    if os.path.exists(init_file):
        os.unlink(init_file)
    return [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]


def _max_rank(rows: list[dict[str, Any]], field: str) -> float:
    return max(float(row[field]) for row in rows)


def _summarize_10m(policy_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summaries = []
    for policy in POLICIES:
        rows = policy_rows[policy.value]
        summaries.append(
            {
                "policy": policy.value,
                "max_rank_step_seconds": _max_rank(rows, "step_seconds"),
                "min_rank_tokens_per_second": min(float(row["tokens_per_second"]) for row in rows),
                "max_rank_peak_rss_step_bytes": int(_max_rank(rows, "peak_rss_step_bytes")),
                "max_rank_peak_rss_step_delta_bytes": int(_max_rank(rows, "peak_rss_step_delta_bytes")),
                "per_rank_parameter_state_bytes": [row["parameter_state_bytes"] for row in rows],
                "per_rank_gradient_state_bytes": [row["gradient_state_bytes"] for row in rows],
                "per_rank_optimizer_state_bytes": [row["optimizer_state_bytes_after_step"] for row in rows],
                "per_rank_persistent_train_state_bytes": [row["persistent_train_state_bytes_after_step"] for row in rows],
                "per_rank_transfer_proxy_bytes": rows[0]["communication_proxy"]["per_rank_transfer_proxy_bytes"],
                "all_dcp_exact": all(
                    row["checkpoint_local_shard_exact_after_reload"]
                    and row["continued_local_shard_exact"]
                    and row["continued_loss_exact"]
                    for row in rows
                ),
                "all_tied_alias_preserved": all(
                    row["tied_alias_before_shard"]
                    and row["tied_alias_after_shard"]
                    and row["tied_alias_after_reload_build"]
                    for row in rows
                ),
                "loss_step_1_by_rank": [row["loss_step_1"] for row in rows],
                "grad_norm_step_1_by_rank": [row["grad_norm_step_1"] for row in rows],
                "final_local_shard_sha256_by_rank": [row["final_local_shard_sha256"] for row in rows],
            }
        )
    return summaries


def _select_cpu_gloo_default(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["policy"]: row for row in summaries}
    full = by_name[FSDP2ReshardPolicy.FULL_SHARD.value]
    root = by_name[FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED.value]
    shard_grad = by_name[FSDP2ReshardPolicy.SHARD_GRAD_OP.value]
    correctness = all(row["all_dcp_exact"] and row["all_tied_alias_preserved"] for row in summaries)
    if not correctness:
        return {"policy": None, "status": "NO_DEFAULT_CORRECTNESS_FAILURE"}

    root_time_ratio = root["max_rank_step_seconds"] / full["max_rank_step_seconds"]
    root_peak_ratio = root["max_rank_peak_rss_step_bytes"] / full["max_rank_peak_rss_step_bytes"]
    if root_time_ratio <= 1.05 and root_peak_ratio <= 1.05:
        return {
            "policy": FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED.value,
            "status": "DEFAULT_FOR_TESTED_2RANK_CPU_GLOO_ONLY",
            "rule": "prefer PyTorch-recommended root retention when within 5% of FULL_SHARD max-rank step time and peak RSS",
            "root_vs_full_step_time_ratio": root_time_ratio,
            "root_vs_full_peak_rss_ratio": root_peak_ratio,
            "shard_grad_op_transfer_proxy_ratio_vs_full": shard_grad["per_rank_transfer_proxy_bytes"]
            / full["per_rank_transfer_proxy_bytes"],
        }
    return {
        "policy": FSDP2ReshardPolicy.FULL_SHARD.value,
        "status": "DEFAULT_FOR_TESTED_2RANK_CPU_GLOO_ONLY",
        "rule": "fall back to current FULL_SHARD when root retention exceeds a 5% observed time or peak-RSS penalty",
        "root_vs_full_step_time_ratio": root_time_ratio,
        "root_vs_full_peak_rss_ratio": root_peak_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ten-m-stage",
        type=Path,
        default=Path("configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json"),
    )
    parser.add_argument(
        "--hundred-m-stage",
        type=Path,
        default=Path("configs/stages/s4_100m_accelerator.candidate.json"),
    )
    args = parser.parse_args()
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("SCALE-144 LOCAL_FREE evidence requires PyTorch Gloo")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scale144-results-") as tmp:
        result_dir = Path(tmp)
        ten_m_rows: dict[str, list[dict[str, Any]]] = {}
        hundred_m_rows: dict[str, list[dict[str, Any]]] = {}
        for policy in POLICIES:
            checkpoint = result_dir / f"checkpoint-{policy.value}"
            ten_m_rows[policy.value] = _run_two_rank(
                _ten_m_worker,
                (
                    str(args.ten_m_stage.resolve()),
                    policy.value,
                    str(checkpoint),
                ),
                result_dir,
                f"10m-{policy.value}",
            )
        for policy in POLICIES:
            hundred_m_rows[policy.value] = _run_two_rank(
                _hundred_m_worker,
                (str(args.hundred_m_stage.resolve()), policy.value),
                result_dir,
                f"100m-{policy.value}",
            )

        ten_m_summary = _summarize_10m(ten_m_rows)
        first_hashes = {
            policy: tuple(row["final_local_shard_sha256"] for row in rows)
            for policy, rows in ten_m_rows.items()
        }
        exact_cross_policy_final = len(set(first_hashes.values())) == 1
        first_losses = {
            policy: tuple(float(row["loss_step_1"]) for row in rows)
            for policy, rows in ten_m_rows.items()
        }
        max_loss_delta = max(
            abs(value - first_losses[FSDP2ReshardPolicy.FULL_SHARD.value][rank])
            for values in first_losses.values()
            for rank, value in enumerate(values)
        )
        report = {
            "schema_version": 1,
            "worker_id": "SCALE-144-FSDP2-SHARD-POLICY",
            "authority": "LOCAL_FREE_ENGINEERING_EVIDENCE_ONLY",
            "source_sha": os.environ.get("GITHUB_SHA"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "gloo_available": dist.is_gloo_available(),
                "nccl_available": dist.is_nccl_available(),
                "cuda_nccl_executed": False,
            },
            "tested_topology": {
                "world_size": WORLD_SIZE,
                "backend": "gloo",
                "device_type": "cpu",
                "single_host": True,
                "multi_node_claim_allowed": False,
            },
            "policies": [policy.value for policy in POLICIES],
            "ten_m": {
                "stage_path": str(args.ten_m_stage),
                "parameter_count": next(iter(ten_m_rows.values()))[0]["parameter_count"],
                "sequence_length": TEN_M_SEQUENCE,
                "rank_rows": ten_m_rows,
                "summary": ten_m_summary,
                "cross_policy_exact_final_local_shards": exact_cross_policy_final,
                "cross_policy_max_abs_loss_delta_step_1": max_loss_delta,
            },
            "hundred_m_boundary": {
                "stage_path": str(args.hundred_m_stage),
                "parameter_count": next(iter(hundred_m_rows.values()))[0]["parameter_count"],
                "sequence_length": HUNDRED_M_SEQUENCE,
                "scope": "real two-rank materialization plus no-grad forward only; no backward, optimizer state, throughput, or training claim",
                "rank_rows": hundred_m_rows,
            },
            "engineering_default": _select_cpu_gloo_default(ten_m_summary),
            "truth_boundary": [
                "CPU/Gloo timing is not a CUDA/NCCL or multi-node performance predictor.",
                "The ~100M run is a real materialization/forward boundary, not 100M training evidence.",
                "Collective bytes are an algorithm-independent logical tensor-payload proxy, not measured wire bytes.",
                "No paid compute was used.",
            ],
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "default": report["engineering_default"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
