from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

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
from twelve_six.model import TwelveSixDecoder, canonical_json_sha256, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import build_optimizer

SCHEMA_VERSION = "12-6.checkpoint204-dcp-cuda-recovery.v1"
AUTHORITY = "CHECKPOINT204_REAL_CUDA_ONLY_WHEN_TWO_AUTHORIZED_GPUS_VISIBLE"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cuda_preflight() -> dict[str, Any]:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": device_count,
        "nccl_available": bool(dist.is_available() and dist.is_nccl_available()),
        "devices": devices,
    }


def select_last_known_good(
    checkpoint_root: Path,
    *,
    verifier: Callable[[Path], dict[str, Any]] = verify_scale_checkpoint,
) -> tuple[Path, list[dict[str, str]]]:
    candidates = sorted(
        (path for path in checkpoint_root.glob("generation-*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            verifier(candidate)
        except Exception as exc:  # fail-closed: newer invalid generations are never selected
            rejected.append({"generation": candidate.name, "reason": type(exc).__name__})
            continue
        return candidate, rejected
    raise RuntimeError("no verified last-known-good DCP generation exists")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight_report(source_sha: str) -> dict[str, Any]:
    hardware = cuda_preflight()
    runnable = hardware["cuda_device_count"] >= 2 and hardware["nccl_available"]
    return {
        "schema_version": SCHEMA_VERSION,
        "swarm_worker_id": "CHECKPOINT-204-DCP-CUDA-RECOVERY",
        "source_sha": source_sha,
        "status": "READY_FOR_REAL_CUDA_RECOVERY" if runnable else "CUDA_UNTESTED_NO_MULTI_GPU",
        "hardware": hardware,
        "checkpoint_format_changed": False,
        "async_dcp_enabled": False,
        "cpu_gloo_incumbent": {
            "source_sha": "c5212411d09d6ce4189dfbd4d3c182b52840cd7c",
            "test": "tests/test_scale_fsdp2_dcp_integration.py",
            "authority": "terminal-success two-rank CPU/Gloo native FSDP2+DCP integration",
        },
        "claims": {
            "cuda_validated": runnable,
            "paid_compute_used": False,
            "production_readiness": False,
        },
    }


def _batch(index: int, *, vocab_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20_400 + index)
    input_ids = torch.randint(0, vocab_size, (1, 16), generator=generator, dtype=torch.long)
    return {"input_ids": input_ids.to(device)}


def _snapshot(model: TwelveSixDecoder) -> tuple[torch.Tensor, ...]:
    rows: list[torch.Tensor] = []
    for parameter in model.parameters():
        to_local = getattr(parameter, "to_local", None)
        local = to_local() if callable(to_local) else parameter
        rows.append(local.detach().cpu().clone())
    return tuple(rows)


def _snapshot_equal(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> bool:
    return len(left) == len(right) and all(
        a.shape == b.shape and torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _build_stack(stage_path: Path, local_rank: int, world_size: int):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20_426)
    torch.cuda.manual_seed_all(20_426)
    stage = load_stage_config(stage_path)
    model = TwelveSixDecoder(stage.model, stage.init).to(device)
    plan = ParallelPlan(data_parallel=world_size, shard_model_state_across_data_parallel=True)
    mesh_spec = build_torch_mesh_spec(plan, fsdp_shard_degree=world_size)
    mesh = mesh_spec.create_device_mesh("cuda")
    model = apply_fsdp2(model, **mesh_spec.fsdp2_kwargs(mesh, reshard_after_forward=True))
    config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=3,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=20_426,
        deterministic_algorithms=True,
    )
    optimizer = build_optimizer(model, config)
    trainer = FSDP2Trainer(model, config, device=device, optimizer=optimizer)
    return stage, model, optimizer, trainer, plan, config, device


def _identity(
    stage,
    trainer: FSDP2Trainer,
    config: TrainerConfig,
    source_sha: str,
) -> ScaleCheckpointIdentity:
    purpose_profile = Path("requirements/profiles/linux-x86_64-cuda-training/profile.json")
    training_identity = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": list(config.betas),
        "eps": config.eps,
        "max_steps": config.max_steps,
        "warmup_steps": config.warmup_steps,
        "scheduler": config.scheduler,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clip_norm": config.gradient_clip_norm,
        "precision": config.precision,
        "seed": config.seed,
        "deterministic_algorithms": config.deterministic_algorithms,
        "deterministic_warn_only": config.deterministic_warn_only,
    }
    return ScaleCheckpointIdentity(
        git_sha=source_sha.lower(),
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256=stage.init.identity_sha256(),
        tokenizer_config_sha256=hashlib.sha256(
            b"checkpoint204.synthetic-token-trace.v1"
        ).hexdigest(),
        tokenizer_vocab_sha256=hashlib.sha256(str(stage.model.vocab_size).encode()).hexdigest(),
        data_manifest_sha256=hashlib.sha256(
            b"checkpoint204.bounded-synthetic-trace.v1"
        ).hexdigest(),
        packing_sha256=hashlib.sha256(b"checkpoint204.fixed-seq16.v1").hexdigest(),
        training_config_sha256=canonical_json_sha256(training_identity),
        environment_lock_sha256=_sha256_file(purpose_profile),
        seed=20_426,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )


def _init_group() -> None:
    # torchrun owns the rendezvous store. Destroying and calling init again creates a fresh
    # NCCL default process group without relying on a racy shared file-store lifecycle.
    dist.init_process_group("nccl")


def _wait_for(path: Path, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def run_cuda_recovery(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("CHECKPOINT-204 real CUDA recovery requires at least two ranks/GPUs")
    hardware = cuda_preflight()
    if hardware["cuda_device_count"] <= local_rank or not hardware["nccl_available"]:
        raise RuntimeError("visible CUDA/NCCL topology does not satisfy requested local rank")

    root = args.checkpoint_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    valid_checkpoint = root / "generation-000002"
    invalid_newer = root / "generation-000003"
    ready_marker = root / "fallback-selection.json"
    if rank == 0:
        for path in (valid_checkpoint, invalid_newer, ready_marker):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    _init_group()
    stage, model, optimizer, trainer, plan, config, device = _build_stack(
        args.stage, local_rank, world_size
    )
    if stage.model.parameter_count() < 9_000_000:
        raise AssertionError("CHECKPOINT-204 must use the bounded ~10M stage first")
    first = trainer.train_microbatch(_batch(0, vocab_size=stage.model.vocab_size, device=device))
    second = trainer.train_microbatch(_batch(1, vocab_size=stage.model.vocab_size, device=device))
    if not (first.optimizer_stepped and second.optimizer_stepped and trainer.optimizer_step == 2):
        raise AssertionError("expected two real optimizer updates before DCP commit")
    checkpoint_snapshot = _snapshot(model)
    identity = _identity(stage, trainer, config, args.source_sha)

    dist.barrier()
    save_start = time.perf_counter()
    manifest = save_scale_checkpoint(
        valid_checkpoint,
        model=model,
        optimizer=optimizer,
        plan=plan,
        identity=identity,
        metadata={
            "authority": AUTHORITY,
            "stage": stage.stage,
            "parameter_count": stage.model.parameter_count(),
            "bounded_training_only": True,
            "async_dcp": False,
        },
        rank_state={
            "micro_step": trainer.micro_step,
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
        },
    )
    dist.barrier()
    save_wall = time.perf_counter() - save_start
    if manifest["identity_sha256"] != identity.sha256:
        raise AssertionError("saved checkpoint identity mismatch")

    control = trainer.train_microbatch(_batch(2, vocab_size=stage.model.vocab_size, device=device))
    control_snapshot = _snapshot(model)
    control_loss = float(control.loss)
    del trainer, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        verify_start = time.perf_counter()
        verified = verify_scale_checkpoint(valid_checkpoint)
        verify_wall = time.perf_counter() - verify_start
        if verified["identity_sha256"] != identity.sha256:
            raise AssertionError("standalone verification identity mismatch")
        storage_bytes = _tree_bytes(valid_checkpoint)
        invalid_newer.mkdir()
        (invalid_newer / "COMMITTED").write_text(
            "corrupt-newer-generation\n", encoding="utf-8"
        )
        selected, rejected = select_last_known_good(root)
        if selected != valid_checkpoint:
            raise AssertionError("failed to fall back to last-known-good checkpoint")
        _write_json(
            ready_marker,
            {
                "selected": selected.name,
                "rejected": rejected,
                "verify_wall_seconds": verify_wall,
                "storage_bytes": storage_bytes,
            },
        )
    else:
        _wait_for(ready_marker)
    selection = json.loads(ready_marker.read_text(encoding="utf-8"))
    if selection["selected"] != valid_checkpoint.name:
        raise AssertionError("all ranks must agree on the LKG generation")

    _init_group()
    stage2, model2, optimizer2, trainer2, plan2, _, device2 = _build_stack(
        args.stage, local_rank, world_size
    )

    def restore_rank_state(value: Any) -> None:
        trainer2.micro_step = int(value["micro_step"])
        trainer2.optimizer_step = int(value["optimizer_step"])
        trainer2.tokens_seen = int(value["tokens_seen"])

    dist.barrier()
    load_start = time.perf_counter()
    loaded = load_scale_checkpoint(
        valid_checkpoint,
        model=model2,
        optimizer=optimizer2,
        target_plan=plan2,
        mode=ResumeMode.EXACT_TOPOLOGY,
        expected_identity_sha256=identity.sha256,
        restore_rank_state=restore_rank_state,
    )
    dist.barrier()
    load_wall = time.perf_counter() - load_start
    checkpoint_equal = _snapshot_equal(_snapshot(model2), checkpoint_snapshot)
    resumed = trainer2.train_microbatch(
        _batch(2, vocab_size=stage2.model.vocab_size, device=device2)
    )
    final_equal = _snapshot_equal(_snapshot(model2), control_snapshot)
    loss_equal = float(resumed.loss) == control_loss
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "save_wall_seconds": save_wall,
        "load_wall_seconds": load_wall,
        "checkpoint_shard_exact": checkpoint_equal,
        "final_shard_exact": final_equal,
        "loss_exact": loss_equal,
        "control_loss": control_loss,
        "resumed_loss": float(resumed.loss),
        "restored_optimizer_step": 2,
        "final_optimizer_step": trainer2.optimizer_step,
        "exact_topology": loaded.exact_topology,
        "exact_trajectory_claim_allowed": loaded.exact_trajectory_claim_allowed,
    }
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    dist.barrier()
    if rank == 0:
        rows = [row for row in gathered if row is not None]
        passed = all(
            row["checkpoint_shard_exact"]
            and row["final_shard_exact"]
            and row["loss_exact"]
            and row["exact_topology"]
            and row["exact_trajectory_claim_allowed"]
            for row in rows
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "swarm_worker_id": "CHECKPOINT-204-DCP-CUDA-RECOVERY",
            "source_sha": args.source_sha,
            "status": (
                "PASS_REAL_CUDA_DCP_RECOVERY" if passed else "FAIL_REAL_CUDA_DCP_RECOVERY"
            ),
            "hardware": hardware,
            "topology": {"backend": "nccl", "world_size": world_size},
            "model": {
                "stage": stage.stage,
                "parameter_count": stage.model.parameter_count(),
                "model_spec_sha256": stage.model.identity_sha256(),
                "init_spec_sha256": stage.init.identity_sha256(),
                "bounded_training_only": True,
            },
            "checkpoint": {
                "format_changed": False,
                "async_dcp_enabled": False,
                "identity_sha256": identity.sha256,
                "aggregate_checkpoint_sha256": loaded.aggregate_checkpoint_sha256,
                "committed_generation": valid_checkpoint.name,
                "injected_newer_invalid_generation": invalid_newer.name,
                "fallback_selected_generation": selection["selected"],
                "fallback_rejections": selection["rejected"],
                "storage_bytes": selection["storage_bytes"],
            },
            "process_group": {
                "teardown_after_commit": True,
                "fresh_group_before_load": True,
            },
            "timing": {
                "save_wall_seconds_max_rank": max(row["save_wall_seconds"] for row in rows),
                "verify_wall_seconds_rank0": selection["verify_wall_seconds"],
                "load_wall_seconds_max_rank": max(row["load_wall_seconds"] for row in rows),
            },
            "resume": {
                "precommit_optimizer_steps": 2,
                "post_resume_step": 3,
                "ranks": rows,
            },
            "claims": {
                "cuda_validated": passed,
                "paid_compute_used": False,
                "production_readiness": False,
                "learned_10m_ladder_evidence": False,
            },
        }
        _write_json(args.output, report)
        if not passed:
            raise AssertionError("CHECKPOINT-204 exact recovery comparison failed")
    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHECKPOINT-204 synchronous DCP CUDA recovery probe")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoint204-report.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path(".checkpoint204"))
    parser.add_argument("--stage", type=Path, default=Path("configs/stages/s3_10m.json"))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        _write_json(args.output, _preflight_report(args.source_sha))
        return
    run_cuda_recovery(args)


if __name__ == "__main__":
    main()
