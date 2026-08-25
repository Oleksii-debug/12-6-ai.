from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from twelve_six.checkpoint.core import CheckpointIdentity, hash_json
from twelve_six.checkpoint.v2 import ResumeTopology, load_checkpoint_v2, save_checkpoint_v2

SCHEMA = "12-6.checkpoint-v2-distributed-writer-smoke.v1"
AUTHORITY = "LOCAL_FREE_DISTRIBUTED_CHECKPOINT_MECHANICS_NOT_SHARDED_RESHARD_EVIDENCE"


class RepresentativeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(512, 64)
        self.ff_up = nn.Linear(64, 192, bias=False)
        self.ff_down = nn.Linear(192, 64, bias=False)
        self.lm_head = nn.Linear(64, 512, bias=False)
        self.lm_head.weight = self.embedding.weight


def _populate_optimizer(model: nn.Module) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _fingerprint(model: nn.Module) -> str:
    inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(model.state_dict().items())
    ]
    return hash_json(inventory)


def _identity(source_sha: str, model: nn.Module, world_size: int) -> CheckpointIdentity:
    marker = {
        "schema": SCHEMA,
        "world_size": world_size,
        "scope": "replicated-two-rank-dcp-writer-smoke",
    }
    digest = hash_json(marker)
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec={
            "name": "checkpoint-v2-representative-distributed-smoke",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_hash=digest,
        tokenizer_vocab_hash=digest,
        dataset_manifest_hash=digest,
        run_manifest_hash=digest,
        training_config={"training_executed": False, "distributed_smoke": True},
        seed=20260825,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
        scheduler=None,
    )


def _write_report(
    *,
    output: Path,
    source_sha: str,
    manifest: dict[str, Any],
    all_rank_results: list[dict[str, Any]],
    checkpoint: Path,
) -> None:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "world_size": dist.get_world_size(),
        "writers": manifest["writers"],
        "source_topology": manifest["source_topology"],
        "checkpoint_id": manifest["checkpoint_id"],
        "dcp_files": sorted(path.name for path in (checkpoint / "dcp").glob("*.distcp")),
        "control_files": sorted(path.name for path in (checkpoint / "control").iterdir()),
        "rank_results": all_rank_results,
        "all_rank_round_trip_pass": all(result["round_trip_pass"] for result in all_rank_results),
        "truth_boundary": {
            "replicated_model": True,
            "fsdp_or_dtensor_sharding": "NOT_TESTED",
            "topology_change_reshard": "NOT_TESTED",
            "gpu": "NOT_TESTED",
            "paid_compute": False,
            "quality_or_capability_evidence": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or args.source_sha != args.source_sha.lower() or any(
        char not in "0123456789abcdef" for char in args.source_sha
    ):
        raise ValueError("--source-sha must be an exact lowercase 40-hex Git SHA")

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 2:
            raise RuntimeError("distributed writer smoke requires at least two ranks")
        torch.manual_seed(20260825)
        model = RepresentativeModel()
        optimizer = _populate_optimizer(model)
        expected_fingerprint = _fingerprint(model)
        identity = _identity(args.source_sha, model, world_size)
        topology = ResumeTopology(
            world_size=world_size,
            parallelism={"data": world_size, "tensor": 1, "pipeline": 1},
        )
        manifest = save_checkpoint_v2(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            identity=identity,
            trainer_state={"rank": rank, "distributed_smoke": True},
            topology=topology,
        )

        torch.manual_seed(99 + rank)
        restored = RepresentativeModel()
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=1e-3, weight_decay=0.0
        )
        loaded = load_checkpoint_v2(
            args.checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            expected_identity=identity,
            topology=topology,
        )
        local_result = {
            "rank": rank,
            "model_fingerprint_exact": _fingerprint(restored) == expected_fingerprint,
            "optimizer_state_entries": len(restored_optimizer.state),
            "expected_optimizer_state_entries": len(list(restored.parameters())),
            "trainer_state_rank_exact": loaded.trainer_state.get("rank") == rank,
            "rng_restored": loaded.rng_restored,
        }
        local_result["round_trip_pass"] = bool(
            local_result["model_fingerprint_exact"]
            and local_result["optimizer_state_entries"]
            == local_result["expected_optimizer_state_entries"]
            and local_result["trainer_state_rank_exact"]
            and local_result["rng_restored"]
        )
        gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result)
        rank_results = [result for result in gathered if result is not None]
        if len(rank_results) != world_size or not all(
            result["round_trip_pass"] for result in rank_results
        ):
            raise RuntimeError("distributed checkpoint-v2 round-trip failed")
        if rank == 0:
            _write_report(
                output=args.output,
                source_sha=args.source_sha,
                manifest=manifest,
                all_rank_results=rank_results,
                checkpoint=args.checkpoint,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
