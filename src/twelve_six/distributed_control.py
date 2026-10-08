"""Plan 7 S10: LOCAL_FREE distributed execution-control and exactly-once proxy ledger.

Adapter-neutral ranks, elastic recovery, sealed tiny shards and deterministic
logical exposure commits. Not a torchrun/optimizer/cloud/production publisher.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import (
    _digest, _finite, _positive, _sha, inspect_shards, shard_manifest,
)
from twelve_six.extreme_moe_paths import MoEPolicy

BACKENDS = frozenset({"FSDP2", "DeepSpeed", "TorchTitan", "Megatron"})
MAX_WORKERS = 128


class DistributedDenied(ValueError):
    """Untrusted topology, partial checkpoint or duplicate exposure."""


@dataclass(frozen=True, slots=True)
class ParallelMesh:
    schema_version: int
    backend: str
    adapter_version_sha256: str
    run_sha256: str
    recipe_sha256: str
    data_order_sha256: str
    epoch: int
    data_parallel: int
    tensor_parallel: int
    pipeline_parallel: int
    context_parallel: int
    expert_parallel: int  # EP is a subgroup of DP, not another world-size factor.
    workers: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DistributedDenied("mesh version")
        if type(self.backend) is not str or self.backend not in BACKENDS:
            raise DistributedDenied("unrecognized replaceable adapter")
        for name in ("adapter_version_sha256", "run_sha256",
                     "recipe_sha256", "data_order_sha256"):
            _sha(name, getattr(self, name))
        if type(self.epoch) is not int or not 0 <= self.epoch <= 1_000_000:
            raise DistributedDenied("mesh epoch")
        for name in ("data_parallel", "tensor_parallel", "pipeline_parallel",
                     "context_parallel", "expert_parallel"):
            _positive(name, getattr(self, name))
        n = (self.data_parallel * self.tensor_parallel
             * self.pipeline_parallel * self.context_parallel)
        if (n > MAX_WORKERS or self.data_parallel % self.expert_parallel
                or type(self.workers) is not tuple or len(self.workers) != n
                or len(set(self.workers)) != n
                or tuple(sorted(self.workers)) != self.workers
                or any(type(w) is not str or not w or len(w) > 64
                       or not w.isascii() for w in self.workers)):
            raise DistributedDenied("invalid DP/TP/PP/CP/EP rank geometry")


def adapter_packet(mesh: ParallelMesh) -> dict[str, Any]:
    if not isinstance(mesh, ParallelMesh):
        raise DistributedDenied("typed mesh required")
    ParallelMesh(**asdict(mesh))  # Detect mutation of frozen evidence.
    ranks = []
    for rank, worker in enumerate(mesh.workers):
        i = rank
        cp = i % mesh.context_parallel
        i //= mesh.context_parallel
        pp = i % mesh.pipeline_parallel
        i //= mesh.pipeline_parallel
        tp = i % mesh.tensor_parallel
        dp = i // mesh.tensor_parallel
        ranks.append({"rank": rank, "worker": worker, "data": dp, "tensor": tp,
                      "pipeline": pp, "context": cp,
                      "expert_group": dp % mesh.expert_parallel})
    record = {
        "schema_version": 1, "backend": mesh.backend,
        "adapter_version_sha256": mesh.adapter_version_sha256,
        "run_sha256": mesh.run_sha256,
        "recipe_sha256": mesh.recipe_sha256,
        "data_order_sha256": mesh.data_order_sha256,
        "epoch": mesh.epoch, "world_size": len(mesh.workers),
        "rank_map": ranks, "mode": "LOCAL_FREE_PROXY_NO_BACKEND_DISPATCH",
        "training_authorized": False, "paid_compute_authorized": False,
    }
    record["packet_sha256"] = _digest(record)
    return record


def bind_expert_groups(mesh: ParallelMesh, policy: MoEPolicy) -> dict[str, Any]:
    if not isinstance(policy, MoEPolicy):
        raise DistributedDenied("typed S9 MoE policy required")
    MoEPolicy(**asdict(policy))
    packet = adapter_packet(mesh)
    if (policy.expert_parallel_groups != mesh.expert_parallel
            or policy.expert_count % mesh.expert_parallel):
        raise DistributedDenied("S9 expert placement and S10 mesh mismatch")
    return {"schema_version": 1, "mesh_sha256": packet["packet_sha256"],
            "policy_sha256": _digest(asdict(policy)),
            "mode": "EXPERT_GROUP_FIXTURE_ONLY", "launch_authorized": False}


@dataclass(frozen=True, slots=True)
class ResourceAdmission:
    schema_version: int
    free_bytes_per_worker: int
    required_bytes_per_worker: int
    checkpoint_free_bytes: int
    checkpoint_required_bytes: int
    interconnect_bytes_per_second: float
    minimum_interconnect_bytes_per_second: float
    estimated_dollars: float
    max_dollars: float
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DistributedDenied("resource version")
        for key in ("free_bytes_per_worker", "required_bytes_per_worker",
                    "checkpoint_free_bytes", "checkpoint_required_bytes"):
            _positive(key, getattr(self, key))
        for key in ("interconnect_bytes_per_second",
                    "minimum_interconnect_bytes_per_second",
                    "estimated_dollars", "max_dollars"):
            _finite(key, getattr(self, key), allow_zero=True)
        if type(self.paid_compute_requested) is not bool:
            raise DistributedDenied("paid compute authority type")


def admit_distributed(mesh: ParallelMesh, evidence: ResourceAdmission) -> dict[str, Any]:
    packet = adapter_packet(mesh)
    if not isinstance(evidence, ResourceAdmission):
        raise DistributedDenied("typed resources required")
    ResourceAdmission(**asdict(evidence))
    reasons = []
    if evidence.paid_compute_requested:
        reasons.append("paid_compute_denied")
    if evidence.free_bytes_per_worker < evidence.required_bytes_per_worker:
        reasons.append("worker_memory_denied")
    if evidence.checkpoint_free_bytes < evidence.checkpoint_required_bytes:
        reasons.append("checkpoint_storage_denied")
    if (evidence.interconnect_bytes_per_second
            < evidence.minimum_interconnect_bytes_per_second):
        reasons.append("interconnect_denied")
    if evidence.estimated_dollars > evidence.max_dollars:
        reasons.append("budget_denied")
    record = {
        "schema_version": 1, "mesh_sha256": packet["packet_sha256"],
        "resource_sha256": _digest(asdict(evidence)),
        "status": "NO_GO" if reasons else "FIXTURE_ADMISSIBLE_NOT_AUTHORIZED",
        "reasons": sorted(reasons), "launch_authorized": False,
        "training_authorized": False, "paid_compute_authorized": False,
    }
    record["packet_sha256"] = _digest(record)
    return record


@dataclass(frozen=True, slots=True)
class RunLedger:
    schema_version: int
    run_sha256: str
    recipe_sha256: str
    data_order_sha256: str
    epoch: int
    next_step: int
    next_exposure: int
    checkpoint_sha256: str | None
    ledger_sha256: str


def _ledger_payload(ledger: RunLedger) -> dict[str, Any]:
    return {k: v for k, v in asdict(ledger).items() if k != "ledger_sha256"}


def _verify_ledger(ledger: RunLedger, mesh: ParallelMesh) -> None:
    if not isinstance(ledger, RunLedger) or type(ledger.schema_version) is not int:
        raise DistributedDenied("ledger type")
    if ledger.schema_version != 1 or ledger.ledger_sha256 != _digest(_ledger_payload(ledger)):
        raise DistributedDenied("ledger integrity")
    for name in ("run_sha256", "recipe_sha256", "data_order_sha256"):
        if getattr(ledger, name) != getattr(mesh, name):
            raise DistributedDenied("scientific run/recipe/order drift")
    if ledger.epoch != mesh.epoch:
        raise DistributedDenied("stale epoch")
    if (type(ledger.next_step) is not int or ledger.next_step < 0
            or type(ledger.next_exposure) is not int or ledger.next_exposure < 0):
        raise DistributedDenied("invalid committed cursor")
    if ledger.checkpoint_sha256 is not None:
        _sha("checkpoint root", ledger.checkpoint_sha256)


def _seal_ledger(fields: dict[str, Any]) -> RunLedger:
    return RunLedger(**fields, ledger_sha256=_digest(fields))


def begin_run(mesh: ParallelMesh) -> RunLedger:
    adapter_packet(mesh)
    return _seal_ledger({
        "schema_version": 1, "run_sha256": mesh.run_sha256,
        "recipe_sha256": mesh.recipe_sha256,
        "data_order_sha256": mesh.data_order_sha256,
        "epoch": mesh.epoch, "next_step": 0, "next_exposure": 0,
        "checkpoint_sha256": None,
    })


def begin_step(mesh: ParallelMesh, ledger: RunLedger,
               exposures: tuple[int, ...]) -> dict[str, Any]:
    adapter_packet(mesh)
    _verify_ledger(ledger, mesh)
    if (type(exposures) is not tuple or not 1 <= len(exposures) <= 256
            or any(type(x) is not int for x in exposures)
            or exposures != tuple(range(ledger.next_exposure,
                                       ledger.next_exposure + len(exposures)))):
        raise DistributedDenied("noncanonical data order, gap or repeated exposure")
    result = {
        "schema_version": 1, "parent_ledger_sha256": ledger.ledger_sha256,
        "mesh_sha256": adapter_packet(mesh)["packet_sha256"],
        "run_sha256": mesh.run_sha256,
        "step": ledger.next_step, "epoch": mesh.epoch,
        "exposures": exposures, "exposure_sha256": _digest(exposures),
        "status": "PREPARED_NOT_COMMITTED",
    }
    result["ticket_sha256"] = _digest(result)
    return result


def _verify_worker_shards(mesh: ParallelMesh,
                          shards: tuple[tuple[str, bytes, str], ...]) -> str:
    if (type(shards) is not tuple or len(shards) != len(mesh.workers)
            or any(not isinstance(x, tuple) or len(x) != 3 for x in shards)):
        raise DistributedDenied("partial worker set; no checkpoint publication")
    group_receipts = []
    for start in range(0, len(shards), 16):
        group = shards[start:start + 16]
        parts = []
        for local_rank, (worker, data, declared_sha) in enumerate(group):
            if worker != mesh.workers[start + local_rank]:
                raise DistributedDenied("missing/reordered/duplicate worker")
            if type(data) is not bytes or not 0 < len(data) <= 65_536:
                raise DistributedDenied("invalid bounded shard")
            _sha("shard", declared_sha)
            if hashlib.sha256(data).hexdigest() != declared_sha:
                raise DistributedDenied("shard hash mismatch")
            parts.append(data)
        named = tuple((f"part-{i:02d}", part, hashlib.sha256(part).hexdigest())
                      for i, part in enumerate(parts))
        group_receipts.append(inspect_shards(named, shard_manifest(tuple(parts))))
    return _digest(group_receipts)


def commit_step(mesh: ParallelMesh, ledger: RunLedger, ticket: dict[str, Any],
                shards: tuple[tuple[str, bytes, str], ...]) -> RunLedger:
    _verify_ledger(ledger, mesh)
    if type(ticket) is not dict:
        raise DistributedDenied("invalid step ticket")
    if (ticket.get("ticket_sha256") != _digest({
            k: v for k, v in ticket.items() if k != "ticket_sha256"})
            or ticket.get("parent_ledger_sha256") != ledger.ledger_sha256
            or ticket.get("mesh_sha256") != adapter_packet(mesh)["packet_sha256"]
            or ticket.get("step") != ledger.next_step
            or ticket.get("epoch") != mesh.epoch
            or ticket.get("run_sha256") != mesh.run_sha256
            or ticket.get("status") != "PREPARED_NOT_COMMITTED"):
        raise DistributedDenied("stale/forged prepared step")
    exposures = ticket.get("exposures")
    if (type(exposures) not in (tuple, list)
            or ticket.get("exposure_sha256") != _digest(exposures)):
        raise DistributedDenied("exposure digest forged")
    # Re-derive from incumbent cursor, not from a caller-claimed count.
    expected = begin_step(mesh, ledger, tuple(exposures))
    if ticket["ticket_sha256"] != expected["ticket_sha256"]:
        raise DistributedDenied("prepared step differs from canonical order")
    # This is an atomic *proxy* commit only after all physical tiny bytes verify.
    checkpoint_sha = _verify_worker_shards(mesh, shards)
    return _seal_ledger({
        "schema_version": 1, "run_sha256": ledger.run_sha256,
        "recipe_sha256": ledger.recipe_sha256,
        "data_order_sha256": ledger.data_order_sha256,
        "epoch": mesh.epoch, "next_step": ledger.next_step + 1,
        "next_exposure": ledger.next_exposure + len(exposures),
        "checkpoint_sha256": checkpoint_sha,
    })


def resume_after_loss(old_mesh: ParallelMesh, new_mesh: ParallelMesh,
                      ledger: RunLedger,
                      previous_shards: tuple[tuple[str, bytes, str], ...]) -> RunLedger:
    adapter_packet(old_mesh)
    adapter_packet(new_mesh)
    _verify_ledger(ledger, old_mesh)
    if ledger.checkpoint_sha256 is None:
        raise DistributedDenied("cannot resume from an uncommitted checkpoint")
    if (new_mesh.epoch != old_mesh.epoch + 1
            or new_mesh.adapter_version_sha256 != old_mesh.adapter_version_sha256
            or any(getattr(old_mesh, name) != getattr(new_mesh, name)
                   for name in ("run_sha256", "recipe_sha256", "data_order_sha256"))):
        raise DistributedDenied("recovery changed scientific run, adapter or epoch")
    if _verify_worker_shards(old_mesh, previous_shards) != ledger.checkpoint_sha256:
        raise DistributedDenied("checkpoint changed since commit")
    return _seal_ledger({
        "schema_version": 1, "run_sha256": ledger.run_sha256,
        "recipe_sha256": ledger.recipe_sha256,
        "data_order_sha256": ledger.data_order_sha256,
        "epoch": new_mesh.epoch, "next_step": ledger.next_step,
        "next_exposure": ledger.next_exposure,
        "checkpoint_sha256": ledger.checkpoint_sha256,
    })
