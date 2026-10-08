"""Plan 7 S11: fail-closed tiny checkpoint transport and topology telemetry.

Immutable local fixture publication only; not a network, GPU or training backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from twelve_six.billion_systems_gate import _digest, _sha
from twelve_six.distributed_control import (
    ParallelMesh, RunLedger, _verify_ledger, _verify_worker_shards,
    adapter_packet, resume_after_loss,
)


class TransportDenied(ValueError):
    """No unqualified transport can publish a checkpoint."""


@dataclass(frozen=True, slots=True)
class TransportPlan:
    schema_version: int
    artifact_id: str
    run_sha256: str
    recipe_sha256: str
    data_order_sha256: str
    mesh_sha256: str
    ledger_sha256: str
    checkpoint_sha256: str
    epoch: int
    shards: tuple[tuple[str, str, int], ...]  # owner, sha256, bytes
    storage_budget_bytes: int
    network_budget_bytes: int
    plan_sha256: str


def _payload(plan: TransportPlan) -> dict:
    return {k: v for k, v in asdict(plan).items() if k != "plan_sha256"}


def _validate(plan: TransportPlan) -> None:
    if not isinstance(plan, TransportPlan) or type(plan.schema_version) is not int:
        raise TransportDenied("typed transport plan required")
    if plan.schema_version != 1 or plan.plan_sha256 != _digest(_payload(plan)):
        raise TransportDenied("transport plan tampered")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", plan.artifact_id):
        raise TransportDenied("unsafe artifact identifier")
    for v in (plan.run_sha256, plan.recipe_sha256, plan.data_order_sha256,
              plan.mesh_sha256, plan.ledger_sha256, plan.checkpoint_sha256):
        _sha("identity", v)
    if (type(plan.epoch) is not int or plan.epoch < 0
            or type(plan.shards) is not tuple or not plan.shards
            or any(type(owner) is not str or type(size) is not int or size < 1
                   or type(digest) is not str or len(digest) != 64
                   for owner, digest, size in plan.shards)):
        raise TransportDenied("invalid manifest")
    if (type(plan.storage_budget_bytes) is not int
            or type(plan.network_budget_bytes) is not int
            or plan.storage_budget_bytes <= 0 or plan.network_budget_bytes <= 0):
        raise TransportDenied("invalid resource budget")
    total = sum(s[2] for s in plan.shards)
    if total * 2 > plan.storage_budget_bytes or total > plan.network_budget_bytes:
        raise TransportDenied("insufficient storage/network admission")


def prepare_transfer(mesh: ParallelMesh, ledger: RunLedger,
                     shards: tuple[tuple[str, bytes, str], ...],
                     artifact_id: str, *, storage_budget_bytes: int,
                     network_budget_bytes: int) -> TransportPlan:
    _verify_ledger(ledger, mesh)
    if ledger.checkpoint_sha256 is None:
        raise TransportDenied("no committed checkpoint")
    checkpoint = _verify_worker_shards(mesh, shards)
    if checkpoint != ledger.checkpoint_sha256:
        raise TransportDenied("shard root differs from committed ledger")
    packet = adapter_packet(mesh)
    fields = dict(schema_version=1, artifact_id=artifact_id,
                  run_sha256=mesh.run_sha256, recipe_sha256=mesh.recipe_sha256,
                  data_order_sha256=mesh.data_order_sha256,
                  mesh_sha256=packet["packet_sha256"],
                  ledger_sha256=ledger.ledger_sha256, checkpoint_sha256=checkpoint,
                  epoch=mesh.epoch,
                  shards=tuple((owner, digest, len(data)) for owner, data, digest
                               in shards),
                  storage_budget_bytes=storage_budget_bytes,
                  network_budget_bytes=network_budget_bytes)
    plan = TransportPlan(**fields, plan_sha256=_digest(fields))
    _validate(plan)
    return plan


@dataclass(frozen=True, slots=True)
class TransferReceipt:
    plan_sha256: str
    owner: str
    shard_sha256: str
    byte_count: int
    elapsed_ms: float
    throughput_bytes_per_second: float


def receive(plan: TransportPlan, owner: str, data: bytes,
            elapsed_ms: float) -> TransferReceipt:
    _validate(plan)
    if (type(owner) is not str or type(data) is not bytes
            or type(elapsed_ms) not in (int, float)
            or not math.isfinite(elapsed_ms) or elapsed_ms <= 0):
        raise TransportDenied("invalid transport receipt")
    matched = [item for item in plan.shards if item[0] == owner]
    if len(matched) != 1:
        raise TransportDenied("unknown or duplicated owner")
    _, digest, size = matched[0]
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise TransportDenied("transfer truncated or corrupted")
    rate = size * 1000.0 / elapsed_ms
    if not math.isfinite(rate):
        raise TransportDenied("nonfinite transfer throughput")
    return TransferReceipt(plan.plan_sha256, owner, digest, size,
                           float(elapsed_ms), rate)


def verify_receipts(plan: TransportPlan,
                    receipts: tuple[TransferReceipt, ...],
                    *, recovery_ms: float = 0.0) -> dict:
    _validate(plan)
    if (type(receipts) is not tuple or len(receipts) != len(plan.shards)
            or any(not isinstance(r, TransferReceipt) for r in receipts)):
        raise TransportDenied("partial transport is never canonical")
    if (type(recovery_ms) not in (int, float)
            or not math.isfinite(recovery_ms) or recovery_ms < 0):
        raise TransportDenied("untrusted recovery metric")
    for expected, r in zip(plan.shards, receipts):
        if (r.plan_sha256 != plan.plan_sha256
                or (r.owner, r.shard_sha256, r.byte_count) != expected
                or type(r.elapsed_ms) not in (int, float)
                or not math.isfinite(r.elapsed_ms) or r.elapsed_ms <= 0
                or type(r.throughput_bytes_per_second) not in (int, float)
                or not math.isfinite(r.throughput_bytes_per_second)
                or abs(r.throughput_bytes_per_second
                       - r.byte_count * 1000.0 / r.elapsed_ms) > 1e-6):
            raise TransportDenied("receipt mismatch, replay or metric forgery")
    slowest = max(receipts, key=lambda r: r.elapsed_ms)
    total = sum(r.byte_count for r in receipts)
    duration = max(r.elapsed_ms for r in receipts)
    return {"schema_version": 1, "plan_sha256": plan.plan_sha256,
            "checkpoint_sha256": plan.checkpoint_sha256,
            "status": "VERIFIED_NOT_PUBLISHED", "physical_gpu_verified": False,
            "training_authorized": False, "paid_compute_authorized": False,
            "topology_workers": len(receipts),
            "straggler": slowest.owner,
            "bandwidth_bytes_per_second": total * 1000.0 / duration,
            "storage_peak_bytes_estimate": 2 * total,
            "checkpoint_duration_ms": duration,
            "recovery_time_ms": float(recovery_ms)}


def _root(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise TransportDenied("trusted non-symlink root required")


def _readback_bytes(folder: Path, plan: TransportPlan) -> None:
    if not folder.is_dir() or folder.is_symlink():
        raise TransportDenied("untrusted checkpoint directory")
    expected = {f"shard-{i:03}.bin" for i in range(len(plan.shards))}
    expected.add("manifest.json")
    if {p.name for p in folder.iterdir()} != expected:
        raise TransportDenied("missing/extra checkpoint objects")
    for i, (_, digest, size) in enumerate(plan.shards):
        item = folder / f"shard-{i:03}.bin"
        if item.is_symlink() or not item.is_file():
            raise TransportDenied("symlink or missing shard")
        data = item.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise TransportDenied("disk readback mismatch")
    manifest = folder / "manifest.json"
    if (manifest.is_symlink()
            or json.loads(manifest.read_text("utf-8"))
            != json.loads(json.dumps(asdict(plan)))):
        raise TransportDenied("manifest readback mismatch")


def publish_local(root: Path, plan: TransportPlan,
                  shards: tuple[tuple[str, bytes, str], ...],
                  receipts: tuple[TransferReceipt, ...]) -> Path:
    """Durable bounded fixture write; stage is never an accepted checkpoint."""
    verify_receipts(plan, receipts)
    root = Path(root)
    _root(root)
    if len(shards) != len(plan.shards):
        raise TransportDenied("partial shard set")
    final = root / plan.artifact_id
    if final.exists() or final.is_symlink():
        raise TransportDenied("checkpoint ID already published")
    stage = Path(tempfile.mkdtemp(prefix=".transfer-", dir=root))
    try:
        for i, (owner, data, digest) in enumerate(shards):
            if (type(data) is not bytes or (owner, digest, len(data)) != plan.shards[i]
                    or hashlib.sha256(data).hexdigest() != digest):
                raise TransportDenied("shard mutated after transport receipt")
            with (stage / f"shard-{i:03}.bin").open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        with (stage / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(asdict(plan), handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        _readback_bytes(stage, plan)
        os.rename(stage, final)  # atomic same-filesystem directory publication
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def readback_local(root: Path, plan: TransportPlan) -> dict:
    _validate(plan)
    root = Path(root)
    _root(root)
    _readback_bytes(root / plan.artifact_id, plan)
    return {"status": "LOCAL_FIXTURE_PUBLISHED_VERIFIED",
            "checkpoint_sha256": plan.checkpoint_sha256,
            "plan_sha256": plan.plan_sha256, "training_authorized": False}


def recover_transport(old_mesh: ParallelMesh, new_mesh: ParallelMesh,
                      ledger: RunLedger,
                      shards: tuple[tuple[str, bytes, str], ...],
                      recovery_ms: float) -> dict:
    if (type(recovery_ms) not in (int, float)
            or not math.isfinite(recovery_ms) or recovery_ms < 0):
        raise TransportDenied("invalid recovery metric")
    resumed = resume_after_loss(old_mesh, new_mesh, ledger, shards)
    return {"schema_version": 1, "run_sha256": resumed.run_sha256,
            "checkpoint_sha256": resumed.checkpoint_sha256,
            "new_epoch": resumed.epoch,
            "next_exposure": resumed.next_exposure,
            "recovery_time_ms": float(recovery_ms),
            "training_authorized": False}
