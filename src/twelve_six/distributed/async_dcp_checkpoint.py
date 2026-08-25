"""Evaluation-only asynchronous DCP writer preserving the D18 truth model.

The tensor data plane uses only ``torch.distributed.checkpoint.async_save``. The
D18 control plane remains authoritative: payloads are written below a hidden
staging generation, hashed after async completion, and only then receive the
manifest/COMMITTED marker and same-parent rename used by the synchronous writer.

This module intentionally does not change ``async_checkpoint_gate()``. CHECKPOINT-145
must produce measured correctness/performance evidence before async checkpointing
can become an engineering default.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpoint_layout import rank_identity
from .contracts import ParallelPlan
from .dcp_checkpoint import (
    BACKEND,
    COMMITTED,
    MANIFEST,
    MANIFEST_SHA,
    SCHEMA,
    ScaleCheckpointIdentity,
    _aggregate,
    _agree,
    _artifact_set,
    _dist,
    _hash,
    _json_safe,
    _parallel_artifacts,
    _rank0,
    _topology,
    _write_json,
    state_schema_sha256,
)


def _future_result(future: Any) -> Any:
    """Wait for either the classic Future or newer AsyncSaveResponse shape."""

    upload_completion = getattr(future, "upload_completion", None)
    if upload_completion is not None:
        return upload_completion.result()
    return future.result()


@dataclass(slots=True)
class AsyncScaleCheckpointHandle:
    """One in-flight DCP save whose visible publication is deferred until ``wait``.

    A normal shutdown must call ``wait``/``close`` before destroying the process
    group. ``close`` is intentionally blocking: exiting while DCP still owns a
    background write would make process lifetime part of checkpoint correctness.
    """

    future: Any
    staging: Path
    destination: Path
    plan: ParallelPlan
    identity: ScaleCheckpointIdentity
    identity_hash: str
    schema_hash: str
    metadata_value: dict[str, Any]
    metadata_hash: str
    rank_state_present: bool
    _manifest: dict[str, Any] | None = None
    _failed: str | None = None

    @property
    def published(self) -> bool:
        return self._manifest is not None

    @property
    def requires_wait_before_exit(self) -> bool:
        return self._manifest is None and self._failed is None

    def wait(self) -> dict[str, Any]:
        """Wait for DCP completion, verify payload bytes, then publish atomically."""

        if self._manifest is not None:
            return self._manifest
        if self._failed is not None:
            raise RuntimeError(self._failed)

        dist = _dist(self.plan)
        rank = dist.get_rank()
        local_error: str | None = None
        try:
            _future_result(self.future)
        except Exception as exc:  # noqa: BLE001 - failures must be gathered by every rank
            local_error = f"rank={rank} {type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        failures = [item for item in errors if item is not None]
        if failures:
            message = "asynchronous DCP save failed before publication: " + "; ".join(failures)
            self._failed = message
            raise RuntimeError(message)

        dist.barrier()
        records = _parallel_artifacts(dist, self.staging)

        def publish() -> dict[str, Any]:
            artifacts_hash = _artifact_set(records)
            saved_topology = _topology(self.plan)
            topology_hash = _hash(saved_topology)
            aggregate = _aggregate(
                self.identity_hash,
                self.schema_hash,
                topology_hash,
                artifacts_hash,
                self.metadata_hash,
            )
            manifest = {
                "schema": SCHEMA,
                "backend": BACKEND,
                "identity": asdict(self.identity),
                "identity_sha256": self.identity_hash,
                "state_schema_sha256": self.schema_hash,
                "save_topology": saved_topology,
                "topology_sha256": topology_hash,
                "artifacts": [asdict(item) for item in records],
                "artifact_set_sha256": artifacts_hash,
                "metadata": self.metadata_value,
                "metadata_sha256": self.metadata_hash,
                "rank_state_present": self.rank_state_present,
                "aggregate_checkpoint_sha256": aggregate,
            }
            _write_json(self.staging / MANIFEST, manifest)
            manifest_hash = hashlib.sha256((self.staging / MANIFEST).read_bytes()).hexdigest()
            (self.staging / MANIFEST_SHA).write_text(manifest_hash + "\n", encoding="ascii")
            (self.staging / COMMITTED).write_text(manifest_hash + "\n", encoding="ascii")
            if self.destination.exists() or self.destination.is_symlink():
                raise FileExistsError(f"checkpoint destination appeared: {self.destination}")
            os.rename(self.staging, self.destination)
            return manifest

        manifest = _rank0(dist, publish, "async checkpoint publication")
        dist.barrier()
        self._manifest = manifest
        return manifest

    def close(self) -> dict[str, Any]:
        """Blocking shutdown barrier for an in-flight checkpoint."""

        return self.wait()

    def __enter__(self) -> AsyncScaleCheckpointHandle:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def begin_async_scale_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any,
    plan: ParallelPlan,
    identity: ScaleCheckpointIdentity,
    metadata: dict[str, Any] | None = None,
    rank_state: dict[str, Any] | None = None,
) -> AsyncScaleCheckpointHandle:
    """Begin one standard-thread ``dcp.async_save`` without publishing it as good.

    The call includes DCP's default training-safe staging and therefore defines the
    foreground checkpoint stall measured by CHECKPOINT-145. It deliberately does
    not use ``use_collectives=False`` or a custom storage format.
    """

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    dist = _dist(plan)
    rank = dist.get_rank()
    identity.validate()
    metadata_value = _json_safe(dict(metadata or {}), "metadata")
    rank_value = None if rank_state is None else _json_safe(dict(rank_state), "rank_state")
    identity_hash = identity.sha256
    metadata_hash = _hash(metadata_value)
    _agree(dist, identity_hash, "identity")
    _agree(dist, metadata_hash, "metadata")
    destination = Path(checkpoint_dir)

    def create_staging() -> str:
        parent = destination.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("checkpoint parent must pre-exist as real directory")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint destination exists: {destination}")
        staging = parent / f".{destination.name}.dcp-staging-{uuid.uuid4().hex}"
        staging.mkdir()
        return str(staging)

    staging = Path(_rank0(dist, create_staging, "async staging creation"))
    dist.barrier()
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    schema_hash = state_schema_sha256(state)
    _agree(dist, schema_hash, "state schema")

    flags: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(flags, rank_value is not None)
    if len(set(flags)) != 1:
        raise ValueError("rank_state must be supplied by every rank or no rank")
    rank_state_present = bool(flags[0])
    if rank_state_present:
        if rank == 0:
            (staging / "rank-state").mkdir()
        dist.barrier()
        _write_json(staging / "rank-state" / f"rank-{rank:05d}.json", rank_value)
        dist.barrier()

    # Public maintained DCP API. The default THREAD checkpointer preserves the
    # incumbent DCP payload format and stages a training-safe copy before return.
    future = dcp.async_save(state_dict=state, checkpoint_id=staging)

    return AsyncScaleCheckpointHandle(
        future=future,
        staging=staging,
        destination=destination,
        plan=plan,
        identity=identity,
        identity_hash=identity_hash,
        schema_hash=schema_hash,
        metadata_value=metadata_value,
        metadata_hash=metadata_hash,
        rank_state_present=rank_state_present,
    )


def logical_rank_identity(plan: ParallelPlan, rank: int) -> dict[str, int]:
    """Expose the same logical-rank identity in benchmark evidence."""

    return rank_identity(plan, rank)
