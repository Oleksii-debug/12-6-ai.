"""Scale checkpoint successor backed by PyTorch Distributed Checkpoint (DCP).

D05 checkpoint-v1 remains unchanged. Large tensor payloads stay in DCP's sharded data plane;
this module adds only a small identity/integrity/publication control plane. Torch imports are lazy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from .checkpoint_layout import rank_identity
from .contracts import ParallelPlan

SCHEMA = "12-6.distributed-dcp-checkpoint.v1"
BACKEND = "torch.distributed.checkpoint"
MANIFEST = "scale-manifest.json"
MANIFEST_SHA = "scale-manifest.sha256"
COMMITTED = "COMMITTED"
_CONTROL = frozenset({MANIFEST, MANIFEST_SHA, COMMITTED})
_HEX = frozenset("0123456789abcdef")
_CHUNK = 8 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise ValueError(f"{field} must be exact lowercase SHA-256")
    return value


def _git_sha(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise ValueError("git_sha must be exact lowercase 40/64-hex")
    return value


def _json_safe(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains non-finite float")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{field} keys must be non-empty strings")
            out[key] = _json_safe(item, f"{field}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, f"{field}[]") for item in value]
    raise TypeError(f"{field} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ScaleCheckpointIdentity:
    git_sha: str
    model_spec_sha256: str
    init_spec_sha256: str
    tokenizer_config_sha256: str
    tokenizer_vocab_sha256: str
    data_manifest_sha256: str
    packing_sha256: str
    training_config_sha256: str
    environment_lock_sha256: str | None
    seed: int
    step: int
    tokens_seen: int

    def validate(self) -> None:
        _git_sha(self.git_sha)
        for name in (
            "model_spec_sha256",
            "init_spec_sha256",
            "tokenizer_config_sha256",
            "tokenizer_vocab_sha256",
            "data_manifest_sha256",
            "packing_sha256",
            "training_config_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.environment_lock_sha256 is not None:
            _sha(self.environment_lock_sha256, "environment_lock_sha256")
        for name in ("seed", "step", "tokens_seen"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def sha256(self) -> str:
        self.validate()
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    sha256: str
    size_bytes: int

    def validate(self) -> None:
        path = PurePosixPath(self.path)
        if path.is_absolute() or not self.path or ".." in path.parts or self.path in _CONTROL:
            raise ValueError("artifact path must be a safe payload-relative path")
        _sha(self.sha256, "artifact.sha256")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be non-negative integer")


class ResumeMode(str, Enum):
    EXACT_TOPOLOGY = "exact_topology"
    RESHARD = "reshard"


@dataclass(frozen=True, slots=True)
class LoadResult:
    mode: ResumeMode
    saved_world_size: int
    target_world_size: int
    exact_topology: bool
    exact_trajectory_claim_allowed: bool
    rng_policy: str
    aggregate_checkpoint_sha256: str
    identity_sha256: str
    metadata: dict[str, Any]
    rank_state: dict[str, Any] | None


def _topology(plan: ParallelPlan) -> dict[str, Any]:
    plan.validate()
    return {
        "parallel_plan": asdict(plan),
        "world_size": plan.world_size,
        "logical_ranks": [rank_identity(plan, rank) for rank in range(plan.world_size)],
    }


def topology_sha256(plan: ParallelPlan) -> str:
    return _hash(_topology(plan))


def _schema(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "kind": "tensor",
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        entries = [
            (type(key).__name__, repr(key), _schema(item)) for key, item in value.items()
        ]
        entries.sort(key=lambda item: (item[0], item[1]))
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, list):
        return {"kind": "list", "items": [_schema(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_schema(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": type(value).__name__}
    raise TypeError(f"unsupported checkpoint state type {type(value).__name__}")


def state_schema_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash logical state keys/global tensor shape+dtypes, not physical shard ownership."""
    return _hash(_schema(state_dict))


def _file_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _artifacts(root: Path) -> tuple[ArtifactRecord, ...]:
    records: list[ArtifactRecord] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint payload must not be symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in _CONTROL:
            continue
        digest, size = _file_hash(path)
        record = ArtifactRecord(relative, digest, size)
        record.validate()
        records.append(record)
    if not records:
        raise ValueError("checkpoint contains no DCP payload artifacts")
    return tuple(records)


def _artifact_set(records: Sequence[ArtifactRecord]) -> str:
    return _hash([asdict(item) for item in sorted(records, key=lambda item: item.path)])


def _dist(plan: ParallelPlan):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("scale checkpoint requires initialized torch.distributed")
    plan.validate()
    if dist.get_world_size() != plan.world_size:
        raise ValueError("process-group world size does not match ParallelPlan")
    return dist


def _broadcast(dist: Any, value: Any) -> Any:
    payload = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _rank0(dist: Any, operation: Callable[[], Any], phase: str) -> Any:
    if dist.get_rank() == 0:
        try:
            status = {"ok": True, "value": operation()}
        except Exception as exc:
            status = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
    else:
        status = None
    status = _broadcast(dist, status)
    if not isinstance(status, Mapping) or status.get("ok") is not True:
        if not isinstance(status, Mapping):
            raise RuntimeError(f"{phase} failed: invalid rank-0 status")
        raise RuntimeError(f"{phase} failed on rank 0: {status.get('type')}: {status.get('error')}")
    return status.get("value")


def _agree(dist: Any, value: str, field: str) -> None:
    values: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != value for item in values):
        raise RuntimeError(f"ranks disagree on {field}")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _aggregate(
    identity: str, state_schema: str, topology: str, artifacts: str, metadata: str
) -> str:
    return _hash(
        {
            "schema": SCHEMA,
            "backend": BACKEND,
            "identity_sha256": identity,
            "state_schema_sha256": state_schema,
            "topology_sha256": topology,
            "artifact_set_sha256": artifacts,
            "metadata_sha256": metadata,
        }
    )


def _manifest_identity(raw: Mapping[str, Any]) -> ScaleCheckpointIdentity:
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("manifest identity must be mapping")
    result = ScaleCheckpointIdentity(**dict(identity))
    result.validate()
    return result


def verify_scale_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Stream-verify one committed local-filesystem generation without reading shards whole."""
    root = Path(checkpoint_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be real directory")
    for name in _CONTROL:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError(f"missing checkpoint control file: {name}")
    manifest_bytes = (root / MANIFEST).read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    declared = (root / MANIFEST_SHA).read_text(encoding="ascii").strip()
    committed = (root / COMMITTED).read_text(encoding="ascii").strip()
    if manifest_hash != declared or manifest_hash != committed:
        raise ValueError("manifest/commit digest mismatch")
    raw = json.loads(manifest_bytes)
    if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA or raw.get("backend") != BACKEND:
        raise ValueError("unsupported scale checkpoint manifest")
    identity = _manifest_identity(raw)
    if identity.sha256 != raw.get("identity_sha256"):
        raise ValueError("identity_sha256 mismatch")
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be mapping")
    metadata = _json_safe(metadata, "metadata")
    if _hash(metadata) != _sha(raw.get("metadata_sha256"), "metadata_sha256"):
        raise ValueError("metadata_sha256 mismatch")
    saved_topology = raw.get("save_topology")
    if not isinstance(saved_topology, Mapping):
        raise TypeError("save_topology must be mapping")
    topology_hash = _sha(raw.get("topology_sha256"), "topology_sha256")
    if _hash(saved_topology) != topology_hash:
        raise ValueError("topology_sha256 mismatch")
    artifact_rows = raw.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise TypeError("artifacts must be list")
    records = tuple(ArtifactRecord(**item) for item in artifact_rows)
    for record in records:
        record.validate()
    paths = [item.path for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact path")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in _CONTROL
    }
    if actual != set(paths):
        raise ValueError("payload inventory differs from manifest")
    for record in records:
        path = root / record.path
        digest, size = _file_hash(path)
        if path.is_symlink() or digest != record.sha256 or size != record.size_bytes:
            raise ValueError(f"artifact integrity mismatch: {record.path}")
    artifact_hash = _artifact_set(records)
    if artifact_hash != _sha(raw.get("artifact_set_sha256"), "artifact_set_sha256"):
        raise ValueError("artifact_set_sha256 mismatch")
    state_schema = _sha(raw.get("state_schema_sha256"), "state_schema_sha256")
    aggregate = _aggregate(
        identity.sha256, state_schema, topology_hash, artifact_hash, _hash(metadata)
    )
    if aggregate != _sha(raw.get("aggregate_checkpoint_sha256"), "aggregate_checkpoint_sha256"):
        raise ValueError("aggregate checkpoint identity mismatch")
    if not isinstance(raw.get("rank_state_present"), bool):
        raise TypeError("rank_state_present must be bool")
    return dict(raw)


def save_scale_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any,
    plan: ParallelPlan,
    identity: ScaleCheckpointIdentity,
    metadata: Mapping[str, Any] | None = None,
    rank_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save DCP shards then publish one immutable, committed local-FS generation.

    Atomicity boundary: cooperative single writer, same-parent rename, no fsync/power-loss/NFS/
    object-store durability claim. Hard rank loss may block until process-group timeout; a save that
    fails before publication leaves only an uncommitted staging generation for later
    garbage collection.
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

    staging = Path(_rank0(dist, create_staging, "staging creation"))
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

    dcp.save(state_dict=state, checkpoint_id=staging)
    dist.barrier()

    def publish() -> dict[str, Any]:
        records = _artifacts(staging)
        artifacts_hash = _artifact_set(records)
        saved_topology = _topology(plan)
        topology_hash = _hash(saved_topology)
        aggregate = _aggregate(
            identity_hash, schema_hash, topology_hash, artifacts_hash, metadata_hash
        )
        manifest = {
            "schema": SCHEMA,
            "backend": BACKEND,
            "identity": asdict(identity),
            "identity_sha256": identity_hash,
            "state_schema_sha256": schema_hash,
            "save_topology": saved_topology,
            "topology_sha256": topology_hash,
            "artifacts": [asdict(item) for item in records],
            "artifact_set_sha256": artifacts_hash,
            "metadata": metadata_value,
            "metadata_sha256": metadata_hash,
            "rank_state_present": rank_state_present,
            "aggregate_checkpoint_sha256": aggregate,
        }
        _write_json(staging / MANIFEST, manifest)
        manifest_hash = hashlib.sha256((staging / MANIFEST).read_bytes()).hexdigest()
        (staging / MANIFEST_SHA).write_text(manifest_hash + "\n", encoding="ascii")
        (staging / COMMITTED).write_text(manifest_hash + "\n", encoding="ascii")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint destination appeared: {destination}")
        os.rename(staging, destination)
        return manifest

    manifest = _rank0(dist, publish, "checkpoint publication")
    dist.barrier()
    return manifest


def load_scale_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    model: Any,
    optimizer: Any,
    target_plan: ParallelPlan,
    mode: ResumeMode = ResumeMode.EXACT_TOPOLOGY,
    expected_identity_sha256: str | None = None,
    restore_rank_state: Callable[[Mapping[str, Any]], None] | None = None,
) -> LoadResult:
    """Load model+optimizer with DCP; topology change is an explicit reshard semantic."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    dist = _dist(target_plan)
    rank = dist.get_rank()
    mode = ResumeMode(mode)
    raw = _rank0(
        dist,
        lambda: verify_scale_checkpoint(checkpoint_dir),
        "checkpoint verification",
    )
    identity = _manifest_identity(raw)
    if expected_identity_sha256 is not None:
        _sha(expected_identity_sha256, "expected_identity_sha256")
        if identity.sha256 != expected_identity_sha256:
            raise ValueError("checkpoint semantic identity does not match expected identity")
    exact_topology = _hash(_topology(target_plan)) == raw["topology_sha256"]
    if mode is ResumeMode.EXACT_TOPOLOGY and not exact_topology:
        raise ValueError("exact-topology resume requires identical topology identity")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    schema_hash = state_schema_sha256(state)
    _agree(dist, schema_hash, "target state schema")
    if schema_hash != raw["state_schema_sha256"]:
        raise ValueError("canonical model/optimizer state schema differs from checkpoint")
    dcp.load(state_dict=state, checkpoint_id=Path(checkpoint_dir))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    rank_value: dict[str, Any] | None = None
    if exact_topology and raw["rank_state_present"]:
        rank_path = Path(checkpoint_dir) / "rank-state" / f"rank-{rank:05d}.json"
        rank_raw = json.loads(rank_path.read_text(encoding="utf-8"))
        if not isinstance(rank_raw, Mapping):
            raise TypeError("rank-state sidecar must decode to mapping")
        rank_value = dict(rank_raw)
    if rank_value is not None and restore_rank_state is not None:
        restore_rank_state(rank_value)
        trajectory = True
        rng_policy = "caller-restored-rank-state-by-logical-rank"
    elif exact_topology and rank_value is not None:
        trajectory = False
        rng_policy = "rank-state-returned-but-not-restored-by-checkpoint-layer"
    elif exact_topology:
        trajectory = False
        rng_policy = "model-optimizer-exact; rank-local-rng/sampler-state-not-captured"
    else:
        trajectory = False
        rng_policy = "reseed-new-rank-streams-from-global-seed-step; no-bitwise-trajectory-claim"
    return LoadResult(
        mode=mode,
        saved_world_size=int(raw["save_topology"]["world_size"]),
        target_world_size=target_plan.world_size,
        exact_topology=exact_topology,
        exact_trajectory_claim_allowed=trajectory,
        rng_policy=rng_policy,
        aggregate_checkpoint_sha256=raw["aggregate_checkpoint_sha256"],
        identity_sha256=identity.sha256,
        metadata=dict(raw["metadata"]),
        rank_state=rank_value,
    )
