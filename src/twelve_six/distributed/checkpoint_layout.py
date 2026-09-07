"""Distributed checkpoint layout identity layered over D05 logical checkpoint identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import ParallelPlan
from .mesh import coordinate_for_rank

FORMAT_NAME = "12-6-distributed-checkpoint-layout"
FORMAT_VERSION = 1
D05_FORMAT_NAME = "12-6-checkpoint"
D05_FORMAT_VERSION = 1
_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_exact_hex(value: Any, *, field: str, lengths: set[int]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        expected = "/".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{field} must be exact lowercase {expected}-hex")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    return _require_exact_hex(value, field=field, lengths={64})


@dataclass(frozen=True, slots=True)
class D05LogicalBinding:
    source_checkpoint_id: str
    identity_sha256: str
    step: int
    tokens_seen: int
    environment_lock_hash: str | None


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    data_parallel: int
    tensor_parallel: int
    pipeline_parallel: int
    context_parallel: int
    expert_parallel: int
    world_size: int
    rank_order: str = "tp-cp-dp-pp|ep-in-dp"

    @classmethod
    def from_plan(cls, plan: ParallelPlan) -> TopologySnapshot:
        plan.validate()
        return cls(
            data_parallel=plan.data_parallel,
            tensor_parallel=plan.tensor_parallel,
            pipeline_parallel=plan.pipeline_parallel,
            context_parallel=plan.context_parallel,
            expert_parallel=plan.expert_parallel,
            world_size=plan.world_size,
        )

    def to_plan(self) -> ParallelPlan:
        plan = ParallelPlan(
            data_parallel=self.data_parallel,
            tensor_parallel=self.tensor_parallel,
            pipeline_parallel=self.pipeline_parallel,
            context_parallel=self.context_parallel,
            expert_parallel=self.expert_parallel,
        )
        plan.validate()
        if plan.world_size != self.world_size:
            raise ValueError("saved_topology world_size does not match its parallel degrees")
        if self.rank_order != "tp-cp-dp-pp|ep-in-dp":
            raise ValueError("unsupported saved_topology rank_order")
        return plan


@dataclass(frozen=True, slots=True)
class ShardRecord:
    relative_path: str
    writer_rank: int
    rank_identity: str
    sha256: str
    byte_count: int

    def validate(self, *, world_size: int) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or self.relative_path in {"", "."}:
            raise ValueError("relative_path must be a safe relative artifact path")
        if not 0 <= self.writer_rank < world_size:
            raise ValueError("writer_rank is outside the saved topology")
        if not self.rank_identity:
            raise ValueError("rank_identity must be non-empty")
        _require_sha256(self.sha256, field="shard.sha256")
        if self.byte_count < 0:
            raise ValueError("byte_count must be non-negative")


@dataclass(frozen=True, slots=True)
class DistributedCheckpointManifest:
    format: str
    format_version: int
    d05: D05LogicalBinding
    saved_topology: TopologySnapshot
    backend_format: str
    reshardable: bool
    optimizer_reshardable: bool
    shards: tuple[ShardRecord, ...]
    artifact_set_sha256: str
    layout_id: str


@dataclass(frozen=True, slots=True)
class ResumePlan:
    mode: str
    source_world_size: int
    target_world_size: int
    preserves_d05_identity_sha256: str


def bind_d05_manifest(manifest: Mapping[str, Any]) -> D05LogicalBinding:
    """Validate the D05 manifest identity boundary without importing D05 runtime code."""

    if manifest.get("format") != D05_FORMAT_NAME:
        raise ValueError("unsupported D05 checkpoint format")
    if manifest.get("format_version") != D05_FORMAT_VERSION:
        raise ValueError("unsupported D05 checkpoint format version")
    checkpoint_id = _require_sha256(manifest.get("checkpoint_id"), field="checkpoint_id")
    identity = manifest.get("identity")
    files = manifest.get("files")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("D05 identity must be a non-empty mapping")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("D05 files must be a non-empty mapping")
    _require_exact_hex(identity.get("git_sha"), field="identity.git_sha", lengths={40, 64})
    for field in (
        "model_spec_hash",
        "tokenizer_hash",
        "tokenizer_vocab_hash",
        "dataset_manifest_hash",
        "run_manifest_hash",
        "training_config_hash",
        "optimizer_hash",
        "scheduler_hash",
        "environment_hash",
    ):
        _require_sha256(identity.get(field), field=f"identity.{field}")
    expected_checkpoint_id = _hash_json({"identity": identity, "files": files})
    if checkpoint_id != expected_checkpoint_id:
        raise ValueError("D05 checkpoint_id does not match identity and file records")
    step = identity.get("step")
    tokens_seen = identity.get("tokens_seen")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("D05 identity step must be a non-negative integer")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
        raise ValueError("D05 identity tokens_seen must be a non-negative integer")
    lock_hash = identity.get("environment_lock_hash")
    if lock_hash is not None:
        _require_sha256(lock_hash, field="identity.environment_lock_hash")
    return D05LogicalBinding(
        source_checkpoint_id=checkpoint_id,
        identity_sha256=_hash_json(identity),
        step=step,
        tokens_seen=tokens_seen,
        environment_lock_hash=lock_hash,
    )


def rank_identity(plan: ParallelPlan, global_rank: int) -> str:
    coordinate = coordinate_for_rank(global_rank, plan)
    return (
        f"pp={coordinate.pipeline_parallel_rank}/"
        f"dp={coordinate.data_parallel_rank}/"
        f"cp={coordinate.context_parallel_rank}/"
        f"tp={coordinate.tensor_parallel_rank}/"
        f"ep={coordinate.expert_parallel_rank}/"
        f"edp={coordinate.expert_data_parallel_rank}"
    )


def _artifact_payload(shards: Sequence[ShardRecord]) -> list[dict[str, Any]]:
    return [
        asdict(shard)
        for shard in sorted(shards, key=lambda item: (item.relative_path, item.writer_rank))
    ]


def build_distributed_checkpoint_manifest(
    *,
    d05: D05LogicalBinding,
    saved_plan: ParallelPlan,
    backend_format: str,
    shards: Sequence[ShardRecord],
    reshardable: bool,
    optimizer_reshardable: bool,
) -> DistributedCheckpointManifest:
    """Build a topology-specific layout identity while preserving D05 logical identity."""

    saved_plan.validate()
    _require_sha256(d05.source_checkpoint_id, field="d05.source_checkpoint_id")
    _require_sha256(d05.identity_sha256, field="d05.identity_sha256")
    if d05.environment_lock_hash is not None:
        _require_sha256(d05.environment_lock_hash, field="d05.environment_lock_hash")
    if not backend_format.strip():
        raise ValueError("backend_format must be non-empty")
    if not shards:
        raise ValueError("at least one shard artifact is required")
    for shard in shards:
        shard.validate(world_size=saved_plan.world_size)
        expected_rank_identity = rank_identity(saved_plan, shard.writer_rank)
        if shard.rank_identity != expected_rank_identity:
            raise ValueError("shard rank_identity does not match writer_rank and saved topology")
    paths = [shard.relative_path for shard in shards]
    if len(paths) != len(set(paths)):
        raise ValueError("shard relative_path values must be unique")

    topology = TopologySnapshot.from_plan(saved_plan)
    artifact_set_sha256 = _hash_json(_artifact_payload(shards))
    layout_payload = {
        "d05_identity_sha256": d05.identity_sha256,
        "saved_topology": asdict(topology),
        "backend_format": backend_format,
        "reshardable": reshardable,
        "optimizer_reshardable": optimizer_reshardable,
        "artifact_set_sha256": artifact_set_sha256,
    }
    layout_id = _hash_json(layout_payload)
    return DistributedCheckpointManifest(
        format=FORMAT_NAME,
        format_version=FORMAT_VERSION,
        d05=d05,
        saved_topology=topology,
        backend_format=backend_format,
        reshardable=reshardable,
        optimizer_reshardable=optimizer_reshardable,
        shards=tuple(shards),
        artifact_set_sha256=artifact_set_sha256,
        layout_id=layout_id,
    )


def verify_distributed_checkpoint_manifest(manifest: DistributedCheckpointManifest) -> None:
    """Verify pure manifest algebra; artifact bytes are verified separately."""

    if manifest.format != FORMAT_NAME or manifest.format_version != FORMAT_VERSION:
        raise ValueError("unsupported distributed checkpoint layout")
    _require_sha256(manifest.d05.source_checkpoint_id, field="d05.source_checkpoint_id")
    _require_sha256(manifest.d05.identity_sha256, field="d05.identity_sha256")
    if manifest.d05.environment_lock_hash is not None:
        _require_sha256(
            manifest.d05.environment_lock_hash,
            field="d05.environment_lock_hash",
        )
    saved_plan = manifest.saved_topology.to_plan()
    paths: list[str] = []
    for shard in manifest.shards:
        shard.validate(world_size=manifest.saved_topology.world_size)
        if shard.rank_identity != rank_identity(saved_plan, shard.writer_rank):
            raise ValueError("shard rank_identity does not match writer_rank and saved topology")
        paths.append(shard.relative_path)
    if len(paths) != len(set(paths)):
        raise ValueError("shard relative_path values must be unique")
    expected_artifacts = _hash_json(_artifact_payload(manifest.shards))
    if expected_artifacts != manifest.artifact_set_sha256:
        raise ValueError("artifact_set_sha256 mismatch")
    layout_payload = {
        "d05_identity_sha256": manifest.d05.identity_sha256,
        "saved_topology": asdict(manifest.saved_topology),
        "backend_format": manifest.backend_format,
        "reshardable": manifest.reshardable,
        "optimizer_reshardable": manifest.optimizer_reshardable,
        "artifact_set_sha256": manifest.artifact_set_sha256,
    }
    if _hash_json(layout_payload) != manifest.layout_id:
        raise ValueError("layout_id mismatch")


def verify_shard_files(manifest: DistributedCheckpointManifest, root: str | Path) -> None:
    """Verify exact sharded artifact bytes against the topology-specific manifest."""

    verify_distributed_checkpoint_manifest(manifest)
    directory = Path(root)
    for shard in manifest.shards:
        path = directory / shard.relative_path
        if not path.is_file():
            raise ValueError(f"missing shard artifact: {shard.relative_path}")
        if path.stat().st_size != shard.byte_count:
            raise ValueError(f"shard size mismatch: {shard.relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard.sha256:
            raise ValueError(f"shard checksum mismatch: {shard.relative_path}")


def plan_resume(
    manifest: DistributedCheckpointManifest,
    target_plan: ParallelPlan,
    *,
    require_optimizer_state: bool = True,
) -> ResumePlan:
    """Choose direct or backend-reshard resume without changing logical training identity."""

    verify_distributed_checkpoint_manifest(manifest)
    target_plan.validate()
    saved = manifest.saved_topology
    target = TopologySnapshot.from_plan(target_plan)
    if saved == target:
        mode = "direct"
    else:
        if not manifest.reshardable:
            raise ValueError("checkpoint layout is not reshardable to a different topology")
        if require_optimizer_state and not manifest.optimizer_reshardable:
            raise ValueError("optimizer state is not reshardable for this checkpoint layout")
        mode = "backend_reshard"
    return ResumePlan(
        mode=mode,
        source_world_size=saved.world_size,
        target_world_size=target.world_size,
        preserves_d05_identity_sha256=manifest.d05.identity_sha256,
    )
