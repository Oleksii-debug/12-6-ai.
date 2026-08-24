"""Topology-aware checkpoint identity contracts layered above D05 checkpoint v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping

from .rank_layout import RankLayout

_HEX = frozenset("0123456789abcdef")
_SCHEMA = "12-6.distributed-checkpoint-envelope.v1"


def _require_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise ValueError(f"{field} must be exact lowercase 64-hex SHA-256")
    return value


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class D05CheckpointRef:
    checkpoint_id: str
    manifest_sha256: str
    identity_sha256: str
    git_sha: str
    model_spec_hash: str
    run_manifest_hash: str
    environment_lock_hash: str | None
    step: int
    tokens_seen: int

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        manifest_sha256: str,
    ) -> D05CheckpointRef:
        if manifest.get("format") != "12-6-checkpoint" or manifest.get("format_version") != 1:
            raise ValueError("expected a verified D05 12-6-checkpoint v1 manifest")
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("D05 manifest identity must be a mapping")
        checkpoint_id = _require_sha256(str(manifest.get("checkpoint_id")), "checkpoint_id")
        manifest_hash = _require_sha256(manifest_sha256, "manifest_sha256")
        identity_hash = _hash_json(identity)
        model_spec_hash = _require_sha256(str(identity.get("model_spec_hash")), "model_spec_hash")
        run_manifest_hash = _require_sha256(
            str(identity.get("run_manifest_hash")),
            "run_manifest_hash",
        )
        lock_hash = identity.get("environment_lock_hash")
        if lock_hash is not None:
            lock_hash = _require_sha256(str(lock_hash), "environment_lock_hash")
        git_sha = identity.get("git_sha")
        if (
            not isinstance(git_sha, str)
            or len(git_sha) not in {40, 64}
            or git_sha != git_sha.lower()
            or any(ch not in _HEX for ch in git_sha)
        ):
            raise ValueError("git_sha must be an exact lowercase 40/64-hex Git object ID")
        step = identity.get("step")
        tokens_seen = identity.get("tokens_seen")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
            raise ValueError("tokens_seen must be a non-negative integer")
        return cls(
            checkpoint_id=checkpoint_id,
            manifest_sha256=manifest_hash,
            identity_sha256=identity_hash,
            git_sha=git_sha,
            model_spec_hash=model_spec_hash,
            run_manifest_hash=run_manifest_hash,
            environment_lock_hash=lock_hash,
            step=step,
            tokens_seen=tokens_seen,
        )

    @property
    def semantic_parent_sha256(self) -> str:
        """D05 lineage identity excluding topology-dependent distributed shard bytes."""

        return self.identity_sha256


@dataclass(frozen=True, slots=True)
class DistributedShardRecord:
    relative_path: str
    sha256: str
    size_bytes: int
    writer_rank: int

    def validate(self, *, world_size: int) -> None:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or not self.relative_path or ".." in path.parts:
            raise ValueError("relative_path must stay inside the checkpoint directory")
        _require_sha256(self.sha256, "shard.sha256")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("shard.size_bytes must be a non-negative integer")
        if (
            not isinstance(self.writer_rank, int)
            or isinstance(self.writer_rank, bool)
            or not 0 <= self.writer_rank < world_size
        ):
            raise ValueError("shard.writer_rank is outside the save world")


@dataclass(frozen=True, slots=True)
class DistributedCheckpointEnvelope:
    d05_parent: D05CheckpointRef
    save_layout_sha256: str
    save_world_size: int
    state_dict_schema_sha256: str
    shards: tuple[DistributedShardRecord, ...]
    rank_rng_sha256: tuple[str, ...]
    backend: str = "torch.distributed.checkpoint"
    schema: str = _SCHEMA

    def validate(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError(f"unsupported distributed checkpoint schema: {self.schema!r}")
        _require_sha256(self.save_layout_sha256, "save_layout_sha256")
        _require_sha256(self.state_dict_schema_sha256, "state_dict_schema_sha256")
        if not isinstance(self.save_world_size, int) or isinstance(self.save_world_size, bool):
            raise ValueError("save_world_size must be an integer")
        if self.save_world_size < 1:
            raise ValueError("save_world_size must be >= 1")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be non-empty")
        if not self.shards:
            raise ValueError("distributed checkpoint must record at least one physical shard")
        if len(self.rank_rng_sha256) != self.save_world_size:
            raise ValueError("rank_rng_sha256 must contain one digest per logical save rank")
        for digest in self.rank_rng_sha256:
            _require_sha256(digest, "rank_rng_sha256")
        paths: set[str] = set()
        for shard in self.shards:
            shard.validate(world_size=self.save_world_size)
            if shard.relative_path in paths:
                raise ValueError(f"duplicate shard path: {shard.relative_path}")
            paths.add(shard.relative_path)

    @property
    def artifact_set_sha256(self) -> str:
        """Topology-dependent physical identity of sorted shard bytes."""

        self.validate()
        payload = [
            {
                "relative_path": shard.relative_path,
                "sha256": shard.sha256,
                "size_bytes": shard.size_bytes,
                "writer_rank": shard.writer_rank,
            }
            for shard in sorted(self.shards, key=lambda item: item.relative_path)
        ]
        return _hash_json(payload)

    @property
    def envelope_sha256(self) -> str:
        self.validate()
        payload = {
            "schema": self.schema,
            "backend": self.backend,
            "d05_parent": asdict(self.d05_parent),
            "save_layout_sha256": self.save_layout_sha256,
            "save_world_size": self.save_world_size,
            "state_dict_schema_sha256": self.state_dict_schema_sha256,
            "artifact_set_sha256": self.artifact_set_sha256,
            "rank_rng_sha256": list(self.rank_rng_sha256),
        }
        return _hash_json(payload)


class ResumeMode(str, Enum):
    EXACT_TOPOLOGY = "exact_topology"
    RESHARD = "reshard"


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    mode: ResumeMode
    allowed: bool
    exact_trajectory_claim_allowed: bool
    rng_policy: str
    reasons: tuple[str, ...]


def decide_resume(
    envelope: DistributedCheckpointEnvelope,
    target_layout: RankLayout,
    *,
    mode: ResumeMode,
    target_state_dict_schema_sha256: str,
) -> ResumeDecision:
    """Separate physical reshardability from exact rank-local RNG trajectory identity."""

    envelope.validate()
    schema_hash = _require_sha256(
        target_state_dict_schema_sha256,
        "target_state_dict_schema_sha256",
    )
    if schema_hash != envelope.state_dict_schema_sha256:
        return ResumeDecision(
            mode=mode,
            allowed=False,
            exact_trajectory_claim_allowed=False,
            rng_policy="blocked",
            reasons=("canonical FQN/state-dict schema differs from the saved checkpoint",),
        )
    same_layout = (
        target_layout.identity_sha256 == envelope.save_layout_sha256
        and target_layout.plan.world_size == envelope.save_world_size
    )
    if mode is ResumeMode.EXACT_TOPOLOGY:
        if not same_layout:
            return ResumeDecision(
                mode=mode,
                allowed=False,
                exact_trajectory_claim_allowed=False,
                rng_policy="blocked",
                reasons=("exact topology resume requires identical logical rank layout",),
            )
        return ResumeDecision(
            mode=mode,
            allowed=True,
            exact_trajectory_claim_allowed=True,
            rng_policy="restore-rank-local-rng-by-logical-rank",
            reasons=("topology and canonical state-dict schema are unchanged",),
        )
    if mode is not ResumeMode.RESHARD:
        raise ValueError(f"unsupported resume mode: {mode!r}")
    return ResumeDecision(
        mode=mode,
        allowed=True,
        exact_trajectory_claim_allowed=same_layout,
        rng_policy=(
            "restore-rank-local-rng-by-logical-rank"
            if same_layout
            else "reseed-new-rank-streams-from-global-seed-step; no-bitwise-trajectory-claim"
        ),
        reasons=(
            "DCP may load into a different target sharding when canonical state keys are stable",
            "changed rank cardinality has no one-to-one mapping for saved rank-local RNG streams",
        ),
    )
