"""Scale-ready deterministic mixture, sharding and restart contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

MIXTURE_PLAN_SCHEMA = "12-6.mixture-plan.v1"
RESTART_CURSOR_SCHEMA = "12-6.mixture-restart-cursor.v1"
SHARDING_VERSION = "record-id-sha256-v1"
MIXTURE_SELECTION_VERSION = "sha256-integer-units-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MixtureContractError(ValueError):
    """Raised when scale packing/mixture/restart identity fails closed."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if _SHA256_RE.fullmatch(value) is None:
        raise MixtureContractError(f"{field} must be a lowercase SHA-256 hex digest")


def _bounded_hash(key: str, upper_bound: int) -> int:
    """Map a stable text key to [0, upper_bound) without modulo bias."""
    if upper_bound <= 0:
        raise MixtureContractError("upper_bound must be positive")
    space = 1 << 256
    limit = space - (space % upper_bound)
    counter = 0
    while True:
        digest = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        value = int.from_bytes(digest, "big")
        if value < limit:
            return value % upper_bound
        counter += 1


@dataclass(frozen=True)
class MixtureSource:
    name: str
    manifest_sha256: str
    weight_units: int

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise MixtureContractError("source name must be non-empty and whitespace-free")
        _require_sha256(self.manifest_sha256, field="source manifest_sha256")
        if self.weight_units <= 0:
            raise MixtureContractError("weight_units must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "manifest_sha256": self.manifest_sha256,
            "weight_units": self.weight_units,
        }


@dataclass(frozen=True)
class MixturePlan:
    """Content-addressed S1-S4 source-selection and sharding contract."""

    plan_id: str
    tokenizer_config_sha256: str
    tokenizer_vocab_sha256: str
    packing_config_sha256: str
    sources: tuple[MixtureSource, ...]
    seed: int
    num_shards: int
    shard_seed: int = 0

    def __post_init__(self) -> None:
        if not self.plan_id or any(char.isspace() for char in self.plan_id):
            raise MixtureContractError("plan_id must be non-empty and whitespace-free")
        for field_name in (
            "tokenizer_config_sha256",
            "tokenizer_vocab_sha256",
            "packing_config_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if not self.sources:
            raise MixtureContractError("sources must not be empty")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise MixtureContractError("source names must be unique")
        if self.num_shards <= 0:
            raise MixtureContractError("num_shards must be positive")
        if not isinstance(self.seed, int) or not isinstance(self.shard_seed, int):
            raise TypeError("seed and shard_seed must be integers")

    @property
    def ordered_sources(self) -> tuple[MixtureSource, ...]:
        return tuple(sorted(self.sources, key=lambda source: source.name))

    @property
    def total_weight_units(self) -> int:
        return sum(source.weight_units for source in self.ordered_sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MIXTURE_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "tokenizer_vocab_sha256": self.tokenizer_vocab_sha256,
            "packing_config_sha256": self.packing_config_sha256,
            "sources": [source.to_dict() for source in self.ordered_sources],
            "seed": self.seed,
            "selection_version": MIXTURE_SELECTION_VERSION,
            "num_shards": self.num_shards,
            "shard_seed": self.shard_seed,
            "sharding_version": SHARDING_VERSION,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict()))

    def source_for_sample(self, sample_index: int) -> str:
        if sample_index < 0:
            raise MixtureContractError("sample_index must be non-negative")
        ticket = _bounded_hash(
            f"{self.sha256}:{self.seed}:sample:{sample_index}",
            self.total_weight_units,
        )
        cumulative = 0
        for source in self.ordered_sources:
            cumulative += source.weight_units
            if ticket < cumulative:
                return source.name
        raise AssertionError("weighted source selection escaped cumulative range")

    def shard_for_record(self, record_id: str) -> int:
        if not record_id:
            raise MixtureContractError("record_id must be non-empty")
        return _bounded_hash(
            f"{SHARDING_VERSION}:{self.shard_seed}:{record_id}",
            self.num_shards,
        )


@dataclass(frozen=True)
class RestartCursor:
    """Exact restart point bound to one immutable MixturePlan."""

    plan_sha256: str
    next_sample_index: int
    source_offsets: tuple[tuple[str, int], ...]
    emitted_sequences: int
    emitted_loss_tokens: int

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, field="plan_sha256")
        if self.next_sample_index < 0:
            raise MixtureContractError("next_sample_index must be non-negative")
        names = [name for name, _ in self.source_offsets]
        if len(names) != len(set(names)):
            raise MixtureContractError("source_offsets names must be unique")
        if any(not name for name in names):
            raise MixtureContractError("source_offsets names must be non-empty")
        if any(offset < 0 for _, offset in self.source_offsets):
            raise MixtureContractError("source offsets must be non-negative")
        if self.emitted_sequences < 0 or self.emitted_loss_tokens < 0:
            raise MixtureContractError("emitted counters must be non-negative")

    @classmethod
    def initial(cls, plan: MixturePlan) -> RestartCursor:
        return cls(
            plan_sha256=plan.sha256,
            next_sample_index=0,
            source_offsets=tuple((source.name, 0) for source in plan.ordered_sources),
            emitted_sequences=0,
            emitted_loss_tokens=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RESTART_CURSOR_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "next_sample_index": self.next_sample_index,
            "source_offsets": dict(self.source_offsets),
            "emitted_sequences": self.emitted_sequences,
            "emitted_loss_tokens": self.emitted_loss_tokens,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict()))

    def require_compatible(self, plan: MixturePlan) -> None:
        if self.plan_sha256 != plan.sha256:
            raise MixtureContractError("restart cursor belongs to a different mixture plan")
        expected_names = tuple(source.name for source in plan.ordered_sources)
        actual_names = tuple(name for name, _ in self.source_offsets)
        if actual_names != expected_names:
            raise MixtureContractError("restart cursor source set/order does not match plan")

    def next_source_and_offset(self, plan: MixturePlan) -> tuple[str, int]:
        self.require_compatible(plan)
        source = plan.source_for_sample(self.next_sample_index)
        offsets = dict(self.source_offsets)
        return source, offsets[source]

    def advance(
        self,
        plan: MixturePlan,
        *,
        source_name: str,
        emitted_sequences: int,
        emitted_loss_tokens: int,
    ) -> RestartCursor:
        self.require_compatible(plan)
        expected_source = plan.source_for_sample(self.next_sample_index)
        if source_name != expected_source:
            raise MixtureContractError(
                f"source {source_name!r} does not match deterministic schedule {expected_source!r}"
            )
        if emitted_sequences <= 0:
            raise MixtureContractError("advance must emit at least one sequence")
        if emitted_loss_tokens < 0:
            raise MixtureContractError("emitted_loss_tokens must be non-negative")
        offsets = dict(self.source_offsets)
        offsets[source_name] += 1
        return RestartCursor(
            plan_sha256=self.plan_sha256,
            next_sample_index=self.next_sample_index + 1,
            source_offsets=tuple(sorted(offsets.items())),
            emitted_sequences=self.emitted_sequences + emitted_sequences,
            emitted_loss_tokens=self.emitted_loss_tokens + emitted_loss_tokens,
        )
