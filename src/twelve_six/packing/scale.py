"""Deterministic S1-S4 mixture, sharding, accounting and restart contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .core import DEFAULT_IGNORE_INDEX, PackedCausalExample

SCALE_PLAN_SCHEMA = "12-6.packing-scale-plan.v1"
RESTART_SCHEMA = "12-6.packing-restart-cursor.v1"


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class MixtureComponent:
    name: str
    weight_units: int
    dataset_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mixture component name must be non-empty")
        if self.weight_units <= 0:
            raise ValueError("weight_units must be positive")
        _require_sha256(self.dataset_manifest_sha256, field="dataset_manifest_sha256")


@dataclass(frozen=True)
class IntegerMixturePlan:
    """Platform-stable weighted source choice with no floating-point thresholds."""

    components: tuple[MixtureComponent, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("components must not be empty")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("mixture component names must be unique")
        object.__setattr__(self, "components", tuple(sorted(self.components, key=lambda item: item.name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCALE_PLAN_SCHEMA,
            "seed": self.seed,
            "components": [
                {
                    "name": component.name,
                    "weight_units": component.weight_units,
                    "dataset_manifest_sha256": component.dataset_manifest_sha256,
                }
                for component in self.components
            ],
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def source_for_sample(self, sample_index: int) -> str:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        total = sum(component.weight_units for component in self.components)
        digest = hashlib.sha256(f"{self.identity_sha256}:{sample_index}".encode("ascii")).digest()
        slot = int.from_bytes(digest[:16], "big") % total
        cumulative = 0
        for component in self.components:
            cumulative += component.weight_units
            if slot < cumulative:
                return component.name
        raise AssertionError("integer mixture selection fell outside cumulative weights")


@dataclass(frozen=True)
class DeterministicShardPlan:
    dataset_manifest_sha256: str
    split: str
    num_shards: int
    assignment_salt: str = "12-6-shard-v1"

    def __post_init__(self) -> None:
        _require_sha256(self.dataset_manifest_sha256, field="dataset_manifest_sha256")
        if not self.split:
            raise ValueError("split must be non-empty")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if not self.assignment_salt:
            raise ValueError("assignment_salt must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCALE_PLAN_SCHEMA,
            "kind": "content-addressed-record-sharding",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split": self.split,
            "num_shards": self.num_shards,
            "assignment_salt": self.assignment_salt,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def shard_for_record(self, record_id: str) -> int:
        if not record_id:
            raise ValueError("record_id must be non-empty")
        digest = hashlib.sha256(
            (
                f"{self.assignment_salt}:{self.dataset_manifest_sha256}:"
                f"{self.split}:{record_id}"
            ).encode()
        ).digest()
        return int.from_bytes(digest[:16], "big") % self.num_shards


@dataclass(frozen=True)
class PackingRestartCursor:
    """Fail-closed restart identity for deterministic stream reconstruction."""

    mixture_plan_sha256: str
    dataset_manifest_sha256: str
    tokenizer_vocab_sha256: str
    packing_config_sha256: str
    split: str
    global_sample_index: int
    shard_epoch: int
    shard_index: int
    document_index: int
    token_offset: int
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        for field, value in (
            ("mixture_plan_sha256", self.mixture_plan_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("tokenizer_vocab_sha256", self.tokenizer_vocab_sha256),
            ("packing_config_sha256", self.packing_config_sha256),
        ):
            _require_sha256(value, field=field)
        if not self.split:
            raise ValueError("split must be non-empty")
        for field in (
            "global_sample_index",
            "shard_epoch",
            "shard_index",
            "document_index",
            "token_offset",
            "rank",
        ):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RESTART_SCHEMA,
            "mixture_plan_sha256": self.mixture_plan_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "tokenizer_vocab_sha256": self.tokenizer_vocab_sha256,
            "packing_config_sha256": self.packing_config_sha256,
            "split": self.split,
            "global_sample_index": self.global_sample_index,
            "shard_epoch": self.shard_epoch,
            "shard_index": self.shard_index,
            "document_index": self.document_index,
            "token_offset": self.token_offset,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def require_compatible(
        self,
        *,
        mixture_plan_sha256: str,
        dataset_manifest_sha256: str,
        tokenizer_vocab_sha256: str,
        packing_config_sha256: str,
        split: str,
        rank: int,
        world_size: int,
    ) -> None:
        expected = {
            "mixture_plan_sha256": mixture_plan_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
            "packing_config_sha256": packing_config_sha256,
            "split": split,
            "rank": rank,
            "world_size": world_size,
        }
        mismatches = [
            f"{field}={getattr(self, field)!r} expected {value!r}"
            for field, value in expected.items()
            if getattr(self, field) != value
        ]
        if mismatches:
            raise ValueError("restart identity mismatch: " + "; ".join(mismatches))


@dataclass(frozen=True)
class PackedAccounting:
    examples: int
    attended_tokens: int
    loss_tokens: int
    masked_positions: int

    def to_dict(self) -> dict[str, int]:
        return {
            "examples": self.examples,
            "attended_tokens": self.attended_tokens,
            "loss_tokens": self.loss_tokens,
            "masked_positions": self.masked_positions,
        }


def audit_packed_examples(
    examples: Iterable[PackedCausalExample],
    *,
    vocab_size: int,
    require_single_document: bool = True,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> PackedAccounting:
    """Validate masking/target alignment and optional no-cross-document invariant."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    example_count = 0
    attended = 0
    loss_tokens = 0
    masked = 0
    for example in examples:
        example_count += 1
        if require_single_document and len(example.record_ids) != 1:
            raise ValueError("cross-document leakage detected in document-isolated packing")
        if any(token_id < 0 or token_id >= vocab_size for token_id in example.input_ids):
            raise ValueError("input token ID lies outside tokenizer vocabulary")
        if any(value not in (0, 1) for value in example.attention_mask):
            raise ValueError("attention_mask must be binary")
        if any(value not in (0, 1) for value in example.loss_mask):
            raise ValueError("loss_mask must be binary")
        if example.loss_mask[-1] != 0:
            raise ValueError("last sequence position cannot own a shifted causal target")
        for index, keep in enumerate(example.loss_mask[:-1]):
            if keep:
                if not example.attention_mask[index] or not example.attention_mask[index + 1]:
                    raise ValueError("loss_mask exposes a masked input/target position")
                if example.labels[index + 1] == ignore_index:
                    raise ValueError("loss_mask exposes an ignored shifted target")
        for index, attention in enumerate(example.attention_mask):
            if not attention:
                masked += 1
                if example.labels[index] != ignore_index:
                    raise ValueError("masked filler position must use ignore_index label")
                if example.loss_mask[index]:
                    raise ValueError("masked filler position must not contribute loss")
        attended += sum(example.attention_mask)
        loss_tokens += sum(example.loss_mask)
    if example_count == 0:
        raise ValueError("examples must not be empty")
    return PackedAccounting(
        examples=example_count,
        attended_tokens=attended,
        loss_tokens=loss_tokens,
        masked_positions=masked,
    )
