"""Deterministic split-safe sequence construction and packing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Iterator, Mapping, Sequence

from twelve_six.tokenization import TokenizerProtocol


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    text: str
    split: str

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be str")


@dataclass(frozen=True)
class PackedCausalExample:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]
    split: str
    record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if not size:
            raise ValueError("packed example may not be empty")
        if not (
            len(self.target_ids)
            == len(self.attention_mask)
            == len(self.loss_mask)
            == size
        ):
            raise ValueError("all packed fields must have equal length")


class SplitMixError(ValueError):
    """Raised when records from another split enter a split-specific iterator."""


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def iter_packed_examples(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    *,
    expected_split: str,
    sequence_length: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Iterator[PackedCausalExample]:
    """Tokenize and pack one explicit split without silently dropping LM pairs.

    The stream advances by ``sequence_length`` tokens while retaining the final
    target token as the next block's first input token. Therefore every adjacent
    token pair in the concatenated stream appears exactly once as a training
    pair. The final partial block is padded and masked rather than dropped.

    A single terminal token with no successor is naturally not a causal pair.
    With ``add_eos=True`` this is the final EOS marker.
    """
    if not expected_split:
        raise ValueError("expected_split must be non-empty")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    token_carry: list[int] = []
    provenance_carry: list[str] = []

    def emit(window_tokens: list[int], window_sources: list[str]) -> PackedCausalExample:
        pair_count = len(window_tokens) - 1
        if not 1 <= pair_count <= sequence_length:
            raise AssertionError("invalid pair count")

        inputs = window_tokens[:-1]
        targets = window_tokens[1:]
        pad_count = sequence_length - pair_count
        input_ids = tuple(inputs + [tokenizer.pad_id] * pad_count)
        target_ids = tuple(targets + [tokenizer.pad_id] * pad_count)
        mask = tuple([1] * pair_count + [0] * pad_count)
        return PackedCausalExample(
            input_ids=input_ids,
            target_ids=target_ids,
            attention_mask=mask,
            loss_mask=mask,
            split=expected_split,
            record_ids=_ordered_unique(window_sources),
        )

    for record in records:
        if record.split != expected_split:
            raise SplitMixError(
                f"record {record.record_id!r} has split {record.split!r}; "
                f"expected {expected_split!r}"
            )
        encoded = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
        token_carry.extend(encoded)
        provenance_carry.extend([record.record_id] * len(encoded))

        while len(token_carry) >= sequence_length + 1:
            window_tokens = token_carry[: sequence_length + 1]
            window_sources = provenance_carry[: sequence_length + 1]
            yield emit(window_tokens, window_sources)
            token_carry = token_carry[sequence_length:]
            provenance_carry = provenance_carry[sequence_length:]

    if len(token_carry) >= 2:
        yield emit(token_carry, provenance_carry)


def batch_examples(
    examples: Iterable[PackedCausalExample],
    *,
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[tuple[PackedCausalExample, ...]]:
    """Group examples deterministically; default behavior never drops a tail batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[PackedCausalExample] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch and not drop_last:
        yield tuple(batch)


def deterministic_shard(
    records: Iterable[TextRecord],
    *,
    shard_index: int,
    num_shards: int,
) -> Iterator[TextRecord]:
    """Assign stable ordered records by index modulo shard count."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    for index, record in enumerate(records):
        if index % num_shards == shard_index:
            yield record


class DeterministicMixtureSampler:
    """Platform-stable weighted source choice based on SHA-256, not global RNG state."""

    def __init__(self, weights: Mapping[str, float], *, seed: int = 0) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        normalized_items: list[tuple[str, float]] = []
        total = 0.0
        for name, weight in sorted(weights.items()):
            if not name:
                raise ValueError("mixture source names must be non-empty")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("mixture weights must be finite and positive")
            normalized_items.append((name, float(weight)))
            total += float(weight)
        self._items = tuple((name, weight / total) for name, weight in normalized_items)
        self.seed = int(seed)

    @property
    def normalized_weights(self) -> Mapping[str, float]:
        return dict(self._items)

    def source_for_step(self, step: int) -> str:
        if step < 0:
            raise ValueError("step must be non-negative")
        digest = hashlib.sha256(f"{self.seed}:{step}".encode("ascii")).digest()
        numerator = int.from_bytes(digest[:8], "big")
        unit = numerator / (1 << 64)
        cumulative = 0.0
        for name, weight in self._items:
            cumulative += weight
            if unit < cumulative:
                return name
        return self._items[-1][0]
