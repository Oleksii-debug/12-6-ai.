"""Deterministic split-safe sequence construction and packing."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from twelve_six.tokenization import TokenizerProtocol

PACKING_VERSION = "s0-byte-pack-v1"
PACKING_CONFIG_HASH = "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_FILL_TOKEN_ID = 0
DEFAULT_IGNORE_INDEX = -100


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
    """One fixed-length causal block compatible with D02's shifted-label loss."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]
    split: str
    record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if size < 2:
            raise ValueError("packed example sequence length must be at least two")
        if not (
            len(self.labels)
            == len(self.attention_mask)
            == len(self.loss_mask)
            == size
        ):
            raise ValueError("all packed fields must have equal length")

    @property
    def num_loss_tokens(self) -> int:
        return sum(self.loss_mask)


class SplitMixError(ValueError):
    """Raised when records from another split enter a split-specific iterator."""


def _packing_config_dict() -> dict[str, object]:
    from twelve_six.tokenization import BYTE_TOKENIZER_HASH

    return {
        "schema_version": 1,
        "packing_version": PACKING_VERSION,
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "document_boundary_policy": "isolate",
        "cross_document_packing": False,
        "add_bos": False,
        "add_eos": False,
        "masked_fill_token_id": DEFAULT_FILL_TOKEN_ID,
        "masked_fill_is_semantic_special": False,
        "label_ignore_index": DEFAULT_IGNORE_INDEX,
        "window_overlap_tokens": 1,
    }


def canonical_packing_config_json() -> str:
    import json

    return json.dumps(
        _packing_config_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def packing_config_hash() -> str:
    return hashlib.sha256(canonical_packing_config_json().encode("utf-8")).hexdigest()


def _validate_packing_identity() -> None:
    if packing_config_hash() != PACKING_CONFIG_HASH:
        raise RuntimeError("S0 packing config drifted without a version/hash update")


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _emit_window(
    window_tokens: list[int],
    window_sources: list[str],
    *,
    split: str,
    sequence_length: int,
    fill_token_id: int,
    ignore_index: int,
) -> PackedCausalExample:
    actual = len(window_tokens)
    if not 2 <= actual <= sequence_length:
        raise AssertionError("invalid window size")

    pad_count = sequence_length - actual
    input_ids = tuple(window_tokens + [fill_token_id] * pad_count)
    labels = tuple(window_tokens + [ignore_index] * pad_count)
    attention_mask = tuple([1] * actual + [0] * pad_count)

    # D02 shifts labels by one: logits[:, t] predicts labels[:, t + 1].
    # Therefore m actual tokens create exactly m-1 valid loss positions.
    loss_mask = tuple([1] * (actual - 1) + [0] * (sequence_length - actual + 1))

    return PackedCausalExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        split=split,
        record_ids=_ordered_unique(window_sources),
    )


def _iter_token_stream_blocks(
    token_ids: list[int],
    source_ids: list[str],
    *,
    split: str,
    sequence_length: int,
    fill_token_id: int,
    ignore_index: int,
) -> Iterator[PackedCausalExample]:
    while len(token_ids) >= sequence_length:
        window_tokens = token_ids[:sequence_length]
        window_sources = source_ids[:sequence_length]
        yield _emit_window(
            window_tokens,
            window_sources,
            split=split,
            sequence_length=sequence_length,
            fill_token_id=fill_token_id,
            ignore_index=ignore_index,
        )
        # One-token overlap preserves the boundary pair for D02's shifted loss.
        token_ids = token_ids[sequence_length - 1 :]
        source_ids = source_ids[sequence_length - 1 :]

    if len(token_ids) >= 2:
        yield _emit_window(
            token_ids,
            source_ids,
            split=split,
            sequence_length=sequence_length,
            fill_token_id=fill_token_id,
            ignore_index=ignore_index,
        )


def iter_packed_examples(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    *,
    expected_split: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    fill_token_id: int = DEFAULT_FILL_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    add_bos: bool = False,
    add_eos: bool = False,
    cross_document: bool = False,
) -> Iterator[PackedCausalExample]:
    """Tokenize and pack one explicit split with no dropped within-document LM pairs.

    S0 defaults to isolated documents because the 256-ID raw-byte tokenizer has
    no semantic EOS token. Cross-document packing is permitted only when an EOS
    token exists and is explicitly added, avoiding invented boundary semantics.

    Blocks overlap by one actual token. With D02's shifted-label objective this
    makes every adjacent token pair within each document appear exactly once.
    Final partial blocks are masked rather than silently dropped.
    """
    _validate_packing_identity()
    if not expected_split:
        raise ValueError("expected_split must be non-empty")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if not 0 <= fill_token_id < tokenizer.vocab_size:
        raise ValueError("fill_token_id must be inside tokenizer vocabulary")
    if 0 <= ignore_index < tokenizer.vocab_size:
        raise ValueError("ignore_index must be outside tokenizer vocabulary")

    if cross_document and (tokenizer.eos_id is None or not add_eos):
        raise ValueError("cross-document packing requires an explicit EOS token")

    if cross_document:
        token_carry: list[int] = []
        provenance_carry: list[str] = []
        for record in records:
            if record.split != expected_split:
                raise SplitMixError(
                    f"record {record.record_id!r} has split {record.split!r}; "
                    f"expected {expected_split!r}"
                )
            encoded = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
            token_carry.extend(encoded)
            provenance_carry.extend([record.record_id] * len(encoded))

        yield from _iter_token_stream_blocks(
            token_carry,
            provenance_carry,
            split=expected_split,
            sequence_length=sequence_length,
            fill_token_id=fill_token_id,
            ignore_index=ignore_index,
        )
        return

    for record in records:
        if record.split != expected_split:
            raise SplitMixError(
                f"record {record.record_id!r} has split {record.split!r}; "
                f"expected {expected_split!r}"
            )
        encoded = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
        source_ids = [record.record_id] * len(encoded)
        yield from _iter_token_stream_blocks(
            encoded,
            source_ids,
            split=expected_split,
            sequence_length=sequence_length,
            fill_token_id=fill_token_id,
            ignore_index=ignore_index,
        )


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


def _aligned_target_ids(example: PackedCausalExample) -> tuple[int, ...]:
    return tuple(
        example.labels[index + 1] if keep else DEFAULT_IGNORE_INDEX
        for index, keep in enumerate(example.loss_mask)
    )


def collate_rows(
    examples: Sequence[PackedCausalExample],
    *,
    target_mode: str = "labels",
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Return tensor-ready rows for either D02-supported causal target convention.

    ``labels`` emits raw/unshifted labels and deliberately omits ``loss_mask``;
    D02 performs the one-token shift and uses ``-100`` tail labels as ignore
    positions. ``target_ids`` emits already-aligned next-token targets plus the
    binary ``loss_mask`` expected by D02's direct causal-pair loss.
    """
    if not examples:
        raise ValueError("examples must not be empty")
    split = examples[0].split
    if any(example.split != split for example in examples):
        raise SplitMixError("cannot collate examples from different splits")
    size = len(examples[0].input_ids)
    if any(len(example.input_ids) != size for example in examples):
        raise ValueError("all examples in a batch must have the same sequence length")

    rows = {
        "input_ids": tuple(example.input_ids for example in examples),
        "attention_mask": tuple(example.attention_mask for example in examples),
    }
    if target_mode == "labels":
        rows["labels"] = tuple(example.labels for example in examples)
        return rows
    if target_mode == "target_ids":
        rows["target_ids"] = tuple(_aligned_target_ids(example) for example in examples)
        rows["loss_mask"] = tuple(example.loss_mask for example in examples)
        return rows
    raise ValueError("target_mode must be 'labels' or 'target_ids'")


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
