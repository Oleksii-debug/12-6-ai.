"""Context-length engineering utilities for 12-6 AI.

These helpers separate mechanical context capacity from trained/evaluated context
capability. They intentionally expose both linear KV-cache cost and a dense
materialized-attention score equivalent. The latter is a planning diagnostic,
not a claim that a fused SDPA kernel actually allocates an S x S score tensor.

Canonical S0 packing identity remains owned by ``twelve_six.packing``. Future
context candidates use the explicitly versioned contract in this module rather
than weakening S0's fail-closed 128-token manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from twelve_six.model import ModelSpec
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import TokenizerProtocol


@dataclass(frozen=True, slots=True)
class ContextCostEstimate:
    """Transparent context-dependent cost terms for one ModelSpec.

    ``attention_score_equivalent_bytes`` is the byte size of one dense
    [batch, layer, query-head, sequence, sequence] score tensor equivalent.
    Flash/memory-efficient SDPA can avoid materializing that tensor in HBM, so
    callers must not treat this diagnostic as measured allocator peak memory.
    It exists to make the quadratic attention term impossible to hide in scale
    planning.
    """

    sequence_length: int
    batch_size: int
    activation_element_bytes: int
    kv_element_bytes: int
    attention_score_elements: int
    attention_score_equivalent_bytes: int
    kv_cache_elements: int
    kv_cache_bytes: int
    rotary_fraction: float


@dataclass(frozen=True, slots=True)
class IsolatedPackingEstimate:
    """Capacity/utilization for the current isolated-document packing policy."""

    sequence_length: int
    document_count: int
    emitted_blocks: int
    unique_next_token_pairs: int
    pair_capacity: int
    pair_utilization: float


@dataclass(frozen=True, slots=True)
class ContextPackingSpec:
    """Identity-bearing isolated-document packing semantics for future stages.

    This is intentionally a new version namespace. It does not redefine
    ``s0-byte-pack-v1`` and cannot be mistaken for the canonical S0 manifest.
    Until a tokenizer with an explicit semantic EOS is selected, context
    candidates remain document-isolated rather than inventing cross-document
    boundaries.
    """

    schema_version: int = 1
    packing_version: str = "context-candidate-isolated-v1"
    sequence_length: int = 256
    document_boundary_policy: str = "isolate"
    cross_document_packing: bool = False
    add_bos: bool = False
    add_eos: bool = False
    masked_fill_token_id: int = 0
    label_ignore_index: int = -100
    window_overlap_tokens: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported ContextPackingSpec schema_version")
        if self.packing_version != "context-candidate-isolated-v1":
            raise ValueError("unsupported context packing_version")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.document_boundary_policy != "isolate":
            raise ValueError("ContextPackingSpec v1 supports document isolation only")
        if self.cross_document_packing:
            raise ValueError("ContextPackingSpec v1 does not permit cross-document packing")
        if self.add_bos or self.add_eos:
            raise ValueError("ContextPackingSpec v1 does not inject BOS/EOS semantics")
        if self.window_overlap_tokens != 1:
            raise ValueError("ContextPackingSpec v1 requires one-token overlap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def identity_sha256(self, *, tokenizer_config_sha256: str) -> str:
        _require_sha256(tokenizer_config_sha256, "tokenizer_config_sha256")
        payload = {
            **self.to_dict(),
            "tokenizer_config_sha256": tokenizer_config_sha256,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextPackingMeasurement:
    """Measured future-stage packing evidence bound to data/tokenizer identities."""

    schema_version: int
    dataset_id: str
    dataset_identity_sha256: str
    source_jsonl_sha256: str
    split: str
    tokenizer_version: str
    tokenizer_config_sha256: str
    tokenizer_vocab_sha256: str
    vocab_size: int
    context_packing_spec_sha256: str
    sequence_length: int
    document_count: int
    token_count: int
    causal_loss_token_count: int
    packed_example_count: int
    packed_input_token_count: int
    packed_capacity_token_count: int
    causal_pair_capacity: int
    causal_pair_utilization: float
    token_length_min: int
    token_length_p50: int
    token_length_p90: int
    token_length_p95: int
    token_length_max: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _nearest_rank(lengths: tuple[int, ...], percentile: int) -> int:
    if not lengths:
        raise ValueError("cannot compute percentile of empty lengths")
    if not 1 <= percentile <= 100:
        raise ValueError("percentile must be in [1, 100]")
    ordered = sorted(lengths)
    rank = max(1, math.ceil(percentile * len(ordered) / 100))
    return ordered[rank - 1]


def estimate_context_cost(
    spec: ModelSpec,
    *,
    sequence_length: int | None = None,
    batch_size: int = 1,
    activation_element_bytes: int = 2,
    kv_element_bytes: int | None = None,
    enforce_model_limit: bool = True,
) -> ContextCostEstimate:
    """Estimate context-dependent dense-attention and model-native KV costs.

    The KV formula matches the incumbent model-native cache representation:
    unexpanded K/V tensors with shape
    [batch, n_kv_heads, sequence, head_dim] for every decoder layer.

    ``sequence_length`` may exceed ``spec.max_seq_len`` only when
    ``enforce_model_limit=False``. That mode is for hypothetical planning and
    does not imply that the checkpoint was trained or evaluated at the larger
    context.
    """

    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    resolved_sequence = spec.max_seq_len if sequence_length is None else sequence_length
    _positive_int("sequence_length", resolved_sequence)
    _positive_int("batch_size", batch_size)
    _positive_int("activation_element_bytes", activation_element_bytes)
    resolved_kv_bytes = activation_element_bytes if kv_element_bytes is None else kv_element_bytes
    _positive_int("kv_element_bytes", resolved_kv_bytes)
    if enforce_model_limit and resolved_sequence > spec.max_seq_len:
        raise ValueError(
            f"sequence_length {resolved_sequence} exceeds ModelSpec max_seq_len "
            f"{spec.max_seq_len}"
        )

    score_elements = (
        batch_size
        * spec.n_layers
        * spec.n_heads
        * resolved_sequence
        * resolved_sequence
    )
    kv_elements = (
        2
        * batch_size
        * spec.n_layers
        * spec.n_kv_heads
        * resolved_sequence
        * spec.head_dim
    )
    return ContextCostEstimate(
        sequence_length=resolved_sequence,
        batch_size=batch_size,
        activation_element_bytes=activation_element_bytes,
        kv_element_bytes=resolved_kv_bytes,
        attention_score_elements=score_elements,
        attention_score_equivalent_bytes=score_elements * activation_element_bytes,
        kv_cache_elements=kv_elements,
        kv_cache_bytes=kv_elements * resolved_kv_bytes,
        rotary_fraction=spec.rope_rotary_dim / spec.head_dim,
    )


def isolated_document_packing_estimate(
    token_lengths: Iterable[int],
    *,
    sequence_length: int,
    window_overlap_tokens: int = 1,
) -> IsolatedPackingEstimate:
    """Estimate utilization without changing S0's packing identity.

    The current causal packer isolates documents and overlaps adjacent windows
    by one token so every within-document next-token pair appears exactly once.
    This helper generalizes that arithmetic to candidate context lengths; it
    does not mutate the canonical 128-token S0 packing configuration.
    """

    _positive_int("sequence_length", sequence_length)
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if not isinstance(window_overlap_tokens, int) or isinstance(window_overlap_tokens, bool):
        raise ValueError("window_overlap_tokens must be an integer")
    if not 0 <= window_overlap_tokens < sequence_length:
        raise ValueError("window_overlap_tokens must satisfy 0 <= overlap < sequence_length")

    stride = sequence_length - window_overlap_tokens
    if stride <= 0:
        raise ValueError("packing stride must be positive")

    lengths = tuple(token_lengths)
    emitted_blocks = 0
    unique_pairs = 0
    for length in lengths:
        _positive_int("token length", length)
        if length < 2:
            continue
        unique_pairs += length - 1
        emitted_blocks += math.ceil((length - window_overlap_tokens) / stride)

    pair_capacity_per_block = sequence_length - 1
    pair_capacity = emitted_blocks * pair_capacity_per_block
    utilization = unique_pairs / pair_capacity if pair_capacity else 0.0
    return IsolatedPackingEstimate(
        sequence_length=sequence_length,
        document_count=len(lengths),
        emitted_blocks=emitted_blocks,
        unique_next_token_pairs=unique_pairs,
        pair_capacity=pair_capacity,
        pair_utilization=utilization,
    )


def measure_context_candidate_packing(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    *,
    packing_spec: ContextPackingSpec,
    dataset_id: str,
    dataset_identity_sha256: str,
    source_jsonl_sha256: str,
    split: str,
) -> ContextPackingMeasurement:
    """Measure candidate packing without changing canonical S0 packing identity."""

    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if not split:
        raise ValueError("split must be non-empty")
    _require_sha256(dataset_identity_sha256, "dataset_identity_sha256")
    _require_sha256(source_jsonl_sha256, "source_jsonl_sha256")

    identity = tokenizer.identity
    _require_sha256(identity.config_sha256, "tokenizer_config_sha256")
    _require_sha256(identity.vocab_sha256, "tokenizer_vocab_sha256")
    if not 0 <= packing_spec.masked_fill_token_id < tokenizer.vocab_size:
        raise ValueError("masked_fill_token_id must be inside tokenizer vocabulary")
    if 0 <= packing_spec.label_ignore_index < tokenizer.vocab_size:
        raise ValueError("label_ignore_index must be outside tokenizer vocabulary")

    ordered_records = tuple(records)
    if not ordered_records:
        raise ValueError("context candidate packing requires at least one document")
    if any(record.split != split for record in ordered_records):
        raise ValueError("context candidate packing cannot mix dataset splits")

    token_lengths = tuple(len(tokenizer.encode(record.text)) for record in ordered_records)
    token_count = sum(token_lengths)
    packed_example_count = 0
    packed_input_token_count = 0
    causal_loss_token_count = 0

    for record in ordered_records:
        for example in iter_packed_examples(
            [record],
            tokenizer,
            expected_split=split,
            sequence_length=packing_spec.sequence_length,
            fill_token_id=packing_spec.masked_fill_token_id,
            ignore_index=packing_spec.label_ignore_index,
            add_bos=packing_spec.add_bos,
            add_eos=packing_spec.add_eos,
            cross_document=packing_spec.cross_document_packing,
        ):
            packed_example_count += 1
            packed_input_token_count += sum(example.attention_mask)
            causal_loss_token_count += example.num_loss_tokens

    packed_capacity_token_count = packed_example_count * packing_spec.sequence_length
    causal_pair_capacity = packed_example_count * (packing_spec.sequence_length - 1)
    pair_utilization = (
        causal_loss_token_count / causal_pair_capacity if causal_pair_capacity else 0.0
    )
    return ContextPackingMeasurement(
        schema_version=1,
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity_sha256,
        source_jsonl_sha256=source_jsonl_sha256,
        split=split,
        tokenizer_version=identity.version,
        tokenizer_config_sha256=identity.config_sha256,
        tokenizer_vocab_sha256=identity.vocab_sha256,
        vocab_size=identity.vocab_size,
        context_packing_spec_sha256=packing_spec.identity_sha256(
            tokenizer_config_sha256=identity.config_sha256
        ),
        sequence_length=packing_spec.sequence_length,
        document_count=len(ordered_records),
        token_count=token_count,
        causal_loss_token_count=causal_loss_token_count,
        packed_example_count=packed_example_count,
        packed_input_token_count=packed_input_token_count,
        packed_capacity_token_count=packed_capacity_token_count,
        causal_pair_capacity=causal_pair_capacity,
        causal_pair_utilization=pair_utilization,
        token_length_min=min(token_lengths),
        token_length_p50=_nearest_rank(token_lengths, 50),
        token_length_p90=_nearest_rank(token_lengths, 90),
        token_length_p95=_nearest_rank(token_lengths, 95),
        token_length_max=max(token_lengths),
    )


def context_probe_spec(spec: ModelSpec, *, max_seq_len: int) -> ModelSpec:
    """Return an identity-distinct ModelSpec for a mechanical context probe.

    This makes the status boundary explicit: changing ``max_seq_len`` changes
    ModelSpec identity and therefore cannot silently redefine an existing S0 or
    stage checkpoint.
    """

    _positive_int("max_seq_len", max_seq_len)
    return replace(spec, max_seq_len=max_seq_len)
