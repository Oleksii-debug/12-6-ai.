"""D04-owned deterministic sequence construction primitives."""

from .packing import (
    DeterministicMixtureSampler,
    PackedCausalExample,
    SplitMixError,
    TextRecord,
    batch_examples,
    deterministic_shard,
    iter_packed_examples,
)

__all__ = [
    "DeterministicMixtureSampler",
    "PackedCausalExample",
    "SplitMixError",
    "TextRecord",
    "batch_examples",
    "deterministic_shard",
    "iter_packed_examples",
]
