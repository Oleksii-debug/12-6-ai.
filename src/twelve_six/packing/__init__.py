"""D04-owned tokenizer-adjacent packing and dataloader primitives."""

from .core import (
    DEFAULT_FILL_TOKEN_ID,
    DEFAULT_IGNORE_INDEX,
    DEFAULT_SEQUENCE_LENGTH,
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    DeterministicMixtureSampler,
    PackedCausalExample,
    SplitMixError,
    TextRecord,
    batch_examples,
    canonical_packing_config_json,
    collate_rows,
    deterministic_shard,
    iter_packed_examples,
    packing_config_hash,
)
from .jsonl import JsonlRecordError, load_jsonl_records, records_from_jsonl_lines

__all__ = [
    "DEFAULT_FILL_TOKEN_ID",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_SEQUENCE_LENGTH",
    "PACKING_CONFIG_HASH",
    "PACKING_VERSION",
    "DeterministicMixtureSampler",
    "JsonlRecordError",
    "PackedCausalExample",
    "SplitMixError",
    "TextRecord",
    "batch_examples",
    "canonical_packing_config_json",
    "collate_rows",
    "deterministic_shard",
    "iter_packed_examples",
    "load_jsonl_records",
    "packing_config_hash",
    "records_from_jsonl_lines",
]
