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
from .manifest import PackedSplitManifest, measure_packed_split
from .performance import (
    PackedSourceSpan,
    collate_right_trimmed_rows,
    document_window_spans,
    right_trim_width,
    valid_causal_pairs,
)

__all__ = [
    "DEFAULT_FILL_TOKEN_ID",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_SEQUENCE_LENGTH",
    "PACKING_CONFIG_HASH",
    "PACKING_VERSION",
    "DeterministicMixtureSampler",
    "JsonlRecordError",
    "PackedCausalExample",
    "PackedSourceSpan",
    "PackedSplitManifest",
    "SplitMixError",
    "TextRecord",
    "batch_examples",
    "canonical_packing_config_json",
    "collate_right_trimmed_rows",
    "collate_rows",
    "deterministic_shard",
    "document_window_spans",
    "iter_packed_examples",
    "load_jsonl_records",
    "measure_packed_split",
    "packing_config_hash",
    "records_from_jsonl_lines",
    "right_trim_width",
    "valid_causal_pairs",
]
