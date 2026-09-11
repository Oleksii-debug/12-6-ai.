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
from .loss_materialization import (
    MATERIALIZATION_SCHEMA,
    LossMaterializationDocument,
    LossMaterializationError,
    build_postpack_loss_materialization,
    tokenizer_identity_sha256,
)
from .manifest import PackedSplitManifest, measure_packed_split

__all__ = [
    "DEFAULT_FILL_TOKEN_ID",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_SEQUENCE_LENGTH",
    "MATERIALIZATION_SCHEMA",
    "PACKING_CONFIG_HASH",
    "PACKING_VERSION",
    "DeterministicMixtureSampler",
    "JsonlRecordError",
    "LossMaterializationDocument",
    "LossMaterializationError",
    "PackedCausalExample",
    "PackedSplitManifest",
    "SplitMixError",
    "TextRecord",
    "batch_examples",
    "build_postpack_loss_materialization",
    "canonical_packing_config_json",
    "collate_rows",
    "deterministic_shard",
    "iter_packed_examples",
    "load_jsonl_records",
    "measure_packed_split",
    "packing_config_hash",
    "records_from_jsonl_lines",
    "tokenizer_identity_sha256",
]
