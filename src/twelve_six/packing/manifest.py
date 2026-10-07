"""Exact tokenizer/packing manifests bound to an upstream D03 dataset identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from twelve_six.tokenization import TokenizerProtocol

from .core import (
    DEFAULT_SEQUENCE_LENGTH,
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    SplitMixError,
    TextRecord,
    iter_packed_examples,
)


def _require_nonempty(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be non-empty")


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class PackedSplitManifest:
    schema_version: int
    dataset_id: str
    dataset_identity_sha256: str
    split: str
    source_jsonl_sha256: str
    tokenizer_version: str
    tokenizer_config_sha256: str
    tokenizer_vocab_sha256: str
    vocab_size: int
    packing_version: str
    packing_config_sha256: str
    sequence_length: int
    document_count: int
    codepoint_count: int
    utf8_byte_count: int
    token_count: int
    causal_loss_token_count: int
    packed_example_count: int
    packed_input_token_count: int
    packed_capacity_token_count: int
    masked_fill_position_count: int
    documents_without_causal_pair: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def fertility_ratio(self) -> tuple[int, int]:
        """Return exact tokenizer tokens / Unicode code points without float identity drift."""
        return self.token_count, self.codepoint_count

    @property
    def packed_input_utilization_ratio(self) -> tuple[int, int]:
        return self.packed_input_token_count, self.packed_capacity_token_count


def measure_packed_split(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    *,
    dataset_id: str,
    dataset_identity_sha256: str,
    source_jsonl_sha256: str,
    split: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> PackedSplitManifest:
    """Measure one ordered split and bind it to canonical S0 packing identity."""
    _require_nonempty(dataset_id, "dataset_id")
    _require_nonempty(split, "split")
    _require_sha256(dataset_identity_sha256, "dataset_identity_sha256")
    _require_sha256(source_jsonl_sha256, "source_jsonl_sha256")
    if sequence_length != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(
            "packed split manifests bind the canonical S0 packing config; "
            f"sequence_length must be {DEFAULT_SEQUENCE_LENGTH}"
        )

    identity = tokenizer.identity
    _require_sha256(identity.config_sha256, "tokenizer_config_sha256")
    _require_sha256(identity.vocab_sha256, "tokenizer_vocab_sha256")

    document_count = 0
    codepoint_count = 0
    utf8_byte_count = 0
    token_count = 0
    causal_loss_token_count = 0
    packed_example_count = 0
    packed_input_token_count = 0
    documents_without_causal_pair = 0

    for record in records:
        if record.split != split:
            raise SplitMixError(
                f"record {record.record_id!r} has split {record.split!r}; expected {split!r}"
            )
        document_count += 1
        codepoint_count += len(record.text)
        utf8_byte_count += len(record.text.encode("utf-8"))
        encoded = tokenizer.encode(record.text)
        token_count += len(encoded)
        if len(encoded) < 2:
            documents_without_causal_pair += 1

        for example in iter_packed_examples(
            [record],
            tokenizer,
            expected_split=split,
            sequence_length=sequence_length,
        ):
            packed_example_count += 1
            packed_input_token_count += sum(example.attention_mask)
            causal_loss_token_count += example.num_loss_tokens

    packed_capacity_token_count = packed_example_count * sequence_length
    masked_fill_position_count = packed_capacity_token_count - packed_input_token_count

    return PackedSplitManifest(
        schema_version=2,
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity_sha256,
        split=split,
        source_jsonl_sha256=source_jsonl_sha256,
        tokenizer_version=identity.version,
        tokenizer_config_sha256=identity.config_sha256,
        tokenizer_vocab_sha256=identity.vocab_sha256,
        vocab_size=identity.vocab_size,
        packing_version=PACKING_VERSION,
        packing_config_sha256=PACKING_CONFIG_HASH,
        sequence_length=sequence_length,
        document_count=document_count,
        codepoint_count=codepoint_count,
        utf8_byte_count=utf8_byte_count,
        token_count=token_count,
        causal_loss_token_count=causal_loss_token_count,
        packed_example_count=packed_example_count,
        packed_input_token_count=packed_input_token_count,
        packed_capacity_token_count=packed_capacity_token_count,
        masked_fill_position_count=masked_fill_position_count,
        documents_without_causal_pair=documents_without_causal_pair,
    )
