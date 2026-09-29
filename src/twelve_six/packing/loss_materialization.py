"""Build the canonical post-pack loss materialization consumed by NEXT100-064 V2.

This module is deliberately a bridge around the existing S0 byte tokenizer and
packing implementation.  It does not introduce a second packing algorithm.
The current canonical packer has no in-document reservation masking semantics,
so non-empty target reservations fail closed instead of being silently exposed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from twelve_six.tokenization import TokenizerProtocol

from .core import (
    DEFAULT_SEQUENCE_LENGTH,
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    TextRecord,
    iter_packed_examples,
)

MATERIALIZATION_SCHEMA = "12-6.postpack-loss-materialization.v2"
_REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_HEX = frozenset("0123456789abcdef")


class LossMaterializationError(ValueError):
    """Raised when a terminal loss materialization cannot be built safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_obj(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise LossMaterializationError(f"{field} must be exact lowercase SHA-256")
    return value


def tokenizer_identity_sha256(tokenizer: TokenizerProtocol) -> str:
    """Hash the complete existing TokenizerIdentity without inventing token semantics."""
    return _sha256_obj(tokenizer.identity.to_dict())


@dataclass(frozen=True, slots=True)
class LossMaterializationDocument:
    """One normalized terminal-corpus document plus policy metadata needed downstream."""

    document_id: str
    text: str
    source_id: str
    language: str
    modality: str
    family_id: str
    normalized_payload_sha256: str
    source_bytes: int
    split: str
    dedup_cluster_id: str
    retained_after_dedup: bool = True
    evaluation_reserved: bool = False
    reserved_target_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "document_id",
            "source_id",
            "language",
            "modality",
            "family_id",
            "split",
            "dedup_cluster_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise LossMaterializationError(f"{field} must be non-empty text")
        if self.modality not in {"text", "code"}:
            raise LossMaterializationError("modality must be 'text' or 'code'")
        if not isinstance(self.text, str):
            raise LossMaterializationError("text must be a string")
        _require_sha256(self.normalized_payload_sha256, "normalized_payload_sha256")
        payload = self.text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != self.normalized_payload_sha256:
            raise LossMaterializationError(
                f"{self.document_id}: normalized payload hash does not match text"
            )
        if isinstance(self.source_bytes, bool) or not isinstance(self.source_bytes, int):
            raise LossMaterializationError("source_bytes must be a non-negative integer")
        if self.source_bytes < 0 or self.source_bytes != len(payload):
            raise LossMaterializationError(
                f"{self.document_id}: source_bytes does not match normalized UTF-8 bytes"
            )
        if not isinstance(self.retained_after_dedup, bool):
            raise LossMaterializationError("retained_after_dedup must be boolean")
        if not isinstance(self.evaluation_reserved, bool):
            raise LossMaterializationError("evaluation_reserved must be boolean")
        if self.split != "train" and self.retained_after_dedup and not self.evaluation_reserved:
            raise LossMaterializationError(
                "retained held-out documents must be evaluation_reserved"
            )
        previous_end = 1
        for index, value in enumerate(self.reserved_target_ranges):
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or isinstance(value[0], bool)
                or isinstance(value[1], bool)
                or not isinstance(value[0], int)
                or not isinstance(value[1], int)
            ):
                raise LossMaterializationError(
                    f"reserved_target_ranges[{index}] must be an integer tuple"
                )
            start, end = value
            if start < 1 or start >= end or start < previous_end:
                raise LossMaterializationError(
                    "reserved_target_ranges must be sorted, non-overlapping, and start >= 1"
                )
            previous_end = end


def _normalize_stage_bindings(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_REQUIRED_STAGE_BINDINGS):
        raise LossMaterializationError(
            "stage_bindings must contain exactly normalization, evaluation_reservations, "
            "dedup, split and packing"
        )
    return {
        name: _require_sha256(value[name], f"stage_bindings.{name}")
        for name in _REQUIRED_STAGE_BINDINGS
    }


def _document_projection(
    document: LossMaterializationDocument,
    tokenizer: TokenizerProtocol,
) -> tuple[dict[str, Any], list[int]]:
    encoded = tokenizer.encode(document.text)
    token_count = len(encoded)
    for start, end in document.reserved_target_ranges:
        if end > token_count:
            raise LossMaterializationError(
                f"{document.document_id}: reserved target range exceeds token_count"
            )

    optimizable = (
        document.split == "train"
        and document.retained_after_dedup
        and not document.evaluation_reserved
        and token_count > 1
    )
    if optimizable and document.reserved_target_ranges:
        raise LossMaterializationError(
            f"{document.document_id}: canonical {PACKING_VERSION} has no in-document "
            "reservation masking; refuse materialization"
        )
    eligible = [[1, token_count]] if optimizable else []
    return (
        {
            "document_id": document.document_id,
            "source_id": document.source_id,
            "language": document.language,
            "modality": document.modality,
            "family_id": document.family_id,
            "normalized_payload_sha256": document.normalized_payload_sha256,
            "source_bytes": document.source_bytes,
            "token_count": token_count,
            "split": document.split,
            "dedup_cluster_id": document.dedup_cluster_id,
            "retained_after_dedup": document.retained_after_dedup,
            "evaluation_reserved": document.evaluation_reserved,
            "reserved_target_ranges": [list(item) for item in document.reserved_target_ranges],
            "eligible_target_ranges": eligible,
        },
        encoded,
    )


def _pack_document(
    document: LossMaterializationDocument,
    tokenizer: TokenizerProtocol,
    encoded: Sequence[int],
) -> list[dict[str, Any]]:
    if (
        document.split != "train"
        or not document.retained_after_dedup
        or document.evaluation_reserved
        or len(encoded) < 2
    ):
        return []

    examples = tuple(
        iter_packed_examples(
            (TextRecord(document.document_id, document.text, "train"),),
            tokenizer,
            expected_split="train",
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
        )
    )
    packs: list[dict[str, Any]] = []
    target_start = 1
    for index, example in enumerate(examples):
        loss_count = example.num_loss_tokens
        if loss_count <= 0:
            raise AssertionError("canonical packer emitted an empty loss block")
        target_end = target_start + loss_count
        if target_end > len(encoded):
            raise AssertionError("canonical packer emitted targets beyond document length")
        packs.append(
            {
                "pack_id": f"{document.document_id}:{index:08d}",
                "token_count": len(example.input_ids),
                "token_ids": list(example.input_ids),
                "loss_spans": [
                    {
                        "document_id": document.document_id,
                        "target_start": target_start,
                        "target_end": target_end,
                        "pack_target_start": 1,
                    }
                ],
            }
        )
        target_start = target_end
    if target_start != len(encoded):
        raise AssertionError("canonical packer did not cover every document causal target")
    return packs


def build_postpack_loss_materialization(
    documents: Sequence[LossMaterializationDocument],
    tokenizer: TokenizerProtocol,
    *,
    terminal_corpus_authority_identity_sha256: str,
    stage_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic V2 materialization from the canonical S0 packing path.

    Terminal S0 currently permits whole-document evaluation reservation only.  If a
    retained training document carries any in-document target reservation, this
    function fails closed until a separately reviewed packing-version change exists.
    """
    if not documents:
        raise LossMaterializationError("documents must not be empty")
    corpus_identity = _require_sha256(
        terminal_corpus_authority_identity_sha256,
        "terminal_corpus_authority_identity_sha256",
    )
    bindings = _normalize_stage_bindings(stage_bindings)
    if tokenizer.vocab_size <= 0:
        raise LossMaterializationError("tokenizer vocab_size must be positive")

    ordered = sorted(documents, key=lambda item: item.document_id)
    if len({item.document_id for item in ordered}) != len(ordered):
        raise LossMaterializationError("document_id values must be unique")

    document_rows: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    for document in ordered:
        row, encoded = _document_projection(document, tokenizer)
        document_rows.append(row)
        packs.extend(_pack_document(document, tokenizer, encoded))

    tokenizer_identity = tokenizer_identity_sha256(tokenizer)
    value: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "terminal_corpus_authority_identity_sha256": corpus_identity,
        "stage_bindings": bindings,
        "tokenizer": {
            "name": tokenizer.version,
            "identity_sha256": tokenizer_identity,
            "source_bytes_are_loss_positions": False,
        },
        "documents": document_rows,
        "packing": {
            "identity_sha256": PACKING_CONFIG_HASH,
            "complete_one_pass": True,
            "packs": packs,
        },
    }
    value["materialization_identity_sha256"] = _sha256_obj(value)
    return value
