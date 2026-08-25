"""Experimental Base special-token contract layered on frozen S0 byte semantics.

This module is deliberately additive. Canonical ``s0-byte-v1`` remains byte IDs
0..255 with no special tokens. The experimental contract appends exactly one
trained end-of-document/end-of-sequence token at ID 256 and defines no BOS, PAD,
UNK, instruction, system, or chat tokens.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from .base import TokenizerIdentity
from .byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)

EXPERIMENTAL_TOKENIZER_VERSION = "exp-byte-eos-v1"
EXPERIMENTAL_EOS_ID = 256
EXPERIMENTAL_EOS_SURFACE = "<|end_of_document|>"
EXPERIMENTAL_VOCAB_SIZE = 257
EXPERIMENTAL_CONFIG_SHA256 = "9d26ab7c69e51f36192fbbe3313e13327a2a97ff1134dd24e79f2d1227dc59a0"
EXPERIMENTAL_VOCAB_SHA256 = "2ae636644ebff60166fe69e1d83a15a1be45aada86a33977cb888c52cf5dc21d"

_CONTRACT: dict[str, Any] = {
    "schema": "12-6.experimental-special-token-contract.v1",
    "status": "EXPERIMENTAL_NOT_CANONICAL",
    "tokenizer_version": EXPERIMENTAL_TOKENIZER_VERSION,
    "base_tokenizer": {
        "version": BYTE_TOKENIZER_VERSION,
        "config_sha256": BYTE_TOKENIZER_HASH,
        "vocab_sha256": BYTE_VOCAB_HASH,
        "vocab_size": 256,
    },
    "vocab_size": EXPERIMENTAL_VOCAB_SIZE,
    "normalization": "none",
    "encoding": "utf-8",
    "byte_token_ids": {"start": 0, "end": 255, "mapping": "identity"},
    "special_tokens": {
        "eos": {
            "id": EXPERIMENTAL_EOS_ID,
            "surface": EXPERIMENTAL_EOS_SURFACE,
            "trained": True,
        }
    },
    "absent_special_tokens": ["bos", "pad", "unk"],
    "defaults": {"add_bos": False, "add_eos": False},
    "padding": {
        "semantic_pad_id": None,
        "masked_fill_token_id": 0,
        "requires_attention_mask": True,
        "requires_ignored_labels": True,
    },
    "empty_context": {
        "supported": False,
        "reason": "no trained BOS token",
    },
    "hf_bridge": {
        "bos_token_id": None,
        "eos_token_id": EXPERIMENTAL_EOS_ID,
        "pad_token_id": None,
        "unk_token_id": None,
        "chat_template": None,
    },
    "instruction_tokens": [],
    "system_tokens": [],
    "promotion_allowed": False,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _vocab_payload() -> dict[str, object]:
    entries: list[dict[str, object]] = [
        {"id": token_id, "kind": "byte", "byte": token_id}
        for token_id in range(256)
    ]
    entries.append(
        {
            "id": EXPERIMENTAL_EOS_ID,
            "kind": "special",
            "role": "eos",
            "surface": EXPERIMENTAL_EOS_SURFACE,
        }
    )
    return {"schema": "12-6.experimental-vocab.v1", "entries": entries}


def contract_payload() -> dict[str, object]:
    """Return a detached JSON-compatible copy of the immutable contract."""
    return json.loads(_canonical_json(_CONTRACT))


def contract_artifact_bytes() -> bytes:
    """Return canonical contract bytes used for the config identity hash."""
    return _canonical_json(_CONTRACT).encode("utf-8")


def vocab_artifact_bytes() -> bytes:
    """Return canonical complete-vocabulary bytes used for the vocab identity hash."""
    return _canonical_json(_vocab_payload()).encode("utf-8")


def hf_special_token_ids() -> Mapping[str, int | None]:
    """Return the Base-safe HF token-ID mapping; no chat template is implied."""
    return MappingProxyType(
        {
            "bos_token_id": None,
            "eos_token_id": EXPERIMENTAL_EOS_ID,
            "pad_token_id": None,
            "unk_token_id": None,
        }
    )


def apply_to_llama_config(config: Mapping[str, object]) -> dict[str, object]:
    """Apply this contract to an already-built raw-Base Llama config.

    The function refuses to overwrite conflicting special-token semantics and
    requires a model whose embedding/output vocabulary was built for 257 IDs.
    """
    result = dict(config)
    if result.get("vocab_size") != EXPERIMENTAL_VOCAB_SIZE:
        raise ValueError(
            "experimental EOS contract requires model vocab_size=257; "
            "existing S0 checkpoints remain vocab_size=256"
        )
    expected = hf_special_token_ids()
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        current = result.get(field)
        target = expected[field]
        if current is not None and current != target:
            raise ValueError(f"refusing to overwrite conflicting {field}")
        result[field] = target
    if result.get("chat_template") is not None:
        raise ValueError("raw Base special-token contract forbids chat_template")
    return result


class ExperimentalByteEosTokenizer:
    """Lossless UTF-8 byte tokenizer with one appended trained EOS/EOD token."""

    pad_id: int | None = None
    bos_id: int | None = None
    eos_id: int | None = EXPERIMENTAL_EOS_ID
    unk_id: int | None = None
    vocab_size = EXPERIMENTAL_VOCAB_SIZE
    version = EXPERIMENTAL_TOKENIZER_VERSION

    def __init__(self) -> None:
        self._identity = TokenizerIdentity(
            version=self.version,
            config_sha256=EXPERIMENTAL_CONFIG_SHA256,
            vocab_sha256=EXPERIMENTAL_VOCAB_SHA256,
            vocab_size=self.vocab_size,
            normalization="none",
            encoding="utf-8",
            special_tokens=MappingProxyType({"eos": EXPERIMENTAL_EOS_ID}),
        )

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if add_bos:
            raise ValueError("experimental Base contract defines no BOS token")
        ids = list(text.encode("utf-8"))
        if add_eos:
            ids.append(EXPERIMENTAL_EOS_ID)
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        if errors not in {"strict", "replace"}:
            raise ValueError("errors must be 'strict' or 'replace'")
        pieces: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors=errors))
                byte_buffer.clear()

        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token IDs must be integers")
            if 0 <= token_id <= 255:
                byte_buffer.append(token_id)
            elif token_id == EXPERIMENTAL_EOS_ID:
                if not skip_special_tokens:
                    flush_bytes()
                    pieces.append(EXPERIMENTAL_EOS_SURFACE)
            else:
                raise ValueError(f"token ID outside experimental vocabulary: {token_id}")
        flush_bytes()
        return "".join(pieces)

    def oov_count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        text.encode("utf-8")
        return 0


if hashlib.sha256(contract_artifact_bytes()).hexdigest() != EXPERIMENTAL_CONFIG_SHA256:
    raise RuntimeError("experimental special-token contract hash drifted")
if hashlib.sha256(vocab_artifact_bytes()).hexdigest() != EXPERIMENTAL_VOCAB_SHA256:
    raise RuntimeError("experimental special-token vocabulary hash drifted")
