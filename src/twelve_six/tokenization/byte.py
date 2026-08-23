"""Deterministic raw UTF-8 byte tokenizer for the S0 learning-factory stage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from types import MappingProxyType

from .base import TokenizerIdentity

BYTE_TOKENIZER_VERSION = "s0-byte-v1"
BYTE_TOKENIZER_HASH = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"

_CONFIG = {
    "schema_version": 1,
    "tokenizer_version": BYTE_TOKENIZER_VERSION,
    "type": "utf8-byte",
    "normalization": "none",
    "encoding": "utf-8",
    "special_tokens": {},
    "byte_offset": 0,
    "byte_values": 256,
    "vocab_size": 256,
}


def canonical_config_json() -> str:
    """Return the exact canonical serialization used for tokenizer identity."""
    return json.dumps(_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tokenizer_config_hash() -> str:
    """Return the SHA-256 identity of the canonical tokenizer config."""
    return hashlib.sha256(canonical_config_json().encode("utf-8")).hexdigest()


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with frozen 0..255 token semantics.

    S0 intentionally reserves no semantic special-token IDs. Every token ID is
    exactly one raw byte value, so the tokenizer matches D01's 256-token S0
    ModelSpec without enlarging the ~10K model merely for delimiters.
    """

    pad_id = None
    bos_id = None
    eos_id = None
    byte_offset = 0
    vocab_size = 256
    version = BYTE_TOKENIZER_VERSION
    normalization = "none"
    encoding = "utf-8"
    special_tokens = MappingProxyType({})

    def __init__(self) -> None:
        if tokenizer_config_hash() != BYTE_TOKENIZER_HASH:
            raise RuntimeError(
                "Byte tokenizer config identity drifted without a version/hash update"
            )

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            version=self.version,
            config_sha256=BYTE_TOKENIZER_HASH,
            vocab_size=self.vocab_size,
            normalization=self.normalization,
            encoding=self.encoding,
            special_tokens=self.special_tokens,
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if add_bos:
            raise ValueError("s0-byte-v1 has no BOS token")
        if add_eos:
            raise ValueError("s0-byte-v1 has no EOS token")
        return list(text.encode(self.encoding))

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del skip_special_tokens
        data = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if not 0 <= token_id <= 255:
                raise ValueError(f"token ID {token_id} is outside tokenizer vocabulary")
            data.append(token_id)
        return bytes(data).decode(self.encoding, errors=errors)

    @staticmethod
    def oov_count(text: str) -> int:
        """Byte tokenization covers every Python string encodable as UTF-8."""
        text.encode("utf-8")
        return 0

    def fertility(self, text: str) -> float:
        """Tokens per Unicode code point."""
        if not text:
            return 0.0
        return len(text.encode(self.encoding)) / len(text)
