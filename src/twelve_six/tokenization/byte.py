"""Deterministic UTF-8 byte tokenizer for the S0 learning-factory stage."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Iterable

from .base import TokenizerIdentity

BYTE_TOKENIZER_VERSION = "s0-byte-v1"
BYTE_TOKENIZER_HASH = "86e7696e39d04e00105dc0bd1149c67abd703d69734c54f503f4a88343256294"

_CONFIG = {
    "schema_version": 1,
    "tokenizer_version": BYTE_TOKENIZER_VERSION,
    "type": "utf8-byte",
    "normalization": "none",
    "encoding": "utf-8",
    "special_tokens": {"pad": 0, "bos": 1, "eos": 2},
    "byte_offset": 3,
    "byte_values": 256,
    "vocab_size": 259,
}


def canonical_config_json() -> str:
    """Return the exact canonical serialization used for tokenizer identity."""
    return json.dumps(_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tokenizer_config_hash() -> str:
    """Return the SHA-256 identity of the canonical tokenizer config."""
    return hashlib.sha256(canonical_config_json().encode("utf-8")).hexdigest()


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with frozen S0 token-ID semantics.

    IDs:
      0 = PAD
      1 = BOS
      2 = EOS
      3..258 = raw byte values 0..255

    Text is never Unicode-normalized. This preserves exact input code points and
    gives complete Unicode coverage through UTF-8 without an OOV token.
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3
    vocab_size = 259
    version = BYTE_TOKENIZER_VERSION
    normalization = "none"
    encoding = "utf-8"
    special_tokens = MappingProxyType({"pad": pad_id, "bos": bos_id, "eos": eos_id})

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
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self.byte_offset + value for value in text.encode(self.encoding))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        data = bytearray()
        rendered: list[str] = []

        def flush() -> None:
            if data:
                rendered.append(bytes(data).decode(self.encoding, errors=errors))
                data.clear()

        special_by_id = {value: f"<{name}>" for name, value in self.special_tokens.items()}
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id in special_by_id:
                if skip_special_tokens:
                    continue
                flush()
                rendered.append(special_by_id[token_id])
                continue
            byte_value = token_id - self.byte_offset
            if not 0 <= byte_value <= 255:
                raise ValueError(f"token ID {token_id} is outside tokenizer vocabulary")
            data.append(byte_value)
        flush()
        return "".join(rendered)

    @staticmethod
    def oov_count(text: str) -> int:
        """Byte tokenization covers every Python string encodable as UTF-8."""
        text.encode("utf-8")
        return 0

    def fertility(self, text: str) -> float:
        """Tokens per Unicode code point, excluding optional special tokens."""
        if not text:
            return 0.0
        return len(text.encode(self.encoding)) / len(text)
