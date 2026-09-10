"""Stable tokenizer interface and checkpoint-compatibility guards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TokenizerIdentity:
    version: str
    config_sha256: str
    vocab_sha256: str
    vocab_size: int
    normalization: str
    encoding: str
    special_tokens: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "config_sha256": self.config_sha256,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": self.vocab_size,
            "normalization": self.normalization,
            "encoding": self.encoding,
            "special_tokens": dict(self.special_tokens),
        }


class TokenizerProtocol(Protocol):
    pad_id: int | None
    bos_id: int | None
    eos_id: int | None
    vocab_size: int
    version: str

    @property
    def identity(self) -> TokenizerIdentity: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str: ...


class TokenizerCompatibilityError(ValueError):
    """Raised when a runtime tokenizer does not match checkpoint identity."""


def require_tokenizer_identity(
    tokenizer: TokenizerProtocol,
    *,
    expected_version: str,
    expected_config_sha256: str,
    expected_vocab_size: int,
    expected_vocab_sha256: str | None = None,
) -> None:
    """Fail closed if checkpoint-recorded tokenizer identity does not match runtime."""
    identity = tokenizer.identity
    mismatches: list[str] = []
    if identity.version != expected_version:
        mismatches.append(f"version={identity.version!r} expected {expected_version!r}")
    if identity.config_sha256 != expected_config_sha256:
        mismatches.append(
            f"config_sha256={identity.config_sha256!r} expected {expected_config_sha256!r}"
        )
    if identity.vocab_size != expected_vocab_size:
        mismatches.append(f"vocab_size={identity.vocab_size} expected {expected_vocab_size}")
    if expected_vocab_sha256 is not None and identity.vocab_sha256 != expected_vocab_sha256:
        mismatches.append(
            f"vocab_sha256={identity.vocab_sha256!r} expected {expected_vocab_sha256!r}"
        )
    if mismatches:
        raise TokenizerCompatibilityError("; ".join(mismatches))
