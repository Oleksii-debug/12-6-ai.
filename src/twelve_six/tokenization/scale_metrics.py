"""Streaming tokenizer-efficiency measurements for scale decisions.

These metrics are diagnostics, not a tokenizer promotion gate. They quantify sequence-length and
vocabulary-parameter trade-offs without loading model weights or training a language model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .base import TokenizerProtocol

BYTE_BASELINE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class TokenizerScaleMeasurement:
    documents: int
    empty_documents: int
    utf8_bytes: int
    codepoints: int
    whitespace_words: int
    tokens: int
    documents_over_context: int
    max_document_tokens: int
    roundtrip_failures: int
    context_length: int

    def to_dict(self) -> dict[str, int | float]:
        token_ratio = self.tokens / self.utf8_bytes if self.utf8_bytes else 0.0
        bytes_per_token = self.utf8_bytes / self.tokens if self.tokens else 0.0
        tokens_per_codepoint = self.tokens / self.codepoints if self.codepoints else 0.0
        tokens_per_word = self.tokens / self.whitespace_words if self.whitespace_words else 0.0
        over_context_fraction = (
            self.documents_over_context / self.documents if self.documents else 0.0
        )
        return {
            "documents": self.documents,
            "empty_documents": self.empty_documents,
            "utf8_bytes": self.utf8_bytes,
            "codepoints": self.codepoints,
            "whitespace_words": self.whitespace_words,
            "tokens": self.tokens,
            "tokens_per_utf8_byte": token_ratio,
            "bytes_per_token": bytes_per_token,
            "tokens_per_codepoint": tokens_per_codepoint,
            "tokens_per_whitespace_word": tokens_per_word,
            "relative_sequence_length_vs_raw_byte": token_ratio,
            "relative_dense_attention_pairs_vs_raw_byte": token_ratio * token_ratio,
            "documents_over_context": self.documents_over_context,
            "documents_over_context_fraction": over_context_fraction,
            "max_document_tokens": self.max_document_tokens,
            "roundtrip_failures": self.roundtrip_failures,
            "context_length": self.context_length,
        }


def measure_tokenizer(
    tokenizer: TokenizerProtocol,
    texts: Iterable[str],
    *,
    context_length: int,
) -> TokenizerScaleMeasurement:
    """Measure one tokenizer in a single streaming pass over normalized text."""
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 2:
        raise ValueError("context_length must be an integer >= 2")

    documents = 0
    empty_documents = 0
    utf8_bytes = 0
    codepoints = 0
    whitespace_words = 0
    tokens = 0
    documents_over_context = 0
    max_document_tokens = 0
    roundtrip_failures = 0

    for text in texts:
        if not isinstance(text, str):
            raise TypeError("tokenizer scale measurement expects strings")
        documents += 1
        if not text:
            empty_documents += 1
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        byte_count = len(text.encode("utf-8"))

        utf8_bytes += byte_count
        codepoints += len(text)
        whitespace_words += len(text.split())
        tokens += token_count
        max_document_tokens = max(max_document_tokens, token_count)
        if token_count > context_length:
            documents_over_context += 1

        try:
            decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        except (TypeError, ValueError, UnicodeError):
            roundtrip_failures += 1
        else:
            if decoded != text:
                roundtrip_failures += 1

    return TokenizerScaleMeasurement(
        documents=documents,
        empty_documents=empty_documents,
        utf8_bytes=utf8_bytes,
        codepoints=codepoints,
        whitespace_words=whitespace_words,
        tokens=tokens,
        documents_over_context=documents_over_context,
        max_document_tokens=max_document_tokens,
        roundtrip_failures=roundtrip_failures,
        context_length=context_length,
    )


def vocabulary_parameter_cost(
    *,
    vocab_size: int,
    d_model: int,
    tied_embeddings: bool,
    target_parameters: int | None = None,
    baseline_vocab_size: int = BYTE_BASELINE_VOCAB_SIZE,
) -> dict[str, int | float | bool | None]:
    """Expose the embedding-parameter cost that sequence compression can hide."""
    for name, value in (
        ("vocab_size", vocab_size),
        ("d_model", d_model),
        ("baseline_vocab_size", baseline_vocab_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if target_parameters is not None and (
        isinstance(target_parameters, bool)
        or not isinstance(target_parameters, int)
        or target_parameters <= 0
    ):
        raise ValueError("target_parameters must be a positive integer when provided")

    matrices = 1 if tied_embeddings else 2
    embedding_parameters = vocab_size * d_model * matrices
    baseline_embedding_parameters = baseline_vocab_size * d_model * matrices
    delta = embedding_parameters - baseline_embedding_parameters
    share = (
        embedding_parameters / target_parameters if target_parameters is not None else None
    )
    return {
        "vocab_size": vocab_size,
        "baseline_vocab_size": baseline_vocab_size,
        "d_model": d_model,
        "tied_embeddings": tied_embeddings,
        "embedding_matrices": matrices,
        "embedding_parameters": embedding_parameters,
        "baseline_embedding_parameters": baseline_embedding_parameters,
        "embedding_parameter_delta_vs_byte_vocab": delta,
        "target_parameters": target_parameters,
        "embedding_share_of_target_parameters": share,
    }
