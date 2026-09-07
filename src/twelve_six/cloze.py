"""First-party raw conditional-likelihood primitives for 12-6 AI.

The scorer is deliberately small: it evaluates a continuation under the decoder
without chat formatting, generation, or third-party benchmark frameworks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


class DecoderOutputLike(Protocol):
    logits: torch.Tensor


class DecoderLike(Protocol):
    training: bool

    @property
    def spec(self): ...

    def __call__(self, input_ids: torch.Tensor) -> DecoderOutputLike: ...

    def eval(self): ...

    def train(self, mode: bool = True): ...


@dataclass(frozen=True)
class ConditionalLikelihood:
    log_likelihood: float
    target_tokens: int
    target_utf8_bytes: int
    mean_log_likelihood_per_token: float
    mean_log_likelihood_per_utf8_byte: float
    bits_per_byte: float
    greedy_match: bool


@dataclass(frozen=True)
class TextLikelihood:
    log_likelihood: float
    scored_tokens: int
    input_utf8_bytes: int
    scored_utf8_bytes: int
    bits_per_scored_byte: float


def _validate_logits(logits: torch.Tensor, *, vocab_size: int) -> None:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != vocab_size:
        raise ValueError("decoder logits must have shape [1, time, vocab_size]")
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("decoder returned non-finite logits")


@torch.no_grad()
def conditional_log_likelihood(
    model: DecoderLike,
    tokenizer: TokenizerLike,
    context: str,
    continuation: str,
) -> ConditionalLikelihood:
    """Score ``continuation`` conditioned on ``context`` using raw LM likelihood."""

    if not isinstance(context, str) or not context:
        raise ValueError("context must be a non-empty string")
    if not isinstance(continuation, str) or not continuation:
        raise ValueError("continuation must be a non-empty string")

    context_ids = [int(token) for token in tokenizer.encode(context)]
    continuation_ids = [int(token) for token in tokenizer.encode(continuation)]
    combined_ids = [int(token) for token in tokenizer.encode(context + continuation)]
    if not context_ids or not continuation_ids:
        raise ValueError("context and continuation must each produce tokens")
    if combined_ids != context_ids + continuation_ids:
        raise ValueError(
            "tokenizer boundary is not compositional for this pair; "
            "author the cloze boundary explicitly"
        )

    max_seq_len = int(model.spec.max_seq_len)
    if len(combined_ids) > max_seq_len:
        raise ValueError(
            f"cloze sequence has {len(combined_ids)} tokens but model limit is {max_seq_len}"
        )

    was_training = bool(model.training)
    model.eval()
    try:
        input_ids = torch.tensor(combined_ids, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits
        vocab_size = int(model.spec.vocab_size)
        _validate_logits(logits, vocab_size=vocab_size)
        start = len(context_ids) - 1
        stop = start + len(continuation_ids)
        prediction_logits = logits[0, start:stop, :]
        targets = torch.tensor(continuation_ids, dtype=torch.long)
        log_probs = F.log_softmax(prediction_logits.float(), dim=-1)
        selected = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        total = float(selected.sum().item())
        greedy = bool(torch.equal(torch.argmax(prediction_logits, dim=-1).cpu(), targets))
    finally:
        model.train(was_training)

    target_tokens = len(continuation_ids)
    target_bytes = len(continuation.encode("utf-8"))
    if target_tokens <= 0 or target_bytes <= 0:
        raise RuntimeError("cloze target unexpectedly became empty")
    return ConditionalLikelihood(
        log_likelihood=total,
        target_tokens=target_tokens,
        target_utf8_bytes=target_bytes,
        mean_log_likelihood_per_token=total / target_tokens,
        mean_log_likelihood_per_utf8_byte=total / target_bytes,
        bits_per_byte=-total / (math.log(2.0) * target_bytes),
        greedy_match=greedy,
    )


@torch.no_grad()
def text_log_likelihood(
    model: DecoderLike,
    tokenizer: TokenizerLike,
    text: str,
    *,
    require_byte_tokenizer: bool = False,
) -> TextLikelihood:
    """Score all predictable tokens in a raw text; the first token is conditioning."""

    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    token_ids = [int(token) for token in tokenizer.encode(text)]
    if len(token_ids) < 2:
        raise ValueError("text must produce at least two tokens")
    encoded = text.encode("utf-8")
    if require_byte_tokenizer and token_ids != list(encoded):
        raise ValueError("context BPB requires the canonical byte tokenizer")
    if len(token_ids) > int(model.spec.max_seq_len):
        raise ValueError("text exceeds model context")

    was_training = bool(model.training)
    model.eval()
    try:
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits
        vocab_size = int(model.spec.vocab_size)
        _validate_logits(logits, vocab_size=vocab_size)
        prediction_logits = logits[0, :-1, :]
        targets = input_ids[0, 1:]
        log_probs = F.log_softmax(prediction_logits.float(), dim=-1)
        total = float(log_probs.gather(1, targets.unsqueeze(1)).sum().item())
    finally:
        model.train(was_training)

    scored_tokens = len(token_ids) - 1
    if require_byte_tokenizer:
        scored_bytes = len(encoded) - 1
    else:
        scored_bytes = len(encoded)
    if scored_bytes <= 0:
        raise RuntimeError("text produced no scored bytes")
    return TextLikelihood(
        log_likelihood=total,
        scored_tokens=scored_tokens,
        input_utf8_bytes=len(encoded),
        scored_utf8_bytes=scored_bytes,
        bits_per_scored_byte=-total / (math.log(2.0) * scored_bytes),
    )
