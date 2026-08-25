from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import DecoderKVCache, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


@dataclass(frozen=True, slots=True)
class TorchBatchCallStats:
    batch_size: int
    min_sequence_length: int
    max_sequence_length: int
    logical_input_positions: int
    padded_input_positions: int
    right_padding_positions: int
    input_tensor_bytes: int
    output_logits_bytes: int


def _validate_padding_token_id(model: TwelveSixDecoder, padding_token_id: int) -> None:
    if (
        not isinstance(padding_token_id, int)
        or isinstance(padding_token_id, bool)
        or not 0 <= padding_token_id < model.spec.vocab_size
    ):
        raise ValueError("padding_token_id must be an integer inside the model vocabulary")


def _validated_rows(
    model: TwelveSixDecoder,
    input_ids: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    if not input_ids:
        raise ValueError("input_ids batch must be non-empty")

    rows: list[list[int]] = []
    lengths: list[int] = []
    for row in input_ids:
        if not row:
            raise ValueError("batched input sequences must be non-empty")
        if len(row) > model.spec.max_seq_len:
            raise ValueError("batched input sequence exceeds model context")
        values: list[int] = []
        for token_id in row:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("batched input token IDs must be integers")
            if not 0 <= token_id < model.spec.vocab_size:
                raise ValueError("batched input token ID is outside model vocabulary")
            values.append(token_id)
        rows.append(values)
        lengths.append(len(values))
    return rows, lengths


@torch.no_grad()
def right_padded_next_token_logits(
    model: TwelveSixDecoder,
    input_ids: Sequence[Sequence[int]],
    *,
    padding_token_id: int = 0,
) -> tuple[list[list[float]], TorchBatchCallStats]:
    """Evaluate heterogeneous prefixes in one canonical causal model forward.

    Padding is appended only after each request's final real token. Causal attention
    prevents those future filler positions from affecting logits gathered at that real
    token, so no semantic padding token or attention-mask convention is introduced.
    """
    _validate_padding_token_id(model, padding_token_id)
    rows, lengths = _validated_rows(model, input_ids)
    max_length = max(lengths)
    padded_rows = [
        [*row, *([padding_token_id] * (max_length - len(row)))]
        for row in rows
    ]

    device = next(model.parameters()).device
    tensor = torch.tensor(padded_rows, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        full_logits = model(tensor).logits
        row_indices = torch.arange(len(rows), device=full_logits.device)
        final_indices = torch.tensor(
            [length - 1 for length in lengths],
            dtype=torch.long,
            device=full_logits.device,
        )
        selected = full_logits[row_indices, final_indices]
        values = selected.detach().float().cpu().tolist()
        output_logits_bytes = full_logits.numel() * full_logits.element_size()
    finally:
        model.train(was_training)

    logical_positions = sum(lengths)
    padded_positions = len(rows) * max_length
    stats = TorchBatchCallStats(
        batch_size=len(rows),
        min_sequence_length=min(lengths),
        max_sequence_length=max_length,
        logical_input_positions=logical_positions,
        padded_input_positions=padded_positions,
        right_padding_positions=padded_positions - logical_positions,
        input_tensor_bytes=tensor.numel() * tensor.element_size(),
        output_logits_bytes=output_logits_bytes,
    )
    return values, stats


class S0TorchBatchedGenerationSession:
    """One fixed-row equal-length batch backed by the incumbent model-native KV cache."""

    def __init__(
        self,
        backend: S0TorchBatchedInferenceBackend,
        input_ids: Sequence[Sequence[int]],
    ) -> None:
        rows, lengths = _validated_rows(backend.model, input_ids)
        if len(set(lengths)) != 1:
            raise ValueError("KV-cache batch prefill requires exact-equal sequence lengths")

        self._backend = backend
        self._model = backend.model
        self._batch_size = len(rows)
        self._closed = False
        self.tokens_processed = sum(lengths)
        self._cache: DecoderKVCache
        self._logits: list[list[float]]

        self._backend._acquire_generation_session()
        try:
            tensor = torch.tensor(
                rows,
                dtype=torch.long,
                device=next(self._model.parameters()).device,
            )
            output, self._cache = self._model.prefill_kv_cache(tensor)
            self._logits = output.logits[:, -1].detach().float().cpu().tolist()
        except Exception:
            self._backend._release_generation_session()
            self._closed = True
            raise

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def sequence_length(self) -> int:
        return self._cache.sequence_length

    @property
    def cache_bytes(self) -> int:
        self._require_open()
        return sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            for layer in self._cache.layers
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("batched generation session is closed")

    def next_token_logits_batch(self) -> Sequence[Sequence[float]]:
        self._require_open()
        return [list(row) for row in self._logits]

    def append_batch(self, token_ids: Sequence[int]) -> None:
        self._require_open()
        if len(token_ids) != self._batch_size:
            raise ValueError("batched KV append must provide exactly one token per cache row")
        self._backend._validate_token_ids(token_ids)
        if self._cache.sequence_length >= self._backend.max_context_tokens:
            raise ValueError("KV cache is already at model context limit")
        tensor = torch.tensor(
            [[token_id] for token_id in token_ids],
            dtype=torch.long,
            device=next(self._model.parameters()).device,
        )
        output, self._cache = self._model.decode_one_with_kv_cache(tensor, self._cache)
        self._logits = output.logits[:, -1].detach().float().cpu().tolist()
        self.tokens_processed += self._batch_size

    def close(self) -> None:
        if self._closed:
            return
        self._backend._release_generation_session()
        self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class S0TorchBatchedInferenceBackend(S0TorchInferenceBackend):
    """S0 raw-Base adapter exposing stateless and model-native batched inference."""

    def __init__(
        self,
        model: TwelveSixDecoder,
        tokenizer: ByteTokenizer,
        *,
        padding_token_id: int = 0,
    ) -> None:
        super().__init__(model, tokenizer)
        _validate_padding_token_id(model, padding_token_id)
        self.padding_token_id = padding_token_id
        self.cache_row_filler_token_id = padding_token_id
        self.last_batch_call_stats: TorchBatchCallStats | None = None

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]:
        values, stats = right_padded_next_token_logits(
            self.model,
            input_ids,
            padding_token_id=self.padding_token_id,
        )
        self.last_batch_call_stats = stats
        return values

    def begin_generation_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> S0TorchBatchedGenerationSession:
        """Open one fixed-row equal-length batch on the accepted model-native cache."""
        return S0TorchBatchedGenerationSession(self, input_ids)
