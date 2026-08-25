from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder
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


class S0TorchBatchedInferenceBackend(S0TorchInferenceBackend):
    """S0 raw-Base adapter exposing a lower-level stateless batched forward API."""

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
