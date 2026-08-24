"""Collision-safe adapters that compose accepted S0 lane contracts."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


class S0TorchInferenceBackend:
    """Adapt the D01 decoder + D04 tokenizer to the D07 inference protocol."""

    eos_token_id: int | None = None

    def __init__(self, model: TwelveSixDecoder, tokenizer: ByteTokenizer) -> None:
        if model.spec.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "model/tokenizer vocabulary mismatch: "
                f"model={model.spec.vocab_size} tokenizer={tokenizer.vocab_size}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_tokens = model.spec.max_seq_len

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, errors="replace")

    @torch.no_grad()
    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(input_ids) > self.max_context_tokens:
            raise ValueError("input_ids exceed model context")
        tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=next(self.model.parameters()).device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(tensor).logits[0, -1]
            return logits.detach().float().cpu().tolist()
        finally:
            self.model.train(was_training)
