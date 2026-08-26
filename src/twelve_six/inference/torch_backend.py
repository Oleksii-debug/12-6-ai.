from __future__ import annotations

from collections.abc import Sequence

import torch

from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


class TorchInferenceBackend:
    """ModelSpec-driven first-party PyTorch inference backend.

    This backend is intentionally stage-neutral. It adapts any compatible
    ``TwelveSixDecoder`` plus tokenizer to the generic inference contract.
    """

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
        for token_id in input_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("input token IDs must be integers")
            if not 0 <= token_id < self.tokenizer.vocab_size:
                raise ValueError(
                    f"input token ID {token_id} is outside vocabulary "
                    f"[0, {self.tokenizer.vocab_size})"
                )

        device = next(self.model.parameters()).device
        tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(tensor).logits[0, -1]
            return logits.detach().float().cpu().tolist()
        finally:
            self.model.train(was_training)
