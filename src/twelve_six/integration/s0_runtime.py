"""Collision-safe adapters that compose accepted S0 lane contracts."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from twelve_six.model import DecoderKVCache, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


class S0TorchGenerationSession:
    """One ephemeral, model-native KV-cache session for incremental generation."""

    def __init__(
        self,
        backend: S0TorchInferenceBackend,
        input_ids: Sequence[int],
    ) -> None:
        backend._validate_token_ids(input_ids)
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(input_ids) > backend.max_context_tokens:
            raise ValueError("input_ids exceed model context")

        self._backend = backend
        self._model = backend.model
        self._was_training = self._model.training
        self._closed = False
        self.tokens_processed = len(input_ids)
        self._cache: DecoderKVCache
        self._logits: list[float]

        self._model.eval()
        tensor = torch.tensor(
            [list(input_ids)],
            dtype=torch.long,
            device=next(self._model.parameters()).device,
        )
        try:
            output, self._cache = self._model.prefill_kv_cache(tensor)
            self._logits = output.logits[0, -1].detach().float().cpu().tolist()
        except (RuntimeError, TypeError, ValueError):
            self._model.train(self._was_training)
            raise

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
            raise RuntimeError("generation session is closed")

    def next_token_logits(self) -> Sequence[float]:
        self._require_open()
        return list(self._logits)

    def append(self, token_id: int) -> None:
        self._require_open()
        self._backend._validate_token_ids((token_id,))
        if self._cache.sequence_length >= self._backend.max_context_tokens:
            raise ValueError("KV cache is already at model context limit")
        tensor = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=next(self._model.parameters()).device,
        )
        output, self._cache = self._model.decode_one_with_kv_cache(tensor, self._cache)
        self._logits = output.logits[0, -1].detach().float().cpu().tolist()
        self.tokens_processed += 1

    def close(self) -> None:
        if self._closed:
            return
        self._model.train(self._was_training)
        self._closed = True

    def __enter__(self) -> S0TorchGenerationSession:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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

    def _validate_token_ids(self, input_ids: Sequence[int]) -> None:
        for token_id in input_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("input token IDs must be integers")
            if not 0 <= token_id < self.tokenizer.vocab_size:
                raise ValueError(
                    f"input token ID {token_id} is outside vocabulary "
                    f"[0, {self.tokenizer.vocab_size})"
                )

    def begin_generation(self, input_ids: Sequence[int]) -> S0TorchGenerationSession:
        """Create an inference-only incremental session without changing D07 semantics."""
        return S0TorchGenerationSession(self, input_ids)

    @torch.no_grad()
    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(input_ids) > self.max_context_tokens:
            raise ValueError("input_ids exceed model context")
        self._validate_token_ids(input_ids)
        tensor = torch.tensor(
            [list(input_ids)],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(tensor).logits[0, -1]
            return logits.detach().float().cpu().tolist()
        finally:
            self.model.train(was_training)
