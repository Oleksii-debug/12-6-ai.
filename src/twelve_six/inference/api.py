"""Generic first-party inference API for local 12-6 Base decoders.

This module is deliberately completion-only. It owns no chat roles, prompt
formatting, tool policy, second decoder, trainer, or remote inference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

from .contracts import GenerationConfig, GenerationResult
from .first_party import FirstPartyInferenceBackend, load_first_party_backend
from .generation import CacheMode, generate, generate_token_ids


class FirstPartyInference:
    """One first-party decoder exposed through text and token-ID generation APIs."""

    def __init__(
        self,
        backend: S0TorchInferenceBackend,
        *,
        source_kind: str,
        source_path: Path | None = None,
        init_seed: int | None = None,
    ) -> None:
        if not isinstance(backend, S0TorchInferenceBackend):
            raise TypeError("FirstPartyInference requires the maintained torch backend")
        if not isinstance(backend.model, TwelveSixDecoder):
            raise TypeError("FirstPartyInference requires the maintained TwelveSixDecoder")
        if not isinstance(backend.tokenizer, ByteTokenizer):
            raise TypeError("FirstPartyInference requires the bound canonical ByteTokenizer")
        self._backend = backend
        self._source_kind = source_kind
        self._source_path = source_path
        self._init_seed = init_seed

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> FirstPartyInference:
        """Load one D05-verified learned checkpoint without changing decoder semantics."""

        path = Path(checkpoint)
        return cls(
            load_first_party_backend(path),
            source_kind="checkpoint",
            source_path=path,
        )

    @classmethod
    def from_random_init_stage(
        cls,
        stage_config: str | Path,
        *,
        seed: int = 0,
    ) -> FirstPartyInference:
        """Construct mechanics-only random-init inference from one exact StageConfig.

        The torch RNG is forked so constructing the mechanics fixture does not mutate
        the caller's global CPU RNG state. The same maintained ``TwelveSixDecoder``
        is used for random-init and learned checkpoints.
        """

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("random-init seed must be an integer")
        path = Path(stage_config)
        config = load_stage_config(path)
        tokenizer = ByteTokenizer()
        if config.model.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "random-init ModelSpec vocabulary does not match canonical tokenizer: "
                f"model={config.model.vocab_size} tokenizer={tokenizer.vocab_size}"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TwelveSixDecoder(config.model, config.init)
        model.eval()
        return cls(
            S0TorchInferenceBackend(model, tokenizer),
            source_kind="random_init",
            source_path=path,
            init_seed=seed,
        )

    @property
    def backend(self) -> S0TorchInferenceBackend:
        """Return the maintained backend; no alternate decoder is constructed."""

        return self._backend

    @property
    def model_spec(self) -> ModelSpec:
        return self._backend.model.spec

    def _validate_input_token_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        values: list[int] = []
        vocab_size = self.model_spec.vocab_size
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("prompt token IDs must contain integers")
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"prompt token ID {token_id} is outside vocabulary [0, {vocab_size})"
                )
            values.append(token_id)
        if not values:
            raise ValueError("prompt token IDs must be non-empty")
        return tuple(values)

    def generate_text(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        cache_mode: CacheMode = "static",
    ) -> GenerationResult:
        """Encode text with the tokenizer bound to this model, then generate."""

        with torch.inference_mode():
            return generate(self._backend, prompt, config, cache_mode=cache_mode)

    def generate_token_ids(
        self,
        prompt_token_ids: Sequence[int],
        config: GenerationConfig | None = None,
        *,
        cache_mode: CacheMode = "static",
    ) -> GenerationResult:
        """Generate directly from validated raw token IDs without text re-encoding."""

        values = self._validate_input_token_ids(prompt_token_ids)
        with torch.inference_mode():
            return generate_token_ids(
                self._backend,
                values,
                config,
                cache_mode=cache_mode,
            )

    def diagnostics(self) -> dict[str, object]:
        """Return local provenance without adding serving or chat semantics."""

        if isinstance(self._backend, FirstPartyInferenceBackend):
            payload = self._backend.diagnostics()
            payload["source_kind"] = self._source_kind
            return payload

        tokenizer = self._backend.tokenizer.identity
        return {
            "backend": "first_party_torch",
            "source_kind": self._source_kind,
            "source_path": str(self._source_path) if self._source_path is not None else None,
            "random_init_seed": self._init_seed,
            "model_spec_sha256": self.model_spec.identity_sha256(),
            "init_spec_sha256": self._backend.model.init_spec.identity_sha256(),
            "parameter_count": self.model_spec.parameter_count(),
            "vocab_size": self.model_spec.vocab_size,
            "max_context_tokens": self._backend.max_context_tokens,
            "tokenizer_version": tokenizer.version,
            "tokenizer_config_sha256": tokenizer.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.vocab_sha256,
            "device": str(next(self._backend.model.parameters()).device),
        }


def load_first_party_inference(checkpoint: str | Path) -> FirstPartyInference:
    """Convenience library entrypoint for a learned first-party checkpoint."""

    return FirstPartyInference.from_checkpoint(checkpoint)


def load_random_init_inference(
    stage_config: str | Path,
    *,
    seed: int = 0,
) -> FirstPartyInference:
    """Convenience entrypoint for bounded random-init mechanics."""

    return FirstPartyInference.from_random_init_stage(stage_config, seed=seed)
