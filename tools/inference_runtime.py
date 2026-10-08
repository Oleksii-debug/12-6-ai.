"""Plan 4 Section 3: one reference inference path, reusing TwelveSixDecoder.generate.

The model owns next-token semantics. This adapter owns tokenizer binding,
request identity, deterministic per-request RNG, stream/cancel and decoding.
It is LOCAL_FREE/reference backend, not a production serving process.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator, Literal

import torch

from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization.base import TokenizerProtocol

SCHEMA = "12-6.plan4-reference-inference.v1"


class InferenceError(ValueError):
    """Fail-closed reference inference error without private model internals."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise InferenceError("noncanonical inference identity") from exc


def _object_sha(value: object) -> str:
    return _sha(_json(value))


def _model_weights_sha(model: TwelveSixDecoder) -> str:
    """Physically bind every state tensor rather than trusting a caller-supplied label."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor) or value.is_sparse:
            raise InferenceError("unsupported model state")
        tensor = value.detach().to("cpu").contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise InferenceError("nonfinite model state")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = _json({"key": key, "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype), "bytes": len(raw)})
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int
    strategy: Literal["greedy", "sample"] = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    decode_errors: Literal["replace", "strict"] = "replace"

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or not 0 <= self.max_new_tokens <= 32768:
            raise InferenceError("invalid generation length")
        if self.strategy not in ("greedy", "sample"):
            raise InferenceError("unsupported sampling strategy")
        if (type(self.temperature) not in (float, int)
                or not math.isfinite(self.temperature) or self.temperature <= 0):
            raise InferenceError("invalid temperature")
        if self.decode_errors not in ("replace", "strict"):
            raise InferenceError("unsupported decode error policy")
        if self.strategy == "greedy":
            if self.temperature != 1.0 or self.top_k is not None or self.seed is not None:
                raise InferenceError("greedy mode forbids ignored sampling parameters")
        else:
            if type(self.seed) is not int or not 0 <= self.seed < 2**63:
                raise InferenceError("sampling requires a bounded deterministic seed")
            if self.top_k is not None and (
                type(self.top_k) is not int or self.top_k <= 0
            ):
                raise InferenceError("invalid top_k")

    def to_dict(self) -> dict[str, object]:
        return {"max_new_tokens": self.max_new_tokens, "strategy": self.strategy,
                "temperature": self.temperature, "top_k": self.top_k,
                "seed": self.seed, "decode_errors": self.decode_errors}


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    result_sha256: str
    model_spec_sha256: str
    model_weights_sha256: str
    tokenizer_sha256: str
    config_sha256: str
    prompt_sha256: str
    status: Literal["RUNNING", "COMPLETED", "CANCELLED"]
    prompt_token_count: int
    output_token_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    kind: Literal["TOKEN", "COMPLETED", "CANCELLED"]
    sequence: int
    token_id: int | None
    result: GenerationResult


class ReferenceInference:
    """Bound reference-model/tokenizer authority; no second next-token implementation."""

    def __init__(self, model: TwelveSixDecoder, tokenizer: TokenizerProtocol) -> None:
        if not isinstance(model, TwelveSixDecoder):
            raise InferenceError("TwelveSixDecoder reference model required")
        if (not isinstance(getattr(tokenizer, "vocab_size", None), int)
                or tokenizer.vocab_size != model.spec.vocab_size):
            raise InferenceError("tokenizer/model vocabulary incompatible")
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.tokenizer_sha256 = _object_sha(tokenizer.identity.to_dict())
            self.model_spec_sha256 = model.spec.identity_sha256()
            self.model_weights_sha256 = _model_weights_sha(model)
        except (AttributeError, KeyError, TypeError) as exc:
            raise InferenceError("invalid model/tokenizer identity") from exc

    def start(self, prompt: str, config: GenerationConfig) -> InferenceSession:
        if not isinstance(config, GenerationConfig) or type(prompt) is not str or not prompt:
            raise InferenceError("nonempty prompt and frozen generation config required")
        if self.model.training:
            raise InferenceError("reference model must be in eval mode")
        if (_model_weights_sha(self.model) != self.model_weights_sha256
                or _object_sha(self.tokenizer.identity.to_dict()) != self.tokenizer_sha256):
            raise InferenceError("model/tokenizer identity drift")
        if config.strategy == "sample" and config.top_k is not None:
            if config.top_k > self.model.spec.vocab_size:
                raise InferenceError("top_k exceeds vocabulary")
        try:
            tokens = self.tokenizer.encode(prompt)
        except (ValueError, UnicodeError, TypeError):
            raise InferenceError("prompt tokenization failed") from None
        if (type(tokens) is not list or not tokens or any(type(t) is not int
                or t < 0 or t >= self.model.spec.vocab_size for t in tokens)):
            raise InferenceError("invalid prompt token IDs")
        if len(tokens) + config.max_new_tokens > self.model.spec.max_seq_len:
            raise InferenceError("request exceeds reference context capacity")
        return InferenceSession(self, tuple(tokens), prompt, config)

    def stream(self, prompt: str, config: GenerationConfig) -> InferenceSession:
        """Return cancellable single-use iterator with one terminal status event."""
        return self.start(prompt, config)

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Drain the very same streaming state machine used by reference streaming."""
        return self.start(prompt, config).result()


class InferenceSession:
    def __init__(
        self, runtime: ReferenceInference, prompt_ids: tuple[int, ...],
        prompt: str, config: GenerationConfig,
    ) -> None:
        self.runtime = runtime
        self.prompt_ids = prompt_ids
        self.config = config
        self.config_sha256 = _object_sha(config.to_dict())
        self.prompt_sha256 = _sha(prompt.encode("utf-8"))
        self.request_id = _object_sha({
            "schema": SCHEMA, "model_spec_sha256": runtime.model_spec_sha256,
            "model_weights_sha256": runtime.model_weights_sha256,
            "tokenizer_sha256": runtime.tokenizer_sha256,
            "config_sha256": self.config_sha256, "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": list(prompt_ids),
        })
        self._cancel_requested = False
        self._started = False
        self._terminal: GenerationResult | None = None
        self._generated: list[int] = []
        self._input = torch.tensor([list(prompt_ids)], dtype=torch.long,
                                   device=next(runtime.model.parameters()).device)
        self._generator: torch.Generator | None = None
        if config.strategy == "sample":
            self._generator = torch.Generator(device=self._input.device)
            self._generator.manual_seed(config.seed)

    def cancel(self) -> None:
        if self._terminal is not None:
            raise InferenceError("terminal generation cannot be cancelled")
        self._cancel_requested = True

    def _snapshot(self, status: Literal["RUNNING", "COMPLETED", "CANCELLED"]) -> GenerationResult:
        try:
            text = self.runtime.tokenizer.decode(
                self._generated, errors=self.config.decode_errors,
            )
        except (ValueError, UnicodeError, TypeError):
            raise InferenceError("reference decode failed") from None
        content = {
            "schema": SCHEMA, "request_id": self.request_id,
            "model_spec_sha256": self.runtime.model_spec_sha256,
            "model_weights_sha256": self.runtime.model_weights_sha256,
            "tokenizer_sha256": self.runtime.tokenizer_sha256,
            "config_sha256": self.config_sha256,
            "prompt_sha256": self.prompt_sha256,
            "status": status, "prompt_token_count": len(self.prompt_ids),
            "output_token_ids": self._generated, "text": text,
        }
        return GenerationResult(
            request_id=self.request_id, result_sha256=_object_sha(content),
            model_spec_sha256=self.runtime.model_spec_sha256,
            model_weights_sha256=self.runtime.model_weights_sha256,
            tokenizer_sha256=self.runtime.tokenizer_sha256,
            config_sha256=self.config_sha256, prompt_sha256=self.prompt_sha256,
            status=status, prompt_token_count=len(self.prompt_ids),
            output_token_ids=tuple(self._generated), text=text,
        )

    def __iter__(self) -> Iterator[GenerationEvent]:
        if self._started:
            raise InferenceError("stream session is single-use")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[GenerationEvent]:
        for _ in range(self.config.max_new_tokens):
            if self._cancel_requested:
                break
            try:
                produced = self.runtime.model.generate(
                    self._input, max_new_tokens=1,
                    do_sample=self.config.strategy == "sample",
                    temperature=self.config.temperature, top_k=self.config.top_k,
                    generator=self._generator,
                )
            except Exception:
                raise InferenceError("reference model generation failed") from None
            if (produced.ndim != 2 or produced.shape != (1, self._input.shape[1] + 1)
                    or not torch.equal(produced[:, :-1], self._input)):
                raise InferenceError("reference generator violated causal prefix contract")
            token = int(produced[0, -1].item())
            if not 0 <= token < self.runtime.model.spec.vocab_size:
                raise InferenceError("reference generator emitted out-of-vocabulary ID")
            self._input = produced
            self._generated.append(token)
            yield GenerationEvent("TOKEN", len(self._generated), token,
                                  self._snapshot("RUNNING"))
        status = "CANCELLED" if self._cancel_requested else "COMPLETED"
        self._terminal = self._snapshot(status)
        yield GenerationEvent(status, len(self._generated), None, self._terminal)

    def result(self) -> GenerationResult:
        if self._terminal is not None:
            return self._terminal
        if self._started:
            raise InferenceError("cannot report incomplete streaming session")
        for _ in self:
            pass
        assert self._terminal is not None
        return self._terminal
