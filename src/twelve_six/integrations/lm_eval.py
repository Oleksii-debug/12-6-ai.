"""Optional EleutherAI lm-evaluation-harness integration for verified 12-6 Base."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Protocol

from twelve_six.inference.contracts import GenerationConfig, InferenceBackend
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate

LM_EVAL_DISTRIBUTION = "lm-eval"
LM_EVAL_VERSION = "0.4.12"
LM_EVAL_WHEEL_SHA256 = "02971ff68284dd14cfa7fce9310a58452c4162e8d413ba96aa7988a0ff9352ef"
LM_EVAL_SOURCE_COMMIT = "6d642546f4688648fced259eb3302efd36ece5af"
NO_BOS_POLICY = "first_token_unscored"


class HarnessRequest(Protocol):
    args: tuple[Any, ...]


def component_manifest() -> dict[str, object]:
    """Return the exact external component identity and semantic boundary."""
    return {
        "schema": "12-6.external-component.lm-eval.v1",
        "distribution": LM_EVAL_DISTRIBUTION,
        "version": LM_EVAL_VERSION,
        "wheel_sha256": LM_EVAL_WHEEL_SHA256,
        "source_commit": LM_EVAL_SOURCE_COMMIT,
        "foreign_pretrained_weights": False,
        "model_backend_extra_required": False,
        "no_bos_policy": NO_BOS_POLICY,
        "authority": "OPTIONAL_EVALUATION_ENGINE_NOT_BASE_OR_PROMOTION",
    }


def require_lm_eval_version() -> str:
    """Require the exact reviewed lm-eval release before importing its runtime."""
    try:
        observed = metadata.version(LM_EVAL_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{LM_EVAL_DISTRIBUTION}=={LM_EVAL_VERSION} is required for this optional integration"
        ) from exc
    if observed != LM_EVAL_VERSION:
        raise RuntimeError(
            f"unsupported {LM_EVAL_DISTRIBUTION} version: {observed}; "
            f"expected exactly {LM_EVAL_VERSION}"
        )
    return observed


def _validate_backend(backend: InferenceBackend) -> None:
    if not isinstance(backend.max_context_tokens, int) or isinstance(
        backend.max_context_tokens, bool
    ):
        raise TypeError("backend max_context_tokens must be an integer")
    if backend.max_context_tokens < 2:
        raise ValueError("lm-eval integration requires max_context_tokens >= 2")


def _require_s0_byte_identity(backend: InferenceBackend) -> None:
    diagnostics = getattr(backend, "diagnostics", None)
    if not callable(diagnostics):
        raise ValueError("verified S0 lm-eval integration requires backend diagnostics")
    payload = diagnostics()
    if payload.get("backend") != "first_party_torch":
        raise ValueError("lm-eval integration requires verified first_party_torch backend")
    if payload.get("tokenizer_version") != "s0-byte-v1":
        raise ValueError("lm-eval integration currently supports s0-byte-v1 only")
    if payload.get("vocab_size") != 256:
        raise ValueError("s0-byte-v1 lm-eval integration requires vocabulary size 256")


def _normalize_until(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        if not raw:
            raise ValueError("lm-eval until string must not be empty")
        return (raw,)
    if not isinstance(raw, Sequence):
        raise TypeError("lm-eval until must be a string or sequence of strings")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError("lm-eval until must contain only strings")
        if not item:
            raise ValueError("lm-eval until strings must not be empty")
        values.append(item)
    return tuple(values)


def _request_args(request: HarnessRequest, expected: int) -> tuple[Any, ...]:
    args = request.args
    if not isinstance(args, tuple) or len(args) != expected:
        raise TypeError(f"lm-eval request must expose a {expected}-item args tuple")
    return args


class TwelveSixHarnessCore:
    """Dependency-light adapter logic shared with the optional lm-eval subclass."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        batch_size: int = 1,
        default_max_gen_toks: int = 64,
        require_s0_byte_identity: bool = True,
    ) -> None:
        _validate_backend(backend)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            not isinstance(default_max_gen_toks, int)
            or isinstance(default_max_gen_toks, bool)
            or default_max_gen_toks < 0
        ):
            raise ValueError("default_max_gen_toks must be a non-negative integer")
        if require_s0_byte_identity:
            _require_s0_byte_identity(backend)
        self.backend = backend
        self.batch_size = batch_size
        self.default_max_gen_toks = default_max_gen_toks

    @property
    def max_length(self) -> int:
        return self.backend.max_context_tokens

    @property
    def tokenizer_name(self) -> str:
        diagnostics = getattr(self.backend, "diagnostics", None)
        if callable(diagnostics):
            payload = diagnostics()
            version = payload.get("tokenizer_version")
            fingerprint = payload.get("tokenizer_vocab_sha256")
            if isinstance(version, str) and isinstance(fingerprint, str):
                return f"12-6:{version}:{fingerprint}"
        return f"12-6:{type(self.backend).__name__}"

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise TypeError("context and continuation must be strings")
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]
        context_ids = list(self.backend.encode(context))
        whole_ids = list(self.backend.encode(context + continuation))
        if len(context_ids) > len(whole_ids):
            raise ValueError("tokenizer produced an invalid context/continuation boundary")
        return context_ids, whole_ids[len(context_ids) :]

    def _score(
        self,
        context_ids: Sequence[int],
        continuation_ids: Sequence[int],
    ) -> tuple[float, bool]:
        context = list(context_ids)
        continuation = list(continuation_ids)
        if not continuation:
            return 0.0, True

        # Canonical 12-6 S0 has no BOS/EOS token. For an empty context we use the
        # first observed token only as conditioning context and deliberately do
        # not assign it an invented probability.
        if not context:
            if len(continuation) == 1:
                return 0.0, True
            context = [continuation.pop(0)]

        total = 0.0
        greedy = True
        prefix = list(context)
        for target in continuation:
            model_input = prefix[-self.backend.max_context_tokens :]
            logits = tuple(float(value) for value in self.backend.next_token_logits(model_input))
            if not logits:
                raise ValueError("backend returned an empty logits vector")
            if not 0 <= target < len(logits):
                raise ValueError(f"target token {target} is outside logits vocabulary")
            if not all(math.isfinite(value) for value in logits):
                raise ValueError("backend returned non-finite logits")
            peak = max(logits)
            logsumexp = peak + math.log(sum(math.exp(value - peak) for value in logits))
            total += logits[target] - logsumexp
            greedy = greedy and target == max(range(len(logits)), key=logits.__getitem__)
            prefix.append(target)
        return total, greedy

    def loglikelihood(self, requests: Sequence[HarnessRequest]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = _request_args(request, 2)
            if not isinstance(context, str) or not isinstance(continuation, str):
                raise TypeError("loglikelihood request arguments must be strings")
            context_ids, continuation_ids = self._encode_pair(context, continuation)
            results.append(self._score(context_ids, continuation_ids))
        return results

    def loglikelihood_rolling(self, requests: Sequence[HarnessRequest]) -> list[float]:
        results: list[float] = []
        for request in requests:
            (text,) = _request_args(request, 1)
            if not isinstance(text, str):
                raise TypeError("rolling loglikelihood request argument must be a string")
            token_ids = list(self.backend.encode(text))
            if len(token_ids) < 2:
                results.append(0.0)
                continue
            score, _ = self._score(token_ids[:1], token_ids[1:])
            results.append(score)
        return results

    def generate_until(self, requests: Sequence[HarnessRequest]) -> list[str]:
        results: list[str] = []
        for request in requests:
            context, raw_kwargs = _request_args(request, 2)
            if not isinstance(context, str) or not context:
                raise ValueError("12-6 raw Base generation requires a non-empty context")
            if not isinstance(raw_kwargs, dict):
                raise TypeError("generate_until kwargs must be a dict")
            kwargs = dict(raw_kwargs)
            allowed = {
                "until",
                "max_gen_toks",
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_k",
                "top_p",
                "seed",
            }
            unknown = sorted(set(kwargs) - allowed)
            if unknown:
                raise ValueError("unsupported lm-eval generation kwargs: " + ", ".join(unknown))

            max_gen_toks = kwargs.get("max_gen_toks")
            max_new_tokens = kwargs.get("max_new_tokens")
            if max_gen_toks is not None and max_new_tokens is not None:
                if max_gen_toks != max_new_tokens:
                    raise ValueError("max_gen_toks and max_new_tokens disagree")
            requested_max = (
                max_gen_toks
                if max_gen_toks is not None
                else max_new_tokens
                if max_new_tokens is not None
                else self.default_max_gen_toks
            )
            sample = kwargs.get("do_sample", False)
            if not isinstance(sample, bool):
                raise TypeError("do_sample must be a boolean")
            temperature = kwargs.get("temperature", 1.0) if sample else 1.0
            top_k = kwargs.get("top_k") if sample else None
            top_p = kwargs.get("top_p", 1.0) if sample else 1.0
            config = GenerationConfig(
                max_new_tokens=requested_max,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=kwargs.get("seed", 0),
                stop_strings=_normalize_until(kwargs.get("until")),
            )
            results.append(generate(self.backend, context, config).text)
        return results


def build_lm_eval_adapter(
    backend: InferenceBackend,
    *,
    batch_size: int = 1,
    default_max_gen_toks: int = 64,
    require_s0_byte_identity: bool = True,
) -> Any:
    """Build an lm-eval ``LM`` instance without adding a Base model backend dependency."""
    require_lm_eval_version()
    lm_model_module = import_module("lm_eval.api.model")
    lm_base = lm_model_module.LM
    core = TwelveSixHarnessCore(
        backend,
        batch_size=batch_size,
        default_max_gen_toks=default_max_gen_toks,
        require_s0_byte_identity=require_s0_byte_identity,
    )

    class TwelveSixLMEval(lm_base):
        def __init__(self) -> None:
            super().__init__()

        @property
        def batch_size(self) -> int:
            return core.batch_size

        @property
        def max_length(self) -> int:
            return core.max_length

        @property
        def tokenizer_name(self) -> str:
            return core.tokenizer_name

        def loglikelihood(
            self,
            requests: list[Any],
            disable_tqdm: bool = False,
        ) -> list[tuple[float, bool]]:
            del disable_tqdm
            output = core.loglikelihood(requests)
            for request, result in zip(requests, output, strict=True):
                self.cache_hook.add_partial("loglikelihood", request.args, result)
            return output

        def loglikelihood_rolling(
            self,
            requests: list[Any],
            disable_tqdm: bool = False,
        ) -> list[float]:
            del disable_tqdm
            output = core.loglikelihood_rolling(requests)
            for request, result in zip(requests, output, strict=True):
                self.cache_hook.add_partial("loglikelihood_rolling", request.args, result)
            return output

        def generate_until(
            self,
            requests: list[Any],
            disable_tqdm: bool = False,
        ) -> list[str]:
            del disable_tqdm
            output = core.generate_until(requests)
            for request, result in zip(requests, output, strict=True):
                self.cache_hook.add_partial("generate_until", request.args, result)
            return output

    TwelveSixLMEval.__name__ = "TwelveSixLMEval"
    return TwelveSixLMEval()


def build_lm_eval_adapter_from_checkpoint(
    checkpoint: str | Path,
    *,
    batch_size: int = 1,
    default_max_gen_toks: int = 64,
) -> Any:
    """Load verified first-party checkpoint bytes and expose them to lm-eval."""
    backend = load_first_party_backend(Path(checkpoint))
    return build_lm_eval_adapter(
        backend,
        batch_size=batch_size,
        default_max_gen_toks=default_max_gen_toks,
        require_s0_byte_identity=True,
    )


def simple_evaluate_checkpoint(
    checkpoint: str | Path,
    tasks: Sequence[str],
    *,
    limit: int | float | None = None,
    num_fewshot: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run lm-eval's maintained evaluator against one verified 12-6 checkpoint."""
    if not tasks:
        raise ValueError("at least one lm-eval task is required")
    adapter = build_lm_eval_adapter_from_checkpoint(checkpoint, batch_size=batch_size)
    lm_eval = import_module("lm_eval")
    kwargs: dict[str, Any] = {
        "model": adapter,
        "tasks": list(tasks),
    }
    if limit is not None:
        kwargs["limit"] = limit
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    result = lm_eval.simple_evaluate(**kwargs)
    if not isinstance(result, dict):
        raise TypeError("lm-eval simple_evaluate returned a non-dict result")
    return result
