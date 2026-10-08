"""Plan 4 S5: fail-closed LOCAL_FREE optimized-backend qualification.

This module never replaces ReferenceInference or TwelveSixDecoder.generate as the
canonical decode authority. A quantized candidate is derived from a frozen CPU
reference model; promotion is only a scoped, evidence-bound capability decision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from tools.inference_runtime import (
    GenerationConfig,
    InferenceError,
    ReferenceInference,
    _model_weights_sha,
)

SCHEMA = "12-6.plan4-backend-parity.v1"
CANDIDATE = "torch-dynamic-int8-cpu-v1"
DENIED = frozenset({"gguf", "llama.cpp", "vllm", "onnx", "cuda", "unknown"})


class BackendQualificationError(ValueError):
    """Unsupported backend or unsafe qualification state."""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParityPolicy:
    max_logit_abs_delta: float = 0.20
    max_logit_mean_delta: float = 0.05
    require_greedy_token_parity: bool = True

    def __post_init__(self) -> None:
        for value in (self.max_logit_abs_delta, self.max_logit_mean_delta):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise BackendQualificationError("nonfinite or negative parity tolerance")
        if type(self.require_greedy_token_parity) is not bool:
            raise BackendQualificationError("token parity policy must be boolean")


@dataclass(frozen=True, slots=True)
class BackendEvidence:
    schema: str
    backend: str
    reference_model_sha256: str
    model_spec_sha256: str
    tokenizer_sha256: str
    torch_version: str
    backend_identity_sha256: str
    prompt_sha256: str
    config_sha256: str
    policy_sha256: str
    reference_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    logit_max_abs_delta: float
    logit_mean_abs_delta: float
    reference_seconds: float
    candidate_seconds: float
    reference_state_bytes: int
    candidate_state_bytes_estimate: int
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feature_matrix() -> dict[str, dict[str, object]]:
    """Explicit versioned capability boundary: unverified vendors are never usable."""
    return {
        "reference-torch-fp32": {
            "supported": True, "quantized": False, "device": "cpu",
            "greedy": True, "sample": True, "authority": "ReferenceInference",
        },
        CANDIDATE: {
            "supported": True, "quantized": True, "device": "cpu",
            "greedy": True, "sample": False, "authority": "QUALIFICATION_ONLY",
        },
        **{name: {"supported": False, "quantized": None, "device": None,
                  "greedy": False, "sample": False,
                  "authority": "UNQUALIFIED_NO_APPROXIMATION"} for name in sorted(DENIED)},
    }


def require_backend(name: str, config: GenerationConfig) -> None:
    if not isinstance(config, GenerationConfig):
        raise BackendQualificationError("frozen generation configuration required")
    entry = feature_matrix().get(name)
    if entry is None or not entry["supported"]:
        raise BackendQualificationError("backend/architecture is not qualified")
    if config.strategy != "greedy" and not entry["sample"]:
        raise BackendQualificationError("sampling unsupported for this optimized backend")


def _bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )


def _candidate_bytes(candidate: nn.Module) -> int:
    # Logical weights only, not peak RSS or a measured deployment-memory claim.
    total = _bytes(candidate)
    for layer in candidate.modules():
        if isinstance(layer, torch.ao.nn.quantized.dynamic.Linear):
            packed = layer.weight()
            total += packed.numel() * packed.element_size()
    return total


def _derive_candidate(runtime: ReferenceInference) -> nn.Module:
    model = runtime.model
    if (model.training or next(model.parameters()).device.type != "cpu"
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise BackendQualificationError("candidate requires frozen eval-mode CPU float32")
    if _model_weights_sha(model) != runtime.model_weights_sha256:
        raise BackendQualificationError("reference model identity drift")
    try:
        candidate = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8, inplace=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BackendQualificationError("dynamic int8 conversion unavailable") from exc
    if not any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
               for m in candidate.modules()):
        raise BackendQualificationError("candidate has no verified int8 linear layers")
    candidate.eval()
    return candidate


def qualify_backend(
    runtime: ReferenceInference, prompt: str, config: GenerationConfig,
    *, backend: str = CANDIDATE, policy: ParityPolicy | None = None,
) -> BackendEvidence:
    """Compare actual CPU int8 logits/tokens against the frozen reference path.

    Timings are observations, NOT proof of sustained throughput or speedup.
    A failed check returns accepted=False; unsupported semantics fail closed.
    """
    require_backend(backend, config)
    if backend != CANDIDATE:
        raise BackendQualificationError("reference needs no optimized promotion")
    if type(prompt) is not str or not prompt:
        raise BackendQualificationError("nonempty exact prompt required")
    policy = policy if policy is not None else ParityPolicy()
    if not isinstance(policy, ParityPolicy):
        raise BackendQualificationError("invalid parity policy")
    # Enforce the incumbent tokenizer/context/mode/drift admission path first.
    runtime.start(prompt, config)
    model = runtime.model
    candidate = _derive_candidate(runtime)
    ids = runtime.tokenizer.encode(prompt)
    tensor = torch.tensor([ids], dtype=torch.long)
    reference_before = runtime.model_weights_sha256
    with torch.no_grad():
        start = time.perf_counter()
        reference_logits = model(tensor).logits.detach().float()
        ref = runtime.generate(prompt, config)
        reference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        candidate_logits = candidate(tensor).logits.detach().float()
        candidate_generated = candidate.generate(tensor, max_new_tokens=config.max_new_tokens)
        candidate_seconds = time.perf_counter() - start
    if not bool(torch.isfinite(reference_logits).all()) or not bool(
        torch.isfinite(candidate_logits).all()
    ):
        raise BackendQualificationError("nonfinite reference or candidate logits")
    if candidate_logits.shape != reference_logits.shape:
        raise BackendQualificationError("optimized backend changes output geometry")
    candidate_ids = tuple(int(x) for x in candidate_generated[0, len(ids):].tolist())
    if _model_weights_sha(model) != reference_before:
        raise BackendQualificationError("reference drift during qualification")
    delta = (reference_logits - candidate_logits).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    if not all(math.isfinite(x) for x in (max_abs, mean_abs,
                                          reference_seconds, candidate_seconds)):
        raise BackendQualificationError("nonfinite diagnostic evidence")
    # Bind to the reference weights plus deterministic conversion recipe; no
    # independent arbitrary int8 checkpoint or vendor/backend promise.
    candidate_identity = _digest({
        "schema": SCHEMA, "backend": CANDIDATE, "torch": torch.__version__,
        "reference": reference_before, "spec": runtime.model_spec_sha256,
        "conversion": "torch.ao.quantization.quantize_dynamic.Linear.qint8.cpu",
    })
    return BackendEvidence(
        schema=SCHEMA, backend=CANDIDATE, reference_model_sha256=reference_before,
        model_spec_sha256=runtime.model_spec_sha256,
        tokenizer_sha256=runtime.tokenizer_sha256,
        torch_version=str(torch.__version__), backend_identity_sha256=candidate_identity,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        config_sha256=ref.config_sha256, policy_sha256=_digest(asdict(policy)),
        reference_token_ids=ref.output_token_ids, candidate_token_ids=candidate_ids,
        logit_max_abs_delta=max_abs, logit_mean_abs_delta=mean_abs,
        reference_seconds=reference_seconds, candidate_seconds=candidate_seconds,
        reference_state_bytes=_bytes(model),
        candidate_state_bytes_estimate=_candidate_bytes(candidate),
        accepted=(
            max_abs <= policy.max_logit_abs_delta
            and mean_abs <= policy.max_logit_mean_delta
            and (not policy.require_greedy_token_parity
                 or ref.output_token_ids == candidate_ids)
        ),
    )
