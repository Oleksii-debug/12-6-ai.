"""Plan 4 Section 5: fail-closed fixture-only real CPU INT8 backend parity.

Reuses TwelveSixDecoder forward/generate as the sole semantic authority.
This is NOT another weight/export authority or production backend promotion.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from twelve_six.model import TwelveSixDecoder
from tools.inference_runtime import InferenceError, _model_weights_sha

SCHEMA = "12-6.plan4-backend-parity.v1"
BACKEND = "torch-cpu-dynamic-int8-linear"
DENIAL = "DENIED_NO_PRODUCTION_CHECKPOINT_OR_NATIVE_EXTERNAL_BACKEND_PARITY"
STATUS = "QUALIFIED_FIXTURE_ONLY"
FEATURE_MATRIX = {
    "reference-torch-fp32": {
        "architecture": "12-6-ModelSpec-v1", "availability": "REFERENCE",
        "greedy": "SUPPORTED", "sample": "SUPPORTED",
        "stream": "SUPPORTED", "cancel": "SUPPORTED",
    },
    BACKEND: {
        "architecture": "12-6-ModelSpec-v1-untied-embeddings-only",
        "availability": STATUS, "greedy": "MEASURED",
        "sample": "NOT_QUALIFIED", "stream": "NOT_QUALIFIED",
        "cancel": "NOT_QUALIFIED",
    },
    "gguf-llama.cpp": {
        "architecture": "UNVERIFIED_CONVERTER_AND_CUSTOM_OPS",
        "availability": "UNSUPPORTED", "greedy": "UNSUPPORTED",
        "sample": "UNSUPPORTED", "stream": "UNSUPPORTED",
        "cancel": "UNSUPPORTED",
    },
    "vllm": {
        "architecture": "UNVERIFIED_ARCHITECTURE_ADAPTER_AND_RUNTIME",
        "availability": "UNSUPPORTED", "greedy": "UNSUPPORTED",
        "sample": "UNSUPPORTED", "stream": "UNSUPPORTED",
        "cancel": "UNSUPPORTED",
    },
}


class BackendParityError(ValueError):
    """Unqualified backend or parity evidence; never coerce to PASS."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_bounds(max_abs: float, max_ce: float, match: float) -> None:
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for x in (max_abs, max_ce, match)):
        raise BackendParityError("nonfinite/non-numeric tolerance")
    if not 0 < max_abs <= 0.05 or not 0 <= max_ce <= 0.05:
        raise BackendParityError("tolerance exceeds bounded policy")
    if match != 1.0:
        raise BackendParityError("fixture greedy token parity must be exact")


def _tensor_bytes(model: nn.Module) -> int:
    """In-memory tensor payload lower bound; NEVER process RSS."""
    total = sum(p.numel() * p.element_size()
                for p in list(model.parameters()) + list(model.buffers()))
    for module in model.modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            weight, bias = module.weight(), module.bias()
            total += weight.numel() * weight.element_size()
            if bias is not None:
                total += bias.numel() * bias.element_size()
    return total


def _candidate_identity(model: TwelveSixDecoder) -> str:
    h = hashlib.sha256(model.spec.identity_sha256().encode("ascii"))
    for name, tensor in sorted(model.named_parameters()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(_canonical({
            "name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
        }))
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    for name, module in sorted(model.named_modules()):
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            weight = module.weight()
            h.update(name.encode("utf-8") + b"\x00")
            h.update(weight.int_repr().contiguous().numpy().tobytes())
            h.update(_canonical({
                "scale": weight.q_scale(), "zero_point": weight.q_zero_point(),
            }))
            bias = module.bias()
            if bias is not None:
                h.update(bias.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _preflight(reference: TwelveSixDecoder, backend: str) -> str:
    if backend != BACKEND:
        raise BackendParityError("backend unsupported; no approximation")
    if not isinstance(reference, TwelveSixDecoder):
        raise BackendParityError("canonical TwelveSixDecoder required")
    if reference.training:
        raise BackendParityError("reference must be in eval mode")
    if reference.spec.tie_word_embeddings:
        raise BackendParityError("tied embeddings unsupported by INT8 adapter")
    if any(p.device.type != "cpu" or p.dtype != torch.float32
           for p in reference.parameters()):
        raise BackendParityError("only CPU fp32 reference supported")
    if not any(isinstance(m, nn.Linear) for m in reference.modules()):
        raise BackendParityError("no eligible linear operations")
    try:
        return _model_weights_sha(reference)
    except InferenceError as exc:
        raise BackendParityError("nonfinite or unsupported reference state") from exc


def _measure_ns(model: TwelveSixDecoder, tokens: torch.Tensor) -> int:
    with torch.inference_mode():
        model(tokens)
        times = []
        for _ in range(5):
            start = time.perf_counter_ns()
            logits = model(tokens).logits
            times.append(time.perf_counter_ns() - start)
            if not bool(torch.isfinite(logits).all()):
                raise BackendParityError("nonfinite measured logits")
    return int(statistics.median(times))


def qualify_cpu_dynamic_int8(
    reference: TwelveSixDecoder,
    *,
    backend: str = BACKEND,
    max_abs_logit_error: float = 0.05,
    max_cross_entropy_increase: float = 0.05,
    min_greedy_match: float = 1.0,
    probes: tuple[tuple[int, ...], ...] = ((65, 66), (67, 68, 69), (70, 71)),
) -> dict[str, Any]:
    """Run real dynamic quantized LINEAR ops against frozen synthetic fixtures."""
    _validate_bounds(max_abs_logit_error, max_cross_entropy_increase, min_greedy_match)
    source_sha = _preflight(reference, backend)
    if (type(probes) is not tuple or not probes or len(probes) > 16
            or any(type(seq) is not tuple or not 1 <= len(seq) <= 16
                   or any(type(i) is not int or not 0 <= i < reference.spec.vocab_size
                          for i in seq)
                   or len(seq) + 2 > reference.spec.max_seq_len for seq in probes)):
        raise BackendParityError("invalid bounded synthetic probe geometry")
    try:
        # Torch.ao is deprecated: this fixture adapter requires requalification
        # on torchao upgrade. Never publish optimized weights as canonical.
        candidate = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(reference), {nn.Linear}, dtype=torch.qint8, inplace=True,
        ).eval()
        n_linear = sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
                       for m in candidate.modules())
        if n_linear == 0 or any(isinstance(m, nn.Linear) for m in candidate.modules()):
            raise BackendParityError("candidate has unconverted linear operations")
        max_error, max_ce, matches, last = 0.0, -math.inf, 0, None
        with torch.inference_mode():
            for seq in probes:
                tokens = torch.tensor([seq], dtype=torch.long)
                last = tokens
                reference_logits = reference(tokens).logits[0, -1].double()
                candidate_logits = candidate(tokens).logits[0, -1].double()
                if (reference_logits.shape != candidate_logits.shape
                        or not bool(torch.isfinite(candidate_logits).all())):
                    raise BackendParityError("different-width/nonfinite logits")
                max_error = max(max_error, float(
                    (reference_logits - candidate_logits).abs().max()
                ))
                target = (seq[-1] + 1) % reference.spec.vocab_size
                ref_ce = float(torch.logsumexp(reference_logits, 0) - reference_logits[target])
                opt_ce = float(torch.logsumexp(candidate_logits, 0) - candidate_logits[target])
                max_ce = max(max_ce, opt_ce - ref_ce)
                matches += int(reference_logits.argmax() == candidate_logits.argmax())
                if not torch.equal(
                    reference.generate(tokens, max_new_tokens=2),
                    candidate.generate(tokens, max_new_tokens=2),
                ):
                    raise BackendParityError("canonical greedy-generation parity failed")
        match = matches / len(probes)
        if max_error > max_abs_logit_error or max_ce > max_cross_entropy_increase:
            raise BackendParityError("candidate outside fixed parity tolerance")
        if match < min_greedy_match:
            raise BackendParityError("greedy mismatch")
        if _model_weights_sha(reference) != source_sha:
            raise BackendParityError("reference weights changed")
        assert last is not None
        envelope = {
            "schema": SCHEMA, "status": STATUS, "backend": BACKEND,
            "production_promotion": DENIAL,
            "model_spec_sha256": reference.spec.identity_sha256(),
            "reference_weights_sha256": source_sha,
            "candidate_runtime_sha256": _candidate_identity(candidate),
            "torch_version": str(torch.__version__),
            "cpu": platform.machine(),
            "quantized_engine": str(torch.backends.quantized.engine),
            "probe_kind": "SYNTHETIC_FIXTURE_ONLY",
            "probe_sha256": _digest(probes), "probe_count": len(probes),
            "converted_linear_count": n_linear,
            "max_abs_logit_error": max_error,
            "max_cross_entropy_increase": max_ce,
            "greedy_match_rate": match,
            "reference_median_ns": _measure_ns(reference, last),
            "optimized_median_ns": _measure_ns(candidate, last),
            "reference_tensor_bytes": _tensor_bytes(reference),
            "optimized_tensor_bytes": _tensor_bytes(candidate),
            "resource_unit": "TENSOR_PAYLOAD_BYTES_NOT_RSS",
            "feature_matrix": copy.deepcopy(FEATURE_MATRIX),
        }
        return dict(envelope, report_sha256=_digest(envelope))
    except BackendParityError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        raise BackendParityError("int8 backend failed; cannot promote") from exc


def verify_report(report: Mapping[str, Any]) -> None:
    """Reject falsified claims; acceptance also requires exact executable replay."""
    if not isinstance(report, Mapping):
        raise BackendParityError("report must be a mapping")
    keys = {
        "schema", "status", "backend", "production_promotion", "model_spec_sha256",
        "reference_weights_sha256", "candidate_runtime_sha256", "torch_version",
        "cpu", "quantized_engine", "probe_kind", "probe_sha256", "probe_count",
        "converted_linear_count", "max_abs_logit_error", "max_cross_entropy_increase",
        "greedy_match_rate", "reference_median_ns", "optimized_median_ns",
        "reference_tensor_bytes", "optimized_tensor_bytes", "resource_unit",
        "feature_matrix", "report_sha256",
    }
    if set(report) != keys:
        raise BackendParityError("report shape invalid")
    if (report["schema"] != SCHEMA or report["status"] != STATUS
            or report["backend"] != BACKEND or report["production_promotion"] != DENIAL
            or report["probe_kind"] != "SYNTHETIC_FIXTURE_ONLY"
            or report["resource_unit"] != "TENSOR_PAYLOAD_BYTES_NOT_RSS"
            or report["feature_matrix"] != FEATURE_MATRIX):
        raise BackendParityError("unqualified parity or promotion claim")
    for name in ("model_spec_sha256", "reference_weights_sha256",
                 "candidate_runtime_sha256", "probe_sha256", "report_sha256"):
        digest = report[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise BackendParityError("noncanonical digest")
    for name in ("converted_linear_count", "probe_count", "reference_median_ns",
                 "optimized_median_ns", "reference_tensor_bytes", "optimized_tensor_bytes"):
        if type(report[name]) is not int or report[name] <= 0:
            raise BackendParityError("invalid resource/count metric")
    if report["probe_count"] > 16:
        raise BackendParityError("unbounded probe count")
    if any(not isinstance(report[n], str) or not report[n]
           for n in ("torch_version", "cpu", "quantized_engine")):
        raise BackendParityError("missing environment identity")
    _validate_bounds(0.05, 0.05, report["greedy_match_rate"])
    for n in ("max_abs_logit_error", "max_cross_entropy_increase"):
        v = report[n]
        if type(v) not in (float, int) or not math.isfinite(v) or v > 0.05:
            raise BackendParityError("invalid parity metric")
    if report["max_abs_logit_error"] < 0:
        raise BackendParityError("negative absolute error")
    unsigned = {k: v for k, v in report.items() if k != "report_sha256"}
    if _digest(unsigned) != report["report_sha256"]:
        raise BackendParityError("tampered parity report")
