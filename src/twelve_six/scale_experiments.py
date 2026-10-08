"""Plan 7 / Section 1: versioned, bounded, non-promoting scale experiments.

Owns only experimental candidate construction, admission and comparable LOCAL_FREE
forward proxies. ModelSpec/InitSpec and canonical stage configs remain read-only.
This is not a training-run authorizer or a predictive GPU memory profiler.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder

SCHEMA_VERSION = 1
ALLOWED_VARIANTS = frozenset(
    {"baseline", "mlp_width", "depth", "gqa", "heads", "context"}
)


class ScaleAdmissionError(ValueError):
    """Candidate or experiment exceeds a fail-closed LOCAL_FREE budget."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProxyProtocol:
    """Identity of one fixed-control, synthetic-data comparison protocol."""

    version: int
    fixture_sha256: str
    seed: int
    batch: int
    sequence: int
    repeats: int = 2
    dtype: str = "float32"
    resource_class: str = "LOCAL_FREE"

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError("unsupported proxy protocol version")
        if len(self.fixture_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.fixture_sha256
        ):
            raise ValueError("fixture identity must be a lowercase SHA-256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("batch", "sequence", "repeats"):
            _positive_int(name, getattr(self, name))
        if self.dtype != "float32" or self.resource_class != "LOCAL_FREE":
            raise ScaleAdmissionError("only LOCAL_FREE CPU float32 proxies are supported")

    def identity(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ProxyBudget:
    max_parameters: int = 100_000
    max_flops: int = 200_000_000
    max_memory_bytes: int = 128_000_000
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_int(name, value)


@dataclass(frozen=True, slots=True)
class ArchitectureHypothesis:
    """Non-promoted ModelSpec delta; cannot mutate canonical architecture."""

    version: int
    hypothesis_id: str
    variant: str
    field: str
    value: int

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION or not self.hypothesis_id.strip():
            raise ValueError("invalid hypothesis identity/version")
        if self.variant not in ALLOWED_VARIANTS:
            raise ValueError("unsupported experimental variant")
        field_by_variant = {
            "baseline": None,
            "mlp_width": "d_ff",
            "depth": "n_layers",
            "gqa": "n_kv_heads",
            "heads": "n_heads",
            "context": "max_seq_len",
        }
        if self.field != field_by_variant[self.variant]:
            raise ValueError("variant/field mismatch")
        if self.variant != "baseline":
            _positive_int("hypothesis value", self.value)
        elif self.value != 0:
            raise ValueError("baseline must have value 0")

    def candidate(self, parent: ModelSpec) -> ModelSpec:
        if not isinstance(parent, ModelSpec):
            raise TypeError("parent must be a canonical ModelSpec")
        return parent if self.variant == "baseline" else replace(parent, **{self.field: self.value})


def estimate_proxy_resources(
    spec: ModelSpec, protocol: ProxyProtocol,
) -> dict[str, int]:
    """Conservative score-equivalent planning model; not observed RAM or GPU peak."""
    batch, seq = protocol.batch, protocol.sequence
    if seq > spec.max_seq_len:
        raise ScaleAdmissionError("sequence exceeds candidate context")
    q, kv, d, ff, layers = (
        spec.q_dim, spec.kv_dim, spec.d_model, spec.d_ff, spec.n_layers
    )
    tokens = batch * seq
    # 2 FLOPs per multiply-add; causal S^2 attention/softmax planning bound.
    forward_flops = (
        2 * tokens * layers * (d * (2 * q + 2 * kv) + 3 * d * ff)
        + 4 * batch * layers * spec.n_heads * seq * seq * spec.head_dim
        + 2 * tokens * d * spec.vocab_size
    )
    parameter_bytes = spec.parameter_count() * 4
    activation_bytes = 4 * (
        tokens * (d + spec.vocab_size + layers * (d + q + 2 * kv + 3 * ff))
        + batch * layers * spec.n_heads * seq * seq
    )
    return {
        "parameters": spec.parameter_count(),
        "parameter_bytes": parameter_bytes,
        "score_equivalent_activation_bytes": activation_bytes,
        "planning_memory_bytes": parameter_bytes + activation_bytes,
        "forward_flops": forward_flops,
        "total_proxy_flops": forward_flops * protocol.repeats,
        "total_proxy_tokens": tokens * protocol.repeats,
    }


def admit_proxy(
    spec: ModelSpec, protocol: ProxyProtocol, budget: ProxyBudget,
) -> dict[str, int]:
    resources = estimate_proxy_resources(spec, protocol)
    if resources["parameters"] > budget.max_parameters:
        raise ScaleAdmissionError("parameter budget exceeded")
    if resources["total_proxy_flops"] > budget.max_flops:
        raise ScaleAdmissionError("FLOP budget exceeded")
    if resources["planning_memory_bytes"] > budget.max_memory_bytes:
        raise ScaleAdmissionError("memory budget exceeded")
    if resources["total_proxy_tokens"] > budget.max_tokens:
        raise ScaleAdmissionError("token budget exceeded")
    return resources


def run_proxy(
    parent: ModelSpec,
    hypothesis: ArchitectureHypothesis,
    protocol: ProxyProtocol,
    budget: ProxyBudget,
    *,
    init: InitSpec | None = None,
) -> dict[str, Any]:
    """Execute an admitted tiny decoder on deterministic synthetic tokens only."""
    candidate = hypothesis.candidate(parent)
    resources = admit_proxy(candidate, protocol, budget)  # BEFORE allocation
    if not isinstance(parent, ModelSpec):
        raise TypeError("parent must be ModelSpec")
    init_spec = InitSpec() if init is None else init
    if not isinstance(init_spec, InitSpec):
        raise TypeError("init must be InitSpec")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.seed)
        model = TwelveSixDecoder(candidate, init_spec).cpu().eval()
        generator = torch.Generator(device="cpu").manual_seed(protocol.seed + 1)
        samples = torch.randint(
            0, candidate.vocab_size,
            (protocol.batch, protocol.sequence),
            generator=generator,
        )
        losses = []
        output_sha256 = []
        with torch.no_grad():
            for _ in range(protocol.repeats):
                logits = model(samples).logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("nonfinite proxy logits")
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, candidate.vocab_size),
                    samples[:, 1:].reshape(-1),
                ) if protocol.sequence > 1 else logits.square().mean()
                if not math.isfinite(float(loss)):
                    raise RuntimeError("nonfinite proxy loss")
                losses.append(float(loss))
                output_sha256.append(hashlib.sha256(
                    logits.contiguous().numpy().tobytes()
                ).hexdigest())
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _digest({
            "parent": parent.identity_sha256(),
            "hypothesis": asdict(hypothesis),
            "protocol": protocol.identity(),
            "init": init_spec.identity_sha256(),
        }),
        "hypothesis": asdict(hypothesis),
        "parent_modelspec_sha256": parent.identity_sha256(),
        "candidate_modelspec_sha256": candidate.identity_sha256(),
        "init_sha256": init_spec.identity_sha256(),
        "protocol_sha256": protocol.identity(),
        "resources": resources,
        "losses": losses,
        "output_sha256": output_sha256,
        "promotion_authorized": False,
        "training_executed": False,
        "paid_compute_authorized": False,
        "evidence_class": "LOCAL_FREE_SYNTHETIC_PROXY_ONLY",
    }


def compare_proxies(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Only same-protocol, same-parent/init records are comparable."""
    required = (
        "schema_version", "parent_modelspec_sha256", "init_sha256",
        "protocol_sha256", "resources", "losses", "evidence_class",
        "promotion_authorized", "training_executed",
    )
    if any(k not in left or k not in right for k in required):
        raise ValueError("incomplete experiment receipt")
    if any(
        left[k] != right[k] for k in (
            "schema_version", "parent_modelspec_sha256",
            "init_sha256", "protocol_sha256",
        )
    ) or left["schema_version"] != SCHEMA_VERSION:
        raise ValueError("non-comparable experiment protocol/identity")
    for receipt in (left, right):
        if (
            receipt["evidence_class"] != "LOCAL_FREE_SYNTHETIC_PROXY_ONLY"
            or receipt["promotion_authorized"] is not False
            or receipt["training_executed"] is not False
            or not receipt["losses"]
            or any(not math.isfinite(float(v)) for v in receipt["losses"])
        ):
            raise ValueError("invalid or promoted proxy receipt")
    return {
        "left_experiment_id": left.get("experiment_id"),
        "right_experiment_id": right.get("experiment_id"),
        "left_loss_mean": sum(left["losses"]) / len(left["losses"]),
        "right_loss_mean": sum(right["losses"]) / len(right["losses"]),
        "promotion_authorized": False,
        "interpretation": "synthetic mechanics comparison, not model quality",
    }
