"""Plan 7 Section 5: bounded ~200M and conditional 300–500M design packets.

All scale-size models are arithmetic ModelSpec recipes only. LOCAL_FREE probes
reuse Plan-7 Section-4's already accepted tiny analog authority. Neither a
GO decision nor a receipt grants training, deployment, or paid compute.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.model import ModelSpec
from twelve_six.optional_risk_probes import digest, tiny_analog

# width, depth, FFN width; keep the accepted v1 decoder and GQA shape semantics.
PROFILES = {
    "200M": (1024, 16, 3072),
    "300M": (1280, 16, 3840),
    "500M": (1536, 18, 4608),
}
PROXY_TIERS = {"200M": "35M", "300M": "50M", "500M": "100M"}


class ScaleRecipeDenied(ValueError):
    """Unsafe, malformed, or unsupported scale proposal."""


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScaleRecipeDenied(f"{name}: positive integer required")


def _nonnegative(name: str, value: float) -> None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ScaleRecipeDenied(f"{name}: finite nonnegative number required")


def _bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ScaleRecipeDenied(f"{name}: boolean required")


def scale_spec(tier: str) -> ModelSpec:
    if not isinstance(tier, str) or tier not in PROFILES:
        raise ScaleRecipeDenied("unsupported scale tier")
    width, layers, ff = PROFILES[tier]
    heads = width // 64
    return ModelSpec(
        schema_version=1, vocab_size=8192, max_seq_len=1024,
        d_model=width, n_layers=layers, n_heads=heads,
        n_kv_heads=heads // 4, head_dim=64, d_ff=ff,
        rope_rotary_dim=64,
    )


def scale_recipe(tier: str) -> dict[str, Any]:
    """Deterministic planning data, not measured hardware or a trained model."""
    model = scale_spec(tier)
    parameters = model.parameter_count()
    target = int(tier[:-1]) * 1_000_000
    if abs(parameters - target) > target // 20:
        raise ScaleRecipeDenied("parameter target tolerance exceeded")
    weights = 2 * parameters  # planned BF16 weights; not validated backend support
    working = 24 * parameters  # conservative planning floor, not peak RAM proof
    record = {
        "schema_version": 1, "tier": tier,
        "role": "PRIMARY" if tier == "200M" else "OPTIONAL_EVIDENCE_BRIDGE",
        "modelspec": model.to_dict(),
        "modelspec_sha256": model.identity_sha256(),
        "parameters": parameters, "target_parameters": target,
        "weights_bf16_bytes": weights,
        "train_state_planning_floor_bytes": working,
        "checkpoint_min_storage_bytes": 2 * weights,
        "tokens_per_step_template": 1024,
        "checkpoint_contract": "MODELSPEC_V1_EXACT_STATE_AND_PARENT_SHA256",
        "export_contract": "PLAN4_VERSIONED_SAFE_EXPORT_PRODUCER_REQUIRED",
        "evaluation_contract": "PLAN4_ISOLATED_EVALUATOR_PRODUCER_REQUIRED",
        "migration": "S2_GROWTH_PROOF_OR_FRESH_INIT_NEW_RUN",
        "topology": "EXACT_VERSIONED_BACKEND_AND_SHARD_RECEIPTS_REQUIRED",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
        "mandatory_bridge": False,
        "measurement_class": "ARITHMETIC_ONLY_NOT_PEAK_OR_THROUGHPUT",
    }
    record["recipe_sha256"] = digest(record)
    return record


@dataclass(frozen=True, slots=True)
class CapacityEvidence:
    schema_version: int
    measured_proxy_receipt_sha256: str
    available_memory_bytes_per_worker: int
    available_checkpoint_bytes: int
    healthy_workers: int
    measured_tokens_per_second: float
    measured_checkpoint_bytes_per_second: float
    measured_interconnect_bytes_per_second: float
    unique_training_tokens: int
    terminal_20m_proven: bool
    terminal_200m_proven: bool
    measured_bridge_risk_reduction: float
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ScaleRecipeDenied("unsupported evidence schema")
        if (not isinstance(self.measured_proxy_receipt_sha256, str)
                or len(self.measured_proxy_receipt_sha256) != 64
                or any(c not in "0123456789abcdef"
                       for c in self.measured_proxy_receipt_sha256)):
            raise ScaleRecipeDenied("invalid proxy receipt digest")
        for key in ("available_memory_bytes_per_worker", "available_checkpoint_bytes",
                    "healthy_workers", "unique_training_tokens"):
            _positive_int(key, getattr(self, key))
        for key in ("measured_tokens_per_second", "measured_checkpoint_bytes_per_second",
                    "measured_interconnect_bytes_per_second", "measured_bridge_risk_reduction"):
            _nonnegative(key, getattr(self, key))
        for key in ("terminal_20m_proven", "terminal_200m_proven", "paid_compute_requested"):
            _bool(key, getattr(self, key))
        if self.measured_bridge_risk_reduction > 1:
            raise ScaleRecipeDenied("risk reduction must be <= 1")


@dataclass(frozen=True, slots=True)
class ScaleLimits:
    max_total_memory_bytes: int
    max_checkpoint_bytes: int
    max_workers: int
    max_seconds: float
    min_tokens_per_second: float
    min_bridge_risk_reduction: float = 0.1

    def __post_init__(self) -> None:
        for key in ("max_total_memory_bytes", "max_checkpoint_bytes", "max_workers"):
            _positive_int(key, getattr(self, key))
        for key in ("max_seconds", "min_tokens_per_second", "min_bridge_risk_reduction"):
            _nonnegative(key, getattr(self, key))
        if self.min_bridge_risk_reduction > 1:
            raise ScaleRecipeDenied("risk threshold must be <= 1")


def qualify_proxy(tier: str, protocol: Any, budget: Any) -> dict[str, Any]:
    """Reuse the already accepted S4 CPU proxy, never build full-scale weights."""
    scale_recipe(tier)
    proxy = tiny_analog(PROXY_TIERS[tier], protocol, budget)
    result = {
        "schema_version": 1, "tier": tier,
        "recipe_sha256": scale_recipe(tier)["recipe_sha256"],
        "s4_proxy_receipt_sha256": digest(proxy),
        "proxy": proxy, "full_scale_executed": False,
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False,
    }
    result["receipt_sha256"] = digest(result)
    return result


def assess_scale(tier: str, evidence: CapacityEvidence, limits: ScaleLimits,
                 proxy: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on forged receipts and resource deficits; advice, never launch."""
    if not isinstance(evidence, CapacityEvidence) or not isinstance(limits, ScaleLimits):
        raise ScaleRecipeDenied("typed evidence and limits required")
    CapacityEvidence(**asdict(evidence))
    ScaleLimits(**asdict(limits))
    recipe = scale_recipe(tier)
    if (not isinstance(proxy, dict) or proxy.get("schema_version") != 1
            or type(proxy.get("schema_version")) is not int
            or proxy.get("tier") != tier
            or proxy.get("recipe_sha256") != recipe["recipe_sha256"]
            or proxy.get("full_scale_executed") is not False
            or proxy.get("launch_authorized") is not False
            or proxy.get("training_authorized") is not False
            or proxy.get("paid_compute_authorized") is not False
            or proxy.get("receipt_sha256") != digest({
                k: v for k, v in proxy.items() if k != "receipt_sha256"})
            or evidence.measured_proxy_receipt_sha256 != proxy["receipt_sha256"]):
        raise ScaleRecipeDenied("proxy identity or authority mismatch")
    # Check nested S4 receipts at their trust boundary; no forged opaque grant.
    inner = proxy.get("proxy")
    if (not isinstance(inner, dict)
            or proxy.get("s4_proxy_receipt_sha256") != digest(inner)
            or inner.get("tier") != PROXY_TIERS[tier]
            or inner.get("full_scale_executed") is not False
            or inner.get("paid_compute_authorized") is not False
            or not isinstance(inner.get("proxy_receipt"), dict)
            or inner["proxy_receipt"].get("promotion_authorized") is not False):
        raise ScaleRecipeDenied("invalid S4 proxy evidence")
    reasons = []
    if evidence.paid_compute_requested:
        reasons.append("paid_compute_not_authorized")
    if not evidence.terminal_20m_proven:
        reasons.append("terminal_20m_proof_missing")
    if tier != "200M" and not evidence.terminal_200m_proven:
        reasons.append("bridge_requires_200m_proof")
    if (tier != "200M" and evidence.measured_bridge_risk_reduction
            < limits.min_bridge_risk_reduction):
        reasons.append("bridge_has_no_measured_need")
    workers = evidence.healthy_workers
    if workers > limits.max_workers:
        reasons.append("worker_budget_exceeded")
    if workers > 1 and evidence.measured_interconnect_bytes_per_second <= 0:
        reasons.append("distributed_link_unmeasured")
    available = workers * evidence.available_memory_bytes_per_worker
    minimum = recipe["train_state_planning_floor_bytes"]
    if available < minimum or available > limits.max_total_memory_bytes:
        reasons.append("memory_admission_denied")
    needed_checkpoint = recipe["checkpoint_min_storage_bytes"]
    if (evidence.available_checkpoint_bytes < needed_checkpoint
            or needed_checkpoint > limits.max_checkpoint_bytes):
        reasons.append("checkpoint_storage_denied")
    if (evidence.measured_tokens_per_second < limits.min_tokens_per_second
            or evidence.measured_tokens_per_second == 0):
        reasons.append("throughput_unqualified")
    else:
        seconds = evidence.unique_training_tokens / evidence.measured_tokens_per_second
        if seconds > limits.max_seconds:
            reasons.append("wallclock_budget_denied")
    if evidence.measured_checkpoint_bytes_per_second <= 0:
        reasons.append("checkpoint_transport_unqualified")
    result = {
        "schema_version": 1, "tier": tier,
        "recipe_sha256": recipe["recipe_sha256"],
        "evidence_sha256": digest(asdict(evidence)),
        "limits_sha256": digest(asdict(limits)),
        "proxy_receipt_sha256": proxy["receipt_sha256"],
        "status": "NO_GO" if reasons else "PROXY_CAPACITY_PLAUSIBLE_NOT_AUTHORIZED",
        "reasons": sorted(reasons),
        "evidence_class": "UNTRUSTED_PROXY_EXTRAPOLATION",
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False,
    }
    result["assessment_sha256"] = digest(result)
    return result


def migration_packet(
    tier: str, parent_model_sha256: str, parent_checkpoint_sha256: str,
    mode: str, proxy_growth_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a smaller parent without treating proxy growth as full optimizer migration."""
    recipe = scale_recipe(tier)
    for name, value in (("parent_model_sha256", parent_model_sha256),
                        ("parent_checkpoint_sha256", parent_checkpoint_sha256)):
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ScaleRecipeDenied(f"{name}: SHA256 required")
    if mode not in {"GROWTH_PROXY_ONLY", "FRESH_INIT_NEW_RUN"}:
        raise ScaleRecipeDenied("unsupported migration mode")
    if mode == "GROWTH_PROXY_ONLY":
        if (not isinstance(proxy_growth_receipt, dict)
                or proxy_growth_receipt.get("function_preserved_on_fixture") is not True
                or proxy_growth_receipt.get("training_authorized") is not False):
            raise ScaleRecipeDenied("growth requires valid proxy-only receipt")
    elif proxy_growth_receipt is not None:
        raise ScaleRecipeDenied("fresh init cannot inherit a growth receipt")
    result = {
        "schema_version": 1, "target_recipe_sha256": recipe["recipe_sha256"],
        "parent_model_sha256": parent_model_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "mode": mode,
        "fixture_growth_receipt_sha256": (
            digest(proxy_growth_receipt) if proxy_growth_receipt is not None else None
        ),
        "full_optimizer_migration_proven": False,
        "fresh_run_identity_required": mode == "FRESH_INIT_NEW_RUN",
        "training_authorized": False, "launch_authorized": False,
    }
    result["migration_sha256"] = digest(result)
    return result
