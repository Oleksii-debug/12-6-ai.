"""Plan 7 Section 4: optional 35M/50M/100M recipes, never launch authority.

Full-size recipes are arithmetic-only. Actual execution uses admitted tiny
LOCAL_FREE analogs with the existing Plan-7 experiment authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from twelve_six.accelerated_scaling import (
    ArchitectureHypothesis,
    ProxyBudget,
    ProxyProtocol,
    admit_proxy,
    run_proxy,
)
from twelve_six.model import ModelSpec

FAMILY = {"35M": (512, 10, 1536), "50M": (512, 14, 1728),
          "100M": (768, 14, 2304)}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RiskProbeDenied(ValueError):
    """The proposed optional probe has no admissible evidence or resources."""


def digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def valid_nonnegative(name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise RiskProbeDenied(f"{name}: expected finite nonnegative number")


def valid_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RiskProbeDenied(f"{name}: expected positive integer")


def valid_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise RiskProbeDenied(f"{name}: expected lowercase SHA256")


def model_for_tier(tier: str) -> ModelSpec:
    """Resolve a ModelSpec without allocating weights or granting compute."""
    if tier not in FAMILY:
        raise RiskProbeDenied("unsupported tier")
    width, depth, ff = FAMILY[tier]
    heads = width // 64
    return ModelSpec(
        schema_version=1, vocab_size=8192, max_seq_len=1024,
        d_model=width, n_layers=depth, n_heads=heads,
        n_kv_heads=heads // 4, head_dim=64, d_ff=ff,
        rope_rotary_dim=64,
    )


@dataclass(frozen=True, slots=True)
class RiskEvidence:
    schema_version: int
    fixture_sha256: str
    protocol_sha256: str
    run_id: str
    observations: int
    measured_risk: float
    expected_risk_reduction: float
    expected_probe_cost_units: float
    observed_free_capacity_bytes: int
    expected_duration_seconds: float
    healthy_workers: int
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != 1 or not isinstance(self.run_id, str)):
            raise RiskProbeDenied("unsupported evidence")
        if not self.run_id.strip():
            raise RiskProbeDenied("missing run")
        valid_digest("fixture", self.fixture_sha256)
        valid_digest("protocol", self.protocol_sha256)
        for key in ("observations", "observed_free_capacity_bytes", "healthy_workers"):
            valid_positive(key, getattr(self, key))
        for key in ("measured_risk", "expected_risk_reduction",
                    "expected_probe_cost_units", "expected_duration_seconds"):
            valid_nonnegative(key, getattr(self, key))
        if self.expected_risk_reduction > self.measured_risk:
            raise RiskProbeDenied("claimed reduction exceeds measured risk")
        if not isinstance(self.paid_compute_requested, bool):
            raise RiskProbeDenied("invalid paid-compute marker")


@dataclass(frozen=True, slots=True)
class ProbeLimits:
    max_cost_units: float
    max_duration_seconds: float
    max_peak_memory_bytes: int
    max_workers: int
    min_observations: int = 3
    min_risk_reduction: float = 0.1

    def __post_init__(self) -> None:
        for key in ("max_cost_units", "max_duration_seconds", "min_risk_reduction"):
            valid_nonnegative(key, getattr(self, key))
        for key in ("max_peak_memory_bytes", "max_workers", "min_observations"):
            valid_positive(key, getattr(self, key))


def full_recipe(tier: str) -> dict[str, Any]:
    """Immutable recipe identity; never represents a full-scale measurement."""
    spec = model_for_tier(tier)
    count = spec.parameter_count()
    target = int(tier[:-1]) * 1_000_000
    if abs(count - target) > target // 20:
        raise RiskProbeDenied("target parameter count out of tolerance")
    record = {
        "schema_version": 1, "tier": tier, "model": spec.to_dict(),
        "modelspec_sha256": spec.identity_sha256(),
        "parameters": count, "parameter_target": target,
        "parameter_bytes_fp32": 4 * count,
        "train_state_lower_bound_bytes_fp32_adam": 16 * count,
        "sequence_template": 1024, "batch_template": 1,
        "checkpoint_format": "CANONICAL_MODELSPEC_STATE_DICT_FIXTURE_V1",
        "growth_policy": "QUALIFIED_FUNCTION_PRESERVING_OR_FRESH_INIT_NEW_RUN",
        "serving_contract": "MODELSPEC_V1_DECODER_LOGITS_ONLY",
        "evidence_class": "ARITHMETIC_RECIPE_NOT_A_SCALE_BENCHMARK",
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False, "mandatory_campaign": False,
    }
    record["recipe_sha256"] = digest(record)
    return record


def decide_probe(tier: str, evidence: RiskEvidence, limits: ProbeLimits) -> dict[str, Any]:
    """Fail-closed optional recommendation; never an execution or budget grant."""
    if not isinstance(evidence, RiskEvidence) or not isinstance(limits, ProbeLimits):
        raise RiskProbeDenied("typed evidence and limits required")
    # Revalidate at trust boundary; a frozen dataclass can be forged with setattr.
    RiskEvidence(**asdict(evidence))
    ProbeLimits(**asdict(limits))
    recipe = full_recipe(tier)
    reasons = []
    if evidence.paid_compute_requested:
        reasons.append("paid_compute_never_authorized")
    if evidence.observations < limits.min_observations:
        reasons.append("insufficient_observations")
    if evidence.expected_risk_reduction < limits.min_risk_reduction:
        reasons.append("no_material_risk_reduction")
    if evidence.expected_probe_cost_units > limits.max_cost_units:
        reasons.append("cost_budget_exceeded")
    if evidence.expected_duration_seconds > limits.max_duration_seconds:
        reasons.append("duration_budget_exceeded")
    if evidence.healthy_workers > limits.max_workers:
        reasons.append("worker_budget_exceeded")
    lower = recipe["train_state_lower_bound_bytes_fp32_adam"]
    if evidence.observed_free_capacity_bytes < lower:
        reasons.append("insufficient_observed_capacity")
    if lower > limits.max_peak_memory_bytes:
        reasons.append("peak_memory_budget_exceeded")
    result = {
        "schema_version": 1, "tier": tier,
        "evidence_sha256": digest(asdict(evidence)),
        "limits_sha256": digest(asdict(limits)),
        "recipe_sha256": recipe["recipe_sha256"],
        "status": "NO_GO" if reasons else "OPTIONAL_RECIPE_ELIGIBLE_NOT_AUTHORIZED",
        "reasons": sorted(reasons),
        "proxy_only": True, "launch_authorized": False,
        "training_authorized": False, "paid_compute_authorized": False,
        "evidence_class": "UNTRUSTED_LOCAL_FREE_FIXTURE_ADVISORY",
    }
    result["receipt_sha256"] = digest(result)
    return result


def tiny_analog(tier: str, protocol: ProxyProtocol, budget: ProxyBudget) -> dict[str, Any]:
    """Execute a real admitted CPU synthetic proxy; never allocate a full tier."""
    full = full_recipe(tier)
    depth, ff = {"35M": (2, 64), "50M": (3, 96), "100M": (4, 128)}[tier]
    analog = replace(
        model_for_tier(tier), vocab_size=64, max_seq_len=16,
        d_model=32, n_layers=depth, n_heads=4, n_kv_heads=2,
        head_dim=8, d_ff=ff, rope_rotary_dim=8,
    )
    resource = admit_proxy(analog, protocol, budget)
    result = run_proxy(
        analog, ArchitectureHypothesis(1, f"risk-{tier}", "baseline", "", 0),
        protocol, budget,
    )
    return {
        "schema_version": 1, "tier": tier, "full_recipe_sha256": full["recipe_sha256"],
        "analog_modelspec_sha256": analog.identity_sha256(),
        "proxy_protocol_sha256": protocol.identity(),
        "admission": resource, "proxy_receipt": result,
        "training_authorized": False, "paid_compute_authorized": False,
        "full_scale_executed": False,
    }


def verify_recipe(value: dict[str, Any]) -> bool:
    """Byte-stable canonical-structure check, not evidence of any large run."""
    if not isinstance(value, dict) or value.get("tier") not in FAMILY:
        return False
    return value == full_recipe(value["tier"])
