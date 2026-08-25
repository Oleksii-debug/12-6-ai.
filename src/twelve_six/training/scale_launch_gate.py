"""Fail-closed qualification and cost gate for materially paid scale training.

This module does not launch compute.  It consumes a planned run plus measured
qualification evidence and reports whether the technical prerequisites are met.
Owner authorization remains a separate, explicit gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

AUTHORIZATION_TOKEN = "COMPUTE_AUTHORIZED"
GPU_MEASUREMENT_KIND = "GPU_MEASURED"
_UNFROZEN_IDENTITIES = frozenset({"", "TBD", "NOT_FROZEN", "NOT_TESTED", "NONE"})


class ScaleLaunchGateError(ValueError):
    """Raised when a launch-plan or qualification-evidence record is malformed."""


def _require_positive_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScaleLaunchGateError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ScaleLaunchGateError(f"{name} must be a finite positive number")
    return result


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScaleLaunchGateError(f"{name} must be a non-negative integer")
    return value


def _require_identity(name: str, value: Any) -> str:
    if not isinstance(value, str) or value.strip().upper() in _UNFROZEN_IDENTITIES:
        raise ScaleLaunchGateError(f"{name} must be a frozen non-placeholder identity")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CostProjection:
    target_tokens: int
    measured_global_tokens_per_second: float
    projected_wall_hours: float
    gpu_count: int
    eur_per_gpu_hour: float
    projected_compute_eur: float
    budget_eur: float
    reserve_fraction: float
    spend_ceiling_eur: float
    headroom_eur: float


@dataclass(frozen=True, slots=True)
class LaunchGateReport:
    technical_qualified: bool
    owner_authorized: bool
    launch_allowed: bool
    reasons: tuple[str, ...]
    projection: CostProjection | None


def project_cost(plan: Mapping[str, Any], evidence: Mapping[str, Any]) -> CostProjection:
    """Project main-run wall time/cost from *measured GPU* throughput only."""
    if evidence.get("measurement_kind") != GPU_MEASUREMENT_KIND:
        raise ScaleLaunchGateError(
            "cost projection requires GPU_MEASURED evidence; CPU throughput is not accepted"
        )

    target_tokens = _require_nonnegative_int("plan.target_tokens", plan.get("target_tokens"))
    if target_tokens == 0:
        raise ScaleLaunchGateError("plan.target_tokens must be > 0")
    gpu_count = _require_nonnegative_int("plan.gpu_count", plan.get("gpu_count"))
    if gpu_count == 0:
        raise ScaleLaunchGateError("plan.gpu_count must be > 0")

    tps = _require_positive_number(
        "evidence.global_tokens_per_second", evidence.get("global_tokens_per_second")
    )
    eur_per_gpu_hour = _require_positive_number(
        "plan.eur_per_gpu_hour", plan.get("eur_per_gpu_hour")
    )
    budget_eur = _require_positive_number("plan.budget_eur", plan.get("budget_eur"))
    reserve_fraction = plan.get("reserve_fraction")
    if (
        isinstance(reserve_fraction, bool)
        or not isinstance(reserve_fraction, (int, float))
        or not math.isfinite(float(reserve_fraction))
        or not 0 <= float(reserve_fraction) < 1
    ):
        raise ScaleLaunchGateError("plan.reserve_fraction must be in [0, 1)")
    reserve_fraction = float(reserve_fraction)

    wall_hours = target_tokens / tps / 3600.0
    compute_eur = wall_hours * gpu_count * eur_per_gpu_hour
    spend_ceiling = budget_eur * (1.0 - reserve_fraction)
    return CostProjection(
        target_tokens=target_tokens,
        measured_global_tokens_per_second=tps,
        projected_wall_hours=wall_hours,
        gpu_count=gpu_count,
        eur_per_gpu_hour=eur_per_gpu_hour,
        projected_compute_eur=compute_eur,
        budget_eur=budget_eur,
        reserve_fraction=reserve_fraction,
        spend_ceiling_eur=spend_ceiling,
        headroom_eur=spend_ceiling - compute_eur,
    )


def evaluate_launch_gate(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    authorization: str | None = None,
) -> LaunchGateReport:
    """Evaluate qualification without ever launching compute.

    ``authorization`` is intentionally external to the plan/evidence records so a
    committed JSON file cannot silently authorize materially paid execution.
    """
    reasons: list[str] = []

    expected_source = _require_identity("plan.source_sha", plan.get("source_sha"))
    expected_tokenizer = _require_identity(
        "plan.tokenizer_identity", plan.get("tokenizer_identity")
    )
    expected_corpus = _require_identity(
        "plan.corpus_manifest_sha256", plan.get("corpus_manifest_sha256")
    )
    expected_architecture = _require_identity(
        "plan.architecture_identity", plan.get("architecture_identity")
    )

    if evidence.get("measurement_kind") != GPU_MEASUREMENT_KIND:
        reasons.append("qualification throughput must be measured on GPU, not CPU/extrapolated")
    if evidence.get("source_sha") != expected_source:
        reasons.append("qualification source SHA does not match the planned source")
    if evidence.get("tokenizer_identity") != expected_tokenizer:
        reasons.append("qualification tokenizer identity does not match the planned tokenizer")
    if evidence.get("corpus_manifest_sha256") != expected_corpus:
        reasons.append("qualification corpus identity does not match the planned corpus")
    if evidence.get("architecture_identity") != expected_architecture:
        reasons.append("qualification architecture family/identity does not match the plan")
    if evidence.get("precision") != plan.get("precision"):
        reasons.append("qualification precision does not match the planned precision")
    if evidence.get("gpu_class") != plan.get("gpu_class"):
        reasons.append("qualification GPU class does not match the planned GPU class")

    measured_tps = _require_positive_number(
        "evidence.global_tokens_per_second", evidence.get("global_tokens_per_second")
    )
    minimum_tps = _require_positive_number(
        "plan.minimum_global_tokens_per_second",
        plan.get("minimum_global_tokens_per_second"),
    )
    if measured_tps < minimum_tps:
        reasons.append(
            f"measured throughput {measured_tps:.3f} tok/s is below qualification floor "
            f"{minimum_tps:.3f} tok/s"
        )

    peak_hbm_fraction = evidence.get("peak_hbm_fraction")
    maximum_hbm_fraction = plan.get("maximum_peak_hbm_fraction")
    if (
        isinstance(peak_hbm_fraction, bool)
        or not isinstance(peak_hbm_fraction, (int, float))
        or not 0 < float(peak_hbm_fraction) <= 1
    ):
        raise ScaleLaunchGateError("evidence.peak_hbm_fraction must be in (0, 1]")
    if (
        isinstance(maximum_hbm_fraction, bool)
        or not isinstance(maximum_hbm_fraction, (int, float))
        or not 0 < float(maximum_hbm_fraction) <= 1
    ):
        raise ScaleLaunchGateError("plan.maximum_peak_hbm_fraction must be in (0, 1]")
    if float(peak_hbm_fraction) > float(maximum_hbm_fraction):
        reasons.append("qualification exceeds the peak-HBM safety threshold")

    qualification_tokens = _require_nonnegative_int(
        "evidence.qualification_tokens", evidence.get("qualification_tokens")
    )
    minimum_qualification_tokens = _require_nonnegative_int(
        "plan.minimum_qualification_tokens", plan.get("minimum_qualification_tokens")
    )
    if qualification_tokens < minimum_qualification_tokens:
        reasons.append("qualification run is too short to support launch authorization")

    if evidence.get("loss_decreased") is not True:
        reasons.append("qualification loss-decrease criterion did not pass")
    if _require_nonnegative_int(
        "evidence.non_finite_steps", evidence.get("non_finite_steps")
    ) != 0:
        reasons.append("qualification observed non-finite training steps")
    if evidence.get("checkpoint_roundtrip_passed") is not True:
        reasons.append("checkpoint round-trip evidence is missing or failed")
    if evidence.get("resume_continuity_passed") is not True:
        reasons.append("interrupted-resume continuity evidence is missing or failed")
    if evidence.get("data_cursor_resume_passed") is not True:
        reasons.append("data-cursor restart evidence is missing or failed")

    planned_gpu_count = _require_nonnegative_int("plan.gpu_count", plan.get("gpu_count"))
    if planned_gpu_count > 1 and evidence.get("distributed_same_topology_passed") is not True:
        reasons.append("multi-GPU plan requires real same-topology distributed training evidence")

    projection: CostProjection | None = None
    if evidence.get("measurement_kind") == GPU_MEASUREMENT_KIND:
        projection = project_cost(plan, evidence)
        if projection.projected_compute_eur > projection.spend_ceiling_eur:
            reasons.append(
                "measured-throughput cost projection exceeds the budget after required reserve"
            )

    technical_qualified = not reasons
    owner_authorized = authorization == AUTHORIZATION_TOKEN
    launch_allowed = technical_qualified and owner_authorized
    if technical_qualified and not owner_authorized:
        reasons.append("technical gates pass, but owner authorization COMPUTE_AUTHORIZED is absent")

    return LaunchGateReport(
        technical_qualified=technical_qualified,
        owner_authorized=owner_authorized,
        launch_allowed=launch_allowed,
        reasons=tuple(reasons),
        projection=projection,
    )
