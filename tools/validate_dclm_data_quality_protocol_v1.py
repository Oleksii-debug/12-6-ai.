#!/usr/bin/env python3
"""Validate and compare DCLM-style data-quality experiment evidence.

This module is deliberately stdlib-only and non-authorizing. It validates a
project-owned comparison contract and produces a deterministic recommendation
record. It does not admit corpora, train models, access final tests, or promote
dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ProtocolError(ValueError):
    """Raised when evidence violates the fail-closed comparison contract."""


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON numeric constant is forbidden: {value}")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ProtocolError("top-level JSON value must be an object")
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not canonical-JSON serializable: {exc}") from exc
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{field} must be an array")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{field} must be boolean")
    return value


def _require_finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be finite")
    if positive and result <= 0:
        raise ProtocolError(f"{field} must be > 0")
    return result


def _require_sha256(value: Any, field: str) -> str:
    text = _require_str(value, field)
    if not SHA256_RE.fullmatch(text):
        raise ProtocolError(f"{field} must be a lowercase 64-hex SHA-256")
    return text


def _require_git_sha(value: Any, field: str) -> str:
    text = _require_str(value, field)
    if not GIT_SHA_RE.fullmatch(text):
        raise ProtocolError(f"{field} must be a lowercase 40-hex Git SHA")
    return text


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ProtocolError("unsupported protocol schema_version")
    if protocol.get("protocol_id") != "DCLM-DATA-QUALITY-PROTOCOL-V1":
        raise ProtocolError("unexpected protocol_id")
    if protocol.get("status") != "RESEARCH_ONLY_NON_AUTHORIZING":
        raise ProtocolError("protocol status must remain non-authorizing")

    bindings = _require_dict(protocol.get("authority_bindings"), "authority_bindings")
    if bindings.get("swarm_protocol") != "SWARM-300-V2":
        raise ProtocolError("protocol must bind SWARM-300-V2")
    _require_git_sha(bindings.get("project_main_sha"), "authority_bindings.project_main_sha")

    registry = _require_dict(bindings.get("reuse_registry"), "authority_bindings.reuse_registry")
    if registry.get("path") != "configs/research/open_source_reuse_registry_v2.json":
        raise ProtocolError("unexpected reuse-registry path")
    if registry.get("registry_id") != "OPEN-SOURCE-REUSE-REGISTRY-V2":
        raise ProtocolError("unexpected reuse-registry id")
    _require_git_sha(registry.get("git_blob_sha"), "authority_bindings.reuse_registry.git_blob_sha")
    if registry.get("component_id") != "DCLM":
        raise ProtocolError("registry component must be DCLM")
    if registry.get("decision") != "P0_DATA_QUALITY_EXPERIMENT_PROTOCOL":
        raise ProtocolError("unexpected DCLM registry decision")

    upstream = _require_dict(bindings.get("upstream"), "authority_bindings.upstream")
    if upstream.get("repository") != "mlfoundations/dclm":
        raise ProtocolError("unexpected upstream repository")
    _require_git_sha(upstream.get("commit_sha"), "authority_bindings.upstream.commit_sha")
    if upstream.get("license") != "MIT":
        raise ProtocolError("upstream license binding must be MIT")
    _require_git_sha(upstream.get("readme_blob_sha"), "authority_bindings.upstream.readme_blob_sha")

    contract = _require_dict(protocol.get("comparison_contract"), "comparison_contract")
    min_arms = contract.get("min_arms")
    if isinstance(min_arms, bool) or not isinstance(min_arms, int) or min_arms < 2:
        raise ProtocolError("comparison_contract.min_arms must be an integer >= 2")
    metric = _require_dict(contract.get("metric"), "comparison_contract.metric")
    _require_str(metric.get("name"), "comparison_contract.metric.name")
    if metric.get("direction") not in {"maximize", "minimize"}:
        raise ProtocolError("comparison metric direction must be maximize or minimize")
    budget = _require_dict(contract.get("budget"), "comparison_contract.budget")
    _require_str(budget.get("unit"), "comparison_contract.budget.unit")
    if _require_bool(
        budget.get("require_equal_across_arms"),
        "comparison_contract.budget.require_equal_across_arms",
    ) is not True:
        raise ProtocolError("equal comparison budgets are mandatory")
    if _require_bool(
        contract.get("require_identical_input_snapshot"),
        "comparison_contract.require_identical_input_snapshot",
    ) is not True:
        raise ProtocolError("identical input snapshots are mandatory")
    gates = _require_list(contract.get("required_hard_gates"), "comparison_contract.required_hard_gates")
    if gates != ["rights", "provenance", "privacy", "contamination"]:
        raise ProtocolError("hard-gate set/order changed unexpectedly")
    if contract.get("required_gate_value") != "PASS":
        raise ProtocolError("hard gates must require PASS")
    tie_epsilon = _require_finite_number(contract.get("tie_epsilon"), "comparison_contract.tie_epsilon")
    if tie_epsilon < 0:
        raise ProtocolError("tie_epsilon must be >= 0")
    if contract.get("tie_policy") != "REJECT_AMBIGUOUS_TIE":
        raise ProtocolError("tie policy must fail closed")

    boundary = _require_dict(protocol.get("authority_boundaries"), "authority_boundaries")
    if boundary.get("calibration_purpose") != "SYNTHETIC_CALIBRATION_ONLY_NON_TRAINING":
        raise ProtocolError("calibration purpose must remain non-training")
    if boundary.get("recommendation_state") != "CANDIDATE_RECOMMENDATION_ONLY":
        raise ProtocolError("recommendation state must remain non-authorizing")
    false_fields = (
        "training_authorized",
        "automatic_adoption_allowed",
        "final_test_access_allowed",
        "paid_compute_authorized",
        "canonical_base_weights_changed",
        "foreign_pretrained_weights_allowed",
    )
    for field in false_fields:
        if _require_bool(boundary.get(field), f"authority_boundaries.{field}") is not False:
            raise ProtocolError(f"authority_boundaries.{field} must remain false")


def compare_evidence(protocol: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)

    if evidence.get("schema_version") != 1:
        raise ProtocolError("unsupported evidence schema_version")
    if evidence.get("purpose") != protocol["authority_boundaries"]["calibration_purpose"]:
        raise ProtocolError("evidence purpose is not calibration-only/non-training")
    experiment_id = _require_str(evidence.get("experiment_id"), "experiment_id")
    if _require_bool(evidence.get("training_authorized"), "training_authorized") is not False:
        raise ProtocolError("evidence may not authorize training")
    if evidence.get("requested_promotion_state") == "ADOPTED":
        raise ProtocolError("automatic ADOPTED promotion is forbidden")
    if evidence.get("requested_promotion_state") not in {"DISCOVERED", "CANDIDATE"}:
        raise ProtocolError("requested_promotion_state must be DISCOVERED or CANDIDATE")

    arms = _require_list(evidence.get("arms"), "arms")
    min_arms = protocol["comparison_contract"]["min_arms"]
    if len(arms) < min_arms:
        raise ProtocolError(f"at least {min_arms} experiment arms are required")

    metric_spec = protocol["comparison_contract"]["metric"]
    budget_spec = protocol["comparison_contract"]["budget"]
    gate_names = protocol["comparison_contract"]["required_hard_gates"]
    gate_value = protocol["comparison_contract"]["required_gate_value"]

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    common_input: str | None = None
    common_budget_value: float | None = None

    for index, raw_arm in enumerate(arms):
        arm = _require_dict(raw_arm, f"arms[{index}]")
        arm_id = _require_str(arm.get("arm_id"), f"arms[{index}].arm_id")
        if arm_id in seen_ids:
            raise ProtocolError(f"duplicate arm_id: {arm_id}")
        seen_ids.add(arm_id)

        config_sha = _require_sha256(arm.get("config_sha256"), f"arms[{index}].config_sha256")
        input_sha = _require_sha256(
            arm.get("input_snapshot_sha256"), f"arms[{index}].input_snapshot_sha256"
        )
        if common_input is None:
            common_input = input_sha
        elif input_sha != common_input:
            raise ProtocolError("all arms must bind the identical input snapshot")

        budget = _require_dict(arm.get("budget"), f"arms[{index}].budget")
        if budget.get("unit") != budget_spec["unit"]:
            raise ProtocolError("arm budget unit does not match preregistered unit")
        budget_value = _require_finite_number(
            budget.get("value"), f"arms[{index}].budget.value", positive=True
        )
        if common_budget_value is None:
            common_budget_value = budget_value
        elif budget_value != common_budget_value:
            raise ProtocolError("all arms must use equal comparison budgets")

        metric = _require_dict(arm.get("metric"), f"arms[{index}].metric")
        if metric.get("name") != metric_spec["name"]:
            raise ProtocolError("arm metric name does not match preregistration")
        if metric.get("direction") != metric_spec["direction"]:
            raise ProtocolError("arm metric direction does not match preregistration")
        score = _require_finite_number(metric.get("score"), f"arms[{index}].metric.score")

        gates = _require_dict(arm.get("hard_gates"), f"arms[{index}].hard_gates")
        if set(gates) != set(gate_names):
            raise ProtocolError("arm hard-gate keys must exactly match the protocol")
        for gate in gate_names:
            if gates.get(gate) != gate_value:
                raise ProtocolError(f"arm {arm_id} hard gate {gate} is not PASS")

        if _require_bool(
            arm.get("training_authorized"), f"arms[{index}].training_authorized"
        ) is not False:
            raise ProtocolError(f"arm {arm_id} may not authorize training")
        if arm.get("promotion_state") == "ADOPTED":
            raise ProtocolError(f"arm {arm_id} may not claim ADOPTED")
        if arm.get("promotion_state") not in {"DISCOVERED", "CANDIDATE"}:
            raise ProtocolError(f"arm {arm_id} promotion_state is invalid")

        normalized.append(
            {
                "arm_id": arm_id,
                "config_sha256": config_sha,
                "input_snapshot_sha256": input_sha,
                "budget": {"unit": budget_spec["unit"], "value": budget_value},
                "metric": {
                    "name": metric_spec["name"],
                    "direction": metric_spec["direction"],
                    "score": score,
                },
            }
        )

    normalized.sort(key=lambda item: item["arm_id"])
    reverse = metric_spec["direction"] == "maximize"
    ranked = sorted(
        normalized,
        key=lambda item: (item["metric"]["score"], item["arm_id"]),
        reverse=reverse,
    )
    best, runner_up = ranked[0], ranked[1]
    gap = abs(best["metric"]["score"] - runner_up["metric"]["score"])
    epsilon = float(protocol["comparison_contract"]["tie_epsilon"])
    if gap <= epsilon:
        raise ProtocolError("ambiguous best-arm tie rejected by fail-closed tie policy")

    report: dict[str, Any] = {
        "schema_version": 1,
        "report_id": f"{experiment_id}-COMPARISON-REPORT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_sha256(protocol),
        "evidence_sha256": canonical_sha256(evidence),
        "authority_bindings": protocol["authority_bindings"],
        "experiment_id": experiment_id,
        "purpose": evidence["purpose"],
        "comparison": {
            "metric": metric_spec,
            "budget_unit": budget_spec["unit"],
            "budget_value": common_budget_value,
            "input_snapshot_sha256": common_input,
            "winner_arm_id": best["arm_id"],
            "winner_score": best["metric"]["score"],
            "runner_up_arm_id": runner_up["arm_id"],
            "runner_up_score": runner_up["metric"]["score"],
            "winner_gap": gap,
            "arm_count": len(normalized),
        },
        "recommendation_state": protocol["authority_boundaries"]["recommendation_state"],
        "training_authorized": False,
        "automatic_adoption_allowed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "canonical_base_weights_changed": False,
        "validated_arms": normalized,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        protocol = load_json(args.protocol)
        evidence = load_json(args.evidence)
        report = compare_evidence(protocol, evidence)
    except (OSError, json.JSONDecodeError, ProtocolError) as exc:
        parser.error(str(exc))

    if args.output:
        write_report(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
