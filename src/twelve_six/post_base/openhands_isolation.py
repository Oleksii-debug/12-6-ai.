"""Fail-closed qualification contract for optional OpenHands SDK isolation.

This module does not import or execute OpenHands.  It defines the 12-6-owned
boundary that any future adapter must satisfy before backend execution can be
considered parity evidence or adoption evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

PROMOTION_STATES = ("DISCOVERED", "CANDIDATE", "PARITY_PROVEN", "ADOPTED")
_REQUIRED_POLICY_FIELDS = {
    "isolation_mode",
    "filesystem_roots",
    "network_hosts",
    "inherit_host_env",
    "secret_names",
    "secrets_injected_by_broker",
    "max_wall_seconds",
    "max_processes",
    "memory_mb",
    "cpu_cores",
    "persistence",
    "audit_log",
    "provenance_log",
    "allowed_tools",
    "promotion_state",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def evidence_identity(value: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 for machine-readable evidence."""
    payload = dict(value)
    payload.pop("evidence_id", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_wildcard(value: str) -> bool:
    return value.strip() in {"*", "**", "/", "0.0.0.0/0", "::/0"} or "*" in value


def validate_policy(
    policy: Mapping[str, Any],
    declared_tools: Sequence[str],
    backend_evidence: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return violations; an empty tuple means candidate mechanics are bounded."""
    violations: list[str] = []
    missing = sorted(_REQUIRED_POLICY_FIELDS - set(policy))
    if missing:
        violations.append("missing_fields:" + ",".join(missing))
        return tuple(violations)

    if policy["isolation_mode"] != "ephemeral_sandbox":
        violations.append("isolation_mode_must_be_ephemeral_sandbox")

    roots = policy["filesystem_roots"]
    if not isinstance(roots, list) or not roots:
        violations.append("filesystem_roots_must_be_nonempty_list")
    elif any(not isinstance(root, str) or _is_wildcard(root) for root in roots):
        violations.append("filesystem_wildcard_or_host_root_forbidden")

    hosts = policy["network_hosts"]
    if not isinstance(hosts, list):
        violations.append("network_hosts_must_be_list")
    elif any(not isinstance(host, str) or _is_wildcard(host) for host in hosts):
        violations.append("network_wildcard_forbidden")

    if policy["inherit_host_env"] is not False:
        violations.append("host_environment_inheritance_forbidden")

    secret_names = policy["secret_names"]
    if not isinstance(secret_names, list):
        violations.append("secret_names_must_be_list")
    elif any(not isinstance(name, str) or not name or _is_wildcard(name) for name in secret_names):
        violations.append("implicit_or_wildcard_secrets_forbidden")
    elif secret_names and policy["secrets_injected_by_broker"] is not True:
        violations.append("explicit_secret_broker_required")

    numeric_limits = {
        "max_wall_seconds": (1, 3600),
        "max_processes": (1, 128),
        "memory_mb": (64, 32768),
        "cpu_cores": (1, 32),
    }
    for field, (minimum, maximum) in numeric_limits.items():
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations.append(f"{field}_must_be_numeric")
        elif not minimum <= value <= maximum:
            violations.append(f"{field}_out_of_bounds")

    if policy["persistence"] not in {"ephemeral", "workspace_snapshot"}:
        violations.append("unbounded_persistence_forbidden")
    if policy["audit_log"] is not True:
        violations.append("audit_log_required")
    if policy["provenance_log"] is not True:
        violations.append("provenance_log_required")

    allowed_tools = policy["allowed_tools"]
    declared = set(declared_tools)
    if not isinstance(allowed_tools, list) or not allowed_tools:
        violations.append("allowed_tools_must_be_nonempty_list")
    else:
        unknown = sorted(set(allowed_tools) - declared)
        if unknown:
            violations.append("undeclared_tools:" + ",".join(unknown))

    state = policy["promotion_state"]
    if state not in PROMOTION_STATES:
        violations.append("unknown_promotion_state")
    evidence = backend_evidence or {}
    if state in {"PARITY_PROVEN", "ADOPTED"}:
        if evidence.get("backend_executed") is not True:
            violations.append("backend_execution_evidence_required")
        if evidence.get("isolation_parity_verified") is not True:
            violations.append("isolation_parity_evidence_required")
    if state == "ADOPTED" and evidence.get("rollback_verified") is not True:
        violations.append("rollback_evidence_required_for_adoption")

    return tuple(sorted(set(violations)))


def qualify_candidate(
    policy: Mapping[str, Any],
    declared_tools: Sequence[str],
    backend_evidence: Mapping[str, Any] | None = None,
    project_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit deterministic evidence without executing a foreign backend."""
    violations = validate_policy(policy, declared_tools, backend_evidence)
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "OPENHANDS-SDK-ISOLATION-V1",
        "verdict": "PASS_CANDIDATE_MECHANICS" if not violations else "REJECTED",
        "promotion_state": policy.get("promotion_state"),
        "backend_executed": bool((backend_evidence or {}).get("backend_executed", False)),
        "external_agent_quality_claimed": False,
        "canonical_base_dependency": False,
        "violations": list(violations),
        "policy": dict(policy),
        "declared_tools": sorted(set(declared_tools)),
        "project_binding": dict(project_binding or {}),
    }
    report["evidence_id"] = evidence_identity(report)
    return report


def validate_report(report: Mapping[str, Any]) -> tuple[str, ...]:
    violations: list[str] = []
    if report.get("contract") != "OPENHANDS-SDK-ISOLATION-V1":
        violations.append("wrong_contract")
    if report.get("canonical_base_dependency") is not False:
        violations.append("canonical_base_dependency_forbidden")
    if report.get("external_agent_quality_claimed") is not False:
        violations.append("unexecuted_quality_claim_forbidden")
    supplied = report.get("evidence_id")
    if not isinstance(supplied, str) or supplied != evidence_identity(report):
        violations.append("evidence_identity_mismatch")
    if report.get("verdict") == "PASS_CANDIDATE_MECHANICS" and report.get("violations"):
        violations.append("pass_with_violations_forbidden")
    return tuple(sorted(set(violations)))


def validate_project_binding(
    config: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[str, ...]:
    """Bind the qualification to the exact project registry semantics."""
    violations: list[str] = []
    binding = config.get("project_binding", {})
    expected_component = binding.get("registry_component", {})
    if registry.get("registry_id") != "OPEN-SOURCE-REUSE-REGISTRY-V2":
        violations.append("wrong_project_registry")
    policy = registry.get("canonical_policy", {})
    if policy.get("agent_frameworks_are_postbase_only") is not True:
        violations.append("postbase_only_policy_missing")
    if policy.get("canonical_base_random_init_only") is not True:
        violations.append("canonical_base_random_init_boundary_missing")
    component = next(
        (
            item
            for item in registry.get("components", [])
            if item.get("id") == "OPENHANDS_AGENT_SDK"
        ),
        None,
    )
    if component is None:
        violations.append("openhands_registry_component_missing")
    else:
        for field, expected in expected_component.items():
            if component.get(field) != expected:
                violations.append(f"openhands_registry_component_drift:{field}")
    campaign = registry.get("campaigns", {}).get("P1_E_AGENT_RUNTIME", [])
    if "OPENHANDS_AGENT_SDK" not in campaign:
        violations.append("openhands_campaign_binding_missing")
    return tuple(sorted(set(violations)))
