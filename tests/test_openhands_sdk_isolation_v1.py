import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/twelve_six/post_base/openhands_isolation.py"
CONFIG_PATH = ROOT / "configs/post_base/openhands_sdk_isolation_v1.json"

spec = importlib.util.spec_from_file_location("openhands_isolation", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TOOLS = CONFIG["project_tool_registry_fixture"]
POLICY = CONFIG["candidate_policy"]


def _policy(**changes):
    value = copy.deepcopy(POLICY)
    value.update(changes)
    return value


def test_bounded_candidate_mechanics_pass_without_backend_claim():
    report = module.qualify_candidate(POLICY, TOOLS, project_binding=CONFIG["project_binding"])
    assert report["verdict"] == "PASS_CANDIDATE_MECHANICS"
    assert report["backend_executed"] is False
    assert report["external_agent_quality_claimed"] is False
    assert module.validate_report(report) == ()


def test_evidence_identity_is_deterministic_and_sensitive_to_policy_drift():
    first = module.qualify_candidate(POLICY, TOOLS)
    second = module.qualify_candidate(copy.deepcopy(POLICY), list(reversed(TOOLS)))
    assert first["evidence_id"] == second["evidence_id"]
    drifted = module.qualify_candidate(_policy(max_wall_seconds=901), TOOLS)
    assert drifted["evidence_id"] != first["evidence_id"]


def test_host_workspace_rejected():
    assert "isolation_mode_must_be_ephemeral_sandbox" in module.validate_policy(
        _policy(isolation_mode="host"), TOOLS
    )


def test_wildcard_filesystem_rejected():
    assert "filesystem_wildcard_or_host_root_forbidden" in module.validate_policy(
        _policy(filesystem_roots=["/"]), TOOLS
    )


def test_wildcard_network_rejected():
    assert "network_wildcard_forbidden" in module.validate_policy(
        _policy(network_hosts=["*"]), TOOLS
    )


def test_host_environment_inheritance_rejected():
    assert "host_environment_inheritance_forbidden" in module.validate_policy(
        _policy(inherit_host_env=True), TOOLS
    )


def test_secret_requires_explicit_broker():
    violations = module.validate_policy(
        _policy(secret_names=["GITHUB_TOKEN"], secrets_injected_by_broker=False), TOOLS
    )
    assert "explicit_secret_broker_required" in violations


def test_undeclared_tool_rejected():
    violations = module.validate_policy(_policy(allowed_tools=["terminal", "host_shell"]), TOOLS)
    assert "undeclared_tools:host_shell" in violations


def test_missing_resource_bound_rejected():
    violations = module.validate_policy(_policy(max_wall_seconds=0), TOOLS)
    assert "max_wall_seconds_out_of_bounds" in violations


def test_unbounded_persistence_rejected():
    violations = module.validate_policy(_policy(persistence="host_persistent"), TOOLS)
    assert "unbounded_persistence_forbidden" in violations


def test_parity_state_requires_real_backend_and_isolation_evidence():
    violations = module.validate_policy(_policy(promotion_state="PARITY_PROVEN"), TOOLS)
    assert "backend_execution_evidence_required" in violations
    assert "isolation_parity_evidence_required" in violations


def test_adopted_state_requires_rollback_evidence():
    evidence = {"backend_executed": True, "isolation_parity_verified": True}
    violations = module.validate_policy(_policy(promotion_state="ADOPTED"), TOOLS, evidence)
    assert "rollback_evidence_required_for_adoption" in violations


def test_adopted_state_can_only_pass_with_all_terminal_evidence():
    evidence = {
        "backend_executed": True,
        "isolation_parity_verified": True,
        "rollback_verified": True,
    }
    assert module.validate_policy(_policy(promotion_state="ADOPTED"), TOOLS, evidence) == ()


def test_report_identity_tamper_is_detected():
    report = module.qualify_candidate(POLICY, TOOLS)
    report["policy"]["cpu_cores"] = 8
    assert "evidence_identity_mismatch" in module.validate_report(report)


def test_live_registry_semantics_are_fail_closed():
    registry = {
        "registry_id": "OPEN-SOURCE-REUSE-REGISTRY-V2",
        "canonical_policy": {
            "agent_frameworks_are_postbase_only": True,
            "canonical_base_random_init_only": True,
        },
        "components": [copy.deepcopy(CONFIG["project_binding"]["registry_component"])],
        "campaigns": {"P1_E_AGENT_RUNTIME": ["OPENHANDS_AGENT_SDK"]},
    }
    assert module.validate_project_binding(CONFIG, registry) == ()
    registry["components"][0]["canonical_base_dependency"] = True
    violations = module.validate_project_binding(CONFIG, registry)
    assert "openhands_registry_component_drift:canonical_base_dependency" in violations


def test_committed_mechanics_evidence_is_self_consistent():
    evidence_path = ROOT / "evidence/post_base/openhands/isolation_v1_local_free.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["scientific_truth"]["openhands_executed"] is False
    assert evidence["scientific_truth"]["adoption_authorized"] is False
    assert module.validate_report(evidence["report"]) == ()
