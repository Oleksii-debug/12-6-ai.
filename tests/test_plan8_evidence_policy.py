import base64
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_ai_qa_control_section4 import _candidate, _failure, _pass_receipt, _policy

from twelve_six import evidence_policy
from twelve_six.ai_qa_control import GateKind, GateVerdict
from twelve_six.capability_map import load_capability_registry
from twelve_six.evidence_control_bridge import build_host_dispatch
from twelve_six.evidence_policy import (
    EvidenceTarget,
    PhysicalEvidenceSource,
    admit_component_receipt,
    admit_signed_evidence,
    admit_sil_receipt,
    capability_readiness,
    evaluate_qualified_repair,
    load_evidence_policy,
    scale_readiness,
)
from twelve_six.physical_qualification import (
    ExecutionMode,
    QualificationAction,
    QualificationPacket,
    ResourceKind,
    execute_qualification,
    inventory_host,
    load_verified_signed_packet,
    write_evidence_bundle,
)
from twelve_six.scenario_harness import Scenario, ScenarioStep, _digest, run_scenario
from twelve_six.sil_qualification import build_package_manifest_bytes, load_sil_scenario

_ROOT = Path(__file__).resolve().parents[1]
_POLICY = _ROOT / "configs/control/plan8_evidence_requirements_v1.json"
_REGISTRY = _ROOT / "configs/control/product_capabilities_v1.json"
_SIGNED_SCHEMA = "12-6.plan8-evidence-attestation.v1"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _authority(key_id: str):
    key = Ed25519PrivateKey.generate()
    public = key.public_key()

    def verify(observed_id: str, message: bytes, signature: bytes) -> bool:
        if observed_id != key_id:
            return False
        try:
            public.verify(signature, message)
        except InvalidSignature:
            return False
        return True

    return key, verify


def _signature(key, key_id: str, body: dict) -> dict[str, str]:
    return {"algorithm": "ED25519", "key_id": key_id,
            "signature_b64": base64.b64encode(key.sign(_json(body))).decode()}


def _attestation(key, target, kind, subject, **scope) -> bytes:
    body = {"schema_version": _SIGNED_SCHEMA, "kind": kind, "subject": subject,
            "git_sha": target.git_sha, "package_sha256": target.package_sha256,
            "artifact_sha256": "3" * 64, "config_sha256": None, "registry_sha256": None,
            "physical_gate_id": None, "backend": None}
    body.update(scope)
    return _json({"statement": body, "signature": _signature(key, "independent", body)})


def _registry_policy():
    registry = load_capability_registry(_REGISTRY)
    policy = load_evidence_policy(_POLICY, registry, expected_sha256=_sha(_POLICY.read_bytes()))
    return registry, policy


def _git_head(root: Path) -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()


def _commit(root: Path) -> str:
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "-c", "core.autocrlf=false", "add", "--all"), cwd=root, check=True)
    subprocess.run(
        ("git", "-c", "user.name=S8 test", "-c", "user.email=s8@example.invalid",
         "commit", "-qm", "Fixture source"), cwd=root, check=True,
    )
    return _git_head(root)


def test_complete_policy_refuses_downgrade_and_registry_omission(tmp_path):
    registry, policy = _registry_policy()
    assert len(policy.requirements) == len(registry.capabilities) == 26
    assert policy.required("windows-nvda-final-product").value == "Q3"
    altered = tmp_path / "policy.json"
    doc = json.loads(_POLICY.read_bytes())
    doc["requirements"]["windows-nvda-final-product"] = "Q0"
    altered.write_bytes(_json(doc))
    with pytest.raises(ValueError, match="trusted SHA"):
        load_evidence_policy(altered, registry, expected_sha256=policy.policy_sha256)
    doc["requirements"].pop("package-runtime")
    altered.write_bytes(_json(doc))
    with pytest.raises(ValueError, match="every registry capability"):
        load_evidence_policy(altered, registry, expected_sha256=_sha(altered.read_bytes()))


def test_q0_is_trusted_source_only_and_cannot_be_reclassified(tmp_path):
    registry, policy = _registry_policy()
    (tmp_path / "input.txt").write_bytes(b"candidate")
    head = _commit(tmp_path)
    scenario = Scenario(
        "component-model-contract", ("model-spec-identity",), ("fixture-ready",),
        (ScenarioStep("readback", "read-input", _digest({"observed": "candidate"})),), "Q0",
    )
    receipt = run_scenario(
        scenario, repo_root=tmp_path, source_sha=head,
        available_prerequisites=frozenset({"fixture-ready"}),
        adapters={"read-input": lambda root: {"observed": (root / "input.txt").read_text()}},
    )
    args = {"capability_id": "model-spec-identity", "source_sha": head, "registry": registry}
    with pytest.raises(ValueError, match="trusted authority"):
        admit_component_receipt(scenario, receipt, **args, trusted_scenario_sha256="a" * 64)
    proof = admit_component_receipt(scenario, receipt, **args,
                                   trusted_scenario_sha256=scenario.identity_sha256())
    assert proof.target.package_sha256 is None
    current = capability_readiness(policy, "model-spec-identity", EvidenceTarget(head, "b" * 64), (proof,))
    assert current["evidence_complete"]
    assert current["package_bound_levels"] == []
    stale = capability_readiness(policy, "model-spec-identity", EvidenceTarget("c" * 40, "b" * 64), (proof,))
    assert not stale["evidence_complete"]
    with pytest.raises(ValueError, match="source-only|verified ingress"):
        replace(proof, kind="Q4", target=EvidenceTarget(head, "b" * 64))


def test_signed_operator_scope_is_registry_package_and_role_bound():
    registry, policy = _registry_policy()
    key, verify = _authority("independent")
    target = EvidenceTarget("a" * 40, "b" * 64)
    subject = "windows-nvda-final-product"
    raw = _attestation(key, target, "Q3", subject, registry_sha256=registry.identity_sha256())
    trusted = {("Q3", subject): frozenset({"independent"})}
    proof = admit_signed_evidence(raw, target=target, signature_verifier=verify, trusted_keys=trusted)
    assert capability_readiness(policy, subject, target, (proof,))["evidence_complete"]
    stale = replace(target, package_sha256="d" * 64)
    assert not capability_readiness(policy, subject, stale, (proof,))["evidence_complete"]
    with pytest.raises(ValueError, match="stale"):
        admit_signed_evidence(raw, target=stale, signature_verifier=verify, trusted_keys=trusted)
    with pytest.raises(ValueError, match="not trusted"):
        admit_signed_evidence(raw, target=target, signature_verifier=verify,
                              trusted_keys={("Q3", "package-runtime"): frozenset({"independent"})})
    tampered = json.loads(raw)
    tampered["statement"]["artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="signature verification"):
        admit_signed_evidence(_json(tampered), target=target, signature_verifier=verify, trusted_keys=trusted)
    with pytest.raises(ValueError, match="verified ingress"):
        replace(proof, kind="Q4")
    foreign = _attestation(key, target, "Q3", subject, registry_sha256="e" * 64)
    foreign_proof = admit_signed_evidence(foreign, target=target,
                                          signature_verifier=verify, trusted_keys=trusted)
    assert not capability_readiness(policy, subject, target, (foreign_proof,))["evidence_complete"]


def test_qualification_preserves_independent_repair_gate_authority():
    registry, policy = _registry_policy()
    failure = _failure()
    candidate = _candidate(failure)
    target = EvidenceTarget(candidate.candidate_git_sha, "b" * 64)
    receipts = (
        _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation"),
        _pass_receipt(GateKind.ADVERSARIAL, actor_id="regression-automation"),
        _pass_receipt(GateKind.SIL, actor_id="independent-sil"),
        _pass_receipt(GateKind.PHYSICAL, actor_id="scope-auditor",
                      verdict=GateVerdict.NOT_APPLICABLE, reason="software-only defect"),
    )
    key, verify = _authority("independent")
    subject = "windows-nvda-final-product"
    proof = admit_signed_evidence(
        _attestation(key, target, "Q3", subject, registry_sha256=registry.identity_sha256()),
        target=target, signature_verifier=verify,
        trusted_keys={("Q3", subject): frozenset({"independent"})},
    )
    args = {"repair_policy": _policy(), "evidence_policy": policy,
            "capability_id": subject, "target": target}
    ready = evaluate_qualified_repair(
        failure, candidate, receipts, certifier_actor_id="independent-certifier",
        evidence=(proof,), **args,
    )
    assert ready["decision"] == "READY_FOR_INDEPENDENT_PROMOTION"
    blocked = evaluate_qualified_repair(
        failure, candidate, receipts, certifier_actor_id=candidate.proposer_actor_id,
        evidence=(proof,), **args,
    )
    assert blocked["decision"] == "BLOCK"
    missing = evaluate_qualified_repair(
        failure, candidate, receipts, certifier_actor_id="independent-certifier",
        evidence=(), **args,
    )
    assert missing["decision"] == "BLOCK"


def test_sil_ingress_reuses_verified_registry_journey_and_package(monkeypatch):
    registry, policy = _registry_policy()
    source = "a" * 40
    package = b"exact package"
    target = EvidenceTarget(source, _sha(package))
    scenario = load_sil_scenario(_ROOT / "configs/control/sil_scenario_v1.json")
    journey = next(item for item in registry.journeys if "training-mechanics" in item.capability_ids)

    def verify(evidence, log, **kwargs):
        assert (evidence, log) == ("evidence.json", "sil.log")
        assert kwargs["expected_git_sha"] == source
        assert kwargs["expected_package_bytes"] == package
        assert kwargs["expected_registry"] is registry
        return {"available_journey_ids": [journey.journey_id], "evidence_identity_sha256": "c" * 64}

    monkeypatch.setattr(evidence_policy, "verify_sil_evidence", verify)
    args = {"target": target, "package_bytes": package, "environment_receipt": {},
            "registry": registry, "scenario": scenario}
    proof = admit_sil_receipt("evidence.json", "sil.log", capability_id="training-mechanics", **args)
    assert capability_readiness(policy, "training-mechanics", target, (proof,))["evidence_complete"]
    with pytest.raises(ValueError, match="does not cover"):
        admit_sil_receipt("evidence.json", "sil.log", capability_id="package-runtime", **args)
    with pytest.raises(ValueError, match="package differs"):
        admit_sil_receipt("evidence.json", "sil.log", capability_id="training-mechanics",
                          **{**args, "package_bytes": b"other package"})


def test_real_signed_host_verification_is_required_for_q2(tmp_path):
    registry, policy = _registry_policy()
    root = tmp_path / "candidate"
    agent = root / "src/twelve_six/physical_qualification.py"
    agent.parent.mkdir(parents=True)
    agent.write_bytes((_ROOT / "src/twelve_six/physical_qualification.py").read_bytes())
    (root / ".gitignore").write_bytes(b".pytest_cache/\n__pycache__/\n")
    (root / "pyproject.toml").write_bytes(b"[project]\nname = 'fixture'\nversion = '0.1.0'\n")
    (root / "tests").mkdir()
    (root / "tests/test_fixture.py").write_bytes(b"def test_fixture():\n    assert True\n")
    head = _commit(root)
    package = _sha(build_package_manifest_bytes(root))
    target = EvidenceTarget(head, package)
    authority, packet_verify = _authority("packet-authority")
    host, host_verify = _authority("fixture-host")
    role, role_verify = _authority("independent")
    now = int(time.time())
    packet = QualificationPacket(
        schema_version="12-6.physical-qualification-packet.v1",
        packet_id="hil-test", target_git_sha=head, agent_source_sha256=_sha(agent.read_bytes()),
        execution_mode=ExecutionMode.REAL_HOST, allowed_os_families=(inventory_host(root).os_family,),
        not_before_epoch_seconds=now - 1, expires_epoch_seconds=now + 300,
        actions=(QualificationAction("hil-test", ("tests/test_fixture.py",), 45, 65536, (ResourceKind.CPU,)),),
        artifact_paths=(),
    )
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(_json({
        "schema_version": "12-6.signed-physical-qualification-packet.v1",
        "packet": packet.to_dict(), "signature": _signature(authority, "packet-authority", packet.to_dict()),
    }))
    verified = load_verified_signed_packet(packet_path, signature_verifier=packet_verify)
    dispatch = build_host_dispatch(verified, repo_root=root, dispatch_id="hil-test",
                                   scenario_id="hil-test", physical_gate_id="fixture-hil")
    receipt, log = execute_qualification(
        verified, repo_root=root, agent_source_bytes=agent.read_bytes(),
        evidence_signing_key_id="fixture-host", evidence_signer=lambda _, data: host.sign(data),
    )
    assert receipt["verdict"] == "PASS", receipt["reasons"]
    receipt_path, log_path = tmp_path / "host.json", tmp_path / "host.log"
    write_evidence_bundle(receipt_path, log_path, evidence=receipt, log_bytes=log)
    source = PhysicalEvidenceSource(root, dispatch, packet_path, receipt_path, log_path,
                                    packet_verify, host_verify)
    subject = "package-runtime"
    args = {"target": target, "signature_verifier": role_verify,
            "trusted_keys": {("Q2", subject): frozenset({"independent"})}}
    scope = {"artifact_sha256": receipt["evidence_identity_sha256"],
             "registry_sha256": registry.identity_sha256(), "physical_gate_id": "fixture-hil"}
    raw = _attestation(role, target, "Q2", subject, **scope)
    with pytest.raises(ValueError, match="verified physical"):
        admit_signed_evidence(raw, **args)
    proof = admit_signed_evidence(raw, **args, physical_source=source)
    assert capability_readiness(policy, subject, target, (proof,))["evidence_complete"]
    wrong_gate = _attestation(role, target, "Q2", subject,
                              **{**scope, "physical_gate_id": "different-hil"})
    with pytest.raises(ValueError, match="gate differs"):
        admit_signed_evidence(wrong_gate, **args, physical_source=source)
    log_path.write_bytes(log + b"tamper")
    with pytest.raises(ValueError, match="log"):
        admit_signed_evidence(raw, **args, physical_source=source)


def test_seven_profiles_keep_independent_exact_config_states():
    target = EvidenceTarget(_git_head(_ROOT), _sha(build_package_manifest_bytes(_ROOT)))
    key, verify = _authority("independent")
    profile = "SCALE-500M"
    config_sha = _sha((_ROOT / "configs/stages/scale_500m.json").read_bytes())
    trusted = {(kind, profile): frozenset({"independent"}) for kind in (
        "PROXY_VERIFIED", "HOST_VERIFIED", "TRAINING_AUTHORIZED", "LEARNED_QUALIFIED",
    )}
    args = {"target": target, "signature_verifier": verify, "trusted_keys": trusted}
    proxy = admit_signed_evidence(_attestation(key, target, "PROXY_VERIFIED", profile,
                                               config_sha256=config_sha), **args)
    training = admit_signed_evidence(_attestation(key, target, "TRAINING_AUTHORIZED", profile,
                                                  config_sha256=config_sha), **args)
    stale = admit_signed_evidence(_attestation(key, target, "LEARNED_QUALIFIED", profile,
                                               config_sha256="f" * 64), **args)
    host = _attestation(key, target, "HOST_VERIFIED", profile, config_sha256=config_sha,
                        backend="CPU", physical_gate_id="fixture-host")
    with pytest.raises(ValueError, match="verified physical"):
        admit_signed_evidence(host, **args)
    rows = scale_readiness(_ROOT, target, (proxy, training, stale))
    assert tuple(row["profile_id"] for row in rows) == (
        "SCALE-500M", "SCALE-750M", "SCALE-1B", "SCALE-2B", "SCALE-3B", "SCALE-4B", "SCALE-5B",
    )
    assert rows[0]["states"] == {
        "CONFIG_DEFINED": True, "PROXY_VERIFIED": True, "HOST_VERIFIED": False,
        "TRAINING_AUTHORIZED": True, "LEARNED_QUALIFIED": False,
    }
    assert rows[0]["host_backends"] == []
    for row in rows[1:]:
        assert [state for state, present in row["states"].items() if present] == ["CONFIG_DEFINED"]
