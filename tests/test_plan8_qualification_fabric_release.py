import json
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from test_plan8_evidence_policy import _authority, _registry_policy, _signature

from twelve_six import physical_qualification, sil_qualification
from twelve_six.ai_qa_control import (
    GateKind,
    GateReceipt,
    GateVerdict,
    build_regression_chain,
    execute_automated_regressions,
    failure_packet_from_sil,
    load_ai_qa_policy,
    materialize_local_repair_candidate,
    verify_candidate_sil_evidence,
)
from twelve_six.evidence_lineage import EvidenceArtifact, EvidenceLineage, defect_identity
from twelve_six.integration_qualification import (
    freeze_candidate,
    prepare_integration,
    verify_accepted_integration,
)
from twelve_six.physical_qualification import (
    ExecutionMode,
    QualificationAction,
    QualificationPacket,
    ResourceKind,
    execute_qualification,
    inventory_host,
    load_verified_signed_packet,
    verify_qualification_evidence,
    write_evidence_bundle,
)
from twelve_six.qualification_fabric_release import verify_qualification_fabric_release
from twelve_six.sil_qualification import (
    CommandExecution,
    build_package_manifest_bytes,
    build_sil_plan,
    canonical_sil_environment_receipt_v1,
    load_sil_scenario,
    probe_git_state,
)

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE = _ROOT / "configs/plan8/qualification_fabric_release_v1.json"
_RELEASE_SHA256 = "de7110efb59f95af2386ca24f2e913dd2098ec11c3a5e05949c7af58b588d726"
_FIXTURE_MARKER = "src/twelve_six/value.txt"
_FIXTURE_RUNNER = """
import hashlib, os, runpy, sys
raw = os.environ['TWELVE_SIX_SIL_INPUT_ENVELOPE_JSON'].encode('utf-8')
identity = hashlib.sha256(raw).hexdigest()
if identity != os.environ['TWELVE_SIX_SIL_INPUT_IDENTITY_SHA256']:
    raise SystemExit(86)
print('SIL_INPUT_VERIFIED_SHA256=' + identity, flush=True)
sys.argv[:] = sys.argv[1:]
runpy.run_module('pytest', run_name='__main__', alter_sys=True)
"""


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ("git", "-c", "user.name=Fabric fixture", "-c", "user.email=fabric@example.invalid", *args),
        cwd=root, stderr=subprocess.PIPE,
    )


def _head(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD").decode().strip()


def _make_fixture_repo(root: Path, vectors) -> str:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "core.autocrlf", "false")
    (root / ".gitignore").write_bytes(b".pytest_cache/\n__pycache__/\n")
    (root / "pyproject.toml").write_bytes(b"[project]\nname='fabric-fixture'\nversion='0.0.1'\n")
    marker = root / _FIXTURE_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"BROKEN\n")
    first_target = vectors[0].argv[4]
    paths = sorted({path for vector in vectors for path in vector.argv[4:]})
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == first_target:
            target.write_bytes(
                b"from pathlib import Path\n"
                b"def test_fixture():\n"
                b"    assert Path('src/twelve_six/value.txt').read_bytes() == b'READY\\n'\n"
            )
        else:
            target.write_bytes(b"def test_fixture():\n    assert True\n")
    _git(root, "add", "--all")
    _git(root, "commit", "-m", "Failing qualification fixture")
    return first_target


def _run_fixture_command(argv, cwd, timeout_seconds, envelope, expected_identity) -> CommandExecution:
    assert argv[:3] == (sys.executable, "-m", "pytest")
    env = sil_qualification._qualification_subprocess_env()
    env["TWELVE_SIX_SIL_INPUT_ENVELOPE_JSON"] = envelope.decode("utf-8")
    env["TWELVE_SIX_SIL_INPUT_IDENTITY_SHA256"] = expected_identity
    started = time.monotonic_ns()
    result = subprocess.run(
        (sys.executable, "-c", _FIXTURE_RUNNER, "pytest", *argv[3:]),
        cwd=cwd, env=env, stdin=subprocess.DEVNULL, text=True, capture_output=True,
        timeout=timeout_seconds, check=False,
    )
    proof_line = f"SIL_INPUT_VERIFIED_SHA256={expected_identity}\n"
    consumed = expected_identity if result.stdout.startswith(proof_line) else None
    return CommandExecution(
        result.returncode, result.stdout, result.stderr, (time.monotonic_ns() - started) // 1_000_000, consumed,
    )


def _run_local_sil_fixture(root: Path, outputs: Path, label: str, *, registry, scenario, environment):
    """Exercise real subprocess tests using the explicit SIL contract fixture backend."""
    package = build_package_manifest_bytes(root)
    evidence, log = sil_qualification._qualify_sil_with_backends(
        repo_root=root, expected_git_sha=_head(root), registry=registry, scenario=scenario,
        package_bytes=package, environment_receipt=environment,
        command_runner=_run_fixture_command, git_probe=probe_git_state,
    )
    evidence_path = outputs / f"{label}.json"
    log_path = outputs / f"{label}.log"
    evidence_path.write_bytes(_json(evidence) + b"\n")
    log_path.write_bytes(log.encode("utf-8"))
    return package, evidence, evidence_path, log_path


def _retained(path: Path, identity: str, provenance: str) -> EvidenceArtifact:
    raw = path.read_bytes()
    return EvidenceArtifact(identity, _sha(raw), len(raw), provenance, raw)


def _reopen_public_key(path: Path, expected_id: str):
    def verify(key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != expected_id:
            return False
        public = Ed25519PublicKey.from_public_bytes(path.read_bytes())
        try:
            public.verify(signature, message)
        except InvalidSignature:
            return False
        return True

    return verify


def _physical_simulation(root: Path, outputs: Path, target: str) -> str:
    packet_key, _ = _authority("fixture-packet")
    host_key, _ = _authority("fixture-host")
    packet_public, host_public = outputs / "packet.pub", outputs / "host.pub"
    packet_public.write_bytes(packet_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
    host_public.write_bytes(host_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
    agent_bytes = Path(physical_qualification.__file__).read_bytes()
    now = int(time.time())
    packet = QualificationPacket(
        schema_version="12-6.physical-qualification-packet.v1",
        packet_id="fabric-simulation", target_git_sha=_head(root), agent_source_sha256=_sha(agent_bytes),
        execution_mode=ExecutionMode.SIMULATION, allowed_os_families=(inventory_host(root).os_family,),
        not_before_epoch_seconds=now - 1, expires_epoch_seconds=now + 300,
        actions=(QualificationAction("simulated-cpu", (target,), 45, 65_536, (ResourceKind.CPU,)),),
        artifact_paths=(_FIXTURE_MARKER,),
    )
    packet_path = outputs / "signed-packet.json"
    packet_path.write_bytes(_json({
        "schema_version": "12-6.signed-physical-qualification-packet.v1",
        "packet": packet.to_dict(),
        "signature": _signature(packet_key, "fixture-packet", packet.to_dict()),
    }))
    verified = load_verified_signed_packet(
        packet_path, signature_verifier=_reopen_public_key(packet_public, "fixture-packet"),
    )
    evidence, log = execute_qualification(
        verified, repo_root=root, agent_source_bytes=agent_bytes,
        evidence_signing_key_id="fixture-host", evidence_signer=lambda _, data, key=host_key: key.sign(data),
    )
    assert evidence["verdict"] == "SIMULATION_PASS"
    evidence_path, log_path = outputs / "simulation.json", outputs / "simulation.log"
    write_evidence_bundle(evidence_path, log_path, evidence=evidence, log_bytes=log)
    del packet_key, host_key
    readback = {
        "verified_packet": load_verified_signed_packet(
            packet_path, signature_verifier=_reopen_public_key(packet_public, "fixture-packet"),
        ),
        "agent_source_bytes": agent_bytes,
        "artifact_root": root,
        "evidence_signature_verifier": _reopen_public_key(host_public, "fixture-host"),
    }
    assert verify_qualification_evidence(
        evidence_path, log_path, require_real_pass=False, **readback,
    )["evidence_identity_sha256"] == evidence["evidence_identity_sha256"]
    with pytest.raises(ValueError, match="simulation|physical PASS"):
        verify_qualification_evidence(evidence_path, log_path, require_real_pass=True, **readback)
    return evidence["evidence_identity_sha256"]


def test_local_fabric_campaign(tmp_path):
    root = tmp_path / "repo"
    registry, evidence_policy = _registry_policy()
    scenario = load_sil_scenario(_ROOT / "configs/control/sil_scenario_v1.json")
    environment = canonical_sil_environment_receipt_v1()
    repair_policy = load_ai_qa_policy(_ROOT / "configs/control/ai_qa_policy_v1.json")
    sil_inputs = {"registry": registry, "scenario": scenario, "environment": environment}
    vectors = build_sil_plan(registry, scenario).vectors
    first_target = _make_fixture_repo(root, vectors)
    failing_sha = _head(root)
    before_package, before, before_path, before_log = _run_local_sil_fixture(
        root, tmp_path, "before", **sil_inputs,
    )
    assert before["verdict"] == "FAIL"
    assert [row["return_code"] for row in before["executions"]] == [1] + [0] * (len(vectors) - 1)
    failure = failure_packet_from_sil(
        before_path, before_log, defect_id="fabric-fixture-defect", policy=repair_policy,
        expected_package_bytes=before_package, expected_environment_receipt=environment,
        expected_registry=registry, expected_scenario=scenario, expected_git_sha=failing_sha,
    )
    journal = EvidenceLineage(tmp_path / "lineage")
    observed = journal.record_failure(
        failure, artifacts=(_retained(before_path, failure.source_evidence_identity_sha256, "local-sil-fixture"),),
    )
    marker = root / _FIXTURE_MARKER
    marker.write_bytes(b"READY\n")
    patch = _git(root, "diff", "--binary", "--", _FIXTURE_MARKER)
    marker.write_bytes(b"BROKEN\n")
    candidate, branch = materialize_local_repair_candidate(
        failure, repo_root=root, patch_bytes=patch, proposer_actor_id="fixture-repair", policy=repair_policy,
    )
    _git(root, "switch", branch)
    after_package = build_package_manifest_bytes(root)
    package_sha = _sha(after_package)
    frozen = journal.freeze_candidate(
        observed, candidate, package_sha256=package_sha,
        artifacts=(EvidenceArtifact(_sha(patch), _sha(patch), len(patch), "local-git-patch", patch),),
    )
    freeze, freeze_sha = freeze_candidate(
        root, base_sha=failing_sha, expected_head_sha=candidate.candidate_git_sha,
        owned_paths=(_FIXTURE_MARKER,), dependency_paths=(first_target,),
    )
    regression = build_regression_chain(
        failure, candidate, adversarial_command=f"pytest -q {vectors[1].argv[4]}",
    )
    automated = execute_automated_regressions(regression, repo_root=root, actor_id="fixture-regression")
    assert all(receipt.verdict is GateVerdict.PASS for receipt in automated)
    package, after, after_path, after_log = _run_local_sil_fixture(root, tmp_path, "after", **sil_inputs)
    assert package == after_package and after["verdict"] == "PASS"
    sil_readback = verify_candidate_sil_evidence(
        after_path, after_log, repo_root=root, candidate_git_sha=candidate.candidate_git_sha,
        expected_environment_receipt=environment, expected_registry=registry, expected_scenario=scenario,
    )
    assert sil_readback["evidence_identity_sha256"] == after["evidence_identity_sha256"]
    sil_receipt = GateReceipt(
        GateKind.SIL, GateVerdict.PASS, candidate.candidate_git_sha,
        sil_readback["evidence_identity_sha256"], "fixture-sil-verifier",
    )
    physical_scope = GateReceipt(
        GateKind.PHYSICAL, GateVerdict.NOT_APPLICABLE, candidate.candidate_git_sha,
        _sha(b"fixture software-only repair"), "fixture-scope-reviewer", "fixture software-only repair",
    )
    retests = tuple(journal.record_retest(frozen, receipt) for receipt in (*automated, sil_receipt, physical_scope))
    subject = "training-mechanics"
    decision = journal.record_promotion_decision(
        frozen, retests, failure=failure, candidate=candidate, certifier_actor_id="fixture-certifier",
        repair_policy=repair_policy, evidence_policy=evidence_policy, capability_id=subject, evidence=(),
    )
    snapshot = journal.readback(defect_identity(failure))
    outcome = snapshot["records"][decision]["payload"]["decision"]
    assert outcome["decision"] == "BLOCK"
    assert outcome["repair_gate"]["decision"] == "READY_FOR_INDEPENDENT_PROMOTION"
    assert EvidenceLineage(journal.root).readback(
        defect_identity(failure), expected_snapshot_sha256=snapshot["snapshot_sha256"],
    ) == snapshot
    simulation = _physical_simulation(root, tmp_path, first_target)
    preview, preview_sha = prepare_integration(
        root, freeze, expected_freeze_sha256=freeze_sha,
        qualified_head_sha=candidate.candidate_git_sha, qualified_tree_sha=freeze.tree_sha,
        qualification_sha256=snapshot["snapshot_sha256"], observed_pr_head_sha=candidate.candidate_git_sha,
        live_base_sha=failing_sha, mergeable=True, main_ref="refs/heads/main",
    )
    _git(root, "switch", "main")
    _git(root, "merge", "--squash", branch)
    _git(root, "commit", "-m", "Accept qualified local fixture repair")
    accepted = verify_accepted_integration(
        root, preview, expected_preview_sha256=preview_sha, accepted_sha=_head(root), observed_main_sha=_head(root),
    )
    assert accepted["accepted_tree_sha"] == freeze.tree_sha
    (tmp_path / "local-campaign.json").write_bytes(_json({
        "schema_version": "12-6.plan8-local-qualification-campaign.v1",
        "scope": "LOCAL_SIL_CONTRACT_FIXTURE_AND_PHYSICAL_SIMULATION",
        "qualification_source_sha": _head(_ROOT),
        "qualification_tree_sha": _git(_ROOT, "rev-parse", "HEAD^{tree}").decode().strip(),
        "release_sha256": _RELEASE_SHA256,
        "platform": sys.platform,
        "vectors": len(vectors),
        "failing_evidence_sha256": before["evidence_identity_sha256"],
        "retested_evidence_sha256": after["evidence_identity_sha256"],
        "simulation_evidence_sha256": simulation,
        "lineage_snapshot_sha256": snapshot["snapshot_sha256"],
        "promotion_decision": outcome["decision"],
        "fixture_accepted_integration": accepted,
    }) + b"\n")


def test_exact_release_reopens_repository_and_historical_provenance(tmp_path):
    release = verify_qualification_fabric_release(_ROOT, _RELEASE, expected_sha256=_RELEASE_SHA256)
    assert release["source"]["git_sha"] == "65986e78989d13430dbc32590919d014ad0a9372"
    assert release["historical_linux_sil"]["artifact_id"] == 11762209803
    assert release["historical_linux_sil"]["accepted_sha"] != release["source"]["git_sha"]
    path = tmp_path / "release.json"
    release.pop("release_sha256")
    release["components"][0]["sha256"] = "0" * 64
    modified = _json(release) + b"\n"
    path.write_bytes(modified)
    with pytest.raises(ValueError, match="trusted SHA-256"):
        verify_qualification_fabric_release(_ROOT, path, expected_sha256=_RELEASE_SHA256)
    with pytest.raises(ValueError, match="exact Git component objects"):
        verify_qualification_fabric_release(_ROOT, path, expected_sha256=_sha(modified))
