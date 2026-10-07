from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import twelve_six.ai_qa_control as ai_qa_control

from twelve_six.ai_qa_control import (
    ExternalObservation,
    FailureClass,
    FailureSource,
    GateKind,
    GateReceipt,
    GateVerdict,
    PhysicalScope,
    RepairCandidate,
    _validate_repair_index_entries,
    build_regression_chain,
    build_repair_candidate,
    classify_failure,
    evaluate_promotion,
    execute_automated_regressions,
    failure_packet_from_observation,
    failure_packet_from_sil,
    load_ai_qa_policy,
    load_external_observation,
    load_failure_packet,
    load_gate_receipt_bundle,
    load_repair_candidate,
    materialize_local_repair_candidate,
    verify_candidate_sil_evidence,
)
from twelve_six.capability_map import load_capability_registry
from twelve_six.sil_qualification import (
    CommandExecution,
    GitState,
    build_package_manifest_bytes,
    canonical_sil_environment_receipt_v1,
    load_sil_scenario,
    qualify_sil,
)


_ROOT = Path(__file__).parents[1]
_POLICY = _ROOT / "configs" / "control" / "ai_qa_policy_v1.json"
_CAPABILITIES = _ROOT / "configs" / "control" / "product_capabilities_v1.json"
_SCENARIO = _ROOT / "configs" / "control" / "sil_scenario_v1.json"
_FAIL_SHA = "a" * 40
_CANDIDATE_SHA = "b" * 40


def _policy():
    return load_ai_qa_policy(_POLICY)


def _package_bytes() -> bytes:
    return build_package_manifest_bytes(_ROOT)


def _environment_receipt() -> dict[str, object]:
    return canonical_sil_environment_receipt_v1()


def _ci_observation(*, physical: bool = False) -> ExternalObservation:
    return ExternalObservation(
        schema_version="12-6.aiqa-observation.v1",
        source=FailureSource.PHYSICAL if physical else FailureSource.CI,
        git_sha=_FAIL_SHA,
        evidence_identity_sha256="c" * 64,
        failure_summary=(
            "physical keyboard journey failed"
            if physical
            else "AssertionError: deterministic contract failed"
        ),
        reproducer_command="pytest -q tests/test_sil_qualification_section3.py",
        physical_gate_id="windows-nvda-keyboard" if physical else None,
    )


def _failure(*, physical: bool = False):
    return failure_packet_from_observation(
        _ci_observation(physical=physical),
        defect_id="defect-section4-regression",
        policy=_policy(),
    )


def _candidate(failure):
    return build_repair_candidate(
        failure,
        base_git_sha=_FAIL_SHA,
        candidate_git_sha=_CANDIDATE_SHA,
        patch_bytes=b"diff --git a/x b/x\n+repair\n",
        proposer_actor_id="repair-agent",
        policy=_policy(),
    )


def _pass_runner(
    argv: tuple[str, ...],
    cwd: Path,
    timeout_seconds: int,
    input_envelope_bytes: bytes,
    expected_input_identity_sha256: str,
) -> CommandExecution:
    assert argv[1:4] == ("-m", "pytest", "-q")
    assert cwd
    assert timeout_seconds == 300
    assert hashlib.sha256(input_envelope_bytes).hexdigest() == (
        expected_input_identity_sha256
    )
    return CommandExecution(
        return_code=0,
        stdout="passed",
        stderr="",
        duration_ms=3,
        consumed_input_identity_sha256=expected_input_identity_sha256,
    )


def _candidate_git_probe(_: str | Path) -> GitState:
    return GitState(sha=_CANDIDATE_SHA, tracked_clean=True)


def _candidate_parent_probe(_: Path, __: str) -> tuple[str, ...]:
    return (_FAIL_SHA,)


@pytest.mark.parametrize(
    ("source", "summary", "expected"),
    (
        (FailureSource.CI, "job timed out after 300 seconds", FailureClass.TIMEOUT),
        (FailureSource.SIL, "checkpoint SHA mismatch", FailureClass.INTEGRITY),
        (FailureSource.CI, "ModuleNotFoundError: missing dependency", FailureClass.ENVIRONMENT),
        (FailureSource.CI, "AssertionError: output failed", FailureClass.TEST),
        (FailureSource.CI, "unexpected opaque signal", FailureClass.UNKNOWN),
        (FailureSource.PHYSICAL, "keyboard path failed", FailureClass.PHYSICAL),
    ),
)
def test_failure_classification_is_deterministic(
    source: FailureSource,
    summary: str,
    expected: FailureClass,
) -> None:
    assert classify_failure(source, summary) is expected


def test_external_ci_and_physical_observations_become_exact_failure_packets() -> None:
    ci = _failure()
    assert ci.source is FailureSource.CI
    assert ci.failure_class is FailureClass.TEST
    assert ci.failing_git_sha == _FAIL_SHA
    assert ci.reproducer_argv[1:4] == ("-m", "pytest", "-q")
    assert ci.physical_scope is PhysicalScope.NONE
    assert ci.physical_gate_id is None
    assert len(ci.identity_sha256()) == 64

    physical = _failure(physical=True)
    assert physical.source is FailureSource.PHYSICAL
    assert physical.failure_class is FailureClass.PHYSICAL
    assert physical.physical_scope is PhysicalScope.REQUIRED
    assert physical.physical_gate_id == "windows-nvda-keyboard"


def test_external_observation_loader_rejects_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"12-6.aiqa-observation.v1",'
        '"source":"CI","source":"CI","git_sha":"' + _FAIL_SHA + '",'
        '"evidence_identity_sha256":"' + "c" * 64 + '",'
        '"failure_summary":"failed","reproducer_command":'
        '"pytest -q tests/test_sil_qualification_section3.py",'
        '"physical_gate_id":null}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict unambiguous"):
        load_external_observation(duplicate)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {
                "schema_version": "12-6.aiqa-observation.v1",
                "source": "CI",
                "git_sha": _FAIL_SHA,
                "evidence_identity_sha256": "c" * 64,
                "failure_summary": "failed",
                "reproducer_command": (
                    "pytest -q tests/test_sil_qualification_section3.py"
                ),
                "physical_gate_id": None,
                "forged": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields"):
        load_external_observation(unknown)


def test_native_sil_fail_evidence_yields_minimal_failed_vector_reproducer(
    tmp_path: Path,
) -> None:
    calls = 0

    def fail_first(
        argv: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int,
        input_envelope_bytes: bytes,
        expected_input_identity_sha256: str,
    ) -> CommandExecution:
        nonlocal calls
        calls += 1
        if calls == 1:
            return CommandExecution(
                return_code=9,
                stdout="",
                stderr="AssertionError: integration failed",
                duration_ms=2,
                consumed_input_identity_sha256=expected_input_identity_sha256,
            )
        return _pass_runner(
            argv,
            cwd,
            timeout_seconds,
            input_envelope_bytes,
            expected_input_identity_sha256,
        )

    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_FAIL_SHA,
        registry=load_capability_registry(_CAPABILITIES),
        scenario=load_sil_scenario(_SCENARIO),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
        command_runner=fail_first,
        git_probe=lambda _: GitState(sha=_FAIL_SHA, tracked_clean=True),
    )
    assert evidence["verdict"] == "FAIL"

    evidence_path = tmp_path / "sil-evidence.json"
    log_path = tmp_path / "sil.log"
    evidence_path.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text(log_text, encoding="utf-8")

    packet = failure_packet_from_sil(
        evidence_path,
        log_path,
        defect_id="sil-integration-defect",
        policy=_policy(),
        expected_package_bytes=_package_bytes(),
        expected_environment_receipt=_environment_receipt(),
        expected_registry=load_capability_registry(_CAPABILITIES),
        expected_scenario=load_sil_scenario(_SCENARIO),
    )
    assert packet.source is FailureSource.SIL
    assert packet.failing_git_sha == _FAIL_SHA
    assert packet.reproducer_argv[1:4] == ("-m", "pytest", "-q")
    assert packet.reproducer_argv[4:]
    assert packet.source_evidence_identity_sha256 == evidence[
        "evidence_identity_sha256"
    ]


def test_sil_failure_ingestion_rejects_environment_authority_mismatch(
    tmp_path: Path,
) -> None:
    registry = load_capability_registry(_CAPABILITIES)
    scenario = load_sil_scenario(_SCENARIO)

    def fail_runner(
        argv: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int,
        input_envelope_bytes: bytes,
        expected_input_identity_sha256: str,
    ) -> CommandExecution:
        assert argv and cwd and timeout_seconds
        assert input_envelope_bytes
        return CommandExecution(
            return_code=9,
            stdout="",
            stderr="AssertionError: integration failed",
            duration_ms=1,
            consumed_input_identity_sha256=expected_input_identity_sha256,
        )

    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_FAIL_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
        command_runner=fail_runner,
        git_probe=lambda _: GitState(sha=_FAIL_SHA, tracked_clean=True),
    )
    evidence_path = tmp_path / "sil-evidence.json"
    log_path = tmp_path / "sil.log"
    evidence_path.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text(log_text, encoding="utf-8")

    forged_environment = dict(_environment_receipt())
    forged_environment["identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="environment receipt|exact accepted authority"):
        failure_packet_from_sil(
            evidence_path,
            log_path,
            defect_id="sil-environment-authority-mismatch",
            policy=_policy(),
            expected_package_bytes=_package_bytes(),
            expected_environment_receipt=forged_environment,
            expected_registry=registry,
            expected_scenario=scenario,
        )


def test_candidate_sil_verifier_rejects_checkout_drift_after_verification(
    tmp_path: Path,
) -> None:
    registry = load_capability_registry(_CAPABILITIES)
    scenario = load_sil_scenario(_SCENARIO)
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_CANDIDATE_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
        command_runner=_pass_runner,
        git_probe=lambda _: GitState(sha=_CANDIDATE_SHA, tracked_clean=True),
    )
    evidence_path = tmp_path / "candidate-sil-evidence.json"
    log_path = tmp_path / "candidate-sil.log"
    evidence_path.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text(log_text, encoding="utf-8")

    states = iter(
        (
            GitState(sha=_CANDIDATE_SHA, tracked_clean=True),
            GitState(sha=_CANDIDATE_SHA, tracked_clean=False),
        )
    )
    with pytest.raises(ValueError, match="dirty"):
        verify_candidate_sil_evidence(
            evidence_path,
            log_path,
            repo_root=_ROOT,
            candidate_git_sha=_CANDIDATE_SHA,
            expected_environment_receipt=_environment_receipt(),
            expected_registry=registry,
            expected_scenario=scenario,
            git_probe=lambda _: next(states),
        )


def test_repair_candidate_is_bound_to_failure_patch_and_exact_candidate_sha() -> None:
    failure = _failure()
    first = _candidate(failure)
    second = build_repair_candidate(
        failure,
        base_git_sha=_FAIL_SHA,
        candidate_git_sha=_CANDIDATE_SHA,
        patch_bytes=b"different repair",
        proposer_actor_id="repair-agent",
        policy=_policy(),
    )

    assert first.failure_packet_identity_sha256 == failure.identity_sha256()
    assert first.candidate_git_sha == _CANDIDATE_SHA
    assert first.identity_sha256() != second.identity_sha256()

    with pytest.raises(ValueError, match="base Git SHA must equal failing Git SHA"):
        build_repair_candidate(
            failure,
            base_git_sha="d" * 40,
            candidate_git_sha=_CANDIDATE_SHA,
            patch_bytes=b"wrong-base repair",
            proposer_actor_id="repair-agent",
            policy=_policy(),
        )


def test_regression_chain_runs_component_then_adversarial_on_exact_candidate() -> None:
    failure = _failure()
    candidate = _candidate(failure)
    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command="pytest -q tests/test_ai_qa_control_section4.py",
    )

    component, adversarial = execute_automated_regressions(
        chain,
        repo_root=_ROOT,
        actor_id="regression-automation",
        command_runner=_pass_runner,
        git_probe=_candidate_git_probe,
        candidate_parent_probe=_candidate_parent_probe,
    )

    assert component.gate is GateKind.COMPONENT
    assert component.verdict is GateVerdict.PASS
    assert adversarial.gate is GateKind.ADVERSARIAL
    assert adversarial.verdict is GateVerdict.PASS
    assert component.git_sha == _CANDIDATE_SHA
    assert adversarial.git_sha == _CANDIDATE_SHA


def test_regression_chain_rejects_wrong_sha_dirty_tree_and_shell_reproducer() -> None:
    failure = _failure()
    candidate = _candidate(failure)

    with pytest.raises(ValueError):
        build_regression_chain(
            failure,
            candidate,
            adversarial_command=(
                "pytest -q tests/test_ai_qa_control_section4.py; echo forged"
            ),
        )

    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command="pytest -q tests/test_ai_qa_control_section4.py",
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        execute_automated_regressions(
            chain,
            repo_root=_ROOT,
            actor_id="regression-automation",
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha="d" * 40, tracked_clean=True),
        )
    with pytest.raises(ValueError, match="direct child"):
        execute_automated_regressions(
            chain,
            repo_root=_ROOT,
            actor_id="regression-automation",
            command_runner=_pass_runner,
            git_probe=_candidate_git_probe,
            candidate_parent_probe=lambda _root, _sha: ("d" * 40,),
        )
    with pytest.raises(ValueError, match="dirty"):
        execute_automated_regressions(
            chain,
            repo_root=_ROOT,
            actor_id="regression-automation",
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha=_CANDIDATE_SHA, tracked_clean=False),
        )


def test_regression_chain_rejects_checkout_mutation_during_gate() -> None:
    failure = _failure()
    candidate = _candidate(failure)
    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command="pytest -q tests/test_ai_qa_control_section4.py",
    )
    states = iter(
        (
            GitState(sha=_CANDIDATE_SHA, tracked_clean=True),
            GitState(sha=_CANDIDATE_SHA, tracked_clean=True),
            GitState(sha=_CANDIDATE_SHA, tracked_clean=False),
        )
    )

    with pytest.raises(ValueError, match="became dirty during component gate"):
        execute_automated_regressions(
            chain,
            repo_root=_ROOT,
            actor_id="regression-automation",
            command_runner=_pass_runner,
            git_probe=lambda _: next(states),
            candidate_parent_probe=_candidate_parent_probe,
        )


def _pass_receipt(
    gate: GateKind,
    *,
    actor_id: str,
    verdict: GateVerdict = GateVerdict.PASS,
    reason: str | None = None,
    git_sha: str = _CANDIDATE_SHA,
) -> GateReceipt:
    return GateReceipt(
        gate=gate,
        verdict=verdict,
        git_sha=git_sha,
        evidence_identity_sha256={
            GateKind.COMPONENT: "1" * 64,
            GateKind.ADVERSARIAL: "2" * 64,
            GateKind.SIL: "3" * 64,
            GateKind.PHYSICAL: "4" * 64,
        }[gate],
        actor_id=actor_id,
        reason=reason,
    )


def test_complete_gate_chain_is_only_ready_for_independent_promotion() -> None:
    failure = _failure()
    candidate = _candidate(failure)
    receipts = (
        _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation"),
        _pass_receipt(GateKind.ADVERSARIAL, actor_id="regression-automation"),
        _pass_receipt(GateKind.SIL, actor_id="independent-sil"),
        _pass_receipt(
            GateKind.PHYSICAL,
            actor_id="physical-scope-auditor",
            verdict=GateVerdict.NOT_APPLICABLE,
            reason="defect has explicit software-only physical scope NONE",
        ),
    )

    decision = evaluate_promotion(
        failure,
        candidate,
        receipts,
        certifier_actor_id="independent-certifier",
        policy=_policy(),
    )
    assert decision.decision == "READY_FOR_INDEPENDENT_PROMOTION"
    assert decision.decision != "PROMOTE"
    assert decision.reasons == ()
    assert len(decision.identity_sha256()) == 64


def test_same_proposer_missing_failed_or_wrong_sha_evidence_blocks_promotion() -> None:
    failure = _failure()
    candidate = _candidate(failure)
    receipts = (
        _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation"),
        _pass_receipt(
            GateKind.ADVERSARIAL,
            actor_id="regression-automation",
            verdict=GateVerdict.FAIL,
        ),
        _pass_receipt(
            GateKind.SIL,
            actor_id="repair-agent",
            git_sha="d" * 40,
        ),
    )

    decision = evaluate_promotion(
        failure,
        candidate,
        receipts,
        certifier_actor_id="repair-agent",
        policy=_policy(),
    )
    assert decision.decision == "BLOCK"
    joined = " | ".join(decision.reasons)
    assert "certifier is the repair proposer" in joined
    assert "adversarial gate is not PASS" in joined
    assert "sil receipt is bound to a different Git SHA" in joined
    assert "missing physical gate receipt" in joined
    assert "sil evidence is not independent" in joined


def test_required_physical_gate_cannot_be_resealed_not_applicable() -> None:
    failure = _failure(physical=True)
    candidate = _candidate(failure)
    receipts = (
        _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation"),
        _pass_receipt(GateKind.ADVERSARIAL, actor_id="regression-automation"),
        _pass_receipt(GateKind.SIL, actor_id="independent-sil"),
        _pass_receipt(
            GateKind.PHYSICAL,
            actor_id="physical-auditor",
            verdict=GateVerdict.NOT_APPLICABLE,
            reason="attempted bypass",
        ),
    )

    decision = evaluate_promotion(
        failure,
        candidate,
        receipts,
        certifier_actor_id="independent-certifier",
        policy=_policy(),
    )
    assert decision.decision == "BLOCK"
    assert "required physical gate is not PASS" in decision.reasons


def test_durable_failure_candidate_and_receipt_bundles_reject_resealing(
    tmp_path: Path,
) -> None:
    failure = _failure()
    candidate = _candidate(failure)

    failure_payload = failure.to_dict()
    failure_payload["failure_packet_identity_sha256"] = failure.identity_sha256()
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(
        json.dumps(failure_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert load_failure_packet(failure_path) == failure

    candidate_payload = candidate.to_dict()
    candidate_payload["candidate_identity_sha256"] = candidate.identity_sha256()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert load_repair_candidate(candidate_path) == candidate

    receipt = _pass_receipt(GateKind.SIL, actor_id="independent-sil")
    bundle_path = tmp_path / "receipts.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "12-6.aiqa-gate-receipts.v1",
                "candidate_identity_sha256": candidate.identity_sha256(),
                "receipts": [receipt.to_dict()],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_gate_receipt_bundle(
        bundle_path,
        expected_candidate_identity_sha256=candidate.identity_sha256(),
        trusted_receipts=(receipt,),
    ) == (receipt,)

    resealed = dict(candidate_payload)
    resealed["patch_sha256"] = "f" * 64
    candidate_path.write_text(json.dumps(resealed), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_repair_candidate(candidate_path)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["candidate_identity_sha256"] = "e" * 64
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate identity mismatch"):
        load_gate_receipt_bundle(
            bundle_path,
            expected_candidate_identity_sha256=candidate.identity_sha256(),
            trusted_receipts=(receipt,),
        )


@pytest.mark.parametrize(
    "bad_argv",
    [
        (
            "/bin/sh",
            "-m",
            "pytest",
            "-q",
            "tests/test_ai_qa_control_section4.py",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "../outside.py",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "tests/test_ai_qa_control_section4.py",
        ),
    ],
)
def test_durable_failure_packet_rejects_self_consistent_noncanonical_reproducer(
    tmp_path: Path,
    bad_argv: tuple[str, ...],
) -> None:
    failure = _failure()
    payload = failure.to_dict()
    payload["reproducer_argv"] = list(bad_argv)
    payload["failure_packet_identity_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "forged-failure.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical pytest argv|current interpreter"):
        load_failure_packet(path)


def test_receipt_bundle_cannot_promote_hand_authored_pass_hash(tmp_path: Path) -> None:
    candidate = _candidate(_failure())
    trusted = _pass_receipt(GateKind.SIL, actor_id="independent-sil")
    forged = GateReceipt(
        gate=GateKind.SIL,
        verdict=GateVerdict.PASS,
        git_sha=_CANDIDATE_SHA,
        evidence_identity_sha256="f" * 64,
        actor_id="independent-sil",
    )
    bundle_path = tmp_path / "forged-receipt.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "12-6.aiqa-gate-receipts.v1",
                "candidate_identity_sha256": candidate.identity_sha256(),
                "receipts": [forged.to_dict()],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match live trusted evidence"):
        load_gate_receipt_bundle(
            bundle_path,
            expected_candidate_identity_sha256=candidate.identity_sha256(),
            trusted_receipts=(trusted,),
        )


def test_regression_chain_rejects_durable_wrong_base_even_if_failure_hash_matches() -> None:
    failure = _failure()
    candidate = RepairCandidate(
        schema_version=1,
        defect_id=failure.defect_id,
        base_git_sha="d" * 40,
        candidate_git_sha=_CANDIDATE_SHA,
        patch_sha256="e" * 64,
        proposer_actor_id="repair-agent",
        failure_packet_identity_sha256=failure.identity_sha256(),
    )
    with pytest.raises(ValueError, match="base Git SHA does not match failing Git SHA"):
        build_regression_chain(
            failure,
            candidate,
            adversarial_command="pytest -q tests/test_ai_qa_control_section4.py",
        )


def test_repair_index_rejects_symlink_and_gitlink_modes() -> None:
    regular = (
        ":100644 100755 " + ("a" * 40) + " " + ("b" * 40) + " M\ttools/run.py"
    )
    _validate_repair_index_entries(regular)

    for mode, path in (
        ("120000", "src/twelve_six/forged.py"),
        ("160000", "vendor/forged-submodule"),
    ):
        raw = (
            ":000000 "
            + mode
            + " "
            + ("0" * 40)
            + " "
            + ("c" * 40)
            + " A\t"
            + path
        )
        with pytest.raises(ValueError, match="regular Git files"):
            _validate_repair_index_entries(raw)


def test_materialize_local_repair_candidate_creates_exact_base_isolated_branch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Section 4 test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "section4@example.invalid"],
        cwd=repo,
        check=True,
    )
    target = repo / "payload.txt"
    target.write_text("broken\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "failing base"], cwd=repo, check=True)
    failing_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    target.write_text("repaired\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "payload.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["git", "checkout", "--", "payload.txt"], cwd=repo, check=True)

    observation = ExternalObservation(
        schema_version="12-6.aiqa-observation.v1",
        source=FailureSource.CI,
        git_sha=failing_sha,
        evidence_identity_sha256="a" * 64,
        failure_summary="AssertionError: payload is broken",
        reproducer_command="pytest -q tests/test_ai_qa_control_section4.py",
        physical_gate_id=None,
    )
    failure = failure_packet_from_observation(
        observation,
        defect_id="materialized-repair",
        policy=_policy(),
    )

    hooks = tmp_path / "hostile-hooks"
    hooks.mkdir()
    hook_sentinel = tmp_path / "hook-executed"
    for hook_name in ("pre-commit", "post-checkout", "reference-transaction"):
        hook = hooks / hook_name
        hook.write_text(
            "#!/bin/sh\\nprintf hook-ran > "
            + str(hook_sentinel)
            + "\\nexit 91\\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)
    subprocess.run(
        ["git", "config", "core.hooksPath", str(hooks)],
        cwd=repo,
        check=True,
    )

    candidate, branch_name = materialize_local_repair_candidate(
        failure,
        repo_root=repo,
        patch_bytes=patch,
        proposer_actor_id="repair-agent",
        policy=_policy(),
    )
    assert candidate.base_git_sha == failing_sha
    assert candidate.patch_sha256 == hashlib.sha256(patch).hexdigest()
    assert branch_name.startswith("aiqa/repair/")
    branch_sha = subprocess.run(
        ["git", "rev-parse", branch_name],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch_sha == candidate.candidate_git_sha
    parent_sha = subprocess.run(
        ["git", "rev-parse", f"{branch_name}^"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert parent_sha == failing_sha
    assert not hook_sentinel.exists()
    repaired = subprocess.run(
        ["git", "show", f"{branch_name}:payload.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert repaired == "repaired\n"

    repeated, repeated_branch = materialize_local_repair_candidate(
        failure,
        repo_root=repo,
        patch_bytes=patch,
        proposer_actor_id="repair-agent",
        policy=_policy(),
    )
    assert repeated == candidate
    assert repeated_branch == branch_name


def test_live_local_defect_repair_retest_round_trip_uses_exact_candidate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "roundtrip"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Section 4 roundtrip"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "section4-roundtrip@example.invalid"],
        cwd=repo,
        check=True,
    )
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (repo / "payload.txt").write_text("broken\n", encoding="utf-8")
    (tests_dir / "test_payload.py").write_text(
        "from pathlib import Path\n\n"
        "def test_payload_is_repaired() -> None:\n"
        "    assert Path('payload.txt').read_text(encoding='utf-8') == 'repaired\\n'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "failing fixture"], cwd=repo, check=True)
    failing_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    failing_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_payload.py"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failing_run.returncode != 0

    observation = ExternalObservation(
        schema_version="12-6.aiqa-observation.v1",
        source=FailureSource.CI,
        git_sha=failing_sha,
        evidence_identity_sha256=hashlib.sha256(
            (failing_run.stdout + failing_run.stderr).encode("utf-8")
        ).hexdigest(),
        failure_summary="AssertionError: payload is broken",
        reproducer_command="pytest -q tests/test_payload.py",
        physical_gate_id=None,
    )
    failure = failure_packet_from_observation(
        observation,
        defect_id="live-local-roundtrip",
        policy=_policy(),
    )

    payload = repo / "payload.txt"
    payload.write_text("repaired\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "payload.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["git", "checkout", "--", "payload.txt"], cwd=repo, check=True)

    candidate, branch_name = materialize_local_repair_candidate(
        failure,
        repo_root=repo,
        patch_bytes=patch,
        proposer_actor_id="repair-agent",
        policy=_policy(),
    )
    subprocess.run(["git", "switch", branch_name], cwd=repo, check=True, capture_output=True)
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == candidate.candidate_git_sha

    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command="pytest -q tests/test_payload.py",
    )
    component, adversarial = execute_automated_regressions(
        chain,
        repo_root=repo,
        actor_id="independent-regression-runner",
        timeout_seconds=60,
    )

    assert component.verdict is GateVerdict.PASS
    assert adversarial.verdict is GateVerdict.PASS
    assert component.git_sha == candidate.candidate_git_sha
    assert adversarial.git_sha == candidate.candidate_git_sha
    assert component.evidence_identity_sha256 != adversarial.evidence_identity_sha256
    assert payload.read_text(encoding="utf-8") == "repaired\n"

def test_repair_path_guard_blocks_qualification_trust_roots() -> None:
    protected = (
        ".gitignore",
        "tests/test_payload.py",
        ".github/workflows/ci.yml",
        "configs/control/ai_qa_policy_v1.json",
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "tools/validate_section2_repository_surface_coverage.py",
        "pyproject.toml",
        "SEQUENTIAL_CLOSURE_STATE.md",
        "src/twelve_six/ai_qa_control.py",
        "src/twelve_six/capability_map.py",
        "src/twelve_six/ci_workflow_policy.py",
        "src/twelve_six/sil_qualification.py",
    )
    for path in protected:
        with pytest.raises(ValueError, match="qualification trust-root"):
            ai_qa_control._validate_repair_paths(path + "\0")

    ai_qa_control._validate_repair_paths("src/twelve_six/model.py\0")

    with pytest.raises(ValueError, match="NUL delimiter"):
        ai_qa_control._validate_repair_paths("src/twelve_six/model.py")
    with pytest.raises(ValueError, match="non-canonical"):
        ai_qa_control._validate_repair_paths("src/twelve_six/../tests/test_payload.py\0")


def test_materializer_rejects_patch_that_rewrites_reproducer_test(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "trust-root-repair"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Section 4 trust-root"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "section4-trust-root@example.invalid"],
        cwd=repo,
        check=True,
    )
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_payload.py"
    target.write_text(
        "def test_payload() -> None:\n"
        "    assert False, 'original failure'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "failing test authority"], cwd=repo, check=True)
    failing_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    observation = ExternalObservation(
        schema_version="12-6.aiqa-observation.v1",
        source=FailureSource.CI,
        git_sha=failing_sha,
        evidence_identity_sha256="a" * 64,
        failure_summary="AssertionError: original failure",
        reproducer_command="pytest -q tests/test_payload.py",
        physical_gate_id=None,
    )
    failure = failure_packet_from_observation(
        observation,
        defect_id="trust-root-rewrite",
        policy=_policy(),
    )

    target.write_text(
        "def test_payload() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", "tests/test_payload.py"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        ["git", "checkout", "--", "tests/test_payload.py"],
        cwd=repo,
        check=True,
    )

    with pytest.raises(ValueError, match="qualification trust-root"):
        materialize_local_repair_candidate(
            failure,
            repo_root=repo,
            patch_bytes=patch,
            proposer_actor_id="repair-agent",
            policy=_policy(),
        )

