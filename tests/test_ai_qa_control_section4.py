from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.ai_qa_control import (
    ExternalObservation,
    FailureClass,
    FailureSource,
    GateKind,
    GateReceipt,
    GateVerdict,
    PhysicalScope,
    build_regression_chain,
    build_repair_candidate,
    classify_failure,
    evaluate_promotion,
    execute_automated_regressions,
    failure_packet_from_observation,
    failure_packet_from_sil,
    load_ai_qa_policy,
    load_external_observation,
)
from twelve_six.capability_map import load_capability_registry
from twelve_six.sil_qualification import (
    CommandExecution,
    GitState,
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
) -> CommandExecution:
    assert argv[1:4] == ("-m", "pytest", "-q")
    assert cwd
    assert timeout_seconds == 300
    return CommandExecution(
        return_code=0,
        stdout="passed",
        stderr="",
        duration_ms=3,
    )


def _candidate_git_probe(_: str | Path) -> GitState:
    return GitState(sha=_CANDIDATE_SHA, tracked_clean=True)


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
    ) -> CommandExecution:
        nonlocal calls
        calls += 1
        if calls == 1:
            return CommandExecution(
                return_code=9,
                stdout="",
                stderr="AssertionError: integration failed",
                duration_ms=2,
            )
        return _pass_runner(argv, cwd, timeout_seconds)

    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_FAIL_SHA,
        registry=load_capability_registry(_CAPABILITIES),
        scenario=load_sil_scenario(_SCENARIO),
        package_bytes=b"package",
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
    )
    assert packet.source is FailureSource.SIL
    assert packet.failing_git_sha == _FAIL_SHA
    assert packet.reproducer_argv[1:4] == ("-m", "pytest", "-q")
    assert packet.reproducer_argv[4:]
    assert packet.source_evidence_identity_sha256 == evidence[
        "evidence_identity_sha256"
    ]


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
    with pytest.raises(ValueError, match="dirty"):
        execute_automated_regressions(
            chain,
            repo_root=_ROOT,
            actor_id="regression-automation",
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha=_CANDIDATE_SHA, tracked_clean=False),
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


def test_independent_complete_gate_chain_can_promote_software_only_repair() -> None:
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
    assert decision.decision == "PROMOTE"
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
