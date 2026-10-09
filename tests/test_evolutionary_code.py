"""Plan 6 Section 12 LOCAL_FREE isolated code-evolution contract tests."""
from dataclasses import replace

import pytest

from twelve_six.evolutionary_code import (
    CodeCandidate,
    EvolutionPolicy,
    GateEvidence,
    _GATES,
    evaluate_evolution,
    publish_evolution_evidence,
    stage_candidate,
    verify_evolution_restart,
)
from twelve_six.experience_replay import _mac

H = "a" * 64
GIT = "b" * 40


def case(tmp_path):
    candidate = CodeCandidate(
        "fix-001", "model-producer", GIT, "evolution/candidate/fix-001",
        "proposals/patch.txt", "code", "print('safe inert candidate')\n", H,
    )
    policy = EvolutionPolicy(GIT)
    production = tmp_path / "production"
    production.mkdir()
    sandbox = tmp_path / "sandbox"
    verifiers = {g: "host-" + g for g in _GATES}
    keys = {g: bytes([i + 31]) * 32 for i, g in enumerate(_GATES)}
    roots = {g: frozenset({str(i + 1) * 64}) for i, g in enumerate(_GATES)}
    gate_records = []
    for gate in _GATES:
        item = GateEvidence(candidate.identity(), gate, verifiers[gate],
                            H, next(iter(roots[gate])), True, H)
        gate_records.append(replace(item, signature=_mac(keys[gate], item.payload())))
    return candidate, policy, sandbox, production, tuple(gate_records), {
        "trusted_verifiers": verifiers,
        "trusted_keys": keys,
        "trusted_roots": roots,
    }


def stage(s):
    return stage_candidate(s[0], s[1], sandbox_root=s[2], production_root=s[3])


def evaluate(s, gates=None):
    return evaluate_evolution(s[0], s[1], s[4] if gates is None else gates, **s[5])


def restart(s, decision):
    return verify_evolution_restart(
        s[0], s[1], s[4], decision, sandbox_root=s[2],
        production_root=s[3], **s[5],
    )



def test_invalid_directory_types_denied_before_staging(tmp_path):
    s = case(tmp_path)
    with pytest.raises(TypeError, match="explicit local directories required"):
        stage_candidate(s[0], s[1], sandbox_root=str(s[2]), production_root=s[3])
    with pytest.raises(TypeError, match="explicit local directories required"):
        stage_candidate(s[0], s[1], sandbox_root=s[2], production_root=str(s[3]))
    assert not s[2].exists()

def test_isolated_success_and_exact_restart(tmp_path):
    s = case(tmp_path)
    assert stage(s) == s[0].identity() == stage(s)
    path = s[2] / "fix-001" / "proposals" / "patch.txt"
    assert path.read_text() == s[0].text
    result = evaluate(s)
    assert result.state == "EXTERNAL_REVIEW_READY"
    assert not result.promotion_authorized and not result.production_write_authorized
    assert not result.negative_gates
    assert result == evaluate(s)
    assert restart(s, result)
    assert publish_evolution_evidence(
        s[0], s[1], s[4], result, sandbox_root=s[2], production_root=s[3], **s[5]
    ) == result.receipt_sha256
    assert not list(s[3].rglob("*"))
    assert restart(s, result)


def test_missing_gate_and_failed_gate_become_durable_negative_evidence(tmp_path):
    s = case(tmp_path)
    stage(s)
    partial = evaluate(s, s[4][:2])
    assert partial.state == "REJECTED_CANDIDATE"
    assert partial.negative_gates == ("benchmark:MISSING", "review:MISSING")
    bad = replace(s[4][0], passed=False)
    bad = replace(bad, signature=_mac(s[5]["trusted_keys"]["tests"], bad.payload()))
    rejected = evaluate(s, (bad,) + s[4][1:])
    assert rejected.negative_gates == ("tests:FAILED",)
    assert not rejected.production_write_authorized and not rejected.promotion_authorized
    assert publish_evolution_evidence(
        s[0], s[1], (bad,) + s[4][1:], rejected,
        sandbox_root=s[2], production_root=s[3], **s[5],
    ) == rejected.receipt_sha256


def test_tampered_candidate_and_stale_git_parent_fail_closed(tmp_path):
    s = case(tmp_path)
    stage(s)
    assert restart(s, evaluate(s))
    dest = s[2] / "fix-001" / "proposals" / "patch.txt"
    dest.write_text("tampered!")
    with pytest.raises(ValueError, match="drift|changed"):
        restart(s, evaluate(s))
    with pytest.raises(ValueError, match="changed"):
        stage(s)
    changed = replace(s[0], parent_git_sha="c" * 40)
    with pytest.raises(ValueError, match="stale"):
        stage_candidate(changed, s[1], sandbox_root=s[2], production_root=s[3])


def test_candidate_can_never_stage_in_production(tmp_path):
    s = case(tmp_path)
    with pytest.raises(ValueError, match="production"):
        stage_candidate(s[0], s[1], sandbox_root=s[3], production_root=s[3])
    with pytest.raises(ValueError, match="production"):
        stage_candidate(s[0], s[1], sandbox_root=s[3] / "sub", production_root=s[3])
    with pytest.raises(ValueError, match="production"):
        stage_candidate(s[0], s[1], sandbox_root=tmp_path, production_root=s[3])


def test_symlink_attack_is_denied(tmp_path):
    s = case(tmp_path)
    s[2].mkdir()
    (s[2] / "fix-001").symlink_to(s[3], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        stage(s)


def test_path_branch_kind_version_and_content_bounds(tmp_path):
    s = case(tmp_path)
    for field in (
        {"path": "../evil.txt"}, {"path": "/absolute.txt"},
        {"path": "proposals/../evil.txt"}, {"path": "proposals/evil.py"},
        {"path": "proposals/a\\b.txt"}, {"branch": "main"},
        {"candidate_id": "bad/id"}, {"kind": "shell"},
        {"text": "\x00"}, {"text": "x" * 16385},
        {"schema_version": True},
    ):
        with pytest.raises(ValueError):
            replace(s[0], **field).identity()
    with pytest.raises(ValueError, match="policy"):
        stage_candidate(s[0], replace(s[1], max_candidate_bytes=1),
                        sandbox_root=s[2], production_root=s[3])


def test_bad_verifier_signature_and_self_approval_are_denied(tmp_path):
    s = case(tmp_path)
    with pytest.raises(ValueError):
        evaluate(s, (replace(s[4][0], signature=H),) + s[4][1:])
    forged = replace(s[4][0], verifier_id=s[0].producer_id)
    forged = replace(
        forged, signature=_mac(s[5]["trusted_keys"]["tests"], forged.payload())
    )
    with pytest.raises(ValueError):
        evaluate(s, (forged,) + s[4][1:])
    colliding = {g: "same-verifier" for g in _GATES}
    with pytest.raises(ValueError, match="independent"):
        evaluate_evolution(
            s[0], s[1], s[4], **{**s[5], "trusted_verifiers": colliding}
        )


def test_wrong_gate_root_version_and_duplicate_evidence_fail(tmp_path):
    s = case(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(s, (s[4][0], s[4][0]))
    with pytest.raises(ValueError):
        evaluate(s, (replace(s[4][0], evidence_sha256=H),) + s[4][1:])
    with pytest.raises(ValueError):
        evaluate(s, (replace(s[4][0], candidate_sha256=H),) + s[4][1:])
    with pytest.raises(ValueError):
        evaluate(s, (replace(s[4][0], schema_version=True),) + s[4][1:])


def test_forged_decision_promotion_and_replay_denied(tmp_path):
    s = case(tmp_path)
    stage(s)
    good = evaluate(s)
    for fake in (
        replace(good, promotion_authorized=True),
        replace(good, production_write_authorized=True),
        replace(good, receipt_sha256=H),
        replace(good, negative_gates=("tests:FAILED",)),
    ):
        with pytest.raises(ValueError):
            restart(s, fake)


def test_recovery_evidence_write_no_overwrite_and_no_model_side_effect(tmp_path):
    s = case(tmp_path)
    stage(s)
    result = evaluate(s)
    for _ in range(2):
        assert publish_evolution_evidence(
            s[0], s[1], s[4], result,
            sandbox_root=s[2], production_root=s[3], **s[5],
        ) == result.receipt_sha256
    receipt = s[2] / "fix-001" / "decision.json"
    assert receipt.is_file()
    receipt.write_text("forged")
    with pytest.raises(ValueError, match="changed"):
        publish_evolution_evidence(
            s[0], s[1], s[4], result,
            sandbox_root=s[2], production_root=s[3], **s[5],
        )


def test_limit_total_verified_gate_bundle_and_types(tmp_path):
    s = case(tmp_path)
    with pytest.raises(ValueError):
        evaluate(s, [*s[4]])
    with pytest.raises(ValueError):
        evaluate(s, s[4] + (s[4][0],))
    with pytest.raises(ValueError):
        evaluate(s, (None,))
