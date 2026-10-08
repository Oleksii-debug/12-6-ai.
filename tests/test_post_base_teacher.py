"""Plan 6 Section 6: local/free teacher-council acceptance and adversarial tests."""
from dataclasses import replace
import hashlib

import pytest

from twelve_six.post_base_teacher import (
    TeacherSpec, GatewayReply, collect_teacher_candidates,
    attest_council, verify_council,
)

H = "a" * 64
K = b"k" * 32


class FixtureGateway:
    def __init__(self, values=None):
        self.values = values if values is not None else {"one": "yes", "two": "no"}
        self.calls = []

    def generate(self, spec, prompt):
        self.calls.append(spec.teacher_id)
        value = self.values[spec.teacher_id]
        return GatewayReply(
            spec.teacher_id, spec.model_sha256,
            hashlib.sha256(prompt.encode()).hexdigest(),
            value, 5, not bool(value),
        )


def specs():
    return (
        TeacherSpec("one", "fixture", H, H, "qa", 10),
        TeacherSpec("two", "fixture", "b" * 64, H, "qa", 10, transport="external"),
    )


def batch(gateway=None):
    return collect_teacher_candidates(
        gateway or FixtureGateway(), prompt="2+2?", capability="qa",
        policy_sha256=H, teachers=specs(), total_budget_microusd=20,
    )


def verdict(record, *, candidate=None, accepted=True):
    if candidate is None and accepted:
        candidate = record.replies[0].identity()
    return attest_council(
        record, selected_reply_sha256=candidate,
        verifier_id="trusted-host", verifier_version_sha256=H,
        verifier_key=K, independent_passed=accepted,
    )


def check(record, decision, key=K):
    return verify_council(
        record, decision, trusted_verifier_id="trusted-host",
        trusted_verifier_version_sha256=H, verifier_key=key,
    )


def test_disagreement_is_not_truth_and_verification_replays():
    record = batch()
    assert record.status == "UNVERIFIED_CANDIDATES"
    assert record.disagreement and record.cost_microusd == 10
    assert record == batch() and record.identity() == batch().identity()
    assert check(record, verdict(record))


def test_abstention_requires_independent_rejection():
    record = batch(FixtureGateway({"one": "", "two": ""}))
    assert not record.disagreement and all(x.abstained for x in record.replies)
    assert check(record, verdict(record, accepted=False))
    with pytest.raises(ValueError):
        verdict(record)


def test_budget_preflight_denies_before_second_provider():
    gateway = FixtureGateway()
    with pytest.raises(ValueError, match="preflight"):
        collect_teacher_candidates(
            gateway, prompt="ok", capability="qa", policy_sha256=H,
            teachers=specs(), total_budget_microusd=14,
        )
    assert gateway.calls == ["one"]


def test_gateway_wrong_model_and_bad_capability_fail_closed():
    class Foreign(FixtureGateway):
        def generate(self, spec, prompt):
            return replace(super().generate(spec, prompt), model_sha256="f" * 64)

    with pytest.raises(ValueError, match="foreign"):
        batch(Foreign())
    with pytest.raises(ValueError, match="unauthorized"):
        collect_teacher_candidates(
            FixtureGateway(), prompt="ok", capability="wrong",
            policy_sha256=H, teachers=specs(), total_budget_microusd=20,
        )


def test_forged_evidence_and_wrong_key_rejected():
    record = batch()
    genuine = verdict(record)
    for forged in (
        replace(genuine, signature=H),
        replace(genuine, batch_sha256=H),
        replace(genuine, accepted=False),
        replace(genuine, selected_reply_sha256=H),
    ):
        with pytest.raises(ValueError):
            check(record, forged)
    with pytest.raises(ValueError):
        check(record, genuine, key=b"x" * 32)


def test_teacher_cannot_self_issue_verdict():
    record = batch()
    with pytest.raises(ValueError, match="self-issued"):
        attest_council(
            record, selected_reply_sha256=record.replies[0].identity(),
            verifier_id="one", verifier_version_sha256=H,
            verifier_key=K, independent_passed=True,
        )


def test_duplicate_teacher_and_foreign_selection_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        collect_teacher_candidates(
            FixtureGateway(), prompt="ok", capability="qa", policy_sha256=H,
            teachers=(specs()[0], specs()[0]), total_budget_microusd=20,
        )
    with pytest.raises(ValueError):
        verdict(batch(), candidate=H)
    with pytest.raises(ValueError):
        verdict(batch(), candidate=batch().replies[0].identity(), accepted=False)


def test_tampered_council_accounting_and_prompt_fail_closed():
    original = batch()
    for forged in (
        replace(original, cost_microusd=0),
        replace(original, disagreement=False),
        replace(original, prompt="rewritten"),
        replace(original, replies=(original.replies[0],)),
    ):
        with pytest.raises(ValueError):
            verdict(forged)
