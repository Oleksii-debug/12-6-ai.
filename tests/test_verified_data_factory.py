"""LOCAL_FREE Plan 6 Section 7 acceptance, negative and deterministic replay tests."""
from dataclasses import replace

import pytest

from twelve_six.verified_data_factory import (
    CurriculumGoal,
    DataCandidate,
    content_fingerprint,
    generate_verified_pool,
    verify_factory_receipt,
)

H = "a" * 64
K = b"k" * 32


class FixtureGenerator:
    def __init__(self, kind="teacher", producer="p", answer="4",
                 max_cost=0, external=False):
        self.producer_id = producer
        self.origin_kind = kind
        self.max_cost_microunits = max_cost
        self.external = external
        self.calls = []
        self.answer = answer

    def generate(self, goal, attempt, seed):
        self.calls.append((attempt, seed))
        return DataCandidate(
            f"id-{self.producer_id}-{attempt}", goal.goal_id,
            self.producer_id, self.origin_kind, f"q {attempt}",
            self.answer, H, H, "owned", H, 0.9,
        )


class HostVerifier:
    verifier_id = "trusted-host"
    verifier_version_sha256 = H

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []

    def verify(self, goal, candidate):
        self.calls.append(candidate.identity())
        return self.accepted


def goal(**overrides):
    return replace(CurriculumGoal("goal", H, (), 17), **overrides)


def run(current_goal=None, generators=None, verifier=None):
    return generate_verified_pool(
        current_goal or goal(), generators or (FixtureGenerator(),),
        verifier or HostVerifier(), verifier_key=K,
    )


def test_candidate_pool_only_replay_and_stop():
    first, second = run(), run()
    assert first == second and first.attempts == 2 and len(first.accepted) == 2
    assert first.status == "CANDIDATE_POOL_ONLY"
    assert not first.canonical_base_eligible and not first.promotion_authorized
    assert first.manifest_sha256 == second.manifest_sha256
    for candidate, receipt in zip(first.accepted, first.receipts, strict=True):
        assert verify_factory_receipt(
            candidate, goal(), receipt, verifier_id="trusted-host",
            verifier_version_sha256=H, verifier_key=K,
        )


def test_teacher_self_play_and_tool_paths():
    current = goal(target_verified=3)
    result = run(current, (
        FixtureGenerator("teacher", "p"), FixtureGenerator("self_play", "s"),
        FixtureGenerator("tool", "t"),
    ))
    assert [c.origin_kind for c in result.accepted] == [
        "teacher", "self_play", "tool",
    ]


def test_failed_verifier_no_admission_and_bounded_stop():
    result = run(goal(max_attempts=3), verifier=HostVerifier(False))
    assert result.attempts == 3 and not result.accepted
    assert len(result.rejected) == 3
    assert all(reason == "UNVERIFIED_OR_EFFECT" for _, reason in result.rejected)


def test_holdout_contamination_and_duplicate_content():
    fingerprint = content_fingerprint("q 0", "4")
    result = run(goal(holdout_fingerprints=(fingerprint,),
                      target_verified=1, max_attempts=2))
    assert len(result.accepted) == 1
    assert result.rejected[0][1] == "HOLDOUT_CONTAMINATION"

    class Duplicate(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return replace(super().generate(current_goal, attempt, seed), task="same")

    duplicate = run(goal(max_attempts=3, target_verified=2),
                    generators=(Duplicate(),))
    assert len(duplicate.accepted) == 1
    assert [reason for _, reason in duplicate.rejected] == ["DUPLICATE", "DUPLICATE"]


def test_rights_and_eval_leakage_are_rejected():
    class Eval(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return replace(super().generate(current_goal, attempt, seed), split="eval")

    class Rights(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return replace(super().generate(current_goal, attempt, seed), rights="unknown")

    for kind in (Eval, Rights):
        with pytest.raises(ValueError):
            run(generators=(kind(),))


@pytest.mark.parametrize("change", [
    {"quality": float("nan")}, {"quality": 0.1}, {"source_sha256": "oops"},
])
def test_invalid_quality_and_provenance(change):
    class Invalid(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return replace(super().generate(current_goal, attempt, seed), **change)

    with pytest.raises(ValueError):
        run(generators=(Invalid(),))


def test_unapproved_paid_or_external_generators_never_invoked():
    paid = FixtureGenerator(max_cost=1)
    with pytest.raises(ValueError):
        run(generators=(paid,))
    assert paid.calls == []
    external = FixtureGenerator(external=True)
    with pytest.raises(ValueError):
        run(generators=(external,))
    assert external.calls == []
    with pytest.raises(ValueError):
        goal(max_total_microunits=3).validate()


def test_forged_evidence_wrong_key_and_self_approval():
    current_goal = goal()
    result = run(current_goal)
    receipt, candidate = result.receipts[0], result.accepted[0]
    for forged in (
        replace(receipt, signature=H),
        replace(receipt, accepted=False),
        replace(receipt, goal_sha256=H),
    ):
        with pytest.raises(ValueError):
            verify_factory_receipt(
                candidate, current_goal, forged, verifier_id="trusted-host",
                verifier_version_sha256=H, verifier_key=K,
            )
    with pytest.raises(ValueError):
        verify_factory_receipt(
            candidate, current_goal, receipt, verifier_id="trusted-host",
            verifier_version_sha256=H, verifier_key=b"x" * 32,
        )

    class SelfVerifier(HostVerifier):
        verifier_id = "p"

    with pytest.raises(ValueError):
        run(verifier=SelfVerifier())


def test_external_tool_effect_never_becomes_positive():
    class Effect(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return replace(
                super().generate(current_goal, attempt, seed),
                external_side_effect=True,
            )

    result = run(goal(max_attempts=2), generators=(Effect(kind="tool"),))
    assert not result.accepted and len(result.rejected) == 2
    assert len(result.receipts) == 2


def test_untyped_verifier_and_generator_responses_denied():
    class BadVerifier(HostVerifier):
        def verify(self, current_goal, candidate):
            return 1

    class BadGenerator(FixtureGenerator):
        def generate(self, current_goal, attempt, seed):
            return object()

    with pytest.raises(ValueError):
        run(verifier=BadVerifier())
    with pytest.raises(ValueError):
        run(generators=(BadGenerator(),))
