from dataclasses import replace

import pytest

from twelve_six.post_base_instruction import canonical_digest
from twelve_six.teacher_council import (
    TeacherModel, TeacherPrompt, GatewayReply, gather_teacher_candidates,
    attest_teacher_candidate, decide_teacher_council, attest_council_judge,
)

H = "a" * 64
K = b"k" * 32
J = b"j" * 32


class Gateway:
    def __init__(self, answers=("yes", "yes"), error=False):
        self.answers = answers
        self.count = 0
        self.error = error

    def invoke(self, model, prompt):
        i = self.count
        self.count += 1
        if self.error:
            raise TimeoutError("provider unavailable")
        return GatewayReply(model.model_sha256, prompt.prompt_sha256, prompt.policy_sha256,
                            self.answers[i], 0)


def fixture(answers=("yes", "yes")):
    prompt = TeacherPrompt(
        "task1", "2 + 2?", canonical_digest({"prompt": "2 + 2?"}), "text", H,
    )
    teachers = tuple(
        TeacherModel(f"teacher{i}", H, "local", "text", H, 512)
        for i in range(len(answers))
    )
    gateway = Gateway(answers)
    batch = gather_teacher_candidates(prompt, teachers, gateway)
    proofs = tuple(
        attest_teacher_candidate(
            c, verifier_id="host-verifier", verifier_version_sha256=H,
            verifier_key=K, accepted=True, rights_ok=True, quality_ok=True,
        ) for c in batch.candidates
    )
    return prompt, teachers, gateway, batch, proofs


def decide(batch, proofs, **kw):
    return decide_teacher_council(
        batch, proofs, verifier_id="host-verifier",
        verifier_version_sha256=H, verifier_key=K, **kw,
    )


def test_identical_teacher_consensus_is_candidate_only():
    _, _, _, batch, proofs = fixture()
    decision = decide(batch, proofs)
    assert decision.state == "ACCEPTED_CANDIDATE"
    assert not decision.disagreement and not decision.promotion_authorized
    assert decide(batch, proofs) == decision


def test_conflicting_teachers_force_disagreement_until_independent_judge():
    _, _, _, batch, proofs = fixture(("yes", "no"))
    decision = decide(batch, proofs)
    assert decision.state == "DISAGREEMENT" and decision.candidate_sha256 is None
    judge = attest_council_judge(
        batch, batch.candidates[1], judge_id="independent-judge",
        judge_version_sha256=H, judge_key=J,
    )
    chosen = decide(
        batch, proofs, judge=judge, trusted_judge_id="independent-judge",
        trusted_judge_version_sha256=H, judge_key=J,
    )
    assert chosen.state == "ACCEPTED_CANDIDATE" and chosen.disagreement
    assert chosen.candidate_sha256 == batch.candidates[1].identity()
    with pytest.raises(ValueError):
        decide(batch, proofs, judge=replace(judge, signature=H),
               trusted_judge_id="independent-judge",
               trusted_judge_version_sha256=H, judge_key=J)


def test_candidate_not_truth_without_verifier():
    _, _, _, batch, proofs = fixture()
    with pytest.raises(ValueError, match="missing"):
        decide(batch, ())
    rejected = tuple(
        attest_teacher_candidate(
            c, verifier_id="host-verifier", verifier_version_sha256=H,
            verifier_key=K, accepted=False, rights_ok=True, quality_ok=True,
        ) for c in batch.candidates
    )
    assert decide(batch, rejected).state == "ABSTAIN"


def test_low_quality_or_unlicensed_rejected():
    _, _, _, batch, proofs = fixture()
    bad = attest_teacher_candidate(
        batch.candidates[0], verifier_id="host-verifier", verifier_version_sha256=H,
        verifier_key=K, accepted=True, rights_ok=False, quality_ok=True,
    )
    assert decide(batch, (bad, proofs[1])).candidate_sha256 == batch.candidates[1].identity()


def test_self_issued_and_forged_verifiers_fail_closed():
    _, _, _, batch, proofs = fixture()
    with pytest.raises(ValueError, match="self-verify"):
        attest_teacher_candidate(
            batch.candidates[0], verifier_id="teacher0", verifier_version_sha256=H,
            verifier_key=K, accepted=True, rights_ok=True, quality_ok=True,
        )
    with pytest.raises(ValueError, match="forged"):
        decide(batch, (replace(proofs[0], accepted=False), proofs[1]))
    with pytest.raises(ValueError):
        decide(batch, (replace(proofs[0], candidate_sha256=H), proofs[1]))


def test_preflight_paid_external_capability_and_duplicate_do_not_call_gateway():
    prompt, models, _, _, _ = fixture()
    cases = [
        (prompt, (replace(models[0], provider="external"),)),
        (prompt, (replace(models[0], price_microunits=1),)),
        (prompt, (replace(models[0], capability="tools"),)),
        (prompt, (models[0], models[0])),
        (replace(prompt, prompt_sha256=H), models),
        (replace(prompt, max_total_microunits=True), models),
    ]
    for candidate_prompt, candidate_models in cases:
        gateway = Gateway()
        with pytest.raises(ValueError):
            gather_teacher_candidates(candidate_prompt, candidate_models, gateway)
        assert gateway.count == 0


def test_free_external_requires_explicit_policy_and_exact_gateway_receipt():
    prompt, models, _, _, _ = fixture()
    ext = replace(models[0], provider="external")
    gateway = Gateway(("yes",))
    assert gather_teacher_candidates(
        replace(prompt, allow_external=True), (ext,), gateway,
    ).candidates[0].status == "CANDIDATE_UNVERIFIED"

    class BadGateway:
        def invoke(self, model, prompt):
            return GatewayReply(H, H, H, "yes", 0)

    with pytest.raises(ValueError, match="identity"):
        gather_teacher_candidates(replace(prompt, allow_external=True),
                                  (ext,), BadGateway())


def test_gateway_outage_abstains_without_fake_evidence():
    prompt, models, _, _, _ = fixture()
    batch = gather_teacher_candidates(prompt, models, Gateway(error=True))
    assert all(x.status == "GATEWAY_FAILURE" and x.response_text is None
               for x in batch.candidates)
    proofs = tuple(
        attest_teacher_candidate(
            c, verifier_id="host-verifier", verifier_version_sha256=H,
            verifier_key=K, accepted=True, rights_ok=True, quality_ok=True,
        ) for c in batch.candidates
    )
    assert decide(batch, proofs).state == "ABSTAIN"


def test_reject_oversize_and_cost_fraud():
    prompt, models, _, _, _ = fixture()
    with pytest.raises(ValueError, match="oversized"):
        gather_teacher_candidates(prompt, models, Gateway(("x" * 513, "yes")))

    class CostGateway:
        def invoke(self, model, prompt):
            return GatewayReply(model.model_sha256, prompt.prompt_sha256,
                                prompt.policy_sha256, "yes", 100)

    with pytest.raises(ValueError, match="cost"):
        gather_teacher_candidates(prompt, models, CostGateway())


def test_tampered_batch_identity_fails():
    _, _, _, batch, proofs = fixture()
    with pytest.raises(ValueError, match="integrity"):
        decide(replace(batch, charged_microunits=1), proofs)
