"""LOCAL_FREE Plan6 S14 contract, negative, recovery and rollback evidence."""
from dataclasses import replace

import pytest

from twelve_six.champion_lifecycle import (
    Authorization,
    Descendant,
    Evaluation,
    LifecycleState,
    decide_candidate,
    rollback_promotion,
)
from twelve_six.experience_replay import _mac

A = 'a' * 64
B = 'b' * 64
C = 'c' * 64
D = 'd' * 64
E = 'e' * 64
F = 'f' * 64
VER_KEY = b'v' * 32
AUTH_KEY = b'g' * 32


def fixture(qualified=True, compatible=True):
    state = LifecycleState.genesis(A, B)
    candidate = Descendant('candidate-1', 'producer', B, C, A, D, E, F)
    raw = Evaluation(candidate.identity(), B, A, F, D, E, 'verifier', D,
                     qualified, compatible, A)
    evaluation = replace(raw, signature=_mac(VER_KEY, raw.payload()))
    return state, candidate, evaluation


def trusted(state):
    return {
        "trusted_state_sha256": state.head_sha256,
        "verifier_id": "verifier",
        "verifier_version_sha256": D,
        "trusted_evidence_roots": frozenset({D, E}),
        "verifier_key": VER_KEY,
        "authorizer_id": "authorizer",
        "authorizer_version_sha256": F,
        "authorizer_key": AUTH_KEY,
    }


def grant_for(state, candidate, evaluation, action='PROMOTE'):
    raw = Authorization(action, candidate.identity(), evaluation.identity(),
                        state.head_sha256, 'authorizer', F, A)
    return replace(raw, signature=_mac(AUTH_KEY, raw.payload()))


def test_promotion_requires_signed_explicit_grant_and_integrity():
    state, candidate, evaluation = fixture()
    with pytest.raises(ValueError):
        decide_candidate(state, candidate, evaluation, **trusted(state))
    promoted = decide_candidate(state, candidate, evaluation,
                                grant_for(state, candidate, evaluation), **trusted(state))
    promoted.validate()
    assert promoted.champion_sha256 == C
    assert state.champion_sha256 == B
    assert promoted.events[0].action == 'PROMOTED'
    assert promoted == decide_candidate(state, candidate, evaluation,
                                        grant_for(state, candidate, evaluation), **trusted(state))


def test_reject_failed_evaluation_preserves_champion_and_history():
    state, candidate, evaluation = fixture(qualified=False)
    rejected = decide_candidate(state, candidate, evaluation, **trusted(state))
    assert rejected.champion_sha256 == B
    assert rejected.events[0].action == 'REJECTED'
    assert rejected.events[0].authorization_sha256 is None
    assert state.events == ()
    with pytest.raises(ValueError):
        decide_candidate(state, candidate, evaluation,
                         grant_for(state, candidate, evaluation), **trusted(state))


def test_incompatible_candidate_rejected_without_champion_edit():
    state, candidate, evaluation = fixture(compatible=False)
    result = decide_candidate(state, candidate, evaluation, **trusted(state))
    assert result.champion_sha256 == B
    assert result.events[0].action == 'REJECTED'


def test_signed_rollback_restores_immutable_previous_champion():
    state, candidate, evaluation = fixture()
    promoted = decide_candidate(state, candidate, evaluation,
                                grant_for(state, candidate, evaluation), **trusted(state))
    rolled = rollback_promotion(promoted, candidate, evaluation,
                                grant_for(promoted, candidate, evaluation, 'ROLLBACK'),
                                trusted_state_sha256=promoted.head_sha256,
                                authorizer_id='authorizer', authorizer_version_sha256=F,
                                authorizer_key=AUTH_KEY)
    assert rolled.champion_sha256 == B
    assert [event.action for event in rolled.events] == ['PROMOTED', 'ROLLED_BACK']
    assert rolled.events[0] == promoted.events[0]
    assert rolled == rollback_promotion(promoted, candidate, evaluation,
                                        grant_for(promoted, candidate, evaluation, 'ROLLBACK'),
                                        trusted_state_sha256=promoted.head_sha256,
                                        authorizer_id='authorizer', authorizer_version_sha256=F,
                                        authorizer_key=AUTH_KEY)
    with pytest.raises(ValueError):
        rollback_promotion(rolled, candidate, evaluation,
                           grant_for(rolled, candidate, evaluation, 'ROLLBACK'),
                           trusted_state_sha256=rolled.head_sha256,
                           authorizer_id='authorizer', authorizer_version_sha256=F,
                           authorizer_key=AUTH_KEY)


def test_forged_or_stale_evaluation_and_grant_denied():
    state, candidate, evaluation = fixture()
    valid = grant_for(state, candidate, evaluation)
    for e in (replace(evaluation, signature=A),
              replace(evaluation, qualified=False),
              replace(evaluation, verifier_id='producer'),
              replace(evaluation, compatibility_sha256=A),
              replace(evaluation, evidence_sha256=C),
              replace(evaluation, qualified=1)):
        with pytest.raises(ValueError):
            decide_candidate(state, candidate, e, valid, **trusted(state))
    for g in (replace(valid, signature=A), replace(valid, expected_state_sha256=B),
              replace(valid, action='ROLLBACK'),
              replace(valid, authorizer_id='producer'),
              replace(valid, authorizer_id='verifier')):
        with pytest.raises(ValueError):
            decide_candidate(state, candidate, evaluation, g, **trusted(state))


def test_candidate_foreign_base_parent_or_reused_identity_fails_closed():
    state, candidate, evaluation = fixture()
    for c in (replace(candidate, base_sha256=C),
              replace(candidate, parent_sha256=E),
              replace(candidate, descendant_sha256=B),
              replace(candidate, candidate_id='')):
        with pytest.raises(ValueError):
            decide_candidate(state, c, evaluation, **trusted(state))
    rejected = decide_candidate(state, candidate,
                                replace(evaluation, qualified=False,
                                        signature=_mac(VER_KEY, replace(evaluation,
                                        qualified=False).payload())), **trusted(state))
    with pytest.raises(ValueError):
        decide_candidate(rejected, candidate, evaluation, **trusted(rejected))


def test_tampered_history_and_rewrapped_head_fail_external_pin():
    state, candidate, evaluation = fixture()
    promoted = decide_candidate(state, candidate, evaluation,
                                grant_for(state, candidate, evaluation), **trusted(state))
    for forged in (replace(promoted, head_sha256=A),
                   replace(promoted, champion_sha256=B),
                   replace(promoted, events=promoted.events * 2),
                   replace(promoted, events=(replace(promoted.events[0],
                                                    authorization_sha256=None),))):
        with pytest.raises(ValueError):
            forged.validate()
    with pytest.raises(ValueError):
        decide_candidate(state, candidate, evaluation,
                         grant_for(state, candidate, evaluation),
                         **dict(trusted(state), trusted_state_sha256=A))


def test_bad_keys_versions_and_forged_rollback_rejected():
    state, candidate, evaluation = fixture()
    promotion = decide_candidate(state, candidate, evaluation,
                                 grant_for(state, candidate, evaluation), **trusted(state))
    valid = grant_for(promotion, candidate, evaluation, 'ROLLBACK')
    for grant in (replace(valid, signature=A), replace(valid, action='PROMOTE'),
                  replace(valid, expected_state_sha256=state.head_sha256)):
        with pytest.raises(ValueError):
            rollback_promotion(promotion, candidate, evaluation, grant,
                               trusted_state_sha256=promotion.head_sha256,
                               authorizer_id='authorizer', authorizer_version_sha256=F,
                               authorizer_key=AUTH_KEY)
    with pytest.raises(ValueError):
        rollback_promotion(promotion, candidate, evaluation, valid,
                           trusted_state_sha256=promotion.head_sha256,
                           authorizer_id='authorizer', authorizer_version_sha256=F,
                           authorizer_key=b'wrong')


def test_double_spend_stale_state_and_non_alias_lineage():
    state, candidate, evaluation = fixture()
    promoted = decide_candidate(state, candidate, evaluation,
                                grant_for(state, candidate, evaluation), **trusted(state))
    with pytest.raises(ValueError):
        decide_candidate(promoted, candidate, evaluation,
                         grant_for(state, candidate, evaluation), **trusted(promoted))
    with pytest.raises(ValueError):
        replace(candidate, descendant_sha256=B).identity()
    assert state.champion_sha256 == B
