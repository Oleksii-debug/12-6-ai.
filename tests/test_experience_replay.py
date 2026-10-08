"""Plan 6 Section 10 LOCAL_FREE positive, negative and restart replay."""
from dataclasses import replace

import pytest

from twelve_six.experience_replay import (
    ReplayRecipe, VerifiedExperience, _mac, build_replay_candidates,
    verify_experience, verify_replay_restart,
)
from twelve_six.post_base_instruction import canonical_digest

H = "a" * 64
KEY = b"v" * 32
ROOTS = frozenset({"1" * 64, "2" * 64, "3" * 64})


def experience(identifier, context, root, *, value=20, forgetting=30, curriculum=40):
    initial = VerifiedExperience(
        identifier, f"task-{identifier}", context, canonical_digest({"id": identifier}),
        root, H, H, "trusted-reviewer", H, "model-producer", H,
        value, forgetting, curriculum,
    )
    return replace(initial, signature=_mac(KEY, initial.payload()))


def setup():
    records = (
        experience("late", H, "1" * 64, value=90),
        experience("early", "b" * 64, "2" * 64, forgetting=95),
        experience("same-context", H, "3" * 64, curriculum=50),
    )
    ordered = sorted(records, key=lambda r: r.experience_id)
    pool_sha = canonical_digest([r.identity() for r in ordered])
    recipe = ReplayRecipe(H, pool_sha, max_selected=2)
    kwargs = dict(trusted_verifier_id="trusted-reviewer", trusted_verifier_version_sha256=H,
                  verifier_key=KEY, trusted_evidence_roots=ROOTS)
    return records, recipe, kwargs


def run(records=None, recipe=None, kwargs=None):
    r, p, k = setup()
    return build_replay_candidates(records or r, recipe or p, **(kwargs or k))


def test_bounded_selection_uses_novelty_value_forgetting_curriculum_not_recency():
    records, recipe, kwargs = setup()
    result = run(records, recipe, kwargs)
    assert result == run(tuple(reversed(records)), recipe, kwargs)
    assert len(result.selected_ids) == 2
    contexts = {next(r.context_sha256 for r in records if r.experience_id == i)
                for i in result.selected_ids}
    assert len(contexts) == 2
    assert {c.destination for c in result.candidates} == {"dataset", "memory", "skills"}
    assert not result.promotion_authorized
    assert all(not c.promotion_authorized and not c.canonical_base_eligible
               for c in result.candidates)
    assert verify_replay_restart(records, recipe, result, **kwargs)


def test_tampered_receipt_or_wrong_key_denied():
    records, recipe, kwargs = setup()
    for changed in (replace(records[0], value=99),
                    replace(records[0], signature=H),
                    replace(records[0], evidence_sha256="f" * 64)):
        with pytest.raises(ValueError):
            verify_experience(changed, **kwargs)
    with pytest.raises(ValueError):
        verify_experience(records[0], **{**kwargs, "verifier_key": b"x" * 32})


def test_self_approval_and_untrusted_roots_denied():
    records, recipe, kwargs = setup()
    self_signed = replace(records[0], verifier_id=records[0].producer_id)
    self_signed = replace(self_signed, signature=_mac(KEY, self_signed.payload()))
    with pytest.raises(ValueError):
        verify_experience(self_signed, **kwargs)
    with pytest.raises(ValueError):
        run(kwargs={**kwargs, "trusted_evidence_roots": frozenset({H})})


def test_duplicate_evidence_or_same_context_outcome_is_rejected():
    records, recipe, kwargs = setup()
    forged = replace(records[2], evidence_sha256=records[0].evidence_sha256)
    forged = replace(forged, signature=_mac(KEY, forged.payload()))
    for invalid in (records + (records[0],), records[:2] + (forged,)):
        with pytest.raises(ValueError):
            build_replay_candidates(invalid, recipe, **kwargs)


def test_eval_leakage_rights_and_quality_denied():
    records, recipe, kwargs = setup()
    for edit in ({"split": "eval"}, {"rights": "unknown"},
                 {"quality": float("nan")}, {"quality": 0.1}):
        bad = replace(records[0], **edit)
        with pytest.raises(ValueError):
            verify_experience(bad, **kwargs)


def test_frozen_pool_mutation_and_budget_bounds():
    records, recipe, kwargs = setup()
    changed = replace(records[0], value=0)
    changed = replace(changed, signature=_mac(KEY, changed.payload()))
    with pytest.raises(ValueError, match="pool changed"):
        build_replay_candidates((changed,) + records[1:], recipe, **kwargs)
    for attr in ({"max_selected": 65}, {"novelty_weight": -1},
                 {"min_priority": True}, {"value_weight": float("nan")},
                 {"novelty_weight": 0, "value_weight": 0,
                  "forgetting_weight": 0, "curriculum_weight": 0}):
        with pytest.raises(ValueError):
            replace(recipe, **attr).validate()


def test_forged_result_and_consolidation_cannot_promote():
    records, recipe, kwargs = setup()
    actual = run(records, recipe, kwargs)
    candidates = list(actual.candidates)
    candidates[0] = replace(candidates[0], promotion_authorized=True)
    for bogus in (replace(actual, evidence_sha256=H),
                  replace(actual, promotion_authorized=True),
                  replace(actual, candidates=tuple(candidates))):
        with pytest.raises(ValueError):
            verify_replay_restart(records, recipe, bogus, **kwargs)


def test_untyped_and_exhausted_selection_fail_closed():
    records, recipe, kwargs = setup()
    with pytest.raises(ValueError):
        build_replay_candidates([*records], recipe, **kwargs)
    with pytest.raises(ValueError, match="no verified experience"):
        build_replay_candidates(records, replace(recipe, min_priority=1100), **kwargs)
