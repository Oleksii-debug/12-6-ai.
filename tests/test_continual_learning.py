"""Plan 6 S9 LOCAL_FREE frozen-capability and negative/restart tests."""
from dataclasses import replace

import pytest

from twelve_six import continual_learning as cl
from twelve_six.continual_learning import (
    CapabilityProbe,
    ContinualRecipe,
    FrozenCapabilitySuite,
    run_continual_update,
    score_suite,
    verify_continual_replay,
)
from twelve_six.post_base_instruction import canonical_digest as digest
from twelve_six.post_base_reasoning import (
    ReasoningReceipt,
    ReasoningRecipe,
    ReasoningRun,
    TinyPolicy,
    VerifiedTask,
)

H = "a" * 64


def scenario():
    champion = TinyPolicy(("old", "new"), ("A", "B"), ((2., 0.), (0., 0.)))
    fresh = VerifiedTask("train_new", "new", digest({"action": "B"}), H, H)
    replay = VerifiedTask("replay_old", "old", digest({"action": "A"}), H, H)
    roots = frozenset({fresh.root(), replay.root()})
    old = FrozenCapabilitySuite("frozen_old", "retention", H, (
        CapabilityProbe("old_check", "old", digest({"action": "A"})),
    ))
    holdout = FrozenCapabilitySuite("frozen_generalization", "holdout", H, (
        CapabilityProbe("unseen_check", "new", digest({"action": "B"})),
    ))
    rl = ReasoningRecipe(
        champion.identity(), digest(sorted(roots)), seed=0,
        max_steps=32, learning_rate=.2,
    )
    contract = ContinualRecipe(
        champion.identity(), old.identity(), holdout.identity(),
        min_holdout_accuracy=1., min_holdout_gain=1.,
    )
    return (champion, (fresh,), (replay,), rl, contract, old, holdout,
            roots, frozenset({old.identity(), holdout.identity()}))


def run(sc=None):
    a = sc or scenario()
    return run_continual_update(
        *a[:7], trusted_verifier_roots=a[7], trusted_suite_roots=a[8],
    )


def test_qualified_candidate_retains_old_capabilities_and_replays():
    args = scenario()
    original = args[0].identity()
    result = run(args)
    assert result == run(args)
    assert result.state == "CANDIDATE_ONLY" and result.candidate is not None
    assert result.champion_sha256 == original == result.rollback_to_sha256
    assert args[0].identity() == original and not result.promotion_authorized
    cycle = result.cycles[0]
    assert cycle.retention_before == cycle.retention_after == 1.
    assert cycle.holdout_before == 0. and cycle.holdout_after == 1.
    assert verify_continual_replay(
        *args[:7], result,
        trusted_verifier_roots=args[7], trusted_suite_roots=args[8],
    )


def test_unreachable_generalization_rolls_back_without_overwriting_champion():
    args = scenario()
    recipe = replace(args[4], min_holdout_gain=1., max_cycles=2)
    bad_suite = replace(args[6], probes=(
        CapabilityProbe("unseen_check", "new", digest({"action": "A"})),
    ))
    bad_contract = replace(
        recipe, holdout_suite_sha256=bad_suite.identity(),
        min_holdout_accuracy=1.,
    )
    modified = (
        *args[:4], bad_contract, args[5], bad_suite,
        args[7], frozenset({args[5].identity(), bad_suite.identity()}),
    )
    result = run(modified)
    assert result.state == "ROLLED_BACK" and result.candidate is None
    assert result.rollback_to_sha256 == args[0].identity()
    assert len(result.cycles) == 1 and result.cycles[0].state == "ROLLED_BACK"


def test_stale_champion_or_forged_suite_refused_before_training():
    args = scenario()
    with pytest.raises(ValueError):
        run((*args[:4], replace(args[4], champion_sha256=H), *args[5:]))
    with pytest.raises(ValueError):
        run((*args[:4], replace(args[4], holdout_suite_sha256=H), *args[5:]))
    with pytest.raises(ValueError):
        run((*args[:8], frozenset({H})))


def test_eval_leakage_and_duplicate_replay_rejected():
    args = scenario()
    with pytest.raises(ValueError):
        run((args[0], args[1], args[1], *args[3:]))
    leaked = replace(args[1][0], task_id="unseen_check")
    rootset = frozenset({leaked.root(), args[2][0].root()})
    rl = replace(args[3], verifier_set_sha256=digest(sorted(rootset)))
    with pytest.raises(ValueError):
        run((args[0], (leaked,), args[2], rl, *args[4:7], rootset, args[8]))
    malformed = replace(args[1][0], split="eval")
    with pytest.raises(ValueError):
        run((args[0], (malformed,), args[2], args[3], *args[4:]))


def test_forged_replay_evidence_and_unauthorized_promotion_rejected():
    args = scenario()
    result = run(args)
    for forged in (
        replace(result, evidence_sha256=H),
        replace(result, promotion_authorized=True),
    ):
        with pytest.raises(ValueError):
            verify_continual_replay(
                *args[:7], forged, trusted_verifier_roots=args[7],
                trusted_suite_roots=args[8],
            )


def test_threshold_bounds_and_missing_replay():
    args = scenario()
    for kwargs in (
        {"max_cycles": 5}, {"max_retention_drop": float("nan")},
        {"min_holdout_accuracy": 1.01},
    ):
        with pytest.raises(ValueError):
            run((*args[:4], replace(args[4], **kwargs), *args[5:]))
    with pytest.raises(ValueError):
        run((args[0], args[1], (), *args[3:]))


def test_policy_retention_detects_catastrophic_flip():
    args = scenario()
    original = args[0]
    flipped = replace(original, logits=((-1., 1.), (0., 1.)))
    assert score_suite(original, args[5]) == 1.
    assert score_suite(flipped, args[5]) == 0.


def test_catastrophic_candidate_is_rejected_even_with_fake_successful_update(monkeypatch):
    args = scenario()
    champion = args[0]
    bad = replace(champion, logits=((-1., 1.), (0., 1.)))

    def adversarial_update(*unused, **kwargs):
        receipt = ReasoningReceipt(
            champion.identity(), bad.identity(), H, H, H,
            2, "CANDIDATE_ONLY", "claimed_success",
        )
        return ReasoningRun(bad, (), receipt)

    monkeypatch.setattr(cl, "run_verified_rl", adversarial_update)
    monkeypatch.setattr(cl, "verify_reasoning_run", lambda *a, **kw: True)
    result = run(args)
    assert result.state == "ROLLED_BACK" and result.candidate is None
    assert result.cycles[0].reason == "forgetting_or_holdout_regression"
    assert result.rollback_to_sha256 == champion.identity()
