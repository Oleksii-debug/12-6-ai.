"""Section 4 verified-reward contextual bandit fixture tests (LOCAL_FREE)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.post_base_instruction import canonical_digest
from twelve_six.post_base_reasoning import (
    ReasoningRecipe,
    TinyPolicy,
    VerifiedTask,
    run_verified_rl,
    verify_reasoning_run,
)

H = "a" * 64
E = "b" * 64


def fixture(target="correct", logits=(0.0, 0.0), steps=16):
    parent = TinyPolicy(("math",), ("correct", "wrong"), (logits,))
    task = VerifiedTask("trusted-1", "math", canonical_digest({"action": target}), H, E)
    roots = frozenset({task.root()})
    recipe = ReasoningRecipe(parent.identity(), canonical_digest(sorted(roots)), max_steps=steps)
    return parent, (task,), roots, recipe


def test_verified_reward_candidate_is_immutable_deterministic_and_unpromoted():
    parent, tasks, roots, recipe = fixture()
    before = parent.identity()
    result = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots)
    assert result.receipt.status == "CANDIDATE_ONLY"
    assert result.candidate is not None
    assert result.candidate.identity() != before
    assert parent.identity() == before
    assert result.receipt.positive_rewards > 0
    assert all(r.verifier_root_sha256 == tasks[0].root() for r in result.rollouts)
    assert all(r.reward == int(r.action == "correct") for r in result.rollouts)
    assert verify_reasoning_run(parent, tasks, recipe, result, trusted_verifier_roots=roots)
    assert run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots) == result
    assert result.receipt.promotion_authorized is False


def test_unverified_reward_rolls_back_even_with_training_attempt():
    parent, tasks, roots, recipe = fixture(target="unreachable")
    result = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots)
    assert result.candidate is None
    assert result.receipt.status == "ROLLED_BACK"
    assert result.receipt.positive_rewards == 0
    assert parent.identity() == recipe.parent_sha256
    assert verify_reasoning_run(parent, tasks, recipe, result, trusted_verifier_roots=roots)


def test_low_entropy_collapse_rolls_back_before_any_update():
    parent, tasks, roots, recipe = fixture(logits=(30.0, -30.0))
    result = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots)
    assert result.candidate is None
    assert not result.rollouts
    assert result.receipt.cause == "degenerate_policy_entropy"
    assert parent.identity() == recipe.parent_sha256


def test_repeated_misses_reject_even_if_gradient_nonzero():
    parent, tasks, roots, recipe = fixture(target="unknown", steps=16)
    result = run_verified_rl(parent, tasks, replace(recipe, max_consecutive_misses=3),
                             trusted_verifier_roots=roots)
    assert result.receipt.cause == "repeated_unverified_rewards"
    assert len(result.rollouts) == 3 and result.candidate is None


@pytest.mark.parametrize("alter", [
    lambda x: replace(x, split="eval"),
    lambda x: replace(x, accepted_action_sha256="wrong"),
    lambda x: replace(x, verifier_sha256="wrong"),
    lambda x: replace(x, evidence_sha256="wrong"),
    lambda x: replace(x, schema_version=2),
])
def test_untrusted_or_reserved_reward_source_is_rejected(alter):
    parent, tasks, roots, recipe = fixture()
    with pytest.raises(ValueError):
        run_verified_rl(parent, (alter(tasks[0]),), recipe, trusted_verifier_roots=roots)


def test_forged_root_and_duplicate_task_fails_closed():
    parent, tasks, roots, recipe = fixture()
    with pytest.raises(ValueError, match="verifier set"):
        run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=frozenset({H}))
    with pytest.raises(ValueError, match="duplicate"):
        run_verified_rl(parent, tasks + tasks, recipe, trusted_verifier_roots=roots)
    with pytest.raises(ValueError, match="independent"):
        run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=frozenset())


@pytest.mark.parametrize("recipe_patch", [
    {"max_steps": 65}, {"learning_rate": float("nan")},
    {"learning_rate": -0.1}, {"max_consecutive_misses": 0},
    {"seed": True}, {"min_entropy": float("inf")},
    {"parent_sha256": H}, {"schema_version": 42},
])
def test_adversarial_bounded_recipe(recipe_patch):
    parent, tasks, roots, recipe = fixture()
    with pytest.raises(ValueError):
        run_verified_rl(parent, tasks, replace(recipe, **recipe_patch),
                        trusted_verifier_roots=roots)


def test_adversarial_nan_policy_and_oracle_changes_are_rejected():
    parent, tasks, roots, recipe = fixture()
    with pytest.raises(ValueError):
        run_verified_rl(replace(parent, logits=((float("nan"), 0.0),)),
                        tasks, recipe, trusted_verifier_roots=roots)
    with pytest.raises(ValueError):
        run_verified_rl(parent, (replace(tasks[0], context_id="other"),),
                        recipe, trusted_verifier_roots=roots)


def test_tampered_replay_cannot_self_certify_or_promote():
    parent, tasks, roots, recipe = fixture()
    result = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots)
    with pytest.raises(ValueError, match="mismatch"):
        verify_reasoning_run(parent, tasks, recipe,
                             replace(result, receipt=replace(result.receipt, positive_rewards=999)),
                             trusted_verifier_roots=roots)
    with pytest.raises(ValueError, match="mismatch"):
        verify_reasoning_run(parent, tasks, recipe,
                             replace(result, receipt=replace(
                                 result.receipt, promotion_authorized=True)),
                             trusted_verifier_roots=roots)
    with pytest.raises(ValueError, match="mismatch"):
        verify_reasoning_run(parent, tasks, recipe,
                             replace(result, rollouts=result.rollouts[:-1]),
                             trusted_verifier_roots=roots)


def test_last_update_cannot_promote_entropy_collapse():
    parent, tasks, roots, recipe = fixture(
        logits=(5.830638927213939, 0.0), steps=1,
    )
    result = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=roots)
    assert result.candidate is None
    assert result.receipt.cause == "degenerate_final_policy_entropy"
    assert result.receipt.status == "ROLLED_BACK"
    assert parent.identity() == recipe.parent_sha256
