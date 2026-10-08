from dataclasses import replace

import pytest

from twelve_six.post_base_agentic import (
    AgenticRecipe,
    ToolStep,
    ToolTrajectory,
    attest_tool_trajectory,
    train_agentic_descendant,
    verify_agentic_run,
    verify_tool_trajectory,
)
from twelve_six.post_base_reasoning import TinyPolicy

H = "a" * 64
K = b"z" * 32


def fixture(*, effect=False, success=True):
    step = ToolStep("search", "go", H, H, H, H, H, effect)
    t = ToolTrajectory("trace-1", "ctx", "untrusted-model", (step,), H, H)
    ev = attest_tool_trajectory(
        t, verifier_id="trusted-host", verifier_version_sha256=H,
        verifier_key=K, success=success, external_effects_verified=effect,
    )
    policy = TinyPolicy(("ctx",), ("go", "stop"), ((0.0, 0.0),))
    recipe = AgenticRecipe(policy.identity(), H, "imitation")
    return t, ev, policy, recipe


@pytest.mark.parametrize("mode", ["imitation", "feedback", "rl_style"])
def test_three_modes_exact_replay_and_immutable_parent(mode):
    t, ev, parent, recipe = fixture()
    recipe = replace(recipe, mode=mode)
    run = train_agentic_descendant(
        parent, ((t, ev),), recipe, trusted_verifier_id="trusted-host", verifier_key=K,
    )
    assert run.status == "CANDIDATE_ONLY" and not run.promotion_authorized
    assert run.candidate.identity() != parent.identity()
    assert verify_agentic_run(
        parent, ((t, ev),), recipe, run, trusted_verifier_id="trusted-host", verifier_key=K,
    )
    assert run == train_agentic_descendant(
        parent, ((t, ev),), recipe, trusted_verifier_id="trusted-host", verifier_key=K,
    )


@pytest.mark.parametrize("patch", [
    {"success": False}, {"verifier_id": "untrusted"},
    {"signature": H}, {"external_effects_verified": False},
    {"verifier_version_sha256": "b" * 64},
])
def test_forged_evidence_fails(patch):
    t, ev, parent, recipe = fixture(effect=True)
    with pytest.raises(ValueError):
        train_agentic_descendant(
            parent, ((t, replace(ev, **patch)),), recipe,
            trusted_verifier_id="trusted-host", verifier_key=K,
        )


def test_signed_external_effect_needs_independent_confirmation():
    t, ev, p, r = fixture(effect=True)
    assert train_agentic_descendant(
        p, ((t, ev),), r, trusted_verifier_id="trusted-host", verifier_key=K,
    )
    with pytest.raises(ValueError, match="forged"):
        train_agentic_descendant(
            p, ((t, replace(ev, external_effects_verified=False)),), r,
            trusted_verifier_id="trusted-host", verifier_key=K,
        )


def test_failed_trajectory_cannot_be_imitation_label():
    t, ev, p, r = fixture(success=False)
    with pytest.raises(ValueError, match="no independently verified"):
        train_agentic_descendant(
            p, ((t, ev),), r, trusted_verifier_id="trusted-host", verifier_key=K,
        )


def test_reject_self_attestation_and_wrong_key():
    t, ev, _, _ = fixture()
    with pytest.raises(ValueError, match="self-issued"):
        attest_tool_trajectory(
            t, verifier_id="untrusted-model", verifier_version_sha256=H,
            verifier_key=K, success=True, external_effects_verified=True,
        )
    with pytest.raises(ValueError):
        verify_tool_trajectory(
            t, ev, trusted_verifier_id="trusted-host",
            trusted_verifier_version_sha256=H, verifier_key=b"x" * 32,
        )


def test_continuity_version_eval_leakage_and_wrong_action():
    t, ev, p, r = fixture()
    variants = [
        replace(t, split="eval"), replace(t, schema_version=2),
        replace(t, steps=(t.steps[0], replace(t.steps[0], state_before_sha256="b" * 64))),
        replace(t, steps=(replace(t.steps[0], action_name="unknown"),)),
    ]
    for bad in variants:
        with pytest.raises(ValueError):
            train_agentic_descendant(
                p, ((bad, ev),), r, trusted_verifier_id="trusted-host", verifier_key=K,
            )


def test_duplicate_and_recipe_bounds_and_tampering():
    t, ev, p, r = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        train_agentic_descendant(
            p, ((t, ev), (t, ev)), r, trusted_verifier_id="trusted-host", verifier_key=K,
        )
    for patch in (
        {"learning_rate": float("nan")}, {"max_updates": 100},
        {"mode": "nonsense"}, {"parent_sha256": H},
    ):
        with pytest.raises(ValueError):
            train_agentic_descendant(
                p, ((t, ev),), replace(r, **patch),
                trusted_verifier_id="trusted-host", verifier_key=K,
            )
    run = train_agentic_descendant(
        p, ((t, ev),), r, trusted_verifier_id="trusted-host", verifier_key=K,
    )
    with pytest.raises(ValueError):
        verify_agentic_run(
            p, ((t, ev),), r, replace(run, promotion_authorized=True),
            trusted_verifier_id="trusted-host", verifier_key=K,
        )
    with pytest.raises(ValueError, match="reproducible"):
        verify_agentic_run(
            p, ((t, ev),), r, replace(run, data_sha256=H),
            trusted_verifier_id="trusted-host", verifier_key=K,
        )
