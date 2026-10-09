"""Plan 6 S8 negative/recovery/replay tests; LOCAL_FREE independent fixtures."""
from dataclasses import replace

import pytest

from twelve_six.post_base_instruction import canonical_digest
from twelve_six.self_play_engine import (
    PlayRecipe,
    PlayTask,
    run_self_play,
    verify_self_play_replay,
)

H = "a" * 64


class Agent:
    def __init__(self, agent_id, corrupt=False):
        self.agent_id = agent_id
        self.corrupt = corrupt
        self.calls = []

    def answer(self, task, *, role, opponent_id, seed):
        self.calls.append((task.task_id, role, opponent_id, seed))
        return "bad" if self.corrupt else task.context


class Oracle:
    verifier_id = "independent"
    verifier_version_sha256 = H

    def __init__(self, approved=True):
        self.approved = approved

    def verify(self, task, answer):
        return self.approved


def setup():
    tasks = tuple(PlayTask(
        f"t{i}", f"v{i}", canonical_digest({"answer": f"v{i}"}), H,
        "easy" if i == 1 else "hard",
    ) for i in range(4))
    recipe = PlayRecipe(
        canonical_digest([t.identity() for t in tasks]), seed=71,
        max_rounds=4, min_unique=2, max_easy_fraction=0.5,
    )
    return tasks, recipe


def run(tasks=None, recipe=None, solver=None, challenger=None, oracle=None, roots=None):
    a, b = setup()
    return run_self_play(
        tasks if tasks is not None else a, recipe or b,
        solver or Agent("solver"), challenger or Agent("challenger"),
        oracle or Oracle(),
        trusted_verifier_roots=roots if roots is not None else frozenset({H}),
    )


def test_determinism_roles_reward_and_replay():
    tasks, recipe = setup()
    result = run(tasks, recipe)
    assert result == run(tasks, recipe)
    assert result.state == "VERIFIED_CANDIDATE"
    assert sum(x.reward for x in result.rounds) == 4
    assert [x.solver_id for x in result.rounds] == [
        "solver", "challenger", "solver", "challenger",
    ]
    assert not result.canonical_base_eligible and not result.promotion_authorized
    assert verify_self_play_replay(
        tasks, recipe, Agent("solver"), Agent("challenger"), Oracle(), result,
        trusted_verifier_roots=frozenset({H}),
    )


def test_corrupt_reward_and_replay_tamper():
    tasks, recipe = setup()
    result = run(tasks, recipe, Agent("solver", corrupt=True))
    assert result.rounds[0].reward == 0
    for forged in (replace(result, evidence_sha256=H),
                   replace(result, promotion_authorized=True)):
        with pytest.raises(ValueError):
            verify_self_play_replay(
                tasks, recipe, Agent("solver"), Agent("challenger"), Oracle(),
                forged, trusted_verifier_roots=frozenset({H}),
            )


def test_self_verify_duplicate_roles_and_untrusted_root_rejected_before_policy():
    tasks, recipe = setup()
    agent = Agent("solver")
    with pytest.raises(ValueError):
        run(tasks, recipe, agent, agent)
    with pytest.raises(ValueError):
        run(tasks, recipe, agent, roots=frozenset({"b" * 64}))
    with pytest.raises(ValueError):
        run(tasks, recipe, agent, oracle=type(
            "SelfOracle", (Oracle,), {"verifier_id": "solver"},
        )())
    assert not agent.calls


def test_contamination_eval_duplicate_and_identity():
    tasks, recipe = setup()
    edited = tuple(
        replace(x, split="eval") if i == 0 else x
        for i, x in enumerate(tasks)
    )
    with pytest.raises(ValueError):
        run(edited, recipe)
    with pytest.raises(ValueError):
        run(tasks + (tasks[0],), recipe)
    with pytest.raises(ValueError):
        run(tasks, replace(recipe, curriculum_sha256="b" * 64))


def test_easy_inflation_repetition_and_collapse():
    tasks, recipe = setup()
    easy = tuple(replace(t, tier="easy") for t in tasks)
    edited_recipe = replace(
        recipe, curriculum_sha256=canonical_digest([t.identity() for t in easy]),
    )
    assert "EASY_TASK_INFLATION" in run(easy, edited_recipe).rejected

    class Same(Agent):
        def answer(self, task, *, role, opponent_id, seed):
            return "same"

    repeated = tuple(
        replace(t, accepted_answer_sha256=canonical_digest({"answer": "same"}))
        for t in tasks
    )
    repeated_recipe = replace(
        recipe, curriculum_sha256=canonical_digest([t.identity() for t in repeated]),
    )
    result = run(repeated, repeated_recipe, Same("solver"), Same("challenger"))
    assert "REPEATED_SOLUTION" in result.rejected
    assert result.state == "REJECTED"
    failed = run(
        tasks, replace(recipe, max_consecutive_failures=2),
        Agent("solver", corrupt=True), Agent("challenger", corrupt=True),
    )
    assert len(failed.rounds) == 2 and "COLLAPSE_STOP" in failed.rejected


def test_bad_verifier_and_bounds():
    tasks, recipe = setup()

    class Untrusted(Oracle):
        def verify(self, task, answer):
            return 1

    with pytest.raises(ValueError):
        run(tasks, recipe, oracle=Untrusted())
    with pytest.raises(ValueError):
        run(tasks, replace(recipe, max_rounds=65))
    with pytest.raises(ValueError):
        run(tasks, replace(recipe, max_easy_fraction=float("nan")))
