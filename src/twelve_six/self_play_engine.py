"""Plan 6 Section 8: deterministic, externally verified LOCAL_FREE self-play.

Candidate research only, not production training or champion authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from .post_base_instruction import bounded_id, canonical_digest, sha_field


@dataclass(frozen=True, slots=True)
class PlayTask:
    task_id: str
    context: str
    accepted_answer_sha256: str
    verifier_root_sha256: str
    tier: str
    split: str = "train"
    schema_version: int = 1

    def identity(self) -> str:
        bounded_id(self.task_id)
        if (type(self.context) is not str or not self.context.strip()
                or len(self.context.encode()) > 4096 or self.tier not in ("easy", "hard")
                or self.split != "train" or type(self.schema_version) is not int
                or self.schema_version != 1):
            raise ValueError("invalid self-play task")
        sha_field(self.accepted_answer_sha256)
        sha_field(self.verifier_root_sha256)
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class PlayRecipe:
    curriculum_sha256: str
    seed: int
    max_rounds: int = 12
    min_unique: int = 2
    max_easy_fraction: float = 0.5
    max_consecutive_failures: int = 4
    schema_version: int = 1

    def validate(self) -> None:
        sha_field(self.curriculum_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.max_rounds) is not int or not 1 <= self.max_rounds <= 64
                or type(self.min_unique) is not int or not 1 <= self.min_unique <= self.max_rounds
                or type(self.max_consecutive_failures) is not int
                or not 1 <= self.max_consecutive_failures <= self.max_rounds
                or type(self.max_easy_fraction) not in (int, float)
                or not 0 <= self.max_easy_fraction <= 1):
            raise ValueError("unbounded or invalid play recipe")


class PlayAgent(Protocol):
    agent_id: str
    def answer(self, task: PlayTask, *, role: str, opponent_id: str, seed: int) -> str: ...


class IndependentOracle(Protocol):
    verifier_id: str
    verifier_version_sha256: str
    def verify(self, task: PlayTask, answer: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlayRound:
    task_sha256: str
    round_index: int
    seed: int
    solver_id: str
    opponent_id: str
    verifier_id: str
    verifier_version_sha256: str
    answer_sha256: str
    reward: int


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    recipe_sha256: str
    curriculum_sha256: str
    rounds: tuple[PlayRound, ...]
    rejected: tuple[str, ...]
    evidence_sha256: str
    state: str
    promotion_authorized: bool = False
    canonical_base_eligible: bool = False


def run_self_play(tasks: tuple[PlayTask, ...], recipe: PlayRecipe,
                  solver: PlayAgent, challenger: PlayAgent, oracle: IndependentOracle, *,
                  trusted_verifier_roots: frozenset[str]) -> SelfPlayResult:
    """Fail closed on self-approval, contamination, repetition and easy-task inflation."""
    recipe.validate()
    if type(tasks) is not tuple or not 1 <= len(tasks) <= 64:
        raise ValueError("bounded frozen curriculum required")
    for participant in (solver.agent_id, challenger.agent_id, oracle.verifier_id):
        bounded_id(participant)
    sha_field(oracle.verifier_version_sha256)
    if len({solver.agent_id, challenger.agent_id, oracle.verifier_id}) != 3:
        raise ValueError("circular self-verification or identical opponents")
    if type(trusted_verifier_roots) is not frozenset or not trusted_verifier_roots:
        raise ValueError("independent verifier roots required")
    for root in trusted_verifier_roots:
        sha_field(root)
    seen: set[str] = set()
    for task in tasks:
        if type(task) is not PlayTask or task.verifier_root_sha256 not in trusted_verifier_roots:
            raise ValueError("untrusted task verifier root")
        task.identity()
        if task.task_id in seen:
            raise ValueError("duplicate curriculum tasks")
        seen.add(task.task_id)
    if canonical_digest([t.identity() for t in tasks]) != recipe.curriculum_sha256:
        raise ValueError("curriculum binding mismatch")
    rounds: list[PlayRound] = []
    rejected: list[str] = []
    accepted_tasks: set[str] = set()
    accepted_answers: set[str] = set()
    failures = 0
    easy = 0
    for index in range(min(recipe.max_rounds, len(tasks))):
        task = tasks[index]
        role_solver, role_challenger = (solver, challenger) if index % 2 == 0 else (challenger, solver)
        answer = role_solver.answer(task, role="solver", opponent_id=role_challenger.agent_id,
                                    seed=(recipe.seed + index) % 2**32)
        if type(answer) is not str or not answer.strip() or len(answer.encode()) > 8192:
            raise ValueError("invalid agent answer")
        answer_sha = canonical_digest({"answer": answer})
        oracle_result = oracle.verify(task, answer)
        if type(oracle_result) is not bool:
            raise ValueError("untyped independent verification")
        valid = oracle_result and answer_sha == task.accepted_answer_sha256
        if task.task_id in accepted_tasks or answer_sha in accepted_answers:
            valid = False
            rejected.append("REPEATED_SOLUTION")
        if valid and task.tier == "easy" and (easy + 1) / (len(rounds) + 1) > recipe.max_easy_fraction:
            valid = False
            rejected.append("EASY_TASK_INFLATION")
        reward = int(valid)
        rounds.append(PlayRound(task.identity(), index, (recipe.seed + index) % 2**32,
                                role_solver.agent_id, role_challenger.agent_id,
                                oracle.verifier_id, oracle.verifier_version_sha256,
                                answer_sha, reward))
        if valid:
            accepted_tasks.add(task.task_id)
            accepted_answers.add(answer_sha)
            easy += int(task.tier == "easy")
            failures = 0
        else:
            failures += 1
        if failures >= recipe.max_consecutive_failures:
            rejected.append("COLLAPSE_STOP")
            break
    positive = sum(r.reward for r in rounds)
    state = ("VERIFIED_CANDIDATE" if positive >= recipe.min_unique
             and len(accepted_answers) >= recipe.min_unique else "REJECTED")
    root = canonical_digest({"recipe": asdict(recipe), "rounds": [asdict(r) for r in rounds],
                             "rejected": rejected, "state": state})
    return SelfPlayResult(canonical_digest(asdict(recipe)), recipe.curriculum_sha256,
                          tuple(rounds), tuple(rejected), root, state)


def verify_self_play_replay(tasks: tuple[PlayTask, ...], recipe: PlayRecipe,
                            solver: PlayAgent, challenger: PlayAgent, oracle: IndependentOracle,
                            result: SelfPlayResult, *, trusted_verifier_roots: frozenset[str]) -> bool:
    if type(result) is not SelfPlayResult or result.promotion_authorized or result.canonical_base_eligible:
        raise ValueError("untrusted play receipt")
    replay = run_self_play(tasks, recipe, solver, challenger, oracle,
                           trusted_verifier_roots=trusted_verifier_roots)
    if result != replay:
        raise ValueError("replay mismatch or forged self-reward")
    return True
