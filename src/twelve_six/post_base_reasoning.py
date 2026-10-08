"""Plan 6 Section 4: independently verified, bounded contextual-bandit RL fixture.

The verifier catalog is supplied by an independent authority. This produces
candidate descendants, never champion promotion or a production training run.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace

from .post_base_instruction import bounded_id, canonical_digest, sha_field


@dataclass(frozen=True, slots=True)
class VerifiedTask:
    task_id: str
    context_id: str
    accepted_action_sha256: str
    verifier_sha256: str
    evidence_sha256: str
    split: str = "train"
    schema_version: int = 1

    def root(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported verifier task version")
        bounded_id(self.task_id)
        bounded_id(self.context_id)
        for value in (self.accepted_action_sha256, self.verifier_sha256, self.evidence_sha256):
            sha_field(value)
        if self.split != "train" or type(self.split) is not str:
            raise ValueError("reserved evaluation cannot enter RL training")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class TinyPolicy:
    contexts: tuple[str, ...]
    actions: tuple[str, ...]
    logits: tuple[tuple[float, ...], ...]
    schema_version: int = 1

    def identity(self) -> str:
        if self.schema_version != 1 or type(self.contexts) is not tuple:
            raise ValueError("unsupported policy version")
        if not 1 <= len(self.contexts) <= 32 or len(set(self.contexts)) != len(self.contexts):
            raise ValueError("invalid contexts")
        if type(self.actions) is not tuple or not 2 <= len(self.actions) <= 32:
            raise ValueError("invalid action catalog")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("duplicate actions")
        for name in self.contexts + self.actions:
            bounded_id(name)
        if type(self.logits) is not tuple or len(self.logits) != len(self.contexts):
            raise ValueError("invalid policy shape")
        for row in self.logits:
            if type(row) is not tuple or len(row) != len(self.actions):
                raise ValueError("invalid policy row")
            if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 30 for x in row):
                raise ValueError("nonfinite or unsafe policy")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ReasoningRecipe:
    parent_sha256: str
    verifier_set_sha256: str
    seed: int = 17
    max_steps: int = 16
    learning_rate: float = 0.2
    min_entropy: float = 0.02
    max_consecutive_misses: int = 12
    schema_version: int = 1

    def validate(self) -> None:
        sha_field(self.parent_sha256)
        sha_field(self.verifier_set_sha256)
        if self.schema_version != 1:
            raise ValueError("unsupported RL recipe version")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("invalid seed")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 64:
            raise ValueError("unbounded RL steps")
        if (type(self.max_consecutive_misses) is not int
                or not 1 <= self.max_consecutive_misses <= 64):
            raise ValueError("invalid stop bound")
        for name, value, upper in (("learning rate", self.learning_rate, 0.2),
                                   ("entropy", self.min_entropy, 0.5)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 < value <= upper):
                raise ValueError(f"unsafe RL {name}")


@dataclass(frozen=True, slots=True)
class VerifiedRollout:
    step: int
    task_id: str
    policy_before_sha256: str
    action: str
    action_sha256: str
    reward: int
    verifier_root_sha256: str


@dataclass(frozen=True, slots=True)
class ReasoningReceipt:
    parent_sha256: str
    descendant_sha256: str | None
    recipe_sha256: str
    verifier_set_sha256: str
    rollout_sha256: str
    positive_rewards: int
    status: str
    cause: str
    promotion_authorized: bool = False


@dataclass(frozen=True, slots=True)
class ReasoningRun:
    candidate: TinyPolicy | None
    rollouts: tuple[VerifiedRollout, ...]
    receipt: ReasoningReceipt


def _distribution(row: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    weights = [math.exp(x - max(row)) for x in row]
    z = sum(weights)
    probabilities = tuple(x / z for x in weights)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return probabilities, entropy


def run_verified_rl(
    parent: TinyPolicy, tasks: tuple[VerifiedTask, ...], recipe: ReasoningRecipe,
    *, trusted_verifier_roots: frozenset[str],
) -> ReasoningRun:
    """REINFORCE on toy categorical policy; no answer/target gradients or auto-promotion."""
    recipe.validate()
    parent_id = parent.identity()
    if parent_id != recipe.parent_sha256:
        raise ValueError("parent model identity mismatch")
    if type(tasks) is not tuple or not 1 <= len(tasks) <= 64:
        raise ValueError("unbounded verifier tasks")
    if type(trusted_verifier_roots) is not frozenset or not trusted_verifier_roots:
        raise ValueError("independent verifier authority required")
    for root in trusted_verifier_roots:
        sha_field(root)
    if (canonical_digest(sorted(trusted_verifier_roots))
            != recipe.verifier_set_sha256):
        raise ValueError("verifier set substitution")
    seen: set[str] = set()
    for task in tasks:
        if type(task) is not VerifiedTask or task.root() not in trusted_verifier_roots:
            raise ValueError("untrusted reward verifier")
        if task.task_id in seen or task.context_id not in parent.contexts:
            raise ValueError("duplicate task or unknown context")
        seen.add(task.task_id)
    # Bound run state and serialize deterministic step-by-step identities.
    candidate = parent
    rng = random.Random(recipe.seed)
    rollouts: list[VerifiedRollout] = []
    consecutive_misses = 0
    cause = "qualified_candidate"
    for step in range(recipe.max_steps):
        task = tasks[step % len(tasks)]
        row_idx = candidate.contexts.index(task.context_id)
        probs, entropy = _distribution(candidate.logits[row_idx])
        if entropy < recipe.min_entropy:
            cause = "degenerate_policy_entropy"
            break
        draw = rng.random()
        cumulative = 0.0
        action_idx = len(probs) - 1
        for i, probability in enumerate(probs):
            cumulative += probability
            if draw < cumulative:
                action_idx = i
                break
        action = candidate.actions[action_idx]
        action_hash = canonical_digest({"action": action})
        reward = int(action_hash == task.accepted_action_sha256)
        rollouts.append(VerifiedRollout(
            step, task.task_id, candidate.identity(), action, action_hash,
            reward, task.root(),
        ))
        consecutive_misses = consecutive_misses + 1 if reward == 0 else 0
        if consecutive_misses >= recipe.max_consecutive_misses:
            cause = "repeated_unverified_rewards"
            break
        # Advantage is strictly the independently verified reward, never the oracle target.
        advantage = float(reward) - 0.5
        updated = tuple(float(value + recipe.learning_rate * advantage *
                              ((1.0 if i == action_idx else 0.0) - probs[i]))
                        for i, value in enumerate(candidate.logits[row_idx]))
        rows = list(candidate.logits)
        rows[row_idx] = updated
        candidate = replace(candidate, logits=tuple(rows))
        try:
            candidate.identity()
        except ValueError:
            cause = "unstable_policy_parameters"
            break
    positive = sum(x.reward for x in rollouts)
    if positive == 0 and cause == "qualified_candidate":
        cause = "no_verified_success"
    accepted = cause == "qualified_candidate" and len(rollouts) == recipe.max_steps
    result = candidate if accepted and candidate.identity() != parent_id else None
    if result is None and accepted:
        cause = "unchanged_policy"
    receipt = ReasoningReceipt(
        parent_id, result.identity() if result else None,
        canonical_digest(asdict(recipe)), recipe.verifier_set_sha256,
        canonical_digest([asdict(x) for x in rollouts]), positive,
        "CANDIDATE_ONLY" if result else "ROLLED_BACK", cause,
    )
    return ReasoningRun(result, tuple(rollouts), receipt)


def verify_reasoning_run(
    parent: TinyPolicy, tasks: tuple[VerifiedTask, ...], recipe: ReasoningRecipe,
    result: ReasoningRun, *, trusted_verifier_roots: frozenset[str],
) -> bool:
    """Independent-root-bound deterministic replay; never treats candidate claim as proof."""
    if type(result) is not ReasoningRun:
        raise ValueError("invalid reasoning receipt")
    replay = run_verified_rl(parent, tasks, recipe, trusted_verifier_roots=trusted_verifier_roots)
    if replay != result or result.receipt.promotion_authorized:
        raise ValueError("reasoning replay or promotion evidence mismatch")
    return True
