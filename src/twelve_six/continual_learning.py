"""Plan 6 Section 9: bounded continual learning with frozen retention gates.

Composes the existing verified Plan-6 RL engine, never a second optimizer.
Candidate-only output; champion and canonical Base remain immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .post_base_instruction import bounded_id, canonical_digest, sha_field
from .post_base_reasoning import (
    ReasoningRecipe,
    TinyPolicy,
    VerifiedTask,
    run_verified_rl,
    verify_reasoning_run,
)


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    probe_id: str
    context_id: str
    expected_action_sha256: str

    def validate(self) -> None:
        bounded_id(self.probe_id)
        bounded_id(self.context_id)
        sha_field(self.expected_action_sha256)


@dataclass(frozen=True, slots=True)
class FrozenCapabilitySuite:
    suite_id: str
    role: str
    version_sha256: str
    probes: tuple[CapabilityProbe, ...]
    schema_version: int = 1

    def identity(self) -> str:
        bounded_id(self.suite_id)
        sha_field(self.version_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.role not in ("retention", "holdout")
                or type(self.probes) is not tuple or not 1 <= len(self.probes) <= 64):
            raise ValueError("unsupported frozen capability suite")
        ids: set[str] = set()
        for probe in self.probes:
            if type(probe) is not CapabilityProbe:
                raise ValueError("invalid capability probe")
            probe.validate()
            if probe.probe_id in ids:
                raise ValueError("duplicate evaluation probe")
            ids.add(probe.probe_id)
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ContinualRecipe:
    champion_sha256: str
    retention_suite_sha256: str
    holdout_suite_sha256: str
    max_cycles: int = 1
    min_holdout_accuracy: float = 0.5
    max_retention_drop: float = 0.0
    min_holdout_gain: float = 0.0
    schema_version: int = 1

    def validate(self) -> None:
        for value in (self.champion_sha256, self.retention_suite_sha256,
                      self.holdout_suite_sha256):
            sha_field(value)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.max_cycles) is not int or not 1 <= self.max_cycles <= 4):
            raise ValueError("unbounded continual learning cycles")
        for value in (self.min_holdout_accuracy, self.max_retention_drop,
                      self.min_holdout_gain):
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("nonfinite or invalid continual learning gate")


@dataclass(frozen=True, slots=True)
class ContinualCycle:
    cycle_index: int
    candidate_sha256: str | None
    train_rollout_sha256: str
    retention_before: float
    retention_after: float
    holdout_before: float
    holdout_after: float
    state: str
    reason: str


@dataclass(frozen=True, slots=True)
class ContinualResult:
    candidate: TinyPolicy | None
    champion_sha256: str
    candidate_sha256: str | None
    input_sha256: str
    cycles: tuple[ContinualCycle, ...]
    evidence_sha256: str
    state: str
    rollback_to_sha256: str
    promotion_authorized: bool = False


def score_suite(policy: TinyPolicy, suite: FrozenCapabilitySuite) -> float:
    """Frozen deterministic top-1 capability measure with no policy feedback."""
    policy.identity()
    suite.identity()
    correct = 0
    for probe in suite.probes:
        if probe.context_id not in policy.contexts:
            raise ValueError("unknown frozen capability context")
        row = policy.logits[policy.contexts.index(probe.context_id)]
        chosen = policy.actions[max(range(len(row)), key=row.__getitem__)]
        correct += int(canonical_digest({"action": chosen}) == probe.expected_action_sha256)
    return correct / len(suite.probes)


def run_continual_update(
    champion: TinyPolicy, new_tasks: tuple[VerifiedTask, ...],
    replay_tasks: tuple[VerifiedTask, ...], reasoning_recipe: ReasoningRecipe,
    recipe: ContinualRecipe, retention_suite: FrozenCapabilitySuite,
    holdout_suite: FrozenCapabilitySuite, *,
    trusted_verifier_roots: frozenset[str], trusted_suite_roots: frozenset[str],
) -> ContinualResult:
    recipe.validate()
    champion_sha = champion.identity()
    reasoning_recipe.validate()
    if recipe.champion_sha256 != champion_sha or reasoning_recipe.parent_sha256 != champion_sha:
        raise ValueError("stale champion/reasoning binding")
    if type(trusted_verifier_roots) is not frozenset or not trusted_verifier_roots:
        raise ValueError("missing independent reward roots")
    for root in trusted_verifier_roots:
        sha_field(root)
    if canonical_digest(sorted(trusted_verifier_roots)) != reasoning_recipe.verifier_set_sha256:
        raise ValueError("reasoning reward authority mismatch")
    if type(trusted_suite_roots) is not frozenset or len(trusted_suite_roots) < 2:
        raise ValueError("independent frozen suite roots required")
    for root in trusted_suite_roots:
        sha_field(root)
    if (retention_suite.role != "retention" or holdout_suite.role != "holdout"
            or retention_suite.suite_id == holdout_suite.suite_id
            or retention_suite.identity() != recipe.retention_suite_sha256
            or holdout_suite.identity() != recipe.holdout_suite_sha256
            or recipe.retention_suite_sha256 not in trusted_suite_roots
            or recipe.holdout_suite_sha256 not in trusted_suite_roots):
        raise ValueError("missing/stale/untrusted frozen suite")
    if (type(new_tasks) is not tuple or type(replay_tasks) is not tuple
            or not new_tasks or not replay_tasks
            or len(new_tasks) + len(replay_tasks) > 64):
        raise ValueError("bounded new and replay tasks are both required")
    train = new_tasks + replay_tasks
    seen: set[str] = set()
    for task in train:
        if type(task) is not VerifiedTask or task.root() not in trusted_verifier_roots:
            raise ValueError("unverified training/replay reward")
        if task.task_id in seen or task.context_id not in champion.contexts:
            raise ValueError("duplicate task or unknown context")
        seen.add(task.task_id)
    probes = retention_suite.probes + holdout_suite.probes
    eval_ids = [p.probe_id for p in probes]
    if len(eval_ids) != len(set(eval_ids)) or seen.intersection(eval_ids):
        raise ValueError("holdout/replay contamination")
    retention_before = score_suite(champion, retention_suite)
    holdout_before = score_suite(champion, holdout_suite)
    input_root = canonical_digest({
        "recipe": asdict(recipe), "rl": asdict(reasoning_recipe),
        "new": [asdict(x) for x in new_tasks],
        "replay": [asdict(x) for x in replay_tasks],
        "retention": recipe.retention_suite_sha256, "holdout": recipe.holdout_suite_sha256,
    })
    current = champion
    records: list[ContinualCycle] = []
    outcome = "CANDIDATE_ONLY"
    for index in range(recipe.max_cycles):
        cycle_recipe = replace(reasoning_recipe, parent_sha256=current.identity(),
                               seed=(reasoning_recipe.seed + index) % 2**32)
        result = run_verified_rl(current, train, cycle_recipe,
                                 trusted_verifier_roots=trusted_verifier_roots)
        verify_reasoning_run(current, train, cycle_recipe, result,
                             trusted_verifier_roots=trusted_verifier_roots)
        next_policy = result.candidate
        after_retention = score_suite(next_policy, retention_suite) if next_policy else retention_before
        after_holdout = score_suite(next_policy, holdout_suite) if next_policy else holdout_before
        allowed = (next_policy is not None
                   and next_policy.contexts == champion.contexts
                   and next_policy.actions == champion.actions
                   and after_retention + 1e-12 >= retention_before - recipe.max_retention_drop
                   and after_holdout + 1e-12 >= recipe.min_holdout_accuracy
                   and after_holdout + 1e-12 >= holdout_before + recipe.min_holdout_gain)
        reason = ("qualified_frozen_capabilities" if allowed else
                  "forgetting_or_holdout_regression" if next_policy else result.receipt.cause)
        records.append(ContinualCycle(
            index, next_policy.identity() if next_policy else None,
            result.receipt.rollout_sha256, retention_before, after_retention,
            holdout_before, after_holdout,
            "QUALIFIED" if allowed else "ROLLED_BACK", reason,
        ))
        if not allowed:
            outcome = "ROLLED_BACK"
            current = champion
            break
        current = next_policy
    candidate = current if outcome == "CANDIDATE_ONLY" and current.identity() != champion_sha else None
    if candidate is None:
        outcome = "ROLLED_BACK"
    receipt = canonical_digest({"input": input_root, "cycles": [asdict(c) for c in records],
                                "candidate": candidate.identity() if candidate else None,
                                "state": outcome, "rollback": champion_sha})
    return ContinualResult(candidate, champion_sha,
                           candidate.identity() if candidate else None,
                           input_root, tuple(records), receipt, outcome, champion_sha)


def verify_continual_replay(
    champion: TinyPolicy, new_tasks: tuple[VerifiedTask, ...],
    replay_tasks: tuple[VerifiedTask, ...], reasoning_recipe: ReasoningRecipe,
    recipe: ContinualRecipe, retention_suite: FrozenCapabilitySuite,
    holdout_suite: FrozenCapabilitySuite, result: ContinualResult, *,
    trusted_verifier_roots: frozenset[str], trusted_suite_roots: frozenset[str],
) -> bool:
    if type(result) is not ContinualResult or result.promotion_authorized:
        raise ValueError("untrusted continual receipt")
    recomputed = run_continual_update(
        champion, new_tasks, replay_tasks, reasoning_recipe, recipe,
        retention_suite, holdout_suite,
        trusted_verifier_roots=trusted_verifier_roots,
        trusted_suite_roots=trusted_suite_roots,
    )
    if recomputed != result:
        raise ValueError("continual restart/replay mismatch")
    return True
