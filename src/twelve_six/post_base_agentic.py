"""Plan 6 Section 5: verified tool trajectories and bounded agentic descendants.

Data ingestion never performs tool calls, trusts model-reported effects, or promotes models.
Only an independent host that holds verifier_key can attest a trajectory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace

from .post_base_instruction import bounded_id, canonical_digest, sha_field, validate_source
from .post_base_reasoning import TinyPolicy


def _seal(payload: object, key: bytes) -> str:
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("independent host verification key required")
    return hmac.new(key, json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolStep:
    tool_name: str
    action_name: str
    tool_schema_sha256: str
    arguments_sha256: str
    result_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    external_side_effect: bool = False

    def validate(self) -> None:
        bounded_id(self.tool_name)
        bounded_id(self.action_name)
        for value in (self.tool_schema_sha256, self.arguments_sha256,
                      self.result_sha256, self.state_before_sha256,
                      self.state_after_sha256):
            sha_field(value)
        if type(self.external_side_effect) is not bool:
            raise ValueError("typed side-effect flag required")


@dataclass(frozen=True, slots=True)
class ToolTrajectory:
    trajectory_id: str
    context_id: str
    producer_id: str
    steps: tuple[ToolStep, ...]
    provenance_sha256: str
    rights_evidence_sha256: str
    rights: str = "owned"
    quality: float = 1.0
    split: str = "train"
    schema_version: int = 1

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported trajectory version")
        bounded_id(self.producer_id)
        bounded_id(self.context_id)
        validate_source(sample_id=self.trajectory_id, provenance_sha256=self.provenance_sha256,
                        rights=self.rights, rights_evidence_sha256=self.rights_evidence_sha256,
                        split=self.split, quality=self.quality)
        if type(self.steps) is not tuple or not 1 <= len(self.steps) <= 16:
            raise ValueError("unbounded tool steps")
        for index, step in enumerate(self.steps):
            if type(step) is not ToolStep:
                raise ValueError("untyped tool step")
            step.validate()
            if index and self.steps[index - 1].state_after_sha256 != step.state_before_sha256:
                raise ValueError("broken tool state lineage")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ToolVerdict:
    trajectory_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    success: bool
    external_effects_verified: bool
    signature: str

    def payload(self) -> dict[str, object]:
        return {"trajectory_sha256": self.trajectory_sha256, "verifier_id": self.verifier_id,
                "verifier_version_sha256": self.verifier_version_sha256,
                "success": self.success, "external_effects_verified": self.external_effects_verified}


def attest_tool_trajectory(trajectory: ToolTrajectory, *, verifier_id: str,
                           verifier_version_sha256: str, verifier_key: bytes,
                           success: bool, external_effects_verified: bool) -> ToolVerdict:
    """Only an independent trusted harness may call this with its secret key."""
    root = trajectory.identity()
    bounded_id(verifier_id)
    sha_field(verifier_version_sha256)
    if verifier_id == trajectory.producer_id or type(success) is not bool or type(external_effects_verified) is not bool:
        raise ValueError("self-issued or untyped evidence")
    data = {"trajectory_sha256": root, "verifier_id": verifier_id,
            "verifier_version_sha256": verifier_version_sha256,
            "success": success, "external_effects_verified": external_effects_verified}
    return ToolVerdict(**data, signature=_seal(data, verifier_key))


def verify_tool_trajectory(trajectory: ToolTrajectory, evidence: ToolVerdict,
                           *, trusted_verifier_id: str, trusted_verifier_version_sha256: str,
                           verifier_key: bytes) -> bool:
    if type(evidence) is not ToolVerdict or type(trajectory) is not ToolTrajectory:
        raise ValueError("typed independent tool verification required")
    if (evidence.trajectory_sha256 != trajectory.identity()
            or evidence.verifier_id != trusted_verifier_id
            or evidence.verifier_id == trajectory.producer_id
            or evidence.verifier_version_sha256 != trusted_verifier_version_sha256):
        raise ValueError("foreign, self-issued or stale tool evidence")
    sha_field(evidence.verifier_version_sha256)
    if type(evidence.success) is not bool or type(evidence.external_effects_verified) is not bool:
        raise ValueError("invalid verdict")
    sha_field(evidence.signature)
    if not hmac.compare_digest(_seal(evidence.payload(), verifier_key), evidence.signature):
        raise ValueError("forged tool verdict")
    if any(step.external_side_effect for step in trajectory.steps) and not evidence.external_effects_verified:
        raise ValueError("unverified external side effect cannot be learned as correct")
    return True


@dataclass(frozen=True, slots=True)
class AgenticRecipe:
    parent_sha256: str
    verifier_version_sha256: str
    mode: str
    learning_rate: float = 0.1
    max_updates: int = 16
    schema_version: int = 1

    def validate(self) -> None:
        sha_field(self.parent_sha256)
        sha_field(self.verifier_version_sha256)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported recipe")
        if self.mode not in ("imitation", "feedback", "rl_style"):
            raise ValueError("unsupported agentic recipe")
        if (type(self.max_updates) is not int or not 1 <= self.max_updates <= 64
                or type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.2):
            raise ValueError("unbounded agentic optimization")


@dataclass(frozen=True, slots=True)
class AgenticRun:
    candidate: TinyPolicy | None
    parent_sha256: str
    recipe_sha256: str
    data_sha256: str
    descendant_sha256: str | None
    status: str
    promotion_authorized: bool = False


def train_agentic_descendant(parent: TinyPolicy, records: tuple[tuple[ToolTrajectory, ToolVerdict], ...],
                             recipe: AgenticRecipe, *, trusted_verifier_id: str,
                             verifier_key: bytes) -> AgenticRun:
    """Tiny categorical optimizer shared by imitation/feedback/RL-style recipe adapters."""
    recipe.validate()
    if type(parent) is not TinyPolicy or parent.identity() != recipe.parent_sha256:
        raise ValueError("parent identity mismatch")
    if type(records) is not tuple or not 1 <= len(records) <= recipe.max_updates:
        raise ValueError("unbounded agentic records")
    seen: set[str] = set()
    verified = []
    for record in records:
        if type(record) is not tuple or len(record) != 2:
            raise ValueError("invalid trajectory/verdict pair")
        trajectory, verdict = record
        verify_tool_trajectory(trajectory, verdict, trusted_verifier_id=trusted_verifier_id,
                               trusted_verifier_version_sha256=recipe.verifier_version_sha256,
                               verifier_key=verifier_key)
        if trajectory.trajectory_id in seen or trajectory.context_id not in parent.contexts:
            raise ValueError("duplicate trajectory or foreign context")
        seen.add(trajectory.trajectory_id)
        if any(step.action_name not in parent.actions for step in trajectory.steps):
            raise ValueError("action outside policy catalog")
        verified.append((trajectory, verdict))
    verified.sort(key=lambda item: item[0].trajectory_id)
    digest = canonical_digest([{"trajectory": t.identity(), "verdict": asdict(v)} for t, v in verified])
    # An unverified or failed external action cannot contribute positive imitation labels.
    if not any(v.success for _, v in verified):
        raise ValueError("no independently verified successful trajectories")
    rows = [list(row) for row in parent.logits]
    for trajectory, verdict in verified:
        if recipe.mode == "imitation" and not verdict.success:
            continue
        target = parent.actions.index(trajectory.steps[-1].action_name)
        idx = parent.contexts.index(trajectory.context_id)
        row = rows[idx]
        maximum = max(row)
        exps = [math.exp(x - maximum) for x in row]
        denom = sum(exps)
        weight = (1 if verdict.success else -1) * (0.5 if recipe.mode == "rl_style" else 1)
        for j in range(len(row)):
            row[j] += recipe.learning_rate * weight * ((1 if j == target else 0) - exps[j] / denom)
    candidate = replace(parent, logits=tuple(tuple(row) for row in rows))
    descendant = candidate.identity()
    if descendant == recipe.parent_sha256:
        raise ValueError("agentic training produced no update")
    if parent.identity() != recipe.parent_sha256:
        raise RuntimeError("mutated parent")
    return AgenticRun(candidate, recipe.parent_sha256, canonical_digest(asdict(recipe)),
                     digest, descendant, "CANDIDATE_ONLY")


def verify_agentic_run(parent: TinyPolicy, records: tuple[tuple[ToolTrajectory, ToolVerdict], ...],
                       recipe: AgenticRecipe, result: AgenticRun, *, trusted_verifier_id: str,
                       verifier_key: bytes) -> bool:
    if type(result) is not AgenticRun or result.promotion_authorized:
        raise ValueError("untrusted promotion")
    replay = train_agentic_descendant(parent, records, recipe,
                                    trusted_verifier_id=trusted_verifier_id, verifier_key=verifier_key)
    if replay != result:
        raise ValueError("non-reproducible agentic receipt")
    return True
