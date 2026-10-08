"""Plan 6 Section 6: ModelGateway teacher candidates, budget gates and independent council.

Injected gateways may be local fixtures or Plan-4 adapters. No network,
training, Base writes, or promotion is performed by this component.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .post_base_instruction import bounded_id, canonical_digest, sha_field


def _seal(value: object, key: bytes) -> str:
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("trusted verifier key must be 32+ bytes")
    return hmac.new(
        key, json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False).encode(), hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherSpec:
    teacher_id: str
    provider: str
    model_sha256: str
    policy_sha256: str
    capability: str
    max_cost_microusd: int
    transport: str = "local"
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported teacher schema")
        for name in (self.teacher_id, self.provider, self.capability):
            bounded_id(name)
        sha_field(self.model_sha256)
        sha_field(self.policy_sha256)
        if self.transport not in ("local", "server", "external"):
            raise ValueError("unknown teacher transport")
        if (type(self.max_cost_microusd) is not int
                or not 0 <= self.max_cost_microusd <= 1_000_000):
            raise ValueError("unbounded teacher cost")


@dataclass(frozen=True, slots=True)
class GatewayReply:
    teacher_id: str
    model_sha256: str
    prompt_sha256: str
    text: str
    cost_microusd: int
    abstained: bool = False
    schema_version: int = 1

    def identity(self) -> str:
        bounded_id(self.teacher_id)
        sha_field(self.model_sha256)
        sha_field(self.prompt_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.abstained) is not bool
                or type(self.text) is not str or len(self.text.encode("utf-8")) > 8192
                or (not self.abstained and not self.text.strip())
                or (self.abstained and self.text != "")
                or type(self.cost_microusd) is not int
                or not 0 <= self.cost_microusd <= 1_000_000):
            raise ValueError("invalid bounded gateway reply")
        return canonical_digest(asdict(self))


class ModelGateway(Protocol):
    def generate(self, spec: TeacherSpec, prompt: str) -> GatewayReply: ...


@dataclass(frozen=True, slots=True)
class CouncilBatch:
    prompt: str
    prompt_sha256: str
    capability: str
    policy_sha256: str
    teachers: tuple[TeacherSpec, ...]
    replies: tuple[GatewayReply, ...]
    cost_microusd: int
    disagreement: bool
    status: str = "UNVERIFIED_CANDIDATES"

    def validate(self) -> None:
        if self.status != "UNVERIFIED_CANDIDATES" or type(self.prompt) is not str:
            raise ValueError("invalid council state")
        if (not self.prompt.strip() or len(self.prompt.encode("utf-8")) > 8192
                or self.prompt_sha256 != hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()):
            raise ValueError("foreign council prompt")
        bounded_id(self.capability)
        sha_field(self.policy_sha256)
        if (type(self.teachers) is not tuple or type(self.replies) is not tuple
                or not 1 <= len(self.teachers) <= 8
                or len(self.teachers) != len(self.replies)):
            raise ValueError("invalid council members")
        seen: set[str] = set()
        total = 0
        responses: set[str] = set()
        for spec, reply in zip(self.teachers, self.replies, strict=True):
            spec.validate()
            reply.identity()
            if (spec.teacher_id in seen or spec.capability != self.capability
                    or spec.policy_sha256 != self.policy_sha256
                    or reply.teacher_id != spec.teacher_id
                    or reply.model_sha256 != spec.model_sha256
                    or reply.prompt_sha256 != self.prompt_sha256
                    or reply.cost_microusd > spec.max_cost_microusd):
                raise ValueError("foreign council member")
            seen.add(spec.teacher_id)
            total += reply.cost_microusd
            if not reply.abstained:
                responses.add(reply.text)
        if (self.cost_microusd != total or type(self.disagreement) is not bool
                or self.disagreement != (len(responses) > 1)):
            raise ValueError("forged council accounting")

    def identity(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


def collect_teacher_candidates(
    gateway: ModelGateway, *, prompt: str, capability: str,
    policy_sha256: str, teachers: tuple[TeacherSpec, ...],
    total_budget_microusd: int,
) -> CouncilBatch:
    """Create immutable untrusted candidate data; no provider result is truth."""
    if (type(prompt) is not str or not prompt.strip()
            or len(prompt.encode("utf-8")) > 8192):
        raise ValueError("invalid prompt")
    bounded_id(capability)
    sha_field(policy_sha256)
    if (type(teachers) is not tuple or not 1 <= len(teachers) <= 8
            or type(total_budget_microusd) is not int
            or not 0 <= total_budget_microusd <= 1_000_000):
        raise ValueError("unbounded council")
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    seen: set[str] = set()
    replies: list[GatewayReply] = []
    spent = 0
    for spec in teachers:
        if type(spec) is not TeacherSpec:
            raise ValueError("untyped teacher")
        spec.validate()
        if (spec.teacher_id in seen or spec.capability != capability
                or spec.policy_sha256 != policy_sha256):
            raise ValueError("duplicate teacher or unauthorized capability/policy")
        seen.add(spec.teacher_id)
        if spent + spec.max_cost_microusd > total_budget_microusd:
            raise ValueError("teacher budget preflight denied")
        reply = gateway.generate(spec, prompt)
        if type(reply) is not GatewayReply:
            raise ValueError("untyped ModelGateway reply")
        reply.identity()
        if (reply.teacher_id != spec.teacher_id
                or reply.model_sha256 != spec.model_sha256
                or reply.prompt_sha256 != prompt_sha
                or reply.cost_microusd > spec.max_cost_microusd):
            raise ValueError("foreign gateway identity, prompt or cost")
        spent += reply.cost_microusd
        replies.append(reply)
    texts = {r.text for r in replies if not r.abstained}
    batch = CouncilBatch(prompt, prompt_sha, capability, policy_sha256,
                         teachers, tuple(replies), spent, len(texts) > 1)
    batch.validate()
    return batch


@dataclass(frozen=True, slots=True)
class CouncilVerdict:
    batch_sha256: str
    selected_reply_sha256: str | None
    verifier_id: str
    verifier_version_sha256: str
    accepted: bool
    signature: str

    def payload(self) -> dict[str, object]:
        return {"batch_sha256": self.batch_sha256,
                "selected_reply_sha256": self.selected_reply_sha256,
                "verifier_id": self.verifier_id,
                "verifier_version_sha256": self.verifier_version_sha256,
                "accepted": self.accepted}


def attest_council(
    batch: CouncilBatch, *, selected_reply_sha256: str | None,
    verifier_id: str, verifier_version_sha256: str,
    verifier_key: bytes, independent_passed: bool,
) -> CouncilVerdict:
    """Only the trusted host controls the secret and independent test outcome."""
    if type(batch) is not CouncilBatch or batch.status != "UNVERIFIED_CANDIDATES":
        raise ValueError("untrusted batch")
    bounded_id(verifier_id)
    sha_field(verifier_version_sha256)
    if verifier_id in {s.teacher_id for s in batch.teachers}:
        raise ValueError("self-issued teacher verdict")
    if type(independent_passed) is not bool:
        raise ValueError("untyped independent result")
    eligible = {r.identity() for r in batch.replies if not r.abstained}
    if selected_reply_sha256 is not None:
        sha_field(selected_reply_sha256)
    if independent_passed and selected_reply_sha256 not in eligible:
        raise ValueError("accepted reply not in candidate batch")
    if not independent_passed and selected_reply_sha256 is not None:
        raise ValueError("rejected verdict must abstain")
    value = {"batch_sha256": batch.identity(),
             "selected_reply_sha256": selected_reply_sha256,
             "verifier_id": verifier_id,
             "verifier_version_sha256": verifier_version_sha256,
             "accepted": independent_passed}
    return CouncilVerdict(**value, signature=_seal(value, verifier_key))


def verify_council(
    batch: CouncilBatch, verdict: CouncilVerdict, *, trusted_verifier_id: str,
    trusted_verifier_version_sha256: str, verifier_key: bytes,
) -> bool:
    if type(batch) is not CouncilBatch or type(verdict) is not CouncilVerdict:
        raise ValueError("untyped council evidence")
    if (verdict.batch_sha256 != batch.identity()
            or verdict.verifier_id != trusted_verifier_id
            or verdict.verifier_id in {s.teacher_id for s in batch.teachers}
            or verdict.verifier_version_sha256 != trusted_verifier_version_sha256
            or type(verdict.accepted) is not bool):
        raise ValueError("foreign or self-issued council evidence")
    eligible = {r.identity() for r in batch.replies if not r.abstained}
    if verdict.accepted != (verdict.selected_reply_sha256 in eligible):
        raise ValueError("unbound accepted candidate")
    sha_field(verdict.signature)
    if not hmac.compare_digest(_seal(verdict.payload(), verifier_key), verdict.signature):
        raise ValueError("forged council verdict")
    return True
