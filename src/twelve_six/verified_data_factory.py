"""Plan 6 Section 7: bounded verified candidate-data factory; never Base-admissible.

Producer adapters and independent verifier are injected LOCAL_FREE fixture interfaces.
No provider credential, real tool effect, training or champion promotion is performed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .post_base_instruction import bounded_id, canonical_digest, sha_field, validate_source

_ALLOWED_KINDS = frozenset({"teacher", "self_play", "tool"})


def _seal(value: object, key: bytes) -> str:
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("trusted verifier key required")
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def content_fingerprint(task: str, answer: str) -> str:
    if type(task) is not str or type(answer) is not str:
        raise ValueError("non-text data")
    if not task.strip() or not answer.strip():
        raise ValueError("empty task or solution")
    if len(task.encode()) > 8192 or len(answer.encode()) > 8192:
        raise ValueError("oversized candidate")
    return canonical_digest({
        "task": " ".join(task.casefold().split()),
        "answer": " ".join(answer.casefold().split()),
    })


@dataclass(frozen=True, slots=True)
class CurriculumGoal:
    goal_id: str
    curriculum_sha256: str
    holdout_fingerprints: tuple[str, ...]
    seed: int
    max_attempts: int = 8
    target_verified: int = 2
    max_total_microunits: int = 0
    max_cost_per_attempt: int = 0
    allow_external: bool = False
    allow_paid: bool = False
    schema_version: int = 1

    def validate(self) -> None:
        bounded_id(self.goal_id)
        sha_field(self.curriculum_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.holdout_fingerprints) is not tuple
                or len(set(self.holdout_fingerprints)) != len(self.holdout_fingerprints)):
            raise ValueError("invalid goal or holdout")
        for value in self.holdout_fingerprints:
            sha_field(value)
        if (type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 64
                or type(self.target_verified) is not int
                or not 1 <= self.target_verified <= self.max_attempts
                or type(self.max_total_microunits) is not int
                or not 0 <= self.max_total_microunits <= 1_000_000
                or type(self.max_cost_per_attempt) is not int
                or not 0 <= self.max_cost_per_attempt <= 1_000_000
                or type(self.allow_external) is not bool or type(self.allow_paid) is not bool):
            raise ValueError("unbounded data generation goal")
        if not self.allow_paid and (self.max_total_microunits or self.max_cost_per_attempt):
            raise ValueError("paid generation not authorized")


@dataclass(frozen=True, slots=True)
class DataCandidate:
    candidate_id: str
    goal_id: str
    producer_id: str
    origin_kind: str
    task: str
    answer: str
    source_sha256: str
    provenance_sha256: str
    rights: str
    rights_evidence_sha256: str
    quality: float
    charged_microunits: int = 0
    split: str = "train"
    external_side_effect: bool = False
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("invalid candidate schema")
        bounded_id(self.producer_id)
        bounded_id(self.goal_id)
        if self.origin_kind not in _ALLOWED_KINDS:
            raise ValueError("invalid generation source")
        sha_field(self.source_sha256)
        validate_source(
            sample_id=self.candidate_id, provenance_sha256=self.provenance_sha256,
            rights=self.rights, rights_evidence_sha256=self.rights_evidence_sha256,
            split=self.split, quality=self.quality,
        )
        content_fingerprint(self.task, self.answer)
        if (type(self.charged_microunits) is not int
                or not 0 <= self.charged_microunits <= 1_000_000
                or type(self.external_side_effect) is not bool):
            raise ValueError("invalid cost or effect")

    def identity(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


class DataGenerator(Protocol):
    producer_id: str
    origin_kind: str
    max_cost_microunits: int
    external: bool

    def generate(self, goal: CurriculumGoal, attempt: int, seed: int) -> DataCandidate: ...


class IndependentVerifier(Protocol):
    verifier_id: str
    verifier_version_sha256: str

    def verify(self, goal: CurriculumGoal, candidate: DataCandidate) -> bool: ...


@dataclass(frozen=True, slots=True)
class FactoryReceipt:
    candidate_sha256: str
    goal_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    accepted: bool
    signature: str

    def payload(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "signature"}


@dataclass(frozen=True, slots=True)
class VerifiedDataPool:
    goal_sha256: str
    accepted: tuple[DataCandidate, ...]
    receipts: tuple[FactoryReceipt, ...]
    rejected: tuple[tuple[str, str], ...]
    attempts: int
    total_cost_microunits: int
    manifest_sha256: str
    status: str = "CANDIDATE_POOL_ONLY"
    canonical_base_eligible: bool = False
    promotion_authorized: bool = False


def verify_factory_receipt(
    candidate: DataCandidate, goal: CurriculumGoal, receipt: FactoryReceipt, *,
    verifier_id: str, verifier_version_sha256: str, verifier_key: bytes,
) -> bool:
    if (type(receipt) is not FactoryReceipt
            or receipt.candidate_sha256 != candidate.identity()
            or receipt.goal_sha256 != canonical_digest(asdict(goal))
            or receipt.verifier_id != verifier_id
            or receipt.verifier_id == candidate.producer_id
            or receipt.verifier_version_sha256 != verifier_version_sha256
            or type(receipt.accepted) is not bool):
        raise ValueError("untrusted or self-issued factory evidence")
    sha_field(receipt.signature)
    if not hmac.compare_digest(receipt.signature, _seal(receipt.payload(), verifier_key)):
        raise ValueError("forged factory evidence")
    return True


def generate_verified_pool(
    goal: CurriculumGoal, generators: tuple[DataGenerator, ...],
    verifier: IndependentVerifier, *, verifier_key: bytes,
) -> VerifiedDataPool:
    """Bounded deterministic generation -> independent verification -> candidate pool only."""
    goal.validate()
    if type(generators) is not tuple or not 1 <= len(generators) <= 3:
        raise ValueError("invalid generator set")
    bounded_id(verifier.verifier_id)
    sha_field(verifier.verifier_version_sha256)
    seen_producers: set[str] = set()
    for gen in generators:
        bounded_id(gen.producer_id)
        if (gen.producer_id in seen_producers or gen.producer_id == verifier.verifier_id
                or gen.origin_kind not in _ALLOWED_KINDS
                or type(gen.external) is not bool or (gen.external and not goal.allow_external)
                or type(gen.max_cost_microunits) is not int
                or not 0 <= gen.max_cost_microunits <= goal.max_cost_per_attempt
                or (gen.max_cost_microunits and not goal.allow_paid)):
            raise ValueError("untrusted/unauthorized generator")
        seen_producers.add(gen.producer_id)
    accepted: list[DataCandidate] = []
    receipts: list[FactoryReceipt] = []
    rejected: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    total = 0
    attempts = 0
    goal_sha = canonical_digest(asdict(goal))
    for i in range(goal.max_attempts):
        if len(accepted) >= goal.target_verified:
            break
        gen = generators[i % len(generators)]
        if total + gen.max_cost_microunits > goal.max_total_microunits:
            break
        produced = gen.generate(goal, i, goal.seed + i)
        attempts += 1
        if type(produced) is not DataCandidate:
            raise ValueError("untyped generator output")
        produced.validate()
        if (produced.goal_id != goal.goal_id or produced.producer_id != gen.producer_id
                or produced.origin_kind != gen.origin_kind
                or produced.charged_microunits > gen.max_cost_microunits
                or (produced.external_side_effect and produced.origin_kind != "tool")):
            raise ValueError("forged source/cost/effect")
        total += produced.charged_microunits
        fingerprint = content_fingerprint(produced.task, produced.answer)
        if produced.candidate_id in seen_ids or fingerprint in seen_content:
            rejected.append((produced.identity(), "DUPLICATE"))
            continue
        seen_ids.add(produced.candidate_id)
        seen_content.add(fingerprint)
        if fingerprint in goal.holdout_fingerprints:
            rejected.append((produced.identity(), "HOLDOUT_CONTAMINATION"))
            continue
        correct = verifier.verify(goal, produced)
        if type(correct) is not bool:
            raise ValueError("untyped verifier outcome")
        approved = correct and not produced.external_side_effect
        payload = {
            "candidate_sha256": produced.identity(),
            "goal_sha256": goal_sha,
            "verifier_id": verifier.verifier_id,
            "verifier_version_sha256": verifier.verifier_version_sha256,
            "accepted": approved,
        }
        receipt = FactoryReceipt(**payload, signature=_seal(payload, verifier_key))
        verify_factory_receipt(
            produced, goal, receipt, verifier_id=verifier.verifier_id,
            verifier_version_sha256=verifier.verifier_version_sha256,
            verifier_key=verifier_key,
        )
        receipts.append(receipt)
        if approved:
            accepted.append(produced)
        else:
            rejected.append((produced.identity(), "UNVERIFIED_OR_EFFECT"))
    manifest = canonical_digest({
        "goal": goal_sha, "accepted": [c.identity() for c in accepted],
        "receipts": [asdict(x) for x in receipts],
        "rejected": rejected, "attempts": attempts, "cost": total,
    })
    return VerifiedDataPool(
        goal_sha, tuple(accepted), tuple(receipts), tuple(rejected),
        attempts, total, manifest,
    )
