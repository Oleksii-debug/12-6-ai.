"""Plan 6 Section 10: independent verified experience replay, candidate only.

Uses the existing Plan-6 source admission and digest contracts. It never
modifies a champion, trains a model, or authorizes dataset/memory/skill promotion.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass

from .post_base_instruction import bounded_id, canonical_digest, sha_field, validate_source


def _mac(key: bytes, payload: object) -> str:
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("independent verifier key required")
    return hmac.new(key, canonical_digest(payload).encode("ascii"), hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedExperience:
    experience_id: str
    task_id: str
    context_sha256: str
    outcome_sha256: str
    evidence_sha256: str
    provenance_sha256: str
    rights_evidence_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    producer_id: str
    signature: str
    value: int
    forgetting: int
    curriculum: int
    quality: float = 1.0
    rights: str = "owned"
    split: str = "train"
    schema_version: int = 1

    def payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported experience schema")
        for item in (self.experience_id, self.task_id, self.verifier_id, self.producer_id):
            bounded_id(item)
        if self.producer_id == self.verifier_id:
            raise ValueError("self-issued experience verification")
        for item in (self.context_sha256, self.outcome_sha256, self.evidence_sha256,
                     self.verifier_version_sha256, self.signature):
            sha_field(item)
        validate_source(
            sample_id=self.experience_id, provenance_sha256=self.provenance_sha256,
            rights=self.rights, rights_evidence_sha256=self.rights_evidence_sha256,
            split=self.split, quality=self.quality,
        )
        for score in (self.value, self.forgetting, self.curriculum):
            if type(score) is not int or not 0 <= score <= 100:
                raise ValueError("unbounded replay priority")
        return canonical_digest(asdict(self))


def verify_experience(
    experience: VerifiedExperience, *, trusted_verifier_id: str,
    trusted_verifier_version_sha256: str, verifier_key: bytes,
    trusted_evidence_roots: frozenset[str],
) -> str:
    if type(experience) is not VerifiedExperience:
        raise ValueError("untyped experience")
    identity = experience.identity()
    bounded_id(trusted_verifier_id)
    sha_field(trusted_verifier_version_sha256)
    if (experience.verifier_id != trusted_verifier_id
            or experience.verifier_version_sha256 != trusted_verifier_version_sha256
            or type(trusted_evidence_roots) is not frozenset
            or not trusted_evidence_roots):
        raise ValueError("untrusted experience verifier")
    for root in trusted_evidence_roots:
        sha_field(root)
    if experience.evidence_sha256 not in trusted_evidence_roots:
        raise ValueError("evidence root not independently admitted")
    if not hmac.compare_digest(experience.signature, _mac(verifier_key, experience.payload())):
        raise ValueError("forged experience evidence")
    return identity


@dataclass(frozen=True, slots=True)
class ReplayRecipe:
    curriculum_sha256: str
    source_pool_sha256: str
    max_selected: int = 4
    min_priority: int = 1
    novelty_weight: int = 4
    value_weight: int = 2
    forgetting_weight: int = 3
    curriculum_weight: int = 2
    schema_version: int = 1

    def validate(self) -> None:
        sha_field(self.curriculum_sha256)
        sha_field(self.source_pool_sha256)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported replay recipe")
        for value, lower, upper in (
            (self.max_selected, 1, 64), (self.min_priority, 0, 1100),
            (self.novelty_weight, 0, 10), (self.value_weight, 0, 10),
            (self.forgetting_weight, 0, 10), (self.curriculum_weight, 0, 10),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid replay budget or priority")
        if not any((self.novelty_weight, self.value_weight,
                    self.forgetting_weight, self.curriculum_weight)):
            raise ValueError("missing replay strategy")


@dataclass(frozen=True, slots=True)
class ConsolidationCandidate:
    destination: str
    selected_sha256: str
    curriculum_sha256: str
    manifest_sha256: str
    promotion_authorized: bool = False
    canonical_base_eligible: bool = False


@dataclass(frozen=True, slots=True)
class ReplayResult:
    pool_sha256: str
    recipe_sha256: str
    selected_ids: tuple[str, ...]
    selected_sha256: str
    candidates: tuple[ConsolidationCandidate, ...]
    evidence_sha256: str
    state: str = "CANDIDATES_ONLY"
    promotion_authorized: bool = False


def build_replay_candidates(
    experiences: tuple[VerifiedExperience, ...], recipe: ReplayRecipe, *,
    trusted_verifier_id: str, trusted_verifier_version_sha256: str,
    verifier_key: bytes, trusted_evidence_roots: frozenset[str],
) -> ReplayResult:
    """Verified pool -> novelty/value/forgetting/curriculum replay -> 3 isolated candidates."""
    recipe.validate()
    if type(experiences) is not tuple or not 1 <= len(experiences) <= 64:
        raise ValueError("unbounded verified replay pool")
    ids: set[str] = set()
    evidence: set[str] = set()
    fingerprints: set[tuple[str, str]] = set()
    for record in experiences:
        verify_experience(
            record, trusted_verifier_id=trusted_verifier_id,
            trusted_verifier_version_sha256=trusted_verifier_version_sha256,
            verifier_key=verifier_key, trusted_evidence_roots=trusted_evidence_roots,
        )
        fingerprint = (record.context_sha256, record.outcome_sha256)
        if (record.experience_id in ids or record.evidence_sha256 in evidence
                or fingerprint in fingerprints):
            raise ValueError("duplicate or replayed verified experience")
        ids.add(record.experience_id)
        evidence.add(record.evidence_sha256)
        fingerprints.add(fingerprint)
    pool_sha = canonical_digest([record.identity() for record in sorted(
        experiences, key=lambda r: r.experience_id,
    )])
    if pool_sha != recipe.source_pool_sha256:
        raise ValueError("replay pool changed after frozen recipe")
    selected: list[VerifiedExperience] = []
    remaining = list(experiences)
    contexts: set[str] = set()
    while remaining and len(selected) < recipe.max_selected:
        def priority(record: VerifiedExperience) -> int:
            novelty = 100 if record.context_sha256 not in contexts else 0
            return (recipe.novelty_weight * novelty + recipe.value_weight * record.value
                    + recipe.forgetting_weight * record.forgetting
                    + recipe.curriculum_weight * record.curriculum)

        current = min(remaining, key=lambda r: (-priority(r), r.experience_id))
        if priority(current) < recipe.min_priority:
            break
        selected.append(current)
        contexts.add(current.context_sha256)
        remaining.remove(current)
    if not selected:
        raise ValueError("no verified experience qualified for replay")
    selected_root = canonical_digest([item.identity() for item in selected])
    recipe_root = canonical_digest(asdict(recipe))
    candidates = tuple(
        ConsolidationCandidate(
            destination=destination, selected_sha256=selected_root,
            curriculum_sha256=recipe.curriculum_sha256,
            manifest_sha256=canonical_digest({
                "destination": destination, "pool": pool_sha, "selection": selected_root,
                "recipe": recipe_root, "curriculum": recipe.curriculum_sha256,
            }),
        ) for destination in ("dataset", "memory", "skills")
    )
    receipt_sha = canonical_digest({
        "pool": pool_sha, "recipe": recipe_root, "selected": selected_root,
        "candidates": [asdict(candidate) for candidate in candidates],
    })
    return ReplayResult(pool_sha, recipe_root, tuple(item.experience_id for item in selected),
                        selected_root, candidates, receipt_sha)


def verify_replay_restart(
    experiences: tuple[VerifiedExperience, ...], recipe: ReplayRecipe,
    result: ReplayResult, *, trusted_verifier_id: str,
    trusted_verifier_version_sha256: str, verifier_key: bytes,
    trusted_evidence_roots: frozenset[str],
) -> bool:
    if (type(result) is not ReplayResult or result.promotion_authorized
            or any(candidate.promotion_authorized or candidate.canonical_base_eligible
                   for candidate in result.candidates)):
        raise ValueError("untrusted replay promotion claim")
    actual = build_replay_candidates(
        experiences, recipe, trusted_verifier_id=trusted_verifier_id,
        trusted_verifier_version_sha256=trusted_verifier_version_sha256,
        verifier_key=verifier_key, trusted_evidence_roots=trusted_evidence_roots,
    )
    if actual != result:
        raise ValueError("replay restart/evidence mismatch")
    return True
