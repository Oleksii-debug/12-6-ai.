"""Plan 6 Section 11: independent frozen capability gaps and advisory curriculum.

This is not the Plan-8 executable product capability map and grants no model,
training, evaluation, budget or promotion authority.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .continual_learning import FrozenCapabilitySuite, score_suite
from .experience_replay import _mac
from .post_base_instruction import bounded_id, canonical_digest, sha_field
from .post_base_reasoning import TinyPolicy


@dataclass(frozen=True, slots=True)
class CapabilityArea:
    capability_id: str
    benchmark: FrozenCapabilitySuite
    generalization: FrozenCapabilitySuite
    target_accuracy: float = 0.8

    def identity(self) -> str:
        bounded_id(self.capability_id)
        if (type(self.benchmark) is not FrozenCapabilitySuite
                or type(self.generalization) is not FrozenCapabilitySuite):
            raise ValueError("untyped capability suite")
        if (self.benchmark.role != "holdout"
                or self.generalization.role != "holdout"):
            raise ValueError("capability evaluation must use frozen holdout suites")
        benchmark_sha = self.benchmark.identity()
        generalization_sha = self.generalization.identity()
        if (benchmark_sha == generalization_sha
                or self.benchmark.suite_id == self.generalization.suite_id
                or {p.probe_id for p in self.benchmark.probes}.intersection(
                    p.probe_id for p in self.generalization.probes)):
            raise ValueError("benchmark and independent generalization must be disjoint")
        if (type(self.target_accuracy) not in (int, float)
                or not math.isfinite(self.target_accuracy)
                or not 0 < self.target_accuracy <= 1):
            raise ValueError("invalid capability target")
        return canonical_digest({"id": self.capability_id, "benchmark": benchmark_sha,
                                 "generalization": generalization_sha,
                                 "target": self.target_accuracy})


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    capability_id: str
    area_sha256: str
    benchmark_accuracy: float
    generalization_accuracy: float
    uncertainty: float
    conservative_gap: float


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    champion_sha256: str
    suite_roots_sha256: str
    areas_sha256: str
    gaps: tuple[CapabilityGap, ...]
    manifest_sha256: str


def measure_capability_gaps(
    champion: TinyPolicy, areas: tuple[CapabilityArea, ...], *,
    trusted_suite_roots: frozenset[str],
) -> CapabilitySnapshot:
    champion_root = champion.identity()
    if type(areas) is not tuple or not 1 <= len(areas) <= 16:
        raise ValueError("invalid capability map size")
    if type(trusted_suite_roots) is not frozenset or not trusted_suite_roots:
        raise ValueError("independent suite roots required")
    for root in trusted_suite_roots:
        sha_field(root)
    seen_ids: set[str] = set()
    seen_probes: set[str] = set()
    seen_suites: set[str] = set()
    gaps: list[CapabilityGap] = []
    for area in sorted(areas, key=lambda a: a.capability_id):
        if type(area) is not CapabilityArea or area.capability_id in seen_ids:
            raise ValueError("invalid/duplicate capability area")
        area_root = area.identity()
        suite_roots = (area.benchmark.identity(), area.generalization.identity())
        if (not set(suite_roots).issubset(trusted_suite_roots)
                or seen_suites.intersection(suite_roots)):
            raise ValueError("untrusted or reused independent capability suite")
        probes = {p.probe_id for suite in (area.benchmark, area.generalization)
                  for p in suite.probes}
        if seen_probes.intersection(probes):
            raise ValueError("cross-area holdout contamination")
        seen_probes.update(probes)
        seen_suites.update(suite_roots)
        seen_ids.add(area.capability_id)
        known = score_suite(champion, area.benchmark)
        generalization = score_suite(champion, area.generalization)
        count = min(len(area.benchmark.probes), len(area.generalization.probes))
        uncertainty = min(1.0, math.sqrt(math.log(40) / (2 * count)))
        # Never treat a perfect familiar benchmark as sufficient holdout evidence.
        gap = max(0.0, area.target_accuracy - min(known, generalization)
                  + uncertainty * 0.25)
        gaps.append(CapabilityGap(area.capability_id, area_root, known,
                                  generalization, uncertainty, gap))
    areas_root = canonical_digest([area.identity() for area in sorted(
        areas, key=lambda x: x.capability_id)])
    suites_root = canonical_digest(sorted(trusted_suite_roots))
    receipt = canonical_digest({"champion": champion_root, "suites": suites_root,
                                "areas": areas_root, "gaps": [asdict(g) for g in gaps]})
    return CapabilitySnapshot(champion_root, suites_root, areas_root, tuple(gaps), receipt)


@dataclass(frozen=True, slots=True)
class CurriculumProposal:
    proposal_id: str
    capability_id: str
    kind: str
    producer_id: str
    verifier_id: str
    verifier_version_sha256: str
    evidence_sha256: str
    signature: str
    train_task_ids: tuple[str, ...]
    expected_gain: int
    evidence_quality: int
    cost_units: int
    split: str = "train"
    schema_version: int = 1

    def payload(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "signature"}

    def identity(self) -> str:
        if (type(self.schema_version) is not int or self.schema_version != 1
                or self.kind not in ("data", "experiment", "training")
                or self.split != "train"):
            raise ValueError("invalid curriculum proposal")
        for item in (self.proposal_id, self.capability_id, self.producer_id, self.verifier_id):
            bounded_id(item)
        if self.producer_id == self.verifier_id:
            raise ValueError("self-assessed curriculum target")
        for root in (self.verifier_version_sha256, self.evidence_sha256, self.signature):
            sha_field(root)
        if (type(self.train_task_ids) is not tuple or not 1 <= len(self.train_task_ids) <= 64
                or len(set(self.train_task_ids)) != len(self.train_task_ids)):
            raise ValueError("invalid curriculum training tasks")
        for task_id in self.train_task_ids:
            bounded_id(task_id)
        for value, minimum, maximum in ((self.expected_gain, 1, 100),
                                        (self.evidence_quality, 1, 100),
                                        (self.cost_units, 1, 10000)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("unbounded curriculum gain/evidence/cost")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class CurriculumRecipe:
    snapshot_sha256: str
    max_targets: int = 4
    budget_units: int = 100
    schema_version: int = 1

    def validate(self) -> None:
        sha_field(self.snapshot_sha256)
        if (type(self.schema_version) is not int or self.schema_version != 1
                or type(self.max_targets) is not int or not 1 <= self.max_targets <= 8
                or type(self.budget_units) is not int or not 1 <= self.budget_units <= 10000):
            raise ValueError("unbounded curriculum recipe")


@dataclass(frozen=True, slots=True)
class CurriculumPlan:
    snapshot_sha256: str
    recipe_sha256: str
    target_ids: tuple[str, ...]
    total_cost_units: int
    manifest_sha256: str
    state: str = "ADVISORY_CANDIDATES_ONLY"
    authorized_paid_compute: bool = False
    authorized_training: bool = False
    promotion_authorized: bool = False


def plan_curriculum(
    snapshot: CapabilitySnapshot, areas: tuple[CapabilityArea, ...],
    proposals: tuple[CurriculumProposal, ...], recipe: CurriculumRecipe, *,
    trusted_verifier_id: str, trusted_verifier_version_sha256: str,
    trusted_proposal_roots: frozenset[str], verifier_key: bytes,
) -> CurriculumPlan:
    recipe.validate()
    if (type(snapshot) is not CapabilitySnapshot
            or snapshot.manifest_sha256 != recipe.snapshot_sha256):
        raise ValueError("stale capability snapshot")
    if (type(areas) is not tuple or canonical_digest([a.identity() for a in sorted(
            areas, key=lambda a: a.capability_id)]) != snapshot.areas_sha256):
        raise ValueError("capability map changed after frozen measurement")
    if type(proposals) is not tuple or not 1 <= len(proposals) <= 64:
        raise ValueError("bounded curriculum proposals required")
    bounded_id(trusted_verifier_id)
    sha_field(trusted_verifier_version_sha256)
    if type(trusted_proposal_roots) is not frozenset or not trusted_proposal_roots:
        raise ValueError("independent curriculum evidence roots required")
    for root in trusted_proposal_roots:
        sha_field(root)
    all_probes = {p.probe_id for area in areas
                  for suite in (area.benchmark, area.generalization)
                  for p in suite.probes}
    gaps = {g.capability_id: g for g in snapshot.gaps}
    if len(gaps) != len(areas) or set(gaps) != {area.capability_id for area in areas}:
        raise ValueError("forged capability map")
    seen: set[str] = set()
    candidates: list[tuple[float, CurriculumProposal]] = []
    for proposal in proposals:
        if type(proposal) is not CurriculumProposal:
            raise ValueError("untyped curriculum proposal")
        proposal.identity()
        if (proposal.proposal_id in seen or proposal.capability_id not in gaps
                or proposal.verifier_id != trusted_verifier_id
                or proposal.verifier_version_sha256 != trusted_verifier_version_sha256
                or proposal.evidence_sha256 not in trusted_proposal_roots
                or all_probes.intersection(proposal.train_task_ids)):
            raise ValueError("untrusted, duplicate or holdout-leaking proposal")
        if proposal.signature != _mac(verifier_key, proposal.payload()):
            raise ValueError("forged proposal evidence")
        seen.add(proposal.proposal_id)
        gap = gaps[proposal.capability_id]
        priority = (gap.conservative_gap * proposal.expected_gain * proposal.evidence_quality
                    * (1 + gap.uncertainty) / proposal.cost_units)
        if priority > 0:
            candidates.append((priority, proposal))
    chosen: list[str] = []
    spent = 0
    for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].proposal_id)):
        if len(chosen) >= recipe.max_targets:
            break
        if spent + item.cost_units > recipe.budget_units:
            continue
        chosen.append(item.proposal_id)
        spent += item.cost_units
    state = "ADVISORY_CANDIDATES_ONLY" if chosen else "STOP_NO_EVIDENCED_VALUE"
    recipe_root = canonical_digest(asdict(recipe))
    evidence_root = canonical_digest({"snapshot": recipe.snapshot_sha256,
                                      "recipe": recipe_root, "selected": chosen,
                                      "cost": spent, "state": state})
    return CurriculumPlan(recipe.snapshot_sha256, recipe_root, tuple(chosen), spent,
                          evidence_root, state)


def verify_curriculum_restart(
    snapshot: CapabilitySnapshot, areas: tuple[CapabilityArea, ...],
    proposals: tuple[CurriculumProposal, ...], recipe: CurriculumRecipe,
    result: CurriculumPlan, *, trusted_verifier_id: str,
    trusted_verifier_version_sha256: str, trusted_proposal_roots: frozenset[str],
    verifier_key: bytes,
) -> bool:
    if (type(result) is not CurriculumPlan or result.promotion_authorized
            or result.authorized_paid_compute or result.authorized_training):
        raise ValueError("unauthorized curriculum promotion")
    actual = plan_curriculum(snapshot, areas, proposals, recipe,
                             trusted_verifier_id=trusted_verifier_id,
                             trusted_verifier_version_sha256=trusted_verifier_version_sha256,
                             trusted_proposal_roots=trusted_proposal_roots,
                             verifier_key=verifier_key)
    if actual != result:
        raise ValueError("curriculum restart mismatch")
    return True
