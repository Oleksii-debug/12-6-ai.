"""Deterministic validation-split robustness contracts.

DATA-36 owns only split-family construction, leakage auditing, and exact identity
binding. It deliberately consumes an already training-eligible corpus; source
rights, normalization, quality filtering, dedup execution, tokenizer selection,
model architecture, optimizer semantics, and benchmark registries remain owned by
their incumbent lanes.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SPLIT_VARIANT_SCHEMA = "12-6.validation-split.v1"
SPLIT_FAMILY_SCHEMA = "12-6.validation-split-family.v1"
SPLIT_EVIDENCE_SCHEMA = "12-6.split-robustness-evidence.v1"
_ALLOWED_PURPOSES = frozenset({"pretraining", "pretraining_eligible", "training_eligible"})
_FORBIDDEN_PURPOSES = frozenset(
    {"benchmark", "evaluation", "evaluation_test", "heldout_test", "test", "probe_test"}
)
_HEX = frozenset("0123456789abcdef")


class SplitRobustnessError(ValueError):
    """Raised when split construction or identity binding would be unsafe."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "UNRESOLVED":
        raise SplitRobustnessError(f"{field} must be resolved non-empty text")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if len(text) != 64 or text != text.lower() or any(ch not in _HEX for ch in text):
        raise SplitRobustnessError(f"{field} must be exact lowercase SHA-256 hex")
    return text


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Minimal post-policy record needed to construct leakage-safe splits."""

    id: str
    text: str
    source_id: str
    modality: str
    content_sha256: str
    near_duplicate_cluster_id: str
    training_eligible: bool = True
    purpose: str = "pretraining_eligible"

    def __post_init__(self) -> None:
        for field in ("id", "source_id", "modality", "near_duplicate_cluster_id"):
            _require_text(getattr(self, field), field)
        if not isinstance(self.text, str) or not self.text:
            raise SplitRobustnessError("text must be a non-empty string")
        _require_sha256(self.content_sha256, "content_sha256")
        actual = _sha256_bytes(self.text.encode("utf-8"))
        if actual != self.content_sha256:
            raise SplitRobustnessError(f"{self.id}: content_sha256 does not match text")
        if self.training_eligible is not True:
            raise SplitRobustnessError(f"{self.id}: record is not training eligible")
        purpose = self.purpose.strip().lower()
        if purpose in _FORBIDDEN_PURPOSES or purpose not in _ALLOWED_PURPOSES:
            raise SplitRobustnessError(f"{self.id}: forbidden/non-training purpose {self.purpose!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitRecord":
        return cls(**dict(value))

    def identity_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "modality": self.modality,
            "content_sha256": self.content_sha256,
            "near_duplicate_cluster_id": self.near_duplicate_cluster_id,
            "training_eligible": self.training_eligible,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class SplitFamilySpec:
    """Predeclared alternative validation partitions for one eligible corpus."""

    eligible_corpus_sha256: str
    dedup_relations_sha256: str
    variant_seeds: tuple[str, ...]
    validation_fraction: float = 0.10
    algorithm: str = "cluster-hash-ranked-greedy-v1"

    def __post_init__(self) -> None:
        _require_sha256(self.eligible_corpus_sha256, "eligible_corpus_sha256")
        _require_sha256(self.dedup_relations_sha256, "dedup_relations_sha256")
        if len(self.variant_seeds) < 2:
            raise SplitRobustnessError("split robustness requires at least two variants")
        normalized = tuple(_require_text(seed, "variant_seed") for seed in self.variant_seeds)
        if len(set(normalized)) != len(normalized):
            raise SplitRobustnessError("variant seeds must be unique")
        if not 0.0 < self.validation_fraction < 0.5:
            raise SplitRobustnessError("validation_fraction must be in (0, 0.5)")
        if self.algorithm != "cluster-hash-ranked-greedy-v1":
            raise SplitRobustnessError("unsupported split algorithm")


def eligible_corpus_identity(records: Sequence[SplitRecord]) -> str:
    """Hash exact eligible record identities without serializing raw text into manifests."""

    ordered = sorted((record.identity_mapping() for record in records), key=lambda item: item["id"])
    return _sha256_bytes(_canonical_json_bytes({"records": ordered}))


def dedup_relations_identity(records: Sequence[SplitRecord]) -> str:
    """Hash the explicit record -> near-duplicate-cluster relation."""

    relation = [
        {"id": record.id, "near_duplicate_cluster_id": record.near_duplicate_cluster_id}
        for record in sorted(records, key=lambda item: item.id)
    ]
    return _sha256_bytes(_canonical_json_bytes({"near_duplicate_relations": relation}))


def _validate_record_set(records: Sequence[SplitRecord]) -> None:
    if len(records) < 4:
        raise SplitRobustnessError("at least four eligible records are required")
    ids = [record.id for record in records]
    if len(set(ids)) != len(ids):
        raise SplitRobustnessError("record ids must be unique")
    content_to_clusters: dict[str, set[str]] = defaultdict(set)
    for record in records:
        content_to_clusters[record.content_sha256].add(record.near_duplicate_cluster_id)
    bad = [digest for digest, clusters in content_to_clusters.items() if len(clusters) > 1]
    if bad:
        raise SplitRobustnessError(
            "exact-content aliases cannot be assigned to different near-duplicate clusters"
        )


def _cluster_groups(records: Sequence[SplitRecord]) -> dict[str, tuple[SplitRecord, ...]]:
    groups: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        groups[record.near_duplicate_cluster_id].append(record)
    return {key: tuple(sorted(value, key=lambda item: item.id)) for key, value in groups.items()}


def _choose_validation_clusters(
    records: Sequence[SplitRecord], *, seed: str, validation_fraction: float
) -> tuple[str, ...]:
    groups = _cluster_groups(records)
    if len(groups) < 2:
        raise SplitRobustnessError("at least two dedup clusters are required")
    target_documents = max(1, round(len(records) * validation_fraction))
    ranked = sorted(
        groups,
        key=lambda cluster_id: hashlib.sha256(
            f"{seed}\0{cluster_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[str] = []
    selected_documents = 0
    for cluster_id in ranked:
        size = len(groups[cluster_id])
        before_delta = abs(target_documents - selected_documents)
        after_delta = abs(target_documents - (selected_documents + size))
        if not selected or after_delta <= before_delta:
            selected.append(cluster_id)
            selected_documents += size
        if selected_documents >= target_documents:
            break
    if not selected:
        selected = [ranked[0]]
    if len(selected) == len(groups):
        selected = selected[:-1]
    return tuple(selected)


def audit_cluster_leakage(
    records: Sequence[SplitRecord], assignments: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return every near-duplicate cluster that straddles train and validation."""

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    cluster_sources: dict[str, set[str]] = defaultdict(set)
    cluster_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        split = assignments.get(record.id)
        if split not in {"train", "validation"}:
            raise SplitRobustnessError(f"missing/invalid split assignment for {record.id}")
        cluster = record.near_duplicate_cluster_id
        cluster_splits[cluster].add(split)
        cluster_sources[cluster].add(record.source_id)
        cluster_ids[cluster].append(record.id)
    leakage = []
    for cluster in sorted(cluster_splits):
        if len(cluster_splits[cluster]) > 1:
            leakage.append(
                {
                    "near_duplicate_cluster_id": cluster,
                    "record_ids": sorted(cluster_ids[cluster]),
                    "source_ids": sorted(cluster_sources[cluster]),
                    "splits": sorted(cluster_splits[cluster]),
                }
            )
    return leakage


def legacy_record_hash_assignments(
    records: Sequence[SplitRecord], *, seed: str, validation_fraction: float
) -> dict[str, str]:
    """Model current record-id hash semantics for a leakage-risk audit only."""

    threshold = round(validation_fraction * 10_000)
    output: dict[str, str] = {}
    for record in records:
        digest = hashlib.sha256(f"{seed}\0{record.id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10_000
        output[record.id] = "validation" if bucket < threshold else "train"
    if set(output.values()) != {"train", "validation"}:
        raise SplitRobustnessError("record-hash audit produced an empty split")
    return output


def _variant_manifest(
    records: Sequence[SplitRecord],
    spec: SplitFamilySpec,
    *,
    variant_index: int,
    seed: str,
) -> dict[str, Any]:
    selected_clusters = set(
        _choose_validation_clusters(
            records, seed=seed, validation_fraction=spec.validation_fraction
        )
    )
    assignments = {
        record.id: (
            "validation"
            if record.near_duplicate_cluster_id in selected_clusters
            else "train"
        )
        for record in records
    }
    leakage = audit_cluster_leakage(records, assignments)
    if leakage:
        raise SplitRobustnessError("cluster-aware split unexpectedly leaked a dedup cluster")
    train_ids = sorted(record_id for record_id, split in assignments.items() if split == "train")
    validation_ids = sorted(
        record_id for record_id, split in assignments.items() if split == "validation"
    )
    if not train_ids or not validation_ids:
        raise SplitRobustnessError("split variant contains an empty partition")
    core = {
        "schema_version": SPLIT_VARIANT_SCHEMA,
        "variant_id": f"v{variant_index:02d}",
        "seed": seed,
        "algorithm": spec.algorithm,
        "eligible_corpus_sha256": spec.eligible_corpus_sha256,
        "dedup_relations_sha256": spec.dedup_relations_sha256,
        "validation_fraction_requested": spec.validation_fraction,
        "validation_clusters": sorted(selected_clusters),
        "train_record_ids": train_ids,
        "validation_record_ids": validation_ids,
        "train_documents": len(train_ids),
        "validation_documents": len(validation_ids),
        "cluster_straddles": [],
    }
    return {**core, "split_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def build_split_family(
    records: Sequence[SplitRecord], spec: SplitFamilySpec
) -> dict[str, Any]:
    """Build multiple cluster-safe validation partitions and one shared train core.

    The shared train core excludes the union of every alternative validation set.
    This permits one trained model to be evaluated across every partition without
    any alternative-validation record ever contributing an optimized token.
    """

    _validate_record_set(records)
    actual_corpus = eligible_corpus_identity(records)
    actual_dedup = dedup_relations_identity(records)
    if actual_corpus != spec.eligible_corpus_sha256:
        raise SplitRobustnessError("eligible corpus identity does not match split spec")
    if actual_dedup != spec.dedup_relations_sha256:
        raise SplitRobustnessError("dedup relation identity does not match split spec")

    variants = [
        _variant_manifest(records, spec, variant_index=index, seed=seed)
        for index, seed in enumerate(spec.variant_seeds, start=1)
    ]
    if len({item["split_identity_sha256"] for item in variants}) != len(variants):
        raise SplitRobustnessError("alternative split variants are not identity-distinct")

    validation_union = sorted(
        {
            record_id
            for variant in variants
            for record_id in variant["validation_record_ids"]
        }
    )
    all_ids = {record.id for record in records}
    shared_train_ids = sorted(all_ids - set(validation_union))
    if not shared_train_ids:
        raise SplitRobustnessError("validation union consumed the entire eligible corpus")
    if set(shared_train_ids) & set(validation_union):
        raise SplitRobustnessError("shared train core overlaps validation union")

    legacy_audit = []
    for seed in spec.variant_seeds:
        legacy = legacy_record_hash_assignments(
            records, seed=seed, validation_fraction=spec.validation_fraction
        )
        leakage = audit_cluster_leakage(records, legacy)
        legacy_audit.append(
            {
                "seed": seed,
                "near_duplicate_cluster_straddles": len(leakage),
                "straddled_cluster_ids": [
                    item["near_duplicate_cluster_id"] for item in leakage
                ],
            }
        )

    core = {
        "schema_version": SPLIT_FAMILY_SCHEMA,
        "eligible_corpus_sha256": actual_corpus,
        "dedup_relations_sha256": actual_dedup,
        "algorithm": spec.algorithm,
        "validation_fraction_requested": spec.validation_fraction,
        "variant_split_identities": [item["split_identity_sha256"] for item in variants],
        "variants": variants,
        "validation_union_record_ids": validation_union,
        "shared_train_record_ids": shared_train_ids,
        "shared_train_documents": len(shared_train_ids),
        "validation_union_documents": len(validation_union),
        "cluster_straddles_across_variants": 0,
        "legacy_record_hash_risk_audit": legacy_audit,
        "training_policy": "optimize_shared_train_core_only_excluding_validation_union",
    }
    return {**core, "split_family_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def verify_split_family_manifest(
    records: Sequence[SplitRecord], manifest: Mapping[str, Any]
) -> None:
    """Fail closed on corpus, dedup, assignment, or identity drift."""

    if manifest.get("schema_version") != SPLIT_FAMILY_SCHEMA:
        raise SplitRobustnessError("unsupported split-family schema")
    expected_corpus = eligible_corpus_identity(records)
    expected_dedup = dedup_relations_identity(records)
    if manifest.get("eligible_corpus_sha256") != expected_corpus:
        raise SplitRobustnessError("split family belongs to another eligible corpus")
    if manifest.get("dedup_relations_sha256") != expected_dedup:
        raise SplitRobustnessError("split family belongs to another dedup relation set")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise SplitRobustnessError("split family must contain at least two variants")
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise SplitRobustnessError("split variant must be an object")
        core = dict(variant)
        claimed = core.pop("split_identity_sha256", None)
        if claimed != _sha256_bytes(_canonical_json_bytes(core)):
            raise SplitRobustnessError("split variant identity/content mismatch")
        assignments = {
            record_id: "train" for record_id in variant.get("train_record_ids", [])
        }
        assignments.update(
            {record_id: "validation" for record_id in variant.get("validation_record_ids", [])}
        )
        if set(assignments) != {record.id for record in records}:
            raise SplitRobustnessError("split variant does not assign the exact eligible corpus")
        leakage = audit_cluster_leakage(records, assignments)
        if leakage:
            raise SplitRobustnessError("near-duplicate cluster straddles train/validation")
    core = dict(manifest)
    claimed_family = core.pop("split_family_identity_sha256", None)
    if claimed_family != _sha256_bytes(_canonical_json_bytes(core)):
        raise SplitRobustnessError("split-family identity/content mismatch")
    shared = set(manifest.get("shared_train_record_ids", []))
    validation_union = set(manifest.get("validation_union_record_ids", []))
    if not shared or shared & validation_union:
        raise SplitRobustnessError("shared train core is empty or contaminated")
    if shared | validation_union != {record.id for record in records}:
        raise SplitRobustnessError("shared-train/validation-union coverage mismatch")


def assert_run_split_binding(
    run_manifest: Mapping[str, Any], split_family_manifest: Mapping[str, Any]
) -> str:
    """Require a launch manifest to name the exact split-family SHA-256."""

    family = _require_sha256(
        split_family_manifest.get("split_family_identity_sha256"),
        "split_family_identity_sha256",
    )
    data = run_manifest.get("data")
    if not isinstance(data, Mapping):
        raise SplitRobustnessError("run manifest data must be a mapping")
    observed = _require_sha256(data.get("split_identity"), "data.split_identity")
    if observed != family:
        raise SplitRobustnessError("run manifest is bound to another split family")
    observed_corpus = _require_sha256(
        data.get("eligible_corpus_sha256"), "data.eligible_corpus_sha256"
    )
    if observed_corpus != split_family_manifest.get("eligible_corpus_sha256"):
        raise SplitRobustnessError("run manifest eligible-corpus identity drift")
    return family


def assert_checkpoint_split_binding(
    checkpoint_identity: Any, split_family_manifest: Mapping[str, Any]
) -> str:
    """Verify D05 checkpoint training_config preserves the exact split-family SHA."""

    family = _require_sha256(
        split_family_manifest.get("split_family_identity_sha256"),
        "split_family_identity_sha256",
    )
    if isinstance(checkpoint_identity, Mapping):
        training_config = checkpoint_identity.get("training_config")
    else:
        training_config = getattr(checkpoint_identity, "training_config", None)
    if not isinstance(training_config, Mapping):
        raise SplitRobustnessError("checkpoint identity lacks training_config mapping")
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise SplitRobustnessError("checkpoint training_config lacks data mapping")
    observed = _require_sha256(data.get("split_identity"), "checkpoint split_identity")
    if observed != family:
        raise SplitRobustnessError("checkpoint split identity does not match split family")
    return family


def split_sensitivity(values: Sequence[float]) -> dict[str, float]:
    """Summarize split sensitivity without selecting a favorable partition."""

    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise SplitRobustnessError("at least two finite split metrics are required")
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "mean": mean,
        "population_stdev": stdev,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "relative_range": (maximum - minimum) / mean if mean else 0.0,
        "max_abs_deviation_from_mean": max(abs(value - mean) for value in values),
    }


def pairwise_ranking_stability(
    candidate_metrics: Mapping[str, Sequence[float]], *, lower_is_better: bool = True
) -> dict[str, Any]:
    """Measure pairwise rank reversals across variants for two or more candidates."""

    names = sorted(candidate_metrics)
    if len(names) < 2:
        raise SplitRobustnessError("ranking stability requires at least two candidates")
    lengths = {len(candidate_metrics[name]) for name in names}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise SplitRobustnessError("candidate metric arrays must have equal length >= 2")
    variant_count = next(iter(lengths))
    pairs = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            preferences: list[str] = []
            for index in range(variant_count):
                left_value = candidate_metrics[left][index]
                right_value = candidate_metrics[right][index]
                if left_value == right_value:
                    preferences.append("tie")
                elif (left_value < right_value) == lower_is_better:
                    preferences.append(left)
                else:
                    preferences.append(right)
            non_ties = [value for value in preferences if value != "tie"]
            winner = None
            if non_ties:
                winner = max(sorted(set(non_ties)), key=non_ties.count)
            reversals = sum(value not in {winner, "tie"} for value in preferences) if winner else 0
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "variant_preferences": preferences,
                    "majority_winner": winner,
                    "rank_reversal_count": reversals,
                    "rank_consistency_fraction": (
                        (len(non_ties) - reversals) / len(non_ties) if non_ties else 1.0
                    ),
                }
            )
    return {
        "variants": variant_count,
        "pairs": pairs,
        "all_pairs_stable": all(item["rank_reversal_count"] == 0 for item in pairs),
    }


def bind_split_evidence(
    payload: Mapping[str, Any], split_family_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Content-address split-sensitivity evidence to one immutable split family."""

    family = _require_sha256(
        split_family_manifest.get("split_family_identity_sha256"),
        "split_family_identity_sha256",
    )
    core = {
        "schema_version": SPLIT_EVIDENCE_SCHEMA,
        "split_family_identity_sha256": family,
        "eligible_corpus_sha256": _require_sha256(
            split_family_manifest.get("eligible_corpus_sha256"),
            "eligible_corpus_sha256",
        ),
        **dict(payload),
    }
    if "evidence_sha256" in core:
        raise SplitRobustnessError("unbound payload must not predeclare evidence_sha256")
    return {**core, "evidence_sha256": _sha256_bytes(_canonical_json_bytes(core))}
