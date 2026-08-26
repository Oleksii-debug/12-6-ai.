"""Source-family diversity metrics for admitted real UA/EN pretraining records.

DATA-228 counts publisher/corpus families rather than files or hosts.  The
module is intentionally measurement-only: it never resamples or duplicates
records and it fails closed when family provenance is missing or aliases the
same publisher/underlying corpus under multiple family names.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from twelve_six.data.multilingual_pretraining import AdmittedRecord


class CorpusDiversityError(ValueError):
    """Raised when source-family provenance or diversity math is invalid."""


@dataclass(frozen=True)
class SourceFamilyIdentity:
    source_id: str
    family_id: str
    publisher_identity: str
    underlying_corpus_identity: str

    def __post_init__(self) -> None:
        for field in (
            "source_id",
            "family_id",
            "publisher_identity",
            "underlying_corpus_identity",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise CorpusDiversityError(f"{field} must be non-empty")


@dataclass(frozen=True)
class DiversityPolicy:
    required_languages: tuple[str, ...] = ("uk", "en")
    minimum_families_per_language: int = 2

    def __post_init__(self) -> None:
        if not self.required_languages or len(set(self.required_languages)) != len(
            self.required_languages
        ):
            raise CorpusDiversityError("required_languages must be unique and non-empty")
        if self.minimum_families_per_language < 1:
            raise CorpusDiversityError("minimum_families_per_language must be positive")


def validate_family_identities(
    identities: Iterable[SourceFamilyIdentity],
) -> dict[str, SourceFamilyIdentity]:
    by_source: dict[str, SourceFamilyIdentity] = {}
    by_family: dict[str, SourceFamilyIdentity] = {}
    publisher_to_family: dict[str, str] = {}
    corpus_to_family: dict[str, str] = {}

    for identity in identities:
        if identity.source_id in by_source:
            raise CorpusDiversityError(f"duplicate source identity: {identity.source_id}")
        by_source[identity.source_id] = identity

        existing = by_family.get(identity.family_id)
        if existing is not None:
            if existing.publisher_identity != identity.publisher_identity:
                raise CorpusDiversityError(
                    f"family {identity.family_id!r} spans multiple publishers"
                )
            if existing.underlying_corpus_identity != identity.underlying_corpus_identity:
                raise CorpusDiversityError(
                    f"family {identity.family_id!r} spans multiple underlying corpora"
                )
        else:
            by_family[identity.family_id] = identity

        previous_family = publisher_to_family.get(identity.publisher_identity)
        if previous_family is not None and previous_family != identity.family_id:
            raise CorpusDiversityError(
                "same publisher cannot be counted as independent source families: "
                f"{identity.publisher_identity!r} -> {previous_family!r}, {identity.family_id!r}"
            )
        publisher_to_family[identity.publisher_identity] = identity.family_id

        previous_family = corpus_to_family.get(identity.underlying_corpus_identity)
        if previous_family is not None and previous_family != identity.family_id:
            raise CorpusDiversityError(
                "same underlying corpus cannot be counted as independent source families: "
                f"{identity.underlying_corpus_identity!r} -> {previous_family!r}, {identity.family_id!r}"
            )
        corpus_to_family[identity.underlying_corpus_identity] = identity.family_id

    if not by_source:
        raise CorpusDiversityError("at least one source-family identity is required")
    return by_source


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _document_length_distribution(lengths: list[int]) -> dict[str, object]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "max": 0,
            "mean": 0.0,
            "buckets": {},
        }
    buckets = Counter()
    for length in lengths:
        if length < 128:
            bucket = "lt128"
        elif length < 256:
            bucket = "128_255"
        elif length < 512:
            bucket = "256_511"
        elif length < 1024:
            bucket = "512_1023"
        else:
            bucket = "ge1024"
        buckets[bucket] += 1
    return {
        "count": len(lengths),
        "min": min(lengths),
        "p50": _nearest_rank(lengths, 0.50),
        "p90": _nearest_rank(lengths, 0.90),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "buckets": dict(sorted(buckets.items())),
    }


def _family_metrics(family_mass: Mapping[str, int]) -> dict[str, object]:
    positive = {family: mass for family, mass in family_mass.items() if mass > 0}
    total = sum(positive.values())
    if total <= 0:
        return {
            "family_count": 0,
            "token_mass": 0,
            "top_family_token_share": 0.0,
            "effective_source_count": 0.0,
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "family_token_mass": {},
        }
    probabilities = [mass / total for mass in positive.values()]
    entropy = -sum(p * math.log(p) for p in probabilities)
    family_count = len(positive)
    normalized_entropy = entropy / math.log(family_count) if family_count > 1 else 0.0
    effective = 1.0 / sum(p * p for p in probabilities)
    return {
        "family_count": family_count,
        "token_mass": total,
        "top_family_token_share": max(probabilities),
        "effective_source_count": effective,
        "entropy_nats": entropy,
        "normalized_entropy": normalized_entropy,
        "family_token_mass": dict(sorted(positive.items())),
    }


def measure_diversity(
    records: Iterable[AdmittedRecord],
    *,
    family_identities: Iterable[SourceFamilyIdentity],
    policy: DiversityPolicy = DiversityPolicy(),
) -> dict[str, object]:
    """Measure diversity using exact UTF-8 byte-token mass.

    The project uses a byte tokenizer for the fixed baseline, so the UTF-8 byte
    length of normalized content is the exact content-token mass before packing
    specials.  Each admitted record is consumed once; this function never
    samples, repeats, or synthesizes documents.
    """
    by_source = validate_family_identities(family_identities)
    records_tuple = tuple(records)
    seen_record_ids: set[str] = set()
    seen_normalized_sha: set[str] = set()
    mass_by_language_family: dict[str, Counter[str]] = defaultdict(Counter)
    lengths_by_language: dict[str, list[int]] = defaultdict(list)
    document_count_by_language: Counter[str] = Counter()

    for record in records_tuple:
        if record.record_id in seen_record_ids:
            raise CorpusDiversityError(f"duplicate record_id: {record.record_id}")
        seen_record_ids.add(record.record_id)
        if record.normalized_sha256 in seen_normalized_sha:
            raise CorpusDiversityError(
                "duplicate normalized document would inflate corpus diversity/volume: "
                f"{record.normalized_sha256}"
            )
        seen_normalized_sha.add(record.normalized_sha256)
        identity = by_source.get(record.source_id)
        if identity is None:
            raise CorpusDiversityError(
                f"missing source-family classification for {record.source_id!r}"
            )
        if record.language not in policy.required_languages:
            continue
        token_mass = len(record.normalized_text.encode("utf-8"))
        if token_mass <= 0:
            raise CorpusDiversityError("admitted record has zero byte-token mass")
        mass_by_language_family[record.language][identity.family_id] += token_mass
        lengths_by_language[record.language].append(token_mass)
        document_count_by_language[record.language] += 1

    languages: dict[str, object] = {}
    total_language_mass = 0
    for language in policy.required_languages:
        metrics = _family_metrics(mass_by_language_family.get(language, {}))
        metrics["document_count"] = document_count_by_language[language]
        metrics["document_length_byte_tokens"] = _document_length_distribution(
            lengths_by_language.get(language, [])
        )
        metrics["meets_minimum_family_count"] = (
            metrics["family_count"] >= policy.minimum_families_per_language
        )
        languages[language] = metrics
        total_language_mass += int(metrics["token_mass"])

    balance = {}
    for language in policy.required_languages:
        token_mass = int(languages[language]["token_mass"])
        balance[language] = token_mass / total_language_mass if total_language_mass else 0.0

    return {
        "schema_version": "12-6.corpus-source-family-diversity.v1",
        "token_mass_unit": "normalized_utf8_byte_tokens_before_packing_specials",
        "document_dedup_required": True,
        "languages": languages,
        "language_balance": balance,
        "total_token_mass": total_language_mass,
        "all_required_languages_meet_family_floor": all(
            bool(languages[language]["meets_minimum_family_count"])
            for language in policy.required_languages
        ),
    }
