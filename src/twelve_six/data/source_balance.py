"""Deterministic provenance-derived source/domain balancing for pretraining corpora.

This layer intentionally sits *under* the incumbent UK/EN/code MixturePlan.  The
incumbent plan chooses the top-level stratum.  SourceBalancePlan only chooses a
source and record inside that already-selected stratum.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

TAXONOMY_SCHEMA = "12-6.source-domain-taxonomy.v1"
POLICY_SCHEMA = "12-6.source-balance-policy.v1"
ANALYSIS_SCHEMA = "12-6.source-balance-analysis.v1"
SELECTION_VERSION = "sha256-unbiased-source-with-replacement-v1"
STRATA = ("uk", "en", "code")
POLICIES = ("raw_proportional", "bounded_source_cap", "tempered_source_sqrt")


class SourceBalanceError(ValueError):
    """Raised when provenance or balancing semantics are ambiguous."""


def canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _url_family(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if host in {"github.com", "gitlab.com"} and parts:
        return f"{host}/{parts[0].casefold()}"
    return host


def provenance_family(record: Mapping[str, Any]) -> str:
    """Derive a stable family only from explicit provenance metadata.

    No text/content inspection is used. Explicit source_family_id wins. URL
    provenance is reduced to host (or host/owner for Git forges). Project-
    authored IDs retain the stratum component so one synthetic generator does
    not pretend to be diverse across modalities.
    """
    explicit = _clean(record.get("source_family_id"))
    if explicit is not None:
        return explicit

    for key in (
        "repository_url",
        "canonical_url",
        "source_url",
        "origin_url",
        "download_url",
    ):
        url = _clean(record.get(key))
        if url is not None:
            family = _url_family(url)
            if family is not None:
                return family

    source_id = _clean(record.get("source_id"))
    if source_id is None:
        raise SourceBalanceError("source_id is required for provenance taxonomy")
    parts = source_id.split(":")
    if parts[0] == "project-authored":
        if len(parts) < 2 or parts[1] not in STRATA:
            raise SourceBalanceError("project-authored source_id must encode uk/en/code")
        return f"project-authored:{parts[1]}"
    if "/" in source_id:
        first = source_id.split("/", 1)[0].casefold()
        if "." in first:
            return first
    if len(parts) >= 2 and parts[0] in {"github", "gitlab"}:
        return f"{parts[0]}.com/{parts[1].casefold()}"
    return source_id


def origin_kind(record: Mapping[str, Any]) -> str:
    external = record.get("external")
    authored = record.get("project_authored")
    if external is True and authored is not True:
        return "real_external"
    if authored is True and external is not True:
        return "project_authored"
    raise SourceBalanceError(
        "record origin must be unambiguously real_external or project_authored"
    )


def length_bucket(byte_tokens: int) -> str:
    if not isinstance(byte_tokens, int) or isinstance(byte_tokens, bool) or byte_tokens <= 0:
        raise SourceBalanceError("byte_tokens must be a positive integer")
    if byte_tokens < 256:
        return "000080-000255"
    if byte_tokens < 1024:
        return "000256-001023"
    if byte_tokens < 4096:
        return "001024-004095"
    if byte_tokens < 16384:
        return "004096-016383"
    return "016384-plus"


@dataclass(frozen=True, slots=True)
class Taxon:
    record_id: str
    stratum: str
    modality: str
    source_id: str
    source_family: str
    length_bucket: str
    origin: str
    byte_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "stratum": self.stratum,
            "modality": self.modality,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "length_bucket": self.length_bucket,
            "origin": self.origin,
            "byte_tokens": self.byte_tokens,
        }


def classify(record: Mapping[str, Any]) -> Taxon:
    record_id = _clean(record.get("record_id"))
    source_id = _clean(record.get("source_id"))
    stratum = _clean(record.get("stratum"))
    modality = _clean(record.get("modality"))
    if record_id is None or source_id is None:
        raise SourceBalanceError("record_id and source_id are required")
    if stratum not in STRATA:
        raise SourceBalanceError("stratum must be uk/en/code")
    expected_modality = "code" if stratum == "code" else "natural"
    if modality != expected_modality:
        raise SourceBalanceError("modality conflicts with stratum")
    byte_tokens = record.get("byte_tokens")
    if not isinstance(byte_tokens, int) or isinstance(byte_tokens, bool):
        raise SourceBalanceError("byte_tokens must be int")
    return Taxon(
        record_id=record_id,
        stratum=stratum,
        modality=modality,
        source_id=source_id,
        source_family=provenance_family(record),
        length_bucket=length_bucket(byte_tokens),
        origin=origin_kind(record),
        byte_tokens=byte_tokens,
    )


def _mass_table(taxa: Iterable[Taxon], key) -> dict[str, dict[str, int]]:
    accum: dict[str, Counter[str]] = defaultdict(Counter)
    for taxon in taxa:
        group = key(taxon)
        accum[group]["documents"] += 1
        accum[group]["byte_tokens"] += taxon.byte_tokens
    return {
        group: {"documents": counts["documents"], "byte_tokens": counts["byte_tokens"]}
        for group, counts in sorted(accum.items())
    }


def concentration(masses: Mapping[str, int]) -> dict[str, object]:
    clean = {str(k): int(v) for k, v in masses.items() if int(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        return {
            "members": 0,
            "total_byte_tokens": 0,
            "top_member": None,
            "top_share": 0.0,
            "effective_count_hill_q2": 0.0,
            "token_entropy_bits": 0.0,
            "normalized_entropy": 0.0,
        }
    ordered = sorted(clean.items(), key=lambda item: (-item[1], item[0]))
    probs = [value / total for _, value in ordered]
    entropy = -sum(p * math.log2(p) for p in probs)
    effective = 1.0 / sum(p * p for p in probs)
    normalized = entropy / math.log2(len(probs)) if len(probs) > 1 else 0.0
    return {
        "members": len(ordered),
        "total_byte_tokens": total,
        "top_member": ordered[0][0],
        "top_share": ordered[0][1] / total,
        "effective_count_hill_q2": effective,
        "token_entropy_bits": entropy,
        "normalized_entropy": normalized,
    }


def analyze_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    ids = [_clean(row.get("record_id")) for row in rows]
    if any(record_id is None for record_id in ids):
        raise SourceBalanceError("all records require record_id")
    if len(ids) != len(set(ids)):
        raise SourceBalanceError("input corpus contains duplicate record_id values")
    taxa = tuple(classify(row) for row in rows)
    tables = {
        "language_modality": _mass_table(taxa, lambda t: f"{t.stratum}:{t.modality}"),
        "source": _mass_table(taxa, lambda t: t.source_id),
        "source_family_domain": _mass_table(taxa, lambda t: t.source_family),
        "document_length_bucket": _mass_table(taxa, lambda t: t.length_bucket),
        "origin": _mass_table(taxa, lambda t: t.origin),
    }
    source_masses = {k: v["byte_tokens"] for k, v in tables["source"].items()}
    family_masses = {
        k: v["byte_tokens"] for k, v in tables["source_family_domain"].items()
    }
    per_stratum: dict[str, Any] = {}
    for stratum in STRATA:
        subset = [taxon for taxon in taxa if taxon.stratum == stratum]
        source = Counter()
        family = Counter()
        for taxon in subset:
            source[taxon.source_id] += taxon.byte_tokens
            family[taxon.source_family] += taxon.byte_tokens
        per_stratum[stratum] = {
            "sources": concentration(source),
            "families": concentration(family),
            "source_byte_tokens": dict(sorted(source.items())),
            "family_byte_tokens": dict(sorted(family.items())),
        }
    taxonomy_core = {
        "schema_version": TAXONOMY_SCHEMA,
        "derivation": "provenance_metadata_only",
        "records": [taxon.to_dict() for taxon in sorted(taxa, key=lambda t: t.record_id)],
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "taxonomy_sha256": sha256_json(taxonomy_core),
        "taxonomy": taxonomy_core,
        "mass": tables,
        "concentration": {
            "overall_sources": concentration(source_masses),
            "overall_families": concentration(family_masses),
            "per_stratum": per_stratum,
        },
    }


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    name: str
    cap_basis_points: int | None = None
    temper_exponent: str | None = None

    def __post_init__(self) -> None:
        if self.name not in POLICIES:
            raise SourceBalanceError(f"unsupported policy: {self.name}")
        if self.name == "bounded_source_cap":
            if (
                not isinstance(self.cap_basis_points, int)
                or isinstance(self.cap_basis_points, bool)
                or not 1 <= self.cap_basis_points <= 10000
            ):
                raise SourceBalanceError("bounded_source_cap requires cap_basis_points")
        elif self.cap_basis_points is not None:
            raise SourceBalanceError("cap_basis_points only belongs to cap policy")
        if self.name == "tempered_source_sqrt":
            if self.temper_exponent != "1/2":
                raise SourceBalanceError("tempered policy is exactly sqrt / exponent 1/2")
        elif self.temper_exponent is not None:
            raise SourceBalanceError("temper_exponent only belongs to tempered policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "cap_basis_points": self.cap_basis_points,
            "temper_exponent": self.temper_exponent,
        }


def source_weight_units(
    source_masses: Mapping[str, int],
    policy: SourcePolicy,
) -> dict[str, int]:
    masses = {str(k): int(v) for k, v in source_masses.items()}
    if not masses or any(value <= 0 for value in masses.values()):
        raise SourceBalanceError("source masses must all be positive")
    if policy.name == "raw_proportional":
        return dict(sorted(masses.items()))
    if policy.name == "bounded_source_cap":
        assert policy.cap_basis_points is not None
        total = sum(masses.values())
        cap = max(1, (total * policy.cap_basis_points) // 10000)
        return {key: min(value, cap) for key, value in sorted(masses.items())}
    if policy.name == "tempered_source_sqrt":
        return {
            key: max(1, math.isqrt(value * 1_000_000))
            for key, value in sorted(masses.items())
        }
    raise AssertionError("unreachable policy")


@dataclass(frozen=True, slots=True)
class SourceBalancePlan:
    corpus_identity_sha256: str
    top_level_mixture_sha256: str
    policy: SourcePolicy
    source_weights_by_stratum: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    seed: int
    sampling_with_replacement: bool = True

    def __post_init__(self) -> None:
        for value, field in (
            (self.corpus_identity_sha256, "corpus_identity_sha256"),
            (self.top_level_mixture_sha256, "top_level_mixture_sha256"),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise SourceBalanceError(f"{field} must be SHA-256")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be integer")
        if self.sampling_with_replacement is not True:
            raise SourceBalanceError("DATA-105 supports explicit with-replacement draws only")
        strata = tuple(name for name, _ in self.source_weights_by_stratum)
        if strata != STRATA:
            raise SourceBalanceError("source weights must be ordered uk/en/code")
        for _, weights in self.source_weights_by_stratum:
            if not weights or any(weight <= 0 for _, weight in weights):
                raise SourceBalanceError("each stratum requires positive source weights")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": POLICY_SCHEMA,
            "corpus_identity_sha256": self.corpus_identity_sha256,
            "top_level_mixture_sha256": self.top_level_mixture_sha256,
            "policy": self.policy.to_dict(),
            "source_weights_by_stratum": {
                stratum: dict(weights) for stratum, weights in self.source_weights_by_stratum
            },
            "seed": self.seed,
            "selection_version": SELECTION_VERSION,
            "sampling": {
                "with_replacement": True,
                "materialize_duplicate_documents": False,
                "replacement_scope": "sampling_trace_only",
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def source_for_draw(self, stratum: str, draw_index: int) -> str:
        if stratum not in STRATA:
            raise SourceBalanceError("unknown stratum")
        if draw_index < 0:
            raise SourceBalanceError("draw_index must be non-negative")
        mapping = dict(self.source_weights_by_stratum)
        weights = mapping[stratum]
        total = sum(weight for _, weight in weights)
        ticket = int.from_bytes(
            hashlib.sha256(
                (
                    f"{self.corpus_identity_sha256}:{self.top_level_mixture_sha256}:"
                    f"{self.seed}:{stratum}:draw:{draw_index}"
                ).encode("utf-8")
            ).digest(),
            "big",
        )
        space = 1 << 256
        cumulative = 0
        for source, weight in weights:
            cumulative += weight
            if ticket * total < cumulative * space:
                return source
        raise AssertionError("source selection escaped range")


def build_source_balance_plan(
    *,
    corpus_identity_sha256: str,
    top_level_mixture_sha256: str,
    analysis: Mapping[str, Any],
    policy: SourcePolicy,
    seed: int,
) -> SourceBalancePlan:
    per_stratum = analysis["concentration"]["per_stratum"]
    ordered = []
    for stratum in STRATA:
        masses = per_stratum[stratum]["source_byte_tokens"]
        weights = source_weight_units(masses, policy)
        ordered.append((stratum, tuple(sorted(weights.items()))))
    return SourceBalancePlan(
        corpus_identity_sha256=corpus_identity_sha256,
        top_level_mixture_sha256=top_level_mixture_sha256,
        policy=policy,
        source_weights_by_stratum=tuple(ordered),
        seed=seed,
    )


def policy_is_effective(plan: SourceBalancePlan, raw_plan: SourceBalancePlan) -> bool:
    current = dict(plan.source_weights_by_stratum)
    baseline = dict(raw_plan.source_weights_by_stratum)
    for stratum in STRATA:
        left = dict(current[stratum])
        right = dict(baseline[stratum])
        if set(left) != set(right):
            return True
        left_total = sum(left.values())
        right_total = sum(right.values())
        if any(
            left[source] * right_total != right[source] * left_total
            for source in left
        ):
            return True
    return False
