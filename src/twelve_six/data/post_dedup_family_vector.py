"""Fail-closed bridge from NEXT100-071 dedup evidence to NEXT100-106 balance input."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data.successor_cross_source_dedup import WORKER, verify_successor_report

INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
STRATUM_MAP = {"uk": "ua", "ua": "ua", "en": "en", "code": "code"}
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


class PostDedupFamilyVectorError(RuntimeError):
    """Raised when a dedup report cannot yield an unambiguous family vector."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDedupFamilyVectorError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _cluster_id(member_family_ids: Sequence[str]) -> str:
    members = sorted(set(member_family_ids))
    if len(members) == 1:
        return members[0]
    digest = hashlib.sha256(_canonical_bytes(members)).hexdigest()
    return f"dedup-family-cluster:{digest}"


def _source_components(
    source_ids: Sequence[str],
    matches: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    pairs = [
        (str(match["left_source_id"]), str(match["right_source_id"]))
        for match in matches
        if match.get("capacity_collapsing") is True
        and match.get("match_type") in v3.CAPACITY_COLLAPSE_MATCH_TYPES
    ]
    return v1._components(sorted(source_ids), sorted(pairs))


def _family_components(
    family_ids: Sequence[str],
    source_family: Mapping[str, str],
    matches: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    pairs: set[tuple[str, str]] = set()
    for match in matches:
        left = str(match["left_source_id"])
        right = str(match["right_source_id"])
        left_family = source_family[left]
        right_family = source_family[right]
        if left_family == right_family:
            continue
        collapses_independence = (
            match.get("capacity_collapsing") is True
            or match.get("independence_collapsing") is True
        )
        if collapses_independence:
            pairs.add(tuple(sorted((left_family, right_family))))
    return v1._components(sorted(set(family_ids)), sorted(pairs))


def build_post_dedup_family_vector(
    report: Mapping[str, Any],
    *,
    head_sha: str,
    workflow_run_id: int,
    workflow_name: str,
) -> dict[str, Any]:
    """Build exact independent-family capacities from a verified successor report.

    The vector is intended for the NEXT100-106 balance gate. It is not itself a
    training or corpus authorization. Trust in ``terminal=true`` additionally
    depends on the referenced workflow run being externally observed as success.
    """
    verify_successor_report(report)
    _require(SHA40_RE.fullmatch(head_sha) is not None, "head_sha must be 40 lowercase hex")
    _require(isinstance(workflow_run_id, int) and workflow_run_id > 0, "workflow_run_id must be positive")
    _require(isinstance(workflow_name, str) and workflow_name.strip(), "workflow_name is required")

    embedded = report.get("v3_report")
    _require(isinstance(embedded, Mapping), "embedded V3 report missing")
    sources_raw = embedded.get("sources")
    matches_raw = embedded.get("matches")
    _require(isinstance(sources_raw, list) and sources_raw, "V3 sources missing")
    _require(isinstance(matches_raw, list), "V3 matches missing")

    sources: dict[str, dict[str, Any]] = {}
    source_family: dict[str, str] = {}
    family_modalities: dict[str, set[str]] = defaultdict(set)
    for raw in sources_raw:
        _require(isinstance(raw, Mapping), "V3 source row must be an object")
        row = dict(raw)
        source_id = row.get("source_id")
        family_id = row.get("source_family")
        modality = row.get("modality")
        capacity = row.get("declared_capacity_bytes")
        _require(isinstance(source_id, str) and source_id, "source_id missing")
        _require(source_id not in sources, f"duplicate source_id: {source_id}")
        _require(isinstance(family_id, str) and family_id, f"{source_id}: family missing")
        _require(modality in STRATUM_MAP, f"{source_id}: unsupported modality {modality!r}")
        _require(isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0, f"{source_id}: invalid capacity")
        sources[source_id] = row
        source_family[source_id] = family_id
        family_modalities[family_id].add(str(modality))

    for family_id, modalities in family_modalities.items():
        _require(len(modalities) == 1, f"family spans multiple strata: {family_id}")

    matches: list[dict[str, Any]] = []
    for raw in matches_raw:
        _require(isinstance(raw, Mapping), "V3 match row must be an object")
        row = dict(raw)
        left = row.get("left_source_id")
        right = row.get("right_source_id")
        _require(left in sources and right in sources and left != right, "match endpoints must be known distinct sources")
        matches.append(row)

    family_components = _family_components(
        list(family_modalities), source_family, matches
    )
    family_component_for: dict[str, tuple[str, ...]] = {}
    for component in family_components:
        frozen = tuple(component)
        modalities = {next(iter(family_modalities[family])) for family in component}
        _require(len(modalities) == 1, "independent family component spans multiple strata")
        for family in component:
            family_component_for[family] = frozen

    capacity_by_component: dict[tuple[str, ...], int] = defaultdict(int)
    for component in _source_components(list(sources), matches):
        component_rows = [sources[source_id] for source_id in component]
        modalities = {str(row["modality"]) for row in component_rows}
        _require(len(modalities) == 1, "capacity duplicate component spans multiple strata")
        family_components_seen = {
            family_component_for[str(row["source_family"])] for row in component_rows
        }
        _require(
            len(family_components_seen) == 1,
            "capacity duplicate component crosses uncollapsed family components",
        )
        family_component = next(iter(family_components_seen))
        capacity_by_component[family_component] += max(
            int(row["declared_capacity_bytes"]) for row in component_rows
        )

    family_rows: list[dict[str, Any]] = []
    for component in sorted(family_components):
        capacity = capacity_by_component.get(tuple(component), 0)
        _require(capacity > 0, "independent family component has zero unique capacity")
        modality = next(iter(family_modalities[component[0]]))
        family_rows.append(
            {
                "family_id": _cluster_id(component),
                "stratum": STRATUM_MAP[modality],
                "unique_bytes": capacity,
            }
        )
    family_rows.sort(key=lambda row: row["family_id"])

    by_stratum = {"ua": 0, "en": 0, "code": 0}
    family_count = {"ua": 0, "en": 0, "code": 0}
    for row in family_rows:
        by_stratum[row["stratum"]] += row["unique_bytes"]
        family_count[row["stratum"]] += 1

    terminal_scope = embedded.get("terminal_candidates")
    _require(isinstance(terminal_scope, Mapping), "terminal candidate scope missing")
    expected_total = int(terminal_scope.get("conservative_unique_capacity_bytes_after", -1))
    _require(sum(by_stratum.values()) == expected_total, "family vector total does not match dedup capacity")
    by_modality = terminal_scope.get("by_modality")
    _require(isinstance(by_modality, Mapping), "dedup modality scope missing")
    for modality, stratum in (("uk", "ua"), ("en", "en"), ("code", "code")):
        scope = by_modality.get(modality)
        _require(isinstance(scope, Mapping), f"dedup modality missing: {modality}")
        expected = int(scope.get("conservative_unique_capacity_bytes_after", -1))
        _require(by_stratum[stratum] == expected, f"family vector capacity mismatch: {stratum}")

    return {
        "schema_version": INPUT_SCHEMA,
        "terminal": True,
        "terminal_semantics": "VALID_ONLY_IF_REFERENCED_DEDUP_WORKFLOW_RUN_IS_EXTERNALLY_TERMINAL_SUCCESS",
        "dedup_authority": {
            "worker_id": WORKER,
            "head_sha": head_sha,
            "evidence_identity_sha256": str(report["report_sha256"]),
            "terminal_verdict": "PASS",
            "workflow_run_id": workflow_run_id,
            "workflow_name": workflow_name,
        },
        "families": family_rows,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": by_stratum,
            "family_count": family_count,
        },
        "claim_boundary": {
            "corpus_materialization_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
