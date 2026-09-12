"""Performance-equivalent candidate indexing for the incumbent NEXT100-065 V3 matcher.

This module does not decide whether two source objects match. It only proves that
certain pairs cannot match any V1 pair predicate, then delegates every retained pair
to the exact incumbent ``v1._pair_matches`` implementation. V3 lineage and summary
semantics remain delegated to the exact incumbent V3 module.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_V3_GIT_BLOB_SHA1 = "11490b1803e0aa2266d8ac0053676efcfb0f91ba"
EXPECTED_V1_GIT_BLOB_SHA1 = "84cdf00b2d468d2709a542ac3ee2ea372aae5716"


class IndexedExecutionError(RuntimeError):
    """Fail-closed indexed-execution contract error."""


def _git_blob_sha1(payload: bytes) -> str:
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()


def _module_blob_sha1(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path:
        raise IndexedExecutionError("matcher module has no inspectable __file__")
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise IndexedExecutionError(f"cannot read matcher module bytes: {path}") from exc
    return _git_blob_sha1(payload)


def attest_incumbent_runtime(
    v3: Any,
    *,
    expected_v3_blob: str = EXPECTED_V3_GIT_BLOB_SHA1,
    expected_v1_blob: str = EXPECTED_V1_GIT_BLOB_SHA1,
) -> None:
    """Bind execution to the exact terminal V3/V1 source blobs and safe thresholds."""
    v1 = getattr(v3, "v1", None)
    if v1 is None:
        raise IndexedExecutionError("V3 runtime does not expose incumbent v1 module")
    if _module_blob_sha1(v3) != expected_v3_blob:
        raise IndexedExecutionError("V3 matcher source blob drift")
    if _module_blob_sha1(v1) != expected_v1_blob:
        raise IndexedExecutionError("V1 matcher source blob drift")

    for name in (
        "_validate_inventory",
        "_fingerprint",
        "_lineage_matches",
        "_summary_for_ids",
    ):
        if not callable(getattr(v3, name, None)):
            raise IndexedExecutionError(f"V3 runtime missing callable {name}")
    for name in ("_validate_inventory", "_pair_matches", "_sha256", "_canonical_bytes"):
        if not callable(getattr(v1, name, None)):
            raise IndexedExecutionError(f"V1 runtime missing callable {name}")

    thresholds = getattr(v1, "DEFAULT_THRESHOLDS", None)
    if not isinstance(thresholds, Mapping):
        raise IndexedExecutionError("incumbent thresholds are not a mapping")
    positive_float = (
        "natural_near_jaccard",
        "natural_fragment_containment",
        "code_near_jaccard",
        "code_fragment_containment",
        "code_copy_jaccard",
    )
    positive_int = (
        "natural_shingle_tokens",
        "natural_fragment_min_tokens",
        "code_shingle_tokens",
        "code_fragment_min_tokens",
        "code_copy_min_tokens",
    )
    for key in positive_float:
        value = thresholds.get(key)
        if type(value) is not float or not 0.0 < value <= 1.0:
            raise IndexedExecutionError(f"unsafe threshold semantics: {key}")
    for key in positive_int:
        value = thresholds.get(key)
        if type(value) is not int or value <= 0:
            raise IndexedExecutionError(f"unsafe threshold semantics: {key}")


def _add_bucket_pairs(bucket: Sequence[int], packed: set[int], count: int, limit: int) -> None:
    unique = sorted(set(bucket))
    for offset, left in enumerate(unique):
        for right in unique[offset + 1 :]:
            packed.add(left * count + right)
            if len(packed) > limit:
                raise IndexedExecutionError(
                    f"candidate pair budget exceeded: >{limit}; refusing unbounded execution"
                )


def _edge_lines(v1: Any, text: str) -> frozenset[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    values = {
        v1.normalize_for_contamination(line, "text")
        for line in lines[:10] + lines[-10:]
    }
    return frozenset(value for value in values if 32 <= len(value) <= 320)


def candidate_pair_indices(
    v1: Any,
    fingerprints: Sequence[Mapping[str, Any]],
    *,
    max_candidate_pairs: int = 5_000_000,
) -> list[tuple[int, int]]:
    """Return a conservative superset of all pairs that can satisfy V1 predicates."""
    if type(max_candidate_pairs) is not int or max_candidate_pairs <= 0:
        raise IndexedExecutionError("max_candidate_pairs must be a positive exact int")
    count = len(fingerprints)
    packed: set[int] = set()
    equal_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
    overlap_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)

    for index, item in enumerate(fingerprints):
        row = item["row"]
        equal_indexes[("origin", str(row["origin_key"]))].append(index)
        equal_indexes[("raw", str(item["raw_sha256"]))].append(index)
        equal_indexes[("normalized", str(item["normalized_sha256"]))].append(index)

        kind = "code" if row["modality"] == "code" else "natural"
        for shingle in item["shingles"]:
            overlap_indexes[(f"{kind}:content", str(shingle))].append(index)
        if kind == "code":
            for shingle in item["skeleton_shingles"]:
                overlap_indexes[("code:skeleton", str(shingle))].append(index)
        for line in _edge_lines(v1, str(item["text"])):
            overlap_indexes[("edge", line)].append(index)

    for bucket in equal_indexes.values():
        if len(bucket) > 1:
            _add_bucket_pairs(bucket, packed, count, max_candidate_pairs)
    for bucket in overlap_indexes.values():
        if len(bucket) > 1:
            _add_bucket_pairs(bucket, packed, count, max_candidate_pairs)

    return [(value // count, value % count) for value in sorted(packed)]


def audit_payloads_indexed(
    v3: Any,
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    max_candidate_pairs: int = 5_000_000,
    expected_v3_blob: str = EXPECTED_V3_GIT_BLOB_SHA1,
    expected_v1_blob: str = EXPECTED_V1_GIT_BLOB_SHA1,
) -> dict[str, Any]:
    """Execute exact incumbent V3 semantics with necessary-condition pair indexing."""
    attest_incumbent_runtime(
        v3,
        expected_v3_blob=expected_v3_blob,
        expected_v1_blob=expected_v1_blob,
    )
    v1 = v3.v1
    rows, edges = v3._validate_inventory(inventory)
    expected_ids = {row["source_id"] for row in rows}
    if set(payloads) != expected_ids:
        raise IndexedExecutionError("payload coverage must equal exact terminal source inventory")
    validated_rows = v1._validate_inventory(v3._as_v1_inventory(rows))
    fingerprints = [
        v3._fingerprint(row, payloads[row["source_id"]]) for row in validated_rows
    ]
    pairs = candidate_pair_indices(
        v1,
        fingerprints,
        max_candidate_pairs=max_candidate_pairs,
    )

    matches = [
        match
        for left_index, right_index in pairs
        for match in v1._pair_matches(fingerprints[left_index], fingerprints[right_index])
    ]
    matches.extend(v3._lineage_matches(fingerprints, edges))
    matches.sort(
        key=lambda item: (
            item["left_source_id"],
            item["right_source_id"],
            item["match_type"],
        )
    )

    all_ids = {item["row"]["source_id"] for item in fingerprints}
    terminal = v3._summary_for_ids(fingerprints, matches, all_ids)
    modalities = sorted({item["row"]["modality"] for item in fingerprints})
    by_modality = {
        modality: v3._summary_for_ids(
            fingerprints,
            matches,
            {
                item["row"]["source_id"]
                for item in fingerprints
                if item["row"]["modality"] == modality
            },
        )
        for modality in modalities
    }
    match_counts = Counter(match["match_type"] for match in matches)
    collapsing_counts = Counter(
        match["match_type"]
        for match in matches
        if match.get("capacity_collapsing") is True
    )
    sources = [
        {
            "source_id": item["row"]["source_id"],
            "source_family": item["row"]["source_family"],
            "stable_origin_id_sha256": v1._sha256(
                item["row"]["stable_origin_id"].encode()
            ),
            "stable_object_id_sha256": v1._sha256(
                item["row"]["stable_object_id"].encode()
            ),
            "modality": item["row"]["modality"],
            "evidence_status": item["row"]["evidence_status"],
            "declared_capacity_bytes": item["row"]["declared_capacity_bytes"],
            "verified_raw_bytes": item["raw_bytes"],
            "verified_raw_sha256": item["raw_sha256"],
            "comparison_policy": item["comparison_policy"],
            "comparison_payload_bytes": item["comparison_payload_bytes"],
            "comparison_payload_sha256": item["comparison_payload_sha256"],
            "normalized_sha256": item["normalized_sha256"],
        }
        for item in fingerprints
    ]
    sources.sort(key=lambda item: item["source_id"])
    core = {
        "schema_version": v3.SCHEMA,
        "algorithm": v3.ALGORITHM,
        "local_free_only": True,
        "model_training_executed": False,
        "source_admission_authority": False,
        "source_count": len(fingerprints),
        "matching_authority": (
            "DATA-232 / DATA-298 exact, normalized, near, fragment and code-skeleton "
            "semantics plus exact source-authority comparison normalization and "
            "NEXT100-065 lineage graph"
        ),
        "thresholds": dict(v1.DEFAULT_THRESHOLDS),
        "sources": sources,
        "matches": matches,
        "match_counts": dict(sorted(match_counts.items())),
        "capacity_collapsing_match_counts": dict(sorted(collapsing_counts.items())),
        "terminal_candidates": {**terminal, "by_modality": by_modality},
        "capacity_policy": {
            "metric": "conservative_unique_capacity_bytes_after",
            "connected_component_rule": (
                "count at most the largest declared-capacity member of each connected "
                "capacity-collapsing duplicate cluster"
            ),
            "url_rule": "URL distinctness is never independence evidence",
            "stable_object_rule": (
                "identical stable_object_id collapses capacity even when acquisition "
                "URLs or wrapper bytes differ"
            ),
            "origin_rule": (
                "stable_origin_id and explicit lineage edges determine independence accounting"
            ),
            "sibling_rule": (
                "same-origin sibling files collapse independence but retain capacity unless "
                "byte/copy/derivative evidence creates a capacity-collapsing edge"
            ),
            "normalization_rule": (
                "when a terminal authority defines an exact training-text normalization, "
                "verify its output hash/size before the common contamination normalization "
                "and matching pass"
            ),
        },
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": v1._sha256(v1._canonical_bytes(core))}


def execution_stats(source_count: int, candidate_pair_count: int) -> dict[str, int | float]:
    if type(source_count) is not int or source_count < 0:
        raise IndexedExecutionError("source_count must be a nonnegative exact int")
    if type(candidate_pair_count) is not int or candidate_pair_count < 0:
        raise IndexedExecutionError("candidate_pair_count must be a nonnegative exact int")
    all_pairs = source_count * (source_count - 1) // 2
    if candidate_pair_count > all_pairs:
        raise IndexedExecutionError("candidate pair count exceeds all-pairs count")
    return {
        "source_count": source_count,
        "incumbent_all_pair_dispatches": all_pairs,
        "indexed_candidate_pair_dispatches": candidate_pair_count,
        "dispatches_avoided": all_pairs - candidate_pair_count,
        "candidate_fraction": candidate_pair_count / all_pairs if all_pairs else 0.0,
    }
