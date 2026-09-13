"""Performance-equivalent candidate indexing for the incumbent NEXT100-065 V3 matcher.

This module does not decide whether two source objects match. It only proves that
certain pairs cannot match any V1 pair predicate, then delegates every retained pair
to the exact incumbent ``v1._pair_matches`` implementation. V3 lineage and summary
semantics remain delegated to the exact incumbent V3 module.
"""
from __future__ import annotations

import hashlib
import inspect
import marshal
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import CodeType
from typing import Any

EXPECTED_V3_GIT_BLOB_SHA1 = "11490b1803e0aa2266d8ac0053676efcfb0f91ba"
EXPECTED_V1_GIT_BLOB_SHA1 = "84cdf00b2d468d2709a542ac3ee2ea372aae5716"
EXPECTED_DATA232_GIT_BLOB_SHA1 = "dab5da98dfc43133aa8f3c2e3c78c809252b741b"
EXPECTED_THRESHOLDS = {
    "natural_shingle_tokens": 3,
    "natural_near_jaccard": 0.80,
    "natural_fragment_containment": 0.88,
    "natural_fragment_min_tokens": 18,
    "code_shingle_tokens": 7,
    "code_near_jaccard": 0.86,
    "code_fragment_containment": 0.90,
    "code_fragment_min_tokens": 16,
    "code_copy_jaccard": 0.82,
    "code_copy_min_tokens": 16,
}
DEFAULT_MAX_INDEX_POSTINGS = 100_000_000
DEFAULT_MAX_PAIR_EXPANSIONS = 100_000_000


class IndexedExecutionError(RuntimeError):
    """Fail-closed indexed-execution contract error."""


def _git_blob_sha1(payload: bytes) -> str:
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()


def _module_path(module: Any) -> Path:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path:
        raise IndexedExecutionError("matcher module has no inspectable __file__")
    return Path(path)


def _module_blob_sha1(module: Any) -> str:
    path = _module_path(module)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IndexedExecutionError(f"cannot read matcher module bytes: {path}") from exc
    return _git_blob_sha1(payload)


def _code_digest(code: CodeType) -> str:
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _canonical_namespace(module: Any, label: str) -> dict[str, Any]:
    path = _module_path(module)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IndexedExecutionError(f"cannot read {label} module bytes: {path}") from exc
    namespace: dict[str, Any] = {
        "__name__": f"_twelve_six_indexed_attested_{label}",
        "__file__": str(path),
        "__package__": getattr(module, "__package__", None),
    }
    try:
        exec(  # noqa: S102 - exact local authority source bytes are intentionally re-executed
            compile(payload, str(path), "exec", dont_inherit=True), namespace
        )
    except Exception as exc:
        raise IndexedExecutionError(f"cannot reconstruct attested {label} executable closure") from exc
    return namespace


def _attest_executable_module(module: Any, label: str) -> dict[str, Any]:
    """Reconstruct exact source and bind all source-defined live function objects to it."""
    canonical = _canonical_namespace(module, label)
    for name, expected in canonical.items():
        if not inspect.isfunction(expected) or expected.__globals__ is not canonical:
            continue
        live = getattr(module, name, None)
        if not inspect.isfunction(live):
            raise IndexedExecutionError(f"{label} executable closure missing function {name}")
        if live.__globals__ is not vars(module):
            raise IndexedExecutionError(f"{label} callable global ownership drift: {name}")
        if _code_digest(live.__code__) != _code_digest(expected.__code__):
            raise IndexedExecutionError(f"{label} callable code drift: {name}")
        if live.__defaults__ != expected.__defaults__ or live.__kwdefaults__ != expected.__kwdefaults__:
            raise IndexedExecutionError(f"{label} callable default drift: {name}")
    return canonical


def _semantic_snapshot(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, re.Pattern):
        return ("regex", value.pattern, value.flags)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _semantic_snapshot(child)) for key, child in value.items())),
        )
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_semantic_snapshot(child) for child in value), key=repr)))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_semantic_snapshot(child) for child in value))
    raise IndexedExecutionError(f"unsupported semantic global type: {type(value).__name__}")


def _attest_globals(module: Any, canonical: Mapping[str, Any], label: str, names: Sequence[str]) -> None:
    for name in names:
        if name not in canonical or not hasattr(module, name):
            raise IndexedExecutionError(f"{label} semantic global missing: {name}")
        if _semantic_snapshot(getattr(module, name)) != _semantic_snapshot(canonical[name]):
            raise IndexedExecutionError(f"{label} semantic global drift: {name}")


def attest_incumbent_runtime(v3: Any) -> None:
    """Bind execution to exact terminal V3 + V1 + DATA-232 executable authority."""
    v1 = getattr(v3, "v1", None)
    if v1 is None:
        raise IndexedExecutionError("V3 runtime does not expose incumbent v1 module")
    if _module_blob_sha1(v3) != EXPECTED_V3_GIT_BLOB_SHA1:
        raise IndexedExecutionError("V3 matcher source blob drift")
    if _module_blob_sha1(v1) != EXPECTED_V1_GIT_BLOB_SHA1:
        raise IndexedExecutionError("V1 matcher source blob drift")

    normalize = getattr(v1, "normalize_for_contamination", None)
    module_name = getattr(normalize, "__module__", None)
    data232 = sys.modules.get(module_name) if isinstance(module_name, str) else None
    if data232 is None:
        raise IndexedExecutionError("V1 runtime does not expose terminal DATA-232 module closure")
    if _module_blob_sha1(data232) != EXPECTED_DATA232_GIT_BLOB_SHA1:
        raise IndexedExecutionError("DATA-232 matcher source blob drift")

    for name in ("DEFAULT_THRESHOLDS", "TOKEN_RE", "normalize_for_contamination", "code_skeleton_tokens"):
        if getattr(v1, name, None) is not getattr(data232, name, None):
            raise IndexedExecutionError(f"V1/DATA-232 mixed executable closure: {name}")

    thresholds = getattr(v1, "DEFAULT_THRESHOLDS", None)
    if type(thresholds) is not dict or thresholds != EXPECTED_THRESHOLDS:
        raise IndexedExecutionError("incumbent threshold vector drift")
    if tuple(thresholds) != tuple(EXPECTED_THRESHOLDS):
        raise IndexedExecutionError("incumbent threshold key/order drift")

    data_canonical = _attest_executable_module(data232, "DATA232")
    _attest_globals(
        data232,
        data_canonical,
        "DATA232",
        (
            "SCHEMA", "ALGORITHM", "NORMALIZATION", "CODE_SKELETON", "DEFAULT_THRESHOLDS",
            "OUTCOME_PARTS", "INVISIBLE", "TOKEN_RE", "CODE_TOKEN_RE", "LINE_COMMENT",
            "BLOCK_COMMENT", "STRING", "KEYWORDS",
        ),
    )

    v1_canonical = _attest_executable_module(v1, "V1")
    if v1_canonical.get("DEFAULT_THRESHOLDS") is not data232.DEFAULT_THRESHOLDS:
        raise IndexedExecutionError("reconstructed V1 does not bind terminal DATA-232 thresholds")
    _attest_globals(
        v1,
        v1_canonical,
        "V1",
        ("SCHEMA", "ALGORITHM", "MAX_SOURCE_BYTES", "COLLAPSE_MATCH_TYPES", "STATUS_SCOPES"),
    )

    v3_canonical = _attest_executable_module(v3, "V3")
    if v3_canonical.get("v1") is not v1:
        raise IndexedExecutionError("reconstructed V3 does not bind exact incumbent V1 module")
    _attest_globals(
        v3,
        v3_canonical,
        "V3",
        (
            "SCHEMA", "INVENTORY_SCHEMA", "ALGORITHM", "TERMINAL_STATUSES",
            "RUST_BOOK_PROSE_POLICY", "RELATION_MATCH_TYPES", "LINEAGE_COLLAPSE_MATCH_TYPES",
            "CAPACITY_COLLAPSE_MATCH_TYPES",
        ),
    )


def _positive_exact_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise IndexedExecutionError(f"{label} must be a positive exact int")
    return value


def _add_bucket_pairs(
    bucket: Sequence[int],
    packed: set[int],
    count: int,
    candidate_limit: int,
    pair_expansion_limit: int,
    seen_buckets: set[tuple[int, ...]],
    work: dict[str, int],
) -> None:
    unique = tuple(sorted(set(bucket)))
    if len(unique) < 2 or unique in seen_buckets:
        return
    seen_buckets.add(unique)
    work["unique_bucket_signatures"] += 1
    expansion = len(unique) * (len(unique) - 1) // 2
    if work["pair_expansion_attempts"] + expansion > pair_expansion_limit:
        raise IndexedExecutionError(
            f"pair expansion work budget exceeded: >{pair_expansion_limit}; refusing unbounded execution"
        )
    work["pair_expansion_attempts"] += expansion
    for offset, left in enumerate(unique):
        for right in unique[offset + 1 :]:
            packed.add(left * count + right)
            if len(packed) > candidate_limit:
                raise IndexedExecutionError(
                    f"candidate pair budget exceeded: >{candidate_limit}; refusing unbounded execution"
                )


def _edge_lines(v1: Any, text: str) -> frozenset[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    values = {
        v1.normalize_for_contamination(line, "text")
        for line in lines[:10] + lines[-10:]
    }
    return frozenset(value for value in values if 32 <= len(value) <= 320)


def candidate_pair_indices_with_stats(
    v1: Any,
    fingerprints: Sequence[Mapping[str, Any]],
    *,
    max_candidate_pairs: int = 5_000_000,
    max_index_postings: int = DEFAULT_MAX_INDEX_POSTINGS,
    max_pair_expansions: int = DEFAULT_MAX_PAIR_EXPANSIONS,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Return the conservative candidate union plus deterministic index-work telemetry."""
    candidate_limit = _positive_exact_int(max_candidate_pairs, "max_candidate_pairs")
    posting_limit = _positive_exact_int(max_index_postings, "max_index_postings")
    expansion_limit = _positive_exact_int(max_pair_expansions, "max_pair_expansions")
    count = len(fingerprints)
    packed: set[int] = set()
    equal_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
    overlap_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
    work = {"index_postings": 0, "pair_expansion_attempts": 0, "unique_bucket_signatures": 0}

    def post(index_map: dict[tuple[str, str], list[int]], key: tuple[str, str], index: int) -> None:
        if work["index_postings"] >= posting_limit:
            raise IndexedExecutionError(
                f"index posting work budget exceeded: >{posting_limit}; refusing unbounded execution"
            )
        index_map[key].append(index)
        work["index_postings"] += 1

    for index, item in enumerate(fingerprints):
        row = item["row"]
        post(equal_indexes, ("origin", str(row["origin_key"])), index)
        post(equal_indexes, ("raw", str(item["raw_sha256"])), index)
        post(equal_indexes, ("normalized", str(item["normalized_sha256"])), index)

        kind = "code" if row["modality"] == "code" else "natural"
        for shingle in item["shingles"]:
            post(overlap_indexes, (f"{kind}:content", str(shingle)), index)
        if kind == "code":
            for shingle in item["skeleton_shingles"]:
                post(overlap_indexes, ("code:skeleton", str(shingle)), index)
        for line in _edge_lines(v1, str(item["text"])):
            post(overlap_indexes, ("edge", line), index)

    seen_buckets: set[tuple[int, ...]] = set()
    for bucket in equal_indexes.values():
        _add_bucket_pairs(bucket, packed, count, candidate_limit, expansion_limit, seen_buckets, work)
    for bucket in overlap_indexes.values():
        _add_bucket_pairs(bucket, packed, count, candidate_limit, expansion_limit, seen_buckets, work)

    pairs = [(value // count, value % count) for value in sorted(packed)] if count else []
    work["unique_candidate_pairs"] = len(pairs)
    return pairs, work


def candidate_pair_indices(
    v1: Any,
    fingerprints: Sequence[Mapping[str, Any]],
    *,
    max_candidate_pairs: int = 5_000_000,
    max_index_postings: int = DEFAULT_MAX_INDEX_POSTINGS,
    max_pair_expansions: int = DEFAULT_MAX_PAIR_EXPANSIONS,
) -> list[tuple[int, int]]:
    """Return a conservative superset of all pairs that can satisfy V1 predicates."""
    pairs, _ = candidate_pair_indices_with_stats(
        v1,
        fingerprints,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    return pairs


def audit_payloads_indexed(
    v3: Any,
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    max_candidate_pairs: int = 5_000_000,
    max_index_postings: int = DEFAULT_MAX_INDEX_POSTINGS,
    max_pair_expansions: int = DEFAULT_MAX_PAIR_EXPANSIONS,
) -> dict[str, Any]:
    """Execute exact incumbent V3 semantics with necessary-condition pair indexing."""
    attest_incumbent_runtime(v3)
    v1 = v3.v1
    rows, edges = v3._validate_inventory(inventory)
    expected_ids = {row["source_id"] for row in rows}
    if set(payloads) != expected_ids:
        raise IndexedExecutionError("payload coverage must equal exact terminal source inventory")
    validated_rows = v1._validate_inventory(v3._as_v1_inventory(rows))
    fingerprints = [v3._fingerprint(row, payloads[row["source_id"]]) for row in validated_rows]
    pairs = candidate_pair_indices(
        v1,
        fingerprints,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )

    matches = [
        match
        for left_index, right_index in pairs
        for match in v1._pair_matches(fingerprints[left_index], fingerprints[right_index])
    ]
    matches.extend(v3._lineage_matches(fingerprints, edges))
    matches.sort(key=lambda item: (item["left_source_id"], item["right_source_id"], item["match_type"]))

    all_ids = {item["row"]["source_id"] for item in fingerprints}
    terminal = v3._summary_for_ids(fingerprints, matches, all_ids)
    modalities = sorted({item["row"]["modality"] for item in fingerprints})
    by_modality = {
        modality: v3._summary_for_ids(
            fingerprints,
            matches,
            {item["row"]["source_id"] for item in fingerprints if item["row"]["modality"] == modality},
        )
        for modality in modalities
    }
    match_counts = Counter(match["match_type"] for match in matches)
    collapsing_counts = Counter(
        match["match_type"] for match in matches if match.get("capacity_collapsing") is True
    )
    sources = [
        {
            "source_id": item["row"]["source_id"],
            "source_family": item["row"]["source_family"],
            "stable_origin_id_sha256": v1._sha256(item["row"]["stable_origin_id"].encode()),
            "stable_object_id_sha256": v1._sha256(item["row"]["stable_object_id"].encode()),
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
                "identical stable_object_id collapses capacity even when acquisition URLs or wrapper bytes differ"
            ),
            "origin_rule": "stable_origin_id and explicit lineage edges determine independence accounting",
            "sibling_rule": (
                "same-origin sibling files collapse independence but retain capacity unless "
                "byte/copy/derivative evidence creates a capacity-collapsing edge"
            ),
            "normalization_rule": (
                "when a terminal authority defines an exact training-text normalization, "
                "verify its output hash/size before the common contamination normalization and matching pass"
            ),
        },
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": v1._sha256(v1._canonical_bytes(core))}


def execution_stats(
    source_count: int,
    candidate_pair_count: int,
    *,
    index_postings: int | None = None,
    pair_expansion_attempts: int | None = None,
    unique_bucket_signatures: int | None = None,
) -> dict[str, int | float]:
    if type(source_count) is not int or source_count < 0:
        raise IndexedExecutionError("source_count must be a nonnegative exact int")
    if type(candidate_pair_count) is not int or candidate_pair_count < 0:
        raise IndexedExecutionError("candidate_pair_count must be a nonnegative exact int")
    all_pairs = source_count * (source_count - 1) // 2
    if candidate_pair_count > all_pairs:
        raise IndexedExecutionError("candidate pair count exceeds all-pairs count")
    result: dict[str, int | float] = {
        "source_count": source_count,
        "incumbent_all_pair_dispatches": all_pairs,
        "indexed_candidate_pair_dispatches": candidate_pair_count,
        "dispatches_avoided": all_pairs - candidate_pair_count,
        "candidate_fraction": candidate_pair_count / all_pairs if all_pairs else 0.0,
    }
    for name, value in (
        ("index_postings", index_postings),
        ("pair_expansion_attempts", pair_expansion_attempts),
        ("unique_bucket_signatures", unique_bucket_signatures),
    ):
        if value is not None:
            if type(value) is not int or value < 0:
                raise IndexedExecutionError(f"{name} must be a nonnegative exact int")
            result[name] = value
    return result
