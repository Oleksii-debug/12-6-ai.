"""Execute exact current-Rada replacement through incumbent indexed global dedup.

Execution-only carrier. It reconstructs the incumbent Nomis-free V8 graph, removes
the exact predecessor Rada family object, validates the exact current-Rada replay
candidate through the Product adapter, then delegates matching to the already
qualified incumbent indexed matcher. Outputs contain hashes/metadata only.

This module grants no corpus, tokenizer-fit, training, final-test, paid-compute,
or scale-promotion authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import ssl
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for location in (str(SRC), str(TOOLS)):
    if location not in sys.path:
        sys.path.insert(0, location)

from twelve_six.data import rada_current_snapshot_dedup_adapter_v1 as rada

PARENT_HEAD = "0294dd8f04e28cd7992f2afce4cdb3af39846d98"
PARENT_AUTHORITY_ID = "543f9cdd5a9aacaf2cc00b5d4057ad8b142aa685870545cff8bd518f8085055e"
PARENT_AUTHORITY_BLOB = "9953940072fdd973cd6c83eb5899314d3484da30"
PARENT_ADAPTER_BLOB = "b07bf36c4efd84b524b5f1330d8650179dd0e5a1"
PARENT_QP_AUTHORITY_BLOB = "6b0c7e173460f23d3d3b702448b228253135c6cc"
PARENT_RIGHTS_BLOB = "4a6cb0bd6b009ef36c9d2fb712967a4ae1cbfe0b"

NBU_HELPER_HEAD = "b5235cfd83852854e345dea6c2e89cfbcd1cef79"
NBU_HELPER_BLOB = "dc0ed88924610fc7ac19ba4fc7d960d8679c9742"
V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
NBU_HELPER_PATH = "tools/run_d03_nbu_current40_global_dedup_v1.py"

EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_DECLARED_BYTES = 6_093_965
REPLACED_SOURCE_ID = "ua.rada.open-data.laws-texts.d23314"
REPLACED_DECLARED_BYTES = 88_565
REPLACED_RAW_BYTES = 332_400
REPLACED_RAW_SHA256 = "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4"
EXPECTED_REPLACED_OBJECTS = 1
EXPECTED_POST_REPLACEMENT_OBJECTS = 262
EXPECTED_POST_REPLACEMENT_DECLARED_BYTES = 6_005_400

EXPECTED_CURRENT_OBJECTS = 101_733
EXPECTED_CURRENT_PAYLOAD_BYTES = 192_393_157
EXPECTED_CURRENT_JSONL_BYTES = 224_897_989
EXPECTED_CURRENT_JSONL_SHA256 = (
    "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
)
EXPECTED_CURRENT_INVENTORY_SHA256 = (
    "468a3132819527e1ce2b3aa375a55a15d55b7af53e66bf06ae1e5a69daf73e7a"
)
EXPECTED_CURRENT_DUPLICATE_HASHES = 765
CURRENT_REPLAY_EVIDENCE_STATUS = "CURRENT_SNAPSHOT_REPLAY_ZERO_CREDIT"
MATCHER_ONLY_EVIDENCE_STATUS = "DEDICATED_TERMINAL"

EXPECTED_COMBINED_OBJECTS = 101_995
EXPECTED_COMBINED_DECLARED_BYTES = 198_398_557

AUTHORITY_PATH = (
    ROOT / "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
)
ADAPTER_PATH = "src/twelve_six/data/rada_current_snapshot_dedup_adapter_v1.py"
QP_AUTHORITY_PATH = "src/twelve_six/data/rada_current_snapshot_qp_authority_v1.py"
RIGHTS_PATH = "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json"

SURVIVOR_SCHEMA = "12-6.d03-rada-current-global-dedup-survivors.v1"
EVIDENCE_SCHEMA = "12-6.d03-rada-current-global-dedup-execution.v1"
TWO_CLEAN_SCHEMA = "12-6.d03-rada-current-global-dedup-two-clean.v1"

PRODUCT_INDEXED_HEAD = "d34f5cf35cc53724d3b1f93967a29fe901c5342c"
PRODUCT_INDEXED_CORE_BLOB = "b7bb13c7a96f6c8e9a7f88d940aab0d0ed6a93e7"
EXPECTED_QUALIFIED_REPORT_SHA256 = (
    "64e687ae431804862003d5839b9a90c715838e794daad907200f0abfc73b4333"
)
EXPECTED_SELECTION_PROJECTION_SHA256 = (
    "601398c39769dabb930997f25506eaaf71bd8960aa82d8af9c697d1cc4075e22"
)
EXPECTED_RADA_SURVIVOR_AUTHORITY_SHA256 = (
    "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
)


class RadaCurrentGlobalDedupError(RuntimeError):
    """Fail-closed current-Rada global-dedup execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCurrentGlobalDedupError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def git(*args: str, cwd: Path = ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RadaCurrentGlobalDedupError(f"git execution failed: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or str(proc.returncode)
        raise RadaCurrentGlobalDedupError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return proc.stdout.strip()


def git_blob(path: str, *, cwd: Path = ROOT) -> str:
    return git("hash-object", str(cwd / path), cwd=cwd)


def verify_product_parent(expected_execution_head: str) -> None:
    require(is_hex(expected_execution_head, 40), "execution head must be lowercase 40-hex")
    require(git("rev-parse", "HEAD") == expected_execution_head, "execution HEAD drift")
    require(
        git("merge-base", PARENT_HEAD, "HEAD") == PARENT_HEAD,
        "exact #2808 parent is not merge-base",
    )
    expected_blobs = {
        str(AUTHORITY_PATH.relative_to(ROOT)): PARENT_AUTHORITY_BLOB,
        ADAPTER_PATH: PARENT_ADAPTER_BLOB,
        QP_AUTHORITY_PATH: PARENT_QP_AUTHORITY_BLOB,
        RIGHTS_PATH: PARENT_RIGHTS_BLOB,
    }
    for path, expected in expected_blobs.items():
        observed = git("rev-parse", f"{PARENT_HEAD}:{path}")
        require(observed == expected, f"#2808 parent blob drift: {path}")
        require(git("rev-parse", f"HEAD:{path}") == expected, f"child changed parent path: {path}")
        require(git_blob(path) == expected, f"worktree parent path drift: {path}")


def load_helper(nbu_root: Path) -> ModuleType:
    require(nbu_root.is_dir(), "NBU helper worktree missing")
    require(git("rev-parse", "HEAD", cwd=nbu_root) == NBU_HELPER_HEAD, "NBU helper HEAD drift")
    helper_path = nbu_root / NBU_HELPER_PATH
    require(helper_path.is_file(), "NBU helper runner missing")
    require(git_blob(NBU_HELPER_PATH, cwd=nbu_root) == NBU_HELPER_BLOB, "NBU helper blob drift")

    import twelve_six
    import twelve_six.data

    nbu_pkg = str((nbu_root / "src/twelve_six").resolve(strict=True))
    nbu_data = str((nbu_root / "src/twelve_six/data").resolve(strict=True))
    if nbu_pkg not in twelve_six.__path__:
        twelve_six.__path__.append(nbu_pkg)
    if nbu_data not in twelve_six.data.__path__:
        twelve_six.data.__path__.append(nbu_data)

    for location in (
        str((nbu_root / "tools").resolve(strict=True)),
        str((nbu_root / "src").resolve(strict=True)),
    ):
        if location not in sys.path:
            sys.path.append(location)

    spec = importlib.util.spec_from_file_location("exact_nbu_global_dedup_helper", helper_path)
    require(spec is not None and spec.loader is not None, "cannot load exact NBU helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RadaCurrentGlobalDedupError("cannot import exact NBU helper") from exc
    require(module.EXPECTED_MAIN == "019944d5fe12334791f05f1232d13de4a12e37d3", "helper main drift")
    return module


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RadaCurrentGlobalDedupError(f"cannot load JSON: {path}") from exc
    require(type(value) is dict, f"JSON root must be exact object: {path}")
    return value


def declared_capacity(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    label: str,
) -> int:
    rows = inventory.get("sources")
    require(type(rows) is list, f"{label}: source vector missing")
    ids: set[str] = set()
    total = 0
    for row in rows:
        require(type(row) is dict, f"{label}: source row invalid")
        source_id = row.get("source_id")
        declared = row.get("declared_capacity_bytes")
        require(
            type(source_id) is str and source_id and source_id not in ids,
            f"{label}: source id invalid or duplicate",
        )
        require(type(declared) is int and declared >= 0, f"{label}: declared bytes invalid")
        ids.add(source_id)
        total += declared
    require(ids == set(payloads), f"{label}: inventory/payload coverage mismatch")
    require(
        all(type(key) is str and type(value) is bytes for key, value in payloads.items()),
        f"{label}: payload map type drift",
    )
    return total


def remove_incumbent_rada_family(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    require(type(inventory) is dict, "base inventory must be exact object")
    rows = inventory.get("sources")
    require(type(rows) is list, "base sources missing")
    require(len(rows) == EXPECTED_BASE_OBJECTS, "base object count drift")
    require(
        declared_capacity(inventory, payloads, label="base")
        == EXPECTED_BASE_DECLARED_BYTES,
        "base declared-capacity drift",
    )

    removed = [
        row
        for row in rows
        if type(row) is dict and row.get("source_family") == rada.SOURCE_FAMILY
    ]
    require(len(removed) == EXPECTED_REPLACED_OBJECTS, "incumbent Rada family cardinality drift")
    row = removed[0]
    require(row.get("source_id") == REPLACED_SOURCE_ID, "incumbent Rada source id drift")
    require(
        row.get("declared_capacity_bytes") == REPLACED_DECLARED_BYTES,
        "incumbent Rada declared bytes drift",
    )
    raw = payloads.get(REPLACED_SOURCE_ID)
    require(type(raw) is bytes, "incumbent Rada raw payload missing")
    require(len(raw) == REPLACED_RAW_BYTES, "incumbent Rada raw byte count drift")
    require(sha256(raw) == REPLACED_RAW_SHA256, "incumbent Rada raw SHA drift")

    replacement_inventory = copy.deepcopy(dict(inventory))
    replacement_inventory["sources"] = [
        copy.deepcopy(source)
        for source in rows
        if source.get("source_id") != REPLACED_SOURCE_ID
    ]
    replacement_payloads = dict(payloads)
    del replacement_payloads[REPLACED_SOURCE_ID]
    require(
        len(replacement_inventory["sources"]) == EXPECTED_POST_REPLACEMENT_OBJECTS,
        "post-replacement base object count drift",
    )
    require(
        declared_capacity(
            replacement_inventory,
            replacement_payloads,
            label="post-replacement base",
        )
        == EXPECTED_POST_REPLACEMENT_DECLARED_BYTES,
        "post-replacement base declared bytes drift",
    )
    proof = {
        "source_family": rada.SOURCE_FAMILY,
        "replaced_source_ids": [REPLACED_SOURCE_ID],
        "replaced_source_object_count": EXPECTED_REPLACED_OBJECTS,
        "replaced_declared_capacity_bytes": REPLACED_DECLARED_BYTES,
        "replaced_raw_bytes": REPLACED_RAW_BYTES,
        "replaced_raw_sha256": REPLACED_RAW_SHA256,
        "replacement_required_by_current_adapter": True,
        "append_forbidden_by_current_adapter": True,
    }
    return replacement_inventory, replacement_payloads, proof


def validate_current_projection(
    candidate_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projection = rada.validate_and_project_current_rada_candidate(
        candidate_jsonl,
        AUTHORITY_PATH,
        repository_root=ROOT,
        retain_payloads=True,
    )
    require(projection.sources is not None, "current Rada source projection missing")
    require(projection.payloads is not None, "current Rada payload projection missing")
    sources = [dict(row) for row in projection.sources]
    payloads = dict(projection.payloads)
    receipt = projection.receipt
    require(receipt.get("replay_authority_identity_sha256") == PARENT_AUTHORITY_ID, "Rada authority identity drift")
    require(receipt.get("candidate_jsonl_sha256") == EXPECTED_CURRENT_JSONL_SHA256, "Rada candidate identity drift")
    require(receipt.get("candidate_jsonl_file_bytes") == EXPECTED_CURRENT_JSONL_BYTES, "Rada candidate transport bytes drift")
    require(receipt.get("source_object_count") == EXPECTED_CURRENT_OBJECTS, "Rada projected object count drift")
    require(receipt.get("source_payload_utf8_bytes") == EXPECTED_CURRENT_PAYLOAD_BYTES, "Rada projected payload bytes drift")
    require(receipt.get("accepted_inventory_sha256") == EXPECTED_CURRENT_INVENTORY_SHA256, "Rada accepted inventory drift")
    require(receipt.get("exact_duplicate_payload_hashes_preserved") == EXPECTED_CURRENT_DUPLICATE_HASHES, "Rada duplicate observation drift")
    require(receipt.get("requires_replace_existing_source_family") is True, "Rada replacement requirement missing")
    require(receipt.get("must_not_append_to_existing_source_family") is True, "Rada append prohibition missing")
    require(receipt.get("consumer_gate") == "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY", "Rada consumer gate drift")
    require(receipt.get("rights_scope") == "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY", "Rada rights scope drift")
    require(receipt.get("bulk_corpus_admission_granted") is False, "Rada corpus admission widened")
    require(receipt.get("training_authority_granted") is False, "Rada training authority widened")
    require(receipt.get("rights_recheck_for_training_required") is True, "Rada rights recheck boundary lost")
    require(receipt.get("canonical_capacity_credited") == 0, "Rada capacity credit widened")
    require(receipt.get("training_authorized_bytes") == 0, "Rada training bytes widened")
    require(receipt.get("tokenizer_fit_authorized") is False, "Rada tokenizer fit widened")
    require(len(sources) == EXPECTED_CURRENT_OBJECTS, "Rada projected source cardinality drift")
    require(len(payloads) == EXPECTED_CURRENT_OBJECTS, "Rada projected payload cardinality drift")
    require(sum(len(raw) for raw in payloads.values()) == EXPECTED_CURRENT_PAYLOAD_BYTES, "Rada payload sum drift")
    return sources, payloads, receipt


def project_current_for_matcher(
    current_sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create an ephemeral V3-compatible projection without widening authority."""
    require(bool(current_sources), "current Rada source projection empty")
    projected: list[dict[str, Any]] = []
    for row in current_sources:
        require(type(row) is dict, "current Rada projected source row invalid")
        require(
            row.get("evidence_status") == CURRENT_REPLAY_EVIDENCE_STATUS,
            "current Rada replay evidence status drift",
        )
        matcher_row = copy.deepcopy(row)
        matcher_row["evidence_status"] = MATCHER_ONLY_EVIDENCE_STATUS
        projected.append(matcher_row)

    require(
        all(
            row.get("evidence_status") == CURRENT_REPLAY_EVIDENCE_STATUS
            for row in current_sources
        ),
        "matcher projection mutated Product replay rows",
    )
    proof = {
        "projection_scope": "GLOBAL_DEDUP_MATCHER_INPUT_ONLY",
        "input_evidence_status": CURRENT_REPLAY_EVIDENCE_STATUS,
        "matcher_evidence_status": MATCHER_ONLY_EVIDENCE_STATUS,
        "projected_source_object_count": len(projected),
        "source_admission_authority_granted": False,
        "training_authority_granted": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
    }
    return projected, proof


PREFIX_FILTER_ALGORITHM = "INCUMBENT_V1_CONSERVATIVE_RARE_PREFIX_V1"
PREFIX_NATURAL_OVERLAP_NUMERATOR = 88
PREFIX_NATURAL_OVERLAP_DENOMINATOR = 100
PREFIX_CODE_OVERLAP_NUMERATOR = 90
PREFIX_CODE_OVERLAP_DENOMINATOR = 100
PREFIX_CODE_SKELETON_OVERLAP_NUMERATOR = 90
PREFIX_CODE_SKELETON_OVERLAP_DENOMINATOR = 100
PREFIX_EDGE_SHARED_CHARACTERS = 80
_PREFIX_FILTER_LAST_STATS: dict[str, int | str] | None = None


def _positive_exact_int(value: object, label: str) -> int:
    require(type(value) is int and value > 0, f"{label} must be a positive exact int")
    return int(value)


def _ceil_ratio(numerator: int, denominator: int, count: int) -> int:
    require(
        type(numerator) is int
        and type(denominator) is int
        and type(count) is int
        and numerator > 0
        and denominator > 0
        and numerator <= denominator
        and count >= 0,
        "invalid prefix-overlap ratio",
    )
    return (numerator * count + denominator - 1) // denominator


def _verify_prefix_threshold_contract(v1: Any) -> None:
    thresholds = getattr(v1, "DEFAULT_THRESHOLDS", None)
    require(type(thresholds) is dict, "incumbent threshold vector missing")

    natural_near = thresholds.get("natural_near_jaccard")
    natural_fragment = thresholds.get("natural_fragment_containment")
    code_near = thresholds.get("code_near_jaccard")
    code_fragment = thresholds.get("code_fragment_containment")
    code_copy = thresholds.get("code_copy_jaccard")
    for label, value in (
        ("natural_near_jaccard", natural_near),
        ("natural_fragment_containment", natural_fragment),
        ("code_near_jaccard", code_near),
        ("code_fragment_containment", code_fragment),
        ("code_copy_jaccard", code_copy),
    ):
        require(type(value) is float and 0.0 < value <= 1.0, f"{label} threshold drift")

    natural_q = PREFIX_NATURAL_OVERLAP_NUMERATOR / PREFIX_NATURAL_OVERLAP_DENOMINATOR
    code_q = PREFIX_CODE_OVERLAP_NUMERATOR / PREFIX_CODE_OVERLAP_DENOMINATOR
    skeleton_q = (
        PREFIX_CODE_SKELETON_OVERLAP_NUMERATOR
        / PREFIX_CODE_SKELETON_OVERLAP_DENOMINATOR
    )
    # For Jaccard J, every positive pair satisfies
    # intersection/min(|A|,|B|) >= 2J/(1+J).  Fragment containment directly
    # lower-bounds the same ratio.  These carrier ratios are deliberately no
    # stronger than either incumbent-positive path.
    require(
        natural_q <= float(natural_fragment)
        and natural_q <= (2.0 * float(natural_near)) / (1.0 + float(natural_near)),
        "natural prefix ratio is not a conservative incumbent necessary condition",
    )
    require(
        code_q <= float(code_fragment)
        and code_q <= (2.0 * float(code_near)) / (1.0 + float(code_near)),
        "code prefix ratio is not a conservative incumbent necessary condition",
    )
    require(
        skeleton_q <= (2.0 * float(code_copy)) / (1.0 + float(code_copy)),
        "code-skeleton prefix ratio is not a conservative incumbent necessary condition",
    )


def _prefix_candidate_pairs(
    v1: Any,
    fingerprints: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | Any,
    *,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
    edge_lines: Any,
) -> tuple[list[tuple[int, int]], dict[str, int | str]]:
    """Conservative stronger blocking; exact pair decisions remain incumbent V1."""
    candidate_limit = _positive_exact_int(max_candidate_pairs, "max_candidate_pairs")
    posting_limit = _positive_exact_int(max_index_postings, "max_index_postings")
    expansion_limit = _positive_exact_int(max_pair_expansions, "max_pair_expansions")
    _verify_prefix_threshold_contract(v1)

    count = len(fingerprints)
    require(count > 0, "prefix matcher fingerprint vector empty")
    packed: set[int] = set()
    work: dict[str, int | str] = {
        "algorithm": PREFIX_FILTER_ALGORITHM,
        "source_count": count,
        "index_postings": 0,
        "pair_expansion_attempts": 0,
        "frequency_scan_items": 0,
        "prefix_queries": 0,
        "exact_bucket_signatures": 0,
        "edge_prefix_sources": 0,
    }

    def source_id(index: int) -> str:
        row = fingerprints[index].get("row")
        require(type(row) is dict, "prefix matcher row missing")
        value = row.get("source_id")
        require(type(value) is str and value, "prefix matcher source id missing")
        return value

    def add_pair(left: int, right: int) -> None:
        if left == right:
            return
        if left > right:
            left, right = right, left
        packed.add(left * count + right)
        if len(packed) > candidate_limit:
            raise RadaCurrentGlobalDedupError(
                f"prefix candidate pair budget exceeded: >{candidate_limit}"
            )

    def reserve_expansions(amount: int, label: str) -> None:
        require(type(amount) is int and amount >= 0, f"{label} expansion count invalid")
        current = int(work["pair_expansion_attempts"])
        if current + amount > expansion_limit:
            raise RadaCurrentGlobalDedupError(
                f"{label} pair expansion work budget exceeded: >{expansion_limit}"
            )
        work["pair_expansion_attempts"] = current + amount

    def post(
        index_map: dict[str, list[int]],
        key: str,
        index: int,
        label: str,
    ) -> None:
        current = int(work["index_postings"])
        if current >= posting_limit:
            raise RadaCurrentGlobalDedupError(
                f"{label} index posting work budget exceeded: >{posting_limit}"
            )
        index_map[key].append(index)
        work["index_postings"] = current + 1

    # Exact/origin relations are independent of modality and must never be
    # pruned by the overlap filter.
    exact_maps: list[dict[str, list[int]]] = [
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    ]
    for index, item in enumerate(fingerprints):
        row = item.get("row")
        require(type(row) is dict, "prefix matcher row invalid")
        origin = row.get("origin_key")
        raw_sha = item.get("raw_sha256")
        normalized_sha = item.get("normalized_sha256")
        for label, value, index_map in (
            ("origin", origin, exact_maps[0]),
            ("raw", raw_sha, exact_maps[1]),
            ("normalized", normalized_sha, exact_maps[2]),
        ):
            require(type(value) is str and value, f"prefix matcher {label} key invalid")
            post(index_map, value, index, f"exact-{label}")

    seen_exact_buckets: set[tuple[int, ...]] = set()
    for index_map in exact_maps:
        for bucket in index_map.values():
            unique = tuple(sorted(set(bucket)))
            if len(unique) < 2 or unique in seen_exact_buckets:
                continue
            seen_exact_buckets.add(unique)
            work["exact_bucket_signatures"] = int(work["exact_bucket_signatures"]) + 1
            expansion = len(unique) * (len(unique) - 1) // 2
            reserve_expansions(expansion, "exact-bucket")
            for offset, left in enumerate(unique):
                for right in unique[offset + 1 :]:
                    add_pair(left, right)

    def add_high_overlap_candidates(
        indices: list[int],
        *,
        field: str,
        numerator: int,
        denominator: int,
        label: str,
    ) -> None:
        if not indices:
            return
        frequencies: Counter[str] = Counter()
        values_by_index: dict[int, frozenset[str]] = {}
        for index in indices:
            values = fingerprints[index].get(field)
            require(
                isinstance(values, frozenset)
                and all(type(value) is str for value in values),
                f"{label} set shape drift",
            )
            values_by_index[index] = values
            frequencies.update(values)
            scanned = int(work["frequency_scan_items"]) + len(values)
            if scanned > posting_limit:
                raise RadaCurrentGlobalDedupError(
                    f"{label} frequency scan work budget exceeded: >{posting_limit}"
                )
            work["frequency_scan_items"] = scanned

        # Descending cardinality means the current item is always the smaller
        # (or equal-size) side of every previously indexed pair.  If an incumbent
        # positive pair requires R shared values from that smaller set, then a
        # prefix of |S|-R+1 rare-first values must intersect the larger full set.
        ordered_indices = sorted(
            indices,
            key=lambda index: (-len(values_by_index[index]), source_id(index)),
        )
        full_index: dict[str, list[int]] = defaultdict(list)
        for index in ordered_indices:
            values = values_by_index[index]
            if values:
                ordered_values = sorted(
                    values,
                    key=lambda value: (frequencies[value], value),
                )
                required_overlap = _ceil_ratio(numerator, denominator, len(values))
                prefix_length = len(values) - required_overlap + 1
                require(
                    1 <= prefix_length <= len(values),
                    f"{label} prefix length invalid",
                )
                prefix = ordered_values[:prefix_length]
                work["prefix_queries"] = int(work["prefix_queries"]) + len(prefix)
                for value in prefix:
                    bucket = full_index.get(value, ())
                    reserve_expansions(len(bucket), f"{label}-prefix")
                    for prior in bucket:
                        add_pair(prior, index)
                for value in ordered_values:
                    post(full_index, value, index, f"{label}-full")
            else:
                # Empty shingle sets cannot satisfy positive Jaccard/containment.
                continue

    natural_indices: list[int] = []
    code_indices: list[int] = []
    for index, item in enumerate(fingerprints):
        row = item.get("row")
        require(type(row) is dict, "prefix matcher modality row invalid")
        modality = row.get("modality")
        require(type(modality) is str and modality, "prefix matcher modality missing")
        (code_indices if modality == "code" else natural_indices).append(index)

    add_high_overlap_candidates(
        natural_indices,
        field="shingles",
        numerator=PREFIX_NATURAL_OVERLAP_NUMERATOR,
        denominator=PREFIX_NATURAL_OVERLAP_DENOMINATOR,
        label="natural-content",
    )
    add_high_overlap_candidates(
        code_indices,
        field="shingles",
        numerator=PREFIX_CODE_OVERLAP_NUMERATOR,
        denominator=PREFIX_CODE_OVERLAP_DENOMINATOR,
        label="code-content",
    )
    add_high_overlap_candidates(
        code_indices,
        field="skeleton_shingles",
        numerator=PREFIX_CODE_SKELETON_OVERLAP_NUMERATOR,
        denominator=PREFIX_CODE_SKELETON_OVERLAP_DENOMINATOR,
        label="code-skeleton",
    )

    # Publisher boilerplate is report-significant even though it does not collapse
    # capacity.  A positive pair has >=80 shared normalized edge-line characters.
    # For each current source, choose a rare-first prefix whose excluded suffix has
    # <80 total characters.  A positive pair therefore must share at least one
    # prefix line with the previously indexed full edge set.
    edge_sets: dict[int, frozenset[str]] = {}
    edge_frequencies: Counter[str] = Counter()
    for index, item in enumerate(fingerprints):
        text_value = item.get("text")
        require(type(text_value) is str, "prefix matcher text missing")
        values = edge_lines(v1, text_value)
        require(
            isinstance(values, frozenset)
            and all(type(value) is str and 32 <= len(value) <= 320 for value in values),
            "edge-line set shape drift",
        )
        edge_sets[index] = values
        edge_frequencies.update(values)
        scanned = int(work["frequency_scan_items"]) + len(values)
        if scanned > posting_limit:
            raise RadaCurrentGlobalDedupError(
                f"edge frequency scan work budget exceeded: >{posting_limit}"
            )
        work["frequency_scan_items"] = scanned

    edge_index: dict[str, list[int]] = defaultdict(list)
    for index in sorted(range(count), key=source_id):
        values = edge_sets[index]
        ordered_values = sorted(
            values,
            key=lambda value: (edge_frequencies[value], value),
        )
        total_weight = sum(len(value) for value in ordered_values)
        if total_weight >= PREFIX_EDGE_SHARED_CHARACTERS:
            remaining = total_weight
            prefix: list[str] = []
            for value in ordered_values:
                prefix.append(value)
                remaining -= len(value)
                if remaining < PREFIX_EDGE_SHARED_CHARACTERS:
                    break
            require(prefix, "edge prefix unexpectedly empty")
            work["edge_prefix_sources"] = int(work["edge_prefix_sources"]) + 1
            work["prefix_queries"] = int(work["prefix_queries"]) + len(prefix)
            for value in prefix:
                bucket = edge_index.get(value, ())
                reserve_expansions(len(bucket), "edge-prefix")
                for prior in bucket:
                    add_pair(prior, index)
        for value in ordered_values:
            post(edge_index, value, index, "edge-full")

    pairs = [
        (value // count, value % count)
        for value in sorted(packed)
    ]
    work["unique_candidate_pairs"] = len(pairs)
    work["candidate_limit"] = candidate_limit
    work["index_posting_limit"] = posting_limit
    work["pair_expansion_limit"] = expansion_limit
    return pairs, work


def _run_prefix_filter_selftest(v1: Any, edge_lines: Any) -> dict[str, int]:
    """Differential bounded fixture: every incumbent V1 match must remain reachable."""
    def fp(
        source_id: str,
        *,
        modality: str = "text",
        origin: str | None = None,
        raw: str | None = None,
        normalized: str | None = None,
        shingles: frozenset[str] | None = None,
        tokens: tuple[str, ...] | None = None,
        skeleton: tuple[str, ...] | None = None,
        skeleton_shingles: frozenset[str] | None = None,
        text_value: str = "short synthetic fixture",
    ) -> dict[str, Any]:
        return {
            "row": {
                "source_id": source_id,
                "source_family": "synthetic-prefix-selftest",
                "modality": modality,
                "origin_key": origin or f"origin:{source_id}",
            },
            "text": text_value,
            "raw_sha256": raw or ("a" * 63 + source_id[-1]),
            "normalized_sha256": normalized or ("b" * 63 + source_id[-1]),
            "tokens": tokens or tuple(f"t{source_id}-{i}" for i in range(24)),
            "shingles": shingles or frozenset({f"s{source_id}-{i}" for i in range(12)}),
            "skeleton": skeleton or tuple(f"k{source_id}-{i}" for i in range(24)),
            "skeleton_shingles": skeleton_shingles
            or frozenset({f"ks{source_id}-{i}" for i in range(12)}),
        }

    fixtures: list[list[dict[str, Any]]] = []

    fixtures.append([
        fp("e0", origin="shared-origin"),
        fp("e1", origin="shared-origin"),
    ])
    fixtures.append([
        fp("r0", raw="c" * 64),
        fp("r1", raw="c" * 64),
    ])
    fixtures.append([
        fp("n0", raw="d" * 64, normalized="e" * 64),
        fp("n1", raw="f" * 64, normalized="e" * 64),
    ])

    natural_common = frozenset({f"natural-common-{i}" for i in range(9)})
    fixtures.append([
        fp("j0", shingles=natural_common | {"natural-left"}),
        fp("j1", shingles=natural_common | {"natural-right"}),
    ])
    fragment_common = frozenset({f"fragment-common-{i}" for i in range(9)})
    fixtures.append([
        fp(
            "f0",
            shingles=fragment_common | {"fragment-left"},
            tokens=tuple(f"f0-token-{i}" for i in range(24)),
        ),
        fp(
            "f1",
            shingles=fragment_common
            | frozenset({f"fragment-right-{i}" for i in range(11)}),
            tokens=tuple(f"f1-token-{i}" for i in range(24)),
        ),
    ])

    code_common = frozenset({f"code-common-{i}" for i in range(13)})
    fixtures.append([
        fp("c0", modality="code", shingles=code_common | {"code-left"}),
        fp("c1", modality="code", shingles=code_common | {"code-right"}),
    ])
    code_fragment_common = frozenset({f"code-fragment-common-{i}" for i in range(9)})
    fixtures.append([
        fp(
            "g0",
            modality="code",
            shingles=code_fragment_common | {"code-fragment-left"},
            tokens=tuple(f"g0-token-{i}" for i in range(24)),
        ),
        fp(
            "g1",
            modality="code",
            shingles=code_fragment_common
            | frozenset({f"code-fragment-right-{i}" for i in range(11)}),
            tokens=tuple(f"g1-token-{i}" for i in range(24)),
        ),
    ])
    skeleton_common = frozenset({f"skeleton-common-{i}" for i in range(10)})
    fixtures.append([
        fp(
            "k0",
            modality="code",
            skeleton=tuple(f"k0-skeleton-{i}" for i in range(24)),
            skeleton_shingles=skeleton_common | {"skeleton-left"},
        ),
        fp(
            "k1",
            modality="code",
            skeleton=tuple(f"k1-skeleton-{i}" for i in range(24)),
            skeleton_shingles=skeleton_common | {"skeleton-right"},
        ),
    ])

    shared_line_a = "A" * 45
    shared_line_b = "B" * 45
    fixtures.append([
        fp(
            "p0",
            text_value="\n".join([shared_line_a, shared_line_b, "left-only-" + "L" * 40]),
        ),
        fp(
            "p1",
            text_value="\n".join([shared_line_a, shared_line_b, "right-only-" + "R" * 40]),
        ),
    ])

    positive_pairs = 0
    candidate_pairs = 0
    for fixture in fixtures:
        expected = {
            (left, right)
            for left in range(len(fixture))
            for right in range(left + 1, len(fixture))
            if v1._pair_matches(fixture[left], fixture[right])
        }
        candidates, _stats = _prefix_candidate_pairs(
            v1,
            fixture,
            max_candidate_pairs=1_000,
            max_index_postings=100_000,
            max_pair_expansions=100_000,
            edge_lines=edge_lines,
        )
        observed = set(candidates)
        require(
            expected <= observed,
            "prefix selftest pruned an incumbent-positive pair",
        )
        positive_pairs += len(expected)
        candidate_pairs += len(observed)

    require(positive_pairs >= 9, "prefix selftest did not exercise all incumbent match families")
    return {
        "fixture_groups": len(fixtures),
        "incumbent_positive_pairs": positive_pairs,
        "prefix_candidate_pairs": candidate_pairs,
    }


def execute_product_indexed_matcher(
    helper: ModuleType,
    matcher: Any,
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> tuple[dict[str, Any], float, dict[str, int], dict[str, Any]]:
    """Execute the exact #2827 Product candidate generator on the real Rada graph."""
    indexed = helper.indexed
    core = getattr(indexed, "_core", None)
    require(core is not None, "indexed executor core missing")
    original = getattr(core, "candidate_pair_indices", None)
    measured = getattr(core, "candidate_pair_indices_with_stats", None)
    require(callable(original), "Product candidate generator missing")
    require(callable(measured), "Product measured candidate generator missing")
    require(
        getattr(indexed, "candidate_pair_indices", None) is original,
        "indexed facade/core candidate generator identity drift",
    )
    require(
        getattr(indexed, "candidate_pair_indices_with_stats", None) is measured,
        "indexed facade/core measured generator identity drift",
    )

    # Attest the exact incumbent matcher closure before any compatibility dispatch.
    indexed.attest_incumbent_runtime(matcher)
    captured_stats: dict[str, int] = {}

    def exact_product_generator(
        v1: Any,
        fingerprints: Any,
        *,
        max_candidate_pairs: int = 5_000_000,
        max_index_postings: int = 100_000_000,
        max_pair_expansions: int = 100_000_000,
    ) -> list[tuple[int, int]]:
        pairs, stats = measured(
            v1,
            fingerprints,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
        )
        captured_stats.clear()
        captured_stats.update(stats)
        return pairs

    # audit_payloads_indexed resolves this exact core global. The temporary wrapper
    # captures telemetry only; all candidate generation is delegated to the exact
    # Product candidate_pair_indices_with_stats implementation above.
    core.candidate_pair_indices = exact_product_generator
    try:
        started = time.perf_counter()
        report = indexed.audit_payloads_indexed(
            matcher,
            inventory,
            payloads,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
        )
        elapsed = time.perf_counter() - started
    finally:
        core.candidate_pair_indices = original

    require(
        getattr(core, "candidate_pair_indices", None) is original,
        "Product candidate generator restore failed",
    )
    require(captured_stats, "Product candidate work telemetry missing")
    for key, limit in (
        ("unique_candidate_pairs", max_candidate_pairs),
        ("index_postings", max_index_postings),
        ("pair_expansion_attempts", max_pair_expansions),
    ):
        value = captured_stats.get(key)
        require(
            type(value) is int and 0 <= value <= limit,
            f"Product candidate telemetry invalid: {key}",
        )
    matcher.verify_report(report)
    require(
        report.get("report_sha256") == EXPECTED_QUALIFIED_REPORT_SHA256,
        "Product candidate generator changed incumbent Rada report identity",
    )
    qualification = {
        "product_head_sha": PRODUCT_INDEXED_HEAD,
        "product_core_blob_sha1": PRODUCT_INDEXED_CORE_BLOB,
        "exact_product_candidate_generator_executed": True,
        "report_identity_matches_qualified_carrier": True,
    }
    return report, elapsed, dict(captured_stats), qualification


def compose_current_graph(
    helper: ModuleType,
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    current_sources: list[dict[str, Any]],
    current_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    replacement_inventory, replacement_payloads, replacement_proof = (
        remove_incumbent_rada_family(base_inventory, base_payloads)
    )
    base_ids = {row["source_id"] for row in replacement_inventory["sources"]}
    current_ids = {row["source_id"] for row in current_sources}
    require(len(current_ids) == len(current_sources), "current Rada source IDs duplicate")
    require(current_ids == set(current_payloads), "current Rada source/payload coverage mismatch")
    require(not (base_ids & current_ids), "current Rada source ID collides after replacement")
    matcher_sources, matcher_projection = project_current_for_matcher(current_sources)
    require(
        {row["source_id"] for row in matcher_sources} == current_ids,
        "matcher-only Rada projection changed source identities",
    )

    inventory = copy.deepcopy(replacement_inventory)
    inventory["sources"] = [
        *copy.deepcopy(replacement_inventory["sources"]),
        *matcher_sources,
    ]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 source graph with the predecessor "
        "ua.rada.open-data.laws-texts family removed, then exact current-Rada "
        "replay projection supplied by #2808; pair decisions delegate only to "
        "terminal V3 semantics through the qualified indexed executor."
    )
    payloads = dict(replacement_payloads)
    payloads.update(current_payloads)
    require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object count drift")
    require(
        declared_capacity(inventory, payloads, label="combined")
        == EXPECTED_COMBINED_DECLARED_BYTES,
        "combined declared-capacity drift",
    )
    quarantine_authority = load_json(
        helper.ROOT / helper.clean_successor.QUARANTINE_CONFIG_PATH
    )
    helper._verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    replacement_proof = {
        **replacement_proof,
        "matcher_only_status_projection": matcher_projection,
    }
    return inventory, payloads, replacement_proof


def verify_report_current_rada(
    report: Mapping[str, Any],
    current_sources: list[dict[str, Any]],
    current_payloads: Mapping[str, bytes],
) -> dict[str, Mapping[str, Any]]:
    require(report.get("local_free_only") is True, "matcher local-free boundary drift")
    require(
        report.get("model_training_executed") is False,
        "matcher falsely claims model training",
    )
    require(
        report.get("source_admission_authority") is False,
        "matcher falsely grants source admission authority",
    )
    require(report.get("source_count") == EXPECTED_COMBINED_OBJECTS, "report source count drift")
    terminal = report.get("terminal_candidates")
    require(type(terminal) is dict, "terminal matcher summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    clusters = terminal.get("duplicate_cluster_count")
    require(before == EXPECTED_COMBINED_DECLARED_BYTES, "report pre-dedup bytes drift")
    require(type(after) is int and 0 < after <= before, "report post-dedup bytes invalid")
    require(type(discount) is int and discount == before - after, "report discount arithmetic drift")
    require(type(clusters) is int and clusters >= 0, "report cluster count invalid")

    rows = report.get("sources")
    require(type(rows) is list and len(rows) == EXPECTED_COMBINED_OBJECTS, "report source vector drift")
    by_id = {
        row.get("source_id"): row
        for row in rows
        if type(row) is dict and type(row.get("source_id")) is str
    }
    require(len(by_id) == len(rows), "report source IDs invalid or duplicate")

    expected_by_id = {row["source_id"]: row for row in current_sources}
    require(set(expected_by_id) == set(current_payloads), "current projection map drift")
    for source_id, expected in expected_by_id.items():
        observed = by_id.get(source_id)
        require(type(observed) is dict, f"current Rada source absent from report: {source_id}")
        raw = current_payloads[source_id]
        raw_sha = sha256(raw)
        require(
            expected.get("evidence_status") == CURRENT_REPLAY_EVIDENCE_STATUS,
            f"current Rada replay status drift: {source_id}",
        )
        require(
            observed.get("evidence_status") == MATCHER_ONLY_EVIDENCE_STATUS,
            f"current Rada matcher-only status drift: {source_id}",
        )
        for field in ("source_family", "modality", "declared_capacity_bytes"):
            require(
                type(observed.get(field)) is type(expected.get(field))
                and observed.get(field) == expected.get(field),
                f"current Rada report field drift: {source_id}:{field}",
            )
        stable_origin = expected.get("stable_origin_id")
        stable_object = expected.get("stable_object_id")
        require(type(stable_origin) is str and stable_origin, "Rada stable origin missing")
        require(type(stable_object) is str and stable_object, "Rada stable object missing")
        require(
            observed.get("stable_origin_id_sha256") == sha256(stable_origin.encode("utf-8")),
            f"current Rada stable origin drift: {source_id}",
        )
        require(
            observed.get("stable_object_id_sha256") == sha256(stable_object.encode("utf-8")),
            f"current Rada stable object drift: {source_id}",
        )
        require(observed.get("verified_raw_bytes") == len(raw), f"current Rada raw byte drift: {source_id}")
        require(observed.get("verified_raw_sha256") == raw_sha, f"current Rada raw SHA drift: {source_id}")
        require(observed.get("comparison_policy") == "DATA232_GENERIC_FROM_RAW", f"current Rada comparison policy drift: {source_id}")
        require(observed.get("comparison_payload_bytes") == len(raw), f"current Rada comparison byte drift: {source_id}")
        require(observed.get("comparison_payload_sha256") == raw_sha, f"current Rada comparison SHA drift: {source_id}")
    return by_id


def survivor_authority(
    helper: ModuleType,
    report: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    current_source_ids: set[str],
    replacement_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selection = helper.v9_semantics._derive_survivors(report)
    selection_core = dict(selection)
    selection_id = selection_core.pop("survivor_authority_sha256", None)
    require(
        is_hex(selection_id, 64) and selection_id == sha256(canonical(selection_core)),
        "selection projection self-hash mismatch",
    )
    require(
        selection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "selection pre-dedup object count drift",
    )
    require(
        selection.get("pre_dedup_declared_capacity_bytes")
        == EXPECTED_COMBINED_DECLARED_BYTES,
        "selection pre-dedup bytes drift",
    )
    require(
        selection.get("post_dedup_declared_capacity_bytes")
        == report["terminal_candidates"]["conservative_unique_capacity_bytes_after"],
        "selection post-dedup bytes drift",
    )
    require(
        selection.get("duplicate_discount_bytes")
        == report["terminal_candidates"]["duplicate_discount_bytes"],
        "selection duplicate discount drift",
    )
    survivor_ids = selection.get("survivor_source_ids")
    require(
        type(survivor_ids) is list
        and all(type(value) is str and value for value in survivor_ids)
        and len(survivor_ids) == len(set(survivor_ids)),
        "selection survivor IDs invalid",
    )
    require(all(value in by_id for value in survivor_ids), "selection references unknown source")
    current_survivors = sorted(set(survivor_ids) & current_source_ids)
    current_removed = sorted(current_source_ids - set(current_survivors))
    require(
        len(current_survivors) + len(current_removed) == EXPECTED_CURRENT_OBJECTS,
        "current Rada survivor partition drift",
    )

    inventory: list[dict[str, Any]] = []
    survivor_bytes = 0
    for source_id in current_survivors:
        row = by_id[source_id]
        declared = row.get("declared_capacity_bytes")
        require(type(declared) is int and declared > 0, "Rada survivor declared bytes invalid")
        verified_sha = row.get("verified_raw_sha256")
        require(is_hex(verified_sha, 64), "Rada survivor raw SHA invalid")
        inventory.append(
            {
                "source_id": source_id,
                "declared_capacity_bytes": declared,
                "verified_raw_sha256": verified_sha,
                "stable_origin_id_sha256": row.get("stable_origin_id_sha256"),
                "stable_object_id_sha256": row.get("stable_object_id_sha256"),
            }
        )
        survivor_bytes += declared

    terminal = report["terminal_candidates"]
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": report.get("report_sha256"),
        "selection_projection_schema": selection.get("schema_version"),
        "selection_projection_sha256": selection_id,
        "replacement_proof": dict(replacement_proof),
        "pre_dedup_source_object_count": EXPECTED_COMBINED_OBJECTS,
        "pre_dedup_declared_capacity_bytes": EXPECTED_COMBINED_DECLARED_BYTES,
        "post_dedup_survivor_source_object_count": selection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "post_dedup_declared_capacity_bytes": selection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": selection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": selection.get("duplicate_cluster_count"),
        "current_rada_input_source_object_count": EXPECTED_CURRENT_OBJECTS,
        "current_rada_input_payload_bytes": EXPECTED_CURRENT_PAYLOAD_BYTES,
        "current_rada_survivor_source_object_count": len(current_survivors),
        "current_rada_survivor_declared_capacity_bytes": survivor_bytes,
        "current_rada_discounted_source_object_count": len(current_removed),
        "current_rada_survivor_inventory": inventory,
        "current_rada_discounted_source_ids": current_removed,
        "terminal_conservative_unique_capacity_bytes_after": terminal.get(
            "conservative_unique_capacity_bytes_after"
        ),
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "rights_recheck_for_training_required": True,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    return {**core, "survivor_authority_sha256": sha256(canonical(core))}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(dict(value)) + b"\n")


def reconstruct_with_bounded_transport_retry(
    helper: ModuleType,
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Replay exact V7 with bounded retry for transport-only fetch failures.

    The frozen historical fetcher, acquisition URLs, normal TLS verification,
    source hashes, rights gates and matcher semantics stay unchanged. Only a
    timeout or TLS EOF may retry, at most three attempts on the same URL.
    """
    historical_v8 = helper.v8
    original_capture = historical_v8._capture_terminal_v7

    def capture_with_retry(
        historical_root: Path,
        historical_config: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, bytes]]:
        historical_v7 = historical_v8._load_v7(historical_root)
        fetch_module = historical_v7.v6.v5.v1
        original_fetch = fetch_module.fetch_exact_source

        def retry_fetch(url: str) -> bytes:
            for attempt in range(1, 4):
                try:
                    return original_fetch(url)
                except OSError as exc:
                    retryable = (
                        isinstance(exc, (TimeoutError, ssl.SSLEOFError))
                        or (
                            isinstance(exc, URLError)
                            and isinstance(
                                exc.reason,
                                (TimeoutError, ssl.SSLEOFError),
                            )
                        )
                    )
                    if retryable and attempt < 3:
                        time.sleep(0.25 * attempt)
                        continue
                    try:
                        host = urlsplit(url).hostname or "unknown"
                    except ValueError:
                        host = "invalid-url"
                    url_sha256 = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    exc.add_note(
                        "historical V7 source fetch failed: "
                        f"host={host}; acquisition_url_sha256={url_sha256}; "
                        f"attempts={attempt}"
                    )
                    raise
            raise AssertionError("unreachable historical fetch attempt state")

        fetch_module.fetch_exact_source = retry_fetch
        try:
            return original_capture(historical_root, historical_config)
        finally:
            fetch_module.fetch_exact_source = original_fetch

    historical_v8._capture_terminal_v7 = capture_with_retry
    try:
        return helper._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    finally:
        historical_v8._capture_terminal_v7 = original_capture


def execute(args: argparse.Namespace) -> None:
    verify_product_parent(args.expected_execution_head)
    helper = load_helper(args.nbu_helper_root)
    require(
        git("rev-parse", "HEAD", cwd=args.v7_root) == V7_HEAD,
        "historical V7 HEAD drift",
    )
    current_sources, current_payloads, projection_receipt = validate_current_projection(
        args.candidate_jsonl
    )
    config = load_json(
        args.nbu_helper_root / "configs/data/next100_065f_global_dedup_v8.json"
    )
    matcher, base_inventory, base_payloads, removal = (
        reconstruct_with_bounded_transport_retry(
            helper,
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            config=config,
        )
    )
    helper._verify_removal_proof(removal)
    inventory, payloads, replacement_proof = compose_current_graph(
        helper,
        base_inventory,
        base_payloads,
        current_sources,
        current_payloads,
    )
    report, _elapsed, prefix_stats, prefix_selftest = execute_product_indexed_matcher(
        helper,
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    by_id = verify_report_current_rada(report, current_sources, current_payloads)
    authority = survivor_authority(
        helper,
        report,
        by_id,
        set(current_payloads),
        replacement_proof,
    )
    require(
        authority.get("selection_projection_sha256")
        == EXPECTED_SELECTION_PROJECTION_SHA256,
        "Product candidate generator changed survivor selection projection",
    )
    require(
        authority.get("survivor_authority_sha256")
        == EXPECTED_RADA_SURVIVOR_AUTHORITY_SHA256,
        "Product candidate generator changed Rada survivor authority",
    )
    evidence_core = {
        "schema_version": EVIDENCE_SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "parent_product_head_sha": PARENT_HEAD,
        "parent_replay_authority_identity_sha256": PARENT_AUTHORITY_ID,
        "nbu_helper_head_sha": NBU_HELPER_HEAD,
        "nbu_helper_blob_sha1": NBU_HELPER_BLOB,
        "v7_head_sha": V7_HEAD,
        "projection_receipt_identity_sha256": projection_receipt.get(
            "receipt_identity_sha256"
        ),
        "candidate_jsonl_sha256": EXPECTED_CURRENT_JSONL_SHA256,
        "candidate_source_object_count": EXPECTED_CURRENT_OBJECTS,
        "candidate_payload_bytes": EXPECTED_CURRENT_PAYLOAD_BYTES,
        "replacement_proof": replacement_proof,
        "matcher_source_admission_authority": False,
        "matcher_model_training_executed": False,
        "combined_source_object_count": EXPECTED_COMBINED_OBJECTS,
        "combined_declared_capacity_bytes": EXPECTED_COMBINED_DECLARED_BYTES,
        "matcher_report_sha256": report.get("report_sha256"),
        "candidate_filter": {
            "algorithm": "PRODUCT_2827_THRESHOLD_AWARE_INDEXED_CORE",
            "product_head_sha": PRODUCT_INDEXED_HEAD,
            "product_core_blob_sha1": PRODUCT_INDEXED_CORE_BLOB,
            "work_telemetry": prefix_stats,
            "compatibility_qualification": prefix_selftest,
            "pair_decision_authority": "EXACT_INCUMBENT_V1_PAIR_MATCHES",
            "lineage_and_report_authority": "QUALIFIED_INCUMBENT_INDEXED_EXECUTOR",
            "science_changed": False,
        },
        "survivor_authority_sha256": authority["survivor_authority_sha256"],
        "current_rada_survivor_source_object_count": authority[
            "current_rada_survivor_source_object_count"
        ],
        "current_rada_survivor_declared_capacity_bytes": authority[
            "current_rada_survivor_declared_capacity_bytes"
        ],
        "rights_scope": "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
        "rights_recheck_for_training_required": True,
        "raw_text_persisted": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    require(not args.output_dir.exists(), "output directory already exists")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "dedup-report.json", report)
    write_json(args.output_dir / "rada-survivor-authority.json", authority)
    write_json(args.output_dir / "execution-evidence.json", evidence)
    print("CURRENT_RADA_GLOBAL_DEDUP=PASS")
    print("REPORT_SHA256=" + str(report.get("report_sha256")))
    print("SURVIVOR_AUTHORITY_SHA256=" + authority["survivor_authority_sha256"])
    print(
        "CURRENT_RADA_SURVIVOR_OBJECTS="
        + str(authority["current_rada_survivor_source_object_count"])
    )
    print(
        "CURRENT_RADA_SURVIVOR_DECLARED_BYTES="
        + str(authority["current_rada_survivor_declared_capacity_bytes"])
    )
    print("PRODUCT_INDEXED_HEAD=" + PRODUCT_INDEXED_HEAD)
    print("PRODUCT_CANDIDATE_PAIRS=" + str(prefix_stats["unique_candidate_pairs"]))
    print("PRODUCT_PAIR_EXPANSIONS=" + str(prefix_stats["pair_expansion_attempts"]))
    print("PRODUCT_INDEX_POSTINGS=" + str(prefix_stats["index_postings"]))
    print("CAPACITY_CREDIT=0")
    print("RIGHTS_RECHECK_REQUIRED=true")


def compare(args: argparse.Namespace) -> None:
    names = (
        "dedup-report.json",
        "rada-survivor-authority.json",
        "execution-evidence.json",
    )
    hashes: dict[str, str] = {}
    for name in names:
        a = (args.output_a / name).read_bytes()
        b = (args.output_b / name).read_bytes()
        require(a == b, f"two-clean output differs: {name}")
        hashes[name] = sha256(a)

    evidence = load_json(args.output_a / "execution-evidence.json")
    survivor = load_json(args.output_a / "rada-survivor-authority.json")
    report = load_json(args.output_a / "dedup-report.json")
    require(
        report.get("report_sha256") == evidence.get("matcher_report_sha256"),
        "two-clean report/evidence identity mismatch",
    )
    require(
        survivor.get("survivor_authority_sha256")
        == evidence.get("survivor_authority_sha256"),
        "two-clean survivor/evidence identity mismatch",
    )
    require(evidence.get("canonical_capacity_credited") == 0, "two-clean capacity widened")
    require(evidence.get("training_authorized_bytes") == 0, "two-clean training widened")
    require(evidence.get("tokenizer_fit_authorized") is False, "two-clean tokenizer widened")
    require(evidence.get("rights_recheck_for_training_required") is True, "two-clean rights gate lost")

    core = {
        "schema_version": TWO_CLEAN_SCHEMA,
        "execution_head_sha": evidence["execution_head_sha"],
        "fresh_process_count": 2,
        "byte_identical_outputs": True,
        "output_file_sha256": hashes,
        "matcher_report_sha256": evidence["matcher_report_sha256"],
        "survivor_authority_sha256": evidence["survivor_authority_sha256"],
        "current_rada_survivor_source_object_count": evidence[
            "current_rada_survivor_source_object_count"
        ],
        "current_rada_survivor_declared_capacity_bytes": evidence[
            "current_rada_survivor_declared_capacity_bytes"
        ],
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "rights_recheck_for_training_required": True,
        "next_gate": "CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING",
    }
    proof = {**core, "two_clean_identity_sha256": sha256(canonical(core))}
    require(not args.proof.exists(), "two-clean proof already exists")
    write_json(args.proof, proof)
    print("CURRENT_RADA_TWO_CLEAN_GLOBAL_DEDUP=PASS")
    print("TWO_CLEAN_IDENTITY_SHA256=" + proof["two_clean_identity_sha256"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    sub = result.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", allow_abbrev=False)
    run.add_argument("--candidate-jsonl", type=Path, required=True)
    run.add_argument("--nbu-helper-root", type=Path, required=True)
    run.add_argument("--v7-root", type=Path, required=True)
    run.add_argument("--bulk-workspace", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--expected-execution-head", required=True)
    run.add_argument("--max-candidate-pairs", type=int, default=25_000_000)
    run.add_argument("--max-index-postings", type=int, default=100_000_000)
    run.add_argument("--max-pair-expansions", type=int, default=100_000_000)

    cmp_parser = sub.add_parser("compare", allow_abbrev=False)
    cmp_parser.add_argument("--output-a", type=Path, required=True)
    cmp_parser.add_argument("--output-b", type=Path, required=True)
    cmp_parser.add_argument("--proof", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            execute(args)
        else:
            compare(args)
    except (
        RadaCurrentGlobalDedupError,
        rada.RadaCurrentSnapshotDedupAdapterError,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
