"""Fail-closed two-clean Rada global-dedup execution carrier.

This module owns orchestration only. Source authentication remains in the Rada
adapter, matching science remains in the qualified indexed incumbent executor, and
V3 runtime attestation remains the executor's responsibility.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows/local fallback is explicit evidence.
    resource = None

AUTHORITY_SCHEMA = "12-6.d03-rada-two-clean-dependency-authority.v1"
RECEIPT_SCHEMA = "12-6.d03-rada-global-dedup-run-receipt.v1"
SURVIVOR_SCHEMA = "12-6.d03-rada-post-dedup-survivors.v1"
TWO_CLEAN_SCHEMA = "12-6.d03-rada-two-clean-survivor-authority.v1"
SELECTION_RULE = "largest_declared_capacity_then_lexicographically_smallest_source_id"

_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "local_free_only",
        "completed",
        "dependency_authority_raw_sha256",
        "inventory_raw_sha256",
        "base_payload_map_raw_sha256",
        "source_object_count",
        "source_payload_utf8_bytes",
        "work_limits",
        "v3_report_sha256",
        "survivor_authority_sha256",
        "survivor_source_object_count",
        "survivor_declared_capacity_bytes",
        "match_wall_clock_seconds",
        "total_wall_clock_seconds",
        "source_objects_per_match_second",
        "payload_bytes_per_match_second",
        "process_max_rss_kib",
        "resource_measurement_complete",
        "raw_text_emitted",
        "canonical_capacity_credited",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "receipt_identity_sha256",
    }
)

_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "local_free_only",
        "current_main_consumable_rada_intake",
        "rada_intake_pr",
        "rada_intake_head_sha",
        "rada_intake_audit_ref",
        "rada_adapter_module",
        "rada_adapter_git_blob_sha1",
        "matcher_pr",
        "matcher_head_sha",
        "matcher_audit_ref",
        "matcher_different_worker_pass",
        "indexed_module",
        "indexed_module_git_blob_sha1",
        "v3_module",
        "incumbent_base_authority_ref",
        "combined_inventory_sha256",
        "incumbent_base_payload_map_sha256",
        "max_candidate_pairs",
        "max_index_postings",
        "max_pair_expansions",
        "worker_timeout_seconds",
        "canonical_capacity_credit",
        "authorized_optimized_target_exposure",
        "training_executed",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    }
)
_HEX = frozenset("0123456789abcdef")


class RadaTwoCleanExecutionError(RuntimeError):
    """Raised when two-clean execution cannot remain authority-safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaTwoCleanExecutionError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_hex(value: Any, length: int, label: str) -> str:
    _require(type(value) is str, f"{label} must be text")
    _require(
        len(value) == length
        and value == value.lower()
        and all(character in _HEX for character in value),
        f"{label} must be exact lowercase {length}-hex",
    )
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive exact int")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RadaTwoCleanExecutionError(f"{label} contains non-finite JSON: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadaTwoCleanExecutionError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be an exact object")
    return value


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected = _require_hex(expected_sha256, 64, f"{label} expected SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RadaTwoCleanExecutionError(f"cannot read {label}: {path}") from exc
    _require(_sha256(raw) == expected, f"{label} raw SHA-256 drift")
    return _strict_json_bytes(raw, label)


def _git_blob_sha1(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RadaTwoCleanExecutionError(f"cannot read module bytes: {path}") from exc
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()


def _module_blob_sha1(module: Any, label: str) -> str:
    raw = getattr(module, "__file__", None)
    _require(type(raw) is str and bool(raw), f"{label} module has no source path")
    path = Path(raw)
    _require(path.suffix == ".py", f"{label} authority must resolve to Python source")
    return _git_blob_sha1(path)


def validate_dependency_authority(
    path: Path,
    *,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    """Validate externally rooted terminal dependency authority."""
    authority = _read_exact_json(path, expected_raw_sha256, "dependency authority")
    _require(set(authority) == _AUTHORITY_KEYS, "dependency authority schema drift")
    _require(authority["schema_version"] == AUTHORITY_SCHEMA, "authority schema version drift")
    _require(authority["local_free_only"] is True, "LOCAL_FREE boundary weakened")
    _require(
        authority["current_main_consumable_rada_intake"] is True,
        "Rada intake is not current-main consumable",
    )
    _exact_positive_int(authority["rada_intake_pr"], "rada_intake_pr")
    _require_hex(authority["rada_intake_head_sha"], 40, "rada_intake_head_sha")
    _require(
        type(authority["rada_intake_audit_ref"]) is str
        and bool(authority["rada_intake_audit_ref"]),
        "rada_intake_audit_ref missing",
    )
    _require(
        type(authority["rada_adapter_module"]) is str
        and bool(authority["rada_adapter_module"]),
        "rada_adapter_module missing",
    )
    _require_hex(
        authority["rada_adapter_git_blob_sha1"],
        40,
        "rada_adapter_git_blob_sha1",
    )
    _exact_positive_int(authority["matcher_pr"], "matcher_pr")
    _require_hex(authority["matcher_head_sha"], 40, "matcher_head_sha")
    _require(
        type(authority["matcher_audit_ref"]) is str
        and bool(authority["matcher_audit_ref"]),
        "matcher_audit_ref missing",
    )
    _require(
        authority["matcher_different_worker_pass"] is True,
        "matcher lacks terminal different-worker PASS",
    )
    _require(
        type(authority["indexed_module"]) is str and bool(authority["indexed_module"]),
        "indexed_module missing",
    )
    _require_hex(
        authority["indexed_module_git_blob_sha1"],
        40,
        "indexed_module_git_blob_sha1",
    )
    _require(
        type(authority["v3_module"]) is str and bool(authority["v3_module"]),
        "v3_module missing",
    )
    _require(
        type(authority["incumbent_base_authority_ref"]) is str
        and bool(authority["incumbent_base_authority_ref"]),
        "incumbent_base_authority_ref missing",
    )
    _require_hex(
        authority["combined_inventory_sha256"],
        64,
        "combined_inventory_sha256",
    )
    _require_hex(
        authority["incumbent_base_payload_map_sha256"],
        64,
        "incumbent_base_payload_map_sha256",
    )
    _exact_positive_int(authority["max_candidate_pairs"], "max_candidate_pairs")
    _exact_positive_int(authority["max_index_postings"], "max_index_postings")
    _exact_positive_int(authority["max_pair_expansions"], "max_pair_expansions")
    _exact_positive_int(authority["worker_timeout_seconds"], "worker_timeout_seconds")
    _require(
        authority["canonical_capacity_credit"] == 0
        and type(authority["canonical_capacity_credit"]) is int,
        "dependency authority may not grant capacity credit",
    )
    _require(
        authority["authorized_optimized_target_exposure"] == 0
        and type(authority["authorized_optimized_target_exposure"]) is int,
        "dependency authority may not grant optimized-target exposure",
    )
    for key in (
        "training_executed",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        _require(authority[key] is False, f"dependency authority truth drift: {key}")
    return authority


def _windows_peak_working_set_kib() -> int | None:
    """Return current-process peak working set in KiB via the Windows API."""

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    windll = getattr(ctypes, "windll", None)
    kernel32 = getattr(windll, "kernel32", None)
    if kernel32 is None:
        return None
    query = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if query is None:
        psapi = getattr(windll, "psapi", None)
        query = getattr(psapi, "GetProcessMemoryInfo", None)
    get_current_process = getattr(kernel32, "GetCurrentProcess", None)
    if query is None or get_current_process is None:
        return None

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        handle = get_current_process()
        ok = query(handle, ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if not ok:
        return None
    peak_bytes = int(counters.PeakWorkingSetSize)
    if peak_bytes <= 0:
        return None
    return (peak_bytes + 1023) // 1024


def _max_rss_kib() -> int | None:
    if resource is None:
        return _windows_peak_working_set_kib()
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value // 1024 if value > 10_000_000 else value


def _load_base_payloads(
    combined_inventory: Mapping[str, Any],
    rada_sources: Sequence[Mapping[str, Any]],
    rada_payloads: Mapping[str, bytes],
    base_payload_map: Mapping[str, Any],
    *,
    rada_source_family: str,
) -> dict[str, bytes]:
    """Bind exact incumbent-base files to a combined inventory plus exact Rada rows."""
    rows = combined_inventory.get("sources")
    _require(type(rows) is list and bool(rows), "combined V3 inventory has no sources")
    _require(
        type(rada_source_family) is str and bool(rada_source_family),
        "Rada source family missing",
    )

    rada_ids = {row.get("source_id") for row in rada_sources}
    _require(
        len(rada_ids) == len(rada_sources)
        and all(type(source_id) is str and bool(source_id) for source_id in rada_ids),
        "Rada projection contains invalid or duplicate source ids",
    )
    _require(set(rada_payloads) == rada_ids, "Rada projection payload coverage drift")

    observed_rada_rows = [
        row
        for row in rows
        if type(row) is dict and row.get("source_family") == rada_source_family
    ]
    _require(
        observed_rada_rows == list(rada_sources),
        "combined inventory Rada segment does not equal exact authenticated projection",
    )

    all_ids: list[str] = []
    seen_ids: set[str] = set()
    base_ids: set[str] = set()
    for index, row in enumerate(rows):
        _require(type(row) is dict, f"combined inventory source {index} must be exact object")
        source_id = row.get("source_id")
        _require(type(source_id) is str and bool(source_id), "combined inventory source_id invalid")
        _require(source_id not in seen_ids, "combined inventory contains duplicate source_id")
        all_ids.append(source_id)
        seen_ids.add(source_id)
        if source_id not in rada_ids:
            base_ids.add(source_id)
    _require(bool(base_ids), "combined inventory lacks incumbent base sources")
    _require(
        set(base_payload_map) == base_ids,
        "base payload map coverage differs from incumbent base inventory",
    )

    payloads: dict[str, bytes] = {}
    for source_id in sorted(base_ids):
        raw_path = base_payload_map[source_id]
        _require(
            type(raw_path) is str and bool(raw_path),
            f"base payload path invalid: {source_id}",
        )
        try:
            payload = Path(raw_path).read_bytes()
        except OSError as exc:
            raise RadaTwoCleanExecutionError(
                f"cannot read incumbent base payload: {source_id}"
            ) from exc
        payloads[source_id] = payload

    payloads.update(rada_payloads)
    _require(set(payloads) == set(all_ids), "combined payload coverage drift")
    return payloads


def derive_survivor_authority(report: Mapping[str, Any]) -> dict[str, Any]:
    """Make the incumbent V3 capacity representative set explicit, without raw text."""
    report_sha = _require_hex(report.get("report_sha256"), 64, "report_sha256")
    rows = report.get("sources")
    _require(type(rows) is list and bool(rows), "V3 report has no source rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        _require(type(raw) is dict, "V3 source row must be exact object")
        source_id = raw.get("source_id")
        _require(type(source_id) is str and bool(source_id), "invalid V3 source_id")
        _require(source_id not in by_id, "duplicate V3 source_id")
        capacity = raw.get("declared_capacity_bytes")
        _require(type(capacity) is int and capacity > 0, "invalid declared capacity")
        for key in (
            "source_family",
            "modality",
            "verified_raw_sha256",
            "normalized_sha256",
            "stable_origin_id_sha256",
            "stable_object_id_sha256",
        ):
            _require(type(raw.get(key)) is str and bool(raw.get(key)), f"missing {key}")
        by_id[source_id] = raw

    terminal = report.get("terminal_candidates")
    _require(type(terminal) is dict, "V3 terminal_candidates missing")
    clusters = terminal.get("duplicate_clusters")
    _require(type(clusters) is list, "V3 duplicate_clusters missing")

    seen_members: set[str] = set()
    dropped: set[str] = set()
    normalized_clusters: list[dict[str, Any]] = []
    for raw_cluster in clusters:
        _require(
            isinstance(raw_cluster, Sequence) and not isinstance(raw_cluster, (str, bytes)),
            "invalid duplicate cluster",
        )
        cluster = sorted(str(value) for value in raw_cluster)
        _require(len(cluster) >= 2 and len(cluster) == len(set(cluster)), "invalid cluster")
        _require(all(source_id in by_id for source_id in cluster), "cluster has unknown source")
        _require(not (seen_members & set(cluster)), "duplicate clusters overlap")
        seen_members.update(cluster)
        maximum = max(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in cluster)
        survivor = min(
            source_id
            for source_id in cluster
            if int(by_id[source_id]["declared_capacity_bytes"]) == maximum
        )
        dropped.update(source_id for source_id in cluster if source_id != survivor)
        normalized_clusters.append(
            {
                "member_source_ids": cluster,
                "selected_source_id": survivor,
                "selected_declared_capacity_bytes": maximum,
            }
        )
    normalized_clusters.sort(key=lambda item: tuple(item["member_source_ids"]))

    survivor_ids = sorted(source_id for source_id in by_id if source_id not in dropped)
    survivors = [
        {
            "source_id": source_id,
            "source_family": by_id[source_id]["source_family"],
            "modality": by_id[source_id]["modality"],
            "declared_capacity_bytes": by_id[source_id]["declared_capacity_bytes"],
            "verified_raw_sha256": by_id[source_id]["verified_raw_sha256"],
            "normalized_sha256": by_id[source_id]["normalized_sha256"],
            "stable_origin_id_sha256": by_id[source_id]["stable_origin_id_sha256"],
            "stable_object_id_sha256": by_id[source_id]["stable_object_id_sha256"],
        }
        for source_id in survivor_ids
    ]
    survivor_bytes = sum(int(row["declared_capacity_bytes"]) for row in survivors)
    post = terminal.get("conservative_unique_capacity_bytes_after")
    _require(
        type(post) is int and survivor_bytes == post,
        "survivor bytes do not reproduce incumbent V3 capacity",
    )
    before = terminal.get("declared_capacity_bytes_before")
    discount = terminal.get("duplicate_discount_bytes")
    _require(
        type(before) is int
        and type(discount) is int
        and before - survivor_bytes == discount,
        "survivor discount does not reproduce incumbent V3 summary",
    )

    core: dict[str, Any] = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "v3_report_sha256": report_sha,
        "pre_dedup_source_object_count": len(by_id),
        "post_dedup_survivor_source_object_count": len(survivors),
        "pre_dedup_declared_capacity_bytes": before,
        "post_dedup_declared_capacity_bytes": survivor_bytes,
        "duplicate_discount_bytes": discount,
        "duplicate_cluster_count": len(normalized_clusters),
        "duplicate_clusters": normalized_clusters,
        "survivors": survivors,
        "raw_text_emitted": False,
        "truth_boundary": {
            "source_object_authority_only": True,
            "canonical_capacity_credited": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    core["survivor_authority_sha256"] = _sha256(_canonical_bytes(core))
    return core


def execute_once(
    *,
    run_id: str,
    dependency_authority_path: Path,
    expected_dependency_authority_sha256: str,
    inventory_path: Path,
    expected_inventory_sha256: str,
    base_payload_map_path: Path,
    expected_base_payload_map_sha256: str,
    candidate_jsonl: Path,
    quality_report: Path,
    execution_evidence: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute one clean, authority-bound Rada indexed-dedup process."""
    _require(type(run_id) is str and bool(run_id), "run_id must be non-empty text")
    authority = validate_dependency_authority(
        dependency_authority_path,
        expected_raw_sha256=expected_dependency_authority_sha256,
    )
    _require(
        expected_inventory_sha256 == authority["combined_inventory_sha256"],
        "combined inventory SHA differs from dependency authority",
    )
    _require(
        expected_base_payload_map_sha256
        == authority["incumbent_base_payload_map_sha256"],
        "base payload map SHA differs from dependency authority",
    )
    inventory = _read_exact_json(inventory_path, expected_inventory_sha256, "V3 inventory")
    base_payload_map = _read_exact_json(
        base_payload_map_path,
        expected_base_payload_map_sha256,
        "incumbent base payload map",
    )

    adapter = importlib.import_module(authority["rada_adapter_module"])
    indexed = importlib.import_module(authority["indexed_module"])
    v3 = importlib.import_module(authority["v3_module"])

    _require(
        _module_blob_sha1(adapter, "Rada adapter")
        == authority["rada_adapter_git_blob_sha1"],
        "Rada adapter Git blob drift",
    )
    _require(
        _module_blob_sha1(indexed, "indexed executor")
        == authority["indexed_module_git_blob_sha1"],
        "indexed executor Git blob drift",
    )
    attest = getattr(indexed, "attest_incumbent_runtime", None)
    _require(callable(attest), "indexed executor lacks runtime attestation")
    attest(v3)

    project = getattr(adapter, "validate_and_project_rada_laws_qp", None)
    _require(callable(project), "Rada adapter lacks projection entrypoint")
    upstream_head = getattr(adapter, "UPSTREAM_FINAL_EVIDENCE_COMMIT", None)
    _require(type(upstream_head) is str and bool(upstream_head), "Rada upstream authority missing")

    total_started = time.perf_counter()
    projection = project(
        candidate_jsonl,
        quality_report,
        execution_evidence,
        upstream_head=upstream_head,
        retain_payloads=True,
    )
    sources = getattr(projection, "sources", None)
    rada_payloads = getattr(projection, "payloads", None)
    _require(type(sources) is tuple and bool(sources), "Rada projection sources missing")
    _require(
        type(rada_payloads) is dict and bool(rada_payloads),
        "Rada projection payloads missing",
    )
    rada_source_family = getattr(adapter, "SOURCE_FAMILY", None)
    payloads = _load_base_payloads(
        inventory,
        sources,
        rada_payloads,
        base_payload_map,
        rada_source_family=rada_source_family,
    )

    execute = getattr(indexed, "audit_payloads_indexed", None)
    _require(callable(execute), "indexed executor entrypoint missing")
    match_started = time.perf_counter()
    report = execute(
        v3,
        inventory,
        payloads,
        max_candidate_pairs=authority["max_candidate_pairs"],
        max_index_postings=authority["max_index_postings"],
        max_pair_expansions=authority["max_pair_expansions"],
    )
    match_seconds = time.perf_counter() - match_started
    verify = getattr(v3, "verify_report", None)
    _require(callable(verify), "incumbent V3 verifier missing")
    verify(report)
    survivor = derive_survivor_authority(report)
    total_seconds = time.perf_counter() - total_started

    inventory_sources = inventory.get("sources")
    assert isinstance(inventory_sources, list)
    source_count = len(inventory_sources)
    source_bytes = sum(len(value) for value in payloads.values())
    rss = _max_rss_kib()
    receipt_core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "run_id": run_id,
        "local_free_only": True,
        "completed": True,
        "dependency_authority_raw_sha256": expected_dependency_authority_sha256,
        "inventory_raw_sha256": expected_inventory_sha256,
        "base_payload_map_raw_sha256": expected_base_payload_map_sha256,
        "source_object_count": source_count,
        "source_payload_utf8_bytes": source_bytes,
        "work_limits": {
            "max_candidate_pairs": authority["max_candidate_pairs"],
            "max_index_postings": authority["max_index_postings"],
            "max_pair_expansions": authority["max_pair_expansions"],
        },
        "v3_report_sha256": report["report_sha256"],
        "survivor_authority_sha256": survivor["survivor_authority_sha256"],
        "survivor_source_object_count": survivor["post_dedup_survivor_source_object_count"],
        "survivor_declared_capacity_bytes": survivor["post_dedup_declared_capacity_bytes"],
        "match_wall_clock_seconds": round(match_seconds, 6),
        "total_wall_clock_seconds": round(total_seconds, 6),
        "source_objects_per_match_second": (
            round(source_count / match_seconds, 6) if match_seconds > 0 else None
        ),
        "payload_bytes_per_match_second": (
            round(source_bytes / match_seconds, 6) if match_seconds > 0 else None
        ),
        "process_max_rss_kib": rss,
        "resource_measurement_complete": rss is not None,
        "raw_text_emitted": False,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": _sha256(_canonical_bytes(receipt_core)),
    }
    return report, survivor, receipt


def build_two_clean_authority(
    first_receipt: Mapping[str, Any],
    second_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind two fresh completed run receipts into one zero-credit survivor authority."""
    _require(first_receipt.get("schema_version") == RECEIPT_SCHEMA, "first receipt schema drift")
    _require(second_receipt.get("schema_version") == RECEIPT_SCHEMA, "second receipt schema drift")
    _require(first_receipt.get("completed") is True, "first run incomplete")
    _require(second_receipt.get("completed") is True, "second run incomplete")
    _require(
        first_receipt.get("run_id") != second_receipt.get("run_id"),
        "two-clean run ids must differ",
    )
    for receipt, label in ((first_receipt, "first"), (second_receipt, "second")):
        _require(set(receipt) == _RECEIPT_KEYS, f"{label} receipt schema drift")
        _require(
            type(receipt.get("run_id")) is str and bool(receipt.get("run_id")),
            f"{label} receipt run_id invalid",
        )
        identity = _require_hex(
            receipt.get("receipt_identity_sha256"),
            64,
            f"{label} receipt identity",
        )
        unsigned = dict(receipt)
        unsigned.pop("receipt_identity_sha256", None)
        _require(
            _sha256(_canonical_bytes(unsigned)) == identity,
            f"{label} receipt self-hash mismatch",
        )
        _require(receipt.get("local_free_only") is True, f"{label} receipt is not LOCAL_FREE")
        _exact_positive_int(receipt.get("source_object_count"), f"{label} source_object_count")
        _exact_positive_int(
            receipt.get("source_payload_utf8_bytes"),
            f"{label} source_payload_utf8_bytes",
        )
        _exact_positive_int(
            receipt.get("survivor_source_object_count"),
            f"{label} survivor_source_object_count",
        )
        _exact_positive_int(
            receipt.get("survivor_declared_capacity_bytes"),
            f"{label} survivor_declared_capacity_bytes",
        )
        for hash_key in (
            "dependency_authority_raw_sha256",
            "inventory_raw_sha256",
            "base_payload_map_raw_sha256",
            "v3_report_sha256",
            "survivor_authority_sha256",
        ):
            _require_hex(receipt.get(hash_key), 64, f"{label} {hash_key}")
        limits = receipt.get("work_limits")
        _require(type(limits) is dict, f"{label} work_limits missing")
        _require(
            set(limits)
            == {"max_candidate_pairs", "max_index_postings", "max_pair_expansions"},
            f"{label} work_limits schema drift",
        )
        for key, value in limits.items():
            _exact_positive_int(value, f"{label} {key}")
        for metric in (
            "match_wall_clock_seconds",
            "total_wall_clock_seconds",
            "source_objects_per_match_second",
            "payload_bytes_per_match_second",
        ):
            value = receipt.get(metric)
            _require(
                type(value) in {int, float}
                and math.isfinite(value)
                and value > 0,
                f"{label} receipt invalid telemetry: {metric}",
            )
        _require(receipt.get("raw_text_emitted") is False, f"{label} receipt emitted raw text")
        _require(
            receipt.get("resource_measurement_complete") is True
            and type(receipt.get("process_max_rss_kib")) is int
            and receipt.get("process_max_rss_kib") > 0,
            f"{label} receipt lacks terminal max-RSS evidence",
        )
        _require(
            receipt["survivor_source_object_count"] <= receipt["source_object_count"],
            f"{label} survivor source count exceeds input",
        )
        _require(
            receipt["survivor_declared_capacity_bytes"]
            <= receipt["source_payload_utf8_bytes"],
            f"{label} survivor bytes exceed input",
        )
        _require(
            receipt["total_wall_clock_seconds"] >= receipt["match_wall_clock_seconds"],
            f"{label} total wall clock is smaller than match wall clock",
        )
        _require(
            receipt.get("canonical_capacity_credited") == 0
            and type(receipt.get("canonical_capacity_credited")) is int
            and receipt.get("authorized_optimized_target_exposure") == 0
            and type(receipt.get("authorized_optimized_target_exposure")) is int,
            f"{label} receipt widened scientific authority",
        )
        for key in (
            "tokenizer_fit_authorized",
            "training_executed",
            "learned_weights_created",
            "final_test_outcomes_read",
            "paid_compute_used",
        ):
            _require(receipt.get(key) is False, f"{label} receipt truth drift: {key}")

    equality_fields = (
        "dependency_authority_raw_sha256",
        "inventory_raw_sha256",
        "base_payload_map_raw_sha256",
        "source_object_count",
        "source_payload_utf8_bytes",
        "work_limits",
        "v3_report_sha256",
        "survivor_authority_sha256",
        "survivor_source_object_count",
        "survivor_declared_capacity_bytes",
    )
    for field in equality_fields:
        _require(
            first_receipt.get(field) == second_receipt.get(field),
            f"two-clean reproducibility drift: {field}",
        )

    core: dict[str, Any] = {
        "schema_version": TWO_CLEAN_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "two_fresh_processes_required": True,
        "run_receipt_identities": [
            first_receipt["receipt_identity_sha256"],
            second_receipt["receipt_identity_sha256"],
        ],
        "dependency_authority_raw_sha256": first_receipt[
            "dependency_authority_raw_sha256"
        ],
        "inventory_raw_sha256": first_receipt["inventory_raw_sha256"],
        "base_payload_map_raw_sha256": first_receipt["base_payload_map_raw_sha256"],
        "v3_report_sha256": first_receipt["v3_report_sha256"],
        "survivor_authority_sha256": first_receipt["survivor_authority_sha256"],
        "source_object_count": first_receipt["source_object_count"],
        "source_payload_utf8_bytes": first_receipt["source_payload_utf8_bytes"],
        "work_limits": first_receipt["work_limits"],
        "survivor_source_object_count": first_receipt["survivor_source_object_count"],
        "survivor_declared_capacity_bytes": first_receipt[
            "survivor_declared_capacity_bytes"
        ],
        "resource_evidence": [
            {
                "run_id": first_receipt["run_id"],
                "match_wall_clock_seconds": first_receipt["match_wall_clock_seconds"],
                "total_wall_clock_seconds": first_receipt["total_wall_clock_seconds"],
                "process_max_rss_kib": first_receipt["process_max_rss_kib"],
                "resource_measurement_complete": first_receipt[
                    "resource_measurement_complete"
                ],
            },
            {
                "run_id": second_receipt["run_id"],
                "match_wall_clock_seconds": second_receipt["match_wall_clock_seconds"],
                "total_wall_clock_seconds": second_receipt["total_wall_clock_seconds"],
                "process_max_rss_kib": second_receipt["process_max_rss_kib"],
                "resource_measurement_complete": second_receipt[
                    "resource_measurement_complete"
                ],
            },
        ],
        "raw_text_emitted": False,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    core["two_clean_authority_sha256"] = _sha256(_canonical_bytes(core))
    return core
