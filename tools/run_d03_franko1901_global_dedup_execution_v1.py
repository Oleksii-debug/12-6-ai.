#!/usr/bin/env python3
"""Execute exact source-admitted Franko1901 through the incumbent global-dedup authority.

This runner is execution glue only.  It composes the exact merged Franko1901 intake
with the exact reconstructed V8 graph and runs the merged, independently qualified
performance-equivalent indexed executor under incumbent V3 verification.  It emits
only text-free zero-credit evidence.
"""
from __future__ import annotations

import argparse
import ctypes
import copy
import hashlib
import importlib
import json
try:
    import resource
except ImportError:  # pragma: no cover - Windows/local fallback
    resource = None
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_expanded_global_dedup_v9 as v9_runner
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import franko1901_dedup_intake as franko1901
from twelve_six.data import incumbent_dedup_indexed_execution as indexed

SCHEMA = "12-6.d03-franko1901-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-franko1901-global-dedup-survivors.v1"
EXPECTED_MAIN = "ba9e49cedba4a110e1c4f7d83702e8fcf8a42461"
EXECUTION_CLAIM = 2394
EXECUTION_PR = 2448
CARRIER_PATH = "tools/run_d03_franko1901_global_dedup_execution_v1.py"
FRANKO1901_FINAL_HEAD = franko1901.UPSTREAM_PRODUCT_HEAD
EXPECTED_BASE_OBJECTS = 264
EXPECTED_BASE_BYTES = 6_095_624
EXPECTED_FRANKO1901_OBJECTS = franko1901.CANDIDATE_RECORDS
EXPECTED_FRANKO1901_BYTES = franko1901.CANDIDATE_TEXT_UTF8_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_FRANKO1901_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_FRANKO1901_BYTES

AUTHORITY_PATHS = (
    "src/twelve_six/data/franko1901_dedup_intake.py",
    "configs/data/d03_franko1901_fresh_execution_authority_v1.json",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "tools/run_d03_expanded_global_dedup_v9.py",
    "tools/run_next100_065f_global_dedup_v8.py",
    "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
    "configs/data/next100_065f_global_dedup_v8.json",
    "configs/data/data_bulk_code1_permissive_python_bundle_v1.json",
    "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json",
)


class Franko1901GlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Franko1901GlobalDedupError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Franko1901GlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise Franko1901GlobalDedupError(f"git {' '.join(args)} failed: {detail}")
    return proc


def verify_repository_authority() -> dict[str, str]:
    """Bind authority-bearing runtime files to exact committed and worktree bytes."""
    ancestor = _git("merge-base", "--is-ancestor", EXPECTED_MAIN, "HEAD", check=False)
    _require(
        ancestor.returncode == 0,
        f"pinned main {EXPECTED_MAIN} is not an ancestor of execution HEAD",
    )
    worktree = _git("diff", "--quiet", "HEAD", "--", *AUTHORITY_PATHS, check=False)
    _require(worktree.returncode == 0, "authority path worktree drift")
    blobs: dict[str, str] = {}
    for path in AUTHORITY_PATHS:
        expected = _git("rev-parse", f"{EXPECTED_MAIN}:{path}").stdout.strip()
        observed = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        _require(bool(expected) and observed == expected, f"authority path drift: {path}")
        blobs[path] = observed
    return blobs


def verify_execution_carrier() -> str:
    """Bind the executing carrier bytes to the exact recorded HEAD."""
    expected = _git("rev-parse", f"HEAD:{CARRIER_PATH}").stdout.strip()
    observed = _git("hash-object", str(ROOT / CARRIER_PATH)).stdout.strip()
    _require(
        len(expected) == 40 and observed == expected,
        "execution carrier worktree drift",
    )
    return observed


def _bind_execution_head(expected_execution_head: str) -> str:
    """Require physical execution on the explicitly selected exact Product head."""
    _require(
        type(expected_execution_head) is str
        and len(expected_execution_head) == 40
        and all(char in "0123456789abcdef" for char in expected_execution_head),
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    observed = _git("rev-parse", "HEAD").stdout.strip()
    _require(
        observed == expected_execution_head,
        "execution HEAD drift: refusing synthetic/stale/unselected checkout",
    )
    return observed


_HISTORICAL_MATCHER_MODULES = (
    # V5 imports pipeline during exact V7 graph reconstruction. Current main no
    # longer carries this path, so require the module to originate from V7 too.
    "twelve_six.data.pipeline",
    "twelve_six.data._data232_decontamination_matching",
    "twelve_six.data.cross_source_capacity_audit",
    "twelve_six.data.cross_source_capacity_audit_v3",
    "twelve_six.data.cross_source_capacity_audit_v4",
    "twelve_six.data.cross_source_capacity_audit_v5",
    "twelve_six.data.cross_source_capacity_audit_v6",
    "twelve_six.data.cross_source_capacity_audit_v7",
)


def _reconstruct_v8_with_historical_namespace(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    """Load exact V7 matcher modules ahead of the cached current-main package paths."""
    for module_name in _HISTORICAL_MATCHER_MODULES:
        _require(module_name not in sys.modules, f"historical matcher preloaded: {module_name}")

    twelve_six_pkg = importlib.import_module("twelve_six")
    data_pkg = importlib.import_module("twelve_six.data")
    current_package_path = list(twelve_six_pkg.__path__)
    current_data_path = list(data_pkg.__path__)
    historical_package = str((v7_root / "src" / "twelve_six").resolve(strict=True))
    historical_data = str(
        (v7_root / "src" / "twelve_six" / "data").resolve(strict=True)
    )
    _require(
        historical_package not in current_package_path
        and historical_data not in current_data_path,
        "historical V7 package path already injected",
    )

    twelve_six_pkg.__path__ = [historical_package, *current_package_path]
    data_pkg.__path__ = [historical_data, *current_data_path]
    importlib.invalidate_caches()
    try:
        matcher, inventory, payloads = v9_runner.reconstruct_v8_source_inputs(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            v8_config=dict(config),
        )
    finally:
        twelve_six_pkg.__path__ = current_package_path
        data_pkg.__path__ = current_data_path
        importlib.invalidate_caches()

    historical_root = Path(historical_data)
    for module_name in _HISTORICAL_MATCHER_MODULES:
        module = sys.modules.get(module_name)
        _require(module is not None, f"historical matcher failed to load: {module_name}")
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        _require(
            module_path.is_relative_to(historical_root),
            f"historical matcher escaped exact V7 worktree: {module_name}",
        )
    return matcher, inventory, payloads


def _compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _require(type(base_inventory) is dict, "base inventory must be exact dict")
    _require(
        all(type(key) is str and type(value) is bytes for key, value in base_payloads.items()),
        "base payload map must be exact str->bytes",
    )
    _require(
        all(type(key) is str and type(value) is bytes for key, value in extension_payloads.items()),
        "extension payload map must be exact str->bytes",
    )
    rows = base_inventory.get("sources")
    _require(isinstance(rows, list) and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if isinstance(row, Mapping)}
    _require(len(base_ids) == len(rows) and None not in base_ids, "base source ids invalid")
    _require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    extension_ids = {
        row.get("source_id") for row in extension_sources if isinstance(row, Mapping)
    }
    _require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "extension source ids invalid",
    )
    _require(extension_ids == set(extension_payloads), "extension inventory/payload mismatch")
    _require(not (base_ids & extension_ids), "Franko1901 source-id collision with incumbent graph")

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed V8 authority plus exact PR #1025 source-admitted Franko1901 "
        "rows; pair decisions delegate to terminal PR #824 V3 semantics through the "
        "independently qualified performance-equivalent indexed executor."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def _outer_survivor_authority(
    dedup_report: Mapping[str, Any],
    selection_projection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = dedup_report.get("sources")
    _require(type(sources) is list, "dedup source vector missing")
    _require(
        all(type(row) is dict and type(row.get("source_id")) is str for row in sources),
        "dedup source row invalid",
    )
    source_ids = [row["source_id"] for row in sources]
    _require(len(set(source_ids)) == len(source_ids), "duplicate dedup source id")
    by_id = {row["source_id"]: row for row in sources}
    survivor_ids = selection_projection.get("survivor_source_ids")
    _require(type(survivor_ids) is list, "selection projection survivor ids missing")
    _require(
        all(type(source_id) is str for source_id in survivor_ids),
        "survivor source id invalid",
    )
    _require(len(set(survivor_ids)) == len(survivor_ids), "duplicate survivor source id")
    _require(all(source_id in by_id for source_id in survivor_ids), "unknown survivor id")
    franko1901_survivors = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == franko1901.SOURCE_FAMILY
    ]
    franko1901_survivor_bytes = 0
    for source_id in franko1901_survivors:
        declared_capacity = by_id[source_id].get("declared_capacity_bytes")
        _require(
            type(declared_capacity) is int and declared_capacity >= 0,
            "Franko1901 survivor declared capacity must be exact nonnegative int",
        )
        franko1901_survivor_bytes += declared_capacity
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_projection_schema": selection_projection.get("schema_version"),
        "selection_projection_sha256": selection_projection.get(
            "survivor_authority_sha256"
        ),
        "matcher_report_sha256": dedup_report.get("report_sha256"),
        "pre_dedup_source_object_count": selection_projection.get(
            "pre_dedup_source_object_count"
        ),
        "post_dedup_survivor_source_object_count": selection_projection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "pre_dedup_declared_capacity_bytes": selection_projection.get(
            "pre_dedup_declared_capacity_bytes"
        ),
        "post_dedup_declared_capacity_bytes": selection_projection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": selection_projection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": selection_projection.get("duplicate_cluster_count"),
        "duplicate_clusters": copy.deepcopy(selection_projection.get("duplicate_clusters")),
        "survivor_source_ids": copy.deepcopy(survivor_ids),
        "franko1901": {
            "source_family": franko1901.SOURCE_FAMILY,
            "pre_dedup_source_object_count": EXPECTED_FRANKO1901_OBJECTS,
            "pre_dedup_payload_bytes": EXPECTED_FRANKO1901_BYTES,
            "post_dedup_survivor_source_object_count": len(franko1901_survivors),
            "post_dedup_survivor_declared_capacity_bytes": franko1901_survivor_bytes,
        },
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "two_clean_builds_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def _validated_terminal_summary(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        type(report.get("source_count")) is int
        and report.get("source_count") == EXPECTED_COMBINED_OBJECTS,
        "combined report source count drift",
    )
    terminal = report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal dedup summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    cluster_count = terminal.get("duplicate_cluster_count")
    _require(
        type(before) is int and before == EXPECTED_COMBINED_BYTES,
        "combined declared capacity drift",
    )
    _require(
        type(after) is int and 0 < after <= before,
        "post-dedup conservative capacity invalid",
    )
    _require(
        type(discount) is int and discount >= 0 and before - after == discount,
        "duplicate discount arithmetic drift",
    )
    _require(
        type(cluster_count) is int and cluster_count >= 0,
        "duplicate cluster count invalid",
    )
    return terminal


def _validate_survivor_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> None:
    """Bind the derived survivor projection back to the verified terminal report."""
    terminal = _validated_terminal_summary(report)
    _require(type(projection) is dict, "survivor projection must be exact object")
    _require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    _require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "survivor projection matcher identity drift",
    )
    survivor_ids = projection.get("survivor_source_ids")
    _require(
        type(survivor_ids) is list
        and all(type(source_id) is str and source_id for source_id in survivor_ids)
        and len(survivor_ids) == len(set(survivor_ids)),
        "survivor projection ids invalid",
    )
    post_count = projection.get("post_dedup_survivor_source_object_count")
    _require(
        type(post_count) is int
        and 0 < post_count <= EXPECTED_COMBINED_OBJECTS
        and post_count == len(survivor_ids),
        "survivor projection post-dedup count drift",
    )
    _require(
        projection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "survivor projection pre-dedup count drift",
    )
    _require(
        projection.get("pre_dedup_declared_capacity_bytes") == EXPECTED_COMBINED_BYTES,
        "survivor projection pre-dedup bytes drift",
    )
    _require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "survivor projection post-dedup bytes drift",
    )
    _require(
        projection.get("duplicate_discount_bytes")
        == terminal.get("duplicate_discount_bytes"),
        "survivor projection duplicate discount drift",
    )
    duplicate_clusters = projection.get("duplicate_clusters")
    duplicate_cluster_count = projection.get("duplicate_cluster_count")
    _require(type(duplicate_clusters) is list, "survivor projection clusters missing")
    _require(
        type(duplicate_cluster_count) is int
        and duplicate_cluster_count == len(duplicate_clusters)
        and duplicate_cluster_count == terminal.get("duplicate_cluster_count"),
        "survivor projection cluster count drift",
    )


def _source_admission_provenance_scope(receipt: Mapping[str, Any]) -> dict[str, bool]:
    truth = receipt.get("truth_boundary")
    _require(type(truth) is dict, "source-admission truth boundary missing")
    _require(
        truth.get("external_llm_or_api_used_for_data_or_intelligence") is False,
        "Franko source-package external-LLM evidence drift",
    )
    return {
        "upstream_source_package_external_llm_or_api_used": False,
        "current_corpus_external_llm_free_claimed_by_this_carrier": False,
        "source_local_negative_promoted_as_corpus_global_truth": False,
    }


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
        get_current_process.restype = ctypes.c_void_p
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
    except (AttributeError, TypeError):
        pass
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
    if sys.platform == "win32":
        return _windows_peak_working_set_kib()
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if sys.platform.startswith("linux"):
        return value
    if sys.platform == "darwin":
        return (value + 1023) // 1024
    return None


def _publish_json_outputs(
    outputs: tuple[tuple[Path, Mapping[str, Any]], ...],
) -> None:
    prepared: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for path, value in outputs:
        _require(path not in seen, f"duplicate output path: {path}")
        seen.add(path)
        prepared.append((path, _canonical(dict(value)) + b"\n"))

    created: list[Path] = []
    try:
        for path, payload in prepared:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(payload)
            except FileExistsError as exc:
                raise Franko1901GlobalDedupError(
                    f"refusing to overwrite: {path}"
                ) from exc
            except OSError as exc:
                raise Franko1901GlobalDedupError(
                    f"cannot write output: {path}"
                ) from exc
    except Exception as exc:
        rollback_errors: list[str] = []
        for created_path in reversed(created):
            try:
                created_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{created_path}: {rollback_exc}")
        if rollback_errors:
            raise Franko1901GlobalDedupError(
                "output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish_json_outputs(((path, value),))


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    historical_terminal_evidence_json: Path,
    fresh_execution_authority_json: Path,
    output_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = _bind_execution_head(expected_execution_head)
    execution_carrier_blob = verify_execution_carrier()
    authority_blobs = verify_repository_authority()
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads = _reconstruct_v8_with_historical_namespace(
        v7_root=v7_root,
        bulk_workspace=bulk_workspace,
        config=config,
    )
    _require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "V8 base object count drift")
    _require(
        sum(len(raw) for raw in base_payloads.values()) == EXPECTED_BASE_BYTES,
        "V8 base payload byte total drift",
    )

    projection = franko1901.validate_and_project_franko(
        candidate_jsonl,
        historical_terminal_evidence_json,
        fresh_execution_authority_json,
        retain_payloads=True,
    )
    _require(
        projection.sources is not None and projection.payloads is not None,
        "payload projection missing",
    )
    source_admission_provenance_scope = _source_admission_provenance_scope(
        projection.receipt
    )
    extension_sources = [dict(row) for row in projection.sources]
    extension_payloads = dict(projection.payloads)
    _require(
        len(extension_sources) == EXPECTED_FRANKO1901_OBJECTS,
        "Franko1901 projection count drift",
    )
    _require(
        sum(len(raw) for raw in extension_payloads.values()) == EXPECTED_FRANKO1901_BYTES,
        "Franko1901 projection bytes drift",
    )
    inventory, payloads = _compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    _require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined source count drift")
    _require(
        sum(len(raw) for raw in payloads.values()) == EXPECTED_COMBINED_BYTES,
        "combined payload byte total drift",
    )

    # PR #1459 independently qualified this indexed executor as performance-equivalent
    # to the incumbent all-pairs authority. Production-scale source execution must not
    # reintroduce an O(N^2) reference pass; bind runtime authority, execute indexed,
    # then require the incumbent V3 verifier to accept the resulting report.
    indexed.attest_incumbent_runtime(matcher)

    indexed_started = time.perf_counter()
    indexed_report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    indexed_seconds = time.perf_counter() - indexed_started
    matcher.verify_report(indexed_report)

    terminal = _validated_terminal_summary(indexed_report)
    selection_projection = v9_semantics._derive_survivors(indexed_report)
    _validate_survivor_projection(indexed_report, selection_projection)
    survivors = _outer_survivor_authority(indexed_report, selection_projection)
    process_max_rss_kib = _max_rss_kib()
    _require(
        type(process_max_rss_kib) is int and process_max_rss_kib > 0,
        "process max RSS unavailable; refusing terminal execution evidence",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "GITHUB_HOSTED_FREE_LOCAL_FREE",
        "execution_claim_issue": EXECUTION_CLAIM,
        "execution_pr": EXECUTION_PR,
        "execution_head_sha": execution_head,
        "execution_carrier_git_blob_sha1": execution_carrier_blob,
        "pinned_main_sha": EXPECTED_MAIN,
        "authority_path_blobs": authority_blobs,
        "baseline_v8": {
            "v7_head_sha": v8.EXPECTED_V7_HEAD,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "payload_bytes": EXPECTED_BASE_BYTES,
        },
        "franko1901": {
            "source_admission_product_pr": franko1901.UPSTREAM_PRODUCT_PR,
            "source_admission_final_head": FRANKO1901_FINAL_HEAD,
            "candidate_sha256": franko1901.CANDIDATE_SHA256,
            "source_object_count": EXPECTED_FRANKO1901_OBJECTS,
            "payload_bytes": EXPECTED_FRANKO1901_BYTES,
            "intake_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
            "source_admission_provenance_scope": source_admission_provenance_scope,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": EXPECTED_COMBINED_BYTES,
            "indexed_report_sha256": indexed_report["report_sha256"],
            "indexed_executor_performance_equivalence_authority": "MERGED_PR_1459",
            "post_dedup_conservative_unique_bytes": terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
            "duplicate_discount_bytes": terminal.get("duplicate_discount_bytes"),
            "duplicate_cluster_count": terminal.get("duplicate_cluster_count"),
        },
        "indexed_execution": {
            "source_count": EXPECTED_COMBINED_OBJECTS,
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "indexed_wall_clock_seconds": round(indexed_seconds, 6),
            "process_max_rss_kib": process_max_rss_kib,
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "raw_text_emitted": False,
            "raw_candidate_written_to_durable_evidence": False,
            "payload_artifact_required": False,
            "dedup_report_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
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
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }
    _publish_json_outputs(
        (
            (output_report, indexed_report),
            (output_survivors, survivors),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--historical-terminal-evidence-json", type=Path, required=True)
    parser.add_argument("--fresh-execution-authority-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument(
        "--expected-execution-head",
        required=True,
        help="Exact 40-hex Product head; synthetic PR merge commits are rejected.",
    )
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()
    evidence = execute(
        v7_root=args.v7_root,
        bulk_workspace=args.bulk_workspace,
        candidate_jsonl=args.candidate_jsonl,
        historical_terminal_evidence_json=args.historical_terminal_evidence_json,
        fresh_execution_authority_json=args.fresh_execution_authority_json,
        output_report=args.output_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        expected_execution_head=args.expected_execution_head,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_FRANKO1901_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print(
        "MATCHER_REPORT_SHA256="
        + evidence["combined"]["indexed_report_sha256"]
    )
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
