#!/usr/bin/env python3
"""Execute audited NBU text through the incumbent indexed global-dedup authority."""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib
import json
import math
import os
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
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
from twelve_six.data import incumbent_dedup_indexed_execution as indexed
from twelve_six.data import nbu_dedup_intake as nbu

SCHEMA = "12-6.d03-nbu-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-nbu-global-dedup-survivors.v1"
TWO_CLEAN_SCHEMA = "12-6.d03-nbu-global-dedup-two-clean.v1"
TWO_CLEAN_INCOMPLETE_SCHEMA = "12-6.d03-nbu-global-dedup-two-clean-incomplete.v1"
TWO_CLEAN_WORKER_TIMEOUT_SECONDS = 3_600
EXPECTED_MAIN = "ba9e49cedba4a110e1c4f7d83702e8fcf8a42461"
EXECUTION_CLAIM = 2398
EXECUTION_PR = 2454
CARRIER_PATH = "tools/run_d03_nbu_global_dedup_execution_v1.py"
INTAKE_PATH = "src/twelve_six/data/nbu_dedup_intake.py"
EXPECTED_BASE_OBJECTS = 264
EXPECTED_BASE_BYTES = 6_095_624
EXPECTED_NBU_OBJECTS = nbu.CANDIDATE_RECORDS
EXPECTED_NBU_BYTES = nbu.CANDIDATE_TEXT_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_NBU_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_NBU_BYTES

MAIN_AUTHORITY_PATHS = (
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
PRODUCT_PATHS = (INTAKE_PATH, CARRIER_PATH)


class NbuGlobalDedupError(RuntimeError):
    """Fail-closed NBU physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NbuGlobalDedupError(message)


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


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


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
        raise NbuGlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise NbuGlobalDedupError(f"git {' '.join(args)} failed: {detail}")
    return proc


def _bind_execution_head(expected_execution_head: str) -> str:
    _require(
        type(expected_execution_head) is str
        and len(expected_execution_head) == 40
        and all(char in "0123456789abcdef" for char in expected_execution_head),
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    observed = _git("rev-parse", "HEAD").stdout.strip()
    _require(
        observed == expected_execution_head,
        "execution HEAD drift: refusing stale or unselected checkout",
    )
    return observed


def verify_repository_authority() -> tuple[dict[str, str], dict[str, str]]:
    ancestor = _git("merge-base", "--is-ancestor", EXPECTED_MAIN, "HEAD", check=False)
    _require(ancestor.returncode == 0, "pinned main is not an ancestor of execution HEAD")
    dirty = _git(
        "diff",
        "--quiet",
        "HEAD",
        "--",
        *MAIN_AUTHORITY_PATHS,
        *PRODUCT_PATHS,
        check=False,
    )
    _require(dirty.returncode == 0, "authority/product worktree drift")

    main_blobs: dict[str, str] = {}
    for path in MAIN_AUTHORITY_PATHS:
        expected = _git("rev-parse", f"{EXPECTED_MAIN}:{path}").stdout.strip()
        observed = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        _require(bool(expected) and observed == expected, f"main authority path drift: {path}")
        main_blobs[path] = observed

    product_blobs: dict[str, str] = {}
    for path in PRODUCT_PATHS:
        committed = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        worktree = _git("hash-object", str(ROOT / path)).stdout.strip()
        _require(len(committed) == 40 and committed == worktree, f"Product path drift: {path}")
        product_blobs[path] = committed
    return main_blobs, product_blobs


_HISTORICAL_MATCHER_MODULES = (
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
    _require(historical_package not in current_package_path, "V7 package path preloaded")
    _require(historical_data not in current_data_path, "V7 data path preloaded")

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
    rows = base_inventory.get("sources")
    _require(type(rows) is list and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if type(row) is dict}
    _require(len(base_ids) == len(rows) and None not in base_ids, "base source ids invalid")
    _require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    extension_ids = {row.get("source_id") for row in extension_sources if type(row) is dict}
    _require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "NBU source ids invalid",
    )
    _require(extension_ids == set(extension_payloads), "NBU inventory/payload mismatch")
    _require(not (base_ids & extension_ids), "NBU source-id collision with incumbent graph")
    _require(
        all(type(key) is str and type(value) is bytes for key, value in base_payloads.items()),
        "base payload map must be exact str->bytes",
    )
    _require(
        all(
            type(key) is str and type(value) is bytes
            for key, value in extension_payloads.items()
        ),
        "NBU payload map must be exact str->bytes",
    )

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed V8 authority plus exact audited NBU materialized records; "
        "pair decisions delegate only to terminal PR #824 V3 semantics through merged "
        "performance-equivalent indexed executor PR #1459."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def _execute_indexed_matcher(
    matcher: Any,
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> tuple[dict[str, Any], float]:
    """Execute only the independently-qualified incumbent indexed matcher path."""

    indexed.attest_incumbent_runtime(matcher)
    indexed_started = time.perf_counter()
    report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    indexed_elapsed = time.perf_counter() - indexed_started
    matcher.verify_report(report)
    return report, indexed_elapsed


def _validated_terminal_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(report.get("source_count") == EXPECTED_COMBINED_OBJECTS, "report count drift")
    terminal = report.get("terminal_candidates")
    _require(type(terminal) is dict, "terminal matcher summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    clusters = terminal.get("duplicate_cluster_count")
    _require(type(before) is int and before == EXPECTED_COMBINED_BYTES, "pre-dedup bytes drift")
    _require(type(after) is int and 0 < after <= before, "post-dedup bytes invalid")
    _require(type(discount) is int and discount == before - after, "duplicate discount drift")
    _require(type(clusters) is int and clusters >= 0, "duplicate cluster count invalid")
    return terminal


def _validate_survivor_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> None:
    terminal = _validated_terminal_summary(report)
    projection_core = dict(projection)
    projection_identity = projection_core.pop("survivor_authority_sha256", None)
    _require(
        _is_lower_hex(projection_identity, 64)
        and projection_identity == _sha256(_canonical(projection_core)),
        "survivor projection self-hash mismatch",
    )
    _require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor schema drift",
    )
    _require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "survivor matcher identity drift",
    )
    survivor_ids = projection.get("survivor_source_ids")
    _require(
        type(survivor_ids) is list
        and all(type(source_id) is str and source_id for source_id in survivor_ids)
        and len(survivor_ids) == len(set(survivor_ids)),
        "survivor ids invalid",
    )
    post_count = projection.get("post_dedup_survivor_source_object_count")
    _require(
        type(post_count) is int
        and 0 < post_count <= EXPECTED_COMBINED_OBJECTS
        and post_count == len(survivor_ids),
        "survivor count drift",
    )
    _require(
        projection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "survivor pre-dedup count drift",
    )
    _require(
        projection.get("pre_dedup_declared_capacity_bytes") == EXPECTED_COMBINED_BYTES,
        "survivor pre-dedup bytes drift",
    )
    _require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "survivor post-dedup bytes drift",
    )
    _require(
        projection.get("duplicate_discount_bytes") == terminal.get("duplicate_discount_bytes"),
        "survivor duplicate discount drift",
    )
    clusters = projection.get("duplicate_clusters")
    cluster_count = projection.get("duplicate_cluster_count")
    _require(type(clusters) is list, "survivor duplicate clusters missing")
    _require(
        type(cluster_count) is int
        and cluster_count == len(clusters)
        and cluster_count == terminal.get("duplicate_cluster_count"),
        "survivor duplicate cluster count drift",
    )


def _outer_survivor_authority(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = report.get("sources")
    _require(type(sources) is list, "dedup source vector missing")
    by_id = {
        row["source_id"]: row
        for row in sources
        if type(row) is dict and type(row.get("source_id")) is str
    }
    _require(len(by_id) == len(sources), "dedup source identity drift")
    survivor_ids = projection["survivor_source_ids"]
    _require(all(source_id in by_id for source_id in survivor_ids), "unknown survivor id")
    nbu_ids = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == nbu.SOURCE_FAMILY
    ]
    nbu_bytes = 0
    for source_id in nbu_ids:
        value = by_id[source_id].get("declared_capacity_bytes")
        _require(type(value) is int and value >= 0, "NBU survivor byte count invalid")
        nbu_bytes += value
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": report.get("report_sha256"),
        "selection_projection_schema": projection.get("schema_version"),
        "selection_projection_sha256": projection.get("survivor_authority_sha256"),
        "pre_dedup_source_object_count": projection.get("pre_dedup_source_object_count"),
        "post_dedup_survivor_source_object_count": projection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "pre_dedup_declared_capacity_bytes": projection.get(
            "pre_dedup_declared_capacity_bytes"
        ),
        "post_dedup_declared_capacity_bytes": projection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": projection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": projection.get("duplicate_cluster_count"),
        "duplicate_clusters": copy.deepcopy(projection.get("duplicate_clusters")),
        "survivor_source_ids": copy.deepcopy(survivor_ids),
        "nbu_survivor_source_ids": nbu_ids,
        "nbu_survivor_source_object_count": len(nbu_ids),
        "nbu_survivor_declared_capacity_bytes": nbu_bytes,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def _validate_two_clean_survivor_readback(
    report: Mapping[str, Any],
    survivor: Mapping[str, Any],
) -> None:
    selection = v9_semantics._derive_survivors(report)
    _validate_survivor_projection(report, selection)
    expected = _outer_survivor_authority(report, selection)
    _require(
        _canonical(survivor) == _canonical(expected),
        "two-clean survivor semantic readback drift",
    )


def _runtime_environment() -> dict[str, Any]:
    github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    result: dict[str, Any] = {
        "python_platform": sys.platform,
        "github_actions": github_actions,
    }
    if github_actions:
        environment = os.environ.get("RUNNER_ENVIRONMENT")
        runner_os = os.environ.get("RUNNER_OS")
        runner_arch = os.environ.get("RUNNER_ARCH")
        _require(environment in {"github-hosted", "self-hosted"}, "runner environment missing")
        _require(type(runner_os) is str and bool(runner_os), "runner OS missing")
        _require(type(runner_arch) is str and bool(runner_arch), "runner arch missing")
        result.update(
            {
                "runner_environment": environment,
                "runner_os": runner_os,
                "runner_arch": runner_arch,
            }
        )
    else:
        result["runner_environment"] = "local"
    return result


def _windows_peak_working_set_kib() -> int | None:
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
    return (peak_bytes + 1023) // 1024 if peak_bytes > 0 else None


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


def _publish_json_outputs(outputs: tuple[tuple[Path, Mapping[str, Any]], ...]) -> None:
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
                raise NbuGlobalDedupError(f"refusing to overwrite: {path}") from exc
    except Exception as exc:
        rollback_errors: list[str] = []
        for created_path in reversed(created):
            try:
                created_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{created_path}: {rollback_exc}")
        if rollback_errors:
            raise NbuGlobalDedupError(
                "output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise



def _strict_generated_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise NbuGlobalDedupError(f"cannot read generated JSON: {path}") from exc

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate generated JSON key: {key}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        _require(math.isfinite(value), f"non-finite generated JSON number: {token}")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_pairs,
            parse_float=finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                NbuGlobalDedupError(f"non-finite generated JSON constant: {token}")
            ),
        )
    except NbuGlobalDedupError:
        raise
    except (json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise NbuGlobalDedupError(f"invalid generated JSON: {path}") from exc
    _require(type(value) is dict, f"generated JSON root must be exact object: {path}")
    return value, raw


def _require_distinct_materialization_copies(
    candidate_a: Path,
    evidence_a: Path,
    candidate_b: Path,
    evidence_b: Path,
) -> None:
    paths = (candidate_a, evidence_a, candidate_b, evidence_b)
    resolved: list[Path] = []
    for path in paths:
        _require(not path.is_symlink(), f"materialization input must not be symlink: {path}")
        try:
            exact = path.resolve(strict=True)
        except OSError as exc:
            raise NbuGlobalDedupError(f"cannot resolve materialization input: {path}") from exc
        _require(exact.is_file(), f"materialization input is not regular file: {path}")
        resolved.append(exact)
    _require(resolved[0] != resolved[2], "two-clean candidate paths must be distinct")
    _require(resolved[1] != resolved[3], "two-clean evidence paths must be distinct")
    try:
        _require(
            not os.path.samefile(resolved[0], resolved[2]),
            "two-clean candidate copies alias one file",
        )
        _require(
            not os.path.samefile(resolved[1], resolved[3]),
            "two-clean evidence copies alias one file",
        )
    except OSError as exc:
        raise NbuGlobalDedupError(
            "cannot attest distinct two-clean materialization inputs"
        ) from exc


def _incumbent_report_identity(report: Mapping[str, Any]) -> str:
    core = dict(report)
    core.pop("report_sha256", None)
    rendered = json.dumps(
        core,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256((rendered + "\n").encode("utf-8"))


def _build_two_clean_authority(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
    first_survivors: Mapping[str, Any],
    second_survivors: Mapping[str, Any],
    first_evidence: Mapping[str, Any],
    second_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        _canonical(first_report) == _canonical(second_report),
        "two-clean dedup reports differ",
    )
    _require(
        _canonical(first_survivors) == _canonical(second_survivors),
        "two-clean survivor authorities differ",
    )
    report_sha = first_report.get("report_sha256")
    survivor_sha = first_survivors.get("survivor_authority_sha256")
    _require(
        _is_lower_hex(report_sha, 64)
        and second_report.get("report_sha256") == report_sha,
        "two-clean report identity drift",
    )
    for report in (first_report, second_report):
        _require(
            report.get("report_sha256") == _incumbent_report_identity(report),
            "two-clean report self-hash mismatch",
        )
    _require(
        _is_lower_hex(survivor_sha, 64)
        and second_survivors.get("survivor_authority_sha256") == survivor_sha,
        "two-clean survivor identity drift",
    )
    for survivor in (first_survivors, second_survivors):
        _require(
            survivor.get("matcher_report_sha256") == report_sha,
            "two-clean survivor/report identity drift",
        )
        survivor_core = dict(survivor)
        survivor_identity = survivor_core.pop("survivor_authority_sha256", None)
        _require(
            survivor_identity == _sha256(_canonical(survivor_core)),
            "two-clean survivor self-hash mismatch",
        )
    execution_head = first_evidence.get("execution_head_sha")
    _require(
        _is_lower_hex(execution_head, 40)
        and second_evidence.get("execution_head_sha") == execution_head,
        "two-clean execution head drift",
    )
    for evidence in (first_evidence, second_evidence):
        evidence_core = dict(evidence)
        evidence_identity = evidence_core.pop("evidence_identity_sha256", None)
        _require(
            _is_lower_hex(evidence_identity, 64)
            and evidence_identity == _sha256(_canonical(evidence_core)),
            "two-clean run evidence self-hash mismatch",
        )
        _require(evidence.get("pinned_main_sha") == EXPECTED_MAIN, "two-clean pinned-main drift")
        nbu_evidence = evidence.get("nbu")
        _require(type(nbu_evidence) is dict, "two-clean NBU evidence missing")
        _require(
            nbu_evidence.get("materialization_head") == nbu.MATERIALIZATION_HEAD
            and nbu_evidence.get("workflow_run_id") == nbu.MATERIALIZATION_RUN
            and nbu_evidence.get("workflow_job_id") == nbu.MATERIALIZATION_JOB
            and nbu_evidence.get("artifact_id") == nbu.MATERIALIZATION_ARTIFACT
            and nbu_evidence.get("independent_audit_issue") == nbu.MATERIALIZATION_AUDIT,
            "two-clean NBU materialization authority drift",
        )
        _require(
            nbu_evidence.get("candidate_sha256") == nbu.CANDIDATE_SHA256,
            "two-clean candidate identity drift",
        )
        _require(
            _is_lower_hex(nbu_evidence.get("intake_receipt_identity_sha256"), 64),
            "two-clean intake receipt identity invalid",
        )
        combined = evidence.get("combined")
        _require(type(combined) is dict, "two-clean combined evidence missing")
        _require(
            combined.get("indexed_report_sha256") == report_sha,
            "two-clean combined report identity drift",
        )
        matcher = evidence.get("matcher_execution")
        _require(type(matcher) is dict, "two-clean matcher evidence missing")
        _require(matcher.get("engine") == "MERGED_PR_1459", "two-clean matcher engine drift")
        _require(matcher.get("report_sha256") == report_sha, "two-clean matcher report drift")
        _require(
            evidence.get("survivor_authority_sha256") == survivor_sha,
            "two-clean evidence/survivor identity drift",
        )
        truth = evidence.get("truth_boundary")
        _require(type(truth) is dict, "two-clean truth boundary missing")
        _require(
            type(truth.get("canonical_capacity_credited")) is int
            and truth.get("canonical_capacity_credited") == 0,
            "two-clean capacity promotion",
        )
        _require(
            type(truth.get("authorized_optimized_target_exposure")) is int
            and truth.get("authorized_optimized_target_exposure") == 0,
            "two-clean exposure promotion",
        )
        _require(truth.get("training_executed") is False, "two-clean training promotion")

    nbu_survivor_count = first_survivors.get("nbu_survivor_source_object_count")
    nbu_survivor_bytes = first_survivors.get("nbu_survivor_declared_capacity_bytes")
    _require(
        type(nbu_survivor_count) is int and nbu_survivor_count >= 0,
        "NBU survivor count invalid",
    )
    _require(
        type(nbu_survivor_bytes) is int and nbu_survivor_bytes >= 0,
        "NBU survivor bytes invalid",
    )

    intake_receipt_sha = first_evidence["nbu"]["intake_receipt_identity_sha256"]
    _require(
        second_evidence["nbu"]["intake_receipt_identity_sha256"] == intake_receipt_sha,
        "two-clean intake receipt identities differ",
    )
    evidence_ids = [
        first_evidence.get("evidence_identity_sha256"),
        second_evidence.get("evidence_identity_sha256"),
    ]
    _require(
        all(_is_lower_hex(value, 64) for value in evidence_ids),
        "two-clean run evidence identity invalid",
    )
    core: dict[str, Any] = {
        "schema_version": TWO_CLEAN_SCHEMA,
        "status": "PASS_TWO_CLEAN_DEDUP_OVER_EXACT_AUDITED_NBU_COPIES_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "pinned_main_sha": EXPECTED_MAIN,
        "materialization_authority": {
            "head_sha": nbu.MATERIALIZATION_HEAD,
            "audit_issue": nbu.MATERIALIZATION_AUDIT,
            "candidate_sha256": nbu.CANDIDATE_SHA256,
            "evidence_identity_sha256": nbu.EVIDENCE_IDENTITY_SHA256,
            "intake_receipt_identity_sha256": intake_receipt_sha,
            "distinct_input_copies_required": True,
            "source_replay_executed_by_this_carrier": False,
        },
        "dedup": {
            "fresh_process_count": 2,
            "engine": "MERGED_PR_1459",
            "report_sha256": report_sha,
            "survivor_authority_sha256": survivor_sha,
            "run_evidence_identity_sha256": evidence_ids,
            "nbu_survivor_source_object_count": nbu_survivor_count,
            "nbu_survivor_declared_capacity_bytes": nbu_survivor_bytes,
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
        },
    }
    return {**core, "two_clean_authority_sha256": _sha256(_canonical(core))}


def _write_two_clean_incomplete(output_root: Path, completed_runs: list[str], reason: str) -> None:
    _publish_json_outputs(
        (
            (
                output_root / "incomplete.json",
                {
                    "schema_version": TWO_CLEAN_INCOMPLETE_SCHEMA,
                    "status": "INCOMPLETE_NO_TWO_CLEAN_AUTHORITY",
                    "reason": reason,
                    "completed_run_ids": list(completed_runs),
                    "canonical_capacity_credited": 0,
                    "authorized_optimized_target_exposure": 0,
                    "training_executed": False,
                    "paid_compute_used": False,
                },
            ),
        )
    )


def _run_two_clean_worker(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=TWO_CLEAN_WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise
    except OSError as exc:
        raise NbuGlobalDedupError("cannot start two-clean worker") from exc
    if completed.returncode != 0:
        raise NbuGlobalDedupError(
            f"two-clean worker failed with exit {completed.returncode}"
        )


def run_two_clean(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl_a: Path,
    materialization_evidence_json_a: Path,
    candidate_jsonl_b: Path,
    materialization_evidence_json_b: Path,
    output_root: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    orchestration_head = _bind_execution_head(expected_execution_head)
    _require_distinct_materialization_copies(
        candidate_jsonl_a,
        materialization_evidence_json_a,
        candidate_jsonl_b,
        materialization_evidence_json_b,
    )
    first_projection = nbu.validate_and_project_nbu(
        candidate_jsonl_a,
        materialization_evidence_json_a,
        retain_payloads=False,
    )
    second_projection = nbu.validate_and_project_nbu(
        candidate_jsonl_b,
        materialization_evidence_json_b,
        retain_payloads=False,
    )
    _require(
        first_projection.receipt["receipt_identity_sha256"]
        == second_projection.receipt["receipt_identity_sha256"],
        "two-clean intake projection identity drift",
    )

    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise NbuGlobalDedupError(
            f"refusing non-fresh two-clean output root: {output_root}"
        ) from exc

    script = Path(__file__).resolve()
    runs = (
        ("clean-a", candidate_jsonl_a, materialization_evidence_json_a),
        ("clean-b", candidate_jsonl_b, materialization_evidence_json_b),
    )
    completed_runs: list[str] = []
    try:
        for run_id, candidate, materialization_evidence in runs:
            run_dir = output_root / run_id
            command = [
                sys.executable,
                str(script),
                "--v7-root",
                str(v7_root),
                "--bulk-workspace",
                str(bulk_workspace),
                "--candidate-jsonl",
                str(candidate),
                "--materialization-evidence-json",
                str(materialization_evidence),
                "--output-report",
                str(run_dir / "dedup-report.json"),
                "--output-survivors",
                str(run_dir / "survivor-authority.json"),
                "--output-evidence",
                str(run_dir / "execution-evidence.json"),
                "--expected-execution-head",
                expected_execution_head,
                "--max-candidate-pairs",
                str(max_candidate_pairs),
                "--max-index-postings",
                str(max_index_postings),
                "--max-pair-expansions",
                str(max_pair_expansions),
            ]
            _run_two_clean_worker(command)
            completed_runs.append(run_id)
    except subprocess.TimeoutExpired as exc:
        _write_two_clean_incomplete(output_root, completed_runs, "worker_timeout")
        raise NbuGlobalDedupError("two-clean worker exceeded fixed timeout") from exc
    except NbuGlobalDedupError:
        _write_two_clean_incomplete(output_root, completed_runs, "worker_execution_failed")
        raise
    except KeyboardInterrupt:
        _write_two_clean_incomplete(output_root, completed_runs, "operator_interrupt")
        raise

    try:
        first_report, first_report_raw = _strict_generated_json(
            output_root / "clean-a" / "dedup-report.json"
        )
        second_report, second_report_raw = _strict_generated_json(
            output_root / "clean-b" / "dedup-report.json"
        )
        first_survivors, first_survivors_raw = _strict_generated_json(
            output_root / "clean-a" / "survivor-authority.json"
        )
        second_survivors, second_survivors_raw = _strict_generated_json(
            output_root / "clean-b" / "survivor-authority.json"
        )
        first_evidence, _ = _strict_generated_json(
            output_root / "clean-a" / "execution-evidence.json"
        )
        second_evidence, _ = _strict_generated_json(
            output_root / "clean-b" / "execution-evidence.json"
        )
        _validate_two_clean_survivor_readback(first_report, first_survivors)
        _validate_two_clean_survivor_readback(second_report, second_survivors)
        _require(first_report_raw == second_report_raw, "two-clean report bytes differ")
        _require(first_survivors_raw == second_survivors_raw, "two-clean survivor bytes differ")
        authority = _build_two_clean_authority(
            first_report,
            second_report,
            first_survivors,
            second_survivors,
            first_evidence,
            second_evidence,
        )
        _require(
            authority.get("execution_head_sha") == orchestration_head,
            "two-clean aggregate execution head drift",
        )
        _publish_json_outputs(((output_root / "two-clean-authority.json", authority),))
    except (NbuGlobalDedupError, OSError) as exc:
        if not (output_root / "incomplete.json").exists():
            _write_two_clean_incomplete(output_root, completed_runs, "post_run_convergence_failed")
        if isinstance(exc, NbuGlobalDedupError):
            raise
        raise NbuGlobalDedupError("cannot finalize two-clean authority") from exc
    return authority


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    materialization_evidence_json: Path,
    output_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = _bind_execution_head(expected_execution_head)
    main_blobs, product_blobs = verify_repository_authority()
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

    projection = nbu.validate_and_project_nbu(
        candidate_jsonl,
        materialization_evidence_json,
        retain_payloads=True,
    )
    _require(
        projection.sources is not None and projection.payloads is not None,
        "NBU payload projection missing",
    )
    extension_sources = [dict(row) for row in projection.sources]
    extension_payloads = dict(projection.payloads)
    _require(len(extension_sources) == EXPECTED_NBU_OBJECTS, "NBU projection count drift")
    _require(
        sum(len(raw) for raw in extension_payloads.values()) == EXPECTED_NBU_BYTES,
        "NBU projection byte total drift",
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

    report, indexed_elapsed = _execute_indexed_matcher(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    terminal = _validated_terminal_summary(report)
    selection = v9_semantics._derive_survivors(report)
    _validate_survivor_projection(report, selection)
    survivors = _outer_survivor_authority(report, selection)
    max_rss_kib = _max_rss_kib()
    _require(type(max_rss_kib) is int and max_rss_kib > 0, "process max RSS unavailable")

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "runtime_environment": _runtime_environment(),
        "execution_claim_issue": EXECUTION_CLAIM,
        "execution_pr": EXECUTION_PR,
        "execution_head_sha": execution_head,
        "pinned_main_sha": EXPECTED_MAIN,
        "main_authority_path_blobs": main_blobs,
        "product_path_blobs": product_blobs,
        "baseline_v8": {
            "v7_head_sha": v8.EXPECTED_V7_HEAD,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "payload_bytes": EXPECTED_BASE_BYTES,
        },
        "nbu": {
            "materialization_head": nbu.MATERIALIZATION_HEAD,
            "workflow_run_id": nbu.MATERIALIZATION_RUN,
            "workflow_job_id": nbu.MATERIALIZATION_JOB,
            "artifact_id": nbu.MATERIALIZATION_ARTIFACT,
            "independent_audit_issue": nbu.MATERIALIZATION_AUDIT,
            "candidate_sha256": nbu.CANDIDATE_SHA256,
            "source_object_count": EXPECTED_NBU_OBJECTS,
            "payload_bytes": EXPECTED_NBU_BYTES,
            "intake_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": EXPECTED_COMBINED_BYTES,
            "indexed_report_sha256": report["report_sha256"],
            "indexed_executor_performance_equivalence_authority": "MERGED_PR_1459",
            "post_dedup_conservative_unique_bytes": terminal[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
            "duplicate_cluster_count": terminal["duplicate_cluster_count"],
        },
        "matcher_execution": {
            "engine": "MERGED_PR_1459",
            "performance_equivalence_authority": "MERGED_PR_1459",
            "incumbent_runtime_attested": True,
            "report_sha256": report["report_sha256"],
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "indexed_wall_clock_seconds": round(indexed_elapsed, 6),
            "process_max_rss_kib": max_rss_kib,
            "all_pairs_reference_executed": False,
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "raw_text_emitted": False,
            "raw_candidate_written_to_durable_evidence": False,
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
            (output_report, report),
            (output_survivors, survivors),
            (output_evidence, evidence),
        )
    )
    return evidence



def _main_two_clean(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run two fresh NBU dedup workers over two distinct exact-audited "
            "materialization copies."
        )
    )
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl-a", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json-a", type=Path, required=True)
    parser.add_argument("--candidate-jsonl-b", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
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
    args = parser.parse_args(argv)
    try:
        authority = run_two_clean(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            candidate_jsonl_a=args.candidate_jsonl_a,
            materialization_evidence_json_a=args.materialization_evidence_json_a,
            candidate_jsonl_b=args.candidate_jsonl_b,
            materialization_evidence_json_b=args.materialization_evidence_json_b,
            output_root=args.output_root,
            expected_execution_head=args.expected_execution_head,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (NbuGlobalDedupError, nbu.NbuDedupIntakeError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_NBU_TWO_CLEAN_DEDUP=PASS_ZERO_CREDIT")
    print("TWO_CLEAN_AUTHORITY_SHA256=" + authority["two_clean_authority_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + authority["dedup"]["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
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
    try:
        evidence = execute(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            candidate_jsonl=args.candidate_jsonl,
            materialization_evidence_json=args.materialization_evidence_json,
            output_report=args.output_report,
            output_survivors=args.output_survivors,
            output_evidence=args.output_evidence,
            expected_execution_head=args.expected_execution_head,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (NbuGlobalDedupError, nbu.NbuDedupIntakeError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_NBU_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("MATCHER_REPORT_SHA256=" + evidence["combined"]["indexed_report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "two-clean":
        raise SystemExit(_main_two_clean(sys.argv[2:]))
    raise SystemExit(main())
