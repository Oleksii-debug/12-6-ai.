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
import platform
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
import run_d03_nomis_free_clean_successor_v1 as clean_successor
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import external_llm_provenance_quarantine_v1 as quarantine
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
EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_BYTES = 6_093_965
EXPECTED_NBU_OBJECTS = nbu.CANDIDATE_RECORDS
EXPECTED_NBU_BYTES = nbu.CANDIDATE_TEXT_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_NBU_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_NBU_BYTES
PAYLOAD_BYTES_SEMANTICS = "DECLARED_CAPACITY_BYTES"

MAIN_AUTHORITY_PATHS = (
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "tools/run_d03_expanded_global_dedup_v9.py",
    "tools/run_d03_nomis_free_clean_successor_v1.py",
    "src/twelve_six/data/external_llm_provenance_quarantine_v1.py",
    "configs/data/d03_external_llm_provenance_quarantine_v1.json",
    "tools/run_next100_065f_global_dedup_v8.py",
    "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
    "configs/data/next100_065f_global_dedup_v8.json",
    "configs/data/data_bulk_code1_permissive_python_bundle_v1.json",
    "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json",
)
PRODUCT_PATHS = (INTAKE_PATH, CARRIER_PATH)
NOMIS_REMOVAL_PROOF = {
    "blocked_family": clean_successor.BLOCKED_FAMILY,
    "blocked_payload_bytes": clean_successor.BLOCKED_BYTES,
    "blocked_payload_sha256": clean_successor.BLOCKED_SHA256,
    "blocked_source_id": clean_successor.BLOCKED_SOURCE_ID,
    "non_blocked_source_ids_byte_for_byte_preserved": True,
    "post_source_capacity_bytes": 2_213_956,
    "post_source_object_count": 34,
    "pre_source_capacity_bytes": 2_215_615,
    "pre_source_object_count": 35,
    "quarantine_identity_sha256": clean_successor.EXPECTED_QUARANTINE_IDENTITY,
    "removed_before_new_global_dedup": True,
}


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


def _verify_removal_proof(proof: Any) -> None:
    _require(
        type(proof) is dict and _canonical(proof) == _canonical(NOMIS_REMOVAL_PROOF),
        "Nomis deauthorization proof drift",
    )


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_work_budgets(
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> None:
    for label, value in (
        ("max_candidate_pairs", max_candidate_pairs),
        ("max_index_postings", max_index_postings),
        ("max_pair_expansions", max_pair_expansions),
    ):
        _require(
            type(value) is int and value > 0,
            f"{label} must be exact positive int",
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


def _verify_clean_payload_graph(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    quarantine_authority: Mapping[str, Any],
) -> None:
    rows = inventory.get("sources")
    _require(type(rows) is list and bool(rows), "clean base source rows missing")
    ids = [row.get("source_id") for row in rows if type(row) is dict]
    _require(
        len(ids) == len(rows)
        and all(type(source_id) is str and bool(source_id) for source_id in ids)
        and len(set(ids)) == len(ids)
        and set(ids) == set(payloads)
        and all(type(raw) is bytes for raw in payloads.values()),
        "clean base inventory/payload coverage mismatch",
    )
    quarantine.reject_quarantined_inventory_rows(
        [
            {
                "source_id": row["source_id"],
                "family": row.get("source_family"),
                "payload_sha256": _sha256(payloads[row["source_id"]]),
                "payload_bytes": len(payloads[row["source_id"]]),
            }
            for row in rows
        ],
        quarantine_authority,
    )


def _declared_capacity_bytes(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    label: str,
) -> int:
    rows = inventory.get("sources")
    _require(type(rows) is list and bool(rows), f"{label} source rows missing")
    seen: set[str] = set()
    total = 0
    for row in rows:
        _require(type(row) is dict, f"{label} source row invalid")
        source_id = row.get("source_id")
        _require(
            type(source_id) is str and bool(source_id) and source_id not in seen,
            f"{label} source identity invalid or duplicate",
        )
        declared = row.get("declared_capacity_bytes")
        _require(
            type(declared) is int and declared >= 0,
            f"{label} declared capacity must be exact nonnegative int",
        )
        seen.add(source_id)
        total += declared
    _require(seen == set(payloads), f"{label} inventory/payload coverage mismatch")
    return total


def _reconstruct_clean_source_inputs(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Reuse incumbent deauthorization before composing the new matcher input."""
    v9_runner.validate_v7_checkout(v7_root)
    clean_successor.validate_runtime_bindings(ROOT)
    quarantine_authority = json.loads(
        (ROOT / clean_successor.QUARANTINE_CONFIG_PATH).read_text(encoding="utf-8")
    )
    v7, _, historical_inventory, historical_payloads = v8._capture_terminal_v7(
        v7_root, config
    )
    clean_inventory, clean_payloads, removal = clean_successor.deauthorize_exact_nomis(
        historical_inventory,
        historical_payloads,
        quarantine_authority,
        quarantine,
    )
    _verify_removal_proof(removal)
    _, bulk_rows, bulk_payloads = v8._materialize_bulk(ROOT, bulk_workspace, config)
    existing_ids = {row["source_id"] for row in clean_inventory["sources"]}
    _require(not (existing_ids & set(bulk_payloads)), "clean base/bulk source-id collision")
    inventory = copy.deepcopy(clean_inventory)
    inventory["sources"] = [*inventory["sources"], *bulk_rows]
    payloads = dict(clean_payloads)
    payloads.update(bulk_payloads)
    _verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    _require(len(payloads) == EXPECTED_BASE_OBJECTS, "clean base object-count drift")
    _require(
        _declared_capacity_bytes(inventory, payloads, label="clean base")
        == EXPECTED_BASE_BYTES,
        "clean base declared-capacity drift",
    )
    return v7.v6.v3, inventory, payloads, removal


def _reconstruct_v8_with_historical_namespace(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
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
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    try:
        matcher, inventory, payloads, removal = _reconstruct_clean_source_inputs(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    finally:
        twelve_six_pkg.__path__ = current_package_path
        data_pkg.__path__ = current_data_path
        sys.dont_write_bytecode = previous_dont_write_bytecode
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
    return matcher, inventory, payloads, removal


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
        "Exact reconstructed Nomis-free V8 source graph plus exact audited NBU materialized records; "
        "pair decisions delegate only to terminal PR #824 V3 semantics through merged "
        "performance-equivalent indexed executor PR #1459."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    quarantine_authority = json.loads(
        (ROOT / clean_successor.QUARANTINE_CONFIG_PATH).read_text(encoding="utf-8")
    )
    _verify_clean_payload_graph(inventory, payloads, quarantine_authority)
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


def _validate_nbu_report_binding(
    report: Mapping[str, Any],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> None:
    report_sources = report.get("sources")
    _require(type(report_sources) is list, "dedup report source vector missing")
    by_id = {
        row.get("source_id"): row
        for row in report_sources
        if type(row) is dict and type(row.get("source_id")) is str
    }
    _require(len(by_id) == len(report_sources), "dedup report source identity drift")
    _require(
        len(extension_sources) == len(extension_payloads),
        "NBU projected source/payload cardinality drift",
    )
    expected_ids: list[str] = []
    for expected in extension_sources:
        _require(type(expected) is dict, "NBU projected source row invalid")
        source_id = expected.get("source_id")
        _require(
            type(source_id) is str and bool(source_id),
            "NBU projected source identity missing",
        )
        expected_ids.append(source_id)
    _require(
        len(expected_ids) == len(set(expected_ids)),
        "NBU projected source ids duplicate",
    )
    _require(
        set(expected_ids) == set(extension_payloads),
        "NBU projected source/payload coverage drift",
    )
    _require(
        all(type(key) is str and type(value) is bytes for key, value in extension_payloads.items()),
        "NBU projected payload map must be exact str->bytes",
    )
    for expected in extension_sources:
        source_id = expected["source_id"]
        observed = by_id.get(source_id)
        _require(type(observed) is dict, f"NBU source missing from report: {source_id}")
        payload = extension_payloads[source_id]
        exact_fields = {
            "source_family": expected.get("source_family"),
            "modality": expected.get("modality"),
            "evidence_status": expected.get("evidence_status"),
            "declared_capacity_bytes": expected.get("declared_capacity_bytes"),
        }
        for field, expected_value in exact_fields.items():
            _require(
                type(observed.get(field)) is type(expected_value)
                and observed.get(field) == expected_value,
                f"NBU report field drift for {source_id}: {field}",
            )
        stable_origin_id = expected.get("stable_origin_id")
        stable_object_id = expected.get("stable_object_id")
        _require(
            type(stable_origin_id) is str and bool(stable_origin_id),
            f"NBU stable origin invalid: {source_id}",
        )
        _require(
            type(stable_object_id) is str and bool(stable_object_id),
            f"NBU stable object invalid: {source_id}",
        )
        _require(
            observed.get("stable_origin_id_sha256")
            == _sha256(stable_origin_id.encode("utf-8")),
            f"NBU stable origin drift: {source_id}",
        )
        _require(
            observed.get("stable_object_id_sha256")
            == _sha256(stable_object_id.encode("utf-8")),
            f"NBU stable object drift: {source_id}",
        )
        _require(
            observed.get("verified_raw_bytes") == len(payload),
            f"NBU verified raw byte drift: {source_id}",
        )
        payload_sha = _sha256(payload)
        _require(
            observed.get("verified_raw_sha256") == payload_sha,
            f"NBU verified raw hash drift: {source_id}",
        )
        _require(
            observed.get("comparison_policy") == "DATA232_GENERIC_FROM_RAW",
            f"NBU comparison policy drift: {source_id}",
        )
        _require(
            observed.get("comparison_payload_bytes") == len(payload),
            f"NBU comparison byte drift: {source_id}",
        )
        _require(
            observed.get("comparison_payload_sha256") == payload_sha,
            f"NBU comparison hash drift: {source_id}",
        )


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
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
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


PUBLICATION_SCHEMA = "12-6.d03-nbu-output-publication.v1"
PUBLICATION_MANIFEST_MAX_BYTES = 64 * 1024
_PUBLICATION_MANIFEST_KEYS = {
    "schema_version",
    "state",
    "publication_pathset_sha256",
    "targets",
    "manifest_identity_sha256",
}
_PUBLICATION_TARGET_KEYS = {"path", "stage_path", "sha256"}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _reject_publication_manifest_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate publication manifest key: {key}")
        result[key] = value
    return result


def _reject_publication_manifest_constant(value: str) -> Any:
    raise ValueError(f"non-finite publication manifest constant: {value}")


def _reject_publication_manifest_float(value: str) -> Any:
    raise ValueError(f"publication manifest float is forbidden: {value}")


def _is_sha256_hex(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in _HEX_DIGITS for char in value)
    )



def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Every marker, manifest and staged payload is flushed with os.fsync().
        # Python has no portable Windows directory-handle fsync; process-crash
        # atomicity is therefore supplied by the marker/manifest/hard-link
        # protocol rather than by assuming a provider or filesystem.
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise NbuGlobalDedupError(
            f"cannot open output directory for fsync: {path}"
        ) from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise NbuGlobalDedupError(
            f"cannot fsync output directory: {path}"
        ) from exc
    finally:
        os.close(fd)


class PublicationWriteCleanupError(NbuGlobalDedupError):
    """A durable-create failure left residue that must remain recovery-visible."""


def _write_create_only_durable(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise NbuGlobalDedupError(f"refusing to overwrite: {path}") from exc
    except OSError as exc:
        cleanup_error: Exception | None = None
        if created:
            try:
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            except (OSError, NbuGlobalDedupError) as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise PublicationWriteCleanupError(
                f"cannot write output and cleanup failed: {path}: {cleanup_error}"
            ) from exc
        raise NbuGlobalDedupError(f"cannot write output: {path}") from exc


def _publication_control_paths(
    prepared: tuple[tuple[Path, bytes], ...],
) -> tuple[Path, Path, tuple[Path, ...], str]:
    _require(bool(prepared), "publication output set must not be empty")
    resolved_paths = [str(path.resolve(strict=False)) for path, _ in prepared]
    pathset_id = _sha256(_canonical({"paths": resolved_paths}))
    marker = prepared[-1][0].with_name(prepared[-1][0].name + ".incomplete")
    manifest = marker.with_name(marker.name + ".manifest")
    stages = tuple(
        path.with_name(f".{path.name}.stage-{pathset_id[:20]}")
        for path, _ in prepared
    )
    return marker, manifest, stages, pathset_id


def _publication_marker_path(
    prepared: tuple[tuple[Path, bytes], ...],
) -> Path:
    return _publication_control_paths(prepared)[0]


def _publication_manifest(
    prepared: tuple[tuple[Path, bytes], ...],
    stages: tuple[Path, ...],
    pathset_id: str,
) -> tuple[dict[str, Any], bytes]:
    targets = [
        {
            "path": str(path.resolve(strict=False)),
            "stage_path": str(stage.resolve(strict=False)),
            "sha256": _sha256(payload),
        }
        for (path, payload), stage in zip(prepared, stages, strict=True)
    ]
    core = {
        "schema_version": PUBLICATION_SCHEMA,
        "state": "INCOMPLETE_NOT_TERMINAL",
        "publication_pathset_sha256": pathset_id,
        "targets": targets,
    }
    manifest = {
        **core,
        "manifest_identity_sha256": _sha256(_canonical(core)),
    }
    return manifest, _canonical(manifest) + b"\n"


def _load_publication_manifest(manifest_path: Path) -> dict[str, Any]:
    _require(
        not manifest_path.is_symlink() and manifest_path.is_file(),
        "incomplete publication manifest is not a regular file",
    )
    try:
        with manifest_path.open("rb") as handle:
            raw = handle.read(PUBLICATION_MANIFEST_MAX_BYTES + 1)
        if len(raw) > PUBLICATION_MANIFEST_MAX_BYTES:
            raise ValueError("publication manifest exceeds bounded size")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_publication_manifest_pairs,
            parse_constant=_reject_publication_manifest_constant,
            parse_float=_reject_publication_manifest_float,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise NbuGlobalDedupError(
            "incomplete publication manifest is unreadable"
        ) from exc
    _require(type(value) is dict, "incomplete publication manifest root invalid")
    _require(
        set(value) == _PUBLICATION_MANIFEST_KEYS,
        "incomplete publication manifest keys invalid",
    )
    _require(
        raw == _canonical(value) + b"\n",
        "incomplete publication manifest is not canonical",
    )
    identity = value.get("manifest_identity_sha256")
    core = {
        key: val for key, val in value.items() if key != "manifest_identity_sha256"
    }
    _require(
        _is_sha256_hex(identity)
        and identity == _sha256(_canonical(core)),
        "incomplete publication manifest identity mismatch",
    )
    _require(
        core.get("schema_version") == PUBLICATION_SCHEMA
        and core.get("state") == "INCOMPLETE_NOT_TERMINAL",
        "incomplete publication manifest semantics invalid",
    )
    _require(
        _is_sha256_hex(core.get("publication_pathset_sha256")),
        "incomplete publication path-set identity invalid",
    )
    targets = core.get("targets")
    _require(
        type(targets) is list and bool(targets),
        "publication manifest targets missing",
    )
    return value

def _remove_control_without_payload(
    marker_path: Path,
    manifest_path: Path,
) -> None:
    if _path_entry_exists(manifest_path):
        manifest_path.unlink()
        _fsync_directory(manifest_path.parent)
    marker_path.unlink()
    _fsync_directory(marker_path.parent)


def _recover_incomplete_publication(
    marker_path: Path,
    manifest_path: Path,
    prepared: tuple[tuple[Path, bytes], ...],
    stages: tuple[Path, ...],
    pathset_id: str,
) -> None:
    _require(
        not marker_path.is_symlink() and marker_path.is_file(),
        "incomplete publication marker is not a regular file",
    )
    expected_final_paths = [str(path.resolve(strict=False)) for path, _ in prepared]
    expected_stage_paths = [str(stage.resolve(strict=False)) for stage in stages]

    if not _path_entry_exists(manifest_path):
        _require(
            not any(_path_entry_exists(path) for path, _ in prepared)
            and not any(_path_entry_exists(stage) for stage in stages),
            "incomplete publication marker lacks manifest but payload paths exist",
        )
        _remove_control_without_payload(marker_path, manifest_path)
        return

    try:
        manifest = _load_publication_manifest(manifest_path)
    except NbuGlobalDedupError:
        _require(
            not any(_path_entry_exists(path) for path, _ in prepared)
            and not any(_path_entry_exists(stage) for stage in stages),
            "invalid incomplete manifest coexists with payload paths",
        )
        _remove_control_without_payload(marker_path, manifest_path)
        return

    targets = manifest["targets"]
    _require(
        manifest.get("publication_pathset_sha256") == pathset_id,
        "incomplete publication path-set identity mismatch",
    )
    marker_final_paths: list[str] = []
    marker_stage_paths: list[str] = []
    for row in targets:
        _require(type(row) is dict, "publication manifest target invalid")
        _require(
            set(row) == _PUBLICATION_TARGET_KEYS,
            "publication manifest target keys invalid",
        )
        final_value = row.get("path")
        stage_value = row.get("stage_path")
        expected_sha = row.get("sha256")
        _require(
            type(final_value) is str
            and final_value
            and type(stage_value) is str
            and stage_value
            and _is_sha256_hex(expected_sha),
            "publication manifest target semantics invalid",
        )
        marker_final_paths.append(final_value)
        marker_stage_paths.append(stage_value)
    _require(
        marker_final_paths == expected_final_paths
        and marker_stage_paths == expected_stage_paths,
        "incomplete publication manifest targets do not match requested outputs",
    )

    touched_dirs: set[Path] = set()
    for row in targets:
        final_path = Path(row["path"])
        stage_path = Path(row["stage_path"])
        expected_sha = row["sha256"]
        final_exists = _path_entry_exists(final_path)
        stage_exists = _path_entry_exists(stage_path)

        if final_exists:
            _require(
                stage_exists,
                f"incomplete final output has no owning stage: {final_path}",
            )
            _require(
                not final_path.is_symlink()
                and final_path.is_file()
                and not stage_path.is_symlink()
                and stage_path.is_file(),
                "incomplete publication payload path is not regular",
            )
            try:
                same_file = os.path.samefile(final_path, stage_path)
            except OSError as exc:
                raise NbuGlobalDedupError(
                    f"cannot verify incomplete publication ownership: {final_path}"
                ) from exc
            _require(
                same_file,
                f"incomplete publication final is not linked to its stage: {final_path}",
            )
            try:
                observed = stage_path.read_bytes()
            except OSError as exc:
                raise NbuGlobalDedupError(
                    f"cannot inspect incomplete publication stage: {stage_path}"
                ) from exc
            _require(
                _sha256(observed) == expected_sha,
                f"incomplete publication output digest mismatch: {final_path}",
            )
            final_path.unlink()
            touched_dirs.add(final_path.parent)

        if stage_exists:
            _require(
                not stage_path.is_symlink() and stage_path.is_file(),
                f"incomplete publication stage is not regular: {stage_path}",
            )
            stage_path.unlink()
            touched_dirs.add(stage_path.parent)

    for directory in sorted(touched_dirs, key=str):
        _fsync_directory(directory)
    manifest_path.unlink()
    _fsync_directory(manifest_path.parent)
    marker_path.unlink()
    _fsync_directory(marker_path.parent)


def _rollback_current_publication(
    *,
    marker_path: Path,
    manifest_path: Path,
    marker_created: bool,
    manifest_created: bool,
    linked_finals: list[Path],
    created_stages: list[Path],
) -> list[str]:
    errors: list[str] = []
    touched_dirs: set[Path] = set()
    for path in reversed(linked_finals):
        try:
            path.unlink(missing_ok=True)
            touched_dirs.add(path.parent)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    for path in reversed(created_stages):
        try:
            path.unlink(missing_ok=True)
            touched_dirs.add(path.parent)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if not errors:
        for directory in sorted(touched_dirs, key=str):
            try:
                _fsync_directory(directory)
            except NbuGlobalDedupError as exc:
                errors.append(str(exc))
    if not errors and manifest_created:
        try:
            manifest_path.unlink(missing_ok=True)
            _fsync_directory(manifest_path.parent)
        except (OSError, NbuGlobalDedupError) as exc:
            errors.append(f"{manifest_path}: {exc}")
    if not errors and marker_created:
        try:
            marker_path.unlink(missing_ok=True)
            _fsync_directory(marker_path.parent)
        except (OSError, NbuGlobalDedupError) as exc:
            errors.append(f"{marker_path}: {exc}")
    return errors


def _link_staged_output(stage_path: Path, final_path: Path) -> None:
    try:
        os.link(stage_path, final_path)
    except FileExistsError as exc:
        raise NbuGlobalDedupError(
            f"refusing to overwrite: {final_path}"
        ) from exc
    except OSError as exc:
        raise NbuGlobalDedupError(
            f"cannot atomically publish output: {final_path}"
        ) from exc


def _cleanup_committed_publication_residue(
    manifest_path: Path,
    stages: tuple[Path, ...],
) -> None:
    touched_dirs: set[Path] = set()
    for stage_path in stages:
        try:
            stage_path.unlink(missing_ok=True)
            touched_dirs.add(stage_path.parent)
        except OSError:
            return
    try:
        manifest_path.unlink(missing_ok=True)
        touched_dirs.add(manifest_path.parent)
    except OSError:
        return
    for directory in sorted(touched_dirs, key=str):
        try:
            _fsync_directory(directory)
        except NbuGlobalDedupError:
            return


def _publish_json_outputs(
    outputs: tuple[tuple[Path, Mapping[str, Any]], ...],
) -> None:
    prepared_list: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    resolved_seen: set[Path] = set()
    for path, value in outputs:
        resolved = path.resolve(strict=False)
        _require(
            path not in seen and resolved not in resolved_seen,
            f"duplicate output path: {path}",
        )
        seen.add(path)
        resolved_seen.add(resolved)
        prepared_list.append((path, _canonical(dict(value)) + b"\n"))
    prepared = tuple(prepared_list)
    _require(bool(prepared), "publication output set must not be empty")

    for path, _ in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)

    marker_path, manifest_path, stages, pathset_id = _publication_control_paths(prepared)
    control_paths = (marker_path, manifest_path, *stages)
    resolved_controls = tuple(path.resolve(strict=False) for path in control_paths)
    _require(
        len(set(resolved_controls)) == len(resolved_controls)
        and not (set(resolved_controls) & resolved_seen),
        "publication control path collision",
    )

    if _path_entry_exists(marker_path):
        _recover_incomplete_publication(
            marker_path,
            manifest_path,
            prepared,
            stages,
            pathset_id,
        )

    for path, _ in prepared:
        _require(not _path_entry_exists(path), f"refusing to overwrite: {path}")
    _require(
        not _path_entry_exists(manifest_path),
        f"stale publication manifest exists without marker: {manifest_path}",
    )
    for stage in stages:
        _require(
            not _path_entry_exists(stage),
            f"stale publication stage exists without marker: {stage}",
        )

    manifest, manifest_payload = _publication_manifest(prepared, stages, pathset_id)
    _require(
        [row["stage_path"] for row in manifest["targets"]]
        == [str(stage.resolve(strict=False)) for stage in stages],
        "publication stage derivation drift",
    )

    marker_created = False
    manifest_created = False
    linked_finals: list[Path] = []
    created_stages: list[Path] = []
    try:
        _write_create_only_durable(marker_path, b"")
        marker_created = True
        _fsync_directory(marker_path.parent)

        _write_create_only_durable(manifest_path, manifest_payload)
        manifest_created = True
        _fsync_directory(manifest_path.parent)

        for (path, payload), stage_path in zip(prepared, stages, strict=True):
            _write_create_only_durable(stage_path, payload)
            created_stages.append(stage_path)
            _fsync_directory(stage_path.parent)

        for (final_path, _), stage_path in zip(prepared, stages, strict=True):
            _link_staged_output(stage_path, final_path)
            linked_finals.append(final_path)
            _fsync_directory(final_path.parent)

        # Marker removal is the sole terminal commit point.  Stages and manifest
        # deliberately remain present until after this durable state transition
        # so an interrupted pre-commit run can prove final-path ownership.
        marker_path.unlink()
        _fsync_directory(marker_path.parent)
        marker_created = False
    except Exception as exc:
        if isinstance(exc, PublicationWriteCleanupError):
            # Preserve the durable marker/control state.  Recovery must decide
            # ownership on the next invocation; do not erase evidence of an
            # incomplete create whose local cleanup already failed.
            raise
        rollback_errors = _rollback_current_publication(
            marker_path=marker_path,
            manifest_path=manifest_path,
            marker_created=marker_created,
            manifest_created=manifest_created,
            linked_finals=linked_finals,
            created_stages=created_stages,
        )
        if rollback_errors:
            raise NbuGlobalDedupError(
                "output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise

    _cleanup_committed_publication_residue(manifest_path, stages)



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
        _require(evidence.get("schema_version") == SCHEMA, "two-clean child schema drift")
        _require(
            evidence.get("execution_profile") == "LOCAL_FREE",
            "two-clean child execution profile drift",
        )
        _require(evidence.get("pinned_main_sha") == EXPECTED_MAIN, "two-clean pinned-main drift")
        _require(
            evidence.get("execution_claim_issue") == EXECUTION_CLAIM
            and evidence.get("execution_pr") == EXECUTION_PR,
            "two-clean child execution lineage drift",
        )
        baseline = evidence.get("baseline_v8")
        _require(type(baseline) is dict, "two-clean baseline evidence missing")
        _verify_removal_proof(baseline.get("nomis1864_deauthorization"))
        _require(
            baseline.get("v7_head_sha") == v8.EXPECTED_V7_HEAD
            and type(baseline.get("source_object_count")) is int
            and baseline.get("source_object_count") == EXPECTED_BASE_OBJECTS
            and type(baseline.get("payload_bytes")) is int
            and baseline.get("payload_bytes") == EXPECTED_BASE_BYTES
            and baseline.get("payload_bytes_semantics") == PAYLOAD_BYTES_SEMANTICS
            and baseline.get("declared_capacity_bytes") == EXPECTED_BASE_BYTES
            and type(baseline.get("comparison_payload_bytes")) is int
            and baseline.get("comparison_payload_bytes") > 0,
            "two-clean baseline authority drift",
        )
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
            type(nbu_evidence.get("source_object_count")) is int
            and nbu_evidence.get("source_object_count") == EXPECTED_NBU_OBJECTS
            and type(nbu_evidence.get("payload_bytes")) is int
            and nbu_evidence.get("payload_bytes") == EXPECTED_NBU_BYTES
            and nbu_evidence.get("payload_bytes_semantics") == PAYLOAD_BYTES_SEMANTICS
            and nbu_evidence.get("declared_capacity_bytes") == EXPECTED_NBU_BYTES
            and nbu_evidence.get("comparison_payload_bytes") == EXPECTED_NBU_BYTES,
            "two-clean NBU cardinality drift",
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
        _require(
            type(combined.get("source_object_count")) is int
            and combined.get("source_object_count") == EXPECTED_COMBINED_OBJECTS
            and type(combined.get("payload_bytes")) is int
            and combined.get("payload_bytes") == EXPECTED_COMBINED_BYTES
            and combined.get("payload_bytes_semantics") == PAYLOAD_BYTES_SEMANTICS
            and combined.get("declared_capacity_bytes") == EXPECTED_COMBINED_BYTES
            and type(combined.get("comparison_payload_bytes")) is int
            and combined.get("comparison_payload_bytes")
            == baseline["comparison_payload_bytes"] + nbu_evidence["comparison_payload_bytes"],
            "two-clean combined cardinality drift",
        )
        matcher = evidence.get("matcher_execution")
        _require(type(matcher) is dict, "two-clean matcher evidence missing")
        _require(matcher.get("engine") == "MERGED_PR_1459", "two-clean matcher engine drift")
        _require(matcher.get("report_sha256") == report_sha, "two-clean matcher report drift")
        _require(
            matcher.get("performance_equivalence_authority") == "MERGED_PR_1459"
            and matcher.get("incumbent_runtime_attested") is True
            and matcher.get("all_pairs_reference_executed") is False,
            "two-clean matcher authority drift",
        )
        _require(
            evidence.get("survivor_authority_sha256") == survivor_sha,
            "two-clean evidence/survivor identity drift",
        )
        content_boundary = evidence.get("content_boundary")
        expected_content_boundary = {
            "raw_text_emitted": False,
            "raw_candidate_written_to_durable_evidence": False,
            "dedup_report_text_free": True,
            "survivor_authority_text_free": True,
        }
        _require(
            type(content_boundary) is dict
            and all(
                type(content_boundary.get(key)) is bool
                and content_boundary.get(key) is expected
                for key, expected in expected_content_boundary.items()
            ),
            "two-clean child content boundary drift",
        )
        truth = evidence.get("truth_boundary")
        _require(type(truth) is dict, "two-clean truth boundary missing")
        expected_truth = {
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
        }
        for key, expected in expected_truth.items():
            _require(
                type(truth.get(key)) is type(expected) and truth.get(key) == expected,
                f"two-clean truth boundary drift: {key}",
            )

    _require(
        first_evidence["baseline_v8"]["comparison_payload_bytes"]
        == second_evidence["baseline_v8"]["comparison_payload_bytes"],
        "two-clean baseline comparison payload bytes differ",
    )
    _require(
        first_evidence["combined"]["comparison_payload_bytes"]
        == second_evidence["combined"]["comparison_payload_bytes"],
        "two-clean combined comparison payload bytes differ",
    )

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


def _validate_parent_child_budget_binding(
    first_evidence: Mapping[str, Any],
    second_evidence: Mapping[str, Any],
    *,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> None:
    expected = {
        "max_candidate_pairs": max_candidate_pairs,
        "max_index_postings": max_index_postings,
        "max_pair_expansions": max_pair_expansions,
    }
    for evidence in (first_evidence, second_evidence):
        matcher = evidence.get("matcher_execution")
        _require(type(matcher) is dict, "two-clean child matcher evidence missing")
        for key, expected_value in expected.items():
            _require(
                type(matcher.get(key)) is int
                and matcher.get(key) == expected_value,
                f"two-clean child work budget drift: {key}",
            )


def _validate_parent_child_repository_binding(
    first_evidence: Mapping[str, Any],
    second_evidence: Mapping[str, Any],
    *,
    expected_main_blobs: Mapping[str, str],
    expected_product_blobs: Mapping[str, str],
) -> None:
    for evidence in (first_evidence, second_evidence):
        main_blobs = evidence.get("main_authority_path_blobs")
        product_blobs = evidence.get("product_path_blobs")
        _require(
            type(main_blobs) is dict
            and _canonical(main_blobs) == _canonical(dict(expected_main_blobs)),
            "two-clean child main-authority blob drift",
        )
        _require(
            type(product_blobs) is dict
            and _canonical(product_blobs) == _canonical(dict(expected_product_blobs)),
            "two-clean child Product blob drift",
        )


def _validate_parent_aggregate_binding(
    authority: Mapping[str, Any],
    *,
    orchestration_head: str,
    expected_intake_receipt_sha: str,
) -> None:
    _require(
        authority.get("execution_head_sha") == orchestration_head,
        "two-clean aggregate execution head drift",
    )
    materialization_authority = authority.get("materialization_authority")
    _require(
        type(materialization_authority) is dict
        and materialization_authority.get("intake_receipt_identity_sha256")
        == expected_intake_receipt_sha,
        "two-clean parent/child intake receipt drift",
    )


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
    parent_main_blobs, parent_product_blobs = verify_repository_authority()
    _validate_work_budgets(
        max_candidate_pairs,
        max_index_postings,
        max_pair_expansions,
    )
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
    expected_intake_receipt_sha = first_projection.receipt["receipt_identity_sha256"]
    _require(
        expected_intake_receipt_sha
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
                str(run_dir / "bulk-workspace"),
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
        _validate_parent_child_repository_binding(
            first_evidence,
            second_evidence,
            expected_main_blobs=parent_main_blobs,
            expected_product_blobs=parent_product_blobs,
        )
        _validate_parent_child_budget_binding(
            first_evidence,
            second_evidence,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
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
        _validate_parent_aggregate_binding(
            authority,
            orchestration_head=orchestration_head,
            expected_intake_receipt_sha=expected_intake_receipt_sha,
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
    _validate_work_budgets(
        max_candidate_pairs,
        max_index_postings,
        max_pair_expansions,
    )
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = _reconstruct_v8_with_historical_namespace(
        v7_root=v7_root,
        bulk_workspace=bulk_workspace,
        config=config,
    )
    _require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "V8 base object count drift")
    _require(
        _declared_capacity_bytes(base_inventory, base_payloads, label="V8 base")
        == EXPECTED_BASE_BYTES,
        "V8 base declared-capacity drift",
    )
    base_comparison_payload_bytes = sum(len(raw) for raw in base_payloads.values())
    _require(base_comparison_payload_bytes > 0, "V8 base comparison payload is empty")

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
    nbu_comparison_payload_bytes = sum(len(raw) for raw in extension_payloads.values())
    _require(
        nbu_comparison_payload_bytes == EXPECTED_NBU_BYTES,
        "NBU projection comparison-payload byte total drift",
    )
    _require(
        _declared_capacity_bytes(
            {"sources": extension_sources},
            extension_payloads,
            label="NBU projection",
        )
        == EXPECTED_NBU_BYTES,
        "NBU projection declared-capacity drift",
    )
    inventory, payloads = _compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    _require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined source count drift")
    _require(
        _declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )
    combined_comparison_payload_bytes = sum(len(raw) for raw in payloads.values())
    _require(
        combined_comparison_payload_bytes
        == base_comparison_payload_bytes + nbu_comparison_payload_bytes,
        "combined comparison-payload byte arithmetic drift",
    )

    report, indexed_elapsed = _execute_indexed_matcher(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    _validate_nbu_report_binding(report, extension_sources, extension_payloads)
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
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "comparison_payload_bytes": base_comparison_payload_bytes,
            "nomis1864_deauthorization": removal,
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
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_NBU_BYTES,
            "comparison_payload_bytes": nbu_comparison_payload_bytes,
            "intake_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": EXPECTED_COMBINED_BYTES,
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "comparison_payload_bytes": combined_comparison_payload_bytes,
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
