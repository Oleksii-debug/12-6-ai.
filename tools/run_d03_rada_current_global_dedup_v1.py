#!/usr/bin/env python3
"""Execute the exact current Rada snapshot through incumbent indexed global dedup.

This is an execution-only, zero-credit carrier. It reconstructs the exact
Nomis1864-free V8 incumbent source graph, validates the current Rada accepted
candidate against the Product authority from PR #2808, and delegates every
global duplicate decision to the independently qualified PR #1459 indexed
executor under exact V3 runtime attestation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_expanded_global_dedup_v9 as v9_runner
import run_d03_nomis_free_clean_successor_v1 as clean_successor
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import external_llm_provenance_quarantine_v1 as quarantine
from twelve_six.data import incumbent_dedup_indexed_execution as indexed
from twelve_six.data import rada_current_snapshot_dedup_adapter_v1 as rada
from twelve_six.data import rada_current_snapshot_qp_authority_v1 as rada_authority


SCHEMA = "12-6.d03-rada-current-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-rada-current-global-dedup-survivors.v1"
EXPECTED_PRODUCT_PARENT = "fb49b7e212444547219df2bd2aa466db955d51e5"
EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_BYTES = 6_093_965
EXPECTED_RADA_OBJECTS = 101_733
EXPECTED_RADA_BYTES = 192_393_157
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_RADA_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_RADA_BYTES
EXPECTED_MAX_CANDIDATE_PAIRS = 5_000_000
EXPECTED_MAX_INDEX_POSTINGS = 50_000_000
EXPECTED_MAX_PAIR_EXPANSIONS = 50_000_000
PAYLOAD_BYTES_SEMANTICS = "DECLARED_CAPACITY_BYTES"
AUTHORITY_PATH = (
    ROOT / "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
)

DEPENDENCY_PATHS = (
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
    "src/twelve_six/data/rada_current_snapshot_qp_authority_v1.py",
    "src/twelve_six/data/rada_current_snapshot_dedup_adapter_v1.py",
    "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json",
    "tools/normalize_d03_rada_bulk_html.py",
    "configs/data/d03_rada_bulk_normalization_v1.json",
    "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json",
)

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


class RadaCurrentGlobalDedupError(RuntimeError):
    """Raised when current Rada global-dedup execution fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCurrentGlobalDedupError(message)


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


def _git(*args: str, check: bool = True) -> str:
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
        raise RadaCurrentGlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RadaCurrentGlobalDedupError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return proc.stdout.strip()


def _bind_execution_head(expected_execution_head: str) -> str:
    _require(
        type(expected_execution_head) is str
        and len(expected_execution_head) == 40
        and all(char in "0123456789abcdef" for char in expected_execution_head),
        "expected execution head must be lowercase 40-hex SHA",
    )
    observed = _git("rev-parse", "HEAD")
    _require(observed == expected_execution_head, "execution HEAD drift")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            EXPECTED_PRODUCT_PARENT,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _require(
        ancestor.returncode == 0,
        "expected Product parent is not an ancestor of execution HEAD",
    )
    return observed


def verify_repository_authority() -> dict[str, str]:
    """Bind every science/runtime dependency to exact PR #2808 Product bytes."""

    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *DEPENDENCY_PATHS,
        ],
        check=False,
        capture_output=True,
    )
    _require(dirty.returncode == 0, "dependency worktree drift")
    observed: dict[str, str] = {}
    for path in DEPENDENCY_PATHS:
        expected = _git("rev-parse", f"{EXPECTED_PRODUCT_PARENT}:{path}")
        current = _git("rev-parse", f"HEAD:{path}")
        _require(
            len(expected) == 40 and current == expected,
            f"Product dependency drift: {path}",
        )
        physical = _git("hash-object", str(ROOT / path))
        _require(physical == current, f"physical dependency drift: {path}")
        observed[path] = current
    return dict(sorted(observed.items()))


def _validate_work_budgets(
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> None:
    expected = (
        ("max_candidate_pairs", max_candidate_pairs, EXPECTED_MAX_CANDIDATE_PAIRS),
        ("max_index_postings", max_index_postings, EXPECTED_MAX_INDEX_POSTINGS),
        ("max_pair_expansions", max_pair_expansions, EXPECTED_MAX_PAIR_EXPANSIONS),
    )
    for label, value, wanted in expected:
        _require(
            type(value) is int and value == wanted,
            f"{label} must equal independently selected Rada-scale bound {wanted}",
        )


def _verify_removal_proof(proof: Mapping[str, Any]) -> None:
    expected = {
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
    _require(
        type(proof) is dict and _canonical(proof) == _canonical(expected),
        "Nomis deauthorization proof drift",
    )


def _verify_clean_payload_graph(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    quarantine_authority: Mapping[str, Any],
) -> None:
    rows = inventory.get("sources")
    _require(type(rows) is list and bool(rows), "clean graph source rows missing")
    ids = [row.get("source_id") for row in rows if type(row) is dict]
    _require(
        len(ids) == len(rows)
        and all(type(source_id) is str and bool(source_id) for source_id in ids)
        and len(set(ids)) == len(ids)
        and set(ids) == set(payloads)
        and all(type(raw) is bytes for raw in payloads.values()),
        "clean graph inventory/payload coverage mismatch",
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
            f"{label} declared capacity invalid",
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
    v9_runner.validate_v7_checkout(v7_root)
    clean_successor.validate_runtime_bindings(ROOT)
    quarantine_authority = json.loads(
        (ROOT / clean_successor.QUARANTINE_CONFIG_PATH).read_text(encoding="utf-8")
    )
    v7, _, historical_inventory, historical_payloads = v8._capture_terminal_v7(
        v7_root,
        config,
    )
    clean_inventory, clean_payloads, removal = clean_successor.deauthorize_exact_nomis(
        historical_inventory,
        historical_payloads,
        quarantine_authority,
        quarantine,
    )
    _verify_removal_proof(removal)
    _, bulk_rows, bulk_payloads = v8._materialize_bulk(
        ROOT,
        bulk_workspace,
        config,
    )
    existing_ids = {row["source_id"] for row in clean_inventory["sources"]}
    _require(
        not (existing_ids & set(bulk_payloads)),
        "clean base/bulk source-id collision",
    )
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
    _require(
        all(
            row.get("source_family") != rada.SOURCE_FAMILY
            for row in inventory["sources"]
        ),
        "incumbent base already contains Rada family; replacement gate required",
    )
    return v7.v6.v3, inventory, payloads, removal


def _reconstruct_v8_with_historical_namespace(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    for module_name in _HISTORICAL_MATCHER_MODULES:
        _require(
            module_name not in sys.modules,
            f"historical matcher preloaded: {module_name}",
        )

    twelve_six_pkg = importlib.import_module("twelve_six")
    data_pkg = importlib.import_module("twelve_six.data")
    current_package_path = list(twelve_six_pkg.__path__)
    current_data_path = list(data_pkg.__path__)
    historical_package = str((v7_root / "src" / "twelve_six").resolve(strict=True))
    historical_data = str(
        (v7_root / "src" / "twelve_six" / "data").resolve(strict=True)
    )
    _require(
        historical_package not in current_package_path,
        "V7 package path already active",
    )
    _require(
        historical_data not in current_data_path,
        "V7 data path already active",
    )

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
        _require(module is not None, f"historical matcher did not load: {module_name}")
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        _require(
            module_path.is_relative_to(historical_root),
            f"historical matcher escaped exact V7 worktree: {module_name}",
        )
    return matcher, inventory, payloads, removal


def _compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    rada_sources: list[dict[str, Any]],
    rada_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows = base_inventory.get("sources")
    _require(type(base_inventory) is dict, "base inventory must be exact dict")
    _require(type(rows) is list and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if type(row) is dict}
    _require(
        len(base_ids) == len(rows)
        and None not in base_ids
        and base_ids == set(base_payloads),
        "base inventory/payload identity drift",
    )
    _require(
        all(
            type(row) is dict and row.get("source_family") != rada.SOURCE_FAMILY
            for row in rows
        ),
        "base contains pre-existing Rada family; append is forbidden",
    )

    rada_ids = {
        row.get("source_id") for row in rada_sources if type(row) is dict
    }
    _require(
        len(rada_ids) == len(rada_sources)
        and None not in rada_ids
        and rada_ids == set(rada_payloads),
        "Rada inventory/payload identity drift",
    )
    _require(not (base_ids & rada_ids), "Rada source-id collision with base graph")
    _require(
        all(type(key) is str and type(value) is bytes for key, value in base_payloads.items()),
        "base payload map must be exact str->bytes",
    )
    _require(
        all(type(key) is str and type(value) is bytes for key, value in rada_payloads.items()),
        "Rada payload map must be exact str->bytes",
    )

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(rada_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 graph plus exact current Rada replay "
        "candidate authenticated by PR #2808; incumbent base is explicitly "
        "Rada-family-free so this is not additive with a historical Rada revision. "
        "Pair decisions delegate only to terminal PR #824 V3 semantics through "
        "merged performance-equivalent indexed executor PR #1459."
    )
    payloads = dict(base_payloads)
    payloads.update(rada_payloads)
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
    indexed.attest_incumbent_runtime(matcher)
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
    matcher.verify_report(report)
    return report, elapsed


def _validate_rada_report_binding(
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
        len(extension_sources) == len(extension_payloads) == EXPECTED_RADA_OBJECTS,
        "Rada projected source/payload cardinality drift",
    )
    expected_ids = [row.get("source_id") for row in extension_sources]
    _require(
        all(type(value) is str and bool(value) for value in expected_ids)
        and len(expected_ids) == len(set(expected_ids))
        and set(expected_ids) == set(extension_payloads),
        "Rada projected source identities drift",
    )
    for expected in extension_sources:
        source_id = expected["source_id"]
        observed = by_id.get(source_id)
        _require(type(observed) is dict, f"Rada source missing from report: {source_id}")
        payload = extension_payloads[source_id]
        for field in (
            "source_family",
            "modality",
            "evidence_status",
            "declared_capacity_bytes",
        ):
            wanted = expected.get(field)
            _require(
                type(observed.get(field)) is type(wanted)
                and observed.get(field) == wanted,
                f"Rada report field drift for {source_id}: {field}",
            )
        stable_origin_id = expected.get("stable_origin_id")
        stable_object_id = expected.get("stable_object_id")
        _require(
            type(stable_origin_id) is str and bool(stable_origin_id),
            f"Rada stable origin invalid: {source_id}",
        )
        _require(
            type(stable_object_id) is str and bool(stable_object_id),
            f"Rada stable object invalid: {source_id}",
        )
        _require(
            observed.get("stable_origin_id_sha256")
            == _sha256(stable_origin_id.encode("utf-8")),
            f"Rada stable origin drift: {source_id}",
        )
        _require(
            observed.get("stable_object_id_sha256")
            == _sha256(stable_object_id.encode("utf-8")),
            f"Rada stable object drift: {source_id}",
        )
        _require(
            observed.get("verified_raw_bytes") == len(payload)
            and observed.get("verified_raw_sha256") == _sha256(payload),
            f"Rada raw payload verification drift: {source_id}",
        )
        _require(
            observed.get("comparison_policy") == "DATA232_GENERIC_FROM_RAW",
            f"Rada comparison policy drift: {source_id}",
        )
        _require(
            observed.get("comparison_payload_bytes") == len(payload)
            and observed.get("comparison_payload_sha256") == _sha256(payload),
            f"Rada comparison payload drift: {source_id}",
        )


def _validated_terminal_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(
        report.get("source_count") == EXPECTED_COMBINED_OBJECTS,
        "report source count drift",
    )
    terminal = report.get("terminal_candidates")
    _require(type(terminal) is dict, "terminal matcher summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    clusters = terminal.get("duplicate_cluster_count")
    _require(
        type(before) is int and before == EXPECTED_COMBINED_BYTES,
        "pre-dedup declared capacity drift",
    )
    _require(type(after) is int and 0 < after <= before, "post-dedup capacity invalid")
    _require(
        type(discount) is int and discount == before - after,
        "duplicate discount arithmetic drift",
    )
    _require(type(clusters) is int and clusters >= 0, "duplicate cluster count invalid")
    return terminal


def _validate_survivor_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> None:
    terminal = _validated_terminal_summary(report)
    projection_core = dict(projection)
    claimed = projection_core.pop("survivor_authority_sha256", None)
    _require(
        type(claimed) is str
        and len(claimed) == 64
        and _sha256(_canonical(projection_core)) == claimed,
        "survivor projection self-hash mismatch",
    )
    _require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
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
        "survivor source ids invalid",
    )
    _require(
        projection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "survivor pre-dedup object count drift",
    )
    _require(
        projection.get("pre_dedup_declared_capacity_bytes")
        == EXPECTED_COMBINED_BYTES,
        "survivor pre-dedup bytes drift",
    )
    _require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "survivor post-dedup bytes drift",
    )
    _require(
        projection.get("duplicate_discount_bytes")
        == terminal.get("duplicate_discount_bytes"),
        "survivor duplicate discount drift",
    )


def _outer_survivor_authority(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = report.get("sources")
    _require(type(sources) is list, "dedup report source vector missing")
    by_id = {
        row["source_id"]: row
        for row in sources
        if type(row) is dict and type(row.get("source_id")) is str
    }
    _require(len(by_id) == len(sources), "dedup source identity drift")
    survivor_ids = projection.get("survivor_source_ids")
    _require(type(survivor_ids) is list, "survivor id vector missing")
    _require(
        all(source_id in by_id for source_id in survivor_ids),
        "survivor references unknown source",
    )
    rada_ids = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == rada.SOURCE_FAMILY
    ]
    rada_bytes = 0
    for source_id in rada_ids:
        value = by_id[source_id].get("declared_capacity_bytes")
        _require(
            type(value) is int and value >= 0,
            "Rada survivor byte count invalid",
        )
        rada_bytes += value

    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": report.get("report_sha256"),
        "selection_projection_schema": projection.get("schema_version"),
        "selection_projection_sha256": projection.get("survivor_authority_sha256"),
        "pre_dedup_source_object_count": projection.get(
            "pre_dedup_source_object_count"
        ),
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
        "rada_survivor_source_ids": rada_ids,
        "rada_survivor_source_object_count": len(rada_ids),
        "rada_survivor_declared_capacity_bytes": rada_bytes,
        "rights_scope": "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
        "rights_recheck_for_training_required": True,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
    }
    return {
        **core,
        "survivor_authority_sha256": _sha256(_canonical(core)),
    }


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
        _require(
            environment in {"github-hosted", "self-hosted"},
            "runner environment missing",
        )
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


def _max_rss_kib() -> int | None:
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


def _publish_output_directory(
    output_dir: Path,
    values: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(type(values) is dict and bool(values), "output set is empty")
    _require(
        not output_dir.exists() and not output_dir.is_symlink(),
        f"refusing to overwrite output directory: {output_dir}",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for filename, value in values.items():
            _require(
                type(filename) is str
                and filename
                and Path(filename).name == filename,
                "output filename is unsafe",
            )
            payload = _canonical(dict(value)) + b"\n"
            target = temporary / filename
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    output_dir: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = _bind_execution_head(expected_execution_head)
    dependency_blobs = verify_repository_authority()
    _validate_work_budgets(
        max_candidate_pairs,
        max_index_postings,
        max_pair_expansions,
    )
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = (
        _reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    base_comparison_bytes = sum(len(raw) for raw in base_payloads.values())
    _require(base_comparison_bytes > 0, "base comparison payload is empty")

    projection = rada.validate_and_project_current_rada_candidate(
        candidate_jsonl,
        AUTHORITY_PATH,
        repository_root=ROOT,
        retain_payloads=True,
    )
    _require(
        projection.sources is not None and projection.payloads is not None,
        "Rada projection payloads missing",
    )
    _require(
        projection.receipt.get("requires_replace_existing_source_family") is True
        and projection.receipt.get("must_not_append_to_existing_source_family")
        is True,
        "Rada replacement semantics weakened",
    )
    _require(
        projection.receipt.get("bulk_corpus_admission_granted") is False
        and projection.receipt.get("training_authority_granted") is False
        and projection.receipt.get("rights_recheck_for_training_required") is True,
        "Rada rights boundary widened",
    )
    extension_sources = [dict(row) for row in projection.sources]
    extension_payloads = dict(projection.payloads)
    _require(
        len(extension_sources) == EXPECTED_RADA_OBJECTS,
        "Rada projection count drift",
    )
    rada_comparison_bytes = sum(len(raw) for raw in extension_payloads.values())
    _require(
        rada_comparison_bytes == EXPECTED_RADA_BYTES,
        "Rada comparison-payload byte total drift",
    )
    _require(
        _declared_capacity_bytes(
            {"sources": extension_sources},
            extension_payloads,
            label="Rada projection",
        )
        == EXPECTED_RADA_BYTES,
        "Rada declared-capacity drift",
    )

    inventory, payloads = _compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    _require(
        len(payloads) == EXPECTED_COMBINED_OBJECTS,
        "combined source-object count drift",
    )
    _require(
        _declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )
    combined_comparison_bytes = sum(len(raw) for raw in payloads.values())
    _require(
        combined_comparison_bytes == base_comparison_bytes + rada_comparison_bytes,
        "combined comparison-payload arithmetic drift",
    )

    report, indexed_elapsed = _execute_indexed_matcher(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    _validate_rada_report_binding(report, extension_sources, extension_payloads)
    terminal = _validated_terminal_summary(report)
    selection = v9_semantics._derive_survivors(report)
    _validate_survivor_projection(report, selection)
    survivors = _outer_survivor_authority(report, selection)

    max_rss_kib = _max_rss_kib()
    _require(
        type(max_rss_kib) is int and max_rss_kib > 0,
        "positive measured max RSS unavailable",
    )

    evidence_core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE_GITHUB_HOSTED",
        "runtime_environment": _runtime_environment(),
        "execution_head_sha": execution_head,
        "product_parent_sha": EXPECTED_PRODUCT_PARENT,
        "dependency_path_blobs": dependency_blobs,
        "baseline_v8_nomissafe": {
            "v7_head_sha": v8.EXPECTED_V7_HEAD,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "comparison_payload_bytes": base_comparison_bytes,
            "nomis1864_deauthorization": removal,
        },
        "current_rada": {
            "replay_authority_identity_sha256": (
                rada_authority.AUTHORITY_IDENTITY_SHA256
            ),
            "authority_file_sha256": (
                rada_authority.CANONICAL_AUTHORITY_FILE_SHA256
            ),
            "candidate_jsonl_sha256": (
                "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
            ),
            "source_object_count": EXPECTED_RADA_OBJECTS,
            "declared_capacity_bytes": EXPECTED_RADA_BYTES,
            "comparison_payload_bytes": rada_comparison_bytes,
            "projection_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
            "rights_scope": "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
            "bulk_corpus_admission_granted": False,
            "training_authority_granted": False,
            "rights_recheck_for_training_required": True,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "comparison_payload_bytes": combined_comparison_bytes,
            "indexed_report_sha256": report["report_sha256"],
            "post_dedup_conservative_unique_bytes": terminal[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
            "duplicate_cluster_count": terminal["duplicate_cluster_count"],
            "incumbent_base_rada_family_free": True,
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
        "survivor_authority_sha256": survivors[
            "survivor_authority_sha256"
        ],
        "content_boundary": {
            "raw_text_emitted": False,
            "raw_candidate_written_to_output": False,
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
            "rights_recheck_for_training_executed": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }
    _publish_output_directory(
        output_dir,
        {
            "dedup-report.json": report,
            "survivor-authority.json": survivors,
            "execution-evidence.json": evidence,
        },
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=EXPECTED_MAX_CANDIDATE_PAIRS,
    )
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=EXPECTED_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=EXPECTED_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()
    try:
        evidence = execute(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            candidate_jsonl=args.candidate_jsonl,
            output_dir=args.output_dir,
            expected_execution_head=args.expected_execution_head,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (
        RadaCurrentGlobalDedupError,
        rada.RadaCurrentSnapshotDedupAdapterError,
        rada_authority.RadaCurrentSnapshotAuthorityError,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_CURRENT_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("MATCHER_REPORT_SHA256=" + evidence["combined"]["indexed_report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TRAINING_AUTHORIZED_BYTES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
