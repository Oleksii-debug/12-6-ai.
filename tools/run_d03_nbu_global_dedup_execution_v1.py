#!/usr/bin/env python3
"""Execute exact source-admitted NBU through the incumbent global-dedup authority.

This runner is execution glue only.  It composes the exact merged NBU intake
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
import os
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
from twelve_six.data import nbu_dedup_intake as nbu
from twelve_six.data import incumbent_dedup_indexed_execution as indexed

SCHEMA = "12-6.d03-nbu-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-nbu-global-dedup-survivors.v1"
EXPECTED_MAIN = "ba9e49cedba4a110e1c4f7d83702e8fcf8a42461"
EXECUTION_CLAIM = 2398
EXECUTION_PR = 2454
CARRIER_PATH = "tools/run_d03_nbu_global_dedup_execution_v1.py"
EXPECTED_BASE_OBJECTS = 264
EXPECTED_BASE_BYTES = 6_095_624
EXPECTED_NBU_OBJECTS = nbu.CANDIDATE_RECORDS
EXPECTED_NBU_BYTES = nbu.CANDIDATE_TEXT_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_NBU_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_NBU_BYTES

AUTHORITY_PATHS = (
    "src/twelve_six/data/nbu_dedup_intake.py",
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


class NBUGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NBUGlobalDedupError(message)


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
        raise NBUGlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise NBUGlobalDedupError(f"git {' '.join(args)} failed: {detail}")
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
    _require(not (base_ids & extension_ids), "NBU source-id collision with incumbent graph")

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed V8 authority plus exact PR #1025 source-admitted NBU "
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
    nbu_survivors = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == nbu.SOURCE_FAMILY
    ]
    nbu_survivor_bytes = 0
    for source_id in nbu_survivors:
        declared_capacity = by_id[source_id].get("declared_capacity_bytes")
        _require(
            type(declared_capacity) is int and declared_capacity >= 0,
            "NBU survivor declared capacity must be exact nonnegative int",
        )
        nbu_survivor_bytes += declared_capacity
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
        "nbu": {
            "materialization_head": nbu.MATERIALIZATION_HEAD,
            "materialization_run": nbu.MATERIALIZATION_RUN,
            "materialization_job": nbu.MATERIALIZATION_JOB,
            "materialization_artifact": nbu.MATERIALIZATION_ARTIFACT,
            "independent_audit_issue": nbu.MATERIALIZATION_AUDIT,
            "candidate_sha256": nbu.CANDIDATE_SHA256,
            "source_object_count": EXPECTED_NBU_OBJECTS,
            "payload_bytes": EXPECTED_NBU_BYTES,
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
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
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
        materialization_evidence_json=args.materialization_evidence_json,
        output_report=args.output_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        expected_execution_head=args.expected_execution_head,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_NBU_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
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
