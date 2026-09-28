#!/usr/bin/env python3
"""Execute exact source-admitted Caselaw through the incumbent global-dedup authority.

This runner is execution glue only.  It composes the exact merged Caselaw intake
with the exact reconstructed V8 graph, runs the incumbent V3 all-pairs authority
and the merged performance-equivalent indexed executor, requires byte-identical
reports, and emits only text-free zero-credit evidence.
"""
from __future__ import annotations

import argparse
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
from twelve_six.data import caselaw_source_admitted_dedup_intake as caselaw
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import incumbent_dedup_indexed_execution as indexed

SCHEMA = "12-6.d03-caselaw-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-caselaw-global-dedup-survivors.v1"
EXPECTED_MAIN = "bd2d445dfd8fbd7ec6759c1398bb913f4e0c0093"
EXECUTION_CLAIM = 2257
EXECUTION_PR = 2259
CASELAW_FINAL_HEAD = "deaf0730fe04a12e9abb8f3cecb14d6ad2cc7a4d"
EXPECTED_BASE_OBJECTS = 264
EXPECTED_BASE_BYTES = 6_095_624
EXPECTED_CASELAW_OBJECTS = 5_658
EXPECTED_CASELAW_BYTES = 5_962_147
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_CASELAW_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_CASELAW_BYTES

AUTHORITY_PATHS = (
    "src/twelve_six/data/caselaw_source_admitted_dedup_intake.py",
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


class CaselawGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaselawGlobalDedupError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
        raise CaselawGlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise CaselawGlobalDedupError(f"git {' '.join(args)} failed: {detail}")
    return proc


def verify_repository_authority() -> dict[str, str]:
    """Bind authority-bearing runtime files to the exact main that merged #1459."""
    ancestor = _git("merge-base", "--is-ancestor", EXPECTED_MAIN, "HEAD", check=False)
    _require(
        ancestor.returncode == 0,
        f"pinned main {EXPECTED_MAIN} is not an ancestor of execution HEAD",
    )
    blobs: dict[str, str] = {}
    for path in AUTHORITY_PATHS:
        expected = _git("rev-parse", f"{EXPECTED_MAIN}:{path}").stdout.strip()
        observed = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        _require(bool(expected) and observed == expected, f"authority path drift: {path}")
        blobs[path] = observed
    return blobs


_HISTORICAL_MATCHER_MODULES = (
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
    _require(not (base_ids & extension_ids), "Caselaw source-id collision with incumbent graph")

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed V8 authority plus exact PR #1347 source-admitted Caselaw "
        "rows; pair decisions delegate to terminal PR #824 V3 semantics and indexed "
        "execution must be byte-equivalent to the all-pairs reference."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def _outer_survivor_authority(
    dedup_report: Mapping[str, Any],
    selection_projection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = dedup_report.get("sources")
    _require(isinstance(sources, list), "dedup source vector missing")
    by_id = {
        row["source_id"]: row
        for row in sources
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    }
    survivor_ids = selection_projection.get("survivor_source_ids")
    _require(isinstance(survivor_ids, list), "selection projection survivor ids missing")
    _require(all(source_id in by_id for source_id in survivor_ids), "unknown survivor id")
    caselaw_survivors = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == caselaw.SOURCE_FAMILY
    ]
    caselaw_survivor_bytes = sum(
        int(by_id[source_id]["declared_capacity_bytes"])
        for source_id in caselaw_survivors
    )
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
        "caselaw": {
            "source_family": caselaw.SOURCE_FAMILY,
            "pre_dedup_source_object_count": EXPECTED_CASELAW_OBJECTS,
            "pre_dedup_payload_bytes": EXPECTED_CASELAW_BYTES,
            "post_dedup_survivor_source_object_count": len(caselaw_survivors),
            "post_dedup_survivor_declared_capacity_bytes": caselaw_survivor_bytes,
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
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def _max_rss_kib() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if value > 10_000_000 else value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(dict(value)) + b"\n")


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    execution_evidence_json: Path,
    output_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    authority_blobs = verify_repository_authority()
    execution_head = _git("rev-parse", "HEAD").stdout.strip()
    _require(len(execution_head) == 40, "execution HEAD identity missing")
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

    projection = caselaw.validate_and_project_caselaw(
        candidate_jsonl,
        execution_evidence_json,
        upstream_head=CASELAW_FINAL_HEAD,
        retain_payloads=True,
    )
    _require(projection.sources is not None and projection.payloads is not None, "payload projection missing")
    extension_sources = [dict(row) for row in projection.sources]
    extension_payloads = dict(projection.payloads)
    _require(len(extension_sources) == EXPECTED_CASELAW_OBJECTS, "Caselaw projection count drift")
    _require(
        sum(len(raw) for raw in extension_payloads.values()) == EXPECTED_CASELAW_BYTES,
        "Caselaw projection bytes drift",
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

    reference_started = time.perf_counter()
    reference = matcher.audit_payloads(inventory, payloads)
    reference_seconds = time.perf_counter() - reference_started
    matcher.verify_report(reference)

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

    reference_bytes = matcher.v1._canonical_bytes(reference)
    indexed_bytes = matcher.v1._canonical_bytes(indexed_report)
    _require(reference_bytes == indexed_bytes, "indexed report differs from incumbent all-pairs report")

    rows, _ = matcher._validate_inventory(inventory)
    validated = matcher.v1._validate_inventory(matcher._as_v1_inventory(rows))
    fingerprints = [
        matcher._fingerprint(row, payloads[row["source_id"]])
        for row in validated
    ]
    pairs, work = indexed.candidate_pair_indices_with_stats(
        matcher.v1,
        fingerprints,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    work_stats = indexed.execution_stats(
        len(fingerprints),
        len(pairs),
        index_postings=work["index_postings"],
        pair_expansion_attempts=work["pair_expansion_attempts"],
        unique_bucket_signatures=work["unique_bucket_signatures"],
    )

    terminal = indexed_report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal dedup summary missing")
    _require(
        terminal.get("declared_capacity_bytes_before") == EXPECTED_COMBINED_BYTES,
        "combined declared capacity drift",
    )
    selection_projection = v9_semantics._derive_survivors(indexed_report)
    survivors = _outer_survivor_authority(indexed_report, selection_projection)

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "GITHUB_HOSTED_FREE_LOCAL_FREE",
        "execution_claim_issue": EXECUTION_CLAIM,
        "execution_pr": EXECUTION_PR,
        "execution_head_sha": execution_head,
        "pinned_main_sha": EXPECTED_MAIN,
        "authority_path_blobs": authority_blobs,
        "baseline_v8": {
            "v7_head_sha": v8.EXPECTED_V7_HEAD,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "payload_bytes": EXPECTED_BASE_BYTES,
        },
        "caselaw": {
            "source_admission_product_pr": caselaw.UPSTREAM_PRODUCT_PR,
            "source_admission_final_head": CASELAW_FINAL_HEAD,
            "candidate_sha256": caselaw.CANDIDATE_SHA256,
            "source_object_count": EXPECTED_CASELAW_OBJECTS,
            "payload_bytes": EXPECTED_CASELAW_BYTES,
            "intake_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": EXPECTED_COMBINED_BYTES,
            "reference_report_sha256": reference["report_sha256"],
            "indexed_report_sha256": indexed_report["report_sha256"],
            "reports_byte_identical": True,
            "post_dedup_conservative_unique_bytes": terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
            "duplicate_discount_bytes": terminal.get("duplicate_discount_bytes"),
            "duplicate_cluster_count": terminal.get("duplicate_cluster_count"),
        },
        "indexed_execution": {
            **work_stats,
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "reference_wall_clock_seconds": round(reference_seconds, 6),
            "indexed_wall_clock_seconds": round(indexed_seconds, 6),
            "process_max_rss_kib": _max_rss_kib(),
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
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }
    _write_json(output_report, indexed_report)
    _write_json(output_survivors, survivors)
    _write_json(output_evidence, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--execution-evidence-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
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
        execution_evidence_json=args.execution_evidence_json,
        output_report=args.output_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_CASELAW_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
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
