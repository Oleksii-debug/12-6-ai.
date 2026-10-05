#!/usr/bin/env python3
"""Run the four physically verified code sources through incumbent global dedup.

Execution-only carrier. Raw source bytes stay ephemeral in memory. Durable outputs
are text-free matcher reports, survivor authority and zero-credit execution evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
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

import execute_d03_four_code_sources_materialization as physical
import run_d03_franko1901_global_dedup_execution_v1 as incumbent

SCHEMA = "12-6.d03-four-code-source-global-dedup-execution.v1"
PARENT_PHYSICAL_HEAD = "6c0bab662fe0d41c922f87ff230c40062f373202"
EXPECTED_PHYSICAL_TOOL_BLOB = "7a289414bfd1c20db3f859424d43893c7fdc1e8c"
EXPECTED_INCUMBENT_RUNNER_BLOB = "a314c59a90e89a61c6cd8adfab10caf668bbce93"
EXPECTED_PHYSICAL_SUMMARY_SHA256 = (
    "39fef69fea9b65bc1f6957c606de951fb0c93e790e8538e0a08bb9f6f3381031"
)
EXPECTED_OBJECT_SET_SHA256 = (
    "5947436cd926c13576fd14e34459ffb2f6e2e290cca79bead7327cf7e13f4d91"
)
EXPECTED_LICENSE_SET_SHA256 = (
    "0d43b619d24210e51a6e7bc3e9c28a264b6bb5c2b473f820a0c46e2eb2fb29b0"
)
EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_BYTES = 6_093_965
EXPECTED_EXTENSION_OBJECTS = 8
EXPECTED_EXTENSION_BYTES = 336_947
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_EXTENSION_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_EXTENSION_BYTES
CURRENT_POST_QP_CODE_BYTES_REFERENCE = 3_664_247
CODE_TARGET_BYTES = 4_000_000
CODE_GAP_BYTES = CODE_TARGET_BYTES - CURRENT_POST_QP_CODE_BYTES_REFERENCE
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class FourCodeGlobalDedupError(RuntimeError):
    """Fail-closed execution or authority mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FourCodeGlobalDedupError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def git(*args: str) -> str:
    completed = incumbent._git(*args)
    return completed.stdout.strip()


def verify_execution_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(git("rev-parse", "HEAD") == expected_execution_head, "execution HEAD drift")
    ancestor = incumbent._git(
        "merge-base",
        "--is-ancestor",
        PARENT_PHYSICAL_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "green physical parent is not execution ancestor")

    pinned = {
        "tools/execute_d03_four_code_sources_materialization.py": (
            EXPECTED_PHYSICAL_TOOL_BLOB
        ),
        "tools/run_d03_franko1901_global_dedup_execution_v1.py": (
            EXPECTED_INCUMBENT_RUNNER_BLOB
        ),
    }
    for path, expected_blob in pinned.items():
        require(git("rev-parse", f"HEAD:{path}") == expected_blob, f"blob drift: {path}")
        require(
            git("hash-object", str(ROOT / path)) == expected_blob,
            f"worktree drift: {path}",
        )

    authority_blobs = incumbent.verify_repository_authority()
    require(bool(authority_blobs), "incumbent authority closure is empty")
    return authority_blobs


def capture_physical_sources() -> tuple[dict[str, Any], dict[str, bytes]]:
    captured: dict[str, bytes] = {}
    original = physical.download

    def capture(url: str, max_bytes: int = 400_000) -> bytes:
        raw = original(url, max_bytes)
        captured[url] = raw
        return raw

    physical.download = capture
    try:
        summary = physical.execute(ROOT)
    finally:
        physical.download = original

    require(summary.get("candidate_raw_bytes") == EXPECTED_EXTENSION_BYTES, "raw byte drift")
    require(summary.get("object_count") == EXPECTED_EXTENSION_OBJECTS, "object count drift")
    require(summary.get("source_family_count") == 4, "source family count drift")
    require(
        summary.get("object_set_sha256") == EXPECTED_OBJECT_SET_SHA256,
        "physical object-set identity drift",
    )
    require(
        summary.get("license_set_sha256") == EXPECTED_LICENSE_SET_SHA256,
        "physical license-set identity drift",
    )
    require(
        sha256(physical.canonical(summary)) == EXPECTED_PHYSICAL_SUMMARY_SHA256,
        "physical summary identity drift",
    )
    return summary, captured


def _source_row(
    *,
    repository: str,
    commit: str,
    family: str,
    source_id: str,
    path: str,
    url: str,
    expected_bytes: int,
    expected_blob: str,
    raw: bytes,
) -> dict[str, Any]:
    require(captured_blob(raw) == expected_blob, f"{source_id}: captured Git blob drift")
    require(len(raw) == expected_bytes, f"{source_id}: captured byte drift")
    locator = f"github:{repository}@{commit}:{path}"
    return {
        "source_id": source_id,
        "source_family": family,
        "modality": "code",
        "evidence_status": "DEDICATED_TERMINAL",
        "acquisition_url": url,
        "origin_key": locator,
        "stable_origin_id": locator,
        "stable_object_id": f"git-blob-sha1:{expected_blob}",
        "declared_capacity_bytes": expected_bytes,
        "expected_raw_bytes": expected_bytes,
        "expected_raw_sha256": sha256(raw),
        "expected_git_blob_sha1": expected_blob,
    }


def captured_blob(raw: bytes) -> str:
    prefix = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(  # noqa: S324 - exact Git object identity\n        prefix + raw, usedforsecurity=False\n    ).hexdigest()


def _captured(captured: Mapping[str, bytes], url: str, source_id: str) -> bytes:
    raw = captured.get(url)
    require(isinstance(raw, bytes), f"{source_id}: physical payload capture missing")
    return raw


def build_extension_graph(
    captured: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    configs = {
        name: physical.load_config(ROOT, name)
        for name in ("pydantic", "scipy", "pandas", "typer")
    }
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    family_groups: dict[tuple[str, str, str], list[str]] = {}

    pydantic = configs["pydantic"]
    pyd_repo = "pydantic/pydantic"
    pyd_commit = str(pydantic["upstream_commit"])
    pyd_family = str(pydantic["source_family"])
    for item in pydantic["decisions"]:
        source_id = str(item["source_id"])
        url = str(item["raw_url"])
        raw = _captured(captured, url, source_id)
        row = _source_row(
            repository=pyd_repo,
            commit=pyd_commit,
            family=pyd_family,
            source_id=source_id,
            path=str(item["path"]),
            url=url,
            expected_bytes=int(item["size_bytes"]),
            expected_blob=str(item["blob_sha1"]),
            raw=raw,
        )
        rows.append(row)
        payloads[source_id] = raw
        family_groups.setdefault((pyd_repo, pyd_commit, pyd_family), []).append(source_id)

    scipy = configs["scipy"]
    scipy_upstream = scipy["upstream"]
    scipy_repo = str(scipy_upstream["repository"])
    scipy_commit = str(scipy_upstream["commit_sha"])
    scipy_family = str(scipy["source_family"])
    for index, item in enumerate(scipy["allowlist"]):
        source_id = f"code.scipy.project.{index}"
        url = str(item["raw_url"])
        raw = _captured(captured, url, source_id)
        row = _source_row(
            repository=scipy_repo,
            commit=scipy_commit,
            family=scipy_family,
            source_id=source_id,
            path=str(item["path"]),
            url=url,
            expected_bytes=int(item["raw_bytes"]),
            expected_blob=str(item["git_blob_sha1"]),
            raw=raw,
        )
        rows.append(row)
        payloads[source_id] = raw
        family_groups.setdefault((scipy_repo, scipy_commit, scipy_family), []).append(
            source_id
        )

    for name in ("pandas", "typer"):
        config = configs[name]
        item = config["bounded_source"]
        repository = str(item["repository"])
        commit = str(item["commit"])
        path = str(item["path"])
        source_id = str(item["source_id"])
        family = str(item["source_family"])
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
        raw = _captured(captured, url, source_id)
        row = _source_row(
            repository=repository,
            commit=commit,
            family=family,
            source_id=source_id,
            path=path,
            url=url,
            expected_bytes=int(item["size_bytes"]),
            expected_blob=str(item["git_blob_sha1"]),
            raw=raw,
        )
        require(
            row["expected_raw_sha256"] == str(item["raw_sha256"]),
            f"{source_id}: config SHA-256 drift",
        )
        rows.append(row)
        payloads[source_id] = raw
        family_groups.setdefault((repository, commit, family), []).append(source_id)

    require(len(rows) == EXPECTED_EXTENSION_OBJECTS, "extension row count drift")
    require(set(payloads) == {row["source_id"] for row in rows}, "extension coverage drift")
    require(
        sum(row["declared_capacity_bytes"] for row in rows) == EXPECTED_EXTENSION_BYTES,
        "extension declared-capacity drift",
    )

    edges: list[dict[str, Any]] = []
    for (repository, commit, _family), source_ids in sorted(family_groups.items()):
        if len(source_ids) < 2:
            continue
        anchor = source_ids[0]
        for sibling in source_ids[1:]:
            edges.append(
                {
                    "left_source_id": anchor,
                    "right_source_id": sibling,
                    "relation": "sibling_same_origin",
                    "capacity_collapsing": False,
                    "independence_collapsing": True,
                    "evidence": (
                        "Exact sibling objects from one pinned upstream repository "
                        f"revision: {repository}@{commit}"
                    ),
                }
            )
    require(len(edges) == 4, "expected three Pydantic plus one SciPy sibling edges")
    return rows, payloads, edges


def compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_rows: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
    extension_edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(type(base_inventory) is dict, "base inventory must be exact object")
    base_rows = base_inventory.get("sources")
    require(isinstance(base_rows, list) and bool(base_rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in base_rows if isinstance(row, Mapping)}
    require(len(base_ids) == len(base_rows), "base source identities invalid")
    require(base_ids == set(base_payloads), "base inventory/payload mismatch")

    extension_ids = {row["source_id"] for row in extension_rows}
    require(extension_ids == set(extension_payloads), "extension inventory/payload mismatch")
    require(not (base_ids & extension_ids), "extension source-id collision with incumbent")

    lineage_edges = base_inventory.get("lineage_edges", [])
    require(isinstance(lineage_edges, list), "base lineage_edges malformed")

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(base_rows), *copy.deepcopy(extension_rows)]
    inventory["lineage_edges"] = [*copy.deepcopy(lineage_edges), *copy.deepcopy(extension_edges)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed quarantined-source-free incumbent V8 graph plus eight "
        "physically verified Pydantic/SciPy/Pandas/Typer source objects. Pair decisions "
        "delegate to the terminal PR #824 V3 semantics through the independently "
        "qualified PR #1459 indexed executor."
    )

    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    quarantine_authority = json.loads(
        (ROOT / incumbent.clean_successor.QUARANTINE_CONFIG_PATH).read_text(
            encoding="utf-8"
        )
    )
    incumbent._verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object count drift")
    require(
        incumbent._declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )
    return inventory, payloads


def terminal_summary(
    report: Mapping[str, Any],
    *,
    expected_objects: int,
    expected_bytes: int,
    label: str,
) -> Mapping[str, Any]:
    require(report.get("source_count") == expected_objects, f"{label}: source count drift")
    terminal = report.get("terminal_candidates")
    require(isinstance(terminal, Mapping), f"{label}: terminal summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    clusters = terminal.get("duplicate_cluster_count")
    require(before == expected_bytes, f"{label}: declared-capacity before drift")
    require(type(after) is int and 0 < after <= before, f"{label}: unique bytes invalid")
    require(
        type(discount) is int and discount >= 0 and before - after == discount,
        f"{label}: duplicate discount arithmetic drift",
    )
    require(type(clusters) is int and clusters >= 0, f"{label}: cluster count invalid")
    return terminal


def validate_survivor_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    require(
        projection.get("schema_version") == incumbent.v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    supplied = projection.get("survivor_authority_sha256")
    require(isinstance(supplied, str) and len(supplied) == 64, "survivor identity invalid")
    core = dict(projection)
    core.pop("survivor_authority_sha256", None)
    require(supplied == sha256(canonical(core)), "survivor identity drift")
    require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "survivor matcher identity drift",
    )
    require(
        projection.get("pre_dedup_source_object_count") == EXPECTED_COMBINED_OBJECTS,
        "survivor pre-dedup count drift",
    )
    require(
        projection.get("pre_dedup_declared_capacity_bytes") == EXPECTED_COMBINED_BYTES,
        "survivor pre-dedup byte drift",
    )
    require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "survivor post-dedup byte drift",
    )
    survivor_ids = projection.get("survivor_source_ids")
    require(
        isinstance(survivor_ids, list)
        and len(survivor_ids) == len(set(survivor_ids))
        and all(isinstance(value, str) and value for value in survivor_ids),
        "survivor ids invalid",
    )


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    expected_execution_head: str,
    output_base_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    authority_blobs = verify_execution_authority(expected_execution_head)
    physical_summary, captured = capture_physical_sources()
    extension_rows, extension_payloads, extension_edges = build_extension_graph(captured)

    verified_v7_head = incumbent._verify_v7_worktree(v7_root)
    config = incumbent.v8.load_config(
        ROOT / "configs/data/next100_065f_global_dedup_v8.json"
    )
    matcher, base_inventory, base_payloads, removal = (
        incumbent._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    require(
        incumbent._verify_v7_worktree(v7_root) == verified_v7_head,
        "V7 worktree drifted during reconstruction",
    )
    require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "base object count drift")
    require(
        incumbent._declared_capacity_bytes(
            base_inventory,
            base_payloads,
            label="incumbent base",
        )
        == EXPECTED_BASE_BYTES,
        "base declared-capacity drift",
    )

    combined_inventory, combined_payloads = compose_graph(
        base_inventory,
        base_payloads,
        extension_rows,
        extension_payloads,
        extension_edges,
    )

    indexed = incumbent.indexed
    indexed.attest_incumbent_runtime(matcher)

    base_started = time.perf_counter()
    base_report = indexed.audit_payloads_indexed(
        matcher,
        copy.deepcopy(base_inventory),
        dict(base_payloads),
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    base_seconds = time.perf_counter() - base_started
    matcher.verify_report(base_report)
    base_terminal = terminal_summary(
        base_report,
        expected_objects=EXPECTED_BASE_OBJECTS,
        expected_bytes=EXPECTED_BASE_BYTES,
        label="base",
    )

    combined_started = time.perf_counter()
    combined_report = indexed.audit_payloads_indexed(
        matcher,
        combined_inventory,
        combined_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    combined_seconds = time.perf_counter() - combined_started
    matcher.verify_report(combined_report)
    combined_terminal = terminal_summary(
        combined_report,
        expected_objects=EXPECTED_COMBINED_OBJECTS,
        expected_bytes=EXPECTED_COMBINED_BYTES,
        label="combined",
    )

    survivor_projection = incumbent.v9_semantics._derive_survivors(combined_report)
    validate_survivor_projection(
        combined_report,
        survivor_projection,
        combined_terminal,
    )

    base_unique = int(base_terminal["conservative_unique_capacity_bytes_after"])
    combined_unique = int(combined_terminal["conservative_unique_capacity_bytes_after"])
    marginal = combined_unique - base_unique
    loss_vs_raw = EXPECTED_EXTENSION_BYTES - marginal
    closes_gap_at_global_dedup = marginal >= CODE_GAP_BYTES

    extension_ids = {row["source_id"] for row in extension_rows}
    survivor_ids = set(survivor_projection["survivor_source_ids"])
    extension_survivors = sorted(extension_ids & survivor_ids)
    by_id = {row["source_id"]: row for row in extension_rows}
    extension_survivor_declared = sum(
        int(by_id[source_id]["declared_capacity_bytes"])
        for source_id in extension_survivors
    )

    max_rss = incumbent._max_rss_kib()
    require(type(max_rss) is int and max_rss > 0, "process max RSS unavailable")

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": expected_execution_head,
        "parent_physical_head_sha": PARENT_PHYSICAL_HEAD,
        "runtime_environment": incumbent._runtime_environment(),
        "authority_path_blobs": authority_blobs,
        "physical_source_authority": {
            "dedicated_run_id": 37269452704,
            "source_family_count": 4,
            "object_count": EXPECTED_EXTENSION_OBJECTS,
            "candidate_raw_bytes": EXPECTED_EXTENSION_BYTES,
            "summary_sha256": EXPECTED_PHYSICAL_SUMMARY_SHA256,
            "object_set_sha256": EXPECTED_OBJECT_SET_SHA256,
            "license_set_sha256": EXPECTED_LICENSE_SET_SHA256,
            "observed_summary_sha256": sha256(physical.canonical(physical_summary)),
        },
        "incumbent_base": {
            "v7_head_sha": verified_v7_head,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "matcher_report_sha256": base_report["report_sha256"],
            "post_dedup_conservative_unique_bytes": base_unique,
            "duplicate_discount_bytes": base_terminal["duplicate_discount_bytes"],
            "nomis1864_deauthorization": removal,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "matcher_report_sha256": combined_report["report_sha256"],
            "post_dedup_conservative_unique_bytes": combined_unique,
            "duplicate_discount_bytes": combined_terminal["duplicate_discount_bytes"],
            "survivor_authority_sha256": survivor_projection[
                "survivor_authority_sha256"
            ],
        },
        "four_source_delta": {
            "pre_dedup_candidate_bytes": EXPECTED_EXTENSION_BYTES,
            "marginal_global_unique_capacity_bytes": marginal,
            "global_dedup_loss_vs_raw_candidate_bytes": loss_vs_raw,
            "extension_survivor_source_objects": len(extension_survivors),
            "extension_survivor_declared_capacity_bytes": extension_survivor_declared,
            "current_post_qp_code_bytes_reference": (
                CURRENT_POST_QP_CODE_BYTES_REFERENCE
            ),
            "code_target_bytes": CODE_TARGET_BYTES,
            "minimum_marginal_bytes_needed_before_later_gates": CODE_GAP_BYTES,
            "global_dedup_preserves_enough_to_keep_target_possible": (
                closes_gap_at_global_dedup
            ),
            "later_gate_loss_budget_if_target_still_possible": (
                max(0, marginal - CODE_GAP_BYTES)
                if closes_gap_at_global_dedup
                else 0
            ),
            "shortfall_already_proven_after_global_dedup": (
                max(0, CODE_GAP_BYTES - marginal)
            ),
        },
        "indexed_execution": {
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "base_wall_clock_seconds": round(base_seconds, 6),
            "combined_wall_clock_seconds": round(combined_seconds, 6),
            "process_max_rss_kib": max_rss,
            "performance_equivalence_authority": "MERGED_PR_1459",
        },
        "content_boundary": {
            "raw_source_bytes_persisted": False,
            "raw_source_bytes_uploaded": False,
            "base_report_text_free": True,
            "combined_report_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_executed_for_exact_candidate": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "balance_diversity_retest_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "deterministic_pack_two_clean_complete": False,
            "positive_exact_unique_loss_ledger": False,
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
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }

    incumbent._publish_json_outputs(
        (
            (output_base_report, base_report),
            (output_combined_report, combined_report),
            (output_survivors, survivor_projection),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--output-base-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()

    try:
        evidence = execute(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            expected_execution_head=args.expected_execution_head,
            output_base_report=args.output_base_report,
            output_combined_report=args.output_combined_report,
            output_survivors=args.output_survivors,
            output_evidence=args.output_evidence,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (
        FourCodeGlobalDedupError,
        incumbent.Franko1901GlobalDedupError,
        incumbent.indexed.IndexedExecutionError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_FOUR_CODE_GLOBAL_DEDUP",
                    "error_type": type(exc).__name__,
                    "canonical_capacity_credited": 0,
                    "authorized_optimized_target_exposure": 0,
                    "training_executed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    delta = evidence["four_source_delta"]
    print("D03_FOUR_CODE_GLOBAL_DEDUP=PASS_ZERO_CREDIT")
    print(
        "MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(delta["marginal_global_unique_capacity_bytes"])
    )
    print(
        "GLOBAL_DEDUP_PRESERVES_ENOUGH_TO_KEEP_TARGET_POSSIBLE="
        + str(delta["global_dedup_preserves_enough_to_keep_target_possible"]).lower()
    )
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
