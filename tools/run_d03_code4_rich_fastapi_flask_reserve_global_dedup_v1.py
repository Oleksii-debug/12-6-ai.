#!/usr/bin/env python3
"""Measure Rich + FastAPI + Flask reserve capacity after terminal D03 code4.

Execution-only carrier. It composes three already-qualified lawful code-source
families on top of the terminal code4 graph, executes incumbent indexed global
dedup twice, and emits only text-free zero-credit evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_code4_flask_global_dedup_v1 as flask_reserve
import run_d03_code4_rich_fastapi_reserve_global_dedup_v1 as reserve

SCHEMA = "12-6.d03-code4-rich-fastapi-flask-reserve-global-dedup-execution.v1"
PARENT_RESERVE_HEAD = "b3d6fbef0c9cc1f0fbd4340e5eb7376fce01261a"
EXPECTED_RESERVE_RUNNER_BLOB = "5349eacdbeafe357000c3411d5997354bd58b140"
EXPECTED_FLASK_RUNNER_BLOB = "e14aea662fd5c5bb17341e0f9cefe0c9d5ac1485"
FLASK_FAMILY = "github:pallets/flask"

EXPECTED_RESERVE_OBJECTS = reserve.EXPECTED_RESERVE_OBJECTS + flask_reserve.FLASK_OBJECTS
EXPECTED_RESERVE_BYTES = reserve.EXPECTED_RESERVE_BYTES + flask_reserve.FLASK_BYTES
EXPECTED_COMBINED_OBJECTS = (
    reserve.EXPECTED_COMBINED_OBJECTS + flask_reserve.FLASK_OBJECTS
)
EXPECTED_COMBINED_BYTES = reserve.EXPECTED_COMBINED_BYTES + flask_reserve.FLASK_BYTES

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class CombinedReserveGlobalDedupError(RuntimeError):
    """Fail-closed combined reserve authority or execution mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CombinedReserveGlobalDedupError(message)


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        reserve.code4.git("rev-parse", "HEAD") == expected_execution_head,
        "execution HEAD drift",
    )
    ancestor = reserve.code4.incumbent._git(
        "merge-base",
        "--is-ancestor",
        PARENT_RESERVE_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "terminal Rich/FastAPI parent is not ancestor")
    require(
        reserve.code4.git(
            "rev-parse",
            "HEAD:tools/run_d03_code4_rich_fastapi_reserve_global_dedup_v1.py",
        )
        == EXPECTED_RESERVE_RUNNER_BLOB,
        "terminal Rich/FastAPI runner blob drift",
    )
    require(
        reserve.code4.git(
            "rev-parse",
            "HEAD:tools/run_d03_code4_flask_global_dedup_v1.py",
        )
        == EXPECTED_FLASK_RUNNER_BLOB,
        "Flask wrapper runner blob drift",
    )

    rich_fastapi = reserve.verify_execution_authority(expected_execution_head)
    flask = flask_reserve.verify_local_authority(expected_execution_head)
    return {**rich_fastapi, **flask}


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    expected_execution_head: str,
    output_code4_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    authority_blobs = verify_local_authority(expected_execution_head)

    original_build = reserve.build_reserve_sources
    original_compose = reserve.compose_reserve_graph
    original_schema = reserve.SCHEMA
    original_reserve_objects = reserve.EXPECTED_RESERVE_OBJECTS
    original_reserve_bytes = reserve.EXPECTED_RESERVE_BYTES
    original_combined_objects = reserve.EXPECTED_COMBINED_OBJECTS
    original_combined_bytes = reserve.EXPECTED_COMBINED_BYTES
    flask_authority: dict[str, Any] = {}

    def build_all_reserve_sources(
        rich: dict[str, Any],
        fast: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
        nonlocal flask_authority

        reserve.EXPECTED_RESERVE_OBJECTS = original_reserve_objects
        reserve.EXPECTED_RESERVE_BYTES = original_reserve_bytes
        reserve.EXPECTED_COMBINED_OBJECTS = original_combined_objects
        reserve.EXPECTED_COMBINED_BYTES = original_combined_bytes

        rows, payloads, edges = original_build(rich, fast)
        flask_rows, flask_payloads, flask_edges, flask_authority = (
            flask_reserve.capture_flask_sources()
        )
        existing_ids = {row["source_id"] for row in rows}
        flask_ids = {row["source_id"] for row in flask_rows}
        require(not (existing_ids & flask_ids), "Flask reserve source-id collision")

        rows = [*rows, *flask_rows]
        payloads = {**payloads, **flask_payloads}
        edges = [*edges, *flask_edges]

        reserve.EXPECTED_RESERVE_OBJECTS = EXPECTED_RESERVE_OBJECTS
        reserve.EXPECTED_RESERVE_BYTES = EXPECTED_RESERVE_BYTES
        reserve.EXPECTED_COMBINED_OBJECTS = EXPECTED_COMBINED_OBJECTS
        reserve.EXPECTED_COMBINED_BYTES = EXPECTED_COMBINED_BYTES
        return rows, payloads, edges

    def compose_all_reserve_graph(
        code4_inventory: dict[str, Any],
        code4_payloads: dict[str, bytes],
        reserve_rows: list[dict[str, Any]],
        reserve_payloads: dict[str, bytes],
        reserve_edges: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        inventory, payloads = original_compose(
            code4_inventory,
            code4_payloads,
            reserve_rows,
            reserve_payloads,
            reserve_edges,
        )
        inventory["terminal_refresh_rule"] = (
            "Exact terminal code4 graph plus exact terminal Rich v15.0.0, "
            "FastAPI, and Flask 3.1.3 source-authority objects. Pair decisions "
            "delegate to incumbent PR #824 V3 semantics through independently "
            "qualified PR #1459 indexed execution. This carrier grants zero "
            "canonical capacity credit."
        )
        quarantine_authority = json.loads(
            (
                ROOT
                / reserve.code4.incumbent.clean_successor.QUARANTINE_CONFIG_PATH
            ).read_text(encoding="utf-8")
        )
        reserve.code4.incumbent._verify_clean_payload_graph(
            inventory,
            payloads,
            quarantine_authority,
        )
        return inventory, payloads

    inner_evidence = output_evidence.with_name(output_evidence.name + ".inner")
    try:
        reserve.SCHEMA = SCHEMA
        reserve.build_reserve_sources = build_all_reserve_sources
        reserve.compose_reserve_graph = compose_all_reserve_graph
        evidence = reserve.execute(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            expected_execution_head=expected_execution_head,
            output_code4_report=output_code4_report,
            output_combined_report=output_combined_report,
            output_survivors=output_survivors,
            output_evidence=inner_evidence,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
        )
    finally:
        reserve.SCHEMA = original_schema
        reserve.build_reserve_sources = original_build
        reserve.compose_reserve_graph = original_compose
        reserve.EXPECTED_RESERVE_OBJECTS = original_reserve_objects
        reserve.EXPECTED_RESERVE_BYTES = original_reserve_bytes
        reserve.EXPECTED_COMBINED_OBJECTS = original_combined_objects
        reserve.EXPECTED_COMBINED_BYTES = original_combined_bytes

    require(bool(flask_authority), "Flask authority was not captured")
    inner_identity = str(evidence.pop("evidence_identity_sha256"))
    evidence["schema_version"] = SCHEMA
    evidence["parent_rich_fastapi_execution_head_sha"] = PARENT_RESERVE_HEAD
    evidence["wrapper_authority_path_blobs"] = authority_blobs
    evidence["source_authorities"]["flask"] = {
        **flask_authority,
        "source_family": FLASK_FAMILY,
    }

    delta = evidence["reserve_delta"]
    delta["source_families"] = sorted(
        [reserve.FASTAPI_FAMILY, FLASK_FAMILY, reserve.RICH_FAMILY]
    )
    delta["pre_dedup_candidate_objects"] = EXPECTED_RESERVE_OBJECTS
    delta["pre_dedup_candidate_bytes"] = EXPECTED_RESERVE_BYTES
    delta["components"] = {
        "rich_raw_bytes": reserve.RICH_EXPECTED_BYTES,
        "fastapi_raw_bytes": reserve.FASTAPI_EXPECTED_BYTES,
        "flask_raw_bytes": flask_reserve.FLASK_BYTES,
    }
    evidence["inner_rich_fastapi_execution_evidence_identity_sha256"] = inner_identity
    evidence["evidence_identity_sha256"] = reserve.sha256(reserve.canonical(evidence))

    reserve.code4.incumbent._publish_json_outputs(((output_evidence, evidence),))
    try:
        inner_evidence.unlink()
    except FileNotFoundError:
        pass
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--output-code4-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=reserve.code4.incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=reserve.code4.incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()

    try:
        evidence = execute(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            expected_execution_head=args.expected_execution_head,
            output_code4_report=args.output_code4_report,
            output_combined_report=args.output_combined_report,
            output_survivors=args.output_survivors,
            output_evidence=args.output_evidence,
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
    except (
        CombinedReserveGlobalDedupError,
        reserve.ReserveGlobalDedupError,
        flask_reserve.Code4FlaskGlobalDedupError,
        reserve.code4.FourCodeGlobalDedupError,
        reserve.code4.incumbent.Franko1901GlobalDedupError,
        reserve.code4.incumbent.indexed.IndexedExecutionError,
        OSError,
        ValueError,
    ) as exc:
        rendered = " ".join(str(exc).split())[:240]
        print(
            json.dumps(
                {
                    "status": "BLOCKED_CODE4_RICH_FASTAPI_FLASK_RESERVE_GLOBAL_DEDUP",
                    "error_type": type(exc).__name__,
                    "error_detail": rendered,
                    "canonical_capacity_credited": 0,
                    "authorized_optimized_target_exposure": 0,
                    "training_executed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    delta = evidence["reserve_delta"]
    print("D03_CODE4_RICH_FASTAPI_FLASK_RESERVE_GLOBAL_DEDUP=PASS_ZERO_CREDIT")
    print(
        "RESERVE_MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(delta["marginal_global_unique_capacity_bytes"])
    )
    print(
        "PROJECTED_LATE_GATE_HEADROOM_AFTER_GLOBAL_DEDUP="
        + str(delta["projected_late_gate_headroom_after_global_dedup"])
    )
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
