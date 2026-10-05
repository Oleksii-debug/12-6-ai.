#!/usr/bin/env python3
"""Measure Rich + FastAPI reserve capacity after the terminal D03 code4 graph.

Execution-only carrier. It consumes exact terminal source authorities, reconstructs
the exact incumbent quarantined-source-free V8 graph through the terminal code4
runner, and executes the independently qualified indexed matcher over:

1. the terminal code4 graph; and
2. the same graph plus exact Rich v15.0.0 and FastAPI pinned source objects.

Raw source bytes remain ephemeral. Durable outputs are text-free matcher reports,
survivor authority, and zero-credit execution evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_four_code_sources_global_dedup_v1 as code4

SCHEMA = "12-6.d03-code4-rich-fastapi-reserve-global-dedup-execution.v1"
PARENT_CODE4_HEAD = "e33b441a767ee48a182f930d202f1419e3849480"
EXPECTED_CODE4_RUNNER_BLOB = "eb89d36fb8bf7c97d4860d242e960940a0fc0ed3"

RICH_POLICY_PATH = "configs/data/next100_051_rich_code_rights_v1.json"
RICH_POLICY_BLOB = "4b4160814ddb97cb47bf45b4af2ed1b9ce8fef9e"
RICH_SOURCE_HEAD = "01c24cbafb01dfe5c5445f0851265cafedab846f"
RICH_SOURCE_RUN = 34840386134
RICH_EXPECTED_OBJECTS = 6
RICH_EXPECTED_BYTES = 46_162
RICH_FAMILY = "github:Textualize/rich"

FASTAPI_POLICY_PATH = "configs/data/next100_044_fastapi_code_rights_policy_v1.json"
FASTAPI_POLICY_BLOB = "8ee76ccc2ca3ff40d7e3d6463670d99e49051b44"
FASTAPI_SOURCE_HEAD = "341666cf66a1001f4708921d997bb3d3863a1a0d"
FASTAPI_SOURCE_RUN = 32999727387
FASTAPI_REGISTRY_SEAL_RUN = 32999727467
FASTAPI_EXPECTED_OBJECTS = 3
FASTAPI_EXPECTED_BYTES = 19_857
FASTAPI_FAMILY = "github:fastapi/fastapi"

EXPECTED_CODE4_OBJECTS = code4.EXPECTED_COMBINED_OBJECTS
EXPECTED_CODE4_BYTES = code4.EXPECTED_COMBINED_BYTES
EXPECTED_RESERVE_OBJECTS = RICH_EXPECTED_OBJECTS + FASTAPI_EXPECTED_OBJECTS
EXPECTED_RESERVE_BYTES = RICH_EXPECTED_BYTES + FASTAPI_EXPECTED_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_CODE4_OBJECTS + EXPECTED_RESERVE_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_CODE4_BYTES + EXPECTED_RESERVE_BYTES

CODE4_TERMINAL_MARGINAL_BYTES = 336_947
CODE4_TERMINAL_LATE_GATE_HEADROOM_BYTES = (
    CODE4_TERMINAL_MARGINAL_BYTES - code4.CODE_GAP_BYTES
)
CODE4_TERMINAL_RUN = 37276943420
CODE4_TERMINAL_BASE_REPORT = (
    "a83c9c19930e8fb055480a937e8ee3c93df730726765e5053e44e07b20f2834f"
)
CODE4_TERMINAL_COMBINED_REPORT = (
    "b63d89bb3b9d8fe220c4d98c8e75b12c0ec07043ac920efd3b4639c39f416b13"
)
CODE4_TERMINAL_SURVIVOR_AUTHORITY = (
    "cfed7514b3734010455a3f396c28bee21587b24f326b3448b038f16a27629929"
)
CODE4_TERMINAL_PROOF_IDENTITY = (
    "00c20e289038a4f6f09e3b9e44df197f1e6492802a292b2a384abe818350fbe7"
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class ReserveGlobalDedupError(RuntimeError):
    """Fail-closed authority, acquisition, or execution mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReserveGlobalDedupError(message)


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


def _headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": "12-6-d03-code4-rich-fastapi-reserve/1"}
    if urllib.parse.urlparse(url).hostname == "api.github.com":
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                }
            )
    return headers


def download(url: str, max_bytes: int = 400_000) -> bytes:
    request = urllib.request.Request(url, headers=_headers(url))
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(max_bytes + 1)
    except Exception as exc:
        raise ReserveGlobalDedupError(f"download failed for {url}: {exc}") from exc
    require(len(raw) <= max_bytes, f"bounded download exceeded {max_bytes}: {url}")
    return raw


def load_json_url(url: str, max_bytes: int = 400_000) -> dict[str, Any]:
    try:
        value = json.loads(download(url, max_bytes=max_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReserveGlobalDedupError(f"invalid JSON from {url}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not object: {url}")
    return value


def load_json_file(path: str) -> dict[str, Any]:
    try:
        value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReserveGlobalDedupError(f"invalid local policy {path}: {exc}") from exc
    require(isinstance(value, dict), f"policy root is not object: {path}")
    return value


def verify_source_run(run_id: int, expected_head: str, label: str) -> None:
    value = load_json_url(
        f"https://api.github.com/repos/Oleksii-debug/12-6-ai./actions/runs/{run_id}"
    )
    require(value.get("head_sha") == expected_head, f"{label}: workflow head drift")
    require(value.get("status") == "completed", f"{label}: workflow not completed")
    require(value.get("conclusion") == "success", f"{label}: workflow not successful")


def verify_execution_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        code4.git("rev-parse", "HEAD") == expected_execution_head,
        "execution HEAD drift",
    )
    ancestor = code4.incumbent._git(
        "merge-base",
        "--is-ancestor",
        PARENT_CODE4_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "terminal code4 parent is not execution ancestor")
    require(
        code4.git(
            "rev-parse",
            "HEAD:tools/run_d03_four_code_sources_global_dedup_v1.py",
        )
        == EXPECTED_CODE4_RUNNER_BLOB,
        "terminal code4 runner blob drift",
    )
    for path, expected_blob in (
        (RICH_POLICY_PATH, RICH_POLICY_BLOB),
        (FASTAPI_POLICY_PATH, FASTAPI_POLICY_BLOB),
    ):
        require(
            code4.git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"blob drift: {path}",
        )
        require(
            code4.git("hash-object", str(ROOT / path)) == expected_blob,
            f"worktree drift: {path}",
        )

    authority = code4.verify_execution_authority(expected_execution_head)
    verify_source_run(RICH_SOURCE_RUN, RICH_SOURCE_HEAD, "Rich source authority")
    verify_source_run(
        FASTAPI_SOURCE_RUN,
        FASTAPI_SOURCE_HEAD,
        "FastAPI source authority",
    )
    verify_source_run(
        FASTAPI_REGISTRY_SEAL_RUN,
        FASTAPI_SOURCE_HEAD,
        "FastAPI registry seal",
    )
    return {
        **authority,
        RICH_POLICY_PATH: RICH_POLICY_BLOB,
        FASTAPI_POLICY_PATH: FASTAPI_POLICY_BLOB,
        "tools/run_d03_four_code_sources_global_dedup_v1.py": (
            EXPECTED_CODE4_RUNNER_BLOB
        ),
    }


def validate_reserve_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    rich = load_json_file(RICH_POLICY_PATH)
    require(
        rich.get("schema_version") == "12-6.next100-051-rich-code-rights.v1",
        "Rich policy schema drift",
    )
    require(rich.get("source_family") == RICH_FAMILY, "Rich family drift")
    require(
        rich.get("upstream_commit")
        == "6ac483cbea39cab124dfd3483bba70ffafb71050",
        "Rich commit drift",
    )
    rich_decisions = rich.get("decisions")
    require(
        isinstance(rich_decisions, list)
        and len(rich_decisions) == RICH_EXPECTED_OBJECTS,
        "Rich object count drift",
    )
    require(
        sum(int(row["size_bytes"]) for row in rich_decisions)
        == RICH_EXPECTED_BYTES,
        "Rich byte count drift",
    )
    require(
        all(row.get("source_family") == RICH_FAMILY for row in rich_decisions),
        "Rich decision family drift",
    )
    require(
        rich.get("uses", {}).get("model_training") == "ALLOWED",
        "Rich training rights drift",
    )
    require(
        rich.get("uses", {}).get("evaluation") == "NOT_SEPARATELY_ADMITTED",
        "Rich evaluation boundary drift",
    )

    fast = load_json_file(FASTAPI_POLICY_PATH)
    require(
        fast.get("schema_version")
        == "12-6.next100-044-fastapi-code-rights-policy.v1",
        "FastAPI policy schema drift",
    )
    upstream = fast.get("upstream", {})
    require(
        upstream.get("canonical_family_id") == FASTAPI_FAMILY,
        "FastAPI family drift",
    )
    require(
        upstream.get("commit")
        == "49033471594ea5d99a80abdf1043231b7791ee49",
        "FastAPI commit drift",
    )
    inventory = fast.get("inventory")
    require(
        isinstance(inventory, list)
        and len(inventory) == FASTAPI_EXPECTED_OBJECTS,
        "FastAPI object count drift",
    )
    require(
        sum(int(row["expected_bytes"]) for row in inventory)
        == FASTAPI_EXPECTED_BYTES,
        "FastAPI byte count drift",
    )
    training = fast.get("training_purpose_authority", {})
    require(
        training.get("decision") == "ALLOWED",
        "FastAPI training rights drift",
    )
    require(
        training.get("evaluation") == "NOT_ADMITTED",
        "FastAPI evaluation boundary drift",
    )
    require(
        training.get("reserved_for_evaluation") is False,
        "FastAPI evaluation reservation drift",
    )
    return rich, fast


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
    return code4._source_row(
        repository=repository,
        commit=commit,
        family=family,
        source_id=source_id,
        path=path,
        url=url,
        expected_bytes=expected_bytes,
        expected_blob=expected_blob,
        raw=raw,
    )


def build_reserve_sources(
    rich: Mapping[str, Any],
    fast: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    family_groups: dict[tuple[str, str, str], list[str]] = {}

    rich_repo = "Textualize/rich"
    rich_commit = str(rich["upstream_commit"])
    for item in rich["decisions"]:
        source_id = str(item["source_id"])
        raw = download(str(item["raw_url"]), max_bytes=100_000)
        row = _source_row(
            repository=rich_repo,
            commit=rich_commit,
            family=RICH_FAMILY,
            source_id=source_id,
            path=str(item["path"]),
            url=str(item["raw_url"]),
            expected_bytes=int(item["size_bytes"]),
            expected_blob=str(item["blob_sha1"]),
            raw=raw,
        )
        rows.append(row)
        payloads[source_id] = raw
        family_groups.setdefault(
            (rich_repo, rich_commit, RICH_FAMILY),
            [],
        ).append(source_id)

    fast_upstream = fast["upstream"]
    fast_repo = "fastapi/fastapi"
    fast_commit = str(fast_upstream["commit"])
    for item in fast["inventory"]:
        source_id = str(item["source_id"])
        raw = download(str(item["raw_url"]), max_bytes=100_000)
        row = _source_row(
            repository=fast_repo,
            commit=fast_commit,
            family=FASTAPI_FAMILY,
            source_id=source_id,
            path=str(item["path"]),
            url=str(item["raw_url"]),
            expected_bytes=int(item["expected_bytes"]),
            expected_blob=str(item["blob_sha1"]),
            raw=raw,
        )
        rows.append(row)
        payloads[source_id] = raw
        family_groups.setdefault(
            (fast_repo, fast_commit, FASTAPI_FAMILY),
            [],
        ).append(source_id)

    require(len(rows) == EXPECTED_RESERVE_OBJECTS, "reserve object count drift")
    require(
        set(payloads) == {row["source_id"] for row in rows},
        "reserve payload coverage drift",
    )
    require(
        sum(int(row["declared_capacity_bytes"]) for row in rows)
        == EXPECTED_RESERVE_BYTES,
        "reserve declared-capacity drift",
    )
    require(
        len({row["source_id"] for row in rows}) == len(rows),
        "reserve source-id collision",
    )

    edges: list[dict[str, Any]] = []
    for (repository, commit, _family), source_ids in sorted(
        family_groups.items()
    ):
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
    require(len(edges) == 7, "reserve sibling-edge count drift")
    return rows, payloads, edges


def compose_reserve_graph(
    code4_inventory: Mapping[str, Any],
    code4_payloads: Mapping[str, bytes],
    reserve_rows: list[dict[str, Any]],
    reserve_payloads: Mapping[str, bytes],
    reserve_edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(
        type(code4_inventory) is dict,
        "code4 inventory must be exact object",
    )
    base_rows = code4_inventory.get("sources")
    require(isinstance(base_rows, list), "code4 source rows missing")
    require(
        len(base_rows) == EXPECTED_CODE4_OBJECTS,
        "code4 object count drift",
    )
    base_ids = {
        row.get("source_id")
        for row in base_rows
        if isinstance(row, Mapping)
    }
    require(len(base_ids) == len(base_rows), "code4 source identities invalid")
    require(
        base_ids == set(code4_payloads),
        "code4 inventory/payload mismatch",
    )
    require(
        code4.incumbent._declared_capacity_bytes(
            code4_inventory,
            code4_payloads,
            label="terminal code4 graph",
        )
        == EXPECTED_CODE4_BYTES,
        "terminal code4 declared-capacity drift",
    )

    base_families = {
        row.get("source_family")
        for row in base_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("source_family"), str)
    }
    require(
        RICH_FAMILY not in base_families,
        "Rich family already exists in code4 graph",
    )
    require(
        FASTAPI_FAMILY not in base_families,
        "FastAPI family already exists in code4 graph",
    )

    reserve_ids = {row["source_id"] for row in reserve_rows}
    require(
        reserve_ids == set(reserve_payloads),
        "reserve inventory/payload mismatch",
    )
    require(
        not (base_ids & reserve_ids),
        "reserve source-id collision with code4 graph",
    )

    lineage_edges = code4_inventory.get("lineage_edges", [])
    require(isinstance(lineage_edges, list), "code4 lineage_edges malformed")
    inventory = copy.deepcopy(dict(code4_inventory))
    inventory["sources"] = [
        *copy.deepcopy(base_rows),
        *copy.deepcopy(reserve_rows),
    ]
    inventory["lineage_edges"] = [
        *copy.deepcopy(lineage_edges),
        *copy.deepcopy(reserve_edges),
    ]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact terminal code4 graph from PR #2734 plus exact terminal Rich "
        "v15.0.0 and FastAPI source-authority objects. All pair decisions "
        "delegate to the incumbent PR #824 V3 semantics through the "
        "independently qualified PR #1459 indexed executor. This execution "
        "grants zero canonical capacity credit."
    )

    payloads = dict(code4_payloads)
    payloads.update(reserve_payloads)
    quarantine_authority = json.loads(
        (
            ROOT / code4.incumbent.clean_successor.QUARANTINE_CONFIG_PATH
        ).read_text(encoding="utf-8")
    )
    code4.incumbent._verify_clean_payload_graph(
        inventory,
        payloads,
        quarantine_authority,
    )
    require(
        len(payloads) == EXPECTED_COMBINED_OBJECTS,
        "combined object count drift",
    )
    require(
        code4.incumbent._declared_capacity_bytes(
            inventory,
            payloads,
            label="code4 plus Rich/FastAPI reserve",
        )
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )
    return inventory, payloads


def validate_survivor_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    require(
        projection.get("schema_version")
        == code4.incumbent.v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    supplied = projection.get("survivor_authority_sha256")
    require(
        isinstance(supplied, str) and len(supplied) == 64,
        "survivor identity invalid",
    )
    core = dict(projection)
    core.pop("survivor_authority_sha256", None)
    require(
        supplied == sha256(canonical(core)),
        "survivor identity drift",
    )
    require(
        projection.get("matcher_report_sha256")
        == report.get("report_sha256"),
        "survivor matcher identity drift",
    )
    require(
        projection.get("pre_dedup_source_object_count")
        == EXPECTED_COMBINED_OBJECTS,
        "survivor pre-dedup object count drift",
    )
    require(
        projection.get("pre_dedup_declared_capacity_bytes")
        == EXPECTED_COMBINED_BYTES,
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


def _execute_matcher(
    matcher: Any,
    inventory: dict[str, Any],
    payloads: dict[str, bytes],
    *,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
    stage: str,
) -> tuple[dict[str, Any], float]:
    indexed = code4.incumbent.indexed
    try:
        indexed.attest_incumbent_runtime(matcher)
    except indexed.IndexedExecutionError as exc:
        raise ReserveGlobalDedupError(
            f"{stage}_attestation: {exc}"
        ) from exc

    started = time.perf_counter()
    try:
        report = indexed.audit_payloads_indexed(
            matcher,
            inventory,
            payloads,
            max_candidate_pairs=max_candidate_pairs,
            max_index_postings=max_index_postings,
            max_pair_expansions=max_pair_expansions,
        )
    except indexed.IndexedExecutionError as exc:
        raise ReserveGlobalDedupError(
            f"{stage}_indexed_execution: {exc}"
        ) from exc
    seconds = time.perf_counter() - started
    matcher.verify_report(report)
    return report, seconds


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
    authority_blobs = verify_execution_authority(expected_execution_head)
    rich, fast = validate_reserve_contracts()

    physical_summary, captured = code4.capture_physical_sources()
    code4_rows, code4_extension_payloads, code4_edges = (
        code4.build_extension_graph(captured)
    )
    reserve_rows, reserve_payloads, reserve_edges = build_reserve_sources(
        rich,
        fast,
    )

    verified_v7_head = code4.incumbent._verify_v7_worktree(v7_root)
    config = code4.incumbent.v8.load_config(
        ROOT / "configs/data/next100_065f_global_dedup_v8.json"
    )
    matcher, base_inventory, base_payloads, removal = (
        code4.incumbent._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    require(
        code4.incumbent._verify_v7_worktree(v7_root) == verified_v7_head,
        "V7 worktree drifted during reconstruction",
    )
    require(
        len(base_payloads) == code4.EXPECTED_BASE_OBJECTS,
        "incumbent object count drift",
    )
    require(
        code4.incumbent._declared_capacity_bytes(
            base_inventory,
            base_payloads,
            label="incumbent V8 base",
        )
        == code4.EXPECTED_BASE_BYTES,
        "incumbent declared-capacity drift",
    )

    code4_inventory, code4_payloads = code4.compose_graph(
        base_inventory,
        base_payloads,
        code4_rows,
        code4_extension_payloads,
        code4_edges,
    )
    combined_inventory, combined_payloads = compose_reserve_graph(
        code4_inventory,
        code4_payloads,
        reserve_rows,
        reserve_payloads,
        reserve_edges,
    )

    code4_report, code4_seconds = _execute_matcher(
        matcher,
        copy.deepcopy(code4_inventory),
        dict(code4_payloads),
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
        stage="code4_baseline",
    )
    code4_terminal = code4.terminal_summary(
        code4_report,
        expected_objects=EXPECTED_CODE4_OBJECTS,
        expected_bytes=EXPECTED_CODE4_BYTES,
        label="terminal code4 baseline",
    )
    code4_report_identity = str(code4_report["report_sha256"])
    code4_report_bytes = canonical(code4_report)
    code4_report_publication = json.loads(code4_report_bytes.decode("utf-8"))
    require(
        canonical(code4_report_publication) == code4_report_bytes,
        "code4 report canonical round-trip drift",
    )
    del code4_report

    combined_report, combined_seconds = _execute_matcher(
        matcher,
        combined_inventory,
        combined_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
        stage="reserve_combined",
    )
    combined_terminal = code4.terminal_summary(
        combined_report,
        expected_objects=EXPECTED_COMBINED_OBJECTS,
        expected_bytes=EXPECTED_COMBINED_BYTES,
        label="code4 plus reserve",
    )

    survivor_projection = (
        code4.incumbent.v9_semantics._derive_survivors(combined_report)
    )
    validate_survivor_projection(
        combined_report,
        survivor_projection,
        combined_terminal,
    )

    code4_unique = int(
        code4_terminal["conservative_unique_capacity_bytes_after"]
    )
    combined_unique = int(
        combined_terminal["conservative_unique_capacity_bytes_after"]
    )
    reserve_marginal = combined_unique - code4_unique
    reserve_loss_vs_raw = EXPECTED_RESERVE_BYTES - reserve_marginal
    projected_late_gate_headroom = (
        CODE4_TERMINAL_LATE_GATE_HEADROOM_BYTES + reserve_marginal
    )

    reserve_ids = {row["source_id"] for row in reserve_rows}
    survivor_ids = set(survivor_projection["survivor_source_ids"])
    reserve_survivors = sorted(reserve_ids & survivor_ids)
    by_id = {row["source_id"]: row for row in reserve_rows}
    reserve_survivor_declared = sum(
        int(by_id[source_id]["declared_capacity_bytes"])
        for source_id in reserve_survivors
    )

    max_rss = code4.incumbent._max_rss_kib()
    require(
        type(max_rss) is int and max_rss > 0,
        "process max RSS unavailable",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": expected_execution_head,
        "terminal_code4_parent": {
            "head_sha": PARENT_CODE4_HEAD,
            "run_id": CODE4_TERMINAL_RUN,
            "candidate_raw_bytes": CODE4_TERMINAL_MARGINAL_BYTES,
            "marginal_global_unique_capacity_bytes": (
                CODE4_TERMINAL_MARGINAL_BYTES
            ),
            "late_gate_headroom_bytes": (
                CODE4_TERMINAL_LATE_GATE_HEADROOM_BYTES
            ),
            "base_matcher_report_sha256": CODE4_TERMINAL_BASE_REPORT,
            "combined_matcher_report_sha256": (
                CODE4_TERMINAL_COMBINED_REPORT
            ),
            "survivor_authority_sha256": (
                CODE4_TERMINAL_SURVIVOR_AUTHORITY
            ),
            "two_clean_proof_identity_sha256": (
                CODE4_TERMINAL_PROOF_IDENTITY
            ),
        },
        "source_authorities": {
            "rich": {
                "source_head_sha": RICH_SOURCE_HEAD,
                "dedicated_run_id": RICH_SOURCE_RUN,
                "policy_git_blob_sha1": RICH_POLICY_BLOB,
                "source_family": RICH_FAMILY,
                "object_count": RICH_EXPECTED_OBJECTS,
                "raw_bytes": RICH_EXPECTED_BYTES,
                "evaluation_use": "NOT_SEPARATELY_ADMITTED",
            },
            "fastapi": {
                "source_head_sha": FASTAPI_SOURCE_HEAD,
                "dedicated_run_id": FASTAPI_SOURCE_RUN,
                "registry_seal_run_id": FASTAPI_REGISTRY_SEAL_RUN,
                "policy_git_blob_sha1": FASTAPI_POLICY_BLOB,
                "source_family": FASTAPI_FAMILY,
                "object_count": FASTAPI_EXPECTED_OBJECTS,
                "raw_bytes": FASTAPI_EXPECTED_BYTES,
                "evaluation_use": "NOT_ADMITTED",
            },
        },
        "authority_path_blobs": authority_blobs,
        "physical_code4_source_authority": {
            "dedicated_run_id": 37269452704,
            "source_family_count": 4,
            "object_count": code4.EXPECTED_EXTENSION_OBJECTS,
            "candidate_raw_bytes": code4.EXPECTED_EXTENSION_BYTES,
            "summary_sha256": code4.EXPECTED_PHYSICAL_SUMMARY_SHA256,
            "observed_summary_sha256": sha256(
                code4.physical.canonical(physical_summary)
            ),
        },
        "code4_baseline": {
            "v7_head_sha": verified_v7_head,
            "source_object_count": EXPECTED_CODE4_OBJECTS,
            "declared_capacity_bytes": EXPECTED_CODE4_BYTES,
            "matcher_report_sha256": code4_report_identity,
            "post_dedup_conservative_unique_bytes": code4_unique,
            "duplicate_discount_bytes": (
                code4_terminal["duplicate_discount_bytes"]
            ),
            "nomis1864_deauthorization": removal,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "matcher_report_sha256": combined_report["report_sha256"],
            "post_dedup_conservative_unique_bytes": combined_unique,
            "duplicate_discount_bytes": (
                combined_terminal["duplicate_discount_bytes"]
            ),
            "survivor_authority_sha256": survivor_projection[
                "survivor_authority_sha256"
            ],
        },
        "reserve_delta": {
            "source_families": [FASTAPI_FAMILY, RICH_FAMILY],
            "pre_dedup_candidate_objects": EXPECTED_RESERVE_OBJECTS,
            "pre_dedup_candidate_bytes": EXPECTED_RESERVE_BYTES,
            "marginal_global_unique_capacity_bytes": reserve_marginal,
            "global_dedup_loss_vs_raw_candidate_bytes": (
                reserve_loss_vs_raw
            ),
            "reserve_survivor_source_objects": len(reserve_survivors),
            "reserve_survivor_declared_capacity_bytes": (
                reserve_survivor_declared
            ),
            "code4_terminal_late_gate_headroom_before_reserve": (
                CODE4_TERMINAL_LATE_GATE_HEADROOM_BYTES
            ),
            "projected_late_gate_headroom_after_global_dedup": (
                projected_late_gate_headroom
            ),
            "code_target_bytes": code4.CODE_TARGET_BYTES,
            "current_post_qp_code_bytes_reference": (
                code4.CURRENT_POST_QP_CODE_BYTES_REFERENCE
            ),
            "still_possible_to_reach_code_target_after_later_gates": (
                projected_late_gate_headroom >= 0
            ),
        },
        "indexed_execution": {
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "code4_baseline_wall_clock_seconds": round(
                code4_seconds,
                6,
            ),
            "combined_wall_clock_seconds": round(
                combined_seconds,
                6,
            ),
            "process_max_rss_kib": max_rss,
            "performance_equivalence_authority": "MERGED_PR_1459",
        },
        "content_boundary": {
            "raw_source_bytes_persisted": False,
            "raw_source_bytes_uploaded": False,
            "code4_report_text_free": True,
            "combined_report_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_executed_for_exact_reserve": True,
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

    code4.incumbent._publish_json_outputs(
        (
            (output_code4_report, code4_report_publication),
            (output_combined_report, combined_report),
            (output_survivors, survivor_projection),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--v7-root", type=Path)
    parser.add_argument("--bulk-workspace", type=Path)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--output-code4-report", type=Path)
    parser.add_argument("--output-combined-report", type=Path)
    parser.add_argument("--output-survivors", type=Path)
    parser.add_argument("--output-evidence", type=Path)
    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=5_000_000,
    )
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=code4.incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=code4.incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()

    try:
        if args.contract_check:
            verify_execution_authority(args.expected_execution_head)
            validate_reserve_contracts()
            print(
                "D03_CODE4_RICH_FASTAPI_CONTRACT=PASS "
                f"objects={EXPECTED_RESERVE_OBJECTS} "
                f"bytes={EXPECTED_RESERVE_BYTES}"
            )
            return 0

        required_paths = (
            args.v7_root,
            args.bulk_workspace,
            args.output_code4_report,
            args.output_combined_report,
            args.output_survivors,
            args.output_evidence,
        )
        require(
            all(path is not None for path in required_paths),
            "execution paths are required",
        )
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
        ReserveGlobalDedupError,
        code4.FourCodeGlobalDedupError,
        code4.incumbent.Franko1901GlobalDedupError,
        code4.incumbent.indexed.IndexedExecutionError,
        OSError,
        ValueError,
    ) as exc:
        rendered = " ".join(str(exc).split())[:240]
        print(
            json.dumps(
                {
                    "status": (
                        "BLOCKED_CODE4_RICH_FASTAPI_RESERVE_GLOBAL_DEDUP"
                    ),
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
    print(
        "D03_CODE4_RICH_FASTAPI_RESERVE_GLOBAL_DEDUP="
        "PASS_ZERO_CREDIT"
    )
    print(
        "RESERVE_MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(delta["marginal_global_unique_capacity_bytes"])
    )
    print(
        "PROJECTED_LATE_GATE_HEADROOM_AFTER_GLOBAL_DEDUP="
        + str(
            delta["projected_late_gate_headroom_after_global_dedup"]
        )
    )
    print(
        "EVIDENCE_IDENTITY_SHA256="
        + evidence["evidence_identity_sha256"]
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
