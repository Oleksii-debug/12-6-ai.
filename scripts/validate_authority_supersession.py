#!/usr/bin/env python3
"""Validate the CI-286 machine authority/supersession graph.

LOCAL_FREE: Python standard library only.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

DEFAULT_GRAPH = pathlib.Path("evidence/ci286/authority-supersession.v1.json")
REQUIRED_SUBSYSTEMS = (
    "environment",
    "model",
    "trainer",
    "checkpoint",
    "data_rights",
    "corpus",
    "evaluation",
    "inference",
    "performance",
    "post_base",
)
NON_AUTHORITY_DISPOSITIONS = {"candidate", "historical", "parallel_evidence"}
DISALLOWED_INCUMBENT_EVIDENCE = {
    "failed",
    "blocked",
    "stale",
    "no_exact_head_run",
    "partial",
}
ALLOWED_EDGE_TYPES = {
    "supersedes",
    "revalidates",
    "retains",
    "invalidates_as_authority",
    "blocks_promotion",
    "candidate_against",
}


class GraphValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise GraphValidationError(message)


def validate_graph(graph: dict) -> None:
    if graph.get("schema_version") != "ci286.authority-supersession.v1":
        _fail("unexpected schema_version")

    policy = graph.get("policy", {})
    if policy.get("local_free_only") is not True:
        _fail("policy.local_free_only must be true")
    if policy.get("automatic_merge") is not False:
        _fail("policy.automatic_merge must be false")
    if policy.get("automatic_pr_close") is not False:
        _fail("policy.automatic_pr_close must be false")
    if policy.get("circular_supersession") != "fail":
        _fail("policy.circular_supersession must be 'fail'")

    required = graph.get("required_subsystems")
    if required != list(REQUIRED_SUBSYSTEMS):
        _fail("required_subsystems must match CI-286 contract exactly")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _fail("nodes must be a non-empty list")

    node_by_id = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            _fail("every node must have a non-empty string id")
        if node_id in node_by_id:
            _fail(f"duplicate node id: {node_id}")
        subsystem = node.get("subsystem")
        if subsystem not in REQUIRED_SUBSYSTEMS:
            _fail(f"{node_id}: unknown subsystem {subsystem!r}")
        disposition = node.get("disposition")
        if disposition not in {"incumbent", *NON_AUTHORITY_DISPOSITIONS}:
            _fail(f"{node_id}: invalid disposition {disposition!r}")
        if disposition in NON_AUTHORITY_DISPOSITIONS and "non_authority_reason" not in node:
            _fail(f"{node_id}: non-incumbent node requires non_authority_reason")
        if disposition == "incumbent":
            if node.get("evidence_state") in DISALLOWED_INCUMBENT_EVIDENCE:
                _fail(
                    f"{node_id}: incumbent cannot use evidence_state "
                    f"{node.get('evidence_state')!r}"
                )
            source = node.get("source", {})
            head_sha = source.get("head_sha")
            if not isinstance(head_sha, str) or len(head_sha) != 40:
                _fail(f"{node_id}: incumbent requires exact 40-char head_sha")
        node_by_id[node_id] = node

    subsystem_map = graph.get("subsystems")
    if not isinstance(subsystem_map, dict):
        _fail("subsystems must be an object")

    for subsystem in REQUIRED_SUBSYSTEMS:
        record = subsystem_map.get(subsystem)
        if not isinstance(record, dict):
            _fail(f"missing subsystem record: {subsystem}")
        if record.get("resolution") != "resolved":
            _fail(f"{subsystem}: resolution must be resolved")
        incumbents = [
            node
            for node in nodes
            if node["subsystem"] == subsystem
            and node["disposition"] == "incumbent"
        ]
        if len(incumbents) != 1:
            _fail(
                f"{subsystem}: expected exactly one incumbent, found {len(incumbents)}"
            )
        if record.get("incumbent") != incumbents[0]["id"]:
            _fail(
                f"{subsystem}: subsystem incumbent does not match node disposition"
            )

    edges = graph.get("edges")
    if not isinstance(edges, list):
        _fail("edges must be a list")

    supersedes = defaultdict(list)
    for edge in edges:
        edge_type = edge.get("type")
        if edge_type not in ALLOWED_EDGE_TYPES:
            _fail(f"unsupported edge type: {edge_type!r}")
        source = edge.get("from")
        target = edge.get("to")
        if source not in node_by_id:
            _fail(f"edge source does not exist: {source!r}")
        if target not in node_by_id:
            _fail(f"edge target does not exist: {target!r}")
        if source == target:
            _fail(f"self-edge is forbidden: {source}")
        if edge_type == "supersedes":
            supersedes[source].append(target)

    visiting = set()
    visited = set()

    def dfs(node_id: str, path: list[str]) -> None:
        if node_id in visiting:
            start = path.index(node_id)
            cycle = path[start:] + [node_id]
            _fail("circular supersession: " + " -> ".join(cycle))
        if node_id in visited:
            return
        visiting.add(node_id)
        path.append(node_id)
        for child in supersedes[node_id]:
            dfs(child, path)
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_by_id:
        dfs(node_id, [])

    registry = graph.get("registry_input", {})
    if registry.get("mode") != "materialized_retained_registry_plus_live_crosscheck":
        _fail("registry_input.mode does not match the declared provenance strategy")
    if registry.get("standalone_ci274_registry_found") is not False:
        _fail("standalone CI-274 registry must not be claimed without evidence")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = pathlib.Path(argv[0]) if argv else DEFAULT_GRAPH
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
        validate_graph(graph)
    except (OSError, json.JSONDecodeError, GraphValidationError) as exc:
        print(f"CI-286 authority graph: FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "CI-286 authority graph: PASS "
        f"({len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
        f"{len(REQUIRED_SUBSYSTEMS)} resolved incumbents)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
