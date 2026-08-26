#!/usr/bin/env python3
"""Metadata-only exact-tree probe for the pinned PluG/PluG2 upstream revision.

This tool never downloads corpus text. It inventories Git blob identities and sizes
from one immutable GitHub tree and fails closed if GitHub truncates that tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_plug2_historical_ua_probe_v1.json"
API = "https://api.github.com"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"D03_PLUG2_TREE_FAIL: {message}")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "config root must be object")
    return value


def github_get(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "12-6-ai-d03-plug2-tree-probe",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_snapshot(config: dict[str, Any], commit: dict[str, Any], tree: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    expected_commit = source["pinned_commit_sha"]
    expected_tree = source["pinned_root_tree_sha"]
    require(commit.get("sha") == expected_commit, "commit identity moved")
    commit_tree = commit.get("commit", {}).get("tree", {}).get("sha")
    require(commit_tree == expected_tree, "commit root tree mismatch")
    require(tree.get("sha") == expected_tree, "recursive tree identity mismatch")
    require(tree.get("truncated") is False, "recursive GitHub tree is truncated")

    entries = tree.get("tree")
    require(isinstance(entries, list), "tree entries missing")
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
    require(len(by_path) == len(entries), "duplicate or invalid tree paths")

    root_evidence: list[dict[str, Any]] = []
    for expected in config["pinned_root_objects"]:
        path = expected["path"]
        actual = by_path.get(path)
        require(isinstance(actual, dict), f"missing pinned root object: {path}")
        require(actual.get("type") == expected["type"], f"root object type drift: {path}")
        require(actual.get("sha") == expected["oid"], f"root object OID drift: {path}")
        if expected["type"] == "blob":
            require(actual.get("size") == expected["bytes"], f"root blob size drift: {path}")
        root_evidence.append(
            {
                "path": path,
                "type": actual["type"],
                "oid": actual["sha"],
                "bytes": actual.get("size"),
            }
        )

    def summarize(prefix: str) -> dict[str, Any]:
        blobs = [
            entry
            for entry in entries
            if entry.get("type") == "blob"
            and isinstance(entry.get("path"), str)
            and entry["path"].startswith(prefix + "/")
            and entry["path"].lower().endswith(".txt")
        ]
        require(blobs, f"no text blobs found under {prefix}")
        require(all(isinstance(item.get("size"), int) and item["size"] >= 0 for item in blobs), f"missing blob size under {prefix}")
        require(all(isinstance(item.get("sha"), str) and len(item["sha"]) == 40 for item in blobs), f"missing blob OID under {prefix}")
        return {
            "prefix": prefix,
            "text_blob_count": len(blobs),
            "text_blob_bytes": sum(item["size"] for item in blobs),
            "largest_text_blob_bytes": max(item["size"] for item in blobs),
        }

    snapshot: dict[str, Any] = {
        "schema_version": "12-6.d03-plug2-github-tree-snapshot.v1",
        "repository": source["repository"],
        "commit_sha": expected_commit,
        "root_tree_sha": expected_tree,
        "tree_truncated": False,
        "root_objects": root_evidence,
        "payloads": [summarize("PluG_texts"), summarize("PluG2_texts")],
        "raw_text_emitted": False,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()
    return snapshot


def validate_snapshot(snapshot: dict[str, Any], config: dict[str, Any]) -> None:
    require(snapshot["schema_version"] == "12-6.d03-plug2-github-tree-snapshot.v1", "snapshot schema drift")
    require(snapshot["repository"] == config["source"]["repository"], "repository drift")
    require(snapshot["commit_sha"] == config["source"]["pinned_commit_sha"], "snapshot commit drift")
    require(snapshot["root_tree_sha"] == config["source"]["pinned_root_tree_sha"], "snapshot tree drift")
    require(snapshot["tree_truncated"] is False, "truncated snapshot")
    require([item["prefix"] for item in snapshot["payloads"]] == ["PluG_texts", "PluG2_texts"], "payload order drift")
    require(all(item["text_blob_count"] > 0 for item in snapshot["payloads"]), "empty payload")
    require(all(item["largest_text_blob_bytes"] <= config["acquisition_policy"]["max_single_text_blob_bytes"] for item in snapshot["payloads"]), "single-member size cap exceeded")
    require(snapshot["raw_text_emitted"] is False, "raw text must not be emitted")
    require(snapshot["training_authorized_bytes"] == 0, "metadata probe cannot authorize training bytes")
    require(snapshot["tokenizer_fit_authorized"] is False, "metadata probe cannot authorize tokenizer fit")
    require(snapshot["model_training_executed"] is False, "metadata probe cannot execute training")
    require(snapshot["paid_compute_used"] is False, "metadata probe cannot use paid compute")
    claimed = snapshot.get("snapshot_sha256")
    unhashed = dict(snapshot)
    unhashed.pop("snapshot_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    require(claimed == actual, "snapshot self-hash mismatch")


def live_probe(config: dict[str, Any]) -> dict[str, Any]:
    repo = config["source"]["repository"]
    commit_sha = config["source"]["pinned_commit_sha"]
    tree_sha = config["source"]["pinned_root_tree_sha"]
    commit = github_get(f"{API}/repos/{repo}/commits/{commit_sha}")
    tree = github_get(f"{API}/repos/{repo}/git/trees/{tree_sha}?recursive=1")
    snapshot = build_snapshot(config, commit, tree)
    validate_snapshot(snapshot, config)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    snapshot = live_probe(config)
    payload = json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
