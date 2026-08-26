#!/usr/bin/env python3
"""Fail-closed exact/live authority validator for NEXT100-065E V7."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_065e_cross_source_dedup_v7.json")
REPO = "Oleksii-debug/12-6-ai."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065E FAIL: {message}")


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "config root must be object")
    return value


def validate_static(config: dict[str, Any]) -> None:
    require(
        config.get("schema_version") == "12-6.next100-065e-cross-source-dedup.v7",
        "schema drift",
    )
    require(config.get("worker_id") == "NEXT100-065E-CROSSSOURCE-DEDUP-V7", "worker drift")
    attrs = config.get("attrs", {})
    require(attrs.get("pr") == 474, "attrs PR drift")
    require(
        attrs.get("head_sha") == "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b",
        "attrs head drift",
    )
    require(attrs.get("dedicated_workflow_run") == 33006080831, "attrs run drift")
    require(attrs.get("dedicated_workflow_conclusion") == "success", "attrs run not sealed success")
    require(
        attrs.get("authority_identity_sha256")
        == "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4",
        "attrs authority drift",
    )
    require(attrs.get("terminal_artifact_id") == 9621650719, "attrs artifact id drift")
    require(
        attrs.get("terminal_artifact_digest")
        == "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2",
        "attrs artifact digest drift",
    )
    require(attrs.get("exact_capacity_bytes") == 170435, "attrs capacity drift")
    require(len(attrs.get("files", [])) == 4, "attrs file-count drift")
    require(
        sum(int(item["raw_bytes"]) for item in attrs["files"]) == 170435,
        "attrs file-capacity arithmetic drift",
    )
    expected = config.get("expected_vector", {})
    require(expected.get("source_object_count") == 35, "V7 object count drift")
    require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6},
        "V7 family vector drift",
    )
    require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2215615,
        "V7 total vector drift",
    )
    require(
        config.get("registry_v5_reconciliation", {}).get("registry_workflow_terminal") is False,
        "queued registry evidence was promoted",
    )
    for key, value in config.get("claim_boundary", {}).items():
        require(value is False, f"truth boundary weakened: {key}")


def github_get(path: str) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GITHUB_TOKEN is required for --github-live")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-ai-next100-065e-authority-validator",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response not object: {path}")
    return value


def validate_live(config: dict[str, Any]) -> None:
    attrs = config["attrs"]
    run = github_get(f"actions/runs/{attrs['dedicated_workflow_run']}")
    require(run.get("head_sha") == attrs["head_sha"], "attrs run head mismatch")
    require(run.get("status") == "completed", "attrs run is nonterminal")
    require(run.get("conclusion") == "success", "attrs run conclusion is not success")

    artifact = github_get(f"actions/artifacts/{attrs['terminal_artifact_id']}")
    require(artifact.get("expired") is False, "attrs artifact expired")
    require(
        artifact.get("name")
        == "next100-053-attrs-admission-cda0232d5574ef91eae0d7e0b7fa5efddcbe218b",
        "attrs artifact name drift",
    )
    digest = artifact.get("digest")
    if digest is not None:
        require(digest == attrs["terminal_artifact_digest"], "attrs artifact digest drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    validate_static(config)
    if args.github_live:
        validate_live(config)
    print("NEXT100-065E authority bindings: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
