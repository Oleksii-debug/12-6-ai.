#!/usr/bin/env python3
"""Fail-closed immutable/live authority validator for NEXT100-065D V6."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_065d_cross_source_dedup_v6.json")
REPO = "Oleksii-debug/12-6-ai."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065D FAIL: {message}")


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "config root must be object")
    return value


def validate_static(config: dict[str, Any]) -> None:
    require(
        config.get("schema_version") == "12-6.next100-065d-cross-source-dedup.v6",
        "schema drift",
    )
    require(config.get("worker_id") == "NEXT100-065D-CROSSSOURCE-DEDUP-V6", "worker drift")
    require(config.get("local_free_only") is True, "LOCAL_FREE weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        require(config.get(key) is False, f"execution boundary weakened: {key}")

    base = config.get("base_v5", {})
    require(
        base.get("config_path") == "configs/data/next100_065c_cross_source_dedup_v5.json",
        "V5 config path drift",
    )
    require(base.get("source_object_count") == 23, "V5 source count drift")
    require(
        base.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4},
        "V5 family vector drift",
    )

    numpy = config.get("numpy", {})
    require(numpy.get("pr") == 468, "NumPy PR drift")
    require(
        numpy.get("head_sha") == "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8",
        "NumPy head drift",
    )
    require(numpy.get("dedicated_workflow_run") == 32998548535, "NumPy run drift")
    require(numpy.get("dedicated_workflow_conclusion") == "success", "NumPy run not green")
    require(numpy.get("source_family") == "github:numpy/numpy", "NumPy family drift")
    require(numpy.get("exact_capacity_bytes") == 36898, "NumPy capacity drift")
    require(len(numpy.get("files", [])) == 5, "NumPy file count drift")
    require(
        sum(int(item["raw_bytes"]) for item in numpy["files"]) == 36898,
        "NumPy file-byte vector drift",
    )

    gutenberg = config.get("gutenberg", {})
    require(gutenberg.get("pr") == 627, "Gutenberg seal PR drift")
    require(
        gutenberg.get("head_sha") == "c50b3f9cf871792c03886bdc1ccdc144812be88f",
        "Gutenberg seal head drift",
    )
    require(gutenberg.get("parent_pr") == 470, "Gutenberg execution parent PR drift")
    require(
        gutenberg.get("parent_head_sha") == "3f4ad26e1e8f3406a1274418cf5f485814ce3032",
        "Gutenberg execution parent head drift",
    )
    require(gutenberg.get("dedicated_workflow_run") == 32998859164, "Gutenberg run drift")
    require(
        gutenberg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg terminal seal identity drift",
    )
    require(gutenberg.get("exact_capacity_bytes") == 1672110, "Gutenberg capacity drift")
    require(len(gutenberg.get("records", [])) == 3, "Gutenberg record count drift")
    require(
        sum(int(item["normalized_bytes"]) for item in gutenberg["records"]) == 1672110,
        "Gutenberg record-byte vector drift",
    )

    expected = config.get("expected_vector", {})
    require(expected.get("source_object_count") == 31, "V6 source count drift")
    require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 family vector drift",
    )
    require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2045180,
        "V6 pre-dedup planning vector drift",
    )
    require(
        expected.get("planning_gap_if_no_successor_global_dedup_collapse") == 17954820,
        "V6 planning gap drift",
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
            "User-Agent": "12-6-ai-next100-065d-authority-validator",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response not object: {path}")
    return value


def validate_live(config: dict[str, Any]) -> None:
    numpy = config["numpy"]
    numpy_pr = github_get(f"pulls/{numpy['pr']}")
    require(
        numpy_pr.get("head", {}).get("sha") == numpy["head_sha"],
        "NumPy PR head moved",
    )
    numpy_run = github_get(f"actions/runs/{numpy['dedicated_workflow_run']}")
    require(numpy_run.get("head_sha") == numpy["head_sha"], "NumPy run head mismatch")
    require(numpy_run.get("status") == "completed", "NumPy run nonterminal")
    require(numpy_run.get("conclusion") == "success", "NumPy run not success")

    gutenberg = config["gutenberg"]
    seal_pr = github_get(f"pulls/{gutenberg['pr']}")
    require(
        seal_pr.get("head", {}).get("sha") == gutenberg["head_sha"],
        "Gutenberg seal PR head moved",
    )
    parent_pr = github_get(f"pulls/{gutenberg['parent_pr']}")
    require(
        parent_pr.get("head", {}).get("sha") == gutenberg["parent_head_sha"],
        "Gutenberg execution parent head moved",
    )
    pg_run = github_get(f"actions/runs/{gutenberg['dedicated_workflow_run']}")
    require(
        pg_run.get("head_sha") == gutenberg["parent_head_sha"],
        "Gutenberg run head mismatch",
    )
    require(pg_run.get("status") == "completed", "Gutenberg run nonterminal")
    require(pg_run.get("conclusion") == "success", "Gutenberg run not success")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    validate_static(config)
    if args.github_live:
        validate_live(config)
    print("NEXT100-065D AUTHORITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
