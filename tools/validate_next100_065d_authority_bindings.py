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

    numpy = config.get("numpy", {})
    require(numpy.get("pr") == 468, "NumPy PR drift")
    require(
        numpy.get("head_sha") == "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8",
        "NumPy head drift",
    )
    require(numpy.get("dedicated_workflow_run") == 32998548535, "NumPy run drift")
    require(numpy.get("dedicated_workflow_conclusion") == "success", "NumPy run not sealed success")
    require(numpy.get("terminal_artifact_id") == 9618015895, "NumPy artifact id drift")
    require(
        numpy.get("terminal_artifact_zip_sha256")
        == "402016760c2ea5b341ed15537bb173e9bf10a938870313f00fd5e617ba20b020",
        "NumPy artifact digest drift",
    )
    require(numpy.get("source_family") == "github:numpy/numpy", "NumPy family drift")
    require(numpy.get("exact_capacity_bytes") == 36898, "NumPy capacity drift")
    require(len(numpy.get("files", [])) == 5, "NumPy file count drift")

    pg = config.get("gutenberg", {})
    require(pg.get("pr") == 627, "Gutenberg seal PR drift")
    require(
        pg.get("head_sha") == "c50b3f9cf871792c03886bdc1ccdc144812be88f",
        "Gutenberg seal head drift",
    )
    require(pg.get("parent_pr") == 470, "Gutenberg parent PR drift")
    require(
        pg.get("parent_head_sha") == "3f4ad26e1e8f3406a1274418cf5f485814ce3032",
        "Gutenberg parent head drift",
    )
    require(pg.get("dedicated_workflow_run") == 32998859164, "Gutenberg run drift")
    require(pg.get("dedicated_workflow_conclusion") == "success", "Gutenberg run not sealed success")
    require(pg.get("artifact_id") == 9618402768, "Gutenberg artifact id drift")
    require(
        pg.get("artifact_digest")
        == "sha256:63fa5d9b403432074193e290beb0473b5a1f7b74de1ac30bad71b9ec8405e006",
        "Gutenberg artifact digest drift",
    )
    require(
        pg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg authority identity drift",
    )
    require(pg.get("source_family") == "en.project-gutenberg.public-domain-books", "Gutenberg family drift")
    require(pg.get("exact_capacity_bytes") == 1672110, "Gutenberg capacity drift")
    require(len(pg.get("records", [])) == 3, "Gutenberg record count drift")

    expected = config.get("expected_vector", {})
    require(expected.get("source_object_count") == 31, "source object count drift")
    require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "family vector drift",
    )
    require(
        expected.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 106031, "total": 2029640},
        "fixed capacity vector drift",
    )
    require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2045180,
        "pre-dedup planning total drift",
    )
    require(
        expected.get("planning_gap_if_no_successor_global_dedup_collapse") == 17954820,
        "planning gap drift",
    )

    boundary = config.get("claim_boundary", {})
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "balance_release_claimed",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "research_corpus_v1_terminal",
    ):
        require(boundary.get(key) is False, f"truth boundary weakened: {key}")


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


def validate_run(run_id: int, head_sha: str, label: str) -> None:
    run = github_get(f"actions/runs/{run_id}")
    require(run.get("head_sha") == head_sha, f"{label} dedicated run head mismatch")
    require(run.get("status") == "completed", f"{label} dedicated run nonterminal")
    require(run.get("conclusion") == "success", f"{label} dedicated run not success")


def validate_live(config: dict[str, Any]) -> None:
    numpy = config["numpy"]
    numpy_pr = github_get(f"pulls/{numpy['pr']}")
    require(numpy_pr.get("head", {}).get("sha") == numpy["head_sha"], "NumPy PR head moved")
    validate_run(int(numpy["dedicated_workflow_run"]), str(numpy["head_sha"]), "NumPy")

    pg = config["gutenberg"]
    seal_pr = github_get(f"pulls/{pg['pr']}")
    require(seal_pr.get("head", {}).get("sha") == pg["head_sha"], "Gutenberg seal PR head moved")
    parent_pr = github_get(f"pulls/{pg['parent_pr']}")
    require(
        parent_pr.get("head", {}).get("sha") == pg["parent_head_sha"],
        "Gutenberg parent PR head moved",
    )
    validate_run(
        int(pg["dedicated_workflow_run"]),
        str(pg["parent_head_sha"]),
        "Gutenberg parent",
    )


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
