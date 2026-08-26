#!/usr/bin/env python3
"""Fail-closed immutable/live authority validator for NEXT100-065D V6."""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_065d_cross_source_dedup_v6.json")
GUARD_CONFIG = Path("configs/data/next100_065d_convergence_guard_v1.json")
REPO = "Oleksii-debug/12-6-ai."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065D FAIL: {message}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: config root must be object")
    return value


def validate_static(config: dict[str, Any], guard: dict[str, Any]) -> None:
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
    require(numpy.get("exact_capacity_bytes") == 36898, "NumPy capacity drift")

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
    require(pg.get("exact_capacity_bytes") == 1672110, "Gutenberg capacity drift")

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
    require(attrs.get("artifact_id") == 9621650719, "attrs artifact id drift")
    require(
        attrs.get("artifact_digest")
        == "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2",
        "attrs artifact digest drift",
    )
    require(attrs.get("source_family") == "github:python-attrs/attrs", "attrs family drift")
    require(attrs.get("exact_capacity_bytes") == 170435, "attrs capacity drift")
    require(len(attrs.get("files", [])) == 4, "attrs file count drift")

    expected = config.get("expected_vector", {})
    require(expected.get("source_object_count") == 35, "source object count drift")
    require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6},
        "family vector drift",
    )
    require(
        expected.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 276466, "total": 2200075},
        "fixed capacity vector drift",
    )
    require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2215615,
        "pre-dedup planning total drift",
    )
    require(
        expected.get("planning_gap_if_no_successor_global_dedup_collapse") == 17784385,
        "planning gap drift",
    )

    require(
        guard.get("schema_version") == "12-6.next100-065d-convergence-guard.v1",
        "convergence guard schema drift",
    )
    require(
        guard.get("worker_id") == "NEXT100-065D-CANONICAL-CONVERGENCE-GUARD",
        "convergence guard worker drift",
    )
    require(guard.get("local_free_only") is True, "convergence guard LOCAL_FREE weakened")

    registry = guard.get("canonical_registry_v3", {})
    require(registry.get("pr") == 538, "canonical registry PR drift")
    require(
        registry.get("registry_identity_sha256")
        == "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c",
        "canonical registry identity drift",
    )
    require(registry.get("numeric_training_capacity_bytes") == 357530, "canonical registry capacity drift")
    require(
        registry.get("authorized_balanced_no_replay_loss_positions") == 0,
        "canonical registry illegally authorizes loss positions",
    )

    additions = guard.get("post_v3_terminal_additions", {})
    cp = additions.get("cpython_accepted_only", {})
    require(
        cp.get("head_sha") == "8f0cbc16f9a920ca9ab3e3061b53fbfec8838d77",
        "CPython accepted-only head drift",
    )
    require(cp.get("dedicated_workflow_run") == 33005689174, "CPython accepted-only run drift")
    require(cp.get("exact_eligible_capacity_bytes") == 15540, "CPython exact eligible capacity drift")
    attrs_guard = additions.get("attrs", {})
    require(attrs_guard.get("head_sha") == attrs.get("head_sha"), "attrs guard head mismatch")
    require(
        attrs_guard.get("dedicated_workflow_run") == attrs.get("dedicated_workflow_run"),
        "attrs guard run mismatch",
    )
    require(attrs_guard.get("exact_eligible_capacity_bytes") == 170435, "attrs guard capacity drift")

    vector = guard.get("expected_composed_vector_before_successor_global_dedup", {})
    require(vector.get("source_object_count") == 35, "convergence-guard object count drift")
    require(
        vector.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6},
        "convergence-guard family vector drift",
    )
    require(
        vector.get("capacity_bytes")
        == {"uk": 100856, "en": 1838293, "code": 276466, "total": 2215615},
        "convergence-guard capacity drift",
    )
    require(vector.get("planning_gap_bytes") == 17784385, "convergence-guard planning gap drift")
    require(
        vector.get("authorized_balanced_no_replay_loss_positions") == 0,
        "convergence guard illegally authorizes loss positions",
    )

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
        require(config.get("claim_boundary", {}).get(key) is False, f"truth boundary weakened: {key}")


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


def decode_contents_json(payload: dict[str, Any], label: str) -> dict[str, Any]:
    require(payload.get("encoding") == "base64", f"{label} contents encoding drift")
    content = payload.get("content")
    require(isinstance(content, str) and content, f"{label} contents missing")
    raw = base64.b64decode(content, validate=False)
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"{label} JSON root must be object")
    return value


def validate_live_registry(guard: dict[str, Any]) -> None:
    expected = guard["canonical_registry_v3"]
    first_pr = github_get(f"pulls/{expected['pr']}")
    head_sha = first_pr.get("head", {}).get("sha")
    require(isinstance(head_sha, str) and head_sha, "canonical registry PR has no head SHA")
    live = decode_contents_json(
        github_get(f"contents/{expected['path']}?ref={head_sha}"),
        "canonical registry V3",
    )
    second_pr = github_get(f"pulls/{expected['pr']}")
    require(
        second_pr.get("head", {}).get("sha") == head_sha,
        "canonical registry PR moved during validation",
    )
    require(live.get("schema_version") == expected["schema_version"], "live canonical registry schema drift")
    require(
        live.get("registry_identity_sha256") == expected["registry_identity_sha256"],
        "live canonical registry identity drift",
    )
    parent = live.get("dedup_parent", {})
    require(
        parent.get("head_sha") == expected["dedup_parent_head_sha"],
        "live canonical dedup-parent head drift",
    )
    require(
        parent.get("dedicated_workflow_run") == expected["dedup_parent_workflow_run"],
        "live canonical dedup-parent run drift",
    )
    validate_run(
        int(expected["dedup_parent_workflow_run"]),
        str(expected["dedup_parent_head_sha"]),
        "canonical V3 dedup parent",
    )
    inventory = live.get("pre_successor_global_dedup_inventory", {})
    require(
        inventory.get("candidate_numeric_training_capacity_bytes")
        == expected["numeric_training_capacity_bytes"],
        "live canonical numeric capacity drift",
    )
    require(
        inventory.get("candidate_independent_family_count") == expected["independent_family_count"],
        "live canonical independent-family total drift",
    )
    downstream = live.get("downstream_gate_vector", {})
    require(
        downstream.get("authorized_balanced_no_replay_loss_positions") == 0,
        "live canonical registry exposure boundary drift",
    )


def validate_live(config: dict[str, Any], guard: dict[str, Any]) -> None:
    validate_live_registry(guard)
    for key, label in (("numpy", "NumPy"), ("attrs", "attrs")):
        spec = config[key]
        pr = github_get(f"pulls/{spec['pr']}")
        require(pr.get("head", {}).get("sha") == spec["head_sha"], f"{label} PR head moved")
        validate_run(int(spec["dedicated_workflow_run"]), str(spec["head_sha"]), label)

    pg = config["gutenberg"]
    seal_pr = github_get(f"pulls/{pg['pr']}")
    require(
        seal_pr.get("head", {}).get("sha") == pg["head_sha"],
        "Gutenberg seal PR head moved",
    )
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

    cp = guard["post_v3_terminal_additions"]["cpython_accepted_only"]
    cp_pr = github_get(f"pulls/{cp['pr']}")
    require(
        cp_pr.get("head", {}).get("sha") == cp["head_sha"],
        "CPython accepted-only PR head moved",
    )
    validate_run(
        int(cp["dedicated_workflow_run"]),
        str(cp["head_sha"]),
        "CPython accepted-only adapter",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    config = load_json(CONFIG)
    guard = load_json(GUARD_CONFIG)
    validate_static(config, guard)
    if args.github_live:
        validate_live(config, guard)
    print("NEXT100-065D AUTHORITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
