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

    require(
        guard.get("schema_version") == "12-6.next100-065d-convergence-guard.v1",
        "convergence guard schema drift",
    )
    require(guard.get("worker_id") == "NEXT100-065D-CANONICAL-CONVERGENCE-GUARD", "convergence guard worker drift")
    require(guard.get("local_free_only") is True, "convergence guard LOCAL_FREE weakened")

    registry = guard.get("canonical_registry_v3", {})
    require(registry.get("pr") == 538, "canonical registry PR drift")
    require(
        registry.get("path") == "configs/data/next100_063_terminal_source_registry_v3.json",
        "canonical registry path drift",
    )
    require(
        registry.get("schema_version") == "12-6.next100-063-terminal-source-registry.v3",
        "canonical registry schema drift",
    )
    require(
        registry.get("registry_identity_sha256")
        == "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c",
        "canonical registry identity drift",
    )
    require(
        registry.get("dedup_parent_head_sha") == "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13",
        "canonical registry dedup-parent head drift",
    )
    require(registry.get("dedup_parent_workflow_run") == 32999969398, "canonical registry dedup-parent run drift")
    require(registry.get("numeric_training_capacity_bytes") == 357530, "canonical registry capacity drift")
    require(
        registry.get("by_stratum") == {"uk": 100856, "en": 150643, "code": 106031},
        "canonical registry stratum vector drift",
    )
    require(
        registry.get("family_counts") == {"uk": 4, "en": 4, "code": 5},
        "canonical registry family vector drift",
    )
    require(registry.get("independent_family_count") == 13, "canonical registry family total drift")
    require(
        registry.get("authorized_balanced_no_replay_loss_positions") == 0,
        "canonical registry illegally authorizes loss positions",
    )

    reconciliation = guard.get("v5_to_v3_reconciliation", {})
    require(
        reconciliation.get("v5_fixed_without_cpython_accepted_chunks") == 320632,
        "V5 reconciliation capacity drift",
    )
    require(reconciliation.get("numpy_exact_capacity_bytes") == 36898, "NumPy reconciliation capacity drift")
    require(
        reconciliation.get("sum_must_equal_canonical_v3_numeric_bytes") == 357530,
        "canonical reconciliation target drift",
    )
    require(
        int(reconciliation["v5_fixed_without_cpython_accepted_chunks"])
        + int(reconciliation["numpy_exact_capacity_bytes"])
        == int(reconciliation["sum_must_equal_canonical_v3_numeric_bytes"]),
        "V5 + NumPy does not reconcile to canonical V3",
    )

    cp = guard.get("post_v3_terminal_additions", {}).get("cpython_accepted_only", {})
    require(cp.get("pr") == 567, "CPython accepted-only adapter PR drift")
    require(
        cp.get("head_sha") == "8f0cbc16f9a920ca9ab3e3061b53fbfec8838d77",
        "CPython accepted-only adapter head drift",
    )
    require(cp.get("dedicated_workflow_run") == 33005689174, "CPython accepted-only run drift")
    require(cp.get("dedicated_workflow_conclusion") == "success", "CPython accepted-only run not success")
    require(cp.get("artifact_id") == 9620571005, "CPython accepted-only artifact id drift")
    require(
        cp.get("artifact_digest")
        == "sha256:5c04e12f1100fd4012efc1cf693f213d1d7c9ababee2a16367897377cde60379",
        "CPython accepted-only artifact digest drift",
    )
    require(cp.get("accepted_chunk_count") == 14, "CPython accepted chunk count drift")
    require(cp.get("rejected_chunk_count") == 2, "CPython rejected chunk count drift")
    require(cp.get("rejection_reasons") == {"pii_phone": 2}, "CPython rejection-reason drift")
    require(cp.get("exact_eligible_capacity_bytes") == 15540, "CPython exact eligible capacity drift")

    guard_vector = guard.get("expected_composed_vector_before_successor_global_dedup", {})
    require(guard_vector.get("source_object_count") == 31, "convergence-guard object count drift")
    require(
        guard_vector.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "convergence-guard family vector drift",
    )
    require(
        guard_vector.get("capacity_bytes")
        == {"uk": 100856, "en": 1838293, "code": 106031, "total": 2045180},
        "convergence-guard composed capacity vector drift",
    )
    require(guard_vector.get("planning_gap_bytes") == 17954820, "convergence-guard planning gap drift")
    require(
        guard_vector.get("authorized_balanced_no_replay_loss_positions") == 0,
        "convergence guard illegally authorizes loss positions",
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


def decode_contents_json(payload: dict[str, Any], label: str) -> dict[str, Any]:
    require(payload.get("encoding") == "base64", f"{label} contents encoding drift")
    content = payload.get("content")
    require(isinstance(content, str) and content, f"{label} contents missing")
    try:
        raw = base64.b64decode(content, validate=False)
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"NEXT100-065D FAIL: {label} contents decode failed: {exc}") from exc
    require(isinstance(value, dict), f"{label} JSON root must be object")
    return value


def validate_live_canonical_registry(guard: dict[str, Any]) -> None:
    expected = guard["canonical_registry_v3"]
    pr_number = int(expected["pr"])
    first_pr = github_get(f"pulls/{pr_number}")
    head_sha = first_pr.get("head", {}).get("sha")
    require(isinstance(head_sha, str) and head_sha, "canonical registry PR has no head SHA")

    payload = github_get(f"contents/{expected['path']}?ref={head_sha}")
    live = decode_contents_json(payload, "canonical registry V3")

    second_pr = github_get(f"pulls/{pr_number}")
    require(second_pr.get("head", {}).get("sha") == head_sha, "canonical registry PR moved during validation")
    require(live.get("schema_version") == expected["schema_version"], "live canonical registry schema drift")
    require(
        live.get("registry_identity_sha256") == expected["registry_identity_sha256"],
        "live canonical registry identity drift",
    )

    parent = live.get("dedup_parent", {})
    require(parent.get("head_sha") == expected["dedup_parent_head_sha"], "live canonical dedup-parent head drift")
    require(
        parent.get("dedicated_workflow_run") == expected["dedup_parent_workflow_run"],
        "live canonical dedup-parent run drift",
    )
    require(parent.get("dedicated_workflow_conclusion") == "success", "live canonical dedup-parent not success")
    validate_run(
        int(expected["dedup_parent_workflow_run"]),
        str(expected["dedup_parent_head_sha"]),
        "canonical V3 dedup parent",
    )

    inventory = live.get("pre_successor_global_dedup_inventory", {})
    require(
        inventory.get("candidate_numeric_training_capacity_bytes") == expected["numeric_training_capacity_bytes"],
        "live canonical numeric capacity drift",
    )
    by_stratum = inventory.get("by_stratum", {})
    actual_capacity = {
        str(name): int(values.get("numeric_training_capacity_bytes", -1))
        for name, values in by_stratum.items()
        if isinstance(values, dict)
    }
    actual_families = {
        str(name): int(values.get("family_count", -1))
        for name, values in by_stratum.items()
        if isinstance(values, dict)
    }
    require(actual_capacity == expected["by_stratum"], "live canonical stratum capacity drift")
    require(actual_families == expected["family_counts"], "live canonical stratum family drift")
    require(
        inventory.get("candidate_independent_family_count") == expected["independent_family_count"],
        "live canonical independent-family total drift",
    )
    downstream = live.get("downstream_gate_vector", {})
    require(
        downstream.get("authorized_balanced_no_replay_loss_positions")
        == expected["authorized_balanced_no_replay_loss_positions"],
        "live canonical registry exposure boundary drift",
    )


def validate_live(config: dict[str, Any], guard: dict[str, Any]) -> None:
    validate_live_canonical_registry(guard)

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

    cp = guard["post_v3_terminal_additions"]["cpython_accepted_only"]
    cp_pr = github_get(f"pulls/{cp['pr']}")
    require(cp_pr.get("head", {}).get("sha") == cp["head_sha"], "CPython accepted-only PR head moved")
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
