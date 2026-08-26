#!/usr/bin/env python3
"""Fail-close NEXT100-065D against the exact current NEXT100-063 V4 authority."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping

CONFIG = Path("configs/data/next100_065d_registry_v4_guard_v1.json")
REPO = "Oleksii-debug/12-6-ai."
SCHEMA = "12-6.next100-065d-registry-v4-guard.v1"
WORKER_ID = "NEXT100-065D-REGISTRY-V4-GUARD"


class RegistryV4GuardError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryV4GuardError(message)


def mapping(value: Any, name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "guard root must be an object")
    return value


def _validate_boundaries(boundary: Mapping[str, Any]) -> None:
    for key in (
        "corpus_identity_claimed",
        "decontamination_pass_claimed",
        "tokenizer_fit_authorized",
        "training_authorized",
        "learned_20m_claimed",
        "paid_compute_authorized",
    ):
        require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def validate_static(data: Mapping[str, Any]) -> None:
    require(data.get("schema_version") == SCHEMA, "schema drift")
    require(data.get("worker_id") == WORKER_ID, "worker drift")
    require(data.get("local_free_only") is True, "LOCAL_FREE boundary weakened")

    registry = mapping(data.get("canonical_registry"), "canonical_registry")
    require(registry.get("pr") == 538, "canonical registry PR drift")
    require(
        registry.get("path") == "configs/data/next100_063_terminal_source_registry_v4.json",
        "canonical registry path must be V4",
    )
    require(
        registry.get("schema_version") == "12-6.next100-063-terminal-source-registry.v4",
        "canonical registry schema must be V4",
    )
    require(
        registry.get("worker_id") == "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V4",
        "canonical registry worker drift",
    )
    require(
        registry.get("registry_identity_sha256")
        == "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58",
        "canonical V4 registry identity drift",
    )
    require(
        registry.get("superseded_v3_registry_identity_sha256")
        == "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c",
        "V3 supersession identity drift",
    )
    require(registry.get("numeric_training_capacity_bytes") == 2_045_180, "V4 numeric capacity drift")
    require(registry.get("source_normalized_envelope_bytes") == 2_047_541, "V4 envelope drift")
    require(registry.get("uncredited_source_normalized_bytes") == 2_361, "V4 uncredited byte drift")
    require(
        registry.get("by_stratum")
        == {
            "uk": {"family_count": 4, "numeric_training_capacity_bytes": 100_856},
            "en": {"family_count": 5, "numeric_training_capacity_bytes": 1_838_293},
            "code": {"family_count": 5, "numeric_training_capacity_bytes": 106_031},
        },
        "V4 stratum vector drift",
    )
    require(registry.get("independent_family_count") == 14, "V4 family total drift")
    require(registry.get("authorized_balanced_no_replay_loss_positions") == 0, "V4 exposure must remain zero")
    head = registry.get("head_sha")
    blob = registry.get("git_blob_sha1")
    require(isinstance(head, str) and len(head) == 40, "canonical V4 head SHA invalid")
    require(isinstance(blob, str) and len(blob) == 40, "canonical V4 blob SHA invalid")

    authorities = mapping(data.get("required_embedded_authorities"), "required_embedded_authorities")
    cp = mapping(authorities.get("cpython_accepted_only"), "cpython authority")
    require(cp.get("pr") == 467, "CPython source PR drift")
    require(cp.get("workflow_run") == 33005689174, "CPython accepted-only workflow drift")
    require(cp.get("artifact_id") == 9620571005, "CPython artifact drift")
    require(cp.get("numeric_training_capacity_bytes") == 15_540, "CPython eligible capacity drift")
    require(cp.get("source_normalized_bytes") == 17_901, "CPython normalized envelope drift")
    require(cp.get("accepted_chunk_count") == 14, "CPython accepted count drift")
    require(cp.get("rejected_chunk_count") == 2, "CPython rejected count drift")
    require(cp.get("rejection_reasons") == {"pii_phone": 2}, "CPython rejection reason drift")

    pg = mapping(authorities.get("gutenberg"), "Gutenberg authority")
    require(pg.get("pr") == 470, "Gutenberg source PR drift")
    require(pg.get("terminal_seal_pr") == 627, "Gutenberg seal PR drift")
    require(pg.get("workflow_run") == 32998859164, "Gutenberg workflow drift")
    require(pg.get("artifact_id") == 9618402768, "Gutenberg artifact drift")
    require(pg.get("numeric_training_capacity_bytes") == 1_672_110, "Gutenberg capacity drift")
    require(pg.get("source_record_count") == 3, "Gutenberg record count drift")
    require(pg.get("family") == "en.project-gutenberg.public-domain-books", "Gutenberg family drift")
    require(pg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary weakened")

    reconciliation = mapping(data.get("v6_reconciliation"), "v6_reconciliation")
    expected = mapping(
        reconciliation.get("expected_pre_global_dedup_capacity_bytes"),
        "V6 expected capacity",
    )
    require(
        expected == {"uk": 100_856, "en": 1_838_293, "code": 106_031, "total": 2_045_180},
        "V6 expected capacity vector drift",
    )
    require(sum(expected[key] for key in ("uk", "en", "code")) == expected["total"], "V6 capacity arithmetic mismatch")
    require(
        reconciliation.get("must_equal_canonical_v4_numeric_training_capacity") is True,
        "V6/V4 equality gate weakened",
    )
    require(expected["total"] == registry["numeric_training_capacity_bytes"], "V6 does not reconcile to canonical V4")
    require(reconciliation.get("expected_source_object_count") == 31, "V6 object count drift")
    require(reconciliation.get("expected_source_family_counts") == {"uk": 4, "en": 5, "code": 5}, "V6 family vector drift")
    require(reconciliation.get("research_corpus_v1_acquisition_target_bytes") == 20_000_000, "acquisition target drift")
    require(reconciliation.get("planning_gap_bytes") == 17_954_820, "planning gap drift")
    require(reconciliation.get("authorized_unique_causal_loss_positions") == 0, "V6 exposure must remain zero")

    _validate_boundaries(mapping(data.get("claim_boundary"), "claim_boundary"))


def github_get(path: str) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GITHUB_TOKEN is required for --github-live")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-ai-next100-065d-v4-guard",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response not object: {path}")
    return value


def _decode_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    require(payload.get("encoding") == "base64", "registry contents encoding drift")
    encoded = payload.get("content")
    require(isinstance(encoded, str) and encoded, "registry contents missing")
    value = json.loads(base64.b64decode(encoded).decode("utf-8"))
    require(isinstance(value, dict), "live registry root must be object")
    return value


def _find_addition(live: Mapping[str, Any], *, pr: int) -> Mapping[str, Any]:
    additions = live.get("terminal_late_additions")
    require(isinstance(additions, list), "V4 terminal_late_additions missing")
    matches = [item for item in additions if isinstance(item, dict) and item.get("pr") == pr]
    require(len(matches) == 1, f"V4 must contain exactly one terminal addition for PR #{pr}")
    return matches[0]


def validate_live(data: Mapping[str, Any]) -> None:
    registry = mapping(data["canonical_registry"], "canonical_registry")
    pr = github_get(f"pulls/{registry['pr']}")
    live_head = pr.get("head", {}).get("sha")
    require(live_head == registry["head_sha"], "canonical registry PR moved; refresh V4 guard before promotion")

    payload = github_get(f"contents/{registry['path']}?ref={live_head}")
    require(payload.get("sha") == registry["git_blob_sha1"], "canonical V4 Git blob drift")
    live = _decode_json(payload)
    require(live.get("schema_version") == registry["schema_version"], "live V4 schema drift")
    require(live.get("worker_id") == registry["worker_id"], "live V4 worker drift")
    require(live.get("registry_identity_sha256") == registry["registry_identity_sha256"], "live V4 identity drift")
    require(
        mapping(live.get("supersedes"), "live V4 supersedes").get("v3_registry_identity_sha256")
        == registry["superseded_v3_registry_identity_sha256"],
        "live V4 does not supersede the expected V3 authority",
    )

    inventory = mapping(live.get("pre_successor_global_dedup_inventory"), "live V4 inventory")
    require(inventory.get("candidate_numeric_training_capacity_bytes") == registry["numeric_training_capacity_bytes"], "live V4 numeric capacity drift")
    require(inventory.get("candidate_source_normalized_envelope_bytes") == registry["source_normalized_envelope_bytes"], "live V4 envelope drift")
    require(inventory.get("uncredited_source_normalized_bytes") == registry["uncredited_source_normalized_bytes"], "live V4 uncredited byte drift")
    require(inventory.get("candidate_independent_family_count") == registry["independent_family_count"], "live V4 family total drift")

    actual_by_stratum = {
        name: {
            "family_count": values.get("family_count"),
            "numeric_training_capacity_bytes": values.get("numeric_training_capacity_bytes"),
        }
        for name, values in mapping(inventory.get("by_stratum"), "live V4 by_stratum").items()
        if isinstance(values, dict)
    }
    require(actual_by_stratum == registry["by_stratum"], "live V4 stratum vector drift")
    downstream = mapping(live.get("downstream_gate_vector"), "live V4 downstream gates")
    require(
        downstream.get("authorized_balanced_no_replay_loss_positions")
        == registry["authorized_balanced_no_replay_loss_positions"],
        "live V4 exposure boundary drift",
    )

    authorities = mapping(data["required_embedded_authorities"], "required embedded authorities")
    cp_expected = mapping(authorities["cpython_accepted_only"], "CPython expected")
    cp = _find_addition(live, pr=int(cp_expected["pr"]))
    require(cp.get("head") == cp_expected["source_head_sha"], "V4 CPython source head drift")
    require(cp.get("numeric_training_capacity_bytes") == cp_expected["numeric_training_capacity_bytes"], "V4 CPython eligible capacity drift")
    require(cp.get("source_normalized_bytes") == cp_expected["source_normalized_bytes"], "V4 CPython normalized envelope drift")
    require(cp.get("accepted_chunk_count") == cp_expected["accepted_chunk_count"], "V4 CPython accepted count drift")
    require(cp.get("rejected_chunk_count") == cp_expected["rejected_chunk_count"], "V4 CPython rejected count drift")
    cp_mat = mapping(cp.get("accepted_materialization"), "V4 CPython accepted materialization")
    require(cp_mat.get("head_sha") == cp_expected["accepted_materialization_head_sha"], "V4 CPython materialization head drift")
    require(cp_mat.get("workflow_run") == cp_expected["workflow_run"], "V4 CPython materialization run drift")
    require(cp_mat.get("artifact_id") == cp_expected["artifact_id"], "V4 CPython artifact id drift")
    require(cp_mat.get("artifact_digest") == cp_expected["artifact_digest"], "V4 CPython artifact digest drift")
    require(cp_mat.get("rejection_reasons") == cp_expected["rejection_reasons"], "V4 CPython rejection reasons drift")

    pg_expected = mapping(authorities["gutenberg"], "Gutenberg expected")
    pg = _find_addition(live, pr=int(pg_expected["pr"]))
    require(pg.get("head") == pg_expected["source_head_sha"], "V4 Gutenberg source head drift")
    require(pg.get("dedicated_workflow_run") == pg_expected["workflow_run"], "V4 Gutenberg run drift")
    require(pg.get("terminal_artifact_id") == pg_expected["artifact_id"], "V4 Gutenberg artifact id drift")
    require(pg.get("terminal_artifact_digest") == pg_expected["artifact_digest"], "V4 Gutenberg artifact digest drift")
    require(pg.get("authority_identity") == pg_expected["authority_identity_sha256"], "V4 Gutenberg authority identity drift")
    require(pg.get("numeric_training_capacity_bytes") == pg_expected["numeric_training_capacity_bytes"], "V4 Gutenberg capacity drift")
    require(pg.get("source_record_count") == pg_expected["source_record_count"], "V4 Gutenberg record count drift")
    require(pg.get("family") == pg_expected["family"], "V4 Gutenberg family drift")
    require(pg.get("evaluation") == pg_expected["evaluation"], "V4 Gutenberg evaluation boundary drift")
    seal = mapping(pg.get("terminal_seal"), "V4 Gutenberg terminal seal")
    require(seal.get("pr") == pg_expected["terminal_seal_pr"], "V4 Gutenberg seal PR drift")
    require(seal.get("head_sha") == pg_expected["terminal_seal_head_sha"], "V4 Gutenberg seal head drift")

    pr_after = github_get(f"pulls/{registry['pr']}")
    require(pr_after.get("head", {}).get("sha") == live_head, "canonical registry PR moved during validation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    data = load_config()
    validate_static(data)
    if args.github_live:
        validate_live(data)
    print("NEXT100-065D REGISTRY V4 GUARD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
