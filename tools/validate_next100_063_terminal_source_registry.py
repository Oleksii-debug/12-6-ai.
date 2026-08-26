#!/usr/bin/env python3
"""Validate NEXT100-063 terminal source-authority convergence.

This gate deliberately distinguishes source-authority convergence from a final
training corpus. Included sources require an exact current PR head and a
successful dedicated workflow. Failed/nonterminal candidates receive zero
capacity. Global dedup/decontamination/packing remain successor work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_063_terminal_source_registry_v1.json")
SCHEMA = "12-6.next100-063-terminal-source-registry.v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
STRATA = ("uk", "en", "code")


class RegistryError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def identity(config: dict[str, Any]) -> str:
    core = dict(config)
    core.pop("registry_identity_sha256", None)
    return hashlib.sha256(canonical_bytes(core)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def load() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "registry must be a JSON object")
    return value


def validate_static(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == SCHEMA, "schema drift")
    require(config.get("local_free_only") is True, "LOCAL_FREE boundary drift")
    require(config.get("model_training_executed") is False, "training must remain false")
    digest = config.get("registry_identity_sha256")
    require(isinstance(digest, str) and SHA64.fullmatch(digest) is not None, "invalid identity")
    require(digest == identity(config), "registry self-identity mismatch")

    policy = config.get("authority_policy")
    require(isinstance(policy, dict), "authority policy missing")
    require(policy.get("included_only_if_current_exact_head_dedicated_workflow_success") is True,
            "exact-head success policy must remain enabled")
    require(policy.get("pr_text_is_not_terminal_authority") is True,
            "PR prose cannot become authority")
    require(policy.get("failed_queued_in_progress_retest_capacity_credit") == 0,
            "nonterminal capacity must remain zero")

    included = config.get("included_authorities")
    require(isinstance(included, list) and included, "included authorities missing")
    workers: set[str] = set()
    run_ids: set[int] = set()
    family_ids: set[str] = set()
    by_stratum = {
        key: {"source_objects": 0, "normalized_bytes": 0, "family_ids": set()}
        for key in STRATA
    }

    for authority in included:
        require(isinstance(authority, dict), "authority must be a mapping")
        worker = authority.get("worker")
        require(isinstance(worker, str) and worker and worker not in workers,
                f"duplicate/invalid included worker: {worker!r}")
        workers.add(worker)
        require(authority.get("status") == "SUCCESS", f"{worker}: included status must be SUCCESS")
        require(isinstance(authority.get("pr"), int) and authority["pr"] > 0,
                f"{worker}: PR required")
        require(isinstance(authority.get("head_sha"), str) and SHA40.fullmatch(authority["head_sha"]),
                f"{worker}: exact head SHA required")
        run_id = authority.get("dedicated_workflow_run")
        require(isinstance(run_id, int) and run_id > 0 and run_id not in run_ids,
                f"{worker}: unique dedicated workflow run required")
        run_ids.add(run_id)
        contributions = authority.get("contributions")
        require(isinstance(contributions, dict) and contributions,
                f"{worker}: contributions missing")
        for stratum, record in contributions.items():
            require(stratum in STRATA, f"{worker}: unsupported stratum {stratum}")
            require(isinstance(record, dict), f"{worker}/{stratum}: contribution malformed")
            objects = record.get("source_objects")
            capacity = record.get("normalized_bytes")
            families = record.get("family_ids")
            require(isinstance(objects, int) and objects > 0,
                    f"{worker}/{stratum}: positive source count required")
            require(isinstance(capacity, int) and capacity > 0,
                    f"{worker}/{stratum}: positive normalized bytes required")
            require(isinstance(families, list) and families,
                    f"{worker}/{stratum}: family ids required")
            require(len(families) == len(set(families)),
                    f"{worker}/{stratum}: duplicate local family ids")
            for family in families:
                require(isinstance(family, str) and family,
                        f"{worker}/{stratum}: invalid family id")
                require(family not in family_ids,
                        f"cross-authority family collision requires explicit canonicalization: {family}")
                family_ids.add(family)
                by_stratum[stratum]["family_ids"].add(family)
            by_stratum[stratum]["source_objects"] += objects
            by_stratum[stratum]["normalized_bytes"] += capacity

    excluded = config.get("excluded_nonterminal_or_failed")
    require(isinstance(excluded, list), "excluded authority list missing")
    for authority in excluded:
        worker = authority.get("worker")
        require(authority.get("status") != "SUCCESS", f"{worker}: successful authority cannot be excluded")
        require(authority.get("capacity_credit_bytes") == 0,
                f"{worker}: excluded capacity must be zero")
        require(isinstance(authority.get("head_sha"), str) and SHA40.fullmatch(authority["head_sha"]),
                f"{worker}: excluded exact head required")

    aggregate = config.get("aggregate")
    require(isinstance(aggregate, dict), "aggregate missing")
    expected_objects = sum(v["source_objects"] for v in by_stratum.values())
    expected_bytes = sum(v["normalized_bytes"] for v in by_stratum.values())
    require(aggregate.get("source_objects") == expected_objects, "aggregate source-object drift")
    require(aggregate.get("source_level_normalized_bytes") == expected_bytes,
            "aggregate source-capacity drift")
    require(aggregate.get("independent_source_families") == len(family_ids),
            "aggregate family-count drift")
    aggregate_strata = aggregate.get("by_stratum")
    require(isinstance(aggregate_strata, dict) and set(aggregate_strata) == set(STRATA),
            "stratum aggregate inventory drift")
    for stratum in STRATA:
        row = aggregate_strata[stratum]
        actual = by_stratum[stratum]
        require(row.get("source_objects") == actual["source_objects"],
                f"{stratum}: source-object aggregate drift")
        require(row.get("normalized_bytes") == actual["normalized_bytes"],
                f"{stratum}: capacity aggregate drift")
        require(row.get("independent_source_families") == len(actual["family_ids"]),
                f"{stratum}: family aggregate drift")

    gate = config.get("source_diversity_gate")
    require(isinstance(gate, dict), "source diversity gate missing")
    minimum = gate.get("minimum_independent_families_per_stratum")
    require(isinstance(minimum, int) and minimum >= 2, "invalid diversity minimum")
    require(all(len(by_stratum[s]["family_ids"]) >= minimum for s in STRATA),
            "source-authority diversity gate is not satisfied")
    require(gate.get("status") == "PASS_SOURCE_AUTHORITY_LEVEL",
            "diversity status must remain source-authority scoped")
    require(gate.get("does_not_authorize_corpus_release") is True,
            "source diversity cannot authorize corpus release")

    truth = config.get("truth_boundary")
    require(isinstance(truth, dict), "truth boundary missing")
    require(truth.get("canonical_base_training_eligibility") is False,
            "source registry cannot grant Base eligibility")
    require(truth.get("source_level_capacity_only") is True,
            "capacity must remain source-level")
    require(truth.get("global_cross_source_dedup_complete_for_this_vector") is False,
            "global dedup cannot be pre-claimed")
    require(truth.get("global_decontamination_complete_for_this_vector") is False,
            "decontamination cannot be pre-claimed")
    require(truth.get("selection_reservations_applied") is False,
            "selection reservation cannot be pre-claimed")
    require(truth.get("final_split_and_packing_materialized") is False,
            "packing cannot be pre-claimed")
    require(truth.get("unique_loss_positions_materialized") is False,
            "loss positions cannot be pre-claimed")
    require(truth.get("training_authorized_exposure") == 0,
            "training exposure must remain zero")
    require(truth.get("research_corpus_v1_terminal") is False,
            "Research Corpus V1 cannot be pre-claimed")


def github_json(path: str) -> dict[str, Any]:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(repo and token), "GITHUB_REPOSITORY/GITHUB_TOKEN required for --github-live")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-next100-063-validator",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.load(response)
    require(isinstance(value, dict), f"GitHub response for {path} must be an object")
    return value


def validate_live(config: dict[str, Any]) -> None:
    base = config["base_authority"]
    base_run = github_json(f"actions/runs/{base['dedicated_workflow_run']}")
    require(base_run.get("head_sha") == base["head_sha"], "base authority run/head drift")
    require(base_run.get("conclusion") == "success", "base authority is no longer successful")

    for authority in config["included_authorities"]:
        worker = authority["worker"]
        pr = github_json(f"pulls/{authority['pr']}")
        require(pr.get("head", {}).get("sha") == authority["head_sha"],
                f"{worker}: current PR head moved; refresh registry")
        run = github_json(f"actions/runs/{authority['dedicated_workflow_run']}")
        require(run.get("head_sha") == authority["head_sha"],
                f"{worker}: dedicated run is not exact-head")
        require(run.get("status") == "completed" and run.get("conclusion") == "success",
                f"{worker}: dedicated workflow is not completed success")

    for authority in config["excluded_nonterminal_or_failed"]:
        worker = authority["worker"]
        pr = github_json(f"pulls/{authority['pr']}")
        require(pr.get("head", {}).get("sha") == authority["head_sha"],
                f"{worker}: excluded PR head moved; refresh registry")
        run = github_json(f"actions/runs/{authority['dedicated_workflow_run']}")
        require(run.get("head_sha") == authority["head_sha"],
                f"{worker}: excluded dedicated run is not exact-head")
        require(run.get("conclusion", "").upper() == authority["status"],
                f"{worker}: exclusion status changed; refresh registry")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    config = load()
    validate_static(config)
    if args.github_live:
        validate_live(config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "registry_identity_sha256": config["registry_identity_sha256"],
                "source_objects": config["aggregate"]["source_objects"],
                "independent_source_families": config["aggregate"]["independent_source_families"],
                "source_level_normalized_bytes": config["aggregate"]["source_level_normalized_bytes"],
                "training_authorized_exposure": config["truth_boundary"]["training_authorized_exposure"],
                "github_live_checked": args.github_live,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
