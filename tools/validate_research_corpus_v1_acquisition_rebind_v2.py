#!/usr/bin/env python3
"""Validate DATA-BULK-ACQ-V1-R1 against exact parent repository bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REBIND_PATH = ROOT / "configs/data/research_corpus_v1_acquisition_rebind_v2.json"
PARENT_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"

EXPECTED_PARENT_HEAD = "5356d60c8c8af46d6fc34debfd3cb36731045338"
EXPECTED_PARENT_BLOB = "d5b640b386219290f69d02a7f2e30a338c883009"
EXPECTED_SAFE_RESULT = "SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION"
STRATA = ("uk", "en", "code")


class RebindError(ValueError):
    """Raised when the acquisition rebind cannot be proven from exact bytes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RebindError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def canonical_sha(data: dict[str, Any], identity_key: str) -> str:
    body = dict(data)
    body.pop(identity_key, None)
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def byte_vector(value: Any, field: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{field}: expected object")
    require(set(value) == {*STRATA, "total"}, f"{field}: invalid keys")
    out: dict[str, int] = {}
    for key in (*STRATA, "total"):
        item = value[key]
        require(type(item) is int and item >= 0, f"{field}.{key}: invalid integer")
        out[key] = item
    require(out["total"] == sum(out[key] for key in STRATA), f"{field}: total drift")
    return out


def ceil_div(numerator: int, denominator: int) -> int:
    require(numerator >= 0 and denominator > 0, "ceil_div domain error")
    return (numerator + denominator - 1) // denominator


def validate(rebind: dict[str, Any], parent_bytes: bytes) -> dict[str, Any]:
    require(
        rebind.get("schema_version") == "12-6.research-corpus-v1-acquisition-rebind.v2",
        "unexpected rebind schema",
    )
    require(
        rebind.get("worker_id") == "DATA-BULK-ACQ-V1-R1-PROVENANCE-REBIND",
        "unexpected worker id",
    )
    require(rebind.get("execution_class") == "LOCAL_FREE", "must remain LOCAL_FREE")
    require(
        canonical_sha(rebind, "rebind_identity_sha256")
        == rebind.get("rebind_identity_sha256"),
        "rebind identity mismatch",
    )

    parent_binding = rebind.get("parent_convergence")
    require(isinstance(parent_binding, dict), "parent_convergence is required")
    require(parent_binding.get("head_sha") == EXPECTED_PARENT_HEAD, "parent head drift")
    require(
        parent_binding.get("config_blob_sha1") == EXPECTED_PARENT_BLOB,
        "declared parent blob drift",
    )
    require(git_blob_sha1(parent_bytes) == EXPECTED_PARENT_BLOB, "repository parent blob drift")
    require(parent_binding.get("safe_result") == EXPECTED_SAFE_RESULT, "safe-result drift")

    parent = json.loads(parent_bytes.decode("utf-8"))
    require(
        parent.get("schema_version") == "12-6.next100-063-source-registry-convergence.v1",
        "parent schema drift",
    )
    require(
        parent.get("worker_id") == "NEXT100-063-SOURCE-REGISTRY-CONVERGENCE",
        "parent worker drift",
    )
    require(
        parent.get("claim_boundary", {}).get("safe_result") == EXPECTED_SAFE_RESULT,
        "parent safe-result drift",
    )

    parent_vector = byte_vector(
        parent.get("converged_pre_successor_dedup_vector", {}).get(
            "numeric_capacity_bytes"
        ),
        "parent numeric capacity",
    )
    declared_vector = byte_vector(
        rebind.get("candidate_pre_successor_dedup_bytes"),
        "candidate_pre_successor_dedup_bytes",
    )
    require(declared_vector == parent_vector, "candidate byte vector is stale")

    parent_families = byte_vector(
        parent.get("converged_pre_successor_dedup_vector", {}).get(
            "independent_family_counts"
        ),
        "parent family counts",
    )
    declared_families = byte_vector(
        rebind.get("candidate_independent_family_counts"),
        "candidate_independent_family_counts",
    )
    require(declared_families == parent_families, "candidate family vector is stale")

    targets = byte_vector(rebind.get("frozen_targets_bytes"), "frozen_targets_bytes")
    gaps = byte_vector(
        rebind.get("candidate_remaining_gap_bytes"), "candidate_remaining_gap_bytes"
    )
    expected_gaps = {key: targets[key] - declared_vector[key] for key in STRATA}
    require(all(value >= 0 for value in expected_gaps.values()), "candidate exceeds target")
    expected_gaps["total"] = sum(expected_gaps.values())
    require(gaps == expected_gaps, "candidate gap arithmetic drift")

    floor = rebind.get("planning_survival_floor")
    require(isinstance(floor, dict), "planning_survival_floor is required")
    numerator = floor.get("numerator")
    denominator = floor.get("denominator")
    require((numerator, denominator) == (3, 5), "planning survival floor drift")
    require(floor.get("is_measured_evidence") is False, "planning floor is not evidence")

    buffered = byte_vector(
        rebind.get("buffered_gross_required_bytes"), "buffered_gross_required_bytes"
    )
    expected_buffered = {
        key: ceil_div(gaps[key] * denominator, numerator) for key in STRATA
    }
    expected_buffered["total"] = sum(expected_buffered.values())
    require(buffered == expected_buffered, "buffered gross arithmetic drift")

    planned = byte_vector(
        rebind.get("existing_v1_planned_gross_bytes"), "existing_v1_planned_gross_bytes"
    )
    require(
        all(planned[key] >= buffered[key] for key in STRATA),
        "existing V1 gross plan no longer covers corrected buffer",
    )
    headroom = byte_vector(rebind.get("planning_headroom_bytes"), "planning_headroom_bytes")
    expected_headroom = {key: planned[key] - buffered[key] for key in STRATA}
    expected_headroom["total"] = sum(expected_headroom.values())
    require(headroom == expected_headroom, "planning headroom arithmetic drift")

    stale = rebind.get("stale_v1_delta")
    require(isinstance(stale, dict), "stale_v1_delta is required")
    require(stale.get("additional_candidate_en_bytes") == 6492, "EN rebind delta drift")
    require(
        stale.get("remaining_gap_total_reduction_bytes") == 6492,
        "gap rebind delta drift",
    )
    require(
        stale.get("buffered_gross_total_reduction_bytes") == 10820,
        "buffer rebind delta drift",
    )

    workflow = parent_binding.get("observed_exact_head_workflow")
    require(isinstance(workflow, dict), "observed parent workflow is required")
    require(workflow.get("run_id") == 33005956092, "observed workflow run drift")
    require(workflow.get("status") == "queued", "cutoff workflow status drift")
    require(workflow.get("conclusion") is None, "queued workflow cannot have conclusion")
    require(
        parent_binding.get("terminal_for_capacity_authority") is False,
        "nonterminal parent cannot be promoted",
    )

    activation = rebind.get("activation")
    require(isinstance(activation, dict), "activation is required")
    require(
        activation.get("source_capacity_authority_status")
        == "BLOCKED_PARENT_EXACT_HEAD_WORKFLOW_NONTERMINAL",
        "source capacity authority must remain blocked",
    )
    require(activation.get("global_dedup") == "REQUIRED", "global dedup is still required")
    require(activation.get("corpus_release") == "BLOCKED", "corpus release must block")
    require(
        activation.get("training_exposure_loss_positions") == 0,
        "training exposure must remain zero",
    )

    claims = rebind.get("claim_boundary")
    require(isinstance(claims, dict), "claim_boundary is required")
    require(all(value is False for value in claims.values()), "unsafe claim boundary")

    return {
        "status": "PASS_REBIND_PLANNING_ONLY_PARENT_NONTERMINAL",
        "rebind_identity_sha256": rebind["rebind_identity_sha256"],
        "parent_head_sha": parent_binding["head_sha"],
        "parent_config_blob_sha1": parent_binding["config_blob_sha1"],
        "candidate_pre_successor_dedup_bytes": declared_vector,
        "candidate_remaining_gap_bytes": gaps,
        "buffered_gross_required_bytes": buffered,
        "existing_v1_planned_gross_bytes": planned,
        "planning_headroom_bytes": headroom,
        "training_authorized": False,
    }


def main() -> None:
    rebind = load_json(REBIND_PATH)
    parent_bytes = PARENT_PATH.read_bytes()
    report = validate(rebind, parent_bytes)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
