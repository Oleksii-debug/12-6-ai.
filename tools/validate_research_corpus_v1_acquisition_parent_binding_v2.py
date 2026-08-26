#!/usr/bin/env python3
"""Hard-bind DATA-BULK-ACQ-V1 to exact NEXT100-063 repository bytes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BINDING_PATH = ROOT / "configs/data/research_corpus_v1_acquisition_parent_binding_v2.json"
PLAN_PATH = ROOT / "configs/data/research_corpus_v1_acquisition_plan.json"
PARENT_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"
STRATA = ("uk", "en", "code")
SAFE_RESULT = "SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION"


class ParentBindingError(ValueError):
    """Raised when acquisition planning is not bound to its exact parent bytes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentBindingError(message)


def load_json_bytes(data: bytes, field: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    require(isinstance(value, dict), f"{field}: expected JSON object")
    return value


def canonical_sha(data: dict[str, Any], identity_key: str) -> str:
    body = dict(data)
    body.pop(identity_key, None)
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def vector(value: Any, field: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{field}: expected object")
    require(set(value) == {*STRATA, "total"}, f"{field}: invalid keys")
    out: dict[str, int] = {}
    for key in (*STRATA, "total"):
        item = value[key]
        require(type(item) is int and item >= 0, f"{field}.{key}: invalid integer")
        out[key] = item
    require(out["total"] == sum(out[key] for key in STRATA), f"{field}: total drift")
    return out


def validate(binding: dict[str, Any], plan_bytes: bytes, parent_bytes: bytes) -> dict[str, Any]:
    require(
        binding.get("schema_version")
        == "12-6.research-corpus-v1-acquisition-parent-binding.v2",
        "binding schema drift",
    )
    require(
        binding.get("worker_id") == "DATA-BULK-ACQ-V1-R1-PROVENANCE-HARDENING",
        "binding worker drift",
    )
    require(binding.get("execution_class") == "LOCAL_FREE", "must remain LOCAL_FREE")
    require(
        canonical_sha(binding, "binding_identity_sha256")
        == binding.get("binding_identity_sha256"),
        "binding identity mismatch",
    )

    base = binding.get("base_acquisition")
    parent_binding = binding.get("parent_convergence")
    require(isinstance(base, dict), "base_acquisition is required")
    require(isinstance(parent_binding, dict), "parent_convergence is required")
    require(
        git_blob_sha1(plan_bytes) == base.get("plan_blob_sha1"),
        "acquisition plan blob drift",
    )
    require(
        git_blob_sha1(parent_bytes) == parent_binding.get("config_blob_sha1"),
        "parent config blob drift",
    )

    plan = load_json_bytes(plan_bytes, "acquisition plan")
    parent = load_json_bytes(parent_bytes, "parent convergence")
    require(plan.get("schema_version") == "12-6.research-corpus-v1-acquisition.v1", "plan schema drift")
    require(plan.get("worker_id") == "DATA-BULK-ACQ-V1", "plan worker drift")
    require(
        parent.get("schema_version") == "12-6.next100-063-source-registry-convergence.v1",
        "parent schema drift",
    )
    require(
        parent.get("worker_id") == parent_binding.get("worker_id"),
        "parent worker drift",
    )
    require(
        parent.get("claim_boundary", {}).get("safe_result") == SAFE_RESULT,
        "parent safe-result drift",
    )

    plan_base = plan.get("base_authority")
    require(isinstance(plan_base, dict), "plan base_authority is required")
    require(plan_base.get("head_sha") == parent_binding.get("head_sha"), "plan parent head drift")
    require(
        plan_base.get("config_path") == parent_binding.get("config_path"),
        "plan parent path drift",
    )
    require(
        plan_base.get("config_blob_sha1") == parent_binding.get("config_blob_sha1"),
        "plan parent blob declaration drift",
    )
    require(plan_base.get("safe_result") == SAFE_RESULT, "plan safe-result drift")

    parent_vector = vector(
        parent.get("converged_pre_successor_dedup_vector", {}).get("numeric_capacity_bytes"),
        "parent candidate vector",
    )
    plan_vector = vector(plan.get("credited_pre_successor_dedup_bytes"), "plan candidate vector")
    expected_vector = vector(binding.get("expected_candidate_vector"), "expected candidate vector")
    require(plan_vector == parent_vector == expected_vector, "candidate vector provenance drift")

    parent_families = vector(
        parent.get("converged_pre_successor_dedup_vector", {}).get(
            "independent_family_counts"
        ),
        "parent family vector",
    )
    expected_families = vector(binding.get("expected_family_vector"), "expected family vector")
    require(parent_families == expected_families, "family vector provenance drift")

    parent_gap = vector(
        parent.get("acquisition_plan_bytes", {}).get("remaining_gap"),
        "parent remaining gap",
    )
    plan_gap = vector(plan.get("remaining_gap_bytes"), "plan remaining gap")
    expected_gap = vector(binding.get("expected_gap_vector"), "expected remaining gap")
    require(plan_gap == parent_gap == expected_gap, "remaining-gap provenance drift")

    policy = plan.get("planning_policy")
    require(isinstance(policy, dict), "planning_policy is required")
    require(policy.get("survival_floor_is_evidence") is False, "survival floor is not evidence")
    floor = policy.get("planning_survival_floor")
    require(type(floor) in {int, float} and math.isclose(float(floor), 0.6), "survival floor drift")

    expected_buffer = {
        key: math.ceil(expected_gap[key] / float(floor)) for key in STRATA
    }
    expected_buffer["total"] = sum(expected_buffer.values())
    plan_buffer = vector(plan.get("buffered_gross_required_bytes"), "plan buffered gross")
    bound_buffer = vector(binding.get("expected_buffered_gross"), "bound buffered gross")
    require(plan_buffer == expected_buffer == bound_buffer, "buffered-gross provenance drift")

    workflow = parent_binding.get("observed_exact_head_workflow")
    require(isinstance(workflow, dict), "observed parent workflow is required")
    require(workflow.get("run_id") == 33006168870, "parent workflow run drift")
    require(workflow.get("status") == "queued", "parent workflow cutoff status drift")
    require(workflow.get("conclusion") is None, "queued workflow cannot have a conclusion")
    require(
        parent_binding.get("terminal_for_capacity_authority") is False,
        "nonterminal parent cannot be promoted",
    )

    claims = binding.get("claim_boundary")
    require(isinstance(claims, dict), "claim_boundary is required")
    require(claims.get("planning_binding_only") is True, "binding must remain planning-only")
    for field in (
        "post_dedup_capacity_claimed",
        "corpus_identity_created",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
    ):
        require(claims.get(field) is False, f"unsafe claim: {field}")
    require(claims.get("training_exposure_loss_positions") == 0, "training exposure must be zero")

    return {
        "status": "PASS_EXACT_PARENT_BINDING_PLAN_NONTERMINAL",
        "binding_identity_sha256": binding["binding_identity_sha256"],
        "plan_blob_sha1": base["plan_blob_sha1"],
        "parent_config_blob_sha1": parent_binding["config_blob_sha1"],
        "candidate_vector": expected_vector,
        "family_vector": expected_families,
        "remaining_gap": expected_gap,
        "buffered_gross": bound_buffer,
        "training_authorized": False,
    }


def main() -> None:
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    report = validate(binding, PLAN_PATH.read_bytes(), PARENT_PATH.read_bytes())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
