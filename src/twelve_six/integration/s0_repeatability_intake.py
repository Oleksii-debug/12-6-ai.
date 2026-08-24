"""Fail-closed validation for D01 intake of exact S0 repeatability evidence."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_WORKFLOWS = {
    "CI",
    "D02 Real S0 Training",
    "D04 Strict S0 Exact-Candidate Evaluation",
    "D02 S0 Determinism Repeatability",
}


class RepeatabilityIntakeError(ValueError):
    """Raised when repeatability intake evidence is stale, unsafe, or inconsistent."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepeatabilityIntakeError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RepeatabilityIntakeError(f"{field} must be an array")
    return value


def _full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _FULL_SHA.fullmatch(value) is None:
        raise RepeatabilityIntakeError(f"{field} must be a full lowercase Git SHA")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RepeatabilityIntakeError(f"{field} must be a lowercase SHA-256")
    return value


def validate_repeatability_intake(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate static intake evidence without granting promotion authority."""

    if document.get("schema_version") != "12-6.s0-repeatability-intake.v1":
        raise RepeatabilityIntakeError("unsupported schema_version")
    if document.get("repository") != "Oleksii-debug/12-6-ai.":
        raise RepeatabilityIntakeError("repository identity mismatch")
    if document.get("stage") != "S0":
        raise RepeatabilityIntakeError("stage must be S0")
    if document.get("status") != "experimental":
        raise RepeatabilityIntakeError("status must remain experimental")
    if document.get("canonical_base") != "random_init_pretraining_only":
        raise RepeatabilityIntakeError("canonical Base identity mismatch")
    if document.get("composition_complete") is not True:
        raise RepeatabilityIntakeError("composition must remain complete")
    if document.get("repeatability_evidence_complete") is not True:
        raise RepeatabilityIntakeError("repeatability evidence must be complete")
    if document.get("promotion_eligible") is not False:
        raise RepeatabilityIntakeError("repeatability intake cannot self-promote")

    authority = _mapping(document.get("authority_snapshot"), "authority_snapshot")
    parent_sha = _full_sha(authority.get("parent_candidate_sha"), "authority.parent_candidate_sha")
    source_sha = _full_sha(
        authority.get("repeatability_source_sha"), "authority.repeatability_source_sha"
    )
    if authority.get("parent_candidate_pr") != 88 or authority.get("repeatability_pr") != 89:
        raise RepeatabilityIntakeError("unexpected PR authority chain")
    if authority.get("main_protected") is not False:
        raise RepeatabilityIntakeError("snapshot must not invent main protection")

    successor = _mapping(document.get("accepted_successor"), "accepted_successor")
    if successor.get("lane") != "D02" or successor.get("pr_number") != 89:
        raise RepeatabilityIntakeError("accepted successor must be D02 PR #89")
    if _full_sha(successor.get("source_sha"), "accepted_successor.source_sha") != source_sha:
        raise RepeatabilityIntakeError("repeatability source SHA drift")
    if _full_sha(successor.get("direct_parent_sha"), "accepted_successor.direct_parent_sha") != parent_sha:
        raise RepeatabilityIntakeError("parent candidate SHA drift")
    if successor.get("changed_path_overlap_with_parent_product") != 0:
        raise RepeatabilityIntakeError("repeatability intake must remain path-disjoint")
    paths = _sequence(successor.get("changed_paths"), "accepted_successor.changed_paths")
    if len(paths) != 8 or len(set(paths)) != 8:
        raise RepeatabilityIntakeError("PR #89 changed-path inventory must contain 8 unique paths")

    workflows = _sequence(successor.get("workflows"), "accepted_successor.workflows")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(workflows):
        workflow = _mapping(raw, f"workflows[{index}]")
        name = workflow.get("name")
        if not isinstance(name, str) or name in by_name:
            raise RepeatabilityIntakeError("workflow names must be unique strings")
        if not isinstance(workflow.get("run_id"), int) or workflow["run_id"] <= 0:
            raise RepeatabilityIntakeError(f"workflow {name} requires a positive run_id")
        if workflow.get("conclusion") != "success":
            raise RepeatabilityIntakeError(f"workflow {name} is not exact-head success")
        by_name[name] = workflow
    if set(by_name) != _EXPECTED_WORKFLOWS:
        raise RepeatabilityIntakeError("required exact-head workflow set is incomplete")

    artifact = _mapping(successor.get("artifact"), "accepted_successor.artifact")
    if not isinstance(artifact.get("id"), int) or artifact["id"] <= 0:
        raise RepeatabilityIntakeError("artifact id must be positive")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RepeatabilityIntakeError("artifact digest must be sha256-prefixed")
    _sha256(digest.removeprefix("sha256:"), "artifact.digest")

    evidence = _mapping(successor.get("evidence"), "accepted_successor.evidence")
    if evidence.get("status") != "PASS":
        raise RepeatabilityIntakeError("repeatability evidence status must be PASS")
    if _full_sha(evidence.get("source_sha"), "evidence.source_sha") != source_sha:
        raise RepeatabilityIntakeError("evidence source SHA drift")
    for field in (
        "evidence_sha256",
        "identity_sha256",
        "locked_environment_evidence_sha256",
        "same_seed_initial_model_sha256",
        "same_seed_final_model_sha256",
        "same_seed_final_trainer_state_sha256",
        "same_seed_step_trace_sha256",
        "different_seed_initial_model_sha256",
        "different_seed_final_model_sha256",
        "different_seed_step_trace_sha256",
    ):
        _sha256(evidence.get(field), f"evidence.{field}")
    if evidence.get("same_seed_exact_equivalence") is not True:
        raise RepeatabilityIntakeError("same-seed exact equivalence is not proven")
    if evidence.get("different_seed_initialization_diverges") is not True:
        raise RepeatabilityIntakeError("different-seed initialization causality is not proven")
    if evidence.get("different_seed_training_diverges") is not True:
        raise RepeatabilityIntakeError("different-seed trajectory causality is not proven")
    if evidence.get("validation_optimized_tokens") != 0:
        raise RepeatabilityIntakeError("validation data must never be optimized")
    if evidence.get("changed_parameter_elements") != evidence.get("trainable_parameter_elements"):
        raise RepeatabilityIntakeError("all trainable parameter elements must change in proof run")
    if not evidence.get("final_train_loss", evidence.get("same_seed_final_train_loss")) < evidence.get(
        "initial_train_loss", evidence.get("same_seed_initial_train_loss")
    ):
        raise RepeatabilityIntakeError("training loss must decrease")
    if evidence.get("same_seed_final_validation_loss") >= evidence.get(
        "same_seed_initial_validation_loss"
    ):
        raise RepeatabilityIntakeError("held-out validation loss must decrease")

    claims = _mapping(document.get("claims"), "claims")
    forbidden_true = {
        "candidate_or_stable_promotion",
        "cross_hardware_bitwise_reproducibility",
        "gpu_reproducibility",
        "distributed_reproducibility",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
    }
    if any(claims.get(name) is not False for name in forbidden_true):
        raise RepeatabilityIntakeError("unsafe or unsupported claim enabled")

    audits = _mapping(document.get("audits"), "audits")
    if set(audits) != {"AUDIT-A", "AUDIT-B"}:
        raise RepeatabilityIntakeError("both independent audit slots are required")
    for name in ("AUDIT-A", "AUDIT-B"):
        audit = _mapping(audits[name], name)
        if audit.get("verdict") != "CHANGES_REQUIRED":
            raise RepeatabilityIntakeError("current manifest must preserve historical audit verdict")

    blockers = _sequence(document.get("promotion_blockers"), "promotion_blockers")
    if len(blockers) < 2 or any(not isinstance(item, str) or not item.strip() for item in blockers):
        raise RepeatabilityIntakeError("promotion blockers must remain explicit")

    return {
        "parent_candidate_sha": parent_sha,
        "repeatability_source_sha": source_sha,
        "workflow_run_ids": {name: by_name[name]["run_id"] for name in sorted(by_name)},
        "evidence_sha256": evidence["evidence_sha256"],
        "artifact_id": artifact["id"],
        "promotion_eligible": False,
    }


def verify_repeatability_ancestry(
    document: Mapping[str, Any], repo_root: str | Path
) -> dict[str, str]:
    """Require both PR #88 and PR #89 exact sources to be ancestors of HEAD."""

    facts = validate_repeatability_intake(document)
    root = Path(repo_root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _full_sha(head, "git HEAD")
    for sha in (facts["parent_candidate_sha"], facts["repeatability_source_sha"]):
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, head], cwd=root, check=True)
    return {
        "head_sha": head,
        "parent_candidate_sha": facts["parent_candidate_sha"],
        "repeatability_source_sha": facts["repeatability_source_sha"],
    }


def load_and_validate(path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load a committed JSON intake document and validate it."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = validate_repeatability_intake(payload)
    if repo_root is not None:
        facts["ancestry"] = verify_repeatability_ancestry(payload, repo_root)
    return facts
