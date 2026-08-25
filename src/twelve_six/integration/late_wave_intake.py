"""Fail-closed validation for D01/W1 late-wave S0 intake snapshots."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_CLASSIFICATIONS = {
    "INCUMBENT",
    "GREEN",
    "RED",
    "QUEUED",
    "SUPERSEDED",
    "DUPLICATE",
    "COMPOSABLE",
    "HOLD",
}
_BLOCKING_CLASSIFICATIONS = {"RED", "QUEUED", "SUPERSEDED", "DUPLICATE", "HOLD"}


class LateWaveIntakeError(ValueError):
    """Raised when a late-wave intake snapshot violates fail-closed composition rules."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LateWaveIntakeError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LateWaveIntakeError(f"{field} must be an array")
    return value


def _full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _FULL_SHA.fullmatch(value) is None:
        raise LateWaveIntakeError(f"{field} must be a full lowercase 40-hex Git SHA")
    return value


def _workflow_evidence(
    raw: Any,
    field: str,
) -> dict[str, Mapping[str, Any]]:
    workflows = _mapping(raw, field)
    validated: dict[str, Mapping[str, Any]] = {}
    for name, raw_workflow in workflows.items():
        if not isinstance(name, str) or not name:
            raise LateWaveIntakeError(f"{field} workflow names must be non-empty strings")
        workflow = _mapping(raw_workflow, f"{field}.{name}")
        run_id = workflow.get("run_id")
        if not isinstance(run_id, int) or run_id <= 0:
            raise LateWaveIntakeError(f"{field}.{name}.run_id must be a positive integer")
        state = workflow.get("state")
        if state not in {"queued", "in_progress", "completed"}:
            raise LateWaveIntakeError(f"{field}.{name}.state is unsupported")
        conclusion = workflow.get("conclusion")
        if state == "completed" and conclusion not in {
            "success",
            "failure",
            "cancelled",
            "timed_out",
            "action_required",
            "neutral",
            "skipped",
            "stale",
        }:
            raise LateWaveIntakeError(f"{field}.{name}.conclusion is unsupported")
        if state != "completed" and conclusion is not None:
            raise LateWaveIntakeError(
                f"{field}.{name}.conclusion must be null before terminal completion"
            )
        validated[name] = workflow
    return validated


def _all_success(workflows: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(workflows) and all(
        workflow.get("state") == "completed" and workflow.get("conclusion") == "success"
        for workflow in workflows.values()
    )


def _has_terminal_failure(workflows: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        workflow.get("state") == "completed" and workflow.get("conclusion") != "success"
        for workflow in workflows.values()
    )


def _has_pending(workflows: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(workflow.get("state") in {"queued", "in_progress"} for workflow in workflows.values())


def validate_late_wave_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a live-cutoff W1 intake registry without granting promotion authority."""

    if document.get("schema") != "12-6.s0-late-wave-intake.v2":
        raise LateWaveIntakeError("unsupported schema")
    if document.get("run_id") != "12-6-AI-SWARM-EXP-01":
        raise LateWaveIntakeError("unexpected swarm run_id")
    if document.get("repository") != "Oleksii-debug/12-6-ai.":
        raise LateWaveIntakeError("repository identity mismatch")
    if document.get("stage") != "S0":
        raise LateWaveIntakeError("registry must be scoped to S0")

    base = _mapping(document.get("base"), "base")
    if base.get("pr") != 89 or base.get("status") != "EXACT_GREEN":
        raise LateWaveIntakeError("base must be exact-green PR #89")
    base_sha = _full_sha(base.get("sha"), "base.sha")
    base_workflows = _workflow_evidence(base.get("required_workflows"), "base.required_workflows")
    required_base_workflows = {
        "CI",
        "D02 Real S0 Training",
        "D02 S0 Determinism Repeatability",
        "D04 Strict S0 Exact-Candidate Evaluation",
    }
    if set(base_workflows) != required_base_workflows or not _all_success(base_workflows):
        raise LateWaveIntakeError("base exact-green workflow set is incomplete or non-green")

    truth = _mapping(document.get("truth_boundary"), "truth_boundary")
    if truth.get("canonical_base") != "random_init_pretraining_only":
        raise LateWaveIntakeError("canonical Base truth boundary drift")
    for field in (
        "foreign_pretrained_weights_allowed",
        "instruction_or_alignment_base_work_allowed",
        "paid_compute_authorized",
        "promotion_eligible",
        "main_protected",
    ):
        if truth.get(field) is not False:
            raise LateWaveIntakeError(f"truth_boundary.{field} must remain false")
    _full_sha(truth.get("main_sha"), "truth_boundary.main_sha")
    audits = _mapping(truth.get("audits"), "truth_boundary.audits")
    if audits != {"AUDIT-A": "CHANGES_REQUIRED", "AUDIT-B": "CHANGES_REQUIRED"}:
        raise LateWaveIntakeError("historical independent audit authority must be preserved")

    collision_groups = _sequence(document.get("collision_groups"), "collision_groups")
    collision_owners: dict[str, int] = {}
    group_membership: dict[int, str] = {}
    for index, raw_group in enumerate(collision_groups):
        group = _mapping(raw_group, f"collision_groups[{index}]")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise LateWaveIntakeError("collision group id must be non-empty")
        members = _sequence(group.get("members"), f"collision group {group_id}.members")
        if len(members) < 2 or len(set(members)) != len(members):
            raise LateWaveIntakeError(f"collision group {group_id} must have unique competing PRs")
        owner = group.get("owner_pr")
        if owner is not None and owner not in members:
            raise LateWaveIntakeError(f"collision group {group_id} owner must be a member")
        if isinstance(owner, int):
            collision_owners[group_id] = owner
        for member in members:
            if not isinstance(member, int) or member <= 0:
                raise LateWaveIntakeError("collision group PR numbers must be positive integers")
            previous = group_membership.get(member)
            if previous is not None and previous != group_id:
                raise LateWaveIntakeError(f"PR #{member} appears in multiple collision groups")
            group_membership[member] = group_id

    items_raw = _sequence(document.get("items"), "items")
    items: dict[int, Mapping[str, Any]] = {}
    item_classes: dict[int, frozenset[str]] = {}
    by_classification: defaultdict[str, list[int]] = defaultdict(list)
    for index, raw_item in enumerate(items_raw):
        item = _mapping(raw_item, f"items[{index}]")
        pr = item.get("pr")
        if not isinstance(pr, int) or pr <= 0 or pr in items:
            raise LateWaveIntakeError("item PR numbers must be unique positive integers")
        _full_sha(item.get("head"), f"PR #{pr}.head")
        classifications_raw = _sequence(item.get("classifications"), f"PR #{pr}.classifications")
        classifications = frozenset(classifications_raw)
        if not classifications or len(classifications) != len(classifications_raw):
            raise LateWaveIntakeError(f"PR #{pr} classifications must be unique and non-empty")
        if not classifications <= _ALLOWED_CLASSIFICATIONS:
            raise LateWaveIntakeError(f"PR #{pr} has unsupported classification")
        if "GREEN" in classifications and "RED" in classifications:
            raise LateWaveIntakeError(f"PR #{pr} cannot be both GREEN and RED")
        if "COMPOSABLE" in classifications and classifications & _BLOCKING_CLASSIFICATIONS:
            raise LateWaveIntakeError(f"PR #{pr} COMPOSABLE conflicts with blocking classification")
        required = item.get("required_for_next_composition")
        if required not in {True, False}:
            raise LateWaveIntakeError(f"PR #{pr}.required_for_next_composition must be boolean")

        raw_workflows = item.get("workflow_evidence", {})
        workflows = _workflow_evidence(raw_workflows, f"PR #{pr}.workflow_evidence")
        if classifications & {"GREEN", "COMPOSABLE"} and not _all_success(workflows):
            raise LateWaveIntakeError(f"PR #{pr} green/composable claim lacks terminal-success evidence")
        if "RED" in classifications and not _has_terminal_failure(workflows):
            raise LateWaveIntakeError(f"PR #{pr} RED claim lacks terminal failure evidence")
        if "QUEUED" in classifications and not _has_pending(workflows):
            raise LateWaveIntakeError(f"PR #{pr} QUEUED claim lacks queued/in-progress evidence")
        if required and "COMPOSABLE" not in classifications:
            raise LateWaveIntakeError(f"required PR #{pr} must be COMPOSABLE")
        if item.get("stage_scope") == "S1" and "HOLD" not in classifications:
            raise LateWaveIntakeError("S1 work must remain HOLD for S0 composition")
        if item.get("lane") == "D10" and "HOLD" not in classifications:
            raise LateWaveIntakeError("D10 governance work must remain HOLD for W1 S0 Product intake")

        items[pr] = item
        item_classes[pr] = classifications
        for classification in classifications:
            by_classification[classification].append(pr)

    for group_id, owner in collision_owners.items():
        if owner in item_classes and "INCUMBENT" not in item_classes[owner]:
            raise LateWaveIntakeError(f"collision owner PR #{owner} must be classified INCUMBENT")
        if owner not in group_membership:
            raise LateWaveIntakeError(f"collision owner for {group_id} has no membership")

    pruning = _mapping(document.get("pruning_decisions"), "pruning_decisions")
    duplicate_prs = tuple(_sequence(pruning.get("duplicate_closed_unmerged"), "duplicate_closed_unmerged"))
    superseded_prs = tuple(_sequence(pruning.get("superseded_closed_unmerged"), "superseded_closed_unmerged"))
    if len(set(duplicate_prs + superseded_prs)) != len(duplicate_prs) + len(superseded_prs):
        raise LateWaveIntakeError("pruning decisions must not classify one PR twice")
    for pr in duplicate_prs + superseded_prs:
        if not isinstance(pr, int) or pr <= 0:
            raise LateWaveIntakeError("pruning PR numbers must be positive integers")

    policy = _mapping(document.get("next_composition_policy"), "next_composition_policy")
    minimum = tuple(_sequence(policy.get("minimum_required"), "policy.minimum_required"))
    if minimum != (90, 91, 100):
        raise LateWaveIntakeError("minimum late-wave composition set drifted")
    if any(pr not in items for pr in minimum):
        raise LateWaveIntakeError("minimum composition PR missing from registry")
    if any("COMPOSABLE" not in item_classes[pr] for pr in minimum):
        raise LateWaveIntakeError("minimum composition PR is not currently COMPOSABLE")
    if policy.get("never_accept_queued_or_red") is not True:
        raise LateWaveIntakeError("queued/red intake must fail closed")
    if policy.get("preserve_real_git_ancestry") is not True:
        raise LateWaveIntakeError("real Git ancestry preservation is mandatory")
    if policy.get("rerun_full_exact_head_workflows_after_composition") is not True:
        raise LateWaveIntakeError("post-composition exact-head rerun is mandatory")
    if policy.get("request_both_independent_audits_on_final_exact_head") is not True:
        raise LateWaveIntakeError("both independent audit handoffs are mandatory")

    return {
        "base_sha": base_sha,
        "registered_prs": tuple(sorted(items)),
        "minimum_required": minimum,
        "composition_ready": True,
        "collision_group_owners": collision_owners,
        "green_prs": tuple(sorted(by_classification["GREEN"])),
        "red_prs": tuple(sorted(by_classification["RED"])),
        "queued_prs": tuple(sorted(by_classification["QUEUED"])),
        "composable_prs": tuple(sorted(by_classification["COMPOSABLE"])),
        "held_prs": tuple(sorted(by_classification["HOLD"])),
        "duplicate_closed_unmerged": tuple(sorted(duplicate_prs)),
        "superseded_closed_unmerged": tuple(sorted(superseded_prs)),
        "promotion_eligible": False,
    }


def verify_base_ancestry(document: Mapping[str, Any], repo_root: str | Path) -> dict[str, str]:
    """Require the registry checkout to descend from the exact-green PR #89 base."""

    facts = validate_late_wave_snapshot(document)
    root = Path(repo_root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _full_sha(head, "git HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", facts["base_sha"], head],
        cwd=root,
        check=True,
    )
    return {"head_sha": head, "base_sha": facts["base_sha"]}


def load_and_validate(path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the committed snapshot; optionally verify exact Git ancestry."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = validate_late_wave_snapshot(document)
    if repo_root is not None:
        facts["ancestry"] = verify_base_ancestry(document, repo_root)
    return facts
