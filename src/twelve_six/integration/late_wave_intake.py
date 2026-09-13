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


def _workflow_evidence(raw: Any, field: str) -> dict[str, Mapping[str, Any]]:
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
    return any(
        workflow.get("state") in {"queued", "in_progress"}
        for workflow in workflows.values()
    )


def validate_late_wave_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a materialized W1 intake registry without granting promotion authority."""

    if document.get("schema") != "12-6.s0-late-wave-intake.v3":
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
    base_workflows = _workflow_evidence(
        base.get("required_workflows"), "base.required_workflows"
    )
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
            raise LateWaveIntakeError(
                f"collision group {group_id} must have unique competing PRs"
            )
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
        classifications_raw = _sequence(
            item.get("classifications"), f"PR #{pr}.classifications"
        )
        classifications = frozenset(classifications_raw)
        if not classifications or len(classifications) != len(classifications_raw):
            raise LateWaveIntakeError(
                f"PR #{pr} classifications must be unique and non-empty"
            )
        if not classifications <= _ALLOWED_CLASSIFICATIONS:
            raise LateWaveIntakeError(f"PR #{pr} has unsupported classification")
        if "GREEN" in classifications and "RED" in classifications:
            raise LateWaveIntakeError(f"PR #{pr} cannot be both GREEN and RED")
        if "COMPOSABLE" in classifications and classifications & _BLOCKING_CLASSIFICATIONS:
            raise LateWaveIntakeError(
                f"PR #{pr} COMPOSABLE conflicts with blocking classification"
            )
        required = item.get("required_for_next_composition")
        if required not in {True, False}:
            raise LateWaveIntakeError(
                f"PR #{pr}.required_for_next_composition must be boolean"
            )

        workflows = _workflow_evidence(
            item.get("workflow_evidence", {}), f"PR #{pr}.workflow_evidence"
        )
        if classifications & {"GREEN", "COMPOSABLE"} and not _all_success(workflows):
            raise LateWaveIntakeError(
                f"PR #{pr} green/composable claim lacks terminal-success evidence"
            )
        if "RED" in classifications and not _has_terminal_failure(workflows):
            raise LateWaveIntakeError(f"PR #{pr} RED claim lacks terminal failure evidence")
        if "QUEUED" in classifications and not _has_pending(workflows):
            raise LateWaveIntakeError(
                f"PR #{pr} QUEUED claim lacks queued/in-progress evidence"
            )
        if required and "COMPOSABLE" not in classifications:
            raise LateWaveIntakeError(f"required PR #{pr} must be COMPOSABLE")
        if item.get("stage_scope") == "S1" and "HOLD" not in classifications:
            raise LateWaveIntakeError("S1 work must remain HOLD for S0 composition")
        if item.get("lane") == "D10" and "HOLD" not in classifications:
            raise LateWaveIntakeError(
                "D10 governance work must remain HOLD for W1 S0 Product intake"
            )

        items[pr] = item
        item_classes[pr] = classifications
        for classification in classifications:
            by_classification[classification].append(pr)

    for group_id, owner in collision_owners.items():
        if owner in item_classes and "INCUMBENT" not in item_classes[owner]:
            raise LateWaveIntakeError(
                f"collision owner PR #{owner} must be classified INCUMBENT"
            )
        if owner not in group_membership:
            raise LateWaveIntakeError(f"collision owner for {group_id} has no membership")

    composition = _mapping(document.get("composition"), "composition")
    if composition.get("state") != "MATERIALIZED_REQUIRES_EXACT_HEAD_RERUN":
        raise LateWaveIntakeError("composition must remain pending exact-head validation")
    composition_commit = _full_sha(composition.get("commit"), "composition.commit")
    first_parent = _full_sha(composition.get("first_parent"), "composition.first_parent")
    if composition.get("changed_path_overlap") != 0:
        raise LateWaveIntakeError("composition must preserve zero changed-path overlap")
    if composition.get("history_destroyed") is not False:
        raise LateWaveIntakeError("composition must preserve source history")
    if composition.get("exact_head_validation_required") is not True:
        raise LateWaveIntakeError("composition must require fresh exact-head validation")

    integrated_raw = _sequence(
        composition.get("integrated_sources"), "composition.integrated_sources"
    )
    integrated_prs: list[int] = []
    integrated_shas: dict[int, str] = {}
    seen_paths: set[str] = set()
    for index, raw_source in enumerate(integrated_raw):
        source = _mapping(raw_source, f"composition.integrated_sources[{index}]")
        pr = source.get("pr")
        if not isinstance(pr, int) or pr <= 0 or pr in integrated_shas:
            raise LateWaveIntakeError("integrated PR numbers must be unique positive integers")
        sha = _full_sha(source.get("sha"), f"integrated PR #{pr}.sha")
        paths = _sequence(source.get("changed_paths"), f"integrated PR #{pr}.changed_paths")
        if not paths or len(set(paths)) != len(paths):
            raise LateWaveIntakeError(
                f"integrated PR #{pr} changed paths must be unique and non-empty"
            )
        for path in paths:
            if not isinstance(path, str) or not path or path in seen_paths:
                raise LateWaveIntakeError(
                    "integrated changed paths must be non-empty and globally disjoint"
                )
            seen_paths.add(path)
        if pr not in items or items[pr].get("head") != sha:
            raise LateWaveIntakeError(
                f"integrated PR #{pr} must match its registered exact head"
            )
        if "GREEN" not in item_classes[pr] or item_classes[pr] & _BLOCKING_CLASSIFICATIONS:
            raise LateWaveIntakeError(
                f"integrated PR #{pr} must remain terminal-green and unblocked"
            )
        if items[pr].get("required_for_next_composition") is not False:
            raise LateWaveIntakeError(
                f"integrated PR #{pr} cannot remain required for next composition"
            )
        if "COMPOSABLE" in item_classes[pr]:
            raise LateWaveIntakeError(
                f"integrated PR #{pr} must not remain classified COMPOSABLE"
            )
        integrated_prs.append(pr)
        integrated_shas[pr] = sha

    if tuple(integrated_prs) != (90, 91, 100):
        raise LateWaveIntakeError("materialized composition source set drifted")

    pruning = _mapping(document.get("pruning_decisions"), "pruning_decisions")
    duplicate_prs = tuple(
        _sequence(pruning.get("duplicate_closed_unmerged"), "duplicate_closed_unmerged")
    )
    superseded_prs = tuple(
        _sequence(
            pruning.get("superseded_closed_unmerged"), "superseded_closed_unmerged"
        )
    )
    if len(set(duplicate_prs + superseded_prs)) != len(duplicate_prs) + len(
        superseded_prs
    ):
        raise LateWaveIntakeError("pruning decisions must not classify one PR twice")
    for pr in duplicate_prs + superseded_prs:
        if not isinstance(pr, int) or pr <= 0:
            raise LateWaveIntakeError("pruning PR numbers must be positive integers")

    policy = _mapping(document.get("next_composition_policy"), "next_composition_policy")
    minimum = tuple(_sequence(policy.get("minimum_required"), "policy.minimum_required"))
    if minimum:
        raise LateWaveIntakeError(
            "minimum late-wave composition set must be empty after materialization"
        )
    integrated_required = tuple(
        _sequence(policy.get("integrated_required"), "policy.integrated_required")
    )
    if integrated_required != tuple(integrated_prs):
        raise LateWaveIntakeError("integrated-required policy must match composition sources")
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
        "integrated_prs": tuple(integrated_prs),
        "integrated_source_shas": integrated_shas,
        "composition_commit": composition_commit,
        "composition_first_parent": first_parent,
        "composition_materialized": True,
        "next_composition_ready": False,
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


def verify_base_ancestry(
    document: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Require the checkout to descend from the exact base and materialized source parents."""

    facts = validate_late_wave_snapshot(document)
    root = Path(repo_root)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    _full_sha(head, "git HEAD")

    required_ancestors = [
        facts["base_sha"],
        facts["composition_commit"],
        *facts["integrated_source_shas"].values(),
    ]
    for ancestor in required_ancestors:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            check=True,
        )

    parent_text = subprocess.check_output(
        ["git", "show", "-s", "--format=%P", facts["composition_commit"]],
        cwd=root,
        text=True,
    ).strip()
    parents = tuple(parent_text.split())
    expected_parents = (
        facts["composition_first_parent"],
        *facts["integrated_source_shas"].values(),
    )
    if parents != expected_parents:
        raise LateWaveIntakeError("materialized composition parent set/order drifted")

    return {
        "head_sha": head,
        "base_sha": facts["base_sha"],
        "composition_commit": facts["composition_commit"],
        "integrated_source_shas": facts["integrated_source_shas"],
    }


def load_and_validate(
    path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the committed snapshot; optionally verify exact Git ancestry."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = validate_late_wave_snapshot(document)
    if repo_root is not None:
        facts["ancestry"] = verify_base_ancestry(document, repo_root)
    return facts
