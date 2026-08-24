"""Fail-closed validation for D01 late-wave S0 intake snapshots."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_DISPOSITIONS = {
    "PENDING_EXACT_HEAD_CI",
    "RECHECK_NEW_HEAD_AFTER_HISTORICAL_RED",
    "PENDING_AND_COLLIDES_WITH_PR95",
    "EXCLUDE_FROM_S0_COMPOSITION",
    "HELD_D10_OWNED",
    "PENDING_AND_OVERLAPS_EVIDENCE_GROUP",
}


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


def _workflow_success(workflow: Mapping[str, Any], field: str) -> None:
    if workflow.get("state") != "completed" or workflow.get("conclusion") != "success":
        raise LateWaveIntakeError(f"{field} is not terminal success")
    run_id = workflow.get("run_id")
    if not isinstance(run_id, int) or run_id <= 0:
        raise LateWaveIntakeError(f"{field}.run_id must be a positive integer")


def validate_late_wave_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a point-in-time D01 intake registry without accepting pending work."""

    if document.get("schema") != "12-6.s0-late-wave-intake.v1":
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
    workflows = _mapping(base.get("required_workflows"), "base.required_workflows")
    required_base_workflows = {
        "CI",
        "D02 Real S0 Training",
        "D02 S0 Determinism Repeatability",
        "D04 Strict S0 Exact-Candidate Evaluation",
    }
    if set(workflows) != required_base_workflows:
        raise LateWaveIntakeError("base exact-green workflow set is incomplete")
    for name in sorted(workflows):
        _workflow_success(_mapping(workflows[name], f"base workflow {name}"), f"base workflow {name}")

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
    group_members: dict[int, str] = {}
    group_owners: dict[str, int] = {}
    for index, raw in enumerate(collision_groups):
        group = _mapping(raw, f"collision_groups[{index}]")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise LateWaveIntakeError("collision group id must be non-empty")
        members = _sequence(group.get("members"), f"collision group {group_id}.members")
        if len(members) < 2 or len(set(members)) != len(members):
            raise LateWaveIntakeError(f"collision group {group_id} must have unique competing PRs")
        owner = group.get("owner_pr")
        if owner is not None and owner not in members:
            raise LateWaveIntakeError(f"collision group {group_id} owner must be a member")
        if owner is not None:
            group_owners[group_id] = owner
        for member in members:
            if not isinstance(member, int) or member <= 0:
                raise LateWaveIntakeError("collision group PR numbers must be positive integers")
            previous = group_members.get(member)
            if previous is not None and previous != group_id:
                raise LateWaveIntakeError(f"PR #{member} appears in multiple collision groups")
            group_members[member] = group_id

    items_raw = _sequence(document.get("items"), "items")
    items: dict[int, Mapping[str, Any]] = {}
    by_disposition: defaultdict[str, list[int]] = defaultdict(list)
    for index, raw in enumerate(items_raw):
        item = _mapping(raw, f"items[{index}]")
        pr = item.get("pr")
        if not isinstance(pr, int) or pr <= 0 or pr in items:
            raise LateWaveIntakeError("item PR numbers must be unique positive integers")
        _full_sha(item.get("head"), f"PR #{pr}.head")
        disposition = item.get("disposition")
        if disposition not in _ALLOWED_DISPOSITIONS:
            raise LateWaveIntakeError(f"PR #{pr} has unsupported disposition")
        if item.get("required_for_next_composition") not in {True, False}:
            raise LateWaveIntakeError(f"PR #{pr}.required_for_next_composition must be boolean")
        if item.get("lane") == "D02" and item.get("kind") == "s1_numerical_preflight":
            if disposition != "EXCLUDE_FROM_S0_COMPOSITION":
                raise LateWaveIntakeError("S1 numerical preflight must not enter S0 composition")
        if item.get("lane") == "D10" and disposition != "HELD_D10_OWNED":
            raise LateWaveIntakeError("D10 governance work must remain separately owned")
        if disposition.startswith("PENDING") and item.get("required_for_next_composition") is True:
            workflow_state = item.get("workflow_state")
            if workflow_state is not None:
                states = _mapping(workflow_state, f"PR #{pr}.workflow_state")
                if states and all(state == "success" for state in states.values()):
                    raise LateWaveIntakeError(
                        f"PR #{pr} is marked pending even though recorded workflows are all success; refresh snapshot"
                    )
        items[pr] = item
        by_disposition[disposition].append(pr)

    policy = _mapping(document.get("next_composition_policy"), "next_composition_policy")
    minimum = tuple(_sequence(policy.get("minimum_required"), "policy.minimum_required"))
    if minimum != (90, 91, 100):
        raise LateWaveIntakeError("minimum late-wave composition set drifted")
    if any(pr not in items for pr in minimum):
        raise LateWaveIntakeError("minimum composition PR missing from registry")
    if any(items[pr]["disposition"] != "PENDING_EXACT_HEAD_CI" for pr in minimum):
        raise LateWaveIntakeError("minimum composition PRs must remain pending at this cutoff")
    if policy.get("never_accept_queued_or_red") is not True:
        raise LateWaveIntakeError("queued/red intake must fail closed")
    if policy.get("preserve_real_git_ancestry") is not True:
        raise LateWaveIntakeError("real Git ancestry preservation is mandatory")
    if policy.get("rerun_full_exact_head_workflows_after_composition") is not True:
        raise LateWaveIntakeError("post-composition exact-head rerun is mandatory")
    if policy.get("request_both_independent_audits_on_final_exact_head") is not True:
        raise LateWaveIntakeError("both independent audit handoffs are mandatory")

    choose_one = _sequence(policy.get("competing_choose_one"), "policy.competing_choose_one")
    for group in choose_one:
        members = tuple(_sequence(group, "competing choose-one group"))
        if len(members) < 2 or len(set(members)) != len(members):
            raise LateWaveIntakeError("choose-one group must contain unique competitors")
        if any(pr not in items for pr in members):
            raise LateWaveIntakeError("choose-one competitor missing from registry")

    return {
        "base_sha": base_sha,
        "registered_prs": tuple(sorted(items)),
        "minimum_required": minimum,
        "collision_group_owners": group_owners,
        "pending_prs": tuple(sorted(pr for disposition, prs in by_disposition.items() if disposition.startswith("PENDING") for pr in prs)),
        "excluded_from_s0": tuple(sorted(by_disposition["EXCLUDE_FROM_S0_COMPOSITION"])),
        "governance_held": tuple(sorted(by_disposition["HELD_D10_OWNED"])),
        "promotion_eligible": False,
    }


def verify_base_ancestry(document: Mapping[str, Any], repo_root: str | Path) -> dict[str, str]:
    """Require the registry checkout to descend from the exact-green PR #89 base."""

    facts = validate_late_wave_snapshot(document)
    root = Path(repo_root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _full_sha(head, "git HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", facts["base_sha"], head], cwd=root, check=True)
    return {"head_sha": head, "base_sha": facts["base_sha"]}


def load_and_validate(path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the committed snapshot; optionally verify exact Git ancestry."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = validate_late_wave_snapshot(document)
    if repo_root is not None:
        facts["ancestry"] = verify_base_ancestry(document, repo_root)
    return facts
