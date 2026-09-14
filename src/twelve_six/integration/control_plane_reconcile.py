"""Fail-closed D10 gate for reconciling Product candidate and live control-plane main."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RECONCILED = "RECONCILED_EXACT_HEAD_GREEN"
BLOCKED = "BLOCKED_NEEDS_CONTROL_PLANE_RECONCILIATION"


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(ch in "0123456789abcdef" for ch in value)
    )


def assess_control_plane_reconciliation(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Assess whether a candidate may be treated as reconciled with live ``main``.

    Historical exact-green parent runs are ancestry evidence only. A merge/candidate SHA
    must itself have terminal required CI, and it must contain the declared live-main
    control-plane lineage before this gate can return RECONCILED.
    """

    blockers: list[str] = []
    for key in ("live_main_sha", "candidate_sha", "merge_base_sha"):
        if not _sha(evidence.get(key)):
            blockers.append(f"{key}.invalid")

    ahead = evidence.get("ahead_by")
    behind = evidence.get("behind_by")
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead < 0:
        blockers.append("ahead_by.invalid")
    if not isinstance(behind, int) or isinstance(behind, bool) or behind < 0:
        blockers.append("behind_by.invalid")

    if evidence.get("live_main_is_ancestor") is not True:
        blockers.append("live_main.not_ancestor_of_candidate")
    if behind != 0:
        blockers.append("candidate.behind_live_main")

    exact_ci = evidence.get("exact_candidate_ci")
    if not isinstance(exact_ci, Mapping):
        blockers.append("exact_candidate_ci.missing")
    else:
        if exact_ci.get("source_sha") != evidence.get("candidate_sha"):
            blockers.append("exact_candidate_ci.source_sha_mismatch")
        if exact_ci.get("status") != "completed":
            blockers.append("exact_candidate_ci.not_completed")
        if exact_ci.get("conclusion") != "success":
            blockers.append("exact_candidate_ci.not_success")
        run_id = exact_ci.get("run_id")
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            blockers.append("exact_candidate_ci.run_id_invalid")

    if evidence.get("historical_parent_green_promoted") is not False:
        blockers.append("historical_green.must_not_promote")
    if evidence.get("training_authorized") is not False:
        blockers.append("training_authorized.must_remain_false")
    if evidence.get("stage_promoted") is not False:
        blockers.append("stage_promoted.must_remain_false")

    return {
        "status": RECONCILED if not blockers else BLOCKED,
        "ready": not blockers,
        "blockers": blockers,
    }
