"""Fail-closed LOCAL_FREE GitHub-hosted session planning for learned-20M."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from twelve_six.portable_run_packet import assess_portable_run_packet

PROFILE_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
PROFILE_ID = "D08-GITHUB-HOSTED-LOCAL-FREE-CPU-V1"
PLAN_ID = "D08-GITHUB-HOSTED-LOCAL-FREE-SESSION-PLAN-V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_KEYS = {
    "schema_version",
    "profile_id",
    "provider",
    "portable_resource_provider",
    "resource_class",
    "runner_label",
    "maximum_cost_usd",
    "materially_paid",
    "configured_job_budget_minutes",
    "checkpoint_safety_margin_minutes",
    "max_session_work_minutes",
    "checkpoint_early_required",
    "fresh_process_resume_required",
    "content_addressed_checkpoint_required",
}
_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "profile_id",
    "profile_sha256",
    "estimated_runtime_minutes",
    "session_count",
    "sessions",
    "truth_boundary",
    "plan_sha256",
}
_SESSION_KEYS = {
    "session_index",
    "mode",
    "planned_work_minutes",
    "configured_job_budget_minutes",
    "checkpoint_deadline_minutes",
    "requires_previous_handoff",
}
_TRUTH_KEYS = {
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
}
_HANDOFF_KEYS = {
    "plan_sha256",
    "session_index",
    "run_id",
    "checkpoint_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_uri",
}


@dataclass(frozen=True)
class SessionLaunchAssessment:
    ready: bool
    session_index: int | None
    blockers: tuple[str, ...]
    portable_packet_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "session_index": self.session_index,
            "blockers": list(self.blockers),
            "portable_packet_sha256": self.portable_packet_sha256,
        }


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _exact_int(value) and value > 0


def _zero_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value == 0
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(
    value: dict[str, Any], expected: set[str], prefix: str
) -> list[str]:
    errors = [f"{prefix}_{key}_missing" for key in sorted(expected - set(value))]
    errors += [
        f"{prefix}_{key}_unexpected" for key in sorted(set(value) - expected)
    ]
    return errors


def validate_profile(profile: Any) -> list[str]:
    if not isinstance(profile, dict):
        return ["profile_root_must_be_object"]
    errors = _exact_keys(profile, _PROFILE_KEYS, "profile")
    if (
        not _exact_int(profile.get("schema_version"))
        or profile.get("schema_version") != PROFILE_SCHEMA_VERSION
    ):
        errors.append("profile_schema_version_mismatch")
    if profile.get("profile_id") != PROFILE_ID:
        errors.append("profile_id_mismatch")
    if profile.get("provider") != "GITHUB_HOSTED_STANDARD":
        errors.append("profile_provider_must_be_github_hosted_standard")
    if profile.get("portable_resource_provider") != "OTHER_FREE":
        errors.append("profile_portable_resource_provider_must_be_other_free")
    if profile.get("resource_class") != "LOCAL_FREE":
        errors.append("profile_resource_class_must_be_local_free")
    if profile.get("runner_label") != "ubuntu-24.04":
        errors.append("profile_runner_label_mismatch")
    if not _zero_number(profile.get("maximum_cost_usd")):
        errors.append("profile_maximum_cost_usd_must_be_zero")
    if profile.get("materially_paid") is not False:
        errors.append("profile_materially_paid_must_be_false")

    job = profile.get("configured_job_budget_minutes")
    margin = profile.get("checkpoint_safety_margin_minutes")
    work = profile.get("max_session_work_minutes")
    if not _positive_int(job):
        errors.append("profile_configured_job_budget_minutes_invalid")
    if not _positive_int(margin):
        errors.append("profile_checkpoint_safety_margin_minutes_invalid")
    if not _positive_int(work):
        errors.append("profile_max_session_work_minutes_invalid")
    if _positive_int(job) and _positive_int(margin) and margin >= job:
        errors.append("profile_checkpoint_safety_margin_must_precede_job_limit")
    if (
        all(_positive_int(item) for item in (job, margin, work))
        and work > job - margin
    ):
        errors.append("profile_work_window_exceeds_safe_job_budget")

    for key in (
        "checkpoint_early_required",
        "fresh_process_resume_required",
        "content_addressed_checkpoint_required",
    ):
        if profile.get(key) is not True:
            errors.append(f"profile_{key}_must_be_true")
    return sorted(set(errors))


def _truth() -> dict[str, Any]:
    return {
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }


def build_session_plan(
    profile: Any, estimated_runtime_minutes: Any
) -> dict[str, Any]:
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid_profile:" + ",".join(errors))
    if not _positive_int(estimated_runtime_minutes):
        raise ValueError("estimated_runtime_minutes_must_be_positive_integer")

    window = profile["max_session_work_minutes"]
    count = (estimated_runtime_minutes + window - 1) // window
    deadline = (
        profile["configured_job_budget_minutes"]
        - profile["checkpoint_safety_margin_minutes"]
    )
    sessions: list[dict[str, Any]] = []
    remaining = estimated_runtime_minutes
    for index in range(1, count + 1):
        work = min(window, remaining)
        sessions.append(
            {
                "session_index": index,
                "mode": "FRESH_START" if index == 1 else "RESUME",
                "planned_work_minutes": work,
                "configured_job_budget_minutes": profile[
                    "configured_job_budget_minutes"
                ],
                "checkpoint_deadline_minutes": deadline,
                "requires_previous_handoff": index > 1,
            }
        )
        remaining -= work

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "profile_id": profile["profile_id"],
        "profile_sha256": canonical_sha256(profile),
        "estimated_runtime_minutes": estimated_runtime_minutes,
        "session_count": count,
        "sessions": sessions,
        "truth_boundary": _truth(),
        "plan_sha256": None,
    }
    sealed = copy.deepcopy(plan)
    sealed.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(sealed)
    return plan


def validate_session_plan(plan: Any, profile: Any) -> list[str]:
    profile_errors = validate_profile(profile)
    if profile_errors:
        return [f"profile:{item}" for item in profile_errors]
    if not isinstance(plan, dict):
        return ["plan_root_must_be_object"]

    errors = _exact_keys(plan, _PLAN_KEYS, "plan")
    if (
        not _exact_int(plan.get("schema_version"))
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        errors.append("plan_schema_version_mismatch")
    if plan.get("plan_id") != PLAN_ID:
        errors.append("plan_id_mismatch")
    if plan.get("profile_id") != profile["profile_id"]:
        errors.append("plan_profile_id_mismatch")
    if plan.get("profile_sha256") != canonical_sha256(profile):
        errors.append("plan_profile_sha256_mismatch")

    runtime = plan.get("estimated_runtime_minutes")
    count = plan.get("session_count")
    if not _positive_int(runtime):
        errors.append("plan_estimated_runtime_minutes_invalid")
    if not _positive_int(count):
        errors.append("plan_session_count_invalid")

    sessions = plan.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        errors.append("plan_sessions_missing")
        sessions = []
    if _positive_int(count) and len(sessions) != count:
        errors.append("plan_session_count_mismatch")

    total = 0
    deadline = (
        profile["configured_job_budget_minutes"]
        - profile["checkpoint_safety_margin_minutes"]
    )
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            errors.append(f"session_{index}_must_be_object")
            continue
        errors += _exact_keys(session, _SESSION_KEYS, f"session_{index}")
        if (
            not _exact_int(session.get("session_index"))
            or session.get("session_index") != index
        ):
            errors.append(f"session_{index}_index_mismatch")
        if session.get("mode") != (
            "FRESH_START" if index == 1 else "RESUME"
        ):
            errors.append(f"session_{index}_mode_mismatch")
        work = session.get("planned_work_minutes")
        if not _positive_int(work):
            errors.append(f"session_{index}_planned_work_minutes_invalid")
        elif work > profile["max_session_work_minutes"]:
            errors.append(f"session_{index}_work_window_exceeds_profile")
        else:
            total += work
        if session.get("configured_job_budget_minutes") != profile[
            "configured_job_budget_minutes"
        ]:
            errors.append(f"session_{index}_job_budget_mismatch")
        if (
            not _positive_int(session.get("checkpoint_deadline_minutes"))
            or session.get("checkpoint_deadline_minutes") != deadline
        ):
            errors.append(f"session_{index}_checkpoint_deadline_mismatch")
        if session.get("requires_previous_handoff") is not (index > 1):
            errors.append(f"session_{index}_handoff_requirement_mismatch")

    if _positive_int(runtime) and sessions and total != runtime:
        errors.append("plan_work_minutes_do_not_sum_to_estimate")
    if any(
        row.get("planned_work_minutes")
        != profile["max_session_work_minutes"]
        for row in sessions[:-1]
        if isinstance(row, dict)
    ):
        errors.append("plan_nonfinal_session_not_full_safe_window")

    truth = plan.get("truth_boundary")
    if not isinstance(truth, dict):
        errors.append("plan_truth_boundary_missing")
    else:
        errors += _exact_keys(truth, _TRUTH_KEYS, "plan_truth")
        for key in (
            "authorized_optimized_target_exposure",
            "optimizer_updates_executed_on_real_targets",
        ):
            if truth.get(key) != 0 or isinstance(truth.get(key), bool):
                errors.append(f"plan_truth_{key}_must_be_zero")
        for key in _TRUTH_KEYS - {
            "authorized_optimized_target_exposure",
            "optimizer_updates_executed_on_real_targets",
        }:
            if truth.get(key) is not False:
                errors.append(f"plan_truth_{key}_must_be_false")

    digest = plan.get("plan_sha256")
    if not _sha256(digest):
        errors.append("plan_sha256_invalid")
    else:
        sealed = copy.deepcopy(plan)
        sealed.pop("plan_sha256", None)
        if canonical_sha256(sealed) != digest:
            errors.append("plan_sha256_mismatch")
    return sorted(set(errors))


def _checkpoint_uri_valid(value: Any) -> bool:
    if not _nonempty(value):
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"file", "https"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def validate_previous_handoff(
    handoff: Any, *, plan_sha256: str, expected_session_index: int
) -> list[str]:
    """Validate correlation fields only; this handoff is never an authority root."""
    if not isinstance(handoff, dict):
        return ["previous_handoff_missing"]
    errors = _exact_keys(handoff, _HANDOFF_KEYS, "handoff")
    if handoff.get("plan_sha256") != plan_sha256:
        errors.append("handoff_plan_sha256_mismatch")
    if (
        not _exact_int(handoff.get("session_index"))
        or handoff.get("session_index") != expected_session_index - 1
    ):
        errors.append("handoff_session_index_mismatch")
    if not _nonempty(handoff.get("run_id")):
        errors.append("handoff_run_id_missing")
    for key in ("checkpoint_sha256", "checkpoint_manifest_sha256"):
        if not _sha256(handoff.get(key)):
            errors.append(f"handoff_{key}_invalid")
    if not _checkpoint_uri_valid(handoff.get("checkpoint_uri")):
        errors.append("handoff_checkpoint_uri_invalid_or_credential_bearing")
    return sorted(set(errors))


def _portable_same_provider_status(portable: Any) -> tuple[bool, tuple[str, ...]]:
    ready = getattr(
        portable, "ready_for_same_provider_fresh_process_resume", False
    )
    raw = getattr(portable, "same_provider_resume_blockers", None)
    if raw is None:
        return False, ("same_provider_resume_signal_unavailable",)
    if not isinstance(raw, (tuple, list)):
        return False, ("same_provider_resume_blockers_invalid",)
    blockers = tuple(str(item) for item in raw)
    return ready is True, blockers


def _portable_time_blockers(
    session: dict[str, Any],
    profile: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    limit = checkpoint.get("session_time_limit_minutes")
    first = checkpoint.get("first_checkpoint_deadline_minutes")
    if not _positive_int(limit):
        blockers.append("portable_session_time_limit_invalid")
        return blockers

    planned_work = session.get("planned_work_minutes")
    if (
        _positive_int(planned_work)
        and planned_work + profile["checkpoint_safety_margin_minutes"] > limit
    ):
        blockers.append("planned_session_exceeds_portable_time_limit")

    plan_deadline = session.get("checkpoint_deadline_minutes")
    if not _positive_int(plan_deadline) or plan_deadline >= limit:
        blockers.append("plan_checkpoint_deadline_not_before_portable_limit")

    if not _positive_int(first):
        blockers.append("portable_first_checkpoint_deadline_invalid")
    elif first >= limit:
        blockers.append("portable_first_checkpoint_deadline_not_before_limit")
    elif _positive_int(plan_deadline) and first > plan_deadline:
        blockers.append("portable_first_checkpoint_deadline_exceeds_plan_deadline")
    return blockers


def assess_session_launch(
    plan: Any,
    profile: Any,
    session_index: Any,
    portable_packet: Any,
    *,
    previous_handoff: Any = None,
) -> SessionLaunchAssessment:
    blockers = [f"profile:{item}" for item in validate_profile(profile)]
    if blockers:
        return SessionLaunchAssessment(False, None, tuple(blockers), None)

    blockers += validate_session_plan(plan, profile)
    if not _positive_int(session_index):
        blockers.append("session_index_invalid")
        return SessionLaunchAssessment(
            False, None, tuple(sorted(set(blockers))), None
        )

    sessions = plan.get("sessions") if isinstance(plan, dict) else None
    if not isinstance(sessions, list) or session_index > len(sessions):
        blockers.append("session_index_out_of_range")
        return SessionLaunchAssessment(
            False, session_index, tuple(sorted(set(blockers))), None
        )
    session = sessions[session_index - 1]
    if not isinstance(session, dict):
        blockers.append("session_entry_invalid")
        return SessionLaunchAssessment(
            False, session_index, tuple(sorted(set(blockers))), None
        )

    packet_hash = (
        canonical_sha256(portable_packet)
        if isinstance(portable_packet, dict)
        else None
    )
    if not isinstance(portable_packet, dict):
        blockers.append("portable_packet_root_must_be_object")
        return SessionLaunchAssessment(
            False, session_index, tuple(sorted(set(blockers))), packet_hash
        )

    resource = portable_packet.get("resource")
    checkpoint = portable_packet.get("checkpoint")
    if not isinstance(resource, dict):
        resource = {}
        blockers.append("portable_resource_missing")
    if not isinstance(checkpoint, dict):
        checkpoint = {}
        blockers.append("portable_checkpoint_missing")

    if resource.get("resource_class") != profile["resource_class"]:
        blockers.append("portable_resource_class_mismatch")
    if resource.get("provider") != profile["portable_resource_provider"]:
        blockers.append("portable_resource_provider_mismatch")
    if not _zero_number(resource.get("maximum_cost_usd")):
        blockers.append("portable_maximum_cost_usd_must_be_zero")
    if resource.get("materially_paid") is not False:
        blockers.append("portable_materially_paid_must_be_false")

    blockers += _portable_time_blockers(session, profile, checkpoint)
    portable = assess_portable_run_packet(portable_packet)

    if session_index == 1:
        if checkpoint.get("mode") != "FRESH_START":
            blockers.append("initial_session_requires_fresh_start_packet")
        if not portable.ready_for_initial_local_free_launch:
            blockers += [
                f"portable:{item}" for item in portable.launch_blockers
            ]
    else:
        blockers += validate_previous_handoff(
            previous_handoff,
            plan_sha256=plan.get("plan_sha256"),
            expected_session_index=session_index,
        )
        if checkpoint.get("mode") != "RESUME":
            blockers.append("continuation_session_requires_resume_packet")

        same_ready, same_blockers = _portable_same_provider_status(portable)
        if not same_ready:
            blockers.append("canonical_same_provider_resume_authority_unavailable")
            blockers += [f"portable:{item}" for item in same_blockers]

        lineage = checkpoint.get("lineage")
        if not isinstance(lineage, dict):
            blockers.append("portable_checkpoint_lineage_missing")
        elif isinstance(previous_handoff, dict):
            if lineage.get("cross_provider_transfer") is not False:
                blockers.append(
                    "same_provider_resume_must_not_declare_cross_provider_transfer"
                )
            if (
                lineage.get("source_provider")
                != profile["portable_resource_provider"]
            ):
                blockers.append("same_provider_source_identity_mismatch")
            if (
                lineage.get("parent_checkpoint_sha256")
                != previous_handoff.get("checkpoint_sha256")
            ):
                blockers.append("parent_checkpoint_sha256_mismatch")
            if (
                lineage.get("parent_manifest_sha256")
                != previous_handoff.get("checkpoint_manifest_sha256")
            ):
                blockers.append("parent_manifest_sha256_mismatch")
            if (
                lineage.get("previous_run_id")
                != previous_handoff.get("run_id")
            ):
                blockers.append("previous_run_id_mismatch")

    blockers = sorted(set(blockers))
    return SessionLaunchAssessment(
        not blockers, session_index, tuple(blockers), packet_hash
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--profile", type=Path, required=True)
    plan_cmd.add_argument(
        "--estimated-runtime-minutes", type=int, required=True
    )
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("--profile", type=Path, required=True)
    validate_cmd.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        profile = _read_json(args.profile)
        if args.command == "plan":
            result = build_session_plan(
                profile, args.estimated_runtime_minutes
            )
            print(
                json.dumps(
                    result, ensure_ascii=False, sort_keys=True, indent=2
                )
            )
            return 0
        errors = validate_session_plan(_read_json(args.plan), profile)
        print(
            json.dumps(
                {"valid": not errors, "errors": errors},
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if not errors else 2
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(
            json.dumps(
                {"valid": False, "errors": [str(exc)]},
                sort_keys=True,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
