"""Fail-closed exact-candidate S0 execution handoff validation.

This integration-owned module prepares a LOCAL_FREE train -> strict checkpoint ->
reload/resume -> evaluation -> first-party inference run against an already composed
Product candidate. It does not implement domain semantics and cannot authorize
promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_S0_LANES = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
_ALLOWED_DISPOSITIONS = {"accepted", "accepted_selective"}
_REQUIRED_CANDIDATE_RUNS = {"CI", "D02 Real S0 Training"}
_REQUIRED_ARTIFACTS = {
    "resolved_run_manifest",
    "training_evidence",
    "checkpoint_manifest",
    "checkpoint_manifest_checksum",
    "resume_evidence",
    "evaluation_report",
    "inference_report",
}
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExactHandoffValidationError(ValueError):
    """Raised when exact-candidate execution evidence is incomplete or inconsistent."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactHandoffValidationError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExactHandoffValidationError(f"{field} must be an array")
    return value


def _full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _FULL_GIT_SHA.fullmatch(value) is None:
        raise ExactHandoffValidationError(
            f"{field} must be a full lowercase 40-hex Git SHA"
        )
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExactHandoffValidationError(f"{field} must be a lowercase SHA-256")
    return value


def component_map_sha256(components: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical SHA-256 identity of the ordered component evidence map."""
    payload = json.dumps(
        list(components),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_exact_handoff(
    handoff: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a prepared handoff against the exact composed candidate and C01 run."""

    if handoff.get("schema") != "12-6.s0-exact-candidate-handoff.v2":
        raise ExactHandoffValidationError("unsupported handoff schema")
    if handoff.get("run_id") != "12-6-AI-SWARM-EXP-01":
        raise ExactHandoffValidationError("unexpected swarm run_id")
    if handoff.get("repository") != "Oleksii-debug/12-6-ai.":
        raise ExactHandoffValidationError("repository identity mismatch")
    if handoff.get("stage") != "S0":
        raise ExactHandoffValidationError("stage must be S0")
    if handoff.get("handoff_state") != "READY_LOCAL_FREE":
        raise ExactHandoffValidationError("handoff_state must be READY_LOCAL_FREE")
    if handoff.get("promotion_allowed") is not False:
        raise ExactHandoffValidationError("handoff may never self-authorize promotion")

    target = _mapping(handoff.get("target_candidate"), "target_candidate")
    target_sha = _full_sha(target.get("sha"), "target_candidate.sha")
    if target.get("branch") != "d01/s0-candidate-convergence-20260824-b":
        raise ExactHandoffValidationError("target candidate branch mismatch")
    if target.get("pr_number") != 81:
        raise ExactHandoffValidationError("target candidate PR must be #81")
    if target.get("status") != "experimental":
        raise ExactHandoffValidationError("target candidate must remain experimental")

    runs = _sequence(target.get("exact_head_runs"), "target_candidate.exact_head_runs")
    run_names: set[str] = set()
    for index, raw in enumerate(runs):
        run = _mapping(raw, f"target_candidate.exact_head_runs[{index}]")
        name = run.get("name")
        if not isinstance(name, str) or not name:
            raise ExactHandoffValidationError("candidate workflow run requires a name")
        if name in run_names:
            raise ExactHandoffValidationError(f"duplicate candidate workflow {name}")
        if not isinstance(run.get("run_id"), int) or run["run_id"] <= 0:
            raise ExactHandoffValidationError(f"{name} run_id must be positive")
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            raise ExactHandoffValidationError(
                f"{name} must be completed success on the exact target head"
            )
        run_names.add(name)
    if run_names != _REQUIRED_CANDIDATE_RUNS:
        raise ExactHandoffValidationError("exact target CI and D02 training runs are required")

    authorization = _mapping(handoff.get("authorization"), "authorization")
    if authorization.get("execution_class") != "LOCAL_FREE":
        raise ExactHandoffValidationError("execution class must be LOCAL_FREE")
    if authorization.get("local_free_allowed") is not True:
        raise ExactHandoffValidationError("LOCAL_FREE must be explicitly allowed")
    if authorization.get("paid_compute_authorized") is not False:
        raise ExactHandoffValidationError("paid compute must remain unauthorized")
    if authorization.get("foreign_pretrained_base_allowed") is not False:
        raise ExactHandoffValidationError("foreign pretrained Base must remain forbidden")
    if authorization.get("behavioral_base_weights_allowed") is not False:
        raise ExactHandoffValidationError("behavioral Base weights must remain forbidden")

    components_raw = _sequence(handoff.get("components"), "components")
    components: list[Mapping[str, Any]] = []
    by_lane: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(components_raw):
        component = _mapping(raw, f"components[{index}]")
        lane = component.get("lane")
        if lane not in REQUIRED_S0_LANES:
            raise ExactHandoffValidationError(f"invalid required lane at components[{index}]")
        if lane in by_lane:
            raise ExactHandoffValidationError(f"duplicate component lane {lane}")
        _full_sha(component.get("source_sha"), f"{lane}.source_sha")
        if component.get("disposition") not in _ALLOWED_DISPOSITIONS:
            raise ExactHandoffValidationError(f"{lane} must be accepted for execution")
        if component.get("ci_conclusion") != "success":
            raise ExactHandoffValidationError(f"{lane} requires exact-source green CI")
        component_runs = _sequence(component.get("ci_runs"), f"{lane}.ci_runs")
        if not component_runs or any(
            not isinstance(run_id, int) or run_id <= 0 for run_id in component_runs
        ):
            raise ExactHandoffValidationError(f"{lane}.ci_runs must be positive run IDs")
        if component.get("contains_behavioral_weights") is not False:
            raise ExactHandoffValidationError(f"{lane} may not add behavioral Base weights")
        if component.get("contains_foreign_pretrained_weights") is not False:
            raise ExactHandoffValidationError(f"{lane} may not add foreign pretrained weights")
        components.append(component)
        by_lane[lane] = component

    if tuple(sorted(by_lane)) != REQUIRED_S0_LANES:
        raise ExactHandoffValidationError("component map must contain D01-D08 exactly")
    declared_hash = _sha256(handoff.get("component_map_sha256"), "component_map_sha256")
    if declared_hash != component_map_sha256(components):
        raise ExactHandoffValidationError("component_map_sha256 mismatch")

    if candidate_manifest.get("schema_version") != 2:
        raise ExactHandoffValidationError("candidate manifest schema mismatch")
    if candidate_manifest.get("repository") != handoff.get("repository"):
        raise ExactHandoffValidationError("candidate repository mismatch")
    if candidate_manifest.get("stage") != "S0":
        raise ExactHandoffValidationError("candidate manifest stage mismatch")
    if candidate_manifest.get("status") != "experimental":
        raise ExactHandoffValidationError("candidate manifest must remain experimental")
    if candidate_manifest.get("composition_complete") is not True:
        raise ExactHandoffValidationError("candidate composition is not complete")
    if candidate_manifest.get("promotion_eligible") is not False:
        raise ExactHandoffValidationError("candidate manifest may not self-promote")
    if candidate_manifest.get("canonical_base") != "random_init_pretraining_only":
        raise ExactHandoffValidationError("canonical Base contract mismatch")
    if candidate_manifest.get("integration_branch") != target.get("branch"):
        raise ExactHandoffValidationError("candidate branch is not the handoff target")

    candidate_components = _sequence(candidate_manifest.get("components"), "candidate.components")
    candidate_by_lane = {
        str(_mapping(item, "candidate component").get("lane")): _mapping(
            item, "candidate component"
        )
        for item in candidate_components
    }
    for lane in REQUIRED_S0_LANES:
        if lane not in candidate_by_lane:
            raise ExactHandoffValidationError(f"candidate manifest is missing {lane}")
        expected = by_lane[lane]
        actual = candidate_by_lane[lane]
        for key in (
            "source_sha",
            "disposition",
            "pr_number",
            "ci_runs",
            "ci_conclusion",
            "contains_behavioral_weights",
            "contains_foreign_pretrained_weights",
        ):
            if actual.get(key) != expected.get(key):
                raise ExactHandoffValidationError(f"{lane} candidate/handoff drift in {key}")

    identities = _mapping(handoff.get("identities"), "identities")
    identity_pairs = {
        "model_spec_sha256": "model_spec_sha256",
        "init_spec_sha256": "init_spec_sha256",
        "tokenizer_config_sha256": "tokenizer_config_sha256",
        "tokenizer_vocab_sha256": "tokenizer_vocab_sha256",
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "environment_lock_sha256": "environment_lock_index_sha256",
    }
    for handoff_key, candidate_key in identity_pairs.items():
        value = _sha256(identities.get(handoff_key), f"identities.{handoff_key}")
        if candidate_manifest.get(candidate_key) != value:
            raise ExactHandoffValidationError(f"candidate identity drift: {handoff_key}")

    if run_manifest.get("schema_version") != 1:
        raise ExactHandoffValidationError("resolved run manifest schema mismatch")
    if run_manifest.get("run_id") != handoff.get("execution_run_id"):
        raise ExactHandoffValidationError("run_id mismatch")
    if run_manifest.get("stage") != "S0":
        raise ExactHandoffValidationError("resolved run stage mismatch")
    if run_manifest.get("state") != "PREPARED_NOT_LAUNCHED":
        raise ExactHandoffValidationError("resolved run must remain PREPARED_NOT_LAUNCHED")
    run_auth = _mapping(run_manifest.get("authorization"), "run.authorization")
    if run_auth.get("class") != "LOCAL_FREE" or run_auth.get("compute_authorized") is not False:
        raise ExactHandoffValidationError(
            "resolved run authorization is not LOCAL_FREE fail-closed"
        )
    candidate = _mapping(run_manifest.get("candidate"), "run.candidate")
    if candidate.get("repository") != handoff.get("repository"):
        raise ExactHandoffValidationError("resolved run repository mismatch")
    if candidate.get("git_sha") != target_sha:
        raise ExactHandoffValidationError("resolved run candidate SHA mismatch")
    if candidate.get("branch_or_tag") != target.get("branch"):
        raise ExactHandoffValidationError("resolved run candidate branch mismatch")
    if candidate.get("parameter_count") != 10140:
        raise ExactHandoffValidationError("resolved run must bind exact S0 parameter count")
    if candidate.get("modelspec_sha256") != identities.get("model_spec_sha256"):
        raise ExactHandoffValidationError("resolved run ModelSpec identity mismatch")
    if candidate.get("initspec_sha256") != identities.get("init_spec_sha256"):
        raise ExactHandoffValidationError("resolved run InitSpec identity mismatch")

    data = _mapping(run_manifest.get("data"), "run.data")
    data_pairs = {
        "dataset_manifest_sha256": "dataset_manifest_sha256",
        "tokenizer_sha256": "tokenizer_config_sha256",
        "tokenizer_vocab_sha256": "tokenizer_vocab_sha256",
        "packing_sha256": "packing_config_sha256",
    }
    for run_key, identity_key in data_pairs.items():
        if data.get(run_key) != identities.get(identity_key):
            raise ExactHandoffValidationError(f"resolved run data drift: {run_key}")

    environment = _mapping(run_manifest.get("environment"), "run.environment")
    if environment.get("lock_sha256") != identities.get("environment_lock_sha256"):
        raise ExactHandoffValidationError("resolved run environment lock mismatch")
    training = _mapping(run_manifest.get("training"), "run.training")
    if training.get("seed") != 1337:
        raise ExactHandoffValidationError("resolved run seed must bind the proven S0 seed")
    if training.get("device") != "cpu" or training.get("precision") != "fp32":
        raise ExactHandoffValidationError("resolved run must remain CPU fp32")
    if training.get("optimizer") != "AdamW" or training.get("scheduler") != "constant":
        raise ExactHandoffValidationError("resolved run optimizer/scheduler mismatch")
    if training.get("target_steps") != 40:
        raise ExactHandoffValidationError("resolved run target_steps must be 40")

    artifact_contract = _mapping(handoff.get("artifact_contract"), "artifact_contract")
    if set(artifact_contract) != _REQUIRED_ARTIFACTS:
        raise ExactHandoffValidationError("artifact contract is incomplete")
    if any(not isinstance(path, str) or not path.strip() for path in artifact_contract.values()):
        raise ExactHandoffValidationError("artifact paths must be non-empty")

    audits = _mapping(handoff.get("audits"), "audits")
    if set(audits) != {"AUDIT-A", "AUDIT-B"}:
        raise ExactHandoffValidationError("audits must contain exactly AUDIT-A and AUDIT-B")
    for auditor, raw in audits.items():
        audit = _mapping(raw, f"audits.{auditor}")
        if audit.get("status") != "RETEST_REQUESTED":
            raise ExactHandoffValidationError(f"{auditor} may not be treated as PASS")
        if audit.get("candidate_sha") != target_sha:
            raise ExactHandoffValidationError(f"{auditor} candidate SHA mismatch")
        if audit.get("issue_number") not in {13, 14}:
            raise ExactHandoffValidationError(f"{auditor} issue reference is invalid")

    supersession = _sequence(handoff.get("supersession"), "supersession")
    if not any(
        _mapping(item, "supersession item").get("pr_number") == 59
        and _mapping(item, "supersession item").get("disposition") == "superseded"
        for item in supersession
    ):
        raise ExactHandoffValidationError("historical PR #59 supersession must be recorded")

    return {
        "execution_ready": True,
        "target_candidate_sha": target_sha,
        "component_map_sha256": declared_hash,
        "accepted_lanes": REQUIRED_S0_LANES,
        "promotion_allowed": False,
        "audit_status": "RETEST_REQUIRED",
    }
