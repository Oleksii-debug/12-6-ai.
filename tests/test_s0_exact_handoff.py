from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.integration.s0_exact_handoff import (
    ExactHandoffValidationError,
    component_map_sha256,
    validate_exact_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_PATH = ROOT / "configs" / "releases" / "s0_exact_handoff_20260824.prepared.json"
CANDIDATE_PATH = (
    ROOT / "configs" / "releases" / "s0_candidate_convergence_20260824.experimental.json"
)
RUN_PATH = ROOT / "configs" / "runs" / "s0_10k.pr81_exact_candidate.local_free.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return _load(HANDOFF_PATH), _load(CANDIDATE_PATH), _load(RUN_PATH)


def _refresh_component_hash(handoff: dict[str, object]) -> None:
    components = handoff["components"]
    assert isinstance(components, list)
    handoff["component_map_sha256"] = component_map_sha256(components)


def test_exact_pr81_handoff_is_ready_local_free_but_not_promotable() -> None:
    handoff, candidate, run = _bundle()

    result = validate_exact_handoff(handoff, candidate, run)

    assert result["execution_ready"] is True
    assert result["target_candidate_sha"] == (
        "1caa729c8efafc84e7a5c4b1f7295eb8dcdb5a8d"
    )
    assert result["accepted_lanes"] == (
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06",
        "D07",
        "D08",
    )
    assert result["component_map_sha256"] == (
        "710446f99bcb9a4609ed9bf6d0cc7c54f20c733e56967c05fa470a6d9c76ef4b"
    )
    assert result["promotion_allowed"] is False
    assert result["audit_status"] == "RETEST_REQUIRED"


def test_exact_handoff_binds_completed_candidate_ci_and_real_training() -> None:
    handoff, candidate, run = _bundle()
    target = handoff["target_candidate"]
    assert isinstance(target, dict)
    exact_runs = target["exact_head_runs"]
    assert isinstance(exact_runs, list)

    by_name = {entry["name"]: entry for entry in exact_runs}
    assert by_name["CI"] == {
        "name": "CI",
        "run_id": 32761570313,
        "status": "completed",
        "conclusion": "success",
    }
    assert by_name["D02 Real S0 Training"] == {
        "name": "D02 Real S0 Training",
        "run_id": 32761570314,
        "status": "completed",
        "conclusion": "success",
    }

    validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_stale_or_non_green_target_ci() -> None:
    handoff, candidate, run = _bundle()
    target = handoff["target_candidate"]
    assert isinstance(target, dict)
    exact_runs = target["exact_head_runs"]
    assert isinstance(exact_runs, list)
    exact_runs[0]["conclusion"] = "failure"

    with pytest.raises(ExactHandoffValidationError, match="completed success"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_component_source_drift_from_candidate_manifest() -> None:
    handoff, candidate, run = _bundle()
    components = handoff["components"]
    assert isinstance(components, list)
    d06 = next(item for item in components if item["lane"] == "D06")
    d06["source_sha"] = "a" * 40
    _refresh_component_hash(handoff)

    with pytest.raises(ExactHandoffValidationError, match="D06 candidate/handoff drift"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_old_held_lane_semantics() -> None:
    handoff, candidate, run = _bundle()
    components = handoff["components"]
    assert isinstance(components, list)
    d06 = next(item for item in components if item["lane"] == "D06")
    d06["disposition"] = "held"
    _refresh_component_hash(handoff)

    with pytest.raises(ExactHandoffValidationError, match="must be accepted"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_tampered_component_map() -> None:
    handoff, candidate, run = _bundle()
    components = handoff["components"]
    assert isinstance(components, list)
    d02 = next(item for item in components if item["lane"] == "D02")
    d02["ci_runs"] = [32745954583]

    with pytest.raises(ExactHandoffValidationError, match="component_map_sha256"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_paid_compute_or_foreign_base_weights() -> None:
    handoff, candidate, run = _bundle()
    authorization = handoff["authorization"]
    assert isinstance(authorization, dict)
    authorization["paid_compute_authorized"] = True

    with pytest.raises(ExactHandoffValidationError, match="paid compute"):
        validate_exact_handoff(handoff, candidate, run)

    handoff, candidate, run = _bundle()
    components = handoff["components"]
    assert isinstance(components, list)
    d01 = next(item for item in components if item["lane"] == "D01")
    d01["contains_foreign_pretrained_weights"] = True
    _refresh_component_hash(handoff)

    with pytest.raises(ExactHandoffValidationError, match="foreign pretrained"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_resolved_run_candidate_or_identity_drift() -> None:
    handoff, candidate, run = _bundle()
    run_candidate = run["candidate"]
    assert isinstance(run_candidate, dict)
    run_candidate["git_sha"] = "b" * 40

    with pytest.raises(ExactHandoffValidationError, match="candidate SHA mismatch"):
        validate_exact_handoff(handoff, candidate, run)

    handoff, candidate, run = _bundle()
    data = run["data"]
    assert isinstance(data, dict)
    data["packing_sha256"] = "0" * 64

    with pytest.raises(ExactHandoffValidationError, match="packing_sha256"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_rejects_missing_output_contract() -> None:
    handoff, candidate, run = _bundle()
    artifacts = handoff["artifact_contract"]
    assert isinstance(artifacts, dict)
    artifacts.pop("resume_evidence")

    with pytest.raises(ExactHandoffValidationError, match="artifact contract"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_keeps_audits_fail_closed_on_target_sha() -> None:
    handoff, candidate, run = _bundle()
    audits = handoff["audits"]
    assert isinstance(audits, dict)
    audit_a = audits["AUDIT-A"]
    assert isinstance(audit_a, dict)
    audit_a["status"] = "PASS"

    with pytest.raises(ExactHandoffValidationError, match="may not be treated as PASS"):
        validate_exact_handoff(handoff, candidate, run)

    handoff, candidate, run = _bundle()
    audits = handoff["audits"]
    assert isinstance(audits, dict)
    audit_b = audits["AUDIT-B"]
    assert isinstance(audit_b, dict)
    audit_b["candidate_sha"] = "c" * 40

    with pytest.raises(ExactHandoffValidationError, match="AUDIT-B candidate SHA mismatch"):
        validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_records_v1_pr59_as_superseded_history() -> None:
    handoff, candidate, run = _bundle()
    supersession = handoff["supersession"]
    assert isinstance(supersession, list)
    pr59 = next(item for item in supersession if item["pr_number"] == 59)

    assert pr59["head_sha"] == "19f813e780b544959c0bc0a5a4a523101a534461"
    assert pr59["disposition"] == "superseded"

    validate_exact_handoff(handoff, candidate, run)


def test_exact_handoff_cannot_self_promote_even_after_audit_metadata_copy() -> None:
    handoff, candidate, run = _bundle()
    handoff = copy.deepcopy(handoff)
    handoff["promotion_allowed"] = True

    with pytest.raises(ExactHandoffValidationError, match="self-authorize promotion"):
        validate_exact_handoff(handoff, candidate, run)
