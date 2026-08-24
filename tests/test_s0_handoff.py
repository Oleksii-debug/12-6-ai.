from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.integration.s0_handoff import (
    HandoffValidationError,
    component_map_sha256,
    validate_s0_handoff,
)

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_PATH = ROOT / "configs" / "releases" / "s0_handoff_20260824.prepared.json"


def _load_handoff() -> dict[str, object]:
    return json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))


def _refresh_component_hash(document: dict[str, object]) -> None:
    components = document["components"]
    assert isinstance(components, list)
    document["component_map_sha256"] = component_map_sha256(components)


def test_committed_handoff_is_exactly_blocked_on_d06() -> None:
    result = validate_s0_handoff(_load_handoff())

    assert result["execution_ready"] is False
    assert result["handoff_state"] == "PREPARED_BLOCKED"
    assert result["accepted_lanes"] == ("D01", "D02", "D03", "D04", "D05", "D07", "D08")
    assert result["held_lanes"] == ("D06",)
    assert result["promotion_allowed"] is False
    assert result["component_map_sha256"] == (
        "021768ace9f464dd6301abe4dfe37fde8394f97043dc4134aa8578811b83174b"
    )


def test_handoff_rejects_abbreviated_or_non_exact_source_sha() -> None:
    document = _load_handoff()
    components = document["components"]
    assert isinstance(components, list)
    components[0]["source_sha"] = "d646d41"
    _refresh_component_hash(document)

    with pytest.raises(HandoffValidationError, match="full lowercase 40-hex"):
        validate_s0_handoff(document)


def test_handoff_rejects_required_lane_omission() -> None:
    document = _load_handoff()
    components = document["components"]
    assert isinstance(components, list)
    document["components"] = [item for item in components if item["lane"] != "D08"]
    _refresh_component_hash(document)

    with pytest.raises(HandoffValidationError, match="D01-D08 exactly"):
        validate_s0_handoff(document)


def test_handoff_rejects_accepted_lane_with_failed_ci() -> None:
    document = _load_handoff()
    components = document["components"]
    assert isinstance(components, list)
    d06 = next(item for item in components if item["lane"] == "D06")
    d06["disposition"] = "accepted"
    d06["contains_behavioral_weights"] = False
    d06["contains_foreign_pretrained_weights"] = False
    _refresh_component_hash(document)

    with pytest.raises(HandoffValidationError, match="accepted lane D06 requires exact-head success"):
        validate_s0_handoff(document)


def test_handoff_rejects_tampered_component_evidence() -> None:
    document = _load_handoff()
    components = document["components"]
    assert isinstance(components, list)
    components[1]["ci"]["run_id"] += 1

    with pytest.raises(HandoffValidationError, match="component_map_sha256"):
        validate_s0_handoff(document)


def test_handoff_rejects_paid_compute_authorization() -> None:
    document = _load_handoff()
    authorization = document["authorization"]
    assert isinstance(authorization, dict)
    authorization["paid_compute_authorized"] = True

    with pytest.raises(HandoffValidationError, match="paid compute"):
        validate_s0_handoff(document)


def test_new_exact_green_d06_can_transition_handoff_to_ready_local_free() -> None:
    document = copy.deepcopy(_load_handoff())
    components = document["components"]
    assert isinstance(components, list)
    d06 = next(item for item in components if item["lane"] == "D06")
    d06["source_sha"] = "a" * 40
    d06["disposition"] = "accepted"
    d06["ci"] = {"run_id": 99999999999, "conclusion": "success"}
    d06.pop("hold_reason")
    d06["contains_behavioral_weights"] = False
    d06["contains_foreign_pretrained_weights"] = False
    document["blockers"] = []
    document["handoff_state"] = "READY_LOCAL_FREE"
    _refresh_component_hash(document)

    result = validate_s0_handoff(document)

    assert result["execution_ready"] is True
    assert result["held_lanes"] == ()
    assert result["accepted_lanes"] == ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
    assert result["promotion_allowed"] is False


def test_handoff_can_never_self_promote_even_when_execution_ready() -> None:
    document = _load_handoff()
    document["promotion_allowed"] = True

    with pytest.raises(HandoffValidationError, match="self-authorize promotion"):
        validate_s0_handoff(document)
