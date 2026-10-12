"""Deterministic, fixture-safe Plan 8 scenario and coverage contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping

_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_LEVELS = ("Q0", "Q1", "Q2", "Q3", "Q4")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _identifier(value: object) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("scenario identifier is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    step_id: str
    adapter_id: str
    expected_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.step_id)
        _identifier(self.adapter_id)
        if type(self.expected_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_sha256
        ):
            raise ValueError("expected observation must be a SHA-256")


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    capability_ids: tuple[str, ...]
    prerequisites: tuple[str, ...]
    steps: tuple[ScenarioStep, ...]
    required_level: str
    schema_version: str = "12-6.plan8-scenario.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "12-6.plan8-scenario.v1":
            raise ValueError("unsupported scenario version")
        _identifier(self.scenario_id)
        if self.required_level not in _LEVELS:
            raise ValueError("unknown evidence level")
        for values in (self.capability_ids, self.prerequisites):
            if type(values) is not tuple or not values or tuple(sorted(set(values))) != values:
                raise ValueError("scenario lists must be nonempty, unique, sorted tuples")
            for value in values:
                _identifier(value)
        if type(self.steps) is not tuple or not self.steps:
            raise ValueError("scenario needs steps")
        for step in self.steps:
            if type(step) is not ScenarioStep:
                raise ValueError("scenario step must have exact type")
            step.__post_init__()
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate step")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "capability_ids": list(self.capability_ids),
            "prerequisites": list(self.prerequisites),
            "steps": [
                {"step_id": s.step_id, "adapter_id": s.adapter_id, "expected_sha256": s.expected_sha256}
                for s in self.steps
            ],
            "required_level": self.required_level,
        }

    def identity_sha256(self) -> str:
        return _digest(self.to_dict())


def run_scenario(
    scenario: Scenario,
    *,
    source_sha: str,
    available_prerequisites: frozenset[str],
    adapters: Mapping[str, Callable[[], object]],
) -> dict[str, object]:
    """Run bounded named adapters; failures never become passing observations."""
    scenario.__post_init__()
    if type(source_sha) is not str or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("exact source SHA required")
    if type(available_prerequisites) is not frozenset:
        raise ValueError("prerequisites must be immutable")
    if not set(scenario.prerequisites).issubset(available_prerequisites):
        raise ValueError("scenario prerequisites missing")
    observations: list[dict[str, str]] = []
    for step in scenario.steps:
        adapter = adapters.get(step.adapter_id)
        if adapter is None or not callable(adapter):
            raise ValueError("scenario adapter unavailable")
        try:
            actual = _digest(adapter())
        except Exception as exc:
            raise RuntimeError("scenario adapter failed: " + step.step_id) from exc
        if actual != step.expected_sha256:
            raise ValueError("scenario observation mismatch: " + step.step_id)
        observations.append({"step_id": step.step_id, "observation_sha256": actual})
    receipt: dict[str, object] = {
        "schema_version": "12-6.plan8-scenario-receipt.v1",
        "scenario_sha256": scenario.identity_sha256(),
        "source_sha": source_sha,
        "evidence_level": "Q0",
        "environment": "DETERMINISTIC_FIXTURE",
        "observations": observations,
    }
    receipt["identity_sha256"] = _digest(receipt)
    return receipt


def coverage_map(
    scenarios: tuple[Scenario, ...], receipts: tuple[dict[str, object], ...]
) -> dict[str, dict[str, object]]:
    """Report proven fixture coverage and explicit gaps, never promote Q0 to HIL."""
    receipt_ids: set[str] = set()
    for receipt in receipts:
        if type(receipt) is not dict or set(receipt) != {
            "schema_version", "scenario_sha256", "source_sha", "evidence_level",
            "environment", "observations", "identity_sha256",
        }:
            raise ValueError("noncanonical scenario receipt")
        if receipt["schema_version"] != "12-6.plan8-scenario-receipt.v1" or (
            receipt["evidence_level"] != "Q0" or receipt["environment"] != "DETERMINISTIC_FIXTURE"
        ):
            raise ValueError("unsupported scenario receipt evidence")
        if _digest({key: value for key, value in receipt.items() if key != "identity_sha256"}) != receipt["identity_sha256"]:
            raise ValueError("scenario receipt identity mismatch")
        receipt_ids.add(receipt["scenario_sha256"])
    result: dict[str, dict[str, object]] = {}
    for scenario in scenarios:
        scenario.__post_init__()
        for capability in scenario.capability_ids:
            current = result.setdefault(capability, {"required_level": scenario.required_level, "q0_scenarios": []})
            if current["required_level"] != scenario.required_level:
                raise ValueError("conflicting capability evidence requirements")
            if scenario.identity_sha256() in receipt_ids:
                current["q0_scenarios"].append(scenario.scenario_id)
    for current in result.values():
        current["q0_scenarios"].sort()
        current["missing_level"] = None if current["required_level"] == "Q0" and current["q0_scenarios"] else current["required_level"]
    return dict(sorted(result.items()))
