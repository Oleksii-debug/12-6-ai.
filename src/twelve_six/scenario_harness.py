"""Deterministic, fixture-safe Plan 8 scenario and coverage contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from twelve_six.sil_qualification import load_sil_scenario, require_exact_clean_git_state
from twelve_six_agent_runtime.fixture_integration import (
    EchoGatewayFixture,
    FakeHostTool,
    FixtureRuntime,
)
from twelve_six_agent_runtime.fixture_integration import (
    Scenario as AgentFixtureScenario,
)

_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_LEVELS = ("Q0", "Q1", "Q2", "Q3", "Q4")
ScenarioAdapter = Callable[[Path], object]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate scenario JSON key")
        result[key] = value
    return result


def parse_scenario_bytes(raw: bytes) -> Scenario:
    """Parse a bounded, unambiguous versioned scenario manifest."""
    if type(raw) is not bytes or len(raw) > 65_536:
        raise ValueError("scenario manifest must be bounded bytes")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid scenario JSON") from exc
    if type(value) is not dict or set(value) != {
        "schema_version", "scenario_id", "capability_ids", "prerequisites", "steps", "required_level"
    }:
        raise ValueError("noncanonical scenario manifest")
    if type(value["capability_ids"]) is not list or type(value["prerequisites"]) is not list or (
        type(value["steps"]) is not list or len(value["steps"]) > 64
    ):
        raise ValueError("invalid scenario arrays")
    steps: list[ScenarioStep] = []
    for item in value["steps"]:
        if type(item) is not dict or set(item) != {"step_id", "adapter_id", "expected_sha256"}:
            raise ValueError("noncanonical scenario step")
        steps.append(ScenarioStep(**item))
    return Scenario(
        schema_version=value["schema_version"], scenario_id=value["scenario_id"],
        capability_ids=tuple(value["capability_ids"]),
        prerequisites=tuple(value["prerequisites"]), steps=tuple(steps),
        required_level=value["required_level"],
    )


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
        if type(self.schema_version) is not str or self.schema_version != "12-6.plan8-scenario.v1":
            raise ValueError("unsupported scenario version")
        _identifier(self.scenario_id)
        if type(self.required_level) is not str or self.required_level not in _LEVELS:
            raise ValueError("unknown evidence level")
        for values in (self.capability_ids, self.prerequisites):
            if type(values) is not tuple or not 1 <= len(values) <= 64:
                raise ValueError("scenario lists must contain 1..64 identifiers")
            for value in values:
                _identifier(value)
            if tuple(sorted(set(values))) != values:
                raise ValueError("scenario lists must be unique and sorted")
        if type(self.steps) is not tuple or not 1 <= len(self.steps) <= 64:
            raise ValueError("scenario must contain 1..64 steps")
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
    repo_root: str | Path,
    source_sha: str,
    available_prerequisites: frozenset[str],
    adapters: Mapping[str, ScenarioAdapter] | None = None,
) -> dict[str, object]:
    """Run named adapters against a clean, exact checkout and issue a Q0 receipt."""
    if type(scenario) is not Scenario:
        raise ValueError("scenario must have exact type")
    scenario.__post_init__()
    if type(source_sha) is not str or not _GIT_SHA.fullmatch(source_sha):
        raise ValueError("exact source SHA required")
    if type(available_prerequisites) is not frozenset:
        raise ValueError("prerequisites must be immutable")
    if not set(scenario.prerequisites).issubset(available_prerequisites):
        raise ValueError("scenario prerequisites missing")
    root = Path(repo_root).resolve(strict=True)
    require_exact_clean_git_state(root, source_sha)
    if adapters is None:
        adapters = fixture_journey_adapters()
    observations: list[dict[str, str]] = []
    for step in scenario.steps:
        adapter = adapters.get(step.adapter_id)
        if adapter is None or not callable(adapter):
            raise ValueError("scenario adapter unavailable")
        try:
            actual = _digest(adapter(root))
        except Exception as exc:
            raise RuntimeError("scenario adapter failed: " + step.step_id) from exc
        require_exact_clean_git_state(root, source_sha)
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
    scenarios: tuple[Scenario, ...],
    receipts: tuple[dict[str, object], ...],
    *,
    expected_source_sha: str,
) -> dict[str, dict[str, object]]:
    """Report Q0 observations and remaining scenario gaps for one exact candidate."""
    if type(expected_source_sha) is not str or not _GIT_SHA.fullmatch(expected_source_sha):
        raise ValueError("exact source SHA required")
    if any(type(scenario) is not Scenario for scenario in scenarios):
        raise ValueError("scenario must have exact type")
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ID")
    scenario_by_sha = {scenario.identity_sha256(): scenario for scenario in scenarios}
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
        scenario = scenario_by_sha.get(receipt["scenario_sha256"])
        if scenario is None:
            raise ValueError("receipt references unknown scenario")
        if type(receipt["source_sha"]) is not str or receipt["source_sha"] != expected_source_sha:
            raise ValueError("receipt source SHA differs from expected candidate")
        expected = [
            {"step_id": step.step_id, "observation_sha256": step.expected_sha256}
            for step in scenario.steps
        ]
        if receipt["observations"] != expected:
            raise ValueError("receipt observations differ from scenario")
        if receipt["scenario_sha256"] in receipt_ids:
            raise ValueError("duplicate scenario receipt")
        receipt_ids.add(receipt["scenario_sha256"])
    result: dict[str, dict[str, object]] = {}
    for identity, scenario in scenario_by_sha.items():
        observed = identity in receipt_ids
        for capability in scenario.capability_ids:
            current = result.setdefault(capability, {
                "required_level": scenario.required_level,
                "q0_scenarios": [],
                "missing_scenarios": [],
            })
            if current["required_level"] != scenario.required_level:
                raise ValueError("conflicting capability evidence requirements")
            if observed:
                current["q0_scenarios"].append(scenario.scenario_id)
            if not observed or scenario.required_level != "Q0":
                current["missing_scenarios"].append(scenario.scenario_id)
    for current in result.values():
        current["q0_scenarios"].sort()
        current["missing_scenarios"].sort()
        current["missing_level"] = current["required_level"] if current["missing_scenarios"] else None
    return dict(sorted(result.items()))


_FIXTURE_JOURNEYS = {
    "agent-goal": ("tests/fixtures/plan5_agent_runtime_scenario_v1.json", "12-6.plan5-agent-fixture.v1"),
    "reserved-eval": ("configs/data/plan2_reserved_eval_fixture_v1.json", "12-6.plan2-reserved-eval-fixture.v1"),
    "sil-journey": ("configs/control/sil_scenario_v1.json", 1),
}


def _run_agent_fixture(raw: bytes) -> dict[str, object]:
    scenario = AgentFixtureScenario.from_json(raw.decode("utf-8").removesuffix("\n"))
    gateway = EchoGatewayFixture(scenario.observed_value)
    host = FakeHostTool()
    with TemporaryDirectory(prefix="plan8-agent-fixture-") as directory:
        runtime = FixtureRuntime(directory, scenario, gateway, host)
        finished = runtime.run()
        restarted = FixtureRuntime(directory, scenario, gateway, host)
        replayed = restarted.run()
        memory = restarted.memory.history()[0]
        effect = finished.pending_effects[0]
        return {
            "task_id": finished.task_id,
            "terminal_step_id": finished.step_id,
            "effect_status": effect.status,
            "receipt_id": effect.receipt_id,
            "memory_evidence_id": memory.evidence_id,
            "memory_content_sha256": _digest(memory.content),
            "memory_revision": restarted.memory.revision(),
            "host_dispatches": host.dispatches,
            "gateway_proposals": gateway.calls,
            "restart_same_snapshot": replayed == finished,
        }


def fixture_journey_adapters() -> dict[str, ScenarioAdapter]:
    """Build named Q0 adapters that receive the runner's verified checkout."""
    adapters: dict[str, ScenarioAdapter] = {}
    for adapter_id, (relative_path, schema) in _FIXTURE_JOURNEYS.items():
        def observe(root: Path, path: str = relative_path, expected: str | int = schema,
                    adapter: str = adapter_id) -> object:
            target = root / path
            if target.is_symlink() or not target.is_file():
                raise ValueError("fixture is missing or a symlink")
            if not target.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
                raise ValueError("fixture escapes the checkout")
            with target.open("rb") as stream:
                raw = stream.read(65_537)
            if not raw or len(raw) > 65_536:
                raise ValueError("fixture size invalid")
            value = json.loads(
                raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
            )
            key = "schema" if path.startswith("tests/") else "schema_version"
            if (type(value) is not dict or type(value.get(key)) is not type(expected)
                    or value.get(key) != expected):
                raise ValueError("fixture schema mismatch")
            if adapter == "reserved-eval" and (
                value.get("training_allowed") is not False
                or value.get("tokenizer_fit_allowed") is not False
            ):
                raise ValueError("reserved evaluation fixture lost its firewall")
            observation: dict[str, object] = {
                "fixture": path,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "schema": expected,
            }
            if adapter == "agent-goal":
                observation["runtime"] = _run_agent_fixture(raw)
            if adapter == "sil-journey":
                load_sil_scenario(target)
            return observation

        adapters[adapter_id] = observe
    return adapters
