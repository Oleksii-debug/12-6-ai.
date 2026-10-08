"""Local-free, model-neutral Plan-5 Section-11 runtime integration fixture.

This is a deterministic integration driver, not a second tool/task/scheduler
authority, production ModelGateway, external executor, or credential provider.
Only existing TaskStore, ToolRegistry and MemoryStore own durable state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .context import ContextEntry, build_context, canonical_bytes
from .memory import MemoryRecord, MemoryStore
from .security import EffectGrant, prepare_secured_tool
from .task_state import PendingEffect, StateError, TaskSnapshot, TaskStore, _digest, _json
from .tools import (
    DESCRIPTOR_VERSION, ToolDescriptor, ToolRegistry, make_result,
)
from .world import WorldFact, WorldModel

SCHEMA = "12-6.plan5-agent-fixture.v1"
TOOL_ID = "fixture.observe"
INPUT = {
    "type": "object",
    "properties": {"value": {"type": "string", "maxLength": 256}},
    "required": ["value"], "additionalProperties": False,
}
OUTPUT = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"}, "value": {"type": "string", "maxLength": 256},
    },
    "required": ["ok", "value"], "additionalProperties": False,
}


class FixtureError(ValueError):
    """Failed integration authority, incompatible scenario, or unknown effect."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    owner_goal: str
    observed_value: str

    def validate(self) -> None:
        for value in (self.scenario_id, self.owner_goal, self.observed_value):
            if (not isinstance(value, str) or not value.strip()
                    or len(value.encode("utf-8")) > 256):
                raise FixtureError("invalid bounded scenario")
        if not self.scenario_id.replace("-", "").replace("_", "").isalnum():
            raise FixtureError("unsafe scenario identity")

    @property
    def digest(self) -> str:
        self.validate()
        return _digest(_json({"schema": SCHEMA, **asdict(self)}))

    @classmethod
    def from_json(cls, raw: str) -> Scenario:
        try:
            obj = json.loads(raw)
            if (not isinstance(obj, dict)
                    or set(obj) != {"schema", "scenario_id", "owner_goal", "observed_value"}
                    or obj.pop("schema") != SCHEMA):
                raise FixtureError("invalid scenario schema")
            result = cls(**obj)
            result.validate()
            if _json({"schema": SCHEMA, **asdict(result)}) != raw:
                raise FixtureError("noncanonical scenario bytes")
            return result
        except (TypeError, KeyError, ValueError) as exc:
            raise FixtureError("invalid scenario") from exc


@dataclass(frozen=True)
class GatewayInput:
    context_sha256: str
    world_sha256: str
    memory_revision: int
    task_id: str


@dataclass(frozen=True)
class GatewayProposal:
    tool_id: str
    request: dict[str, object]


class ModelGatewayFixture(Protocol):
    def propose(self, request: GatewayInput) -> GatewayProposal:
        """Untrusted proposal; never a permission grant."""


class EchoGatewayFixture:
    """Replaceable deterministic fixture model; produces no host authority."""

    def __init__(self, observed_value: str) -> None:
        self.observed_value = observed_value
        self.calls = 0

    def propose(self, request: GatewayInput) -> GatewayProposal:
        self.calls += 1
        return GatewayProposal(TOOL_ID, {"value": self.observed_value})


class FakeHostTool:
    """Fixture-only host adapter, no network, file mutation or OS effects."""

    def __init__(self) -> None:
        self.dispatches = 0

    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        self.dispatches += 1
        return {"ok": True, "value": request["value"]}


class FixtureRuntime:
    """Reproducible E2E host harness over the accepted Plan-5 authorities."""

    def __init__(
        self, root: str | Path, scenario: Scenario, gateway: ModelGatewayFixture,
        host_tool: FakeHostTool, *, verify_host: Callable[[str], bool] | None = None,
    ) -> None:
        scenario.validate()
        self.scenario = scenario
        self.gateway = gateway
        self.host_tool = host_tool
        self.verify_host = verify_host or (lambda _: True)
        root = Path(root)
        if not root.is_dir():
            raise FixtureError("fixture root must already exist")
        self.tasks = TaskStore(root / "tasks.sqlite")
        self.memory = MemoryStore(root / "memory.sqlite")
        self.world = WorldModel()
        self.world.apply(
            WorldFact(
                fact_id="fixture-observation", subject_id="fixture-environment",
                subject_kind="environment", attribute="state", state="observed",
                value=scenario.observed_value, source_id="host-fixture",
                evidence_id="host-world-receipt", origin_kind="tool", recorded_at=1,
            ),
            expected_revision=0,
        )
        self.registry = ToolRegistry(self.tasks)
        self.registry.register(
            ToolDescriptor(
                tool_id=TOOL_ID, version=DESCRIPTOR_VERSION,
                capabilities=("fixture",), permissions=("fixture.observe",),
                side_effect="read", input_schema=INPUT, output_schema=OUTPUT,
                available=True,
            ),
            verify_descriptor=lambda _: self.verify_host("descriptor") is True,
        )

    @property
    def effect_id(self) -> str:
        return "effect-" + self.scenario.digest[:24]

    @property
    def checkpoint_id(self) -> str:
        return "fixture-" + self.scenario.digest

    def _context_and_model(self) -> GatewayProposal:
        entries = (
            ContextEntry(
                "trusted-goal", "owner-fixture", "owner", 1, 100,
                self.scenario.owner_goal, True,
            ),
            ContextEntry(
                "trusted-observation", "host-fixture", "observation", 2, 50,
                self.scenario.observed_value,
            ),
        )
        context = build_context(entries, max_bytes=4096, max_entries=4)
        observed = self.world.lookup("fixture-environment", "state", now=2)
        if observed.state != "observed" or observed.value != self.scenario.observed_value:
            raise FixtureError("world fixture is not observed")
        snapshot = GatewayInput(
            context_sha256=hashlib.sha256(canonical_bytes(context)).hexdigest(),
            world_sha256=hashlib.sha256(self.world.export().encode("utf-8")).hexdigest(),
            memory_revision=self.memory.revision(), task_id=self.scenario.scenario_id,
        )
        proposal = self.gateway.propose(snapshot)
        if (not isinstance(proposal, GatewayProposal)
                or proposal.tool_id != TOOL_ID
                or proposal.request != {"value": self.scenario.observed_value}):
            raise FixtureError("model proposal cannot change verified host intent")
        return proposal

    def _load_or_reserve(self) -> TaskSnapshot:
        try:
            state = self.tasks.load(self.scenario.scenario_id)
        except StateError as exc:
            if str(exc) != "unknown task_id":
                raise
            state = self.tasks.create(
                task_id=self.scenario.scenario_id, plan_id="plan5",
                step_id="fixture-start",
            )
        if state.revision == 0:
            self._context_and_model()
            state = self.tasks.checkpoint(
                state.task_id, expected_epoch=state.control_epoch,
                expected_revision=state.revision, step_id="fixture-reserved",
                checkpoint_id=self.checkpoint_id,
                pending_effects=(PendingEffect(self.effect_id, "fake host observation"),),
            )
        if (state.plan_id != "plan5" or state.checkpoint_id not in (
                self.checkpoint_id, self.checkpoint_id + "-done")
                or len(state.pending_effects) != 1
                or state.pending_effects[0].effect_id != self.effect_id):
            raise FixtureError("fixture identity or task lineage changed")
        return state

    def _prepare(self, state: TaskSnapshot):
        _, digest = self.registry.get(TOOL_ID)
        request = {"value": self.scenario.observed_value}
        grant = EffectGrant(
            issuer="host", evidence_id="host-fixture-grant",
            task_id=state.task_id, effect_id=self.effect_id, tool_id=TOOL_ID,
            descriptor_digest=digest, request_sha256=_digest(_json(request)),
            permissions=("fixture.observe",), control_epoch=state.control_epoch,
            issued_at_ms=1000, expires_at_ms=2000,
        )
        return prepare_secured_tool(
            self.registry, grant, request=request,
            expected_revision=state.revision, now_ms=1500,
            verify_grant=lambda *_: self.verify_host("grant") is True,
            verify_payload_safety=lambda *_: self.verify_host("payload") is True,
            authorize_host=lambda *_: self.verify_host("permission") is True,
        )

    def _finish(self, state: TaskSnapshot) -> TaskSnapshot:
        if state.pending_effects[0].status != "resolved":
            raise FixtureError("cannot complete unverified or unknown effect")
        if not self.verify_host("completion"):
            raise FixtureError("trusted host refused completion")
        records = self.memory.history()
        memory_id = self.scenario.scenario_id + "-observation"
        if not any(item.memory_id == memory_id for item in records):
            self.memory.append(
                MemoryRecord(
                    memory_id=memory_id, kind="episodic",
                    content="Observed: " + self.scenario.observed_value,
                    source_id="fixture-host", source_kind="tool",
                    evidence_id=state.pending_effects[0].receipt_id,
                    confidence_ppm=1_000_000, created_at=2,
                ),
                expected_revision=len(records),
            )
        if state.step_id != "fixture-done":
            state = self.tasks.checkpoint(
                state.task_id, expected_epoch=state.control_epoch,
                expected_revision=state.revision, step_id="fixture-done",
                checkpoint_id=self.checkpoint_id + "-done",
                pending_effects=state.pending_effects,
            )
        return state

    def run(self, *, stop_after: str | None = None) -> TaskSnapshot:
        """The only fake dispatch is after durable issue; never retry unknown."""
        if stop_after not in (None, "reserved", "issued", "resolved"):
            raise FixtureError("unknown interruption point")
        state = self._load_or_reserve()
        if state.step_id == "fixture-done":
            return state
        effect = state.pending_effects[0]
        if effect.status == "unknown":
            raise FixtureError("unknown effect requires independent receipt; no retry")
        if effect.status == "resolved":
            return self._finish(state)
        if stop_after == "reserved":
            return state
        call = self._prepare(state)
        state = self.tasks.issue_effect(
            state.task_id, self.effect_id, expected_epoch=state.control_epoch,
            expected_revision=state.revision,
        )
        if stop_after == "issued":
            return state
        output = self.host_tool.dispatch({"value": self.scenario.observed_value})
        result = make_result(
            call, receipt_id="fixture-verified-receipt",
            evidence_id="fixture-result-proof", outcome="success", output=output,
        )
        state = self.registry.accept_result(
            call, result,
            verify_call=lambda _: self.verify_host("call") is True,
            verify_result=lambda _, record: (
                self.verify_host("result") is True
                and record.output_json == _json({
                    "ok": True, "value": self.scenario.observed_value,
                })
            ),
        )
        if stop_after == "resolved":
            return state
        return self._finish(state)

    def reconcile_unknown(
        self, *, receipt_id: str, verify_external_receipt: Callable[[str], bool],
    ) -> TaskSnapshot:
        """Host-only recovery proof, never another dispatch or model call."""
        state = self._load_or_reserve()
        if state.pending_effects[0].status != "unknown":
            raise FixtureError("recovery only allowed for unknown effects")
        if (receipt_id != "fixture-verified-receipt"
                or not callable(verify_external_receipt)
                or verify_external_receipt(receipt_id) is not True):
            raise FixtureError("untrusted or mismatched external receipt")
        return self.tasks.resolve_effect(
            state.task_id, self.effect_id, receipt_id,
            expected_epoch=state.control_epoch, expected_revision=state.revision,
        )
