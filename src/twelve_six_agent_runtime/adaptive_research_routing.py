"""Reference adaptive research route planner, not an executor or model provider.

A trusted host supplies its catalog, resource reading and independently
verified outcomes. Selection NEVER loads a model, invokes an API, authorizes
an external effect or turns model agreement into factual verification.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal

SCHEMA = "12-6.adaptive-research-route.v1"


class RouteError(ValueError):
    """Invalid route, forged result admission or incompatible snapshot."""


def _name(value: object, label: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 128 or not value.strip():
        raise RouteError(f"{label} must be bounded nonempty text")


def _number(value: object, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise RouteError(f"{label} must be an integer >= {minimum}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Specialist:
    node_id: str
    capabilities: tuple[str, ...]
    provider: Literal["local", "external", "deterministic"]
    model_sha256: str
    memory_mb: int
    cpu_units: int
    max_charge_microunits: int
    priority: int
    successors: tuple[str, ...]

    def validate(self) -> None:
        _name(self.node_id, "node_id")
        if (type(self.capabilities) is not tuple or not self.capabilities
                or len(self.capabilities) > 32 or len(set(self.capabilities)) != len(self.capabilities)):
            raise RouteError("capabilities must be unique bounded tuple")
        if type(self.successors) is not tuple or len(self.successors) > 32 or len(set(self.successors)) != len(self.successors):
            raise RouteError("successors must be unique bounded tuple")
        for x in self.capabilities + self.successors:
            _name(x, "capability or successor")
        if self.provider not in ("local", "external", "deterministic"):
            raise RouteError("unsupported provider")
        if type(self.model_sha256) is not str or len(self.model_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.model_sha256):
            raise RouteError("invalid pinned model identity")
        _number(self.memory_mb, "memory_mb", 1)
        _number(self.cpu_units, "cpu_units", 1)
        _number(self.max_charge_microunits, "max_charge_microunits")
        _number(self.priority, "priority")
        if self.priority > 100:
            raise RouteError("priority exceeds policy")


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    max_calls: int
    max_visits_per_node: int
    max_no_progress: int
    cost_limit_microunits: int
    allow_external: bool = False
    allow_paid: bool = False

    def validate(self) -> None:
        for key in ("max_calls", "max_visits_per_node", "max_no_progress"):
            _number(getattr(self, key), key, 1)
        if self.max_calls > 128 or self.max_visits_per_node > self.max_calls or self.max_no_progress > self.max_calls:
            raise RouteError("unbounded routing policy")
        _number(self.cost_limit_microunits, "cost_limit_microunits")
        if type(self.allow_external) is not bool or type(self.allow_paid) is not bool:
            raise RouteError("provider admission flags must be booleans")


@dataclass(frozen=True, slots=True)
class ResourceReading:
    free_memory_mb: int
    available_cpu_units: int

    def validate(self) -> None:
        _number(self.free_memory_mb, "free_memory_mb")
        _number(self.available_cpu_units, "available_cpu_units")


@dataclass(frozen=True, slots=True)
class RouteState:
    task_id: str
    required_capability: str
    history: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    spent_microunits: int = 0
    no_progress: int = 0
    revision: int = 0

    def validate(self) -> None:
        _name(self.task_id, "task_id")
        _name(self.required_capability, "required_capability")
        if type(self.history) is not tuple or type(self.receipt_ids) is not tuple or len(self.history) != len(self.receipt_ids) or len(self.history) > 128:
            raise RouteError("malformed route history")
        for item in self.history + self.receipt_ids:
            _name(item, "route history identity")
        if len(set(self.receipt_ids)) != len(self.receipt_ids):
            raise RouteError("duplicate external receipt")
        for key in ("spent_microunits", "no_progress", "revision"):
            _number(getattr(self, key), key)
        if self.revision != len(self.history) or self.no_progress > len(self.history):
            raise RouteError("incoherent route revision")

    def snapshot(self) -> tuple[str, str]:
        """Canonical state bytes and integrity checksum; not a signature."""
        self.validate()
        raw = json.dumps({"schema": SCHEMA, "state": asdict(self)}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return raw, _digest(raw)

    @classmethod
    def from_snapshot(cls, raw: str, checksum: str) -> RouteState:
        if type(raw) is not str or type(checksum) is not str or len(raw) > 65536 or _digest(raw) != checksum:
            raise RouteError("snapshot checksum mismatch")
        try:
            data = json.loads(raw)
            if type(data) is not dict or set(data) != {"schema", "state"} or data["schema"] != SCHEMA:
                raise RouteError("incompatible routing snapshot")
            state = data["state"]
            if type(state) is not dict or set(state) != set(cls.__dataclass_fields__):
                raise RouteError("invalid route state fields")
            state["history"] = tuple(state["history"])
            state["receipt_ids"] = tuple(state["receipt_ids"])
            result = cls(**state)
            result.validate()
            if result.snapshot()[0] != raw:
                raise RouteError("noncanonical snapshot")
            return result
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RouteError("malformed routing snapshot") from exc


@dataclass(frozen=True, slots=True)
class RouteDecision:
    status: Literal["SELECTED", "WAIT_RESOURCES", "NO_ELIGIBLE_ROUTE", "BUDGET_EXHAUSTED", "STOP_NO_PROGRESS"]
    task_id: str
    revision: int
    node_id: str | None


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    task_id: str
    node_id: str
    receipt_id: str
    evidence_sha256: str
    charged_microunits: int
    next_capability: str
    measurable_progress: bool

    def validate(self) -> None:
        for key in ("task_id", "node_id", "receipt_id", "next_capability"):
            _name(getattr(self, key), key)
        if type(self.evidence_sha256) is not str or len(self.evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.evidence_sha256):
            raise RouteError("invalid evidence digest")
        _number(self.charged_microunits, "charged_microunits")
        if type(self.measurable_progress) is not bool:
            raise RouteError("untyped progress signal")


def _catalog(members: tuple[Specialist, ...]) -> dict[str, Specialist]:
    if type(members) is not tuple or not 1 <= len(members) <= 64:
        raise RouteError("catalog capacity invalid")
    nodes: dict[str, Specialist] = {}
    for member in members:
        if type(member) is not Specialist:
            raise RouteError("non-specialist catalog entry")
        member.validate()
        if member.node_id in nodes:
            raise RouteError("duplicate specialist ID")
        nodes[member.node_id] = member
    for member in members:
        if any(next_id not in nodes for next_id in member.successors):
            raise RouteError("successor points outside trusted catalog")
    return nodes


def select_next(state: RouteState, members: tuple[Specialist, ...],
                policy: RoutePolicy, resources: ResourceReading) -> RouteDecision:
    """Deterministic one-model-at-a-time selection; no model is invoked."""
    state.validate()
    policy.validate()
    resources.validate()
    nodes = _catalog(members)
    if any(node_id not in nodes for node_id in state.history):
        raise RouteError("route refers to unknown prior model")
    if state.spent_microunits > policy.cost_limit_microunits:
        raise RouteError("stored spend exceeds policy")
    if len(state.history) >= policy.max_calls:
        return RouteDecision("BUDGET_EXHAUSTED", state.task_id, state.revision, None)
    if state.no_progress >= policy.max_no_progress:
        return RouteDecision("STOP_NO_PROGRESS", state.task_id, state.revision, None)
    allowed = nodes if not state.history else {x: nodes[x] for x in nodes[state.history[-1]].successors}
    visits = Counter(state.history)
    capacity_blocked = False
    selected: list[Specialist] = []
    for member in allowed.values():
        if state.required_capability not in member.capabilities:
            continue
        if visits[member.node_id] >= policy.max_visits_per_node:
            continue
        if member.provider == "external" and not policy.allow_external:
            continue
        if member.max_charge_microunits and not policy.allow_paid:
            continue
        if state.spent_microunits + member.max_charge_microunits > policy.cost_limit_microunits:
            continue
        if member.memory_mb > resources.free_memory_mb or member.cpu_units > resources.available_cpu_units:
            capacity_blocked = True
            continue
        selected.append(member)
    if selected:
        best = min(selected, key=lambda x: (visits[x.node_id], -x.priority, x.max_charge_microunits, x.memory_mb, x.node_id))
        return RouteDecision("SELECTED", state.task_id, state.revision, best.node_id)
    return RouteDecision("WAIT_RESOURCES" if capacity_blocked else "NO_ELIGIBLE_ROUTE",
                         state.task_id, state.revision, None)


def accept_outcome(state: RouteState, decision: RouteDecision, outcome: ModelOutcome,
                   members: tuple[Specialist, ...], policy: RoutePolicy,
                   verifier: Callable[[ModelOutcome], bool]) -> RouteState:
    """Host-only evidence admission; not a transport or claim-truth verifier."""
    state.validate()
    policy.validate()
    nodes = _catalog(members)
    if decision.status != "SELECTED" or decision.node_id not in nodes or decision.task_id != state.task_id or decision.revision != state.revision:
        raise RouteError("selection is stale or not selected")
    outcome.validate()
    if outcome.task_id != state.task_id or outcome.node_id != decision.node_id:
        raise RouteError("result/decision identity mismatch")
    if outcome.receipt_id in state.receipt_ids:
        raise RouteError("replayed evidence receipt")
    if outcome.charged_microunits > nodes[decision.node_id].max_charge_microunits:
        raise RouteError("actual charge exceeds admission")
    if not callable(verifier) or verifier(outcome) is not True:
        raise RouteError("independent host evidence admission failed")
    updated = replace(state, required_capability=outcome.next_capability,
                      history=(*state.history, outcome.node_id),
                      receipt_ids=(*state.receipt_ids, outcome.receipt_id),
                      spent_microunits=state.spent_microunits + outcome.charged_microunits,
                      no_progress=0 if outcome.measurable_progress else state.no_progress + 1,
                      revision=state.revision + 1)
    updated.validate()
    if updated.spent_microunits > policy.cost_limit_microunits or updated.revision > policy.max_calls:
        raise RouteError("outcome overspent policy")
    return updated
