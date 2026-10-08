"""Plan 5 / Section 8: semantic browser and computer-control reference boundary.

No browser, Win32, OS input or Nika product runtime is imported or executed.
The trusted host owns actual observations, permissions and postcondition proofs.
The existing Section-7 ToolRegistry and Section-2 TaskStore remain authorities.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from .task_state import TaskSnapshot, _digest, _id, _json, _uint
from .tools import (
    DESCRIPTOR_VERSION, ToolCall, ToolDescriptor, ToolRegistry, make_result,
)

BACKENDS = ("dom", "ax", "uia")
METHODS = ("semantic", "coordinate", "vision")
ACTIONS = ("activate", "focus", "set_value")
Backend = Literal["dom", "ax", "uia"]
Method = Literal["semantic", "coordinate", "vision"]
ACTION_TOOL = "ui.perform"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class UIControlError(ValueError):
    """Unsupported surface, ambiguous/stale target or unverified UI effect."""


def _identity(value: object) -> str:
    return _digest(_json(value))


def _word(value: object, label: str) -> str:
    try:
        _id(value, label)
    except ValueError as exc:
        raise UIControlError(f"invalid {label}") from exc
    return value


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise UIControlError(f"invalid {label}")
    return value


def _closed(\n    properties: dict[str, dict[str, object]], required: tuple[str, ...],\n) -> dict[str, object]:
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


def _string(max_length: int) -> dict[str, object]:
    return {"type": "string", "maxLength": max_length}


def ui_tool_descriptor() -> ToolDescriptor:
    """A single versioned side-effect contract, registered by a trusted host."""
    return ToolDescriptor(
        tool_id=ACTION_TOOL, version=DESCRIPTOR_VERSION,
        capabilities=("accessibility-tree", "browser-dom", "windows-uia"),
        permissions=("ui.control",), side_effect="external", available=True,
        input_schema=_closed({
            "snapshot_digest": _string(64), "target_digest": _string(64),
            "action": _string(32), "expected_state": _string(256),
            "method": _string(16), "fallback_evidence_id": _string(256),
        }, ("snapshot_digest", "target_digest", "action", "expected_state",\n            "method", "fallback_evidence_id")),
        output_schema=_closed({
            "target_digest": _string(64), "state": _string(256),
            "ok": {"type": "boolean"},
        }, ("target_digest", "state", "ok")),
    )


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    role: str
    name: str
    state: str

    def validate(self) -> None:
        for field in ("node_id", "role", "name", "state"):
            _word(getattr(self, field), field)
        if len(_json(asdict(self)).encode("utf-8")) > 4096:
            raise UIControlError("oversized accessibility node")

    @property
    def digest(self) -> str:
        self.validate()
        return _identity(asdict(self))


@dataclass(frozen=True)
class UISnapshot:
    backend: Backend
    application_id: str
    view_id: str
    generation: int
    tree_digest: str
    evidence_id: str
    nodes: tuple[SemanticNode, ...]

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise UIControlError("unsupported semantic backend")
        for field in ("application_id", "view_id", "evidence_id"):
            _word(getattr(self, field), field)
        _uint(self.generation, "generation")
        _hex(self.tree_digest, "tree digest")
        if type(self.nodes) is not tuple or len(self.nodes) > 1024:
            raise UIControlError("invalid or unbounded semantic tree")
        seen: set[str] = set()
        for node in self.nodes:
            if not isinstance(node, SemanticNode):
                raise UIControlError("typed semantic nodes required")
            node.validate()
            if node.node_id in seen:
                raise UIControlError("duplicate semantic node identity")
            seen.add(node.node_id)
        if len(_json(asdict(self)).encode("utf-8")) > 262_144:
            raise UIControlError("oversized semantic snapshot")

    @property
    def digest(self) -> str:
        self.validate()
        return _identity(asdict(self))


@dataclass(frozen=True)
class UIIntent:
    snapshot: UISnapshot
    target: SemanticNode
    action: str
    expected_state: str
    method: Method
    call: ToolCall
    fallback_evidence_id: str | None = None


class SemanticUIControl:
    """Model-neutral intent, preflight, result and recovery adapter.

    The adapter never performs UI effects itself. The trusted host must call
    issue() immediately before applying the effect and then retain the intent
    and external receipts durably for crash reconciliation.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise UIControlError("canonical typed ToolRegistry required")
        descriptor, _ = registry.get(ACTION_TOOL)
        if descriptor != ui_tool_descriptor():
            raise UIControlError("unexpected UI tool descriptor contract")
        self.registry = registry

    @staticmethod
    def select(
        snapshot: UISnapshot, *, role: str, name: str,
        node_id: str | None = None,
    ) -> SemanticNode:
        if not isinstance(snapshot, UISnapshot):
            raise UIControlError("typed semantic snapshot required")
        snapshot.validate()
        _word(role, "role")
        _word(name, "name")
        if node_id is not None:
            _word(node_id, "node_id")
        matches = tuple(node for node in snapshot.nodes
                        if node.role == role and node.name == name
                        and (node_id is None or node.node_id == node_id))
        if len(matches) != 1:
            raise UIControlError("missing or ambiguous semantic target")
        return matches[0]

    @staticmethod
    def _bind(intent: UIIntent) -> None:
        if not isinstance(intent, UIIntent) or not isinstance(intent.call, ToolCall):
            raise UIControlError("typed UI intent required")
        intent.snapshot.validate()
        intent.target.validate()
        if intent.target not in intent.snapshot.nodes:
            raise UIControlError("unbound UI target")
        if intent.action not in ACTIONS:
            raise UIControlError("unsupported UI action")
        _word(intent.expected_state, "postcondition")
        if len(intent.expected_state) > 256:
            raise UIControlError("postcondition too large")
        if intent.method not in METHODS:
            raise UIControlError("unsupported UI method")
        if intent.method == "semantic":
            if intent.fallback_evidence_id is not None:
                raise UIControlError("unexpected fallback authority")
        else:
            _word(intent.fallback_evidence_id, "fallback evidence")
        expected = {
            "snapshot_digest": intent.snapshot.digest,
            "target_digest": intent.target.digest,
            "action": intent.action,
            "expected_state": intent.expected_state,
            "method": intent.method,
        }
        if intent.call.tool_id != ACTION_TOOL or intent.call.request_json != _json(expected):
            raise UIControlError("UI intent/call binding mismatch")

    def prepare(
        self, snapshot: UISnapshot, *, role: str, name: str,
        action: str, expected_state: str, task_id: str, effect_id: str,
        expected_epoch: int, expected_revision: int, grant_evidence_id: str,
        verify_snapshot: Callable[[UISnapshot], bool],
        authorize: Callable[..., bool], node_id: str | None = None,
        method: Method = "semantic", fallback_evidence_id: str | None = None,
        verify_fallback: Callable[..., bool] | None = None,
    ) -> UIIntent:
        target = self.select(snapshot, role=role, name=name, node_id=node_id)
        if not callable(verify_snapshot) or verify_snapshot(snapshot) is not True:
            raise UIControlError("unverified UI observation")
        if action not in ACTIONS:
            raise UIControlError("unsupported UI action")
        _word(expected_state, "postcondition")
        if len(expected_state) > 256:
            raise UIControlError("postcondition too large")
        if method not in METHODS:
            raise UIControlError("unsupported UI method")
        if method == "semantic" and fallback_evidence_id is not None:
            raise UIControlError("fallback authority with semantic method")
        if method != "semantic":
            _word(fallback_evidence_id, "fallback evidence")
            if (not callable(verify_fallback)
                    or verify_fallback(snapshot, target, method, fallback_evidence_id) is not True):
                raise UIControlError("coordinate/vision fallback lacks explicit evidence")
        request = {
            "snapshot_digest": snapshot.digest, "target_digest": target.digest,
            "action": action, "expected_state": expected_state, "method": method,
        }
        call = self.registry.prepare(
            task_id=task_id, effect_id=effect_id, tool_id=ACTION_TOOL,
            request=request, permissions=("ui.control",),
            grant_evidence_id=grant_evidence_id,
            expected_epoch=expected_epoch, expected_revision=expected_revision,
            authorize=authorize,
        )
        intent = UIIntent(snapshot, target, action, expected_state, method,
                          call, fallback_evidence_id)
        self._bind(intent)
        return intent

    def issue(
        self, intent: UIIntent, current: UISnapshot, *,
        verify_current: Callable[[UISnapshot], bool],
    ) -> TaskSnapshot:
        """Fail closed on changed UI before marking the one external effect unknown."""
        self._bind(intent)
        if (not isinstance(current, UISnapshot) or current.digest != intent.snapshot.digest
                or not callable(verify_current) or verify_current(current) is not True):
            raise UIControlError("stale or unverified UI target; reconcile, never blind retry")
        return self.registry.tasks.issue_effect(
            intent.call.task_id, intent.call.effect_id,
            expected_epoch=intent.call.control_epoch,
            expected_revision=intent.call.task_revision,
        )

    def complete(
        self, intent: UIIntent, *, observed_state: str, receipt_id: str,
        evidence_id: str, verify_call: Callable[..., bool],
        verify_result: Callable[..., bool],
        verify_postcondition: Callable[..., bool],
    ) -> TaskSnapshot:
        """Accept only independently proven outcome matching declared postcondition."""
        self._bind(intent)
        if observed_state != intent.expected_state:
            raise UIControlError("ambiguous or unmet postcondition requires reconciliation")
        if (not callable(verify_postcondition)
                or verify_postcondition(intent, observed_state, evidence_id) is not True):
            raise UIControlError("independent UI postcondition proof missing")
        result = make_result(
            intent.call, receipt_id=receipt_id, evidence_id=evidence_id,
            outcome="success",
            output={"target_digest": intent.target.digest,
                    "state": observed_state, "ok": True},
        )
        return self.registry.accept_result(
            intent.call, result, verify_call=verify_call, verify_result=verify_result,
        )

    def reconcile_unknown(
        self, intent: UIIntent, *, observed_state: str, receipt_id: str,
        evidence_id: str,
        verify_recovery: Callable[[UIIntent, TaskSnapshot, str, str, str], bool],
    ) -> TaskSnapshot:
        """Explicit trusted receipt may resolve a crash-unknown effect after epoch bump.

        It never repeats a UI action and never grants new tool permissions.
        A fresh trusted verifier must attest the original call, target, actual
        postcondition and receipt against the current canonical task snapshot.
        """
        self._bind(intent)
        _word(receipt_id, "receipt")
        _word(evidence_id, "evidence")
        if observed_state != intent.expected_state:
            raise UIControlError("unknown UI outcome cannot be assumed successful")
        state = self.registry.tasks.load(intent.call.task_id)
        if not any(effect.effect_id == intent.call.effect_id and effect.status == "unknown"
                   for effect in state.pending_effects):
            raise UIControlError("unknown effect missing or already reconciled")
        if (not callable(verify_recovery) or
                verify_recovery(intent, state, receipt_id, evidence_id, observed_state) is not True):
            raise UIControlError("trusted recovery evidence missing")
        return self.registry.tasks.resolve_effect(
            intent.call.task_id, intent.call.effect_id, receipt_id,
            expected_epoch=state.control_epoch, expected_revision=state.revision,
        )
