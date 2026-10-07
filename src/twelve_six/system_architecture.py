from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Final


class ArchitectureContractError(ValueError):
    """Raised when a system assembly would violate the canonical architecture contract."""


class ComponentRole(str, Enum):
    """Stable product boundaries for the complete 12-6 system."""

    BASE_MODEL = "base_model"
    POST_BASE_LEARNING = "post_base_learning"
    INFERENCE_GATEWAY = "inference_gateway"
    PERSISTENT_COGNITION_MEMORY = "persistent_cognition_memory"
    TOOLS = "tools"
    LIVE_AGENT_PLANE = "live_agent_plane"
    EVOLUTION_PLANE = "evolution_plane"
    VOICE_ADAPTER = "voice_adapter"
    OPERATOR_UI = "operator_ui"
    ORCHESTRATION = "orchestration"


class Availability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


_REQUIRED_ROLES: Final[tuple[ComponentRole, ...]] = tuple(ComponentRole)
_NON_CORE_ROLES: Final[tuple[ComponentRole, ...]] = tuple(
    role for role in ComponentRole if role is not ComponentRole.BASE_MODEL
)


def _require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureContractError(f"{field} must be a non-empty string")


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ArchitectureContractError(f"{field} must be a lowercase SHA-256 hex digest")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ArchitectureContractError(f"{field} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class CognitiveCoreBinding:
    """Exact identity of the replaceable Base Model cognition core."""

    model_spec_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    parameter_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.model_spec_sha256, "model_spec_sha256")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        if (
            not isinstance(self.parameter_count, int)
            or isinstance(self.parameter_count, bool)
            or self.parameter_count <= 0
        ):
            raise ArchitectureContractError("parameter_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class ComponentSlot:
    """One typed architecture boundary, explicitly available or unavailable."""

    role: ComponentRole
    availability: Availability
    contract_version: str
    implementation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ComponentRole):
            raise ArchitectureContractError("role must be a ComponentRole")
        _require_nonempty(self.contract_version, "contract_version")
        if self.availability is Availability.AVAILABLE:
            if self.implementation_id is None:
                raise ArchitectureContractError(
                    f"available role {self.role.value} requires implementation_id"
                )
            _require_nonempty(self.implementation_id, "implementation_id")
        elif self.availability is Availability.UNAVAILABLE:
            if self.implementation_id is not None:
                raise ArchitectureContractError(
                    f"unavailable role {self.role.value} must not advertise implementation_id"
                )
        else:
            raise ArchitectureContractError("availability must be an Availability value")


@dataclass(frozen=True, slots=True)
class TwelveSixSystemAssembly:
    """Validated composition of the replaceable core and all stable system boundaries."""

    core: CognitiveCoreBinding
    slots: tuple[ComponentSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.core, CognitiveCoreBinding):
            raise ArchitectureContractError("core must be a CognitiveCoreBinding")
        if not isinstance(self.slots, tuple):
            raise ArchitectureContractError("slots must be a tuple")

        by_role: dict[ComponentRole, ComponentSlot] = {}
        for slot in self.slots:
            if not isinstance(slot, ComponentSlot):
                raise ArchitectureContractError("every slot must be a ComponentSlot")
            if slot.role in by_role:
                raise ArchitectureContractError(f"duplicate architecture role: {slot.role.value}")
            by_role[slot.role] = slot

        missing = [role.value for role in _REQUIRED_ROLES if role not in by_role]
        if missing:
            raise ArchitectureContractError(
                "missing architecture roles: " + ", ".join(sorted(missing))
            )

        base_slot = by_role[ComponentRole.BASE_MODEL]
        if base_slot.availability is not Availability.AVAILABLE:
            raise ArchitectureContractError("base_model must be available when an assembly is built")

    def slot(self, role: ComponentRole) -> ComponentSlot:
        for slot in self.slots:
            if slot.role is role:
                return slot
        raise ArchitectureContractError(f"unknown architecture role: {role!r}")

    def replace_cognitive_core(
        self,
        core: CognitiveCoreBinding,
        *,
        implementation_id: str,
    ) -> TwelveSixSystemAssembly:
        """Swap Base Model identity without rewriting peripheral system boundaries."""

        _require_nonempty(implementation_id, "implementation_id")
        replacement_slots = tuple(
            replace(slot, implementation_id=implementation_id)
            if slot.role is ComponentRole.BASE_MODEL
            else slot
            for slot in self.slots
        )
        return TwelveSixSystemAssembly(core=core, slots=replacement_slots)

    def non_core_slots(self) -> tuple[ComponentSlot, ...]:
        return tuple(self.slot(role) for role in _NON_CORE_ROLES)


def canonical_component_roles() -> tuple[ComponentRole, ...]:
    """Return the complete stable role order for machine-readable consumers."""

    return _REQUIRED_ROLES
