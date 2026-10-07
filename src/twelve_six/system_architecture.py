from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


class SystemPlane(str, Enum):
    BASE_MODEL = "base_model"
    POST_BASE_LEARNING = "post_base_learning"
    MODEL_GATEWAY = "model_gateway"
    PERSISTENT_COGNITION = "persistent_cognition"
    TOOLS = "tools"
    LIVE_AGENT_PLANE = "live_agent_plane"
    EVOLUTION_PLANE = "evolution_plane"


_REQUIRED_PLANES = tuple(SystemPlane)
_REQUIRED_BOUNDARY_SPECS = {
    "base_to_gateway": (
        SystemPlane.BASE_MODEL,
        SystemPlane.MODEL_GATEWAY,
        "twelve_six.model_gateway",
        1,
    ),
    "post_base_to_base": (
        SystemPlane.POST_BASE_LEARNING,
        SystemPlane.BASE_MODEL,
        "twelve_six.descendant_model",
        1,
    ),
    "gateway_to_cognition": (
        SystemPlane.MODEL_GATEWAY,
        SystemPlane.PERSISTENT_COGNITION,
        "twelve_six.inference_exchange",
        1,
    ),
    "cognition_to_tools": (
        SystemPlane.PERSISTENT_COGNITION,
        SystemPlane.TOOLS,
        "twelve_six.tool_invocation",
        1,
    ),
    "cognition_to_live_agent": (
        SystemPlane.PERSISTENT_COGNITION,
        SystemPlane.LIVE_AGENT_PLANE,
        "twelve_six.cognition_state",
        1,
    ),
    "evolution_to_post_base": (
        SystemPlane.EVOLUTION_PLANE,
        SystemPlane.POST_BASE_LEARNING,
        "twelve_six.training_candidate",
        1,
    ),
    "evolution_to_gateway": (
        SystemPlane.EVOLUTION_PLANE,
        SystemPlane.MODEL_GATEWAY,
        "twelve_six.model_promotion",
        1,
    ),
}


@dataclass(frozen=True, slots=True)
class InterfaceContract:
    """A stable, versioned interface independent of any one model checkpoint."""

    name: str
    schema_version: int

    def __post_init__(self) -> None:
        _require_nonempty_text("name", self.name)
        _require_positive_int("schema_version", self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TypedBoundary:
    """One directed contract edge between two system planes."""

    name: str
    producer: SystemPlane
    consumer: SystemPlane
    interface: InterfaceContract

    def __post_init__(self) -> None:
        _require_nonempty_text("name", self.name)
        if not isinstance(self.producer, SystemPlane):
            raise ValueError("producer must be a SystemPlane")
        if not isinstance(self.consumer, SystemPlane):
            raise ValueError("consumer must be a SystemPlane")
        if self.producer is self.consumer:
            raise ValueError("typed boundary producer and consumer must differ")
        if not isinstance(self.interface, InterfaceContract):
            raise ValueError("interface must be an InterfaceContract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "producer": self.producer.value,
            "consumer": self.consumer.value,
            "interface": self.interface.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SystemArchitectureManifest:
    """Machine-checkable Section-0 product decomposition and typed boundary graph."""

    schema_version: int
    planes: tuple[SystemPlane, ...]
    boundaries: tuple[TypedBoundary, ...]

    def __post_init__(self) -> None:
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported system architecture schema_version")

        if not isinstance(self.planes, tuple) or any(
            not isinstance(plane, SystemPlane) for plane in self.planes
        ):
            raise ValueError(
                "system architecture planes must be an immutable tuple of SystemPlane values"
            )
        if not isinstance(self.boundaries, tuple):
            raise ValueError("system architecture boundaries must be an immutable tuple")

        if len(self.planes) != len(_REQUIRED_PLANES):
            raise ValueError("system architecture must contain exactly seven required planes")
        if len(set(self.planes)) != len(self.planes):
            raise ValueError("system architecture planes must be unique")
        if self.planes != _REQUIRED_PLANES:
            raise ValueError("system architecture plane order or set is non-canonical")

        if any(not isinstance(boundary, TypedBoundary) for boundary in self.boundaries):
            raise ValueError("boundaries must contain only TypedBoundary values")
        names = [boundary.name for boundary in self.boundaries]
        if len(set(names)) != len(names):
            raise ValueError("typed boundary names must be unique")

        plane_set = set(self.planes)
        for boundary in self.boundaries:
            if boundary.producer not in plane_set or boundary.consumer not in plane_set:
                raise ValueError("typed boundary refers to a plane outside the manifest")

        if names != list(_REQUIRED_BOUNDARY_SPECS):
            raise ValueError("system architecture typed-boundary order or set is non-canonical")

        observed_specs = {
            boundary.name: (
                boundary.producer,
                boundary.consumer,
                boundary.interface.name,
                boundary.interface.schema_version,
            )
            for boundary in self.boundaries
        }
        if observed_specs != _REQUIRED_BOUNDARY_SPECS:
            raise ValueError("system architecture typed-boundary semantics are non-canonical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planes": [plane.value for plane in self.planes],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CognitiveCoreIdentity:
    """Identity of a replaceable 12-6 cognitive core generation."""

    model_spec_sha256: str
    init_spec_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    parameter_count: int

    def __post_init__(self) -> None:
        _require_sha256("model_spec_sha256", self.model_spec_sha256)
        _require_sha256("init_spec_sha256", self.init_spec_sha256)
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("tokenizer_sha256", self.tokenizer_sha256)
        _require_positive_int("parameter_count", self.parameter_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_spec_sha256": self.model_spec_sha256,
            "init_spec_sha256": self.init_spec_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "parameter_count": self.parameter_count,
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeShellContract:
    """Contracts that must survive model/checkpoint/scale replacement unchanged."""

    gateway_api: InterfaceContract
    memory_api: InterfaceContract
    tools_api: InterfaceContract
    voice_api: InterfaceContract
    ui_api: InterfaceContract
    orchestration_api: InterfaceContract

    def __post_init__(self) -> None:
        for name, value in (
            ("gateway_api", self.gateway_api),
            ("memory_api", self.memory_api),
            ("tools_api", self.tools_api),
            ("voice_api", self.voice_api),
            ("ui_api", self.ui_api),
            ("orchestration_api", self.orchestration_api),
        ):
            if not isinstance(value, InterfaceContract):
                raise ValueError(f"{name} must be an InterfaceContract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gateway_api": self.gateway_api.to_dict(),
            "memory_api": self.memory_api.to_dict(),
            "tools_api": self.tools_api.to_dict(),
            "voice_api": self.voice_api.to_dict(),
            "ui_api": self.ui_api.to_dict(),
            "orchestration_api": self.orchestration_api.to_dict(),
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())

    def surface_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, contract.identity_sha256())
            for name, contract in (
                ("gateway", self.gateway_api),
                ("memory", self.memory_api),
                ("tools", self.tools_api),
                ("voice", self.voice_api),
                ("ui", self.ui_api),
                ("orchestration", self.orchestration_api),
            )
        )


@dataclass(frozen=True, slots=True)
class CognitiveCoreBinding:
    core: CognitiveCoreIdentity
    gateway_api: InterfaceContract

    def __post_init__(self) -> None:
        if not isinstance(self.core, CognitiveCoreIdentity):
            raise ValueError("core must be a CognitiveCoreIdentity")
        if not isinstance(self.gateway_api, InterfaceContract):
            raise ValueError("gateway_api must be an InterfaceContract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "core": self.core.to_dict(),
            "gateway_api": self.gateway_api.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProductAssembly:
    """A core generation attached to the persistent product shell."""

    architecture: SystemArchitectureManifest
    shell: RuntimeShellContract
    core_binding: CognitiveCoreBinding

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, SystemArchitectureManifest):
            raise ValueError("architecture must be a SystemArchitectureManifest")
        if not isinstance(self.shell, RuntimeShellContract):
            raise ValueError("shell must be a RuntimeShellContract")
        if not isinstance(self.core_binding, CognitiveCoreBinding):
            raise ValueError("core_binding must be a CognitiveCoreBinding")
        architecture_gateway = next(
            boundary.interface
            for boundary in self.architecture.boundaries
            if boundary.name == "base_to_gateway"
        )
        if self.shell.gateway_api != architecture_gateway:
            raise ValueError("runtime shell gateway contract is incompatible with architecture")
        if self.core_binding.gateway_api != self.shell.gateway_api:
            raise ValueError("cognitive core gateway contract is incompatible with runtime shell")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_identity_sha256": self.architecture.identity_sha256(),
            "shell_identity_sha256": self.shell.identity_sha256(),
            "core_binding": self.core_binding.to_dict(),
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CoreReplacementReceipt:
    previous_core_identity_sha256: str
    candidate_core_identity_sha256: str
    shell_identity_sha256_before: str
    shell_identity_sha256_after: str
    preserved_surface_identities: tuple[tuple[str, str], ...]
    shell_rewrite_required: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("previous_core_identity_sha256", self.previous_core_identity_sha256),
            ("candidate_core_identity_sha256", self.candidate_core_identity_sha256),
            ("shell_identity_sha256_before", self.shell_identity_sha256_before),
            ("shell_identity_sha256_after", self.shell_identity_sha256_after),
        ):
            _require_sha256(name, value)
        if not isinstance(self.preserved_surface_identities, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            for item in self.preserved_surface_identities
        ):
            raise ValueError(
                "preserved_surface_identities must be an immutable tuple of 2-tuples"
            )
        if self.shell_identity_sha256_before != self.shell_identity_sha256_after:
            raise ValueError("core replacement receipt cannot claim a changed runtime shell")
        if self.shell_rewrite_required is not False:
            raise ValueError("canonical core replacement must not require a runtime-shell rewrite")
        expected_surfaces = ("gateway", "memory", "tools", "voice", "ui", "orchestration")
        observed_surfaces = tuple(surface for surface, _ in self.preserved_surface_identities)
        if observed_surfaces != expected_surfaces:
            raise ValueError("preserved surface identity set or order is non-canonical")
        for surface, identity in self.preserved_surface_identities:
            _require_nonempty_text("surface", surface)
            _require_sha256(f"{surface}_identity_sha256", identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_core_identity_sha256": self.previous_core_identity_sha256,
            "candidate_core_identity_sha256": self.candidate_core_identity_sha256,
            "shell_identity_sha256_before": self.shell_identity_sha256_before,
            "shell_identity_sha256_after": self.shell_identity_sha256_after,
            "preserved_surface_identities": [
                {"surface": surface, "identity_sha256": identity}
                for surface, identity in self.preserved_surface_identities
            ],
            "shell_rewrite_required": self.shell_rewrite_required,
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())


def canonical_system_architecture_v1() -> SystemArchitectureManifest:
    model_gateway = InterfaceContract("twelve_six.model_gateway", 1)
    return SystemArchitectureManifest(
        schema_version=1,
        planes=_REQUIRED_PLANES,
        boundaries=(
            TypedBoundary(
                "base_to_gateway",
                SystemPlane.BASE_MODEL,
                SystemPlane.MODEL_GATEWAY,
                model_gateway,
            ),
            TypedBoundary(
                "post_base_to_base",
                SystemPlane.POST_BASE_LEARNING,
                SystemPlane.BASE_MODEL,
                InterfaceContract("twelve_six.descendant_model", 1),
            ),
            TypedBoundary(
                "gateway_to_cognition",
                SystemPlane.MODEL_GATEWAY,
                SystemPlane.PERSISTENT_COGNITION,
                InterfaceContract("twelve_six.inference_exchange", 1),
            ),
            TypedBoundary(
                "cognition_to_tools",
                SystemPlane.PERSISTENT_COGNITION,
                SystemPlane.TOOLS,
                InterfaceContract("twelve_six.tool_invocation", 1),
            ),
            TypedBoundary(
                "cognition_to_live_agent",
                SystemPlane.PERSISTENT_COGNITION,
                SystemPlane.LIVE_AGENT_PLANE,
                InterfaceContract("twelve_six.cognition_state", 1),
            ),
            TypedBoundary(
                "evolution_to_post_base",
                SystemPlane.EVOLUTION_PLANE,
                SystemPlane.POST_BASE_LEARNING,
                InterfaceContract("twelve_six.training_candidate", 1),
            ),
            TypedBoundary(
                "evolution_to_gateway",
                SystemPlane.EVOLUTION_PLANE,
                SystemPlane.MODEL_GATEWAY,
                InterfaceContract("twelve_six.model_promotion", 1),
            ),
        ),
    )


def canonical_runtime_shell_v1() -> RuntimeShellContract:
    return RuntimeShellContract(
        gateway_api=InterfaceContract("twelve_six.model_gateway", 1),
        memory_api=InterfaceContract("twelve_six.memory", 1),
        tools_api=InterfaceContract("twelve_six.tools", 1),
        voice_api=InterfaceContract("twelve_six.voice", 1),
        ui_api=InterfaceContract("twelve_six.ui", 1),
        orchestration_api=InterfaceContract("twelve_six.orchestration", 1),
    )


def replace_cognitive_core(
    assembly: ProductAssembly,
    candidate: CognitiveCoreBinding,
) -> tuple[ProductAssembly, CoreReplacementReceipt]:
    """Replace only the cognitive core while proving the persistent shell stayed identical."""

    if not isinstance(assembly, ProductAssembly):
        raise ValueError("assembly must be a ProductAssembly")
    if not isinstance(candidate, CognitiveCoreBinding):
        raise ValueError("candidate must be a CognitiveCoreBinding")
    if candidate.gateway_api != assembly.shell.gateway_api:
        raise ValueError("candidate cognitive core is incompatible with runtime shell gateway")

    shell_before = assembly.shell.identity_sha256()
    replacement = ProductAssembly(
        architecture=assembly.architecture,
        shell=assembly.shell,
        core_binding=candidate,
    )
    shell_after = replacement.shell.identity_sha256()
    if shell_before != shell_after:
        raise RuntimeError("runtime shell identity changed during cognitive-core replacement")

    receipt = CoreReplacementReceipt(
        previous_core_identity_sha256=assembly.core_binding.core.identity_sha256(),
        candidate_core_identity_sha256=candidate.core.identity_sha256(),
        shell_identity_sha256_before=shell_before,
        shell_identity_sha256_after=shell_after,
        preserved_surface_identities=assembly.shell.surface_identities(),
        shell_rewrite_required=False,
    )
    return replacement, receipt
