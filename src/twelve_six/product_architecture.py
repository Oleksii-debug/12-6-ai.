from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from twelve_six.model import canonical_json_sha256


class ProductLayer(StrEnum):
    """Stable top-level layers in the 12-6 product architecture."""

    BASE_MODEL = "base_model"
    POST_BASE_LEARNING = "post_base_learning"
    MODEL_GATEWAY = "model_gateway"
    COGNITION_MEMORY = "cognition_memory"
    TOOLS = "tools"
    LIVE_AGENT = "live_agent"
    EVOLUTION = "evolution"


REQUIRED_LAYERS = frozenset(ProductLayer)


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    """Versioned implementation bound to one stable product-layer contract."""

    layer: ProductLayer
    component_id: str
    contract_version: int
    implementation_version: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not isinstance(self.contract_version, int) or isinstance(self.contract_version, bool):
            raise ValueError("contract_version must be an integer")
        if self.contract_version <= 0:
            raise ValueError("contract_version must be positive")
        if not self.implementation_version or not self.implementation_version.strip():
            raise ValueError("implementation_version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "component_id": self.component_id,
            "contract_version": self.contract_version,
            "implementation_version": self.implementation_version,
        }


@dataclass(frozen=True, slots=True)
class BoundaryContract:
    """Typed, versioned hand-off between two top-level product layers."""

    producer: ProductLayer
    consumer: ProductLayer
    contract: str
    version: int

    def __post_init__(self) -> None:
        if self.producer == self.consumer:
            raise ValueError("boundary producer and consumer must be different layers")
        if not self.contract or not self.contract.strip():
            raise ValueError("boundary contract must be non-empty")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ValueError("boundary version must be an integer")
        if self.version <= 0:
            raise ValueError("boundary version must be positive")

    @property
    def key(self) -> tuple[ProductLayer, ProductLayer, str]:
        return (self.producer, self.consumer, self.contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer.value,
            "consumer": self.consumer.value,
            "contract": self.contract,
            "version": self.version,
        }


REQUIRED_BOUNDARIES = (
    BoundaryContract(
        ProductLayer.POST_BASE_LEARNING,
        ProductLayer.BASE_MODEL,
        "candidate_model",
        1,
    ),
    BoundaryContract(ProductLayer.BASE_MODEL, ProductLayer.MODEL_GATEWAY, "inference", 1),
    BoundaryContract(ProductLayer.MODEL_GATEWAY, ProductLayer.LIVE_AGENT, "inference", 1),
    BoundaryContract(ProductLayer.COGNITION_MEMORY, ProductLayer.LIVE_AGENT, "memory", 1),
    BoundaryContract(ProductLayer.TOOLS, ProductLayer.LIVE_AGENT, "tools", 1),
    BoundaryContract(ProductLayer.LIVE_AGENT, ProductLayer.EVOLUTION, "verified_trace", 1),
    BoundaryContract(
        ProductLayer.EVOLUTION,
        ProductLayer.POST_BASE_LEARNING,
        "learning_candidate",
        1,
    ),
)


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase hexadecimal SHA-256 digest")


@dataclass(frozen=True, slots=True)
class BaseModelIdentity:
    """Exact swappable cognitive-core identity used by runtime-facing layers."""

    schema_version: int
    model_spec_sha256: str
    init_spec_sha256: str
    tokenizer_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported BaseModelIdentity schema_version: {self.schema_version}")
        for field_name in (
            "model_spec_sha256",
            "init_spec_sha256",
            "tokenizer_sha256",
            "checkpoint_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_spec_sha256": self.model_spec_sha256,
            "init_spec_sha256": self.init_spec_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProductArchitecture:
    """Executable architecture contract for the complete 12-6 AI system.

    The non-model graph is deliberately independent from the active Base-model
    identity. A newly qualified checkpoint can therefore replace the cognitive
    core without rewriting memory, tools, Live Agent, or Evolution components.
    """

    schema_version: int
    components: tuple[ComponentBinding, ...]
    boundaries: tuple[BoundaryContract, ...]
    active_model: BaseModelIdentity

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported ProductArchitecture schema_version: {self.schema_version}")

        layers = [binding.layer for binding in self.components]
        if len(layers) != len(set(layers)):
            raise ValueError("each product layer must have exactly one component binding")
        if set(layers) != REQUIRED_LAYERS:
            missing = sorted(layer.value for layer in REQUIRED_LAYERS.difference(layers))
            extra = sorted(layer.value for layer in set(layers).difference(REQUIRED_LAYERS))
            raise ValueError(f"product architecture layer mismatch: missing={missing}, extra={extra}")

        keys = [boundary.key for boundary in self.boundaries]
        if len(keys) != len(set(keys)):
            raise ValueError("boundary contracts must be unique by producer/consumer/contract")

        present = {boundary.key: boundary.version for boundary in self.boundaries}
        for required in REQUIRED_BOUNDARIES:
            if present.get(required.key) != required.version:
                raise ValueError(
                    "missing or incompatible required boundary: "
                    f"{required.producer.value}->{required.consumer.value}:"
                    f"{required.contract}@v{required.version}"
                )

    def component(self, layer: ProductLayer) -> ComponentBinding:
        for binding in self.components:
            if binding.layer == layer:
                return binding
        raise KeyError(layer)

    def with_active_model(self, model: BaseModelIdentity) -> ProductArchitecture:
        """Return the same product graph with a different exact Base-model identity."""

        return replace(self, active_model=model)

    def non_model_identity_sha256(self) -> str:
        components = sorted(
            (binding.to_dict() for binding in self.components),
            key=lambda item: item["layer"],
        )
        boundaries = sorted(
            (boundary.to_dict() for boundary in self.boundaries),
            key=lambda item: (
                item["producer"],
                item["consumer"],
                item["contract"],
                item["version"],
            ),
        )
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "components": components,
                "boundaries": boundaries,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "non_model_identity_sha256": self.non_model_identity_sha256(),
            "active_model": self.active_model.to_dict(),
        }

    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())
