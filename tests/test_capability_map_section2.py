from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.capability_map import (
    CapabilityRegistry,
    CapabilityStatus,
    load_capability_registry,
)


_ROOT = Path(__file__).parents[1]
_REGISTRY = _ROOT / "configs" / "control" / "product_capabilities_v1.json"


def _load() -> CapabilityRegistry:
    return load_capability_registry(_REGISTRY)


def test_registry_binds_exact_accepted_main_and_terminal_ci() -> None:
    registry = _load()

    assert registry.observed_main_sha == "019944d5fe12334791f05f1232d13de4a12e37d3"
    assert registry.observed_main_ci_run_id == 37248299503
    assert registry.observed_main_ci_conclusion == "success"
    assert len(registry.identity_sha256()) == 64


def test_available_capabilities_have_executable_acceptance_paths() -> None:
    registry = _load()

    available = [
        capability
        for capability in registry.capabilities
        if capability.status is CapabilityStatus.AVAILABLE
    ]
    assert available
    for capability in available:
        path = registry.acceptance_path(capability.capability_id)
        assert path["status"] == "AVAILABLE"
        assert path["component_contract"]
        levels = {vector["level"] for vector in path["test_vectors"]}
        assert {"component", "integration"} <= levels
        assert path["evidence_targets"]
        assert path["integrated_result"]


def test_checked_in_test_vectors_reference_real_test_files() -> None:
    registry = _load()

    for capability in registry.capabilities:
        for vector in capability.test_vectors:
            tokens = vector.command.split()
            assert tokens[:2] == ["pytest", "-q"]
            paths = [token for token in tokens[2:] if token.endswith(".py")]
            assert paths
            for relative_path in paths:
                assert (_ROOT / relative_path).is_file(), (
                    capability.capability_id,
                    vector.vector_id,
                    relative_path,
                )


def test_unavailable_capability_never_simulates_integrated_result() -> None:
    registry = _load()

    unavailable = [
        capability
        for capability in registry.capabilities
        if capability.status is CapabilityStatus.UNAVAILABLE
    ]
    assert unavailable
    for capability in unavailable:
        path = registry.acceptance_path(capability.capability_id)
        assert path == {
            "capability_id": capability.capability_id,
            "status": "UNAVAILABLE",
            "component_contract": capability.component_contract,
            "unavailable_reason": capability.unavailable_reason,
            "integrated_result": None,
        }


def test_known_not_yet_product_capabilities_are_explicitly_unavailable() -> None:
    registry = _load()

    for capability_id in (
        "replaceable-cognitive-core-shell",
        "unified-generation-identity",
        "learned-20m-base",
        "windows-nvda-final-product",
    ):
        capability = registry.capability(capability_id)
        assert capability.status is CapabilityStatus.UNAVAILABLE
        assert capability.unavailable_reason


def test_mechanics_are_not_resealed_as_physical_windows_acceptance() -> None:
    registry = _load()
    cli = registry.capability("windows-operator-cli-packaging")

    assert cli.status is CapabilityStatus.AVAILABLE
    support = {item.environment_id: item.supported for item in cli.environments}
    assert support["python-3.11"] is True
    assert support["windows-11-physical"] is False
    assert registry.journey_available("operator-windows-cli") is True
    assert registry.journey_available("operator-windows-product") is False


def test_available_research_input_journey_is_dependency_closed() -> None:
    registry = _load()

    assert registry.journey_available("researcher-prepare-training-inputs") is True
    packing = registry.capability("deterministic-packing-mechanics")
    assert packing.dependencies == ("byte-tokenizer-runtime",)


def test_registry_rejects_available_capability_with_unavailable_dependency() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    target_index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.capability_id == "unified-generation-identity"
    )
    target = capabilities[target_index]
    capabilities[target_index] = replace(
        target,
        status=CapabilityStatus.AVAILABLE,
        unavailable_reason=None,
        integrated_result="forged",
        environments=tuple(
            replace(environment, supported=True)
            for environment in target.environments
        ),
    )

    with pytest.raises(ValueError, match="depends on UNAVAILABLE"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=tuple(capabilities),
            journeys=registry.journeys,
        )


def test_registry_rejects_dependency_cycles() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    model_index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.capability_id == "model-spec-identity"
    )
    model = capabilities[model_index]
    capabilities[model_index] = replace(
        model,
        dependencies=("checkpoint-integrity-mechanics",),
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=tuple(capabilities),
            journeys=registry.journeys,
        )


def test_registry_rejects_journey_without_capability_back_binding() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    packing_index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.capability_id == "deterministic-packing-mechanics"
    )
    packing = capabilities[packing_index]
    capabilities[packing_index] = replace(packing, journey_ids=())

    with pytest.raises(ValueError, match="is not back-bound"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=tuple(capabilities),
            journeys=registry.journeys,
        )


def test_registry_loader_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    payload["forged_ready"] = True
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level schema"):
        load_capability_registry(path)


def test_registry_identity_changes_on_availability_reseal() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.capability_id == "learned-20m-base"
    )
    target = capabilities[index]
    capabilities[index] = replace(
        target,
        unavailable_reason="different blocker statement",
    )
    changed = CapabilityRegistry(
        schema_version=registry.schema_version,
        observed_main_sha=registry.observed_main_sha,
        observed_main_ci_run_id=registry.observed_main_ci_run_id,
        observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
        capabilities=tuple(capabilities),
        journeys=registry.journeys,
    )

    assert changed.identity_sha256() != registry.identity_sha256()
