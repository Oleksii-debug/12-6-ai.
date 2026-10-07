from __future__ import annotations

import pytest

from twelve_six.system_architecture import (
    ArchitectureContractError,
    Availability,
    CognitiveCoreBinding,
    ComponentRole,
    ComponentSlot,
    TwelveSixSystemAssembly,
    canonical_component_roles,
)


def _sha(char: str) -> str:
    return char * 64


def _core(char: str, *, parameters: int) -> CognitiveCoreBinding:
    return CognitiveCoreBinding(
        model_spec_sha256=_sha(char),
        checkpoint_sha256=_sha("b" if char != "b" else "c"),
        tokenizer_sha256=_sha("d" if char != "d" else "e"),
        parameter_count=parameters,
    )


def _slot(
    role: ComponentRole,
    *,
    available: bool,
    implementation_id: str | None = None,
) -> ComponentSlot:
    return ComponentSlot(
        role=role,
        availability=Availability.AVAILABLE if available else Availability.UNAVAILABLE,
        contract_version="12-6.boundary.v1",
        implementation_id=implementation_id,
    )


def _assembly() -> TwelveSixSystemAssembly:
    slots = []
    for role in canonical_component_roles():
        if role is ComponentRole.BASE_MODEL:
            slots.append(_slot(role, available=True, implementation_id="random-init-20m"))
        elif role is ComponentRole.INFERENCE_GATEWAY:
            slots.append(_slot(role, available=True, implementation_id="reference-cli"))
        elif role is ComponentRole.OPERATOR_UI:
            slots.append(_slot(role, available=True, implementation_id="windows-cli"))
        else:
            slots.append(_slot(role, available=False))
    return TwelveSixSystemAssembly(core=_core("a", parameters=20_613_440), slots=tuple(slots))


def test_canonical_roles_cover_core_runtime_agent_and_adapter_boundaries() -> None:
    assert canonical_component_roles() == (
        ComponentRole.BASE_MODEL,
        ComponentRole.POST_BASE_LEARNING,
        ComponentRole.INFERENCE_GATEWAY,
        ComponentRole.PERSISTENT_COGNITION_MEMORY,
        ComponentRole.TOOLS,
        ComponentRole.LIVE_AGENT_PLANE,
        ComponentRole.EVOLUTION_PLANE,
        ComponentRole.VOICE_ADAPTER,
        ComponentRole.OPERATOR_UI,
        ComponentRole.ORCHESTRATION,
    )


def test_unavailable_capability_cannot_advertise_fake_implementation() -> None:
    with pytest.raises(ArchitectureContractError, match="must not advertise"):
        _slot(ComponentRole.TOOLS, available=False, implementation_id="pretend-tools")


def test_available_capability_requires_concrete_implementation_identity() -> None:
    with pytest.raises(ArchitectureContractError, match="requires implementation_id"):
        _slot(ComponentRole.INFERENCE_GATEWAY, available=True)


def test_assembly_requires_every_boundary_exactly_once() -> None:
    assembly = _assembly()
    assert tuple(slot.role for slot in assembly.slots) == canonical_component_roles()

    with pytest.raises(ArchitectureContractError, match="duplicate architecture role"):
        TwelveSixSystemAssembly(core=assembly.core, slots=assembly.slots + (assembly.slots[0],))

    with pytest.raises(ArchitectureContractError, match="missing architecture roles"):
        TwelveSixSystemAssembly(core=assembly.core, slots=assembly.slots[:-1])


def test_base_model_must_be_available_for_a_runnable_assembly() -> None:
    assembly = _assembly()
    slots = tuple(
        _slot(slot.role, available=False)
        if slot.role is ComponentRole.BASE_MODEL
        else slot
        for slot in assembly.slots
    )
    with pytest.raises(ArchitectureContractError, match="base_model must be available"):
        TwelveSixSystemAssembly(core=assembly.core, slots=slots)


def test_core_identity_is_fail_closed() -> None:
    with pytest.raises(ArchitectureContractError, match="model_spec_sha256"):
        CognitiveCoreBinding(
            model_spec_sha256="A" * 64,
            checkpoint_sha256=_sha("b"),
            tokenizer_sha256=_sha("c"),
            parameter_count=1,
        )
    with pytest.raises(ArchitectureContractError, match="parameter_count"):
        CognitiveCoreBinding(
            model_spec_sha256=_sha("a"),
            checkpoint_sha256=_sha("b"),
            tokenizer_sha256=_sha("c"),
            parameter_count=0,
        )


def test_checkpoint_and_scale_swap_preserves_every_non_core_boundary() -> None:
    before = _assembly()
    replacement = _core("f", parameters=199_884_800)
    after = before.replace_cognitive_core(replacement, implementation_id="learned-200m")

    assert after.core == replacement
    assert after.slot(ComponentRole.BASE_MODEL).implementation_id == "learned-200m"
    assert after.non_core_slots() == before.non_core_slots()


def test_core_swap_preserves_unavailable_truth_without_simulating_capabilities() -> None:
    before = _assembly()
    after = before.replace_cognitive_core(
        _core("1", parameters=1_000_000_000),
        implementation_id="future-1b",
    )
    for role in (
        ComponentRole.POST_BASE_LEARNING,
        ComponentRole.PERSISTENT_COGNITION_MEMORY,
        ComponentRole.TOOLS,
        ComponentRole.LIVE_AGENT_PLANE,
        ComponentRole.EVOLUTION_PLANE,
        ComponentRole.VOICE_ADAPTER,
        ComponentRole.ORCHESTRATION,
    ):
        assert after.slot(role).availability is Availability.UNAVAILABLE
        assert after.slot(role).implementation_id is None
