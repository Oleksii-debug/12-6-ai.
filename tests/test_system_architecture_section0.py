from __future__ import annotations

import hashlib

import pytest

import twelve_six.system_architecture as system_architecture_module
from twelve_six.system_architecture import (
    CognitiveCoreBinding,
    CognitiveCoreIdentity,
    CoreReplacementReceipt,
    InterfaceContract,
    ProductAssembly,
    RuntimeShellContract,
    SystemArchitectureManifest,
    SystemPlane,
    TypedBoundary,
    canonical_runtime_shell_v1,
    canonical_system_architecture_v1,
    replace_cognitive_core,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _core(label: str, parameter_count: int) -> CognitiveCoreIdentity:
    return CognitiveCoreIdentity(
        model_spec_sha256=_sha(f"{label}:model"),
        init_spec_sha256=_sha(f"{label}:init"),
        checkpoint_sha256=_sha(f"{label}:checkpoint"),
        tokenizer_sha256=_sha(f"{label}:tokenizer"),
        parameter_count=parameter_count,
    )


def _assembly(label: str = "20m", parameter_count: int = 20_613_440) -> ProductAssembly:
    architecture = canonical_system_architecture_v1()
    shell = canonical_runtime_shell_v1()
    return ProductAssembly(
        architecture=architecture,
        shell=shell,
        core_binding=CognitiveCoreBinding(
            core=_core(label, parameter_count),
            gateway_api=shell.gateway_api,
        ),
    )


def test_section0_manifest_contains_exact_required_planes() -> None:
    manifest = canonical_system_architecture_v1()

    assert tuple(manifest.planes) == tuple(SystemPlane)
    assert len(manifest.planes) == 7
    assert {plane.value for plane in manifest.planes} == {
        "base_model",
        "post_base_learning",
        "model_gateway",
        "persistent_cognition",
        "tools",
        "live_agent_plane",
        "evolution_plane",
    }


def test_manifest_rejects_raw_string_plane_type_aliases() -> None:
    canonical = canonical_system_architecture_v1()
    raw_string_planes = tuple(plane.value for plane in canonical.planes)

    with pytest.raises(ValueError, match="SystemPlane"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=raw_string_planes,  # type: ignore[arg-type]
            boundaries=canonical.boundaries,
        )


def test_section0_manifest_contains_exact_typed_boundary_set() -> None:
    manifest = canonical_system_architecture_v1()

    assert {boundary.name for boundary in manifest.boundaries} == {
        "base_to_gateway",
        "post_base_to_base",
        "gateway_to_cognition",
        "cognition_to_tools",
        "cognition_to_live_agent",
        "evolution_to_post_base",
        "evolution_to_gateway",
    }
    assert all(boundary.interface.schema_version == 1 for boundary in manifest.boundaries)


def test_manifest_rejects_missing_or_duplicate_planes() -> None:
    canonical = canonical_system_architecture_v1()

    with pytest.raises(ValueError, match="exactly seven"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes[:-1],
            boundaries=canonical.boundaries,
        )

    duplicated = (*canonical.planes[:-1], SystemPlane.BASE_MODEL)
    with pytest.raises(ValueError, match="unique"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=duplicated,
            boundaries=canonical.boundaries,
        )


def test_manifest_rejects_missing_or_duplicate_boundary_contracts() -> None:
    canonical = canonical_system_architecture_v1()

    with pytest.raises(ValueError, match="order or set"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=canonical.boundaries[:-1],
        )

    with pytest.raises(ValueError, match="names must be unique"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(*canonical.boundaries, canonical.boundaries[0]),
        )


def test_manifest_rejects_identity_ambiguous_reordering() -> None:
    canonical = canonical_system_architecture_v1()

    with pytest.raises(ValueError, match="plane order or set"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=(canonical.planes[1], canonical.planes[0], *canonical.planes[2:]),
            boundaries=canonical.boundaries,
        )

    with pytest.raises(ValueError, match="boundary order or set"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(
                canonical.boundaries[1],
                canonical.boundaries[0],
                *canonical.boundaries[2:],
            ),
        )


def test_architecture_boundary_policy_is_runtime_immutable() -> None:
    with pytest.raises(TypeError):
        system_architecture_module._REQUIRED_BOUNDARY_SPECS[
            "base_to_gateway"
        ] = (  # type: ignore[index]
            SystemPlane.BASE_MODEL,
            SystemPlane.PERSISTENT_COGNITION,
            "twelve_six.model_gateway",
            1,
        )


def test_manifest_validation_fails_closed_after_boundary_policy_global_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = canonical_system_architecture_v1()
    first = canonical.boundaries[0]
    forged_policy = dict(system_architecture_module._REQUIRED_BOUNDARY_SPECS)
    forged_policy[first.name] = (
        SystemPlane.BASE_MODEL,
        SystemPlane.PERSISTENT_COGNITION,
        first.interface.name,
        first.interface.schema_version,
    )
    monkeypatch.setattr(
        system_architecture_module,
        "_REQUIRED_BOUNDARY_SPECS",
        forged_policy,
    )
    resealed = TypedBoundary(
        first.name,
        SystemPlane.BASE_MODEL,
        SystemPlane.PERSISTENT_COGNITION,
        first.interface,
    )

    with pytest.raises(ValueError, match="semantics are non-canonical"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(resealed, *canonical.boundaries[1:]),
        )


def test_manifest_rejects_boundary_semantic_resealing() -> None:
    canonical = canonical_system_architecture_v1()
    first = canonical.boundaries[0]
    resealed = TypedBoundary(
        first.name,
        SystemPlane.BASE_MODEL,
        SystemPlane.PERSISTENT_COGNITION,
        first.interface,
    )

    with pytest.raises(ValueError, match="semantics are non-canonical"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(resealed, *canonical.boundaries[1:]),
        )


def test_manifest_rejects_interface_resealing_under_canonical_boundary_name() -> None:
    canonical = canonical_system_architecture_v1()
    first = canonical.boundaries[0]
    resealed = TypedBoundary(
        first.name,
        first.producer,
        first.consumer,
        InterfaceContract("twelve_six.model_gateway", 2),
    )

    with pytest.raises(ValueError, match="semantics are non-canonical"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(resealed, *canonical.boundaries[1:]),
        )


def test_typed_boundary_rejects_self_edge_and_wrong_types() -> None:
    contract = InterfaceContract("test.contract", 1)

    with pytest.raises(ValueError, match="must differ"):
        TypedBoundary(
            "self_edge",
            SystemPlane.TOOLS,
            SystemPlane.TOOLS,
            contract,
        )

    with pytest.raises(ValueError, match="producer"):
        TypedBoundary(  # type: ignore[arg-type]
            "bad_producer",
            "tools",
            SystemPlane.LIVE_AGENT_PLANE,
            contract,
        )


def test_core_identity_is_strict_about_hashes_and_parameter_count() -> None:
    kwargs = {
        "model_spec_sha256": _sha("model"),
        "init_spec_sha256": _sha("init"),
        "checkpoint_sha256": _sha("checkpoint"),
        "tokenizer_sha256": _sha("tokenizer"),
        "parameter_count": 20_613_440,
    }

    CognitiveCoreIdentity(**kwargs)

    with pytest.raises(ValueError, match="model_spec_sha256"):
        CognitiveCoreIdentity(**{**kwargs, "model_spec_sha256": "A" * 64})

    with pytest.raises(ValueError, match="parameter_count"):
        CognitiveCoreIdentity(**{**kwargs, "parameter_count": True})


def test_architecture_and_shell_identities_are_deterministic() -> None:
    architecture_a = canonical_system_architecture_v1()
    architecture_b = canonical_system_architecture_v1()
    shell_a = canonical_runtime_shell_v1()
    shell_b = canonical_runtime_shell_v1()

    assert architecture_a.identity_sha256() == architecture_b.identity_sha256()
    assert shell_a.identity_sha256() == shell_b.identity_sha256()
    assert architecture_a.identity_sha256() != shell_a.identity_sha256()


def test_product_assembly_cross_binds_shell_gateway_to_architecture() -> None:
    shell = canonical_runtime_shell_v1()
    incompatible_shell = RuntimeShellContract(
        gateway_api=InterfaceContract("twelve_six.model_gateway", 2),
        memory_api=shell.memory_api,
        tools_api=shell.tools_api,
        voice_api=shell.voice_api,
        ui_api=shell.ui_api,
        orchestration_api=shell.orchestration_api,
    )

    with pytest.raises(ValueError, match="incompatible with architecture"):
        ProductAssembly(
            architecture=canonical_system_architecture_v1(),
            shell=incompatible_shell,
            core_binding=CognitiveCoreBinding(
                core=_core("20m", 20_613_440),
                gateway_api=incompatible_shell.gateway_api,
            ),
        )


def test_product_assembly_rejects_gateway_generation_mismatch() -> None:
    shell = canonical_runtime_shell_v1()

    with pytest.raises(ValueError, match="incompatible"):
        ProductAssembly(
            architecture=canonical_system_architecture_v1(),
            shell=shell,
            core_binding=CognitiveCoreBinding(
                core=_core("20m", 20_613_440),
                gateway_api=InterfaceContract("twelve_six.model_gateway", 2),
            ),
        )


def test_core_replacement_preserves_all_persistent_shell_contracts() -> None:
    current = _assembly()
    candidate = CognitiveCoreBinding(
        core=_core("200m", 200_000_000),
        gateway_api=current.shell.gateway_api,
    )
    shell_before = current.shell
    assembly_before_identity = current.identity_sha256()

    replacement, receipt = replace_cognitive_core(current, candidate)

    assert replacement.core_binding.core.parameter_count == 200_000_000
    assert replacement.core_binding.core != current.core_binding.core
    assert replacement.shell == shell_before
    assert replacement.shell.identity_sha256() == current.shell.identity_sha256()
    assert replacement.shell.surface_identities() == current.shell.surface_identities()
    assert receipt.shell_identity_sha256_before == receipt.shell_identity_sha256_after
    assert receipt.preserved_surface_identities == current.shell.surface_identities()
    assert receipt.preserved_shell == current.shell
    assert receipt.shell_rewrite_required is False
    assert replacement.identity_sha256() != assembly_before_identity


def test_core_replacement_accepts_new_checkpoint_same_scale_without_shell_rewrite() -> None:
    current = _assembly("20m-a", 20_613_440)
    candidate = CognitiveCoreBinding(
        core=_core("20m-b", 20_613_440),
        gateway_api=current.shell.gateway_api,
    )

    replacement, receipt = replace_cognitive_core(current, candidate)

    assert (
        replacement.core_binding.core.parameter_count
        == current.core_binding.core.parameter_count
    )
    assert (
        replacement.core_binding.core.checkpoint_sha256
        != current.core_binding.core.checkpoint_sha256
    )
    assert replacement.shell is current.shell
    assert receipt.shell_rewrite_required is False


def test_core_replacement_rejects_incompatible_gateway_before_mutation() -> None:
    current = _assembly()
    incompatible = CognitiveCoreBinding(
        core=_core("future", 400_000_000),
        gateway_api=InterfaceContract("twelve_six.model_gateway", 2),
    )
    original_identity = current.identity_sha256()

    with pytest.raises(ValueError, match="incompatible"):
        replace_cognitive_core(current, incompatible)

    assert current.identity_sha256() == original_identity


def test_runtime_shell_identity_changes_when_a_surface_contract_changes() -> None:
    shell = canonical_runtime_shell_v1()
    changed = RuntimeShellContract(
        gateway_api=shell.gateway_api,
        memory_api=InterfaceContract("twelve_six.memory", 2),
        tools_api=shell.tools_api,
        voice_api=shell.voice_api,
        ui_api=shell.ui_api,
        orchestration_api=shell.orchestration_api,
    )

    assert changed.identity_sha256() != shell.identity_sha256()


def test_runtime_shell_rejects_role_resealing() -> None:
    shell = canonical_runtime_shell_v1()

    with pytest.raises(ValueError, match="memory_api contract role semantics are non-canonical"):
        RuntimeShellContract(
            gateway_api=shell.gateway_api,
            memory_api=shell.tools_api,
            tools_api=shell.tools_api,
            voice_api=shell.voice_api,
            ui_api=shell.ui_api,
            orchestration_api=shell.orchestration_api,
        )


def test_replacement_receipt_rejects_false_claim_of_shell_preservation() -> None:
    shell = canonical_runtime_shell_v1()

    with pytest.raises(ValueError, match="changed runtime shell"):
        CoreReplacementReceipt(
            previous_core_identity_sha256=_sha("previous"),
            candidate_core_identity_sha256=_sha("candidate"),
            shell_identity_sha256_before=_sha("shell-a"),
            shell_identity_sha256_after=_sha("shell-b"),
            preserved_surface_identities=(),
            preserved_shell=shell,
            shell_rewrite_required=False,
        )

    with pytest.raises(ValueError, match="must not require"):
        CoreReplacementReceipt(
            previous_core_identity_sha256=_sha("previous"),
            candidate_core_identity_sha256=_sha("candidate"),
            shell_identity_sha256_before=_sha("shell"),
            shell_identity_sha256_after=_sha("shell"),
            preserved_surface_identities=(),
            preserved_shell=shell,
            shell_rewrite_required=True,
        )

def test_manifest_rejects_mutable_boundary_container() -> None:
    canonical = canonical_system_architecture_v1()

    with pytest.raises(ValueError, match="boundaries must be an immutable tuple"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=list(canonical.boundaries),  # type: ignore[arg-type]
        )

def test_manifest_rejects_non_boundary_element_fail_closed() -> None:
    canonical = canonical_system_architecture_v1()

    with pytest.raises(ValueError, match="only TypedBoundary"):
        SystemArchitectureManifest(
            schema_version=1,
            planes=canonical.planes,
            boundaries=(*canonical.boundaries[:-1], None),  # type: ignore[arg-type]
        )

def test_replacement_receipt_rejects_mutable_or_malformed_surface_container() -> None:
    surfaces = canonical_runtime_shell_v1().surface_identities()
    common = {
        "previous_core_identity_sha256": _sha("previous"),
        "candidate_core_identity_sha256": _sha("candidate"),
        "shell_identity_sha256_before": _sha("shell"),
        "shell_identity_sha256_after": _sha("shell"),
        "preserved_shell": canonical_runtime_shell_v1(),
        "shell_rewrite_required": False,
    }

    with pytest.raises(ValueError, match="immutable tuple of 2-tuples"):
        CoreReplacementReceipt(
            **common,
            preserved_surface_identities=list(surfaces),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="immutable tuple of 2-tuples"):
        CoreReplacementReceipt(
            **common,
            preserved_surface_identities=(*surfaces[:-1], ("ui-only",)),  # type: ignore[arg-type]
        )



def test_replacement_receipt_cross_binds_shell_snapshot_and_surface_hashes() -> None:
    shell = canonical_runtime_shell_v1()
    common = {
        "previous_core_identity_sha256": _sha("previous"),
        "candidate_core_identity_sha256": _sha("candidate"),
        "shell_identity_sha256_before": shell.identity_sha256(),
        "shell_identity_sha256_after": shell.identity_sha256(),
        "preserved_shell": shell,
        "shell_rewrite_required": False,
    }
    tampered_surfaces = list(shell.surface_identities())
    tampered_surfaces[1] = ("memory", _sha("tampered-memory"))

    with pytest.raises(ValueError, match="surface identities do not match preserved shell"):
        CoreReplacementReceipt(
            **common,
            preserved_surface_identities=tuple(tampered_surfaces),
        )

    changed_shell = RuntimeShellContract(
        gateway_api=shell.gateway_api,
        memory_api=InterfaceContract("twelve_six.memory", 2),
        tools_api=shell.tools_api,
        voice_api=shell.voice_api,
        ui_api=shell.ui_api,
        orchestration_api=shell.orchestration_api,
    )
    with pytest.raises(ValueError, match="shell identity does not match preserved shell"):
        CoreReplacementReceipt(
            **{**common, "preserved_shell": changed_shell},
            preserved_surface_identities=changed_shell.surface_identities(),
        )
