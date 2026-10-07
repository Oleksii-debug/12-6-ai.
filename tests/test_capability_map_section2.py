from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import twelve_six.capability_map as capability_map_module
from twelve_six.capability_map import (
    CapabilityRegistry,
    CapabilityStatus,
    _changed_existing_source_paths,
    _worktree_python_source_drift,
    _python_source_blob_map,
    load_capability_registry,
    load_source_surface_inventory,
    validate_source_surface_coverage,
)


_ROOT = Path(__file__).parents[1]
_REGISTRY = _ROOT / "configs" / "control" / "product_capabilities_v1.json"
_SURFACE_INVENTORY = (
    _ROOT / "configs" / "control" / "product_source_surface_inventory_v1.json"
)


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


def test_registry_loader_rejects_noncanonical_available_test_command(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    target["test_vectors"][0]["command"] = "python -c print-pass"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="test vector command"):
        load_capability_registry(path)


def test_registry_loader_rejects_test_path_escape_from_tests_tree(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    target["test_vectors"][0]["command"] = (
        "pytest -q tests/../tools/validate_section2_repository_surface_coverage.py"
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical tests"):
        load_capability_registry(path)


def test_registry_loader_rejects_available_evidence_not_bound_to_main_ci(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    main_ci = next(
        evidence
        for evidence in target["evidence_targets"]
        if evidence["evidence_id"] == "main-ci"
    )
    main_ci["target"] = "github-actions:1"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact observed main CI"):
        load_capability_registry(path)


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


def test_registry_rejects_capability_forward_binding_absent_from_journey() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    target_index = 0
    target = capabilities[target_index]
    foreign_journey = next(
        journey
        for journey in registry.journeys
        if target.capability_id not in journey.capability_ids
        and journey.journey_id not in target.journey_ids
    )
    capabilities[target_index] = replace(
        target,
        journey_ids=(*target.journey_ids, foreign_journey.journey_id),
    )

    with pytest.raises(ValueError, match="is not listed by journey"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=tuple(capabilities),
            journeys=registry.journeys,
        )


def test_capability_rejects_missing_user_operator_journey() -> None:
    registry = _load()
    target = registry.capabilities[0]

    with pytest.raises(ValueError, match="must bind at least one user/operator journey"):
        replace(target, journey_ids=())


def test_registry_rejects_journey_without_capability_back_binding() -> None:
    registry = _load()
    journeys = list(registry.journeys)
    target_capability = registry.capability("model-spec-identity")
    target_index = next(
        index
        for index, journey in enumerate(journeys)
        if target_capability.capability_id not in journey.capability_ids
        and journey.journey_id not in target_capability.journey_ids
    )
    target_journey = journeys[target_index]
    journeys[target_index] = replace(
        target_journey,
        capability_ids=(
            *target_journey.capability_ids,
            target_capability.capability_id,
        ),
    )

    with pytest.raises(ValueError, match="is not back-bound"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=registry.capabilities,
            journeys=tuple(journeys),
        )


def test_registry_loader_rejects_bool_schema_version_alias(tmp_path: Path) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_capability_registry(path)


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


def _resolve_contract(path: str) -> object:
    parts = path.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            value: object = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attribute in parts[index:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(f"cannot import component contract: {path}")


def test_available_python_component_contracts_resolve_to_real_symbols() -> None:
    registry = _load()

    for capability in registry.capabilities:
        if capability.status is not CapabilityStatus.AVAILABLE:
            continue
        assert capability.component_contract.startswith("twelve_six.")
        assert _resolve_contract(capability.component_contract) is not None


def test_registry_constructor_rejects_unresolvable_available_contract() -> None:
    registry = _load()
    capabilities = list(registry.capabilities)
    target_index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.status is CapabilityStatus.AVAILABLE
    )
    capabilities[target_index] = replace(
        capabilities[target_index],
        component_contract="twelve_six.nonexistent.Contract",
    )

    with pytest.raises(ValueError, match="component contract"):
        CapabilityRegistry(
            schema_version=registry.schema_version,
            observed_main_sha=registry.observed_main_sha,
            observed_main_ci_run_id=registry.observed_main_ci_run_id,
            observed_main_ci_conclusion=registry.observed_main_ci_conclusion,
            capabilities=tuple(capabilities),
            journeys=registry.journeys,
        )


def test_registry_loader_rejects_unresolvable_available_contract(tmp_path: Path) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    target["component_contract"] = "twelve_six.nonexistent.Contract"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="component contract"):
        load_capability_registry(path)


def test_registry_loader_rejects_external_symbol_reexport_as_component_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    target["component_contract"] = "twelve_six.model.torch.nn.Module"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repository-owned"):
        load_capability_registry(path)


def test_registry_loader_rejects_forged_twelve_six_module_origin(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
    )
    module_name, symbol_name = target["component_contract"].rsplit(".", 1)
    original_module = sys.modules.get(module_name)
    forged = ModuleType(module_name)
    forged.__file__ = str(tmp_path / "forged_component.py")
    forged_symbol = type(symbol_name, (), {})
    forged_symbol.__module__ = module_name
    setattr(forged, symbol_name, forged_symbol)
    sys.modules[module_name] = forged
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="repository-owned module origin"):
            load_capability_registry(path)
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


def test_registry_loader_rejects_unknown_nested_capability_field(tmp_path: Path) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    payload["capabilities"][0]["forged_ready"] = True
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="capability schema"):
        load_capability_registry(path)


def test_registry_loader_rejects_duplicate_json_members(tmp_path: Path) -> None:
    text = _REGISTRY.read_text(encoding="utf-8")
    tampered = text.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "registry.json"
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="strict unambiguous"):
        load_capability_registry(path)


def test_registry_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    text = _REGISTRY.read_text(encoding="utf-8")
    tampered = text.replace(
        '"run_id": 37248299503',
        '"run_id": NaN',
        1,
    )
    path = tmp_path / "registry.json"
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="strict unambiguous"):
        load_capability_registry(path)


def test_worktree_python_source_drift_detects_dirty_tracked_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_command = [
        "git",
        "-C",
        str(tmp_path),
        "diff",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        "src/twelve_six",
    ]

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert command == expected_command
        assert check is False
        assert capture_output is True
        assert text is True
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="src/twelve_six/model.py\0README.md\0",
            stderr="",
        )

    monkeypatch.setattr(capability_map_module.subprocess, "run", fake_run)

    assert _worktree_python_source_drift(tmp_path, "src/twelve_six") == {
        "src/twelve_six/model.py"
    }


def test_library_git_evidence_probes_strip_ambient_git_redirection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hostile = {
        "Git_Dir": str(tmp_path / "forged.git"),
        "git_work_tree": str(tmp_path / "forged-worktree"),
        "gIt_CoNfIg_PaRaMeTeRs": "'core.hooksPath=/forged'",
        "Git_Index_File": str(tmp_path / "forged-index"),
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")

    original_run = capability_map_module.subprocess.run
    observed_commands: list[tuple[str, ...]] = []

    def recording_run(command: list[str], *args: object, **kwargs: object):
        if command and command[0] == "git":
            env = kwargs.get("env")
            assert isinstance(env, dict)
            assert all(
                not key.upper().startswith("GIT_") or key == "GIT_OPTIONAL_LOCKS"
                for key in env
            )
            assert env["GIT_OPTIONAL_LOCKS"] == "0"
            observed_commands.append(tuple(command))
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(capability_map_module.subprocess, "run", recording_run)

    validate_source_surface_coverage(
        _load(),
        load_source_surface_inventory(_SURFACE_INVENTORY),
        repo_root=_ROOT,
    )

    assert any("diff" in command for command in observed_commands)
    assert any("show" in command for command in observed_commands)
    assert any("ls-tree" in command for command in observed_commands)


def test_changed_existing_source_paths_detects_same_path_blob_drift() -> None:
    accepted_main_blobs = {
        "src/twelve_six/model.py": "a" * 40,
        "src/twelve_six/packing.py": "b" * 40,
    }
    checkout_blobs = {
        "src/twelve_six/model.py": "c" * 40,
        "src/twelve_six/packing.py": "b" * 40,
        "src/twelve_six/new_module.py": "d" * 40,
    }

    assert _changed_existing_source_paths(
        accepted_main_blobs,
        checkout_blobs,
    ) == {"src/twelve_six/model.py"}


def test_python_source_blob_map_preserves_unicode_path_and_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = "src/twelve_six/перевірка.py"

    class GitResult:
        returncode = 0
        stdout = "100755 blob " + "a" * 40 + f"\t{path}\0"

    monkeypatch.setattr(
        "twelve_six.capability_map.subprocess.run",
        lambda *args, **kwargs: GitResult(),
    )

    blobs = _python_source_blob_map(tmp_path, "HEAD", "src/twelve_six")

    assert blobs == {path: "100755:" + "a" * 40}


def test_python_source_blob_map_rejects_missing_nul_delimiter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class GitResult:
        returncode = 0
        stdout = "100644 blob " + "a" * 40 + "\tsrc/twelve_six/model.py"

    monkeypatch.setattr(
        "twelve_six.capability_map.subprocess.run",
        lambda *args, **kwargs: GitResult(),
    )

    with pytest.raises(ValueError, match="NUL delimiter"):
        _python_source_blob_map(tmp_path, "HEAD", "src/twelve_six")


def test_python_source_blob_map_rejects_symlink_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class GitResult:
        returncode = 0
        stdout = (
            "120000 blob "
            + "a" * 40
            + "\tsrc/twelve_six/symlinked_module.py\0"
        )

    monkeypatch.setattr(
        "twelve_six.capability_map.subprocess.run",
        lambda *args, **kwargs: GitResult(),
    )

    with pytest.raises(ValueError, match="regular Git blob"):
        _python_source_blob_map(tmp_path, "HEAD", "src/twelve_six")


def test_source_surface_inventory_covers_accepted_main_and_candidate_stack() -> None:
    registry = _load()
    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)

    assert inventory.observed_main_sha == registry.observed_main_sha
    assert inventory.observed_main_sha == "019944d5fe12334791f05f1232d13de4a12e37d3"
    assert inventory.observed_main_tree_sha == "c727add7897dd94bdb02493e0cd7a565be7e8d9f"
    assert inventory.accepted_main_surface_count == 114
    assert inventory.candidate_overlay_surface_count == 3
    assert inventory.source_surface_count == 117
    validate_source_surface_coverage(registry, inventory, repo_root=_ROOT)


def test_every_source_surface_maps_to_a_registered_capability_and_journey() -> None:
    registry = _load()
    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)
    capability_ids = {capability.capability_id for capability in registry.capabilities}

    for surface in inventory.surfaces:
        assert surface.capability_id in capability_ids
        capability = registry.capability(surface.capability_id)
        assert capability.journey_ids


def test_candidate_overlay_surfaces_remain_unavailable_until_integrated() -> None:
    registry = _load()
    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)

    for surface in inventory.surfaces:
        if surface.origin != "stacked_candidate":
            continue
        capability = registry.capability(surface.capability_id)
        assert capability.status is CapabilityStatus.UNAVAILABLE
        assert capability.integrated_result is None


def test_source_surface_coverage_rejects_candidate_overlay_mapped_to_available_capability(
    tmp_path: Path,
) -> None:
    registry = _load()
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    target = next(
        surface
        for surface in payload["surfaces"]
        if surface["origin"] == "stacked_candidate"
    )
    target["capability_id"] = "model-spec-identity"
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inventory = load_source_surface_inventory(path)

    with pytest.raises(ValueError, match="must map to UNAVAILABLE"):
        validate_source_surface_coverage(registry, inventory, repo_root=_ROOT)


def test_source_surface_inventory_accepts_zero_candidate_overlay_after_integration(
    tmp_path: Path,
) -> None:
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["surfaces"] = [
        surface
        for surface in payload["surfaces"]
        if surface["origin"] == "accepted_main"
    ]
    payload["source_surface_count"] = payload["accepted_main_surface_count"]
    payload["candidate_overlay_surface_count"] = 0
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    inventory = load_source_surface_inventory(path)

    assert inventory.candidate_overlay_surface_count == 0
    assert inventory.source_surface_count == inventory.accepted_main_surface_count


def test_source_surface_inventory_rejects_bool_schema_version_alias(
    tmp_path: Path,
) -> None:
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_source_surface_inventory(path)


def test_source_surface_inventory_rejects_path_escape(tmp_path: Path) -> None:
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["surfaces"][0]["path"] = "src/twelve_six/../tools/escape.py"
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical Python path"):
        load_source_surface_inventory(path)


def test_source_surface_inventory_rejects_unknown_nested_field(tmp_path: Path) -> None:
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["surfaces"][0]["forged"] = True
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_surface schema"):
        load_source_surface_inventory(path)


def test_source_surface_inventory_rejects_origin_count_reseal(tmp_path: Path) -> None:
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["accepted_main_surface_count"] -= 1
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted_main_surface_count"):
        load_source_surface_inventory(path)


def test_source_surface_coverage_rejects_unknown_capability_mapping(tmp_path: Path) -> None:
    registry = _load()
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    payload["surfaces"][0]["capability_id"] = "forged-capability"
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inventory = load_source_surface_inventory(path)

    with pytest.raises(ValueError, match="maps unknown capability ids"):
        validate_source_surface_coverage(registry, inventory, repo_root=_ROOT)


def test_source_surface_coverage_rejects_current_checkout_drift(tmp_path: Path) -> None:
    registry = _load()
    payload = json.loads(_SURFACE_INVENTORY.read_text(encoding="utf-8"))
    target = next(
        surface
        for surface in payload["surfaces"]
        if surface["origin"] == "stacked_candidate"
    )
    payload["surfaces"].remove(target)
    payload["source_surface_count"] -= 1
    payload["candidate_overlay_surface_count"] -= 1
    path = tmp_path / "surface-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inventory = load_source_surface_inventory(path)

    with pytest.raises(ValueError, match="unmapped_checkout"):
        validate_source_surface_coverage(registry, inventory, repo_root=_ROOT)


def test_closed_schema_rejects_test_vector_subclass_serialization_resealing() -> None:
    registry = _load()
    capability = next(item for item in registry.capabilities if item.test_vectors)
    vector = capability.test_vectors[0]

    class ForgedTestVector(capability_map_module.TestVector):
        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            payload["command"] = "pytest -q tests/forged_serialized_target.py"
            return payload

    forged = ForgedTestVector(vector.vector_id, vector.level, vector.command)
    with pytest.raises(ValueError, match="test_vectors must contain TestVector values"):
        replace(
            capability,
            test_vectors=(forged, *capability.test_vectors[1:]),
        )


def test_closed_schema_rejects_capability_subclass_registry_resealing() -> None:
    registry = _load()
    capability = registry.capabilities[0]

    class ForgedCapability(capability_map_module.Capability):
        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            payload["component_contract"] = "forged.outside.accepted.contract"
            return payload

    forged = ForgedCapability(
        capability.capability_id,
        capability.schema_version,
        capability.status,
        capability.component_contract,
        capability.dependencies,
        capability.journey_ids,
        capability.environments,
        capability.test_vectors,
        capability.evidence_targets,
        capability.integrated_result,
        capability.unavailable_reason,
    )
    capabilities = tuple(
        forged if item.capability_id == capability.capability_id else item
        for item in registry.capabilities
    )

    with pytest.raises(ValueError, match="capabilities must contain only Capability values"):
        capability_map_module.CapabilityRegistry(
            registry.schema_version,
            registry.observed_main_sha,
            registry.observed_main_ci_run_id,
            registry.observed_main_ci_conclusion,
            capabilities,
            registry.journeys,
        )


def test_closed_schema_rejects_source_surface_subclass_resealing() -> None:
    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)
    surface = inventory.surfaces[0]

    class ForgedSourceSurface(capability_map_module.SourceSurface):
        def to_dict(self) -> dict[str, object]:
            payload = super().to_dict()
            payload["capability_id"] = "forged-capability"
            return payload

    forged = ForgedSourceSurface(surface.path, surface.capability_id, surface.origin)
    surfaces = (forged, *inventory.surfaces[1:])

    with pytest.raises(ValueError, match="surfaces must contain only SourceSurface values"):
        replace(inventory, surfaces=surfaces)


def test_closed_scalar_and_container_schema_boundaries_reject_behavioral_subclasses() -> None:
    class ForgedStr(str):
        def strip(self) -> str:
            return "forged-valid"

    class ForgedInt(int):
        pass

    class ForgedBytes(bytes):
        def decode(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("behavioral bytes subclass must not be decoded")

    class ForgedTuple(tuple):
        pass

    with pytest.raises(ValueError, match="capability registry input must be bytes"):
        capability_map_module._strict_json_object(ForgedBytes(b"{}"))

    with pytest.raises(ValueError, match="environment_id must be a canonical identifier"):
        capability_map_module.EnvironmentSupport(ForgedStr("windows"), True)

    registry = _load()
    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        replace(registry, schema_version=ForgedInt(1))

    capability = registry.capabilities[0]
    with pytest.raises(ValueError, match="dependencies must be an immutable tuple"):
        replace(capability, dependencies=ForgedTuple(capability.dependencies))

    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)
    surface = inventory.surfaces[0]
    with pytest.raises(ValueError, match="source surface origin is unsupported"):
        replace(surface, origin=ForgedStr(surface.origin))


def test_registry_rejects_enum_wire_value_mutation_before_serialization() -> None:
    registry = _load()
    status = capability_map_module.CapabilityStatus.AVAILABLE
    original_status_value = status.value
    object.__setattr__(status, "_value_", "FORGED_AVAILABLE")
    try:
        with pytest.raises(ValueError, match="status wire value is non-canonical"):
            registry.identity_sha256()
    finally:
        object.__setattr__(status, "_value_", original_status_value)

    vector = next(
        item
        for capability in registry.capabilities
        for item in capability.test_vectors
        if item.level is capability_map_module.TestLevel.COMPONENT
    )
    level = capability_map_module.TestLevel.COMPONENT
    original_level_value = level.value
    object.__setattr__(level, "_value_", "forged_component")
    try:
        with pytest.raises(ValueError, match="level wire value is non-canonical"):
            vector.to_dict()
    finally:
        object.__setattr__(level, "_value_", original_level_value)


def test_registry_enum_value_descriptor_rebinding_cannot_reseal_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _load()
    expected_identity = registry.identity_sha256()

    def dispatching_status(_: capability_map_module.CapabilityStatus) -> str:
        raise AssertionError("CapabilityStatus.value descriptor must not be dispatched")

    def dispatching_level(_: capability_map_module.TestLevel) -> str:
        raise AssertionError("TestLevel.value descriptor must not be dispatched")

    monkeypatch.setattr(
        capability_map_module.CapabilityStatus,
        "value",
        property(dispatching_status),
        raising=False,
    )
    monkeypatch.setattr(
        capability_map_module.TestLevel,
        "value",
        property(dispatching_level),
        raising=False,
    )

    assert registry.identity_sha256() == expected_identity
    assert _load().identity_sha256() == expected_identity


def test_registry_loader_ignores_poisoned_enum_value_lookup_tables(
    tmp_path: Path,
) -> None:
    payload = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    target = next(
        capability
        for capability in payload["capabilities"]
        if capability["status"] == "AVAILABLE"
        and any(vector["level"] == "component" for vector in capability["test_vectors"])
    )
    component_vector = next(
        vector for vector in target["test_vectors"] if vector["level"] == "component"
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    level_map = capability_map_module.TestLevel._value2member_map_
    status_map = capability_map_module.CapabilityStatus._value2member_map_
    original_component = level_map["component"]
    original_integration = level_map["integration"]
    original_available = status_map["AVAILABLE"]
    original_unavailable = status_map["UNAVAILABLE"]
    level_map["component"] = capability_map_module.TestLevel.INTEGRATION
    level_map["integration"] = capability_map_module.TestLevel.COMPONENT
    status_map["AVAILABLE"] = capability_map_module.CapabilityStatus.UNAVAILABLE
    status_map["UNAVAILABLE"] = capability_map_module.CapabilityStatus.AVAILABLE
    try:
        registry = load_capability_registry(path)
    finally:
        level_map["component"] = original_component
        level_map["integration"] = original_integration
        status_map["AVAILABLE"] = original_available
        status_map["UNAVAILABLE"] = original_unavailable

    loaded = registry.capability(target["capability_id"])
    loaded_vector = next(
        vector
        for vector in loaded.test_vectors
        if vector.vector_id == component_vector["vector_id"]
    )
    assert loaded.status is capability_map_module.CapabilityStatus.AVAILABLE
    assert loaded_vector.level is capability_map_module.TestLevel.COMPONENT


def test_registry_enum_policy_ignores_module_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _load()
    baseline_identity = registry.identity_sha256()

    monkeypatch.setattr(
        capability_map_module,
        "_CANONICAL_CAPABILITY_STATUSES",
        (
            capability_map_module.CapabilityStatus.UNAVAILABLE,
            capability_map_module.CapabilityStatus.AVAILABLE,
        ),
    )
    monkeypatch.setattr(
        capability_map_module,
        "_CANONICAL_CAPABILITY_STATUS_VALUES",
        ("FORGED_UNAVAILABLE", "FORGED_AVAILABLE"),
    )
    monkeypatch.setattr(
        capability_map_module,
        "_CANONICAL_TEST_LEVELS",
        tuple(reversed(tuple(capability_map_module.TestLevel))),
    )
    monkeypatch.setattr(
        capability_map_module,
        "_CANONICAL_TEST_LEVEL_VALUES",
        ("forged_end_to_end", "forged_integration", "forged_component"),
    )

    assert registry.identity_sha256() == baseline_identity
    assert _load().identity_sha256() == baseline_identity


def test_registry_revalidates_post_construction_nested_mutation() -> None:
    registry = _load()
    capability = next(item for item in registry.capabilities if item.test_vectors)
    vector = capability.test_vectors[0]
    object.__setattr__(vector, "command", "python -c print-pass")

    with pytest.raises(ValueError, match="test vector command"):
        registry.identity_sha256()


def test_registry_accessors_revalidate_mutated_journey_state() -> None:
    registry = _load()
    journey = registry.journeys[0]
    object.__setattr__(journey, "title", "")

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        registry.journey_available(journey.journey_id)

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        registry.acceptance_path(registry.capabilities[0].capability_id)


def test_source_inventory_revalidates_mutated_surface_state() -> None:
    inventory = load_source_surface_inventory(_SURFACE_INVENTORY)
    surface = inventory.surfaces[0]
    object.__setattr__(surface, "origin", "forged")

    with pytest.raises(ValueError, match="source surface origin is unsupported"):
        inventory.identity_sha256()

    with pytest.raises(ValueError, match="source surface origin is unsupported"):
        validate_source_surface_coverage(_load(), inventory, repo_root=_ROOT)

