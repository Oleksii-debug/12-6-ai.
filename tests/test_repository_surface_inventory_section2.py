from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.validate_section2_repository_surface_coverage as surface_validator
from tools.validate_section2_repository_surface_coverage import (
    _candidate_surface_paths,
    _load_strict_json,
    validate_repository_surface_coverage,
)


_ROOT = Path(__file__).parents[1]
_INVENTORY = (
    _ROOT
    / "configs"
    / "control"
    / "product_repository_executable_surface_rules_v1.json"
)
_CAPABILITIES = _ROOT / "configs" / "control" / "product_capabilities_v1.json"


def _validate(
    inventory: Path = _INVENTORY,
    capabilities: Path = _CAPABILITIES,
) -> dict[str, object]:
    return validate_repository_surface_coverage(
        repo_root=_ROOT,
        inventory_path=inventory,
        capability_registry_path=capabilities,
    )


def test_repository_executable_surface_coverage_is_exact_and_complete() -> None:
    result = _validate()

    assert result["observed_main_sha"] == "019944d5fe12334791f05f1232d13de4a12e37d3"
    assert result["observed_main_tree_sha"] == "c727add7897dd94bdb02493e0cd7a565be7e8d9f"
    assert result["current_repository_main_sha"] == "e5dbb7107d5b54f09a26d07d59f093ac05ede9c7"
    assert result["current_repository_main_tree_sha"] == "429a9933512f3d0c17f42d80193365e5df3f195a"
    assert result["qualified_current_equivalent_surface_count"] == 233
    assert result["accepted_main_surface_count"] == 119
    assert result["candidate_overlay_surface_count"] == 1
    assert result["checkout_surface_count"] == 120


def test_repository_executable_surface_distribution_is_pinned() -> None:
    result = _validate()

    assert result["main_capability_counts"] == {
        "accelerated-scaling-research": 2,
        "byte-tokenizer-runtime": 1,
        "checkpoint-integrity-mechanics": 1,
        "data-governance-mechanics": 102,
        "deterministic-packing-mechanics": 2,
        "learned20m-control-plane": 5,
        "model-spec-identity": 2,
        "package-runtime": 1,
        "portable-run-authority": 2,
        "project-control-plane": 1,
    }


def test_section2_validator_is_itself_an_explicit_candidate_overlay() -> None:
    payload = _load_strict_json(_INVENTORY)

    assert payload["candidate_overrides"] == [
        {
            "path": "tools/validate_section2_repository_surface_coverage.py",
            "capability_id": "executable-capability-map",
        }
    ]


def test_candidate_surface_paths_include_changed_existing_surface() -> None:
    current_main_blobs = {
        ".github/workflows/ci.yml": "a" * 40,
        "tools/existing.py": "b" * 40,
    }
    checkout_blobs = {
        ".github/workflows/ci.yml": "c" * 40,
        "tools/existing.py": "b" * 40,
        "tools/new.py": "d" * 40,
    }

    assert _candidate_surface_paths(current_main_blobs, checkout_blobs) == {
        ".github/workflows/ci.yml",
        "tools/new.py",
    }


def test_repository_surface_coverage_rejects_missing_candidate_override(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["candidate_overrides"] = []
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unmapped_checkout"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_unknown_rule_capability(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["rules"][0]["capability_id"] = "forged-capability"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown classified capability"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_count_reseal(tmp_path: Path) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["expected_main_surface_count"] -= 1
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not sum"):
        _validate(inventory)


def test_repository_surface_inventory_rejects_duplicate_json_members(
    tmp_path: Path,
) -> None:
    text = _INVENTORY.read_text(encoding="utf-8")
    tampered = text.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="strict unambiguous"):
        _load_strict_json(inventory)


def test_repository_surface_rules_are_nonambiguous_on_exact_main() -> None:
    # Full validation classifies every accepted-main surface through exactly one rule.
    _validate()


def test_repository_surface_coverage_rejects_resealed_current_main(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = payload["observed_main_sha"]
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the live repository main ref"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_current_main_tree_reseal(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match current_repository_main_sha"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_duplicate_capability_ids(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["capabilities"].append(dict(payload["capabilities"][0]))
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="capability ids must be unique"):
        _validate(capabilities=capabilities)

def test_repository_surface_coverage_rejects_capability_registry_baseline_reseal(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["observed_main_sha"] = "e5dbb7107d5b54f09a26d07d59f093ac05ede9c7"
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match coverage baseline"):
        _validate(capabilities=capabilities)


def test_repository_surface_coverage_rejects_nonterminal_capability_ci(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["observed_main_ci"]["conclusion"] = "failure"
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be terminal success"):
        _validate(capabilities=capabilities)


def test_surface_blob_map_rejects_symlink_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    line = "120000 blob " + ("a" * 40) + "\ttools/forged.py"
    monkeypatch.setattr(surface_validator, "_run_git", lambda *_args: [line])

    with pytest.raises(ValueError, match="regular Git blob"):
        surface_validator._surface_blob_map(tmp_path, "HEAD")

