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

    assert result["observed_main_sha"] == "a6c4c269babbac134679a70b719e86e3fbcd4932"
    assert result["observed_main_tree_sha"] == "9363c3cad0bdf8c58034a4a47a10d6d87421a123"
    assert result["current_repository_main_sha"] == result["observed_main_sha"]
    assert result["current_repository_main_tree_sha"] == result["observed_main_tree_sha"]
    assert result["qualified_current_equivalent_surface_count"] == 239
    assert result["accepted_main_surface_count"] == 120
    assert result["candidate_overlay_surface_count"] == 81
    assert result["checkout_surface_count"] == 199


def test_repository_executable_surface_distribution_is_pinned() -> None:
    result = _validate()

    assert result["main_capability_counts"] == {
        "accelerated-scaling-research": 2,
        "byte-tokenizer-runtime": 1,
        "checkpoint-integrity-mechanics": 1,
        "data-governance-mechanics": 102,
        "deterministic-packing-mechanics": 2,
        "executable-capability-map": 1,
        "learned20m-control-plane": 5,
        "model-spec-identity": 2,
        "package-runtime": 1,
        "portable-run-authority": 2,
        "project-control-plane": 1,
    }



def test_unqualified_peer_executables_are_explicitly_quarantined() -> None:
    payload = _load_strict_json(_INVENTORY)
    capabilities = _load_strict_json(_CAPABILITIES)

    overrides = payload["candidate_overrides"]
    assert len(overrides) == 81
    assert len({entry["path"] for entry in overrides}) == 81
    assert {
        "path": ".github/workflows/ci.yml",
        "capability_id": "peer-source-qualification-pending",
    } in overrides
    capability_by_id = {
        item["capability_id"]: item for item in capabilities["capabilities"]
    }
    assert all(
        capability_by_id[entry["capability_id"]]["status"] == "UNAVAILABLE"
        for entry in overrides
    )
    assert {
        "rule_id": "section2-surface-validator",
        "selector": "exact",
        "pattern": "tools/validate_section2_repository_surface_coverage.py",
        "capability_id": "executable-capability-map",
    } in payload["rules"]

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



def test_repository_surface_coverage_rejects_available_candidate_override(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert payload["candidate_overrides"]
    payload["candidate_overrides"][0]["capability_id"] = "model-spec-identity"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain UNAVAILABLE"):
        _validate(inventory)

def test_repository_surface_coverage_rejects_candidate_override_capability_remap(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["candidate_overrides"].append(
        {
            "path": "tools/validate_d04_learned20m_tokenizer_decision.py",
            "capability_id": "learned-20m-base",
        }
    )

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="cannot remap accepted-main executable capability",
    ):
        _validate(inventory)


def test_repository_surface_coverage_rejects_missing_candidate_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_surface_blob_map = surface_validator._surface_blob_map

    def drifted_surface_blob_map(root: Path, treeish: str) -> dict[str, str]:
        blobs = original_surface_blob_map(root, treeish)
        if treeish == "HEAD":
            blobs = dict(blobs)
            path = "tools/new_unmapped_qualification_candidate.py"
            blobs[path] = f"100644:{'f' * 40}"
        return blobs

    monkeypatch.setattr(
        surface_validator,
        "_surface_blob_map",
        drifted_surface_blob_map,
    )
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["candidate_overrides"] = [
        item for item in payload["candidate_overrides"]
        if item["path"] != ".github/workflows/ci.yml"
    ]
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


def test_repository_surface_coverage_rejects_accepted_main_capability_remap(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    target = next(
        rule
        for rule in payload["rules"]
        if rule["rule_id"] == "section2-surface-validator"
    )
    assert target["capability_id"] == "executable-capability-map"
    target["capability_id"] = "model-spec-identity"
    payload["expected_main_capability_counts"]["model-spec-identity"] += 1
    del payload["expected_main_capability_counts"]["executable-capability-map"]

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="accepted-main executable capability mapping drift",
    ):
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


def test_repository_surface_coverage_accepts_equivalent_historical_main_receipt(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = payload["observed_main_sha"]
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate(inventory)
    assert result["current_repository_main_sha"] == payload["observed_main_sha"]
    assert result["current_repository_main_tree_sha"] == payload["observed_main_tree_sha"]


def test_repository_surface_coverage_rejects_non_equivalent_historical_receipt(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = "5c041ca56edda55a5c3334f722361754051e121c"
