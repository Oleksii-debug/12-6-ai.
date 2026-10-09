"""Final LOCAL_FREE Plan-1 qualification: composed accepted contracts, no promotion."""
from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.artifact_identity import ArtifactRef as AcceptedArtifactRef
from twelve_six.backend_compatibility import load_static_backend_matrix
from twelve_six.contracts import (
    BASELINE_SCHEMA_VERSION,
    ArtifactRef,
    ContractCompatibilityError,
    ContractEvolutionReceipt,
    SystemPlane,
)
from twelve_six.integration.dependency_lock import SUPPORTED_PROFILES, validate_lock_index
from twelve_six.integration.dependency_security import build_lock_sbom, unique_components
from twelve_six.integration.release_fixture import build_notice_inventory, verify_notice_inventory
from twelve_six.system_architecture import SystemPlane as AcceptedSystemPlane

ROOT = Path(__file__).resolve().parents[1]


def test_migration_baseline_same_python_authority_no_new_class() -> None:
    assert BASELINE_SCHEMA_VERSION == 1
    assert ArtifactRef is AcceptedArtifactRef
    assert SystemPlane is AcceptedSystemPlane


def test_same_version_semantics_remain_stable_after_restart() -> None:
    safe = ContractEvolutionReceipt("artifact", 1, 1, "a" * 64, "a" * 64)
    assert safe.needs_requalification is False
    with pytest.raises(ContractCompatibilityError):
        ContractEvolutionReceipt("artifact", 1, 1, "a" * 64, "b" * 64)
    with pytest.raises(ContractCompatibilityError):
        ContractEvolutionReceipt("artifact", 1, 2, "a" * 64, "b" * 64)
    changed = ContractEvolutionReceipt(
        "artifact", 1, 2, "a" * 64, "b" * 64,
        migration_fixture_sha256="c" * 64, affected_plans=(2, 8),
    )
    assert changed.needs_requalification is True
    assert changed.affected_plans == (2, 8)


def test_locked_matrix_never_self_promotes_plan8_readiness() -> None:
    index = validate_lock_index(root=ROOT, index_path="requirements/locks/index.json")
    assert set(index["profiles"]) == SUPPORTED_PROFILES
    matrix = load_static_backend_matrix(root=ROOT)
    assert matrix.lock_index_sha256 == index["index_sha256"]
    for profile in SUPPORTED_PROFILES:
        decision = matrix.lookup(
            profile, "python", "dependency_environment",
            expected_lock_sha256=index["index_sha256"],
        )
        assert (decision.status, decision.qualification) == ("supported", "UNQUALIFIED")
        unknown = matrix.lookup(
            profile, "rocm", "training", expected_lock_sha256=index["index_sha256"],
        )
        assert (unknown.status, unknown.qualification) == ("unsupported", "UNQUALIFIED")


def test_rebuilt_supply_chain_coverage_never_forges_release_approval() -> None:
    sbom = build_lock_sbom(root=ROOT, source_sha="a" * 40)
    assert sbom == build_lock_sbom(root=ROOT, source_sha="a" * 40)
    components = unique_components(sbom)
    assert components
    observations = {
        item["key"]: {
            "status": "UNRESOLVED", "license_expression": None,
            "metadata_sha256": "b" * 64,
        }
        for item in components
    }
    notices = build_notice_inventory(sbom, observations)
    verify_notice_inventory(sbom, notices)
    assert notices["release_approved"] is False
    assert len(notices["notices"]) == len(components)
