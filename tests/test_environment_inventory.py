from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.integration.environment_inventory import (
    EnvironmentInventoryError,
    build_environment_inventory,
    declared_requirements,
    validate_environment_inventory,
)

SOURCE_SHA = "a" * 40
RECORD_SHA = "b" * 64


def _package(name: str, version: str, *, licensed: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "version": version,
        "license_expression": "MIT" if licensed else None,
        "license": None,
        "license_classifiers": [],
        "record_sha256": RECORD_SHA,
        "provenance": {"editable": False, "vcs": None, "vcs_commit_id": None},
    }


def _inventory(packages: list[dict[str, object]]) -> dict[str, object]:
    return build_environment_inventory(
        repository="Oleksii-debug/12-6-ai.",
        source_sha=SOURCE_SHA,
        python_version="3.11.16",
        python_implementation="cpython",
        requirements={"runtime": ["torch>=2.5"], "optional:dev": ["pytest>=8"]},
        packages=packages,
    )


def test_inventory_is_order_stable_and_self_validating() -> None:
    first = _inventory([_package("Example_Pkg", "1.0"), _package("Other", "2.0")])
    second = _inventory([_package("other", "2.0"), _package("example-pkg", "1.0")])
    assert first == second
    validate_environment_inventory(first)


def test_inventory_hash_rejects_tampering() -> None:
    inventory = _inventory([_package("example", "1.0")])
    tampered = json.loads(json.dumps(inventory))
    tampered["packages"][0]["version"] = "2.0"
    with pytest.raises(EnvironmentInventoryError, match="hash mismatch"):
        validate_environment_inventory(tampered)


def test_duplicate_normalized_distribution_name_is_ambiguous() -> None:
    with pytest.raises(EnvironmentInventoryError, match="ambiguous installed distribution"):
        _inventory([_package("Example_Pkg", "1.0"), _package("example-pkg", "1.0")])


def test_unresolved_license_metadata_is_counted_not_silently_passed() -> None:
    inventory = _inventory([_package("unknown-license", "1.0", licensed=False)])
    assert inventory["summary"] == {
        "package_count": 1,
        "unresolved_license_metadata_count": 1,
    }


def test_source_sha_must_be_full_lowercase_git_identity() -> None:
    with pytest.raises(EnvironmentInventoryError, match="full lowercase Git object"):
        build_environment_inventory(
            repository="Oleksii-debug/12-6-ai.",
            source_sha="deadbeef",
            python_version="3.11.16",
            python_implementation="cpython",
            requirements={},
            packages=[],
        )


def test_declared_requirements_preserve_all_groups_deterministically(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "fixture"
version = "0"
dependencies = ["torch>=2.5", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["ruff>=0.12", "pytest>=8"]
export = ["safetensors>=0.5"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert declared_requirements(pyproject) == {
        "runtime": ["numpy>=1.26", "torch>=2.5"],
        "optional:dev": ["pytest>=8", "ruff>=0.12"],
        "optional:export": ["safetensors>=0.5"],
    }
