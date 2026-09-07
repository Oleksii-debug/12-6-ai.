from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "observer", TOOLS / "observe_next100_025_data_gov_package.py"
)
assert SPEC and SPEC.loader
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def config() -> dict:
    return {
        "worker": "NEXT100-025-DATA-UA-GOV-OPEN",
        "dataset": {
            "dataset_id": "public-id",
            "dataset_page": "https://data.gov.ua/dataset/public-id",
            "package_api": "https://data.gov.ua/api/3/action/package_show?id=public-id",
            "publisher": "Expected Publisher",
        },
    }


def package(package_id: str = "internal-id") -> dict:
    return {
        "id": package_id,
        "name": "registry",
        "title": "Registry",
        "metadata_modified": "2026-09-03T11:38:00",
        "license_title": "Creative Commons Attribution",
        "organization": {
            "title": "Expected Publisher",
            "description": "must not be retained",
        },
        "maintainer": "must not be retained",
        "resources": [
            {
                "id": "resource-1",
                "name": "register",
                "format": "JSON",
                "url": "https://data.gov.ua/dataset/x/resource/resource-1/download/register.json",
                "created": "2026-01-01",
                "last_modified": "2026-09-03",
                "size": 1234,
                "hash": "abc",
                "description": "must not be retained either",
            }
        ],
    }


def test_identity_drift_is_retest_with_zero_credit() -> None:
    result = observer.observe(package(), config(), "rights-sha")
    assert result["status"] == "RETEST_PACKAGE_IDENTITY_DRIFT"
    boundary = result["claim_boundary"]
    assert boundary["source_capacity_bytes_credited"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["evaluation_authorized_bytes"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False


def test_observation_does_not_retain_contact_or_resource_prose() -> None:
    result = observer.observe(package(), config(), "rights-sha")
    encoded = json.dumps(result, ensure_ascii=False)
    assert "must not be retained" not in encoded
    assert result["resource_count"] == 1
    assert result["resources"][0]["id"] == "resource-1"


def test_expected_package_identity_still_requires_publisher_and_license() -> None:
    exact = observer.observe(package("public-id"), config(), "rights-sha")
    assert exact["status"] == "OBSERVED_EXPECTED_PACKAGE_IDENTITY"

    publisher_drift = package("public-id")
    publisher_drift["organization"]["title"] = "Other"
    assert observer.observe(publisher_drift, config(), "rights-sha")["status"] == "BLOCK_PUBLISHER_DRIFT"

    license_drift = package("public-id")
    license_drift["license_title"] = "Other License"
    assert observer.observe(license_drift, config(), "rights-sha")["status"] == "BLOCK_LICENSE_DRIFT"


def test_observation_identity_is_deterministic() -> None:
    first = observer.observe(package(), config(), "rights-sha")
    second = observer.observe(package(), config(), "rights-sha")
    assert first == second
    core = {key: value for key, value in first.items() if key != "observation_sha256"}
    assert first["observation_sha256"] == observer.hashlib.sha256(
        observer.canonical_json(core)
    ).hexdigest()
