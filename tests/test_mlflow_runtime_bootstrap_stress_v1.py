from __future__ import annotations

import json
from pathlib import Path

from tools.verify_mlflow_runtime_bootstrap_stress_v1 import (
    EXPECTED_LICENSE_BLOB,
    EXPECTED_MAIN_SHA,
    EXPECTED_PARENT_HEAD,
    EXPECTED_SOURCE_VERSION,
    EXPECTED_TRACKING_BLOB,
    EXPECTED_UPSTREAM_COMMIT,
    identity_sha256,
    validate_contract,
    validate_metadata,
    validate_tracking_uri,
)

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_identity_contract():
    config = load(ROOT / "configs/research/mlflow_runtime_bootstrap_stress_v1.json")
    assert config["project"]["base_main_sha"] == EXPECTED_MAIN_SHA
    assert config["project"]["parent_head_sha"] == EXPECTED_PARENT_HEAD
    assert config["upstream"]["commit_sha"] == EXPECTED_UPSTREAM_COMMIT
    assert config["upstream"]["license_blob_sha"] == EXPECTED_LICENSE_BLOB
    assert config["upstream"]["tracking_source_blob_sha"] == EXPECTED_TRACKING_BLOB
    assert config["upstream"]["source_version"] == EXPECTED_SOURCE_VERSION
    assert not validate_contract(config)


def test_contract_rejects_identity_drift():
    config = load(ROOT / "configs/research/mlflow_runtime_bootstrap_stress_v1.json")
    config["upstream"]["commit_sha"] = "0" * 40
    assert "upstream commit drift" in validate_contract(config)


def test_tracking_uri_boundary():
    assert validate_tracking_uri("file:///tmp/mlruns")
    assert validate_tracking_uri("sqlite:///tmp/mlflow.db")
    assert not validate_tracking_uri("https://example.invalid/mlruns")
    assert not validate_tracking_uri("file://user:secret@host/tmp/mlruns")


def test_secret_metadata_boundary():
    assert validate_metadata({"run_id": "abc", "metric": 1})
    assert not validate_metadata({"api_key": "secret"})
    assert not validate_metadata({"Authorization": "Bearer secret"})


def test_evidence_identity_is_deterministic():
    payload = {"b": 2, "a": 1}
    assert identity_sha256(payload) == identity_sha256({"a": 1, "b": 2})
