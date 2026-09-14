from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_mlflow_local_tracking_qualification.py"
SPEC = importlib.util.spec_from_file_location("mlflow_qualification", TOOL)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def contract():
    return mod.load_json(
        ROOT / "configs" / "research" / "mlflow_local_tracking_qualification_v1.json"
    )


def run_input():
    return {
        "source_git_sha": "1" * 40,
        "run_manifest_sha256": "2" * 64,
        "checkpoint_manifest_sha256": "3" * 64,
        "checkpoint_id": "4" * 64,
        "tracking_uri": "sqlite:///mlflow-local.db",
        "params": {"batch_size": 8, "seed": 1337},
        "metrics": {"heldout_bpb": 3.25, "step": 10},
        "tags": {"stage": "research", "execution_profile": "LOCAL_FREE"},
        "artifact_references": [
            {
                "logical_name": "checkpoint-manifest",
                "sha256": "5" * 64,
                "byte_size": 1200,
            }
        ],
        "canonical_lineage_authority": "GIT_AND_PROJECT_MANIFESTS",
        "mlflow_role": "OPTIONAL_METADATA_SINK_ONLY",
    }


def test_contract_passes():
    mod.validate_contract(contract())


def test_evidence_is_deterministic_and_valid():
    cfg = contract()
    payload = run_input()
    first = mod.build_evidence(cfg, payload)
    second = mod.build_evidence(cfg, copy.deepcopy(payload))
    assert first == second
    mod.validate_evidence(cfg, first)


def test_declared_metric_change_changes_identity():
    cfg = contract()
    first = mod.build_evidence(cfg, run_input())
    changed = run_input()
    changed["metrics"]["heldout_bpb"] = 3.20
    second = mod.build_evidence(cfg, changed)
    assert first["evidence_sha256"] != second["evidence_sha256"]


@pytest.mark.parametrize(
    "uri",
    [
        "https://tracker.example.invalid",
        "http://127.0.0.1:5000",
        "databricks://profile",
        "postgresql://user:password@localhost/db",
        "",
    ],
)
def test_remote_or_implicit_tracking_uri_rejected(uri):
    payload = run_input()
    payload["tracking_uri"] = uri
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_local_file_uri_is_allowed():
    payload = run_input()
    payload["tracking_uri"] = "file:///tmp/12-6-mlflow"
    mod.build_evidence(contract(), payload)


def test_secret_like_metadata_key_rejected():
    payload = run_input()
    payload["tags"]["api_token"] = "redacted"
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_secret_like_metadata_value_rejected():
    payload = run_input()
    payload["tags"]["note"] = "password=do-not-store"
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_missing_exact_binding_rejected():
    payload = run_input()
    del payload["checkpoint_manifest_sha256"]
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_mlflow_cannot_be_lineage_authority():
    payload = run_input()
    payload["canonical_lineage_authority"] = "MLFLOW"
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_unhashed_artifact_payload_rejected():
    payload = run_input()
    payload["artifact_references"][0].pop("sha256")
    with pytest.raises(mod.ContractError):
        mod.build_evidence(contract(), payload)


def test_contract_cannot_self_adopt():
    cfg = contract()
    cfg["promotion"]["adopted"] = True
    with pytest.raises(mod.ContractError):
        mod.validate_contract(cfg)


def test_upstream_identity_drift_rejected():
    cfg = contract()
    cfg["upstream"]["git_sha"] = "0" * 40
    with pytest.raises(mod.ContractError):
        mod.validate_contract(cfg)


def test_evidence_tamper_rejected():
    cfg = contract()
    evidence = mod.build_evidence(cfg, run_input())
    evidence["run"]["metrics"]["heldout_bpb"] = 1.0
    with pytest.raises(mod.ContractError):
        mod.validate_evidence(cfg, evidence)


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(mod.ContractError):
        mod.load_json(path)


def test_canonical_json_survives_round_trip():
    cfg = contract()
    evidence = mod.build_evidence(cfg, run_input())
    restored = json.loads(json.dumps(evidence, sort_keys=True))
    mod.validate_evidence(cfg, restored)
