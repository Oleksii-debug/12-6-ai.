from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.data.cross_source_dedup_v4_intake import (
    BLOCKED_OBJECTS,
    BLOCKED_UPSTREAM,
    READY,
    evaluate_v4_intake,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/next100_065_cross_source_dedup_v4_intake_v1.json"
CONVERGENCE_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"


def _load() -> tuple[dict, dict]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(CONVERGENCE_PATH.read_text(encoding="utf-8")),
    )


def _terminalize(config: dict) -> None:
    upstream = config["upstream_source_convergence"]
    upstream["workflow_status"] = "completed"
    upstream["workflow_conclusion"] = "success"


def test_repository_snapshot_has_complete_objects_but_waits_for_terminal_upstream() -> None:
    config, convergence = _load()
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_UPSTREAM
    assert report["blockers"] == ["upstream_source_convergence_ci_not_terminal_success"]
    assert report["validated_positive_credit_authority_count"] == 4
    assert report["validated_object_manifest_count"] == 4
    assert report["validated_object_count"] == 11
    assert report["validated_object_capacity_bytes"] == 76662
    assert report["post_dedup_capacity_claimed"] is False
    assert report["training_authorized"] is False


def test_terminal_exact_upstream_makes_object_comparison_ready_only() -> None:
    config, convergence = _load()
    _terminalize(config)
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == READY
    assert report["ready_for_global_dedup_object_comparison"] is True
    assert report["post_dedup_capacity_claimed"] is False
    assert report["corpus_identity_claimed"] is False
    assert report["tokenizer_fit_authorized"] is False
    assert report["training_authorized"] is False


def test_terminal_but_aggregate_only_handoff_blocks() -> None:
    config, convergence = _load()
    _terminalize(config)
    config["object_manifests"] = []
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert len([item for item in report["blockers"] if item.startswith("missing_positive_credit_object_manifest:")]) == 4


def test_unbound_positive_credit_authority_blocks() -> None:
    config, convergence = _load()
    _terminalize(config)
    omitted = config["required_positive_credit_late_authorities"].pop()
    config["object_manifests"] = [
        manifest for manifest in config["object_manifests"] if manifest["worker_id"] != omitted["worker_id"]
    ]
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert f"unbound_positive_credit_late_authority:{omitted['worker_id']}" in report["blockers"]


def test_duplicate_stable_object_identity_blocks() -> None:
    config, convergence = _load()
    _terminalize(config)
    duplicate = config["object_manifests"][0]["source_objects"][0]["stable_object_id"]
    config["object_manifests"][1]["source_objects"][0]["stable_object_id"] = duplicate
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert any(item.startswith("duplicate_global_stable_object_id:") for item in report["blockers"])


def test_missing_content_hash_blocks() -> None:
    config, convergence = _load()
    _terminalize(config)
    row = config["object_manifests"][0]["source_objects"][0]
    row.pop("expected_raw_sha256")
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert any("object_record_missing_content_identity" in item for item in report["blockers"])


def test_unsupported_comparison_normalization_blocks() -> None:
    config, convergence = _load()
    _terminalize(config)
    config["object_manifests"][0]["source_objects"][0]["comparison_normalization"] = "UNBOUND_NORMALIZATION"
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert any("object_record_unsupported_comparison_normalization" in item for item in report["blockers"])


def test_object_family_count_and_capacity_are_authority_bound() -> None:
    config, convergence = _load()
    _terminalize(config)
    manifest = config["object_manifests"][0]
    manifest["source_objects"][0]["source_family"] = "wrong.family"
    manifest["source_objects"].pop()
    report = evaluate_v4_intake(config, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert any(item.startswith("object_record_family_mismatch:") for item in report["blockers"])
    assert any(item.startswith("object_manifest_count_mismatch:") for item in report["blockers"])
    assert any(item.startswith("object_manifest_capacity_mismatch:") for item in report["blockers"])


def test_zero_credit_cpython_never_enters_dedup() -> None:
    config, convergence = _load()
    _terminalize(config)
    assert evaluate_v4_intake(config, convergence)["status"] == READY

    poisoned = copy.deepcopy(config)
    zero = poisoned["zero_credit_late_authorities"][0]
    poisoned["object_manifests"].append(
        {
            "worker_id": zero["worker_id"],
            "authority_identity": "0" * 64,
            "source_objects": [
                {
                    "source_id": "forbidden.cpython.credit",
                    "source_family": zero["family_id"],
                    "stable_origin_id": "forbidden",
                    "stable_object_id": "sha256:" + "1" * 64,
                    "origin_key": "forbidden:cpython",
                    "modality": "en",
                    "declared_capacity_bytes": 1,
                    "expected_raw_sha256": "2" * 64,
                }
            ],
        }
    )
    report = evaluate_v4_intake(poisoned, convergence)
    assert report["status"] == BLOCKED_OBJECTS
    assert any(item.startswith("unexpected_positive_credit_object_manifest:") for item in report["blockers"])
    assert any(item.startswith("zero_credit_authority_must_not_enter_dedup:") for item in report["blockers"])
