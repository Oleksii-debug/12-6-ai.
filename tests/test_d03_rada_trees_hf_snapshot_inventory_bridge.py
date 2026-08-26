from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/run_d03_rada_trees_inventory_from_hf_snapshot.py"
SPEC = importlib.util.spec_from_file_location("rada_hf_bridge", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def snapshot() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": module.SNAPSHOT_SCHEMA,
        "execution_profile": "LOCAL_FREE_METADATA_ONLY",
        "source": {
            "repo_id": module.REPO_ID,
            "repo_type": "dataset",
            "revision": module.REVISION,
            "tree_endpoint": "https://example.invalid/exact-tree",
        },
        "files": [
            {
                "path": module.PRIMARY_ARCHIVE,
                "size_bytes": 123,
                "git_blob_oid": "1" * 40,
                "xet_hash": module.inventory.PINNED_XET_HASH,
            },
            {
                "path": module.SECONDARY_ARCHIVE,
                "size_bytes": 456,
                "git_blob_oid": "3" * 40,
                "xet_hash": "4" * 64,
            },
        ],
        "verification": {
            "tree_revision_is_immutable_40hex": True,
            "git_blob_oids_bound": True,
            "xet_hashes_bound": True,
            "resolve_header_xet_hashes_match_tree": True,
        },
        "claim_boundary": {
            "archives_downloaded": False,
            "archive_content_sha256_verified": False,
            "archive_members_inventoried": False,
            "normalized_capacity_claimed": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "safe_result": (
                "EXACT_HF_OBJECT_IDENTITIES_PINNED_DOWNLOAD_AND_MEMBER_AUDIT_REQUIRED"
            ),
        },
    }
    value["snapshot_identity_sha256"] = module.snapshot_identity(value)
    return value


def rehash(value: dict[str, Any]) -> None:
    body = dict(value)
    body.pop("snapshot_identity_sha256", None)
    value["snapshot_identity_sha256"] = module.snapshot_identity(body)


def test_valid_snapshot_returns_exact_primary_object() -> None:
    value = snapshot()
    result = module.validate_hf_object_snapshot(value)
    assert result == {
        "snapshot_identity_sha256": value["snapshot_identity_sha256"],
        "size_bytes": 123,
        "git_blob_oid": "1" * 40,
        "xet_hash": module.inventory.PINNED_XET_HASH,
    }


def test_snapshot_self_hash_tamper_is_rejected() -> None:
    value = snapshot()
    value["source"]["revision"] = "0" * 40
    with pytest.raises(module.HandoffError, match="identity mismatch"):
        module.validate_hf_object_snapshot(value)


def test_training_credit_is_rejected_even_if_snapshot_is_rehashed() -> None:
    value = snapshot()
    value["claim_boundary"]["training_authorized_bytes"] = 1
    rehash(value)
    with pytest.raises(module.HandoffError, match="cannot authorize training"):
        module.validate_hf_object_snapshot(value)


def test_archive_inventory_order_or_path_drift_is_rejected() -> None:
    value = snapshot()
    value["files"].reverse()
    rehash(value)
    with pytest.raises(module.HandoffError, match="inventory/order drift"):
        module.validate_hf_object_snapshot(value)


def test_primary_xet_drift_is_rejected_even_if_snapshot_is_rehashed() -> None:
    value = snapshot()
    value["files"][0]["xet_hash"] = "0" * 64
    rehash(value)
    with pytest.raises(module.HandoffError, match="pinned source authority"):
        module.validate_hf_object_snapshot(value)


def test_build_bridge_derives_size_xet_and_content_sha_from_authorities() -> None:
    value = snapshot()
    config = {
        "primary_archive": {
            "exact_content_sha256": module.inventory.PINNED_CONTENT_SHA256,
        },
        "inventory_policy": {},
    }
    inner = {
        "archive": {
            "size_bytes": 123,
            "upstream_object_identity": module.inventory.PINNED_XET_HASH,
            "sha256": module.inventory.PINNED_CONTENT_SHA256,
        },
        "inventory_identity_sha256": "5" * 64,
        "training_authorized_bytes": 0,
    }
    with mock.patch.object(module.inventory, "build_report", return_value=inner) as build:
        report = module.build_bridge_report(
            config,
            Path("Rada_Trees.7z"),
            value,
            "7z",
        )
    build.assert_called_once_with(
        config,
        Path("Rada_Trees.7z"),
        module.inventory.PINNED_CONTENT_SHA256,
        123,
        module.inventory.PINNED_XET_HASH,
        "7z",
    )
    assert report["parent_object"]["git_blob_oid"] == "1" * 40
    assert report["parent_object"]["xet_hash"] == module.inventory.PINNED_XET_HASH
    assert report["parent_object"]["content_sha256"] == module.inventory.PINNED_CONTENT_SHA256
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert len(report["bridge_report_identity_sha256"]) == 64
