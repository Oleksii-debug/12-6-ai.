from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "classify_d03_rada_trees_members.py"
SPEC = importlib.util.spec_from_file_location("rada_trees_classify_parent", TOOL_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def parent_report() -> dict[str, object]:
    members = [{"path": "plain/2024/session.txt", "size_bytes": 5, "sha256": "a" * 64}]
    archive = {
        "path": tool.PRIMARY_ARCHIVE,
        "upstream_object_identity": tool.PINNED_XET,
        "sha256": tool.PINNED_SHA256,
        "sha256_authority": "HF_XET_POINTER_AT_EXACT_DATASET_REVISION",
        "size_bytes": 123,
    }
    inventory_payload = {
        "dataset_head_sha": tool.DATASET_HEAD,
        "archive_path": tool.PRIMARY_ARCHIVE,
        "upstream_object_identity": tool.PINNED_XET,
        "archive_sha256": tool.PINNED_SHA256,
        "archive_size_bytes": 123,
        "members": members,
    }
    report: dict[str, object] = {
        "schema_version": tool.PARENT_REPORT_SCHEMA,
        "state": "EXACT_ARCHIVE_AND_MEMBER_INVENTORY_MATERIALIZED_CLASSIFICATION_NOT_RUN",
        "parent_probe_head_sha": "92c1fd05d4399b0f0c4a35f0689160383f963c9c",
        "dataset_head_sha": tool.DATASET_HEAD,
        "archive": archive,
        "extractor": {"name": "7z", "version_observed": "fixture"},
        "member_count": 1,
        "uncompressed_bytes_observed": 5,
        "members": members,
        "inventory_identity_sha256": tool.inventory.sha256_bytes(
            tool.inventory.canonical_json(inventory_payload)
        ),
        "member_payload_classification": "NOT_RUN_SUCCESSOR_REQUIRED",
        "normalized_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "training_exposure_authorized": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "next_gate": "CLASSIFY_PLAIN_TEXT_AND_BIND_MEMBER_LEVEL_PROVENANCE",
    }
    stable = dict(report)
    stable["extractor"] = {"name": "7z"}
    report["report_identity_sha256"] = tool.inventory.sha256_bytes(
        tool.inventory.canonical_json(stable)
    )
    return report


def test_canonical_697_parent_report_shape_is_accepted() -> None:
    files = tool.verify_parent_report(parent_report())
    assert files == [{"path": "plain/2024/session.txt", "size_bytes": 5, "sha256": "a" * 64}]


def test_parent_report_training_credit_is_rejected() -> None:
    report = parent_report()
    report["training_authorized_bytes"] = 1
    try:
        tool.verify_parent_report(report)
    except tool.ClassificationError as exc:
        assert "parent training credit" in str(exc)
    else:
        raise AssertionError("mutated parent training credit was accepted")
