from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import materialize_d03_rada_trees_primary_archive as subject


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pin_snapshot() -> dict:
    payload = {
        "schema_version": "12-6.d03-rada-trees-hf-object-identity.v1",
        "execution_profile": "LOCAL_FREE_METADATA_ONLY",
        "source": {
            "repo_id": subject.REPO_ID,
            "repo_type": "dataset",
            "revision": subject.REVISION,
            "tree_endpoint": "https://huggingface.co/api/datasets/example",
        },
        "files": [
            {
                "path": "Rada_Trees.7z",
                "size_bytes": 536_000_000,
                "git_blob_oid": "a" * 40,
                "xet_hash": "b" * 64,
            },
            {
                "path": "rada_xtag_texts.7z",
                "size_bytes": 698_000_000,
                "git_blob_oid": "c" * 40,
                "xet_hash": "d" * 64,
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
            "safe_result": "EXACT_HF_OBJECT_IDENTITIES_PINNED_DOWNLOAD_AND_MEMBER_AUDIT_REQUIRED",
        },
    }
    payload["snapshot_identity_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _listing(*records: str) -> str:
    return "Path = archive.7z\nType = 7z\n\n----------\n" + "\n\n".join(records) + "\n"


def test_inventory_is_deterministic_and_sorted() -> None:
    listing = _listing(
        "Path = z.txt\nSize = 4\nAttributes = A\nEncrypted = -\nCRC = ABCD",
        "Path = a/one.txt\nSize = 3\nAttributes = A\nEncrypted = -\nCRC = EFGH",
    )
    first = subject.build_member_inventory(
        listing, max_single_member_bytes=10, max_total_uncompressed_bytes=20
    )
    second = subject.build_member_inventory(
        listing, max_single_member_bytes=10, max_total_uncompressed_bytes=20
    )
    assert first == second
    assert [row["path"] for row in first["members"]] == ["a/one.txt", "z.txt"]
    assert first["member_count"] == 2
    assert first["total_uncompressed_bytes"] == 7


@pytest.mark.parametrize(
    "path",
    ["../escape.txt", "/abs.txt", "a\\b.txt", "C:evil.txt", "a/../b.txt"],
)
def test_inventory_rejects_unsafe_paths(path: str) -> None:
    listing = _listing(f"Path = {path}\nSize = 1\nAttributes = A\nEncrypted = -")
    with pytest.raises(subject.MaterializationError):
        subject.build_member_inventory(
            listing, max_single_member_bytes=10, max_total_uncompressed_bytes=20
        )


def test_inventory_rejects_case_collisions_and_encryption() -> None:
    listing = _listing(
        "Path = A.txt\nSize = 1\nAttributes = A\nEncrypted = -",
        "Path = a.TXT\nSize = 1\nAttributes = A\nEncrypted = -",
    )
    with pytest.raises(subject.MaterializationError):
        subject.build_member_inventory(
            listing, max_single_member_bytes=10, max_total_uncompressed_bytes=20
        )

    encrypted = _listing("Path = secret.txt\nSize = 1\nAttributes = A\nEncrypted = +")
    with pytest.raises(subject.MaterializationError):
        subject.build_member_inventory(
            encrypted, max_single_member_bytes=10, max_total_uncompressed_bytes=20
        )


def test_inventory_rejects_link_like_and_bombs() -> None:
    linked = _listing(
        "Path = link\nSize = 1\nAttributes = A lrwxrwxrwx\nEncrypted = -\n"
        "Symbolic Link = ../../outside"
    )
    with pytest.raises(subject.MaterializationError):
        subject.build_member_inventory(
            linked, max_single_member_bytes=10, max_total_uncompressed_bytes=20
        )

    large = _listing("Path = huge.txt\nSize = 11\nAttributes = A\nEncrypted = -")
    with pytest.raises(subject.MaterializationError):
        subject.build_member_inventory(
            large, max_single_member_bytes=10, max_total_uncompressed_bytes=20
        )


def test_pin_snapshot_is_fail_closed_on_training_credit() -> None:
    snapshot = _pin_snapshot()
    snapshot["claim_boundary"]["training_authorized_bytes"] = 1
    copy = dict(snapshot)
    copy.pop("snapshot_identity_sha256")
    snapshot["snapshot_identity_sha256"] = hashlib.sha256(_canonical(copy)).hexdigest()
    with pytest.raises(subject.MaterializationError):
        subject.validate_pin_snapshot(snapshot)


def test_report_identity_and_truth_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _pin_snapshot()
    inventory = subject.build_member_inventory(
        _listing("Path = plain/1990.txt\nSize = 5\nAttributes = A\nEncrypted = -"),
        max_single_member_bytes=50_000_000,
        max_total_uncompressed_bytes=10_000_000_000,
    )

    monkeypatch.setattr(
        subject.parent_probe,
        "load_and_validate",
        lambda: {
            "acquisition_policy": {
                "max_single_member_bytes": 50_000_000,
                "max_total_uncompressed_bytes": 10_000_000_000,
            }
        },
    )
    report = subject.build_report(
        snapshot,
        archive_path=Path("Rada_Trees.7z"),
        archive_bytes=536_000_000,
        archive_sha256="e" * 64,
        inventory=inventory,
        seven_zip_runtime="7-Zip 24.09",
    )
    subject.validate_report(report)
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["claim_boundary"]["member_content_sha256_verified"] is False
    assert report["claim_boundary"]["archive_members_extracted"] is False

    report["claim_boundary"]["training_authorized_bytes"] = 1
    with pytest.raises(subject.MaterializationError):
        subject.validate_report(report)
