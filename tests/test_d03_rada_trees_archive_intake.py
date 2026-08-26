from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_d03_rada_trees_archive.py"
SPEC = importlib.util.spec_from_file_location("rada_trees_intake", TOOL_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

LISTING = """7-Zip 24.09

Path = Rada_Trees.7z
Type = 7z
Physical Size = 123

----------
Path = plain/1990/session-001.txt
Size = 12
Packed Size = 10
Attributes = A

Path = ud
Size = 0
Attributes = D

Path = ud/session-001.conllu
Size = 21
Packed Size = 18
Attributes = A
"""


def make_snapshot(primary_size: int = 123) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": tool.OBJECT_SNAPSHOT_SCHEMA,
        "execution_profile": "LOCAL_FREE_METADATA_ONLY",
        "source": {
            "repo_id": tool.DATASET,
            "repo_type": "dataset",
            "revision": tool.DATASET_HEAD,
            "tree_endpoint": "https://example.invalid/exact-tree",
        },
        "files": [
            {
                "path": tool.PRIMARY_ARCHIVE,
                "size_bytes": primary_size,
                "git_blob_oid": "1" * 40,
                "xet_hash": "2" * 64,
            },
            {
                "path": "rada_xtag_texts.7z",
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
    value["snapshot_identity_sha256"] = tool._canonical_sha256(value)
    return value


class RadaTreesArchiveIntakeTests(unittest.TestCase):
    def test_parse_safe_listing(self) -> None:
        members = tool.parse_7z_slt(LISTING)
        self.assertEqual(
            [member["path"] for member in members],
            ["plain/1990/session-001.txt", "ud", "ud/session-001.conllu"],
        )
        self.assertEqual(sum(member["size"] for member in members), 33)
        self.assertFalse(members[0]["is_directory"])
        self.assertTrue(members[1]["is_directory"])

    def test_valid_object_snapshot_binds_primary_archive(self) -> None:
        primary = tool.validate_object_snapshot(make_snapshot())
        self.assertEqual(primary["path"], tool.PRIMARY_ARCHIVE)
        self.assertEqual(primary["size_bytes"], 123)

    def test_object_snapshot_self_hash_tamper_rejected(self) -> None:
        snapshot = make_snapshot()
        snapshot["source"]["revision"] = "0" * 40  # type: ignore[index]
        with self.assertRaises(tool.IntakeError):
            tool.validate_object_snapshot(snapshot)

    def test_object_snapshot_training_credit_rejected(self) -> None:
        snapshot = make_snapshot()
        snapshot["claim_boundary"]["training_authorized_bytes"] = 1  # type: ignore[index]
        snapshot["snapshot_identity_sha256"] = tool._canonical_sha256(
            {key: value for key, value in snapshot.items() if key != "snapshot_identity_sha256"}
        )
        with self.assertRaises(tool.IntakeError):
            tool.validate_object_snapshot(snapshot)

    def test_downloaded_size_must_match_object_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / tool.PRIMARY_ARCHIVE
            archive.write_bytes(b"abc")
            snapshot = make_snapshot(primary_size=4)
            with self.assertRaises(tool.IntakeError):
                tool.build_report(
                    archive,
                    hashlib.sha256(b"abc").hexdigest(),
                    snapshot,
                    executable="definitely-not-needed",
                )

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(
                LISTING.replace("plain/1990/session-001.txt", "../escape.txt")
            )

    def test_absolute_path_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(
                LISTING.replace("plain/1990/session-001.txt", "/tmp/escape.txt")
            )

    def test_duplicate_normalized_path_rejected(self) -> None:
        listing = LISTING + """
Path = plain/1990/session-001.txt
Size = 1
Attributes = A
"""
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_archive_link_rejected(self) -> None:
        listing = LISTING.replace(
            "Attributes = A\n\nPath = ud",
            "Attributes = A\nSymbolic Link = ../../escape\n\nPath = ud",
            1,
        )
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_member_size_bound_rejected(self) -> None:
        listing = LISTING.replace(
            "\nSize = 12\n",
            f"\nSize = {tool.MAX_MEMBER_BYTES + 1}\n",
            1,
        )
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload"
            path.write_bytes(b"rada-trees-fixture")
            self.assertEqual(
                tool.sha256_file(path),
                hashlib.sha256(b"rada-trees-fixture").hexdigest(),
            )

    def test_backslash_path_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(
                LISTING.replace(
                    "plain/1990/session-001.txt",
                    r"plain\1990\session.txt",
                )
            )


if __name__ == "__main__":
    unittest.main()
