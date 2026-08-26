from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import materialize_d03_rada_trees_archive as intake
import probe_d03_rada_trees_hf_objects as hf_probe


def valid_slt() -> str:
    return """7-Zip 24.00
Path = Rada_Trees.7z
Type = 7z
Physical Size = 123

----------
Path = 1990/session-001.txt
Size = 12
Packed Size = 10
Attributes = A
Encrypted = -

Path = 1991/session-002.txt
Size = 7
Packed Size = 6
Attributes = A
Encrypted = -
"""


def hf_fixture() -> tuple[dict[str, object], dict[str, object]]:
    tree = [
        {
            "type": "file",
            "path": "Rada_Trees.7z",
            "size": 123,
            "oid": "1" * 40,
            "xetHash": "a" * 64,
        },
        {
            "type": "file",
            "path": "rada_xtag_texts.7z",
            "size": 456,
            "oid": "2" * 40,
            "xetHash": "b" * 64,
        },
    ]
    resolve = {
        "Rada_Trees.7z": "a" * 64,
        "rada_xtag_texts.7z": "b" * 64,
    }
    snapshot = hf_probe.build_snapshot(tree, resolve)
    return snapshot, snapshot["files"][0]


class RadaTreesArchiveIntakeTests(unittest.TestCase):
    def test_streaming_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sample.bin"
            payload = b"abc" * 100_000
            path.write_bytes(payload)
            self.assertEqual(
                intake.sha256_file(path, chunk_size=257),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_safe_listing_is_deterministic_and_sorted(self) -> None:
        members = intake.parse_7z_slt(
            valid_slt(),
            max_single_member_bytes=100,
            max_total_uncompressed_bytes=1000,
        )
        self.assertEqual(
            [item["path"] for item in members],
            ["1990/session-001.txt", "1991/session-002.txt"],
        )
        self.assertEqual(sum(item["size_bytes"] for item in members), 19)
        self.assertTrue(all(item["sha256"] is None for item in members))

    def test_traversal_and_backslash_paths_fail_closed(self) -> None:
        for bad in ("../escape.txt", "safe/../escape.txt", r"safe\escape.txt", "/abs.txt"):
            listing = valid_slt().replace("1990/session-001.txt", bad)
            with self.subTest(path=bad):
                with self.assertRaises(intake.ArchiveIntakeError):
                    intake.parse_7z_slt(
                        listing,
                        max_single_member_bytes=100,
                        max_total_uncompressed_bytes=1000,
                    )

    def test_casefold_collision_fails_closed(self) -> None:
        listing = valid_slt().replace(
            "1991/session-002.txt",
            "1990/SESSION-001.TXT",
        )
        with self.assertRaises(intake.ArchiveIntakeError):
            intake.parse_7z_slt(
                listing,
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=1000,
            )

    def test_link_and_encrypted_members_fail_closed(self) -> None:
        link_listing = valid_slt().replace(
            "Attributes = A\nEncrypted = -",
            "Attributes = A\nSymbolic Link = target\nEncrypted = -",
            1,
        )
        encrypted_listing = valid_slt().replace("Encrypted = -", "Encrypted = +", 1)
        for listing in (link_listing, encrypted_listing):
            with self.assertRaises(intake.ArchiveIntakeError):
                intake.parse_7z_slt(
                    listing,
                    max_single_member_bytes=100,
                    max_total_uncompressed_bytes=1000,
                )

    def test_expansion_budgets_fail_closed(self) -> None:
        with self.assertRaises(intake.ArchiveIntakeError):
            intake.parse_7z_slt(
                valid_slt(),
                max_single_member_bytes=10,
                max_total_uncompressed_bytes=1000,
            )
        with self.assertRaises(intake.ArchiveIntakeError):
            intake.parse_7z_slt(
                valid_slt(),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=18,
            )

    def test_report_binds_snapshot_and_cannot_grant_training_credit(self) -> None:
        snapshot, archive = hf_fixture()
        members = intake.parse_7z_slt(
            valid_slt(),
            max_single_member_bytes=100,
            max_total_uncompressed_bytes=1000,
        )
        report = intake.build_report(
            hf_snapshot=snapshot,
            hf_archive=archive,
            archive_sha256="c" * 64,
            archive_size_bytes=123,
            members=members,
            hashes_verified=False,
        )
        intake.validate_report(report)
        self.assertEqual(report["claim_boundary"]["training_authorized_bytes"], 0)
        self.assertFalse(report["claim_boundary"]["tokenizer_fit_authorized"])
        self.assertFalse(report["claim_boundary"]["model_training_authorized"])

        mutated = copy.deepcopy(report)
        mutated["claim_boundary"]["training_authorized_bytes"] = 1
        core = dict(mutated)
        core.pop("report_identity_sha256")
        mutated["report_identity_sha256"] = hashlib.sha256(
            intake.canonical_json_bytes(core)
        ).hexdigest()
        with self.assertRaises(intake.ArchiveIntakeError):
            intake.validate_report(mutated)

    def test_hash_verified_report_requires_every_member_hash(self) -> None:
        snapshot, archive = hf_fixture()
        members = intake.parse_7z_slt(
            valid_slt(),
            max_single_member_bytes=100,
            max_total_uncompressed_bytes=1000,
        )
        members[0]["sha256"] = "d" * 64
        with self.assertRaises(intake.ArchiveIntakeError):
            intake.build_report(
                hf_snapshot=snapshot,
                hf_archive=archive,
                archive_sha256="c" * 64,
                archive_size_bytes=123,
                members=members,
                hashes_verified=True,
            )


if __name__ == "__main__":
    unittest.main()
