from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/acquire_d03_rada_trees_primary_archive.py"
SPEC = importlib.util.spec_from_file_location("rada_acquire", TOOL)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def records_text() -> str:
    return """Path = plain/1990/session_001.txt
Size = 1234
Packed Size = 500
Modified = 2025-01-01 00:00:00
Attributes = A
CRC = ABCDEF12
Encrypted = -
Method = LZMA2
Block = 0

Path = annotations/session_001.conllu
Size = 4321
Packed Size = 800
Attributes = A
CRC = 12345678
Encrypted = -
Method = LZMA2
Block = 0

Path = emptydir
Size = 0
Folder = +
Attributes = D
Encrypted = -
"""


class RadaTreesArchiveAcquisitionTests(unittest.TestCase):
    def test_parse_and_normalize_inventory(self) -> None:
        parsed = module.parse_7z_slt(records_text())
        members = module.normalize_inventory(
            parsed,
            max_single_member_bytes=50_000_000,
            max_total_uncompressed_bytes=10_000_000_000,
        )
        self.assertEqual(len(members), 3)
        by_path = {item["path"]: item for item in members}
        self.assertEqual(
            by_path["plain/1990/session_001.txt"]["classification"],
            "PLAIN_TEXT_CANDIDATE_EXTENSION_ONLY",
        )
        self.assertEqual(
            by_path["annotations/session_001.conllu"]["classification"],
            "ANNOTATION_OR_STRUCTURED_DERIVATIVE_HOLD",
        )
        self.assertEqual(by_path["emptydir"]["kind"], "directory")

    def test_parent_traversal_path_is_rejected(self) -> None:
        records = module.parse_7z_slt("Path = ../escape.txt\nSize = 1\nAttributes = A\n\n")
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                records,
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_absolute_windows_path_is_rejected(self) -> None:
        records = module.parse_7z_slt("Path = C:\\escape.txt\nSize = 1\nAttributes = A\n\n")
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                records,
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_casefold_collision_is_rejected(self) -> None:
        text = (
            "Path = A.txt\nSize = 1\nAttributes = A\n\n"
            "Path = a.TXT\nSize = 1\nAttributes = A\n\n"
        )
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                module.parse_7z_slt(text),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_symlink_is_rejected(self) -> None:
        text = "Path = link.txt\nSize = 0\nAttributes = A\nSymbolic Link = ../../secret\n\n"
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                module.parse_7z_slt(text),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_encrypted_member_is_rejected(self) -> None:
        text = "Path = secret.txt\nSize = 10\nAttributes = A\nEncrypted = +\n\n"
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                module.parse_7z_slt(text),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_member_size_limit_is_fail_closed(self) -> None:
        text = "Path = huge.txt\nSize = 101\nAttributes = A\n\n"
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                module.parse_7z_slt(text),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=1000,
            )

    def test_total_uncompressed_limit_is_fail_closed(self) -> None:
        text = (
            "Path = a.txt\nSize = 60\nAttributes = A\n\n"
            "Path = b.txt\nSize = 60\nAttributes = A\n\n"
        )
        with self.assertRaises(module.AcquisitionError):
            module.normalize_inventory(
                module.parse_7z_slt(text),
                max_single_member_bytes=100,
                max_total_uncompressed_bytes=100,
            )

    def test_duplicate_technical_field_is_rejected(self) -> None:
        with self.assertRaises(module.AcquisitionError):
            module.parse_7z_slt("Path = a.txt\nSize = 1\nSize = 2\n\n")


if __name__ == "__main__":
    unittest.main()
