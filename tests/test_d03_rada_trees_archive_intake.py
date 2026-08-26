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


class RadaTreesArchiveIntakeTests(unittest.TestCase):
    def test_parse_safe_listing(self) -> None:
        members = tool.parse_7z_slt(LISTING)
        self.assertEqual([m["path"] for m in members], ["plain/1990/session-001.txt", "ud", "ud/session-001.conllu"])
        self.assertEqual(sum(m["size"] for m in members), 33)
        self.assertFalse(members[0]["is_directory"])
        self.assertTrue(members[1]["is_directory"])

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(LISTING.replace("plain/1990/session-001.txt", "../escape.txt"))

    def test_absolute_path_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(LISTING.replace("plain/1990/session-001.txt", "/tmp/escape.txt"))

    def test_duplicate_normalized_path_rejected(self) -> None:
        listing = LISTING + """
Path = plain/1990/session-001.txt
Size = 1
Attributes = A
"""
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_archive_link_rejected(self) -> None:
        listing = LISTING.replace("Attributes = A\n\nPath = ud", "Attributes = A\nSymbolic Link = ../../escape\n\nPath = ud", 1)
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_member_size_bound_rejected(self) -> None:
        listing = LISTING.replace("\nSize = 12\n", f"\nSize = {tool.MAX_MEMBER_BYTES + 1}\n", 1)
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(listing)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload"
            path.write_bytes(b"rada-trees-fixture")
            self.assertEqual(tool.sha256_file(path), hashlib.sha256(b"rada-trees-fixture").hexdigest())

    def test_backslash_path_rejected(self) -> None:
        with self.assertRaises(tool.IntakeError):
            tool.parse_7z_slt(LISTING.replace("plain/1990/session-001.txt", r"plain\1990\session.txt"))


if __name__ == "__main__":
    unittest.main()
