from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/scan_d03_rada_trees_secondary_plaintext.py"
SPEC = importlib.util.spec_from_file_location("d03_secondary_plaintext_scan", TOOL)
assert SPEC is not None and SPEC.loader is not None
scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)


def _split_sizes(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _listing() -> list[dict[str, object]]:
    text_sizes = _split_sizes(scan.EXPECTED_TEXT_LISTED_BYTES, scan.EXPECTED_TEXT_MEMBER_COUNT)
    xtag_total = scan.EXPECTED_UNCOMPRESSED_BYTES - scan.EXPECTED_TEXT_LISTED_BYTES
    xtag_sizes = _split_sizes(xtag_total, scan.EXPECTED_TEXT_MEMBER_COUNT)
    rows: list[dict[str, object]] = []
    for index, size in enumerate(text_sizes):
        rows.append({"path": f"texts/doc-{index:04d}.txt", "size_bytes": size})
    for index, size in enumerate(xtag_sizes):
        rows.append({"path": f"xtag/VRU/doc-{index:04d}.xtag", "size_bytes": size})
    rows.sort(key=lambda row: str(row["path"]))
    return rows


class SecondaryPlaintextScanTests(unittest.TestCase):
    def test_terminal_evidence_is_exactly_bound(self) -> None:
        evidence = scan.load_terminal_evidence()
        self.assertEqual(evidence["source"]["content_sha256"], scan.EXPECTED_CONTENT_SHA256)
        self.assertEqual(evidence["runtime"]["runtime_identity_sha256"], scan.EXPECTED_RUNTIME_IDENTITY)
        self.assertEqual(evidence["classification"]["plain_text_suffix_member_count"], 4391)

    def test_selects_only_exact_text_layer_under_separate_envelope(self) -> None:
        selected = scan.select_plaintext_listing(_listing())
        self.assertEqual(len(selected), scan.EXPECTED_TEXT_MEMBER_COUNT)
        self.assertEqual(sum(int(row["size_bytes"]) for row in selected), scan.EXPECTED_TEXT_LISTED_BYTES)
        self.assertTrue(all(str(row["path"]).startswith("texts/") for row in selected))
        self.assertTrue(all(str(row["path"]).endswith(".txt") for row in selected))
        self.assertTrue(all("xtag" not in str(row["path"]).lower() for row in selected))

    def test_rejects_txt_suffix_outside_texts_prefix(self) -> None:
        listing = _listing()
        for row in listing:
            if str(row["path"]).startswith("texts/"):
                row["path"] = "xtag/VRU/attacker.txt"
                break
        listing.sort(key=lambda row: str(row["path"]))
        with self.assertRaisesRegex(scan.PlaintextScanError, "outside texts/"):
            scan.select_plaintext_listing(listing)

    def test_rejects_listing_count_or_byte_drift(self) -> None:
        listing = _listing()
        with self.assertRaisesRegex(scan.PlaintextScanError, "member count drift"):
            scan.select_plaintext_listing(listing[:-1])
        listing = _listing()
        listing[0]["size_bytes"] = int(listing[0]["size_bytes"]) + 1
        with self.assertRaisesRegex(scan.PlaintextScanError, "byte total drift"):
            scan.select_plaintext_listing(listing)

    def test_exact_duplicate_collapse_is_candidate_only_and_deterministic(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(scan.EXPECTED_TEXT_MEMBER_COUNT):
            digest = hashlib.sha256(f"payload-{index}".encode()).hexdigest()
            rows.append(
                {
                    "path": f"texts/doc-{index:04d}.txt",
                    "size_bytes": 10,
                    "sha256": digest,
                    "classification": "PLAIN_TEXT_CANDIDATE",
                    "decoded_encoding": "utf-8-sig",
                    "text_metrics": {},
                    "path_year_hints": [],
                    "text_emitted": False,
                }
            )
        rows[1]["sha256"] = rows[0]["sha256"]
        summary = scan.summarize_classified(rows)
        self.assertEqual(summary["plain_text_candidate_members"], scan.EXPECTED_TEXT_MEMBER_COUNT)
        self.assertEqual(summary["plain_text_candidate_exact_unique_payload_count"], scan.EXPECTED_TEXT_MEMBER_COUNT - 1)
        self.assertEqual(summary["plain_text_exact_duplicate_group_count"], 1)
        self.assertEqual(summary["plain_text_exact_duplicate_member_discount"], 1)
        self.assertEqual(summary["plain_text_exact_duplicate_byte_discount"], 10)
        self.assertFalse(summary["raw_member_text_emitted"])

    def test_tool_never_selects_xtag_for_extraction(self) -> None:
        source = Path(scan.__file__).read_text(encoding="utf-8")
        self.assertIn('TEXT_PREFIX = "texts/"', source)
        self.assertIn('TEXT_SUFFIX = ".txt"', source)
        self.assertIn('"xtag_extracted": False', source)
        self.assertNotIn('TEXT_SUFFIX = ".xtag"', source)


if __name__ == "__main__":
    unittest.main()
