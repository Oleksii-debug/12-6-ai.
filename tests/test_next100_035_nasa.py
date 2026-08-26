#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "qualify_next100_035_nasa.py"
spec = importlib.util.spec_from_file_location("qualify_next100_035_nasa", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Next100035NasaTests(unittest.TestCase):
    def test_normalization_is_deterministic_and_text_only(self) -> None:
        got = mod.normalize_text("  Flight\u00a0Test  ", "A\r\n  technical\tabstract.  ")
        self.assertEqual(got, "Flight Test\n\nA\ntechnical abstract.\n")
        self.assertEqual(got, mod.normalize_text("  Flight\u00a0Test  ", "A\r\n  technical\tabstract.  "))

    def test_author_gate_accepts_civil_and_explicit_nasa_affiliation(self) -> None:
        record = {
            "authorAffiliations": [
                {"userType": "CIVIL", "meta": {}},
                {
                    "userType": "OTHER",
                    "meta": {"organization": {"name": "NASA Glenn Research Center"}},
                },
            ]
        }
        passed, reasons = mod.nasa_author_gate(record)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_author_gate_rejects_non_nasa_contractor(self) -> None:
        record = {
            "authorAffiliations": [
                {
                    "userType": "CONTRACTOR",
                    "meta": {"organization": {"name": "Example Aerospace LLC"}},
                }
            ]
        }
        passed, reasons = mod.nasa_author_gate(record)
        self.assertFalse(passed)
        self.assertTrue(any(reason.startswith("AUTHOR_0_NOT_NASA_CIVIL") for reason in reasons))

    def test_privacy_gate_rejects_contact_patterns(self) -> None:
        reasons = mod.privacy_gate("Contact alice@example.org or 202-555-0123.")
        self.assertIn("EMAIL_ADDRESS_IN_TRAINING_TEXT", reasons)
        self.assertIn("PHONE_PATTERN_IN_TRAINING_TEXT", reasons)

    def test_near_duplicate_metric_is_symmetric(self) -> None:
        left = mod.shingles("one two three four five six seven eight")
        right = mod.shingles("one two three four five six seven nine")
        self.assertEqual(mod.jaccard(left, right), mod.jaccard(right, left))
        self.assertLess(mod.jaccard(left, right), 1.0)

    def test_qualify_rejects_unresolved_third_party_metadata(self) -> None:
        config = {
            "required_distribution": "PUBLIC",
            "copyright_determinations_allowed": ["GOV_PUBLIC_USE_PERMITTED", "PUBLIC_USE_PERMITTED"],
            "required_license_type": "NO",
            "minimum_words_per_record": 1,
            "normalization_policy": "NASA_NTRS_TITLE_ABSTRACT_NFKC_WS_V1",
        }
        record = {
            "id": 123,
            "distribution": "PUBLIC",
            "status": "CURATED",
            "copyright": {
                "determinationType": "GOV_PUBLIC_USE_PERMITTED",
                "licenseType": "NO",
                "containsThirdPartyMaterial": None,
                "belongsToContractor": False,
                "belongsToPublisher": False,
            },
            "authorAffiliations": [{"userType": "CIVIL", "meta": {}}],
            "title": "A technical title",
            "abstract": "A sufficiently technical abstract for this isolated gate test.",
            "exportControl": {"ear": "NO", "itar": "NO"},
            "sensitiveInformation": None,
        }
        old_fetch = mod.fetch_record
        try:
            mod.fetch_record = lambda document_id: (b"{}", record)
            with tempfile.TemporaryDirectory() as tmp:
                result = mod.qualify_one(config, 123, Path(tmp))
        finally:
            mod.fetch_record = old_fetch
        self.assertEqual(result["decision"], "REJECT")
        self.assertIn("THIRD_PARTY_CONTENT_NOT_EXPLICITLY_FALSE", result["reasons"])


if __name__ == "__main__":
    unittest.main()
