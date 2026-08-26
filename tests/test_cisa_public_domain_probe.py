from __future__ import annotations

import unittest
from unittest import mock

from tools.probe_cisa_public_domain_sources import (
    ProbeError,
    assert_rights_phrases,
    build_report,
    normalize_text,
)


class CisaPublicDomainProbeTests(unittest.TestCase):
    def test_rights_phrase_accepts_pdf_line_wrapping(self) -> None:
        text = (
            "This publication is in the public domain.\n"
            "Authorization to copy and distribute this publication\n"
            "in whole or in part is granted."
        )
        assert_rights_phrases(
            text,
            [
                "this publication is in the public domain",
                "authorization to copy and distribute this publication "
                "in whole or in part is granted",
            ],
        )

    def test_rights_phrase_fails_closed_when_grant_is_missing(self) -> None:
        with self.assertRaisesRegex(ProbeError, "public-domain phrase missing"):
            assert_rights_phrases(
                "This is a public web page without a document-specific grant.",
                ["this publication is in the public domain"],
            )

    def test_normalization_redacts_email_and_is_deterministic(self) -> None:
        raw = "  Alpha\t beta\r\n\r\nContact: analyst@example.gov\fGamma  "
        first, first_count = normalize_text(raw)
        second, second_count = normalize_text(raw)
        self.assertEqual(first, second)
        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertNotIn("analyst@example.gov", first)
        self.assertIn("<EMAIL_REDACTED>", first)
        self.assertNotIn("\r", first)
        self.assertNotIn("\f", first)

    def test_probe_report_never_grants_training_credit(self) -> None:
        config = {
            "status": "PROBE_ONLY_NO_TRAINING_AUTHORITY",
            "source_family": "en.usgov.dhs.cisa-public-domain-guidance",
            "language": "en",
            "modality": "text",
            "allowed_download_hosts": ["www.cisa.gov"],
            "required_rights_phrases_casefold": [
                "this publication is in the public domain",
                "authorization to copy and distribute this publication "
                "in whole or in part is granted",
            ],
            "quality_minimums": {
                "normalized_utf8_bytes_per_document": 20000,
                "word_count_per_document": 2500,
                "alphabetic_character_ratio": 0.55,
            },
            "documents": [
                {
                    "publication_id": "fixture",
                    "title": "Fixture",
                    "publisher": "CISA",
                    "year": 2024,
                    "url": "https://www.cisa.gov/fixture.pdf",
                }
            ],
            "truth_boundary": {
                "training_authorized_exposure": 0,
                "corpus_admitted": False,
                "paid_compute_authorized": False,
            },
        }
        rights = (
            "This publication is in the public domain.\n"
            "Authorization to copy and distribute this publication "
            "in whole or in part is granted.\n"
        )
        extracted = rights + ("security engineering resilience guidance " * 900)

        with (
            mock.patch(
                "tools.probe_cisa_public_domain_sources._download_pdf",
                return_value=(b"%PDF-fixture", "https://www.cisa.gov/fixture.pdf"),
            ),
            mock.patch(
                "tools.probe_cisa_public_domain_sources._extract_pdf_text",
                return_value=extracted,
            ),
            mock.patch(
                "tools.probe_cisa_public_domain_sources._pdftotext_version",
                return_value="pdftotext fixture",
            ),
        ):
            report = build_report(config)

        self.assertEqual(report["training_authorized_exposure"], 0)
        self.assertIs(report["corpus_admitted"], False)
        self.assertEqual(report["documents"][0]["training_credit"], 0)
        self.assertNotIn("text", report["documents"][0])
        self.assertRegex(report["report_sha256"], r"^[0-9a-f]{64}$")

    def test_nonzero_training_authority_is_rejected_before_network(self) -> None:
        config = {
            "status": "PROBE_ONLY_NO_TRAINING_AUTHORITY",
            "truth_boundary": {
                "training_authorized_exposure": 1,
                "corpus_admitted": False,
                "paid_compute_authorized": False,
            },
        }
        with self.assertRaisesRegex(ProbeError, "zero training exposure"):
            build_report(config)


if __name__ == "__main__":
    unittest.main()
