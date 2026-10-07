from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "probe_kmu_secretariat_bulk_v1.py"
SPEC = importlib.util.spec_from_file_location("probe_kmu_secretariat_bulk_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class KMuBulkProbeTests(unittest.TestCase):
    def test_canonicalize_news_url_strips_tracking(self):
        self.assertEqual(
            m.canonicalize_url("http://kmu.gov.ua/news/example/?utm_source=x#fragment"),
            "https://www.kmu.gov.ua/news/example",
        )

    def test_disallowed_paths_rejected(self):
        for url in (
            "https://www.kmu.gov.ua/api/news",
            "https://www.kmu.gov.ua/npas/example",
            "https://www.kmu.gov.ua/contact/example",
            "https://example.com/news/example",
        ):
            self.assertIsNone(m.canonicalize_url(url))

    def test_page_parser_accepts_exact_secretariat_fixture(self):
        article = " ".join(["Український уряд повідомив важливе рішення для громадян."] * 20)
        html = f"""
        <html><head><title>Тестова новина</title></head><body>
        <div>{m.SECRETARIAT_BYLINE}</div>
        <p>{article}</p>
        <footer>Весь контент доступний за ліцензією {m.LICENSE_MARKER}</footer>
        </body></html>
        """.encode()
        with patch.object(m, "fetch_bytes", return_value=(html, {"content-type": "text/html"})):
            row = m.inspect_page(m.SitemapEntry("https://www.kmu.gov.ua/news/test", "2026-09-01"))
        self.assertTrue(row["eligible_probe_record"])
        self.assertTrue(row["exact_secretariat_byline"])
        self.assertTrue(row["site_cc_by_marker"])
        self.assertGreaterEqual(row["word_count"], 60)

    def test_ministry_fixture_rejected(self):
        article = " ".join(["Український уряд повідомив важливе рішення для громадян."] * 20)
        html = f"""
        <html><body>
        <div>Міністерство економіки України</div>
        <p>{article}</p>
        <footer>Весь контент доступний за ліцензією {m.LICENSE_MARKER}</footer>
        </body></html>
        """.encode()
        with patch.object(m, "fetch_bytes", return_value=(html, {"content-type": "text/html"})):
            row = m.inspect_page(m.SitemapEntry("https://www.kmu.gov.ua/news/test", "2026-09-01"))
        self.assertFalse(row["eligible_probe_record"])
        self.assertIn("not_exact_secretariat_byline", row["rejection_reasons"])

    def test_other_license_marker_rejected(self):
        article = " ".join(["Український уряд повідомив важливе рішення для громадян."] * 20)
        html = f"""
        <html><body><div>{m.SECRETARIAT_BYLINE}</div><p>{article}</p>
        <footer>{m.LICENSE_MARKER} CC BY-NC-ND</footer></body></html>
        """.encode()
        with patch.object(m, "fetch_bytes", return_value=(html, {"content-type": "text/html"})):
            row = m.inspect_page(m.SitemapEntry("https://www.kmu.gov.ua/news/test", "2026-09-01"))
        self.assertFalse(row["eligible_probe_record"])
        self.assertIn("page_other_license_marker", row["rejection_reasons"])

    def test_contact_scalar_rejected(self):
        article = " ".join(["Український уряд повідомив важливе рішення для громадян."] * 20)
        html = f"""
        <html><body><div>{m.SECRETARIAT_BYLINE}</div>
        <p>{article} test@example.gov.ua</p>
        <footer>{m.LICENSE_MARKER}</footer></body></html>
        """.encode()
        with patch.object(m, "fetch_bytes", return_value=(html, {"content-type": "text/html"})):
            row = m.inspect_page(m.SitemapEntry("https://www.kmu.gov.ua/news/test", "2026-09-01"))
        self.assertFalse(row["eligible_probe_record"])
        self.assertIn("contact_like_scalar_present", row["rejection_reasons"])

    def _valid_report(self):
        core = {
            "schema_version": m.SCHEMA,
            "repository": m.REPOSITORY,
            "execution_profile": "LOCAL_FREE",
            "parent_authority": {
                "pr": m.PARENT_PR,
                "head_sha": m.PARENT_HEAD,
                "manifest_identity_sha256": m.PARENT_MANIFEST,
                "family_id": m.FAMILY_ID,
                "license_id": "CC-BY-4.0",
                "training_rights_parent_verdict": "ADMIT_BOUNDED_SIX_ONLY",
            },
            "probe_boundary": {
                "training_authorized_bytes": 0,
                "tokenizer_fit_executed": False,
                "optimizer_updates": 0,
                "final_test_accessed": False,
                "paid_compute_used": False,
                "canonical_registry_mutated": False,
                "family_count_credit_added": 0,
                "probe_is_not_bulk_admission": True,
            },
            "robots_policy": {
                "sitemap": m.SITEMAP_URL,
                "crawl_delay_seconds_minimum": 1.0,
                "configured_delay_seconds": 1.05,
                "disallowed_paths_respected": True,
            },
            "selection_contract": {},
            "sitemap": {"discovered_news_url_count": 0, "snapshots": []},
            "sample": {
                "requested_fetch_limit": 1,
                "attempted_pages": 0,
                "eligible_pages": 0,
                "eligible_fraction": 0.0,
                "duplicate_normalized_payload_count": 0,
                "observed_unique_probe_bytes": 0,
                "sampled_response_bytes": 0,
                "family_cap_bytes": m.FAMILY_CAP_BYTES,
                "observed_fraction_of_family_cap": 0.0,
                "rows": [],
            },
        }
        report = dict(core)
        report["probe_identity_sha256"] = m.canonical_sha256(core)
        report["verdict"] = "PROBE_INSUFFICIENT_OBSERVED_YIELD"
        return report

    def test_valid_probe_report_passes(self):
        m.validate_report(self._valid_report())

    def test_training_credit_mutation_rejected(self):
        report = self._valid_report()
        report["probe_boundary"]["training_authorized_bytes"] = 1
        with self.assertRaises(ValueError):
            m.validate_report(report)

    def test_family_credit_mutation_rejected(self):
        report = self._valid_report()
        report["probe_boundary"]["family_count_credit_added"] = 1
        with self.assertRaises(ValueError):
            m.validate_report(report)

    def test_parent_authority_drift_rejected(self):
        report = self._valid_report()
        report["parent_authority"]["head_sha"] = "0" * 40
        with self.assertRaises(ValueError):
            m.validate_report(report)

    def test_crawl_delay_below_robots_rejected(self):
        report = self._valid_report()
        report["robots_policy"]["configured_delay_seconds"] = 0.5
        with self.assertRaises(ValueError):
            m.validate_report(report)

    def test_self_hash_drift_rejected(self):
        report = copy.deepcopy(self._valid_report())
        report["sample"]["family_cap_bytes"] += 1
        with self.assertRaises(ValueError):
            m.validate_report(report)


if __name__ == "__main__":
    unittest.main()
