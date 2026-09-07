from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "probe_kmu_secretariat_timeline_v2.py"
SPEC = importlib.util.spec_from_file_location("probe_kmu_secretariat_timeline_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class TimelineProbeCompositionTests(unittest.TestCase):
    def test_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            m.build_report(1, 0, 1.0)
        with self.assertRaises(ValueError):
            m.build_report(1, 1, 0.99)

    def test_inventory_is_wired_into_existing_inspector_and_credit_stays_zero(self):
        inventory = {
            "identity_sha256": "d" * 64,
            "news_urls": ["https://www.kmu.gov.ua/news/a", "https://www.kmu.gov.ua/news/b"],
            "training_authorized_bytes": 0,
            "family_count_credit_added": 0,
        }
        rows = [
            {
                "url": inventory["news_urls"][0],
                "eligible_probe_record": True,
                "normalized_probe_sha256": "a" * 64,
                "normalized_probe_bytes": 101,
            },
            {
                "url": inventory["news_urls"][1],
                "eligible_probe_record": True,
                "normalized_probe_sha256": "a" * 64,
                "normalized_probe_bytes": 101,
            },
        ]
        with patch.object(m.discovery, "discover", return_value=inventory), patch.object(
            m.probe, "inspect_page", side_effect=rows
        ), patch.object(m.time, "sleep") as sleep:
            report = m.build_report(5, 10, 1.05)
        self.assertEqual(report["sample"]["attempted_pages"], 2)
        self.assertEqual(report["sample"]["eligible_pages"], 2)
        self.assertEqual(report["sample"]["duplicate_eligible_payloads"], 1)
        self.assertEqual(report["sample"]["observed_unique_probe_bytes"], 101)
        self.assertEqual(report["probe_boundary"]["training_authorized_bytes"], 0)
        self.assertEqual(report["probe_boundary"]["family_count_credit_added"], 0)
        self.assertEqual(report["verdict"], "PROBE_USEFUL_OBSERVED_YIELD")
        sleep.assert_called_once_with(1.05)

    def test_inspector_failure_is_retained_as_rejection_not_credit(self):
        inventory = {
            "identity_sha256": "e" * 64,
            "news_urls": ["https://www.kmu.gov.ua/news/a"],
            "training_authorized_bytes": 0,
            "family_count_credit_added": 0,
        }
        with patch.object(m.discovery, "discover", return_value=inventory), patch.object(
            m.probe, "inspect_page", side_effect=UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")
        ):
            report = m.build_report(1, 1, 1.0)
        self.assertEqual(report["sample"]["eligible_pages"], 0)
        self.assertEqual(report["sample"]["observed_unique_probe_bytes"], 0)
        self.assertEqual(report["verdict"], "PROBE_INSUFFICIENT_OBSERVED_YIELD")
        self.assertIn("fetch_or_parse_error:UnicodeDecodeError", report["sample"]["rows"][0]["rejection_reasons"])

    def test_identity_is_deterministic(self):
        inventory = {
            "identity_sha256": "f" * 64,
            "news_urls": [],
            "training_authorized_bytes": 0,
            "family_count_credit_added": 0,
        }
        with patch.object(m.discovery, "discover", return_value=inventory):
            first = m.build_report(1, 1, 1.0)
            second = m.build_report(1, 1, 1.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
