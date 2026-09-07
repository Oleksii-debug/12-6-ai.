from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "discover_kmu_news_timeline_v1.py"
SPEC = importlib.util.spec_from_file_location("discover_kmu_news_timeline_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class TimelineDiscoveryTests(unittest.TestCase):
    def test_canonical_news_url(self):
        self.assertEqual(
            m.canonical_news_url("/news/example?utm_source=x#frag"),
            "https://www.kmu.gov.ua/news/example",
        )

    def test_cross_origin_and_non_news_rejected(self):
        self.assertIsNone(m.canonical_news_url("https://example.com/news/x"))
        self.assertIsNone(m.canonical_news_url("/npas/x"))
        self.assertIsNone(m.canonical_news_url("/news"))

    def test_timeline_page_is_strict_and_canonical(self):
        self.assertEqual(
            m.canonical_timeline_page("/timeline?type=posts&page=2"),
            "https://www.kmu.gov.ua/timeline?type=posts&page=2",
        )
        self.assertEqual(
            m.canonical_timeline_page("https://kmu.gov.ua/timeline?page=02&type=posts#x"),
            "https://www.kmu.gov.ua/timeline?type=posts&page=2",
        )
        self.assertIsNone(m.canonical_timeline_page("/timeline?type=documents&page=2"))
        self.assertIsNone(m.canonical_timeline_page("/timeline?type=posts&page=0"))
        self.assertIsNone(m.canonical_timeline_page("/timeline?type=posts&page=2&lang=en"))
        self.assertIsNone(m.canonical_timeline_page("https://evil.test/timeline?type=posts&page=2"))

    def test_fetch_rejects_same_origin_redirect_route_drift(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        response.geturl.return_value = "https://www.kmu.gov.ua/news/not-a-timeline"
        response.read.return_value = b"unexpected"
        with patch.object(m.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "timeline redirect drift rejected"):
                m.fetch(m.SEED)

    def test_fetch_accepts_canonical_host_redirect_only(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        response.geturl.return_value = "https://kmu.gov.ua/timeline?type=posts"
        response.read.return_value = b"ok"
        with patch.object(m.urllib.request, "urlopen", return_value=response):
            self.assertEqual(m.fetch(m.SEED), b"ok")

    def test_delay_and_page_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            m.discover(1, 0.99)
        with self.assertRaises(ValueError):
            m.discover(21, 1.0)

    def test_discovers_articles_and_follows_only_post_timeline_pagination(self):
        pages = {
            m.SEED: (
                b'<a href="/news/a">A</a>'
                b'<a href="/timeline?type=posts&page=2">next</a>'
                b'<a href="/timeline?type=documents&page=2">bad</a>'
                b'<a href="/timeline?type=posts&page=3&lang=en">bad-extra</a>'
            ),
            "https://www.kmu.gov.ua/timeline?type=posts&page=2": (
                b'<a href="https://kmu.gov.ua/news/b?x=1">B</a>'
                b'<a href="https://evil.test/news/c">C</a>'
            ),
        }

        def fake_fetch(url: str, timeout: float = 20.0) -> bytes:
            del timeout
            return pages[url]

        with patch.object(m, "fetch", side_effect=fake_fetch), patch.object(m.time, "sleep") as sleep:
            report = m.discover(5, 1.05)
        self.assertEqual(
            report["news_urls"],
            ["https://www.kmu.gov.ua/news/a", "https://www.kmu.gov.ua/news/b"],
        )
        self.assertEqual(report["pages_visited"], 2)
        self.assertEqual(report["training_authorized_bytes"], 0)
        self.assertEqual(report["family_count_credit_added"], 0)
        sleep.assert_called_once_with(1.05)

    def test_identity_is_deterministic(self):
        html = b'<a href="/news/a">A</a>'
        with patch.object(m, "fetch", return_value=html):
            first = m.discover(1, 1.0)
            second = m.discover(1, 1.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
