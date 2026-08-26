import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_next100_023_ua_wikibooks.py"
SPEC = importlib.util.spec_from_file_location("next100_023", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


class TestNext100023(unittest.TestCase):
    def test_static_authority_is_fail_closed(self):
        data = json.loads(
            (ROOT / "configs" / "data" / "next100_023_ua_wikibooks_source_audit_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(data["worker_id"], "NEXT100-023-DATA-UA-WIKIBOOKS")
        self.assertEqual(data["verdict"], "RETEST")
        self.assertEqual(data["family_lineage"]["independent_family_credit"], 0)
        self.assertEqual(data["rights"]["evaluation"], "NOT_SEPARATELY_ADMITTED")
        self.assertEqual(
            data["candidate"]["snapshot"]["sha1"],
            "6975ba549f822ea2394567743fdb3564c36e048a",
        )

    def test_normalization_is_deterministic(self):
        a = mod.normalize_wikitext("  Україна  \r\n\r\n\r\n текст\tтест ")
        b = mod.normalize_wikitext(" Україна \n\n текст тест ")
        self.assertEqual(a, b)
        self.assertEqual(mod.sha256_text(a), mod.sha256_text(b))

    def test_language_metrics_detect_ukrainian_specific_letters(self):
        letters, cyrillic, specific = mod.language_metrics("Їжак і ґрунт — це українські слова")
        self.assertGreater(letters, 0)
        self.assertGreater(cyrillic, 0)
        self.assertGreaterEqual(specific, 4)

    def test_family_credit_does_not_follow_hostname(self):
        data = json.loads(
            (ROOT / "configs" / "data" / "next100_023_ua_wikibooks_source_audit_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("not sufficient", data["family_lineage"]["rule"])
        self.assertIn("Wikipedia", " ".join(data["family_lineage"]["required_before_family_credit"]))
        self.assertIn("Wikisource", " ".join(data["family_lineage"]["required_before_family_credit"]))


if __name__ == "__main__":
    unittest.main()
