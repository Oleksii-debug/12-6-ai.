from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'materialize_data324_kubernetes_ua.py'
)
SPEC = importlib.util.spec_from_file_location('data324_materializer', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Data324ContractTests(unittest.TestCase):
    def test_normalization_removes_english_comments_and_frontmatter(self) -> None:
        raw = (
            b'---\ntitle: test\n---\n'
            b'<!-- English original should disappear. -->\n'
            + 'Український текст із Kubernetes та об’єктами API.\n'.encode('utf-8')
        )
        normalized = MODULE.normalize_markdown_uk(raw)
        self.assertNotIn('English original', normalized)
        self.assertNotIn('title:', normalized)
        self.assertIn('Український текст', normalized)
        self.assertTrue(normalized.endswith('\n'))

    def test_language_gate_requires_ukrainian_signal(self) -> None:
        text = ('Це український технічний текст із літерами і ї є ґ. ' * 20).strip()
        evidence = MODULE.language_evidence(text)
        self.assertEqual(evidence['decision'], 'PASS')
        with self.assertRaises(RuntimeError):
            MODULE.language_evidence('This is only English technical text. ' * 20)

    def test_git_blob_identity_is_content_bound(self) -> None:
        self.assertEqual(
            MODULE.git_blob_sha1(b'test\n'),
            '9daeafb9864cf43055ae93beb0afd6c7d144bfa4',
        )
        self.assertNotEqual(
            MODULE.git_blob_sha1(b'test\n'),
            MODULE.git_blob_sha1(b'test'),
        )


if __name__ == '__main__':
    unittest.main()
