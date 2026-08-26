from __future__ import annotations

import hashlib
import unittest

from twelve_six.data.cross_source_capacity_audit_v4 import (
    ConvergedDedupError,
    _derive_kmu_rows,
    _derive_mdn_rows,
    _derive_verba_rows,
    _git_blob_sha1,
    normalize_mdn_prose,
    normalize_nist_extracted,
)


class Next100102ConvergedDedupTests(unittest.TestCase):
    def test_git_blob_identity_matches_git_object_rule(self) -> None:
        payload = b"abc\n"
        expected = hashlib.sha1(b"blob 4\0" + payload).hexdigest()
        self.assertEqual(_git_blob_sha1(payload), expected)

    def test_nist_normalization_matches_authority_rules(self) -> None:
        payload = normalize_nist_extracted("Ａ  \r\nmail@example.com\f\n\n\nTail  \n")
        self.assertEqual(payload.decode("utf-8"), "A\n<EMAIL>\n\nTail\n")

    def test_kmu_derivation_uses_authorized_normalized_capacity(self) -> None:
        authority = {
            "verdict": "ADMIT",
            "rights": {"training": "ALLOWED_PRETRAINING"},
            "source_family": {"family_id": "ua.kmu.portal.secretariat-news"},
            "aggregate": {"normalized_bytes": 11},
            "records": [{
                "id": "r1", "quality": "PASS", "raw_path": "data/r1.txt",
                "raw_bytes": 10, "raw_sha256": "a" * 64,
                "normalized_bytes": 11, "normalized_sha256": "b" * 64,
            }],
        }
        binding = {"worker_id": "K", "head_sha": "h", "workflow_run": 1}
        rows = _derive_kmu_rows(authority, binding)
        self.assertEqual(rows[0]["declared_capacity_bytes"], 11)
        self.assertEqual(rows[0]["expected_raw_bytes"], 10)

    def test_kmu_rejects_non_admit(self) -> None:
        with self.assertRaises(ConvergedDedupError):
            _derive_kmu_rows({"verdict": "RETEST"}, {})

    def test_verba_is_one_bounded_family_object(self) -> None:
        authority = {
            "verdict": "ADMIT",
            "scope": {"training_admitted": True},
            "family": {"source_family": "ua.verba.public-domain.nomis1864"},
            "snapshot": {"normalized_bytes": 1659, "normalized_sha256": "1" * 64, "normalized_path": "data/verba.txt"},
        }
        binding = {"worker_id": "V", "head_sha": "h", "workflow_run": 2}
        rows = _derive_verba_rows(authority, binding)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["declared_capacity_bytes"], 1659)

    def test_mdn_derivation_materializes_prose_only_capacity(self) -> None:
        authority = {
            "verdict": "ADMIT_PROSE_ONLY",
            "claim_boundary": {"training_source_authority_terminal": True, "prose_only": True},
            "family": {"family_id": "en.mdn.webdocs.prose"},
            "upstream": {"commit": "c" * 40},
            "pages": [{
                "path": "files/en-us/web/http/guides/compression/index.md",
                "quality_status": "PASS", "normalization_policy": "MDN_PROSE_ONLY_MARKDOWN_V1",
                "git_blob_sha1": "a" * 40, "raw_bytes": 100, "raw_sha256": "b" * 64,
                "normalized_bytes": 64, "normalized_sha256": "d" * 64,
            }],
        }
        binding = {"worker_id": "M", "head_sha": "h", "workflow_run": 3}
        rows = _derive_mdn_rows(authority, binding)
        self.assertEqual(rows[0]["declared_capacity_bytes"], 64)
        self.assertEqual(rows[0]["materialization_policy"], "MDN_PROSE_ONLY_MARKDOWN_V1")

    def test_mdn_normalizer_removes_frontmatter_and_inline_code(self) -> None:
        raw = b"---\ntitle: T\n---\n# Head\nUse `code` here.\n\nTail\n"
        self.assertEqual(normalize_mdn_prose(raw), b"Head\nUse here.\n\nTail\n")


if __name__ == "__main__":
    unittest.main()
