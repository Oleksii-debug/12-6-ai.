from __future__ import annotations

import json
import unittest
from pathlib import Path

from twelve_six.next100_043_flask_admission import (
    AdmissionError,
    MANIFEST_PATH,
    _git_blob_sha1,
    _near_jaccard,
    _scan_privacy,
    _scan_secrets,
    validate_manifest,
)


class Next100043FlaskAdmissionTests(unittest.TestCase):
    def test_manifest_is_bounded_training_only_one_family(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        manifest = json.loads((repo / MANIFEST_PATH).read_text(encoding="utf-8"))
        validate_manifest(manifest)
        self.assertEqual(len(manifest["selected_files"]), 8)
        self.assertEqual(
            sum(item["size_bytes"] for item in manifest["selected_files"]), 183088
        )
        self.assertEqual(manifest["upstream"]["source_family"], "github:pallets/flask")
        self.assertFalse(manifest["evaluation_boundary"]["evaluation_use_authorized"])
        self.assertEqual(
            manifest["selection_policy"]["excluded_objects_count_as_capacity"], False
        )

    def test_git_blob_identity_is_content_sensitive(self) -> None:
        self.assertNotEqual(_git_blob_sha1(b"print(1)\n"), _git_blob_sha1(b"print(2)\n"))

    def test_secret_scan_rejects_key_material(self) -> None:
        self.assertIn(
            "private_key",
            _scan_secrets(b"-----BEGIN PRIVATE KEY-----\nnot-real\n"),
        )
        self.assertIn(
            "aws_access_key",
            _scan_secrets(b"x = 'AKIAABCDEFGHIJKLMNOP'\n"),
        )

    def test_privacy_scan_detects_ssn_like_material(self) -> None:
        self.assertIn("us_ssn_like", _scan_privacy(b"value = '123-45-6789'\n"))

    def test_near_dedup_detects_copy(self) -> None:
        text = "def f(x):\n    return x + 1\n" * 20
        self.assertEqual(_near_jaccard(text, text), 1.0)

    def test_manifest_fails_closed_on_evaluation_promotion(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        manifest = json.loads((repo / MANIFEST_PATH).read_text(encoding="utf-8"))
        manifest["evaluation_boundary"]["evaluation_use_authorized"] = True
        with self.assertRaises(AdmissionError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
