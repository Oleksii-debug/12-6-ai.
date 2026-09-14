from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "twelve_six" / "next100_043_flask_admission.py"
_SPEC = importlib.util.spec_from_file_location("next100_043_flask_admission", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load standalone Flask admission verifier: {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

AdmissionError = _MODULE.AdmissionError
MANIFEST_PATH = _MODULE.MANIFEST_PATH
_git_blob_sha1 = _MODULE._git_blob_sha1
_near_jaccard = _MODULE._near_jaccard
_scan_privacy = _MODULE._scan_privacy
_scan_secrets = _MODULE._scan_secrets
_verify_license = _MODULE._verify_license
validate_manifest = _MODULE.validate_manifest


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

    def test_license_semantics_accept_line_wrapped_grant(self) -> None:
        data = (
            b"Redistribution and use in source and binary forms, with or without\n"
            b"modification, are permitted.\n"
            b"Neither the name of the copyright holder may be used.\n"
            b"THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"
            b'\"AS IS\" without warranties.\n'
        )
        manifest = {
            "license_review": {
                "license_url": "https://example.invalid/LICENSE.txt",
                "license_size_bytes": len(data),
                "license_blob_sha1": _git_blob_sha1(data),
                "license_id": "BSD-3-Clause",
                "redistribution_conditions": "retain notice and disclaimer",
            }
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(_MODULE, "_download", return_value=data),
        ):
            evidence, raw = _verify_license(manifest, Path(tmpdir))
            self.assertEqual(
                (Path(tmpdir) / "rights-evidence" / "LICENSE.txt").read_bytes(),
                data,
            )
        self.assertEqual(raw, data)
        self.assertEqual(evidence["license_blob_sha1"], _git_blob_sha1(data))
        self.assertEqual(evidence["training_purpose_decision"], "ALLOWED")

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
