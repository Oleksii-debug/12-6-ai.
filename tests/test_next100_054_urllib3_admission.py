from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twelve_six.next100_054_urllib3_admission import (
    AdmissionError,
    _git_blob_sha1,
    _luhn_valid,
    _near_jaccard,
    _scan_privacy,
    _scan_secrets,
    _sha256,
    _cjson,
    load_manifest,
    validate_manifest,
    validate_report,
)


class Next100054Urllib3AdmissionTests(unittest.TestCase):
    def test_manifest_is_exact_bounded_contract(self) -> None:
        manifest = load_manifest(Path("."))
        self.assertEqual(manifest["worker_id"], "NEXT100-054-CODE-URLLIB3")
        self.assertEqual(manifest["upstream"]["commit"], "9a950b92d999f906b6020bb2d1076ee56cddd5d2")
        self.assertEqual(manifest["license_review"]["license_id"], "MIT")
        self.assertEqual(len(manifest["selected_files"]), 8)
        self.assertEqual(manifest["candidate_bytes"], 228836)
        self.assertEqual(manifest["evaluation_boundary"]["selection_records_at_review"], 0)
        self.assertEqual(manifest["evaluation_boundary"]["final_test_records_at_review"], 0)

    def test_manifest_rejects_non_training_only_boundary(self) -> None:
        manifest = load_manifest(Path("."))
        altered = json.loads(json.dumps(manifest))
        altered["evaluation_boundary"]["candidate_role"] = "EVALUATION"
        with self.assertRaises(AdmissionError):
            validate_manifest(altered)

    def test_git_blob_identity(self) -> None:
        self.assertEqual(
            _git_blob_sha1(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_secret_and_privacy_scanners(self) -> None:
        self.assertIn("github_token", _scan_secrets(b"token=ghp_abcdefghijklmnopqrstuvwxyz1234"))
        self.assertIn("us_ssn_like", _scan_privacy(b"value 123-45-6789 value"))
        self.assertTrue(_luhn_valid("4111111111111111"))

    def test_near_duplicate_metric(self) -> None:
        self.assertEqual(_near_jaccard("a b c d e f", "a b c d e f"), 1.0)
        self.assertLess(_near_jaccard("alpha beta gamma delta epsilon", "one two three four five"), 0.85)

    def test_report_identity_validation(self) -> None:
        report = {
            "schema_version": "12-6.next100-054-urllib3-code-admission-report.v1",
            "source_head": "a" * 40,
            "verdict": "ADMIT_TRAINING_ONLY",
            "terminal": True,
            "selection": {"object_count": 8, "raw_bytes": 228836},
            "deduplication": {"status": "PASS"},
            "evaluation_boundary": {"selected_blob_intersection_count": 0},
            "source_family_decision": {
                "decision": "SEPARATE_CURRENT_BOUNDED_FAMILIES_WITH_LINEAGE_CAVEAT"
            },
        }
        report["report_identity_sha256"] = _sha256(_cjson(report))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            validated = validate_report(path, "a" * 40)
        self.assertEqual(validated["verdict"], "ADMIT_TRAINING_ONLY")


if __name__ == "__main__":
    unittest.main()
