from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tools.verify_open_source_bootstrap_stress_v1 as verifier


class BootstrapStressTests(unittest.TestCase):
    def test_path_safety_rejects_absolute_and_parent_paths(self) -> None:
        self.assertFalse(verifier.safe_relative("/tmp/escape.lock"))
        self.assertFalse(verifier.safe_relative("requirements/../escape.lock"))
        self.assertTrue(verifier.safe_relative("requirements/execution/linux-x86_64/cpu-runtime.lock.txt"))

    def test_canonical_json_identity_is_stable(self) -> None:
        value = {"b": 2, "a": [1, 3]}
        self.assertEqual(verifier.canonical_bytes(value), verifier.canonical_bytes(value))
        self.assertEqual(verifier.sha256_bytes(verifier.canonical_bytes(value)), verifier.sha256_bytes(verifier.canonical_bytes(value)))

    def test_report_is_explicitly_blocked_when_exact_runtime_is_unavailable(self) -> None:
        config = json.loads(verifier.CONFIG.read_text(encoding="utf-8"))
        report = verifier.build_report(config)
        self.assertEqual(report["runtime"]["status"], "NOT_EXECUTED")
        self.assertIn("EXACT_PYTHON_UNAVAILABLE", report["blockers"])
        self.assertIn("NO_LOCAL_EXACT_ARTIFACT_CACHE", report["blockers"])
        self.assertFalse(report["canonical_base_contamination"])
        self.assertFalse(report["paid_compute"])
        self.assertEqual(report["training_updates"], 0)

    def test_report_identity_rejects_reformatted_payload(self) -> None:
        config = json.loads(verifier.CONFIG.read_text(encoding="utf-8"))
        report = verifier.build_report(config)
        payload = dict(report)
        identity = payload.pop("evidence_identity_sha256")
        reformatted = json.dumps(payload, indent=4, sort_keys=True).encode("utf-8")
        self.assertEqual(identity, verifier.sha256_bytes(verifier.canonical_bytes(payload)))
        self.assertNotEqual(identity, verifier.sha256_bytes(reformatted))

    def test_contract_paths_do_not_touch_canonical_surfaces(self) -> None:
        config = json.loads(verifier.CONFIG.read_text(encoding="utf-8"))
        checks = {check["name"]: check["passed"] for check in verifier.static_contract_checks(config)}
        self.assertTrue(checks["canonical_surfaces_avoided"])
        self.assertTrue(checks["no_new_workflow"])
        self.assertTrue(checks["path_safety"])

    def test_report_roundtrip_is_machine_readable(self) -> None:
        config = json.loads(verifier.CONFIG.read_text(encoding="utf-8"))
        report = verifier.build_report(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema"], verifier.REPORT_SCHEMA)
        self.assertEqual(loaded["evidence_identity_sha256"], report["evidence_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
