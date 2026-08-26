from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/validate_next100_062_license_compliance_manifest.py"
spec = importlib.util.spec_from_file_location("next100_062_compliance", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class LicenseComplianceManifestTests(unittest.TestCase):
    def test_manifest_and_training_purpose_pass(self) -> None:
        manifest = mod.validate_manifest()
        self.assertEqual(manifest["source_count"], 6)
        mod.assert_training_authorized()

    def test_canonical_digest_is_deterministic(self) -> None:
        manifest = mod.validate_manifest()
        self.assertEqual(mod.canonical_digest(manifest), mod.canonical_digest(manifest))
        self.assertEqual(len(mod.canonical_digest(manifest)), 64)

    def test_redistribution_fails_closed_on_unresolved_share_alike(self) -> None:
        with self.assertRaisesRegex(mod.ComplianceError, "redistribution fail-closed"):
            mod.assert_redistributable()

    def test_missing_required_sidecar_fails_closed(self) -> None:
        path = ROOT / "data/compliance/next100_062/required/requests-NOTICE.txt"
        hidden = path.with_name(path.name + ".test-hidden")
        path.rename(hidden)
        try:
            with self.assertRaisesRegex(mod.ComplianceError, "required sidecar missing"):
                mod.validate_manifest()
        finally:
            hidden.rename(path)

    def test_no_training_license_is_promoted_to_evaluation(self) -> None:
        manifest = mod.validate_manifest()
        for source in manifest["sources"]:
            purposes = source["project_purpose_authorization"]
            self.assertEqual(purposes["evaluation"], "NOT_SEPARATELY_ADMITTED")
            self.assertEqual(purposes["selection_validation"], "NOT_SEPARATELY_AUTHORIZED_BY_THIS_MANIFEST")
            self.assertEqual(purposes["final_test"], "NOT_SEPARATELY_AUTHORIZED_BY_THIS_MANIFEST")


if __name__ == "__main__":
    unittest.main()
