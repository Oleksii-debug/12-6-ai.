import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_open_lm_bootstrap_stress_v1 import canonical_hash, validate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/research/open_lm_bootstrap_stress_v1.json"


class OpenLMBootstrapStressV1Tests(unittest.TestCase):
    def load(self):
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    def write_temp(self, payload):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return Path(handle.name)

    def test_canonical_manifest_passes(self):
        result = validate(MANIFEST)
        self.assertEqual(result["status"], "PASS")

    def test_manifest_identity_is_deterministic(self):
        payload = self.load()
        self.assertEqual(
            canonical_hash(payload, "evidence_identity_sha256"),
            payload["evidence_identity_sha256"],
        )
        self.assertEqual(canonical_hash(payload, "evidence_identity_sha256"), canonical_hash(payload, "evidence_identity_sha256"))

    def test_base_sha_drift_fails_closed(self):
        payload = self.load()
        payload["project_base_sha"] = "0" * 40
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "project base SHA drift"):
            validate(path)

    def test_upstream_commit_drift_fails_closed(self):
        payload = self.load()
        payload["upstream"]["immutable_commit"] = "1" * 40
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "upstream commit drift"):
            validate(path)

    def test_unverified_tag_cannot_be_bound(self):
        payload = self.load()
        payload["upstream"]["tag_or_release"] = "v9.9.9"
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "unexpected unverified tag/release binding"):
            validate(path)

    def test_floating_requirement_inventory_is_required(self):
        payload = self.load()
        payload["upstream_requirements"]["floating_or_lower_bound_entries"] = []
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "floating dependency inventory unexpectedly empty"):
            validate(path)

    def test_missing_exact_requirement_fails_closed(self):
        payload = self.load()
        payload["upstream_requirements"]["entries"] = [x for x in payload["upstream_requirements"]["entries"] if x != "pandas==2.1.4"]
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "missing exact upstream pandas pin"):
            validate(path)

    def test_fabricated_install_artifact_hash_fails_closed(self):
        payload = self.load()
        payload["installation_attempt"]["artifact_sha256"] = "a" * 64
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "unavailable artifact must not have a fabricated hash"):
            validate(path)

    def test_runtime_pass_claim_fails_closed(self):
        payload = self.load()
        payload["runtime"]["execution_status"] = "PASS"
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "runtime cannot be promoted without execution"):
            validate(path)

    def test_parity_claim_fails_closed(self):
        payload = self.load()
        payload["runtime"]["parity_proven"] = True
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "parity cannot be true"):
            validate(path)

    def test_foreign_weight_flag_fails_closed(self):
        payload = self.load()
        payload["canonical_base_safety"]["foreign_pretrained_weights_used"] = True
        payload["evidence_identity_sha256"] = canonical_hash(payload, "evidence_identity_sha256")
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "canonical Base safety violation"):
            validate(path)

    def test_evidence_tamper_fails_closed(self):
        payload = self.load()
        payload["network"]["pypi_reachable"] = True
        path = self.write_temp(payload)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "evidence identity mismatch"):
            validate(path)

    def test_deep_copy_does_not_change_identity(self):
        payload = self.load()
        clone = copy.deepcopy(payload)
        self.assertEqual(
            canonical_hash(payload, "evidence_identity_sha256"),
            canonical_hash(clone, "evidence_identity_sha256"),
        )


if __name__ == "__main__":
    unittest.main()
