from __future__ import annotations

import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "validate_dclm_data_quality_protocol_v1.py"
SPEC = importlib.util.spec_from_file_location("dclm_protocol", TOOL_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DclmDataQualityProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = MODULE.load_json(ROOT / "configs" / "research" / "dclm_data_quality_protocol_v1.json")
        cls.evidence = MODULE.load_json(ROOT / "evidence" / "research" / "dclm_data_quality_synthetic_v1.json")

    def mutate(self):
        return copy.deepcopy(self.protocol), copy.deepcopy(self.evidence)

    def test_valid_fixture_is_deterministic_and_non_authorizing(self):
        first = MODULE.compare_evidence(self.protocol, self.evidence)
        second = MODULE.compare_evidence(self.protocol, self.evidence)
        self.assertEqual(first, second)
        self.assertEqual(first["comparison"]["winner_arm_id"], "FILTER_A_SYNTHETIC")
        self.assertEqual(first["recommendation_state"], "CANDIDATE_RECOMMENDATION_ONLY")
        self.assertFalse(first["training_authorized"])
        self.assertFalse(first["automatic_adoption_allowed"])
        self.assertFalse(first["final_test_accessed"])
        self.assertFalse(first["paid_compute_used"])
        self.assertEqual(first["report_sha256"], MODULE.canonical_sha256({k: v for k, v in first.items() if k != "report_sha256"}))

    def test_missing_arm_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"] = evidence["arms"][:1]
        with self.assertRaisesRegex(MODULE.ProtocolError, "at least 2"):
            MODULE.compare_evidence(protocol, evidence)

    def test_duplicate_arm_id_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["arm_id"] = evidence["arms"][0]["arm_id"]
        with self.assertRaisesRegex(MODULE.ProtocolError, "duplicate arm_id"):
            MODULE.compare_evidence(protocol, evidence)

    def test_unequal_budget_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["budget"]["value"] = 999
        with self.assertRaisesRegex(MODULE.ProtocolError, "equal comparison budgets"):
            MODULE.compare_evidence(protocol, evidence)

    def test_different_input_snapshot_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["input_snapshot_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.ProtocolError, "identical input snapshot"):
            MODULE.compare_evidence(protocol, evidence)

    def test_unhashed_input_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][0]["input_snapshot_sha256"] = "mutable-latest"
        with self.assertRaisesRegex(MODULE.ProtocolError, "64-hex SHA-256"):
            MODULE.compare_evidence(protocol, evidence)

    def test_metric_direction_mismatch_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][0]["metric"]["direction"] = "minimize"
        with self.assertRaisesRegex(MODULE.ProtocolError, "metric direction"):
            MODULE.compare_evidence(protocol, evidence)

    def test_nonfinite_score_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][0]["metric"]["score"] = math.inf
        with self.assertRaisesRegex(MODULE.ProtocolError, "must be finite"):
            MODULE.compare_evidence(protocol, evidence)

    def test_ambiguous_tie_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["metric"]["score"] = 0.64
        evidence["arms"][2]["metric"]["score"] = 0.64
        with self.assertRaisesRegex(MODULE.ProtocolError, "ambiguous best-arm tie"):
            MODULE.compare_evidence(protocol, evidence)

    def test_failed_hard_gate_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["hard_gates"]["privacy"] = "FAIL"
        with self.assertRaisesRegex(MODULE.ProtocolError, "privacy is not PASS"):
            MODULE.compare_evidence(protocol, evidence)

    def test_extra_hard_gate_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][1]["hard_gates"]["quality"] = "PASS"
        with self.assertRaisesRegex(MODULE.ProtocolError, "hard-gate keys"):
            MODULE.compare_evidence(protocol, evidence)

    def test_training_authorization_rejected(self):
        protocol, evidence = self.mutate()
        evidence["training_authorized"] = True
        with self.assertRaisesRegex(MODULE.ProtocolError, "may not authorize training"):
            MODULE.compare_evidence(protocol, evidence)

    def test_arm_adopted_rejected(self):
        protocol, evidence = self.mutate()
        evidence["arms"][0]["promotion_state"] = "ADOPTED"
        with self.assertRaisesRegex(MODULE.ProtocolError, "may not claim ADOPTED"):
            MODULE.compare_evidence(protocol, evidence)

    def test_top_level_adopted_rejected(self):
        protocol, evidence = self.mutate()
        evidence["requested_promotion_state"] = "ADOPTED"
        with self.assertRaisesRegex(MODULE.ProtocolError, "automatic ADOPTED"):
            MODULE.compare_evidence(protocol, evidence)

    def test_protocol_cannot_enable_paid_compute(self):
        protocol, evidence = self.mutate()
        protocol["authority_boundaries"]["paid_compute_authorized"] = True
        with self.assertRaisesRegex(MODULE.ProtocolError, "paid_compute_authorized must remain false"):
            MODULE.compare_evidence(protocol, evidence)

    def test_protocol_cannot_enable_foreign_pretrained_weights(self):
        protocol, evidence = self.mutate()
        protocol["authority_boundaries"]["foreign_pretrained_weights_allowed"] = True
        with self.assertRaisesRegex(MODULE.ProtocolError, "foreign_pretrained_weights_allowed must remain false"):
            MODULE.compare_evidence(protocol, evidence)

    def test_json_loader_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"score": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(MODULE.ProtocolError, "non-finite JSON"):
                MODULE.load_json(path)

    def test_report_write_is_byte_deterministic(self):
        report = MODULE.compare_evidence(self.protocol, self.evidence)
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.json"
            right = Path(tmp) / "right.json"
            MODULE.write_report(left, report)
            MODULE.write_report(right, report)
            self.assertEqual(left.read_bytes(), right.read_bytes())


if __name__ == "__main__":
    unittest.main()
