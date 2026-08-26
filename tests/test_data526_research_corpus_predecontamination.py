from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/data526_research_corpus_v1_predecontamination_intake_v1.json"
VALIDATOR = ROOT / "tools/validate_data526_research_corpus_predecontamination.py"

spec = importlib.util.spec_from_file_location("data526_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Data526PredecontaminationIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_current_canonical_nonterminal_cutoff_is_valid_blocked_evidence(self) -> None:
        report = module.validate(copy.deepcopy(self.data))
        self.assertEqual(report["status"], "PASS_BLOCKED")
        self.assertEqual(report["dependency_pr"], 538)
        self.assertFalse(report["candidate_manifest_materialized"])

    def test_nonterminal_successor_cannot_be_consumed_as_authority(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["source_convergence_dependency"]["consumed_as_authority"] = True
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_blocked_state_cannot_publish_source_objects(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["predecontamination_candidate_manifest"]["source_object_ids"] = ["forged-source"]
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_blocked_state_cannot_publish_composite_identity(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["predecontamination_candidate_manifest"]["composite_identity_sha256"] = "0" * 64
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_dependency_head_or_registry_drift_fails_closed(self) -> None:
        for key, value in (
            ("head_sha", "f" * 40),
            ("candidate_registry_identity_sha256", "0" * 64),
        ):
            broken = copy.deepcopy(self.data)
            broken["source_convergence_dependency"][key] = value
            with self.assertRaises(module.ValidationError):
                module.validate(broken)

    def test_generic_ci_cannot_be_promoted_to_terminal_authority(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["source_convergence_dependency"]["observed_generic_ci"]["accepted_as_terminal_authority"] = True
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_training_tokenizer_and_decontamination_remain_blocked(self) -> None:
        for key in ("training_authorized_now", "tokenizer_fit_authorized_now", "decontamination_authorized_now"):
            broken = copy.deepcopy(self.data)
            broken["handoff"][key] = True
            with self.assertRaises(module.ValidationError):
                module.validate(broken)

    def test_unsafe_execution_claims_fail_closed(self) -> None:
        for key in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
            broken = copy.deepcopy(self.data)
            broken[key] = True
            with self.assertRaises(module.ValidationError):
                module.validate(broken)


if __name__ == "__main__":
    unittest.main()
