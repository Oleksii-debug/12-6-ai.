from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"
VALIDATOR = ROOT / "tools/validate_next100_063_source_registry_convergence.py"

spec = importlib.util.spec_from_file_location("next100_063_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Next100063SourceRegistryConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_current_authority_passes(self) -> None:
        report = module.validate(copy.deepcopy(self.data))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["capacity_bytes"],
            {"uk": 100856, "en": 144151, "code": 69133, "total": 314140},
        )
        self.assertEqual(
            report["family_counts"],
            {"uk": 4, "en": 2, "code": 4, "total": 10},
        )
        self.assertEqual(report["numeric_source_object_count"], 21)
        self.assertEqual(report["next_gate"], "GLOBAL_CROSS_SOURCE_DEDUP")

    def test_nonterminal_late_workflow_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["late_authorities"][0]["workflow_conclusion"] = "queued"
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_generic_workflow_cannot_substitute_for_dedicated_source_workflow(self) -> None:
        broken = copy.deepcopy(self.data)
        nist = next(
            row
            for row in broken["late_authorities"]
            if row["worker_id"] == "NEXT100-034-DATA-EN-NIST"
        )
        nist["workflow_run"] = 32998704439
        nist["workflow_name"] = "DATA-21-22 External Source Intake"
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_workflow_name_drift_fails_closed_even_if_run_is_success(self) -> None:
        broken = copy.deepcopy(self.data)
        verba = next(
            row
            for row in broken["late_authorities"]
            if row["worker_id"] == "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT"
        )
        verba["workflow_name"] = "DATA-21-22 External Source Intake"
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_cpython_docs_cannot_receive_unmaterialized_capacity_credit(self) -> None:
        broken = copy.deepcopy(self.data)
        cpython = next(
            row
            for row in broken["late_authorities"]
            if row["worker_id"] == "NEXT100-037-DATA-EN-PYTHON-DOCS"
        )
        cpython["numeric_capacity_bytes"] = 17901
        cpython["independent_family_credit"] = 1
        cpython["capacity_object_count"] = 14
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_arithmetic_drift_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["converged_pre_successor_dedup_vector"]["numeric_capacity_bytes"][
            "total"
        ] += 1
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_post_dedup_or_corpus_promotion_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["claim_boundary"]["post_dedup_capacity_claimed"] = True
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_learned_20m_cannot_be_claimed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["claim_boundary"]["learned_20m_checkpoint_claimed"] = True
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_byte_token_truth_boundary_is_mandatory(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["acquisition_plan_bytes"]["note"] = "capacity"
        with self.assertRaises(module.ValidationError):
            module.validate(broken)


if __name__ == "__main__":
    unittest.main()
