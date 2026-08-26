from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_research_corpus_v1_acquisition_parent_binding_v2.py"
BINDING = ROOT / "configs/data/research_corpus_v1_acquisition_parent_binding_v2.json"
PLAN = ROOT / "configs/data/research_corpus_v1_acquisition_plan.json"
PARENT = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"

SPEC = importlib.util.spec_from_file_location("acq_parent_binding_v2", TOOL)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def reseal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["binding_identity_sha256"] = validator.canonical_sha(
        result, "binding_identity_sha256"
    )
    return result


class AcquisitionParentBindingV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = json.loads(BINDING.read_text(encoding="utf-8"))
        cls.plan_bytes = PLAN.read_bytes()
        cls.parent_bytes = PARENT.read_bytes()

    def test_exact_checkout_binding_passes_planning_only(self) -> None:
        report = validator.validate(self.binding, self.plan_bytes, self.parent_bytes)
        self.assertEqual(report["status"], "PASS_EXACT_PARENT_BINDING_PLAN_NONTERMINAL")
        self.assertEqual(report["candidate_vector"]["total"], 320632)
        self.assertEqual(report["remaining_gap"]["total"], 19679368)
        self.assertFalse(report["training_authorized"])

    def test_plan_byte_mutation_fails_blob_binding(self) -> None:
        with self.assertRaisesRegex(validator.ParentBindingError, "acquisition plan blob drift"):
            validator.validate(self.binding, self.plan_bytes + b"\n", self.parent_bytes)

    def test_parent_byte_mutation_fails_blob_binding(self) -> None:
        with self.assertRaisesRegex(validator.ParentBindingError, "parent config blob drift"):
            validator.validate(self.binding, self.plan_bytes, self.parent_bytes + b"\n")

    def test_stale_parent_head_fails_even_when_resealed(self) -> None:
        value = copy.deepcopy(self.binding)
        value["parent_convergence"]["head_sha"] = "5356d60c8c8af46d6fc34debfd3cb36731045338"
        with self.assertRaisesRegex(validator.ParentBindingError, "plan parent head drift"):
            validator.validate(reseal(value), self.plan_bytes, self.parent_bytes)

    def test_candidate_vector_drift_fails(self) -> None:
        value = copy.deepcopy(self.binding)
        value["expected_candidate_vector"]["en"] -= 1
        value["expected_candidate_vector"]["total"] -= 1
        with self.assertRaisesRegex(validator.ParentBindingError, "candidate vector provenance drift"):
            validator.validate(reseal(value), self.plan_bytes, self.parent_bytes)

    def test_buffer_drift_fails(self) -> None:
        value = copy.deepcopy(self.binding)
        value["expected_buffered_gross"]["en"] += 1
        value["expected_buffered_gross"]["total"] += 1
        with self.assertRaisesRegex(validator.ParentBindingError, "buffered-gross provenance drift"):
            validator.validate(reseal(value), self.plan_bytes, self.parent_bytes)

    def test_nonterminal_parent_cannot_be_promoted(self) -> None:
        value = copy.deepcopy(self.binding)
        value["parent_convergence"]["terminal_for_capacity_authority"] = True
        with self.assertRaisesRegex(validator.ParentBindingError, "nonterminal parent cannot be promoted"):
            validator.validate(reseal(value), self.plan_bytes, self.parent_bytes)

    def test_training_authorization_is_rejected(self) -> None:
        value = copy.deepcopy(self.binding)
        value["claim_boundary"]["model_training_authorized"] = True
        with self.assertRaisesRegex(validator.ParentBindingError, "unsafe claim"):
            validator.validate(reseal(value), self.plan_bytes, self.parent_bytes)


if __name__ == "__main__":
    unittest.main()
