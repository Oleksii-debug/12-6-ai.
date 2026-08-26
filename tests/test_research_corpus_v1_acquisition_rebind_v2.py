from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate_research_corpus_v1_acquisition_rebind_v2.py"
REBIND_PATH = ROOT / "configs/data/research_corpus_v1_acquisition_rebind_v2.json"
PARENT_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"

SPEC = importlib.util.spec_from_file_location("acq_rebind_v2", TOOL_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def reseal(data: dict) -> dict:
    value = copy.deepcopy(data)
    value["rebind_identity_sha256"] = validator.canonical_sha(
        value, "rebind_identity_sha256"
    )
    return value


class ResearchCorpusAcquisitionRebindV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rebind = json.loads(REBIND_PATH.read_text(encoding="utf-8"))
        cls.parent_bytes = PARENT_PATH.read_bytes()

    def test_exact_repository_parent_passes_planning_only(self) -> None:
        report = validator.validate(self.rebind, self.parent_bytes)
        self.assertEqual(
            report["status"], "PASS_REBIND_PLANNING_ONLY_PARENT_NONTERMINAL"
        )
        self.assertEqual(report["candidate_pre_successor_dedup_bytes"]["total"], 320632)
        self.assertEqual(report["candidate_remaining_gap_bytes"]["total"], 19679368)
        self.assertFalse(report["training_authorized"])

    def test_parent_byte_mutation_fails_git_blob_binding(self) -> None:
        with self.assertRaisesRegex(validator.RebindError, "repository parent blob drift"):
            validator.validate(self.rebind, self.parent_bytes + b"\n")

    def test_stale_parent_head_fails_even_when_resealed(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["parent_convergence"]["head_sha"] = "9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41"
        with self.assertRaisesRegex(validator.RebindError, "parent head drift"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_stale_candidate_vector_fails_even_when_resealed(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["candidate_pre_successor_dedup_bytes"]["en"] = 144151
        mutated["candidate_pre_successor_dedup_bytes"]["total"] = 314140
        with self.assertRaisesRegex(validator.RebindError, "candidate byte vector is stale"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_family_vector_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["candidate_independent_family_counts"]["en"] = 2
        mutated["candidate_independent_family_counts"]["total"] = 10
        with self.assertRaisesRegex(validator.RebindError, "candidate family vector is stale"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_gap_arithmetic_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["candidate_remaining_gap_bytes"]["en"] += 1
        mutated["candidate_remaining_gap_bytes"]["total"] += 1
        with self.assertRaisesRegex(validator.RebindError, "candidate gap arithmetic drift"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_buffer_arithmetic_drift_fails(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["buffered_gross_required_bytes"]["en"] += 1
        mutated["buffered_gross_required_bytes"]["total"] += 1
        with self.assertRaisesRegex(validator.RebindError, "buffered gross arithmetic drift"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_nonterminal_parent_cannot_be_promoted(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["parent_convergence"]["terminal_for_capacity_authority"] = True
        with self.assertRaisesRegex(validator.RebindError, "nonterminal parent cannot be promoted"):
            validator.validate(reseal(mutated), self.parent_bytes)

    def test_training_claim_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.rebind)
        mutated["claim_boundary"]["model_training_authorized"] = True
        with self.assertRaisesRegex(validator.RebindError, "unsafe claim boundary"):
            validator.validate(reseal(mutated), self.parent_bytes)


if __name__ == "__main__":
    unittest.main()
