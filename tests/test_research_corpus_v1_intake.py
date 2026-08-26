from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import validate_research_corpus_v1_intake as validator


class ResearchCorpusV1IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(validator.MANIFEST.read_text(encoding="utf-8"))

    def test_committed_manifest_passes(self) -> None:
        validator.validate(copy.deepcopy(self.value))

    def test_fabricated_training_capacity_fails_closed(self) -> None:
        value = copy.deepcopy(self.value)
        value["capacity_accounting"]["exact_training_eligible_bytes"] = 210115
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_duplicate_cpython_chunk_identity_fails_closed(self) -> None:
        value = copy.deepcopy(self.value)
        chunks = value["successor_authorities"]["en_cpython_docs"]["accepted_normalized_chunk_sha256"]
        chunks[-1] = chunks[0]
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_authority_head_drift_fails_closed(self) -> None:
        value = copy.deepcopy(self.value)
        value["successor_authorities"]["ua_kmu_secretariat"]["head_sha"] = "0" * 40
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_training_gate_cannot_be_promoted_by_manifest_edit(self) -> None:
        value = copy.deepcopy(self.value)
        value["gates"]["real_20m_training"] = "PASS"
        with self.assertRaises(AssertionError):
            validator.validate(value)


if __name__ == "__main__":
    unittest.main()
