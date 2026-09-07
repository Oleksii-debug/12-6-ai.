from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.validate_data526_research_corpus_predecontam_v4 import validate_doc

DOC = json.loads(
    Path("configs/data/research_corpus_v1_predecontam_freeze_v4.json").read_text(
        encoding="utf-8"
    )
)


class Data526PredecontamV4Tests(unittest.TestCase):
    def test_frozen_terminal_v7_authority_validates(self) -> None:
        validate_doc(copy.deepcopy(DOC))

    def test_source_bytes_cannot_be_promoted_to_optimized_targets(self) -> None:
        mutated = copy.deepcopy(DOC)
        mutated["claim_boundary"]["authorized_unique_optimized_targets"] = 2_215_615
        with self.assertRaises(ValueError):
            validate_doc(mutated)

    def test_candidate_identity_cannot_drift(self) -> None:
        mutated = copy.deepcopy(DOC)
        mutated["candidate_freeze"]["candidate_identity_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_doc(mutated)

    def test_decontamination_cannot_be_marked_executed_by_freeze(self) -> None:
        mutated = copy.deepcopy(DOC)
        mutated["claim_boundary"]["decontamination_executed"] = True
        with self.assertRaises(ValueError):
            validate_doc(mutated)


if __name__ == "__main__":
    unittest.main()
