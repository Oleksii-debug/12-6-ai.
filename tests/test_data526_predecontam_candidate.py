from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_predecontam_candidate_identity import build_candidate  # noqa: E402
from validate_data526_predecontam_candidate import validate  # noqa: E402

INVENTORY_PATH = ROOT / "configs/data/data526_predecontam_source_records_v1.json"
FROZEN_PATH = ROOT / "evidence/data526/predecontam_candidate_v1.json"


class Data526PredecontamCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        self.frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    def test_exact_frozen_candidate_rebuilds(self) -> None:
        self.assertEqual(build_candidate(self.inventory), self.frozen)
        report = validate()
        self.assertEqual(report["status"], "PASS_PRE_DECONTAMINATION_IDENTITY_ONLY")
        self.assertFalse(report["decontamination_executed"])
        self.assertFalse(report["final_training_authorized"])

    def test_family_diversity_is_real_not_replay(self) -> None:
        families = self.frozen["independent_families_by_stratum"]
        self.assertEqual({key: len(value) for key, value in families.items()}, {"ua": 2, "en": 2, "code": 2})
        self.assertFalse(self.frozen["replay_authorized"])
        self.assertEqual(self.frozen["total_normalized_utf8_bytes"], 243898)

    def test_cpython_full_source_is_not_silently_admitted(self) -> None:
        source_ids = {row["source_id"] for row in self.inventory["records"]}
        self.assertNotIn("en.python.docs.tutorial-introduction", source_ids)
        held = self.inventory["held_out_authorities"]["next100_037_cpython_docs"]
        self.assertEqual(held["status"], "HOLD_UNTIL_ACCEPTED_CHUNK_RECORD_MATERIALIZATION")
        self.assertIn("14 of 16 chunks", held["reason"])

    def test_training_promotion_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["records"][0]["final_training_eligible"] = True
        with self.assertRaises(ValueError):
            build_candidate(tampered)

    def test_duplicate_normalized_identity_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["records"][1]["normalized_sha256"] = tampered["records"][0]["normalized_sha256"]
        with self.assertRaises(ValueError):
            build_candidate(tampered)

    def test_single_family_regression_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["records"] = [
            row
            for row in tampered["records"]
            if row["source_family"] != "en.usgov.nist.technical-series"
        ]
        with self.assertRaisesRegex(ValueError, ">=2 independent families"):
            build_candidate(tampered)


if __name__ == "__main__":
    unittest.main()
