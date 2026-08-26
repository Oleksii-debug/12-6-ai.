from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_next100_102_research_corpus_v1_intake.py"
CONFIG = ROOT / "configs" / "data" / "next100_102_research_corpus_v1_intake.json"
spec = importlib.util.spec_from_file_location("next100102_validator", VALIDATOR)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ResearchCorpusV1IntakeTests(unittest.TestCase):
    def load(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def validate_mutation(self, payload):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return module.validate(path)

    def test_frozen_intake_passes(self):
        result = module.validate(CONFIG)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["logical_record_count"], 20)
        self.assertEqual(result["authorized_unique_optimized_targets"], 0)

    def test_cannot_fabricate_training_capacity(self):
        data = self.load()
        data["training_authority"]["authorized_unique_optimized_targets"] = 1
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_cannot_drop_accepted_python_chunk(self):
        data = self.load()
        data["candidate_inventory"]["successor_records"].pop()
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_cannot_relabel_family_to_fake_diversity(self):
        data = self.load()
        data["candidate_inventory"]["successor_records"][0]["family"] = "ua.fake.family"
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_cannot_claim_corpus_or_shard_identity(self):
        for key in ("corpus_identity", "shard_identity"):
            data = self.load()
            data["release_state"][key] = "fabricated"
            with self.assertRaises(SystemExit):
                self.validate_mutation(data)


if __name__ == "__main__":
    unittest.main()
