from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_next100_107_gutenberg_terminal_seal.py"
CONFIG = ROOT / "configs" / "data" / "next100_107_gutenberg_terminal_seal_v1.json"
spec = importlib.util.spec_from_file_location("next100107_validator", VALIDATOR)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class GutenbergTerminalSealTests(unittest.TestCase):
    def load(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def validate_mutation(self, payload):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return module.validate(path)

    def test_frozen_seal_passes(self):
        result = module.validate(CONFIG)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["normalized_utf8_bytes"], 1672110)

    def test_reject_capacity_inflation(self):
        data = self.load()
        data["capacity"]["exact_source_level_normalized_bytes"] += 1
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_reject_evaluation_grant(self):
        data = self.load()
        data["rights"]["evaluation"] = "ALLOWED"
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_reject_family_inflation(self):
        data = self.load()
        data["source_family"]["independent_family_credit"] = 3
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)

    def test_reject_training_promotion(self):
        data = self.load()
        data["claim_boundary"]["long_training_authorized"] = True
        with self.assertRaises(SystemExit):
            self.validate_mutation(data)


if __name__ == "__main__":
    unittest.main()
