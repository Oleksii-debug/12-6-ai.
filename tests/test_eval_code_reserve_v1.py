from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_eval_code_reserve_v1.py"
MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"

spec = importlib.util.spec_from_file_location("eval647_validator", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class EvalCodeReserveV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_contract_is_reserved_but_not_authorized(self) -> None:
        result = validator.validate_document(self.doc)
        self.assertEqual(result["reserved_objects"], 2)
        self.assertEqual(result["independent_families"], 2)
        self.assertEqual(result["selection_validation_records_authorized"], 0)

    def test_training_promotion_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.doc)
        mutated["objects"][0]["training_allowed"] = True
        with self.assertRaises(ValueError):
            validator.validate_document(mutated)

    def test_final_test_access_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.doc)
        mutated["reservation"]["final_test_payload_access_allowed"] = True
        with self.assertRaises(ValueError):
            validator.validate_document(mutated)

    def test_blob_identity_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.doc)
        mutated["objects"][1]["git_blob_sha1"] = "0" * 40
        with self.assertRaises(ValueError):
            validator.validate_document(mutated)

    def test_guessed_raw_sha256_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.doc)
        mutated["objects"][0]["raw_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_document(mutated)


if __name__ == "__main__":
    unittest.main()
