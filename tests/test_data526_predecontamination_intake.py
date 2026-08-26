from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_data526_predecontamination_intake.py"
MANIFEST_PATH = (
    ROOT
    / "configs"
    / "data"
    / "data526_research_corpus_v1_predecontamination_intake_v1.json"
)

spec = importlib.util.spec_from_file_location("data526_validator", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class Data526PredecontaminationIntakeTests(unittest.TestCase):
    def test_exact_manifest_passes(self) -> None:
        result = validator.validate(MANIFEST_PATH)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_object_count"], 7)
        self.assertEqual(result["family_counts"], {"uk": 2, "en": 2, "code": 2})
        self.assertEqual(result["authorized_unique_optimized_targets"], 0)

    def _write_mutation(self, mutate) -> Path:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mutate(payload)
        temp = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".json", delete=False
        )
        json.dump(payload, temp, ensure_ascii=False, indent=2, sort_keys=True)
        temp.write("\n")
        temp.close()
        return Path(temp.name)

    def test_source_hash_mutation_fails_closed(self) -> None:
        path = self._write_mutation(
            lambda data: data["candidate_set"]["sources"][0].__setitem__(
                "raw_sha256", "0" * 64
            )
        )
        with self.assertRaises(SystemExit):
            validator.validate(path)

    def test_nonzero_training_capacity_fails_closed(self) -> None:
        path = self._write_mutation(
            lambda data: data["eligibility"].__setitem__(
                "authorized_unique_optimized_targets", 1
            )
        )
        with self.assertRaises(SystemExit):
            validator.validate(path)

    def test_silent_source_addition_fails_closed(self) -> None:
        def mutate(data):
            extra = copy.deepcopy(data["candidate_set"]["sources"][0])
            extra["source_id"] = "forbidden:extra"
            data["candidate_set"]["sources"].append(extra)

        path = self._write_mutation(mutate)
        with self.assertRaises(SystemExit):
            validator.validate(path)

    def test_wikisource_decontamination_gate_cannot_be_weakened(self) -> None:
        def mutate(data):
            for source in data["candidate_set"]["sources"]:
                if source["source_id"].startswith("ua.wikisource."):
                    source["near_match_decontamination"] = "SKIPPED"

        path = self._write_mutation(mutate)
        with self.assertRaises(SystemExit):
            validator.validate(path)


if __name__ == "__main__":
    unittest.main()
