import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next100063_validator", ROOT / "tools" / "validate_next100_063_research_corpus_v1_intake.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class Next100063Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "configs/data/next100_063_research_corpus_v1_intake_v1.json").read_text())

    def test_authority_validates(self):
        MOD.validate(copy.deepcopy(self.data))
        self.assertEqual(MOD.authority_identity(), self.data["authority_identity"]["sha256"])

    def test_family_double_credit_rejected(self):
        broken = copy.deepcopy(self.data)
        broken["terminal_source_authorities"][0]["family"] = "ua.rada.open-data.laws-texts"
        with self.assertRaises(ValueError):
            MOD.validate(broken)

    def test_fake_cpython_payload_bytes_rejected(self):
        broken = copy.deepcopy(self.data)
        broken["terminal_source_authorities"][2]["materialized_training_payload_bytes"] = 17901
        with self.assertRaises(ValueError):
            MOD.validate(broken)

    def test_training_exposure_rejected(self):
        broken = copy.deepcopy(self.data)
        broken["training_authority"]["authorized_training_exposure"] = 1
        with self.assertRaises(ValueError):
            MOD.validate(broken)

    def test_replay_rejected(self):
        broken = copy.deepcopy(self.data)
        broken["mixture_policy"]["replay_allowed"] = True
        with self.assertRaises(ValueError):
            MOD.validate(broken)


if __name__ == "__main__":
    unittest.main()
