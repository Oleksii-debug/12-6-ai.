from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next100_069_validator",
    ROOT / "tools/validate_next100_069_balance_diversity_v2.py",
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class BalanceDiversityV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = json.loads(
            (ROOT / "configs/data/next100_069_balance_diversity_v2.json").read_text(encoding="utf-8")
        )
        cls.dedup = json.loads(
            (ROOT / "configs/data/next100_065_cross_source_dedup_v3.json").read_text(encoding="utf-8")
        )

    def test_committed_authority_validates(self) -> None:
        validator.validate(copy.deepcopy(self.authority), copy.deepcopy(self.dedup))

    def test_cannot_invent_second_english_family(self) -> None:
        mutated = copy.deepcopy(self.authority)
        mutated["family_count"]["en"] = 2
        with self.assertRaises(AssertionError):
            validator.validate(mutated, copy.deepcopy(self.dedup))

    def test_cannot_relabel_source_bytes_as_loss_positions(self) -> None:
        mutated = copy.deepcopy(self.authority)
        mutated["unique_loss_positions"]["full_current_vector_exact_total"] = mutated[
            "available_unique_source_bytes"
        ]["total"]
        with self.assertRaises(AssertionError):
            validator.validate(mutated, copy.deepcopy(self.dedup))

    def test_cannot_hide_code_family_concentration(self) -> None:
        mutated = copy.deepcopy(self.authority)
        mutated["family_gate"]["whole_available_pool_if_consumed_without_subsampling"][
            "own_stratum_60pct_violations"
        ].remove("github:django/django")
        with self.assertRaises(AssertionError):
            validator.validate(mutated, copy.deepcopy(self.dedup))

    def test_cannot_tune_mixture_from_bpb(self) -> None:
        mutated = copy.deepcopy(self.authority)
        mutated["model_result_guided_tuning"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, copy.deepcopy(self.dedup))


if __name__ == "__main__":
    unittest.main()
