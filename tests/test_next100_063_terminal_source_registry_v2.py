from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_063_terminal_source_registry_v2.json"
VALIDATOR = ROOT / "tools/validate_next100_063_terminal_source_registry_v2.py"

spec = importlib.util.spec_from_file_location("next100_063_v2_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Next100063TerminalSourceRegistryV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(CONFIG.read_text(encoding="utf-8"))

    def reseal(self, data: dict) -> dict:
        data["registry_identity_sha256"] = module.canonical_identity(data)
        return data

    def test_current_registry_passes(self) -> None:
        report = module.validate(copy.deepcopy(self.data))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["candidate_normalized_bytes"], 303374)
        self.assertEqual(report["candidate_independent_family_count"], 11)
        self.assertEqual(
            report["by_stratum"],
            {
                "uk": {"normalized_bytes": 100856, "family_count": 4},
                "en": {"normalized_bytes": 150643, "family_count": 3},
                "code": {"normalized_bytes": 51875, "family_count": 4},
            },
        )
        self.assertEqual(report["held_fail_closed_prs"], [467, 465, 475])

    def test_numpy_terminal_authority_is_required(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["terminal_additions"] = [row for row in broken["terminal_additions"] if row.get("pr") != 468]
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_failed_or_unmaterialized_sources_cannot_receive_credit(self) -> None:
        for blocked_pr in (467, 465, 475):
            with self.subTest(blocked_pr=blocked_pr):
                broken = copy.deepcopy(self.data)
                broken["terminal_additions"][0]["pr"] = blocked_pr
                self.reseal(broken)
                with self.assertRaises(module.ValidationError):
                    module.validate(broken)

    def test_dedicated_workflow_failure_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["terminal_additions"][0]["dedicated_workflow_conclusion"] = "failure"
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_generic_workflow_cannot_replace_dedicated_binding(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["terminal_additions"][0]["dedicated_workflow_name"] = "DATA-21-22 External Source Intake"
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_cpython_requires_exact_accepted_byte_ledger(self) -> None:
        broken = copy.deepcopy(self.data)
        cpython = next(row for row in broken["held_out_or_noncomposable"] if row.get("pr") == 467)
        cpython["reason"] = "full source bytes are good enough"
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_capacity_arithmetic_drift_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["pre_global_dedup_inventory"]["candidate_normalized_bytes"] += 1
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_replay_cannot_repair_capacity(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["composition_policy"]["replay_or_duplication_may_repair_capacity"] = True
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_nonterminal_candidates_cannot_be_counted(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["composition_policy"]["parallel_retest_queued_or_failed_candidates_counted"] = True
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_training_and_tokenizer_remain_blocked(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["downstream_gate_vector"]["authorized_balanced_no_replay_loss_positions"] = 1
        broken["downstream_gate_vector"]["tokenizer_fit"] = "PASS"
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)

    def test_premature_corpus_claim_fails_closed(self) -> None:
        broken = copy.deepcopy(self.data)
        broken["claim_boundary"]["research_corpus_v1_frozen"] = True
        self.reseal(broken)
        with self.assertRaises(module.ValidationError):
            module.validate(broken)


if __name__ == "__main__":
    unittest.main()
