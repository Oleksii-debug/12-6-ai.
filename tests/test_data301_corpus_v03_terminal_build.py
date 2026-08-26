from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_data301_corpus_v03_terminal_build.py"
EVIDENCE = ROOT / "configs/data/data301_corpus_v03_terminal_build_v1.json"

spec = importlib.util.spec_from_file_location("data301_validator", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Data301TerminalBuildTests(unittest.TestCase):
    def test_terminal_blocker_is_exact_and_fail_closed(self) -> None:
        result = validator.validate()
        self.assertEqual(result["status"], "TERMINAL_BLOCKED")
        self.assertIsNone(result["corpus_identity"])
        self.assertIsNone(result["shard_identity"])
        self.assertEqual(result["authorized_balanced_no_replay_capacity"], 0)
        self.assertEqual(result["candidate_docs"], 5)
        self.assertEqual(result["candidate_unique_bytes_prebuild"], 183061)
        self.assertEqual(result["terminal_text_only_unique_loss_positions"], 173355)
        self.assertEqual(result["product_trainer_streaming"], "PASS_AST_SOURCE_SEMANTICS")

    def test_evidence_does_not_promote_blocked_corpus(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        verdict = evidence["terminal_verdict"]
        self.assertFalse(verdict["corpus_frozen"])
        self.assertFalse(verdict["corpus_terminal"])
        self.assertFalse(verdict["release_ready"])
        self.assertIsNone(verdict["corpus_identity"])
        self.assertIsNone(verdict["shard_identity"])

    def test_full_five_source_loss_capacity_is_not_fabricated(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        accounting = evidence["one_pass_loss_position_accounting"]
        self.assertFalse(accounting["full_five_source_terminal_ledger_available"])
        self.assertEqual(accounting["authorized_balanced_no_replay_capacity"], 0)
        self.assertEqual(
            accounting["terminal_text_only_data294"]["by_language"],
            {"uk": 88564, "en": 84791, "code": 0},
        )


if __name__ == "__main__":
    unittest.main()
