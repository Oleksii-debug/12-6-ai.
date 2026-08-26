from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_data300_corpus_v03_build_contract.py"
SPEC = importlib.util.spec_from_file_location("data300_contract", TOOL)
assert SPEC is not None and SPEC.loader is not None
data300 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data300)


class Data300ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = data300._read_contract(ROOT)

    def test_frozen_contract_self_identity_and_truth_boundary(self) -> None:
        data300._assert_contract_invariants(self.contract)
        self.assertEqual(self.contract["corpus_state"], "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL")
        self.assertEqual(
            self.contract["wave3_release_state"]["on_contract_validation"],
            "CONTRACT_VALID_CORPUS_NOT_BUILT_NOT_FROZEN",
        )

    def test_exact_five_source_inventory_is_bound(self) -> None:
        sources = self.contract["terminal_source_inventory"]["sources"]
        self.assertEqual(
            tuple(item["registry_source_id"] for item in sources),
            data300.EXPECTED_SOURCE_IDS,
        )
        self.assertEqual(len({item["source_family"] for item in sources}), 4)

    def test_nonterminal_data228_is_not_admitted(self) -> None:
        excluded = self.contract["terminal_source_inventory"]["excluded_nonterminal_candidates"]
        self.assertEqual(excluded[0]["worker_id"], "DATA-228-REAL-UA-EN-DIVERSITY-V2")
        self.assertEqual(excluded[0]["dedicated_workflow_conclusion"], "failure")
        self.assertEqual(excluded[0]["rule"], "NO_ADMISSION")

    def test_artificial_repetition_cannot_be_enabled(self) -> None:
        weakened = copy.deepcopy(self.contract)
        weakened["build_contract"]["artificial_repetition"]["document_replication"] = True
        with self.assertRaises(data300.ContractError):
            data300._assert_contract_invariants(weakened)

    def test_final_test_cannot_become_selection_data(self) -> None:
        weakened = copy.deepcopy(self.contract)
        weakened["split_contract"]["final_test"]["may_be_read_before_selection_is_locked"] = True
        with self.assertRaises(data300.ContractError):
            data300._assert_contract_invariants(weakened)

    def test_tree_comparison_detects_one_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            left = tmp_path / "left"
            right = tmp_path / "right"
            left.mkdir()
            right.mkdir()
            (left / "x").write_bytes(b"a")
            (right / "x").write_bytes(b"b")
            self.assertNotEqual(data300._tree_inventory(left), data300._tree_inventory(right))

    def test_unique_loss_ledger_rejects_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unique-loss").mkdir()
            row = {
                "source_id": "s",
                "record_id": "r",
                "target_offset": 1,
                "optimized": True,
                "is_padding": False,
            }
            ledger = root / "unique-loss/train-ledger.jsonl"
            ledger.write_text(
                json.dumps(row, sort_keys=True)
                + "\n"
                + json.dumps(row, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (root / "unique-loss/summary.json").write_text(
                json.dumps(
                    {
                        "unique_optimized_loss_positions": 2,
                        "repeated_optimized_loss_positions": 0,
                        "padding_counted_as_data": False,
                        "corpus_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(data300.ContractError):
                data300._validate_unique_loss(root)


if __name__ == "__main__":
    unittest.main()
