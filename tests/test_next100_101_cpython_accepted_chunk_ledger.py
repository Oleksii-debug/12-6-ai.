from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_next100_101_cpython_accepted_chunk_ledger.py"
CONFIG = ROOT / "configs/data/next100_101_cpython_accepted_chunk_ledger_v1.json"

spec = importlib.util.spec_from_file_location("next100_101_ledger", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Next100101CpythonAcceptedChunkLedgerTests(unittest.TestCase):
    def test_production_contract_is_pinned_and_loads(self) -> None:
        contract = module._load_contract(CONFIG)
        self.assertEqual(contract["worker_id"], module.WORKER)
        self.assertEqual(
            contract["source_authority"]["authority_identity_sha256"],
            "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        )
        hashes = contract["quality_privacy"]["accepted_normalized_sha256_in_order"]
        self.assertEqual(len(hashes), 14)
        self.assertEqual(len(set(hashes)), 14)
        self.assertEqual(
            contract["historical_probe_evidence"]["derived_expected_accepted_utf8_bytes_total"],
            15540,
        )

    def test_premature_capacity_promotion_fails_closed(self) -> None:
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        broken = copy.deepcopy(contract)
        broken["purpose_firewall"]["canonical_training_capacity_credit"] = "PROMOTED"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(module.LedgerError):
                module._load_contract(path)

    def test_source_authority_binding_drift_fails_closed(self) -> None:
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        broken = copy.deepcopy(contract)
        broken["source_authority"]["workflow_run"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(module.LedgerError):
                module._load_contract(path)

    def test_normalization_matches_data228_semantics(self) -> None:
        source = "Ａ  B\r\n\r\n C\t D  \n"
        self.assertEqual(module._normalize_text(source), "A B\nC D")

    def test_chunking_is_deterministic_and_nonempty(self) -> None:
        text = "\n".join(
            [
                "alpha " * 25,
                "beta " * 25,
                "gamma " * 25,
                "delta " * 25,
            ]
        )
        first = module._chunk_text(text, max_chars=220, min_chars=40)
        second = module._chunk_text(text, max_chars=220, min_chars=40)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(len(chunk) >= 40 for chunk in first))

    def test_privacy_predicates_reject_phone_email_and_controls(self) -> None:
        self.assertEqual(
            module._quality_reason("Contact the office at +1 212 555 0199 for details about the program."),
            "pii_phone",
        )
        self.assertEqual(
            module._quality_reason("Contact documentation@example.org for detailed information about the program."),
            "pii_email",
        )
        self.assertEqual(
            module._quality_reason("Readable alphabetic material" * 4 + "\x00"),
            "control_character",
        )

    def test_wrong_source_bytes_fail_before_any_ledger_can_be_emitted(self) -> None:
        contract = module._load_contract(CONFIG)
        with self.assertRaises(module.LedgerError):
            module.build_ledger(contract, b"not-the-pinned-cpython-object")

    def test_rejected_payloads_cannot_be_declared_emittable(self) -> None:
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key in ("emit_rejected_chunk_text", "emit_rejected_chunk_hashes"):
            broken = copy.deepcopy(contract)
            broken["ledger_contract"][key] = True
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"broken-{key}.json"
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaises(module.LedgerError):
                    module._load_contract(path)


if __name__ == "__main__":
    unittest.main()
