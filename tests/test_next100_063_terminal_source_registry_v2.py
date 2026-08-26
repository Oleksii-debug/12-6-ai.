from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_063_terminal_source_registry_v2.json"
VALIDATOR_PATH = ROOT / "tools/validate_next100_063_terminal_source_registry_v2.py"

_spec = importlib.util.spec_from_file_location("next100_063_v2_validator", VALIDATOR_PATH)
assert _spec is not None and _spec.loader is not None
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _reseal(data: dict) -> dict:
    sealed = copy.deepcopy(data)
    sealed.pop("registry_identity_sha256", None)
    canonical = json.dumps(
        sealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    sealed["registry_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return sealed


def _validate_resealed(data: dict) -> dict:
    sealed = _reseal(data)
    return validator.validate_registry(
        sealed,
        expected_registry_identity=sealed["registry_identity_sha256"],
    )


class TerminalSourceRegistryV2Tests(unittest.TestCase):
    def test_current_registry_passes_exact_accounting(self) -> None:
        report = validator.validate_registry(_load())
        self.assertEqual(report["credited_normalized_bytes"], 266476)
        self.assertEqual(report["new_credited_normalized_bytes"], 83415)
        self.assertEqual(report["independent_families"], 10)
        self.assertEqual(report["authorized_balanced_no_replay_loss_positions"], 0)
        self.assertEqual(report["long_training"], "BLOCKED")
        self.assertEqual(report["paid_compute"], "NOT_AUTHORIZED")

    def test_self_hash_tamper_rejected(self) -> None:
        data = _load()
        data["observed_live_cutoff_utc"] = "2099-01-01T00:00:00Z"
        with self.assertRaisesRegex(validator.RegistryValidationError, "registry identity mismatch"):
            validator.validate_registry(data)

    def test_evaluation_authorization_rejected_even_after_reseal(self) -> None:
        data = _load()
        data["terminal_additions"][0]["evaluation"] = "AUTHORIZED_FOR_SELECTION"
        with self.assertRaisesRegex(validator.RegistryValidationError, "evaluation permission must remain denied"):
            _validate_resealed(data)

    def test_bool_capacity_rejected_even_after_reseal(self) -> None:
        data = _load()
        data["terminal_additions"][0]["normalized_bytes"] = True
        with self.assertRaisesRegex(validator.RegistryValidationError, "invalid credited bytes"):
            _validate_resealed(data)

    def test_cpython_missing_eligible_byte_ledger_stays_held_out(self) -> None:
        data = _load()
        data["held_out_or_noncomposable"] = [
            item for item in data["held_out_or_noncomposable"] if item.get("pr") != 467
        ]
        with self.assertRaisesRegex(validator.RegistryValidationError, "required held-out authority missing: PR 467"):
            _validate_resealed(data)

    def test_failed_pydantic_workflow_cannot_be_relabelled_success(self) -> None:
        data = _load()
        item = next(item for item in data["held_out_or_noncomposable"] if item.get("pr") == 465)
        item["dedicated_workflow_conclusion"] = "success"
        with self.assertRaisesRegex(validator.RegistryValidationError, "held-out boundary drift for PR 465"):
            _validate_resealed(data)

    def test_global_dedup_requirement_cannot_be_disabled(self) -> None:
        data = _load()
        data["composition_policy"]["global_cross_source_dedup_required_before_corpus_identity"] = False
        with self.assertRaisesRegex(validator.RegistryValidationError, "composition policy drift"):
            _validate_resealed(data)

    def test_paid_compute_cannot_be_authorized_at_source_registry_stage(self) -> None:
        data = _load()
        data["downstream_gate_vector"]["paid_compute"] = "AUTHORIZED"
        with self.assertRaisesRegex(validator.RegistryValidationError, "downstream gate vector weakened"):
            _validate_resealed(data)

    def test_training_claim_cannot_be_promoted_at_source_registry_stage(self) -> None:
        data = _load()
        data["claim_boundary"]["model_training_executed"] = True
        with self.assertRaisesRegex(validator.RegistryValidationError, "premature corpus/training claim"):
            _validate_resealed(data)


if __name__ == "__main__":
    unittest.main()
