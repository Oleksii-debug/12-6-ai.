from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from validate_next100_063_canonical_authority_pointer import (  # noqa: E402
    EXPECTED_CANONICAL_ID,
    EXPECTED_POINTER_ID,
    FORBIDDEN_IDS,
    validate,
)


class Next100063CanonicalAuthorityPointerTests(unittest.TestCase):
    def test_selector_resolves_only_fail_closed_v2(self) -> None:
        report = validate()
        self.assertEqual(report["status"], "PASS_CANONICAL_V2_ONLY")
        self.assertEqual(report["pointer_identity_sha256"], EXPECTED_POINTER_ID)
        self.assertEqual(report["canonical_registry_identity_sha256"], EXPECTED_CANONICAL_ID)
        self.assertEqual(report["candidate_normalized_bytes"], 303374)
        self.assertEqual(report["authorized_balanced_no_replay_loss_positions"], 0)

    def test_all_known_stale_identities_are_forbidden(self) -> None:
        report = validate()
        self.assertEqual(set(report["forbidden_registry_identity_sha256"]), FORBIDDEN_IDS)
        self.assertNotIn(EXPECTED_CANONICAL_ID, FORBIDDEN_IDS)

    def test_historical_v1_path_cannot_masquerade_as_canonical(self) -> None:
        pointer = json.loads(
            (ROOT / "configs/data/next100_063_canonical_authority_pointer_v1.json").read_text(encoding="utf-8")
        )
        old = json.loads(
            (ROOT / "configs/data/next100_063_terminal_source_registry_v1.json").read_text(encoding="utf-8")
        )
        canonical = json.loads(
            (ROOT / pointer["canonical"]["path"]).read_text(encoding="utf-8")
        )
        self.assertNotEqual(old["registry_identity_sha256"], canonical["registry_identity_sha256"])
        self.assertIn(old["registry_identity_sha256"], FORBIDDEN_IDS)
        self.assertEqual(canonical["registry_identity_sha256"], EXPECTED_CANONICAL_ID)

    def test_consumer_rule_preserves_zero_training_exposure(self) -> None:
        pointer = json.loads(
            (ROOT / "configs/data/next100_063_canonical_authority_pointer_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(pointer["canonical"]["authorized_balanced_no_replay_loss_positions"], 0)
        self.assertTrue(all(value is False for value in pointer["truth_boundary"].values()))


if __name__ == "__main__":
    unittest.main()
