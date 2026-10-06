from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "rada_rights",
    Path("tools/validate_d03_rada_current_training_rights_v1.py"),
)
assert SPEC is not None and SPEC.loader is not None
rights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rights)


class RadaCurrentTrainingRightsTests(unittest.TestCase):
    def config(self) -> dict:
        return json.loads(
            Path("configs/data/d03_rada_current_training_rights_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def replay(self) -> tuple[dict, bytes]:
        path = Path(
            "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
        )
        raw = path.read_bytes()
        return json.loads(raw), raw

    def policy(self) -> tuple[dict, bytes]:
        path = Path("configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json")
        raw = path.read_bytes()
        return json.loads(raw), raw

    def test_exact_config_replay_and_policy_pass(self) -> None:
        config = self.config()
        replay, replay_raw = self.replay()
        policy, policy_raw = self.policy()
        rights.validate_config(config)
        rights.validate_replay(replay, replay_raw)
        rights.validate_policy(policy, policy_raw)

    def test_coherent_reseal_cannot_credit_training_bytes(self) -> None:
        mutated = copy.deepcopy(self.config())
        mutated["truth_boundary"]["training_authorized_bytes"] = 1
        core = dict(mutated)
        core.pop("authority_identity_sha256")
        mutated["authority_identity_sha256"] = rights.digest(core)
        with self.assertRaises(rights.RightsRecheckError):
            rights.validate_config(mutated)

    def test_evaluation_cannot_be_admitted_by_rights_recheck(self) -> None:
        mutated = copy.deepcopy(self.config())
        mutated["project_decision"]["evaluation"] = "ALLOWED"
        core = dict(mutated)
        core.pop("authority_identity_sha256")
        mutated["authority_identity_sha256"] = rights.digest(core)
        with self.assertRaises(rights.RightsRecheckError):
            rights.validate_config(mutated)

    def test_current_snapshot_substitution_is_rejected(self) -> None:
        replay, raw = self.replay()
        mutated = copy.deepcopy(replay)
        mutated["source"]["source_archive_sha256"] = "0" * 64
        with self.assertRaises(rights.RightsRecheckError):
            rights.validate_replay(mutated, raw)

    def test_portal_semantics_pass_without_persisting_page_text(self) -> None:
        markers = self.config()["official_portal_recheck"]["required_semantics"]
        page = ("<html><body>" + " ".join(markers.values()) + "</body></html>").encode(
            "utf-8"
        )
        evidence = rights.evaluate_portal(page, markers)
        self.assertTrue(all(evidence["semantic_projection"]["semantic_checks"].values()))
        self.assertEqual(len(evidence["semantic_identity_sha256"]), 64)
        self.assertNotIn(
            "Тексти первинних",
            json.dumps(evidence, ensure_ascii=False),
        )

    def test_missing_attribution_marker_fails_closed(self) -> None:
        markers = self.config()["official_portal_recheck"]["required_semantics"]
        page = " ".join(
            marker
            for key, marker in markers.items()
            if key != "source_attribution"
        ).encode("utf-8")
        with self.assertRaises(rights.RightsRecheckError):
            rights.evaluate_portal(page, markers)

    def test_script_text_cannot_satisfy_license_marker(self) -> None:
        markers = self.config()["official_portal_recheck"]["required_semantics"]
        body = dict(markers)
        license_marker = body.pop("default_license")
        page = (" ".join(body.values()) + f"<script>{license_marker}</script>").encode(
            "utf-8"
        )
        with self.assertRaises(rights.RightsRecheckError):
            rights.evaluate_portal(page, markers)


if __name__ == "__main__":
    unittest.main()
