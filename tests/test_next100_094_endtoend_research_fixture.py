from __future__ import annotations

import json
import unittest

from twelve_six.postbase_research_fixture import canonical_json, run_fixture


class EndToEndResearchFixtureTest(unittest.TestCase):
    def test_objective_fixture_core_is_deterministic_and_private(self) -> None:
        first = canonical_json(run_fixture())
        second = canonical_json(run_fixture())
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["final"], {"answer": 37, "verified": True})
        self.assertEqual(payload["verifier_ensemble"]["wrong_candidate_status"], "FAIL")
        self.assertEqual(payload["verifier_ensemble"]["final_status"], "PASS")
        self.assertEqual(payload["hypothesis_search"]["wrong_candidate"]["status"], "rejected")
        self.assertEqual(payload["hypothesis_search"]["revision"]["parent_id"], "H001")
        self.assertIn("damaged-v1", payload["memory_retrieval"]["superseded_excluded"])
        self.assertNotIn("PRIVATE_NEXT100_094", first)
        self.assertEqual(payload["execution_policy"]["external_llm_calls"], 0)
        self.assertEqual(payload["execution_policy"]["base_weight_changes"], 0)


if __name__ == "__main__":
    unittest.main()
