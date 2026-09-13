from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "swarm_collision_report.py"
SPEC = importlib.util.spec_from_file_location("swarm_collision_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SwarmCollisionReportTests(unittest.TestCase):
    def test_extracts_ephemeral_key_from_title_and_worker_line(self) -> None:
        self.assertEqual(
            MODULE.extract_task_keys("NEXT100-063 — source convergence", None),
            ("NEXT100-063",),
        )
        self.assertEqual(
            MODULE.extract_task_keys(
                "DATA: intake",
                "SWARM_WORKER_ID: `NEXT100-063-RESEARCH-CORPUS-V1-CONVERGENCE`",
            ),
            ("NEXT100-063",),
        )

    def test_permanent_lane_ids_are_not_unique_ephemeral_tasks(self) -> None:
        self.assertEqual(
            MODULE.extract_task_keys("D01 — Model Architecture", "SWARM_WORKER_ID: D01-SCALE"),
            (),
        )

    def test_one_issue_plus_one_pr_is_normal_lifecycle(self) -> None:
        records = MODULE.records_from_github_items(
            [
                {
                    "number": 521,
                    "title": "NEXT100-063 — source convergence",
                    "body": "SWARM_WORKER_ID: NEXT100-063-SOURCE-REGISTRY-CONVERGENCE",
                    "html_url": "https://example.test/issues/521",
                },
                {
                    "number": 527,
                    "title": "NEXT100-063: converge source authorities",
                    "body": "SWARM_WORKER_ID: NEXT100-063-SOURCE-REGISTRY-CONVERGENCE",
                    "html_url": "https://example.test/pull/527",
                    "pull_request": {},
                },
            ]
        )
        self.assertEqual(MODULE.detect_collisions(records), [])

    def test_two_open_issues_for_same_task_are_blocked(self) -> None:
        records = MODULE.records_from_github_items(
            [
                {
                    "number": 521,
                    "title": "NEXT100-063 — source convergence",
                    "body": "",
                    "html_url": "https://example.test/issues/521",
                },
                {
                    "number": 530,
                    "title": "NEXT100-063 — Research Corpus V1 convergence",
                    "body": "",
                    "html_url": "https://example.test/issues/530",
                },
            ]
        )
        collisions = MODULE.detect_collisions(records)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["task_key"], "NEXT100-063")
        self.assertEqual(collisions[0]["reasons"], ["MULTIPLE_OPEN_ISSUES_CLAIM_TASK"])
        self.assertEqual([item["number"] for item in collisions[0]["issues"]], [521, 530])

    def test_two_open_prs_for_same_task_are_blocked(self) -> None:
        records = MODULE.records_from_github_items(
            [
                {
                    "number": 527,
                    "title": "NEXT100-063: implementation A",
                    "body": "",
                    "html_url": "https://example.test/pull/527",
                    "pull_request": {},
                },
                {
                    "number": 531,
                    "title": "NEXT100-063: implementation B",
                    "body": "",
                    "html_url": "https://example.test/pull/531",
                    "pull_request": {},
                },
            ]
        )
        collisions = MODULE.detect_collisions(records)
        self.assertEqual(collisions[0]["reasons"], ["MULTIPLE_OPEN_PRS_CLAIM_TASK"])

    def test_report_is_fail_closed_and_self_hashed(self) -> None:
        report = MODULE.build_report(
            "Oleksii-debug/12-6-ai.",
            [
                {
                    "number": 521,
                    "title": "NEXT100-063 — A",
                    "body": "",
                    "html_url": "https://example.test/issues/521",
                },
                {
                    "number": 530,
                    "title": "NEXT100-063 — B",
                    "body": "",
                    "html_url": "https://example.test/issues/530",
                },
            ],
            "2026-08-26T19:30:00Z",
        )
        self.assertEqual(report["verdict"], "BLOCK_DUPLICATE_EPHEMERAL_TASK_OWNERSHIP")
        self.assertEqual(len(report["report_sha256"]), 64)
        self.assertEqual(report["open_items_with_ephemeral_task_key"], 2)


if __name__ == "__main__":
    unittest.main()
