from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_rada_trees_hf_objects as probe


def fixture_tree() -> list[dict[str, object]]:
    return [
        {"type": "file", "path": ".gitattributes", "size": 2460, "oid": "a" * 40},
        {
            "type": "file",
            "path": "Rada_Trees.7z",
            "size": 536_000_000,
            "oid": "1" * 40,
            "xetHash": "a" * 64,
        },
        {
            "type": "file",
            "path": "rada_xtag_texts.7z",
            "size": 698_000_000,
            "oid": "2" * 40,
            "xetHash": "b" * 64,
        },
    ]


def fixture_resolve() -> dict[str, str]:
    return {
        "Rada_Trees.7z": "a" * 64,
        "rada_xtag_texts.7z": "b" * 64,
    }


class RadaTreesHfObjectProbeTests(unittest.TestCase):
    def test_snapshot_binds_exact_object_metadata_and_stays_zero_credit(self) -> None:
        snapshot = probe.build_snapshot(fixture_tree(), fixture_resolve())
        probe.validate_snapshot(snapshot)
        self.assertEqual(
            [item["path"] for item in snapshot["files"]],
            list(probe.EXPECTED_PATHS),
        )
        self.assertEqual(snapshot["claim_boundary"]["training_authorized_bytes"], 0)
        self.assertFalse(snapshot["claim_boundary"]["archives_downloaded"])

    def test_snapshot_identity_is_deterministic(self) -> None:
        first = probe.build_snapshot(fixture_tree(), fixture_resolve())
        second = probe.build_snapshot(fixture_tree(), fixture_resolve())
        self.assertEqual(
            first["snapshot_identity_sha256"],
            second["snapshot_identity_sha256"],
        )

    def test_missing_xet_identity_fails_closed(self) -> None:
        tree = fixture_tree()
        del tree[1]["xetHash"]
        with self.assertRaises(probe.ProbeError):
            probe.build_snapshot(tree, fixture_resolve())

    def test_resolve_header_must_match_tree_identity(self) -> None:
        hashes = fixture_resolve()
        hashes["Rada_Trees.7z"] = "c" * 64
        with self.assertRaises(probe.ProbeError):
            probe.build_snapshot(fixture_tree(), hashes)

    def test_duplicate_archive_path_fails_closed(self) -> None:
        tree = fixture_tree()
        tree.append(copy.deepcopy(tree[1]))
        with self.assertRaises(probe.ProbeError):
            probe.build_snapshot(tree, fixture_resolve())

    def test_mutated_snapshot_cannot_authorize_training(self) -> None:
        snapshot = probe.build_snapshot(fixture_tree(), fixture_resolve())
        snapshot["claim_boundary"]["training_authorized_bytes"] = 1
        with self.assertRaises(probe.ProbeError):
            probe.validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
