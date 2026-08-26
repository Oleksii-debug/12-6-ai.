from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_plug2_github_tree as tree_probe
import validate_d03_plug2_historical_ua_probe as validator


class PluG2HistoricalUaProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(validator.CONFIG.read_text(encoding="utf-8"))

    def test_committed_probe_is_fail_closed_valid(self) -> None:
        validator.validate(copy.deepcopy(self.value))

    def test_license_version_cannot_be_inferred(self) -> None:
        value = copy.deepcopy(self.value)
        value["rights"]["exact_cc_by_version_pinned"] = True
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_reported_tokens_cannot_become_training_credit(self) -> None:
        value = copy.deepcopy(self.value)
        value["claim_boundary"]["training_authorized_bytes"] = value["source"]["reported_plug2_tokens"]
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_plug_and_plug2_cannot_gain_two_family_credits(self) -> None:
        value = copy.deepcopy(self.value)
        value["lineage"]["plug_and_plug2_share_one_family"] = False
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_historical_orthography_cannot_be_silently_treated_as_modern(self) -> None:
        value = copy.deepcopy(self.value)
        value["quality_strata"]["modern_standard_ukrainian"] = "ASSUME"
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def _commit(self) -> dict[str, object]:
        return {
            "sha": validator.EXPECTED_COMMIT,
            "commit": {"tree": {"sha": validator.EXPECTED_TREE}},
        }

    def _tree(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        for path, (object_type, oid, size) in validator.EXPECTED_ROOT_OBJECTS.items():
            entry: dict[str, object] = {"path": path, "type": object_type, "sha": oid}
            if size is not None:
                entry["size"] = size
            entries.append(entry)
        entries.extend(
            [
                {
                    "path": "PluG_texts/A/author/work.txt",
                    "type": "blob",
                    "sha": "1" * 40,
                    "size": 123,
                },
                {
                    "path": "PluG2_texts/B/author/work.txt",
                    "type": "blob",
                    "sha": "2" * 40,
                    "size": 456,
                },
            ]
        )
        return {"sha": validator.EXPECTED_TREE, "truncated": False, "tree": entries}

    def test_metadata_snapshot_is_deterministic_and_zero_credit(self) -> None:
        config = copy.deepcopy(self.value)
        first = tree_probe.build_snapshot(config, self._commit(), self._tree())
        second = tree_probe.build_snapshot(config, self._commit(), self._tree())
        self.assertEqual(first, second)
        tree_probe.validate_snapshot(first, config)
        self.assertEqual(first["training_authorized_bytes"], 0)
        self.assertFalse(first["raw_text_emitted"])

    def test_truncated_recursive_tree_is_rejected(self) -> None:
        tree = self._tree()
        tree["truncated"] = True
        with self.assertRaises(RuntimeError):
            tree_probe.build_snapshot(copy.deepcopy(self.value), self._commit(), tree)

    def test_root_oid_drift_is_rejected(self) -> None:
        tree = self._tree()
        entries = tree["tree"]
        assert isinstance(entries, list)
        entries[0]["sha"] = "f" * 40
        with self.assertRaises(RuntimeError):
            tree_probe.build_snapshot(copy.deepcopy(self.value), self._commit(), tree)

    def test_oversized_text_member_is_rejected_by_snapshot_validation(self) -> None:
        config = copy.deepcopy(self.value)
        snapshot = tree_probe.build_snapshot(config, self._commit(), self._tree())
        snapshot["payloads"][0]["largest_text_blob_bytes"] = 50_000_001
        unhashed = dict(snapshot)
        unhashed.pop("snapshot_sha256")
        import hashlib

        snapshot["snapshot_sha256"] = hashlib.sha256(tree_probe.canonical_json_bytes(unhashed)).hexdigest()
        with self.assertRaises(RuntimeError):
            tree_probe.validate_snapshot(snapshot, config)


if __name__ == "__main__":
    unittest.main()
