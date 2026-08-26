from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_next100_104_global_cross_source_dedup_v4.py"
BASE = ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"
CONVERGENCE = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"
EXTENSION = ROOT / "configs/data/next100_104_global_cross_source_dedup_v4_extension.json"

spec = importlib.util.spec_from_file_location("next100_104", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Next100104GlobalCrossSourceDedupV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = json.loads(BASE.read_text(encoding="utf-8"))
        self.convergence = json.loads(CONVERGENCE.read_text(encoding="utf-8"))
        self.extension = json.loads(EXTENSION.read_text(encoding="utf-8"))

    def test_successor_inventory_reproduces_parent_vector(self) -> None:
        inventory, capacity, families = module.build_inventory(
            copy.deepcopy(self.base),
            copy.deepcopy(self.convergence),
            copy.deepcopy(self.extension),
        )
        self.assertEqual(len(inventory["sources"]), 21)
        self.assertEqual(capacity, {"uk": 100856, "en": 144151, "code": 69133, "total": 314140})
        self.assertEqual(families, {"uk": 4, "en": 2, "code": 4, "total": 10})
        self.assertFalse(inventory["final_refresh_required"])

    def test_extension_is_exactly_positive_capacity_late_objects(self) -> None:
        rows = self.extension["sources"]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["declared_capacity_bytes"] > 0 for row in rows))
        self.assertEqual(
            module._capacity_by_modality(rows),
            {"uk": 10812, "en": 59358, "code": 0, "total": 70170},
        )
        self.assertNotIn("python.cpython.documentation", {row["source_family"] for row in rows})

    def test_nist_normalization_is_deterministic_and_redacts_email(self) -> None:
        text = "Ａ title  \r\ncontact@example.org\r\n\r\n\r\nBody\fTail  \n"
        payload = module.normalize_nist_extracted(text)
        self.assertEqual(payload.decode("utf-8"), "A title\n<EMAIL>\n\nBody\nTail\n")

    def test_capacity_tamper_fails_closed(self) -> None:
        broken = copy.deepcopy(self.extension)
        broken["sources"][0]["declared_capacity_bytes"] += 1
        with self.assertRaises(module.Next100104Error):
            module.build_inventory(self.base, self.convergence, broken)

    def test_non_local_free_extension_fails_closed(self) -> None:
        broken = copy.deepcopy(self.extension)
        broken["local_free_only"] = False
        with self.assertRaises(module.Next100104Error):
            module.build_inventory(self.base, self.convergence, broken)


if __name__ == "__main__":
    unittest.main()
