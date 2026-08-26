from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/discover_d03_rada_trees_object_identity.py"
SPEC = importlib.util.spec_from_file_location("rada_object_identity", TOOL)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_trees_object_identity_v1.json").read_text(encoding="utf-8")
)

PRIMARY_XET = "1" * 64
SECONDARY_XET = "2" * 64
RESOLVE_XET = PRIMARY_XET


def tree_fixture() -> list[dict[str, object]]:
    return [
        {
            "type": "file",
            "path": "README.md",
            "size": 3960,
            "oid": "a" * 40,
        },
        {
            "type": "file",
            "path": "Rada_Trees.7z",
            "size": 536_000_000,
            "oid": "b" * 40,
            "xetHash": PRIMARY_XET,
        },
        {
            "type": "file",
            "path": "rada_xtag_texts.7z",
            "size": 698_000_000,
            "oid": "c" * 40,
            "xetHash": SECONDARY_XET,
        },
    ]


class RadaTreesObjectIdentityTests(unittest.TestCase):
    def test_urls_pin_exact_revision_not_main(self) -> None:
        revision = CONFIG["source"]["revision"]
        tree_url = module.tree_api_url(CONFIG["source"]["repo_id"], revision)
        resolve_url = module.resolve_url(
            CONFIG["source"]["repo_id"], revision, CONFIG["source"]["primary_archive_path"]
        )
        self.assertIn(revision, tree_url)
        self.assertIn(revision, resolve_url)
        self.assertNotIn("/main/", resolve_url)

    def test_build_report_pins_object_metadata_without_training_credit(self) -> None:
        report = module.build_report(
            CONFIG,
            tree_fixture(),
            resolve_status=302,
            resolve_headers={"X-Xet-Hash": RESOLVE_XET, "Location": "https://example.invalid/large"},
        )
        self.assertEqual(report["primary_archive"]["xet_hash"], PRIMARY_XET)
        self.assertEqual(report["primary_archive"]["git_blob_id"], "b" * 40)
        self.assertGreater(report["primary_archive"]["size_bytes"], 0)
        self.assertFalse(report["resolve_observation"]["redirect_followed"])
        self.assertEqual(report["claim_boundary"]["training_authorized_bytes"], 0)
        self.assertFalse(report["claim_boundary"]["archive_downloaded"])
        module.validate_report(report)

    def test_tree_and_resolve_xet_mismatch_fails_closed(self) -> None:
        with self.assertRaises(module.DiscoveryError):
            module.build_report(
                CONFIG,
                tree_fixture(),
                resolve_status=302,
                resolve_headers={"X-Xet-Hash": "9" * 64},
            )

    def test_missing_immutable_identity_fails_closed(self) -> None:
        tree = tree_fixture()
        primary = next(item for item in tree if item.get("path") == "Rada_Trees.7z")
        primary.pop("xetHash")
        with self.assertRaises(module.DiscoveryError):
            module.build_report(CONFIG, tree, resolve_status=302, resolve_headers={})

    def test_duplicate_primary_tree_entry_fails_closed(self) -> None:
        tree = tree_fixture()
        tree.append(copy.deepcopy(next(item for item in tree if item.get("path") == "Rada_Trees.7z")))
        with self.assertRaises(module.DiscoveryError):
            module.build_report(
                CONFIG,
                tree,
                resolve_status=302,
                resolve_headers={"X-Xet-Hash": RESOLVE_XET},
            )

    def test_lfs_sha256_can_supply_object_identity(self) -> None:
        tree = tree_fixture()
        primary = next(item for item in tree if item.get("path") == "Rada_Trees.7z")
        primary.pop("xetHash")
        primary["lfs"] = {"oid": "sha256:" + "d" * 64, "size": primary["size"]}
        report = module.build_report(CONFIG, tree, resolve_status=302, resolve_headers={})
        self.assertEqual(report["primary_archive"]["lfs_sha256"], "d" * 64)
        module.validate_report(report)

    def test_report_cannot_promote_metadata_to_training(self) -> None:
        report = module.build_report(
            CONFIG,
            tree_fixture(),
            resolve_status=302,
            resolve_headers={"X-Xet-Hash": RESOLVE_XET},
        )
        report["claim_boundary"]["training_authorized_bytes"] = 1
        with self.assertRaises(module.DiscoveryError):
            module.validate_report(report)

    def test_mutable_revision_is_rejected(self) -> None:
        mutated = copy.deepcopy(CONFIG)
        mutated["source"]["revision"] = "main"
        with self.assertRaises(module.DiscoveryError):
            module.validate_config(mutated)


if __name__ == "__main__":
    unittest.main()
