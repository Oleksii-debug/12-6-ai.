from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/probe_next100_105_rada_bulk_expansion.py"
CONFIG = ROOT / "configs/data/next100_105_rada_bulk_expansion_v1.json"

spec = importlib.util.spec_from_file_location("next100_105_rada", TOOL)
assert spec and spec.loader
rada = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rada)


def html(text: str, *, encoding: str = "utf-8") -> bytes:
    charset = "windows-1251" if encoding == "cp1251" else "utf-8"
    payload = (
        f'<html><head><meta charset="{charset}"></head>'
        f"<body><p>{text}</p></body></html>"
    )
    return payload.encode(encoding)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def good_text(seed: str = "Закон") -> str:
    return (
        (seed + " України набирає чинності відповідно до законодавства. ") * 20
    ).strip()


class RadaBulkProbeTests(unittest.TestCase):
    def test_config_is_fail_closed(self) -> None:
        config = rada.load_config(CONFIG)
        self.assertEqual(config["mode"], "PROBE")
        self.assertEqual(config["selection_policy"]["candidate_byte_cap"], 4_911_435)
        self.assertEqual(config["frozen_balance_policy"]["new_family_credit"], 0)

        weakened = copy.deepcopy(config)
        weakened["frozen_balance_policy"]["new_family_credit"] = 1
        with self.assertRaisesRegex(rada.ProbeError, "family credit"):
            rada.validate_config(weakened)

        weakened = copy.deepcopy(config)
        weakened["evaluation_firewall"]["final_test_records_may_enter_training"] = True
        with self.assertRaisesRegex(rada.ProbeError, "final-test firewall"):
            rada.validate_config(weakened)

    def test_probe_rejects_duplicate_privacy_backup_and_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.zip"
            safe = html(good_text())
            write_zip(
                archive,
                {
                    "d1.htm": safe,
                    "d2.htm": safe,
                    "d3.htm": html(good_text() + " Контакт test@example.org"),
                    "d4.htm.1": html(good_text("Рішення")),
                    "d23314.htm": html(good_text("Чинний")),
                },
            )
            config = rada.load_config(CONFIG)
            first = rada.probe_archive(config, archive)
            second = rada.probe_archive(config, archive)

        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "PROBE_LOCK_REQUIRED")
        self.assertEqual(first["gates"]["training_capacity_credit"], 0)
        self.assertIs(first["gates"]["training_eligible"], False)
        self.assertEqual(first["family"]["family_credit_added"], 0)
        self.assertEqual(first["candidate"]["selected_record_count"], 1)
        self.assertEqual(first["candidate"]["records"][0]["member"], "d1.htm")
        rejected = first["candidate"]["rejection_counts"]
        self.assertEqual(rejected["exact_duplicate"], 1)
        self.assertEqual(rejected["email"], 1)
        self.assertEqual(rejected["noncanonical_member"], 1)
        self.assertEqual(rejected["explicit_existing_source_exclusion"], 1)
        rada.verify_report(config, first)

    def test_probe_supports_declared_cp1251_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "cp1251.zip"
            write_zip(
                archive,
                {"d7.htm": html(good_text("Постанова"), encoding="cp1251")},
            )
            report = rada.probe_archive(rada.load_config(CONFIG), archive)
        self.assertEqual(report["candidate"]["selected_record_count"], 1)
        self.assertEqual(
            report["candidate"]["records"][0]["source_encoding"], "cp1251"
        )

    def test_probe_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "traversal.zip"
            write_zip(archive, {"../d8.htm": html(good_text())})
            with self.assertRaisesRegex(rada.ProbeError, "unsafe zip path"):
                rada.probe_archive(rada.load_config(CONFIG), archive)

    def test_locked_mode_rejects_archive_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "locked.zip"
            write_zip(archive, {"d9.htm": html(good_text())})
            probe_config = rada.load_config(CONFIG)
            report = rada.probe_archive(probe_config, archive)

            locked = copy.deepcopy(probe_config)
            locked["mode"] = "LOCKED"
            locked["lock"] = {
                "archive_sha256": report["archive"]["sha256"],
                "archive_bytes": report["archive"]["bytes"],
                "selected_manifest_identity_sha256": report["candidate"][
                    "selected_manifest_identity_sha256"
                ],
                "selected_pre_dedup_normalized_bytes": report["candidate"][
                    "selected_pre_dedup_normalized_bytes"
                ],
            }
            locked_report = rada.probe_archive(locked, archive)
            self.assertEqual(
                locked_report["decision"],
                "LOCKED_SOURCE_CANDIDATE_REQUIRES_GLOBAL_DEDUP_DECONTAMINATION",
            )
            self.assertEqual(locked_report["gates"]["training_capacity_credit"], 0)

            write_zip(archive, {"d9.htm": html(good_text("Змінено"))})
            with self.assertRaisesRegex(
                rada.ProbeError, "locked archive_sha256 mismatch"
            ):
                rada.probe_archive(locked, archive)


if __name__ == "__main__":
    unittest.main()
