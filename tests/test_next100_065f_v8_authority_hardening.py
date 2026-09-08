from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/verify_next100_065f_v8_authority.py"
SPEC = importlib.util.spec_from_file_location("next100_065f_v8_authority", TOOL)
assert SPEC and SPEC.loader
hardening = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardening)


def _report() -> dict[str, object]:
    modalities = {
        "uk": {
            "source_count": 1,
            "declared_source_family_count": 4,
            "declared_capacity_bytes_before": 100,
            "conservative_unique_capacity_bytes_after": 90,
            "duplicate_discount_bytes": 10,
        },
        "en": {
            "source_count": 1,
            "declared_source_family_count": 5,
            "declared_capacity_bytes_before": 200,
            "conservative_unique_capacity_bytes_after": 190,
            "duplicate_discount_bytes": 10,
        },
        "code": {
            "source_count": 1,
            "declared_source_family_count": 12,
            "declared_capacity_bytes_before": 300,
            "conservative_unique_capacity_bytes_after": 280,
            "duplicate_discount_bytes": 20,
        },
    }
    outer_modalities = {
        name: {
            "source_count": row["source_count"],
            "source_family_count": row["declared_source_family_count"],
            "capacity_bytes_before_global_dedup": row["declared_capacity_bytes_before"],
            "conservative_unique_capacity_bytes_after_global_dedup": row[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": row["duplicate_discount_bytes"],
        }
        for name, row in modalities.items()
    }
    nested = {
        "report_sha256": "nested-terminal",
        "source_count": 3,
        "sources": [
            {"source_family": "uk-a", "modality": "uk"},
            {"source_family": "en-a", "modality": "en"},
            {"source_family": "code-a", "modality": "code"},
        ],
        "terminal_candidates": {
            "source_count": 3,
            "declared_capacity_bytes_before": 600,
            "conservative_unique_capacity_bytes_after": 560,
            "duplicate_discount_bytes": 40,
            "duplicate_cluster_count": 2,
            "effective_independent_origin_count": 18,
            "by_modality": modalities,
        },
    }
    return {
        "dedup_v3": nested,
        "source_vector": {
            "source_object_count": 3,
            "source_family_counts": {"uk": 1, "en": 1, "code": 1},
            "source_capacity_bytes_before_global_dedup": 600,
            "conservative_unique_capacity_bytes_after_global_dedup": 560,
            "duplicate_discount_bytes": 40,
            "duplicate_cluster_count": 2,
            "effective_independent_origin_count": 18,
            "by_modality": outer_modalities,
        },
    }


class Next100065FV8AuthorityHardeningTests(unittest.TestCase):
    def test_current_main_v6_registry_is_exactly_bound(self) -> None:
        hardening.validate_current_registry(ROOT)

    def test_data_only_namespace_bypasses_top_level_torch_import(self) -> None:
        saved = {name: sys.modules.get(name) for name in ("twelve_six", "twelve_six.data", "twelve_six.data.probe")}
        for name in saved:
            sys.modules.pop(name, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "src/twelve_six"
                data = package / "data"
                data.mkdir(parents=True)
                (package / "__init__.py").write_text(
                    'raise ModuleNotFoundError("No module named torch")\n', encoding="utf-8"
                )
                (data / "__init__.py").write_text("", encoding="utf-8")
                (data / "probe.py").write_text("VALUE = 7\n", encoding="utf-8")
                hardening._install_v7_namespace(root)
                probe = importlib.import_module("twelve_six.data.probe")
                self.assertEqual(probe.VALUE, 7)
        finally:
            for name in ("twelve_six.data.probe", "twelve_six.data", "twelve_six"):
                sys.modules.pop(name, None)
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module

    def test_nested_summary_cross_binding_rejects_mutation(self) -> None:
        report = _report()
        report["source_vector"]["source_family_counts"] = {"uk": 4, "en": 5, "code": 12}
        report["dedup_v3"]["sources"] = [
            *({"source_family": f"uk-{i}", "modality": "uk"} for i in range(4)),
            *({"source_family": f"en-{i}", "modality": "en"} for i in range(5)),
            *({"source_family": f"code-{i}", "modality": "code"} for i in range(12)),
        ]
        report["dedup_v3"]["source_count"] = 21
        report["dedup_v3"]["terminal_candidates"]["source_count"] = 21
        report["source_vector"]["source_object_count"] = 21
        hardening.cross_bind_nested(report)
        report["dedup_v3"]["terminal_candidates"]["duplicate_discount_bytes"] = 41
        with self.assertRaisesRegex(hardening.V8AuthorityError, "duplicate discount"):
            hardening.cross_bind_nested(report)

    def test_nested_verifier_runs_even_if_outer_hash_is_recomputed(self) -> None:
        report = _report()
        report["source_vector"]["source_family_counts"] = {"uk": 4, "en": 5, "code": 12}
        report["dedup_v3"]["sources"] = [
            *({"source_family": f"uk-{i}", "modality": "uk"} for i in range(4)),
            *({"source_family": f"en-{i}", "modality": "en"} for i in range(5)),
            *({"source_family": f"code-{i}", "modality": "code"} for i in range(12)),
        ]
        report["dedup_v3"]["source_count"] = 21
        report["dedup_v3"]["terminal_candidates"]["source_count"] = 21
        report["source_vector"]["source_object_count"] = 21

        calls: list[str] = []

        def strict_verifier(nested: object) -> None:
            calls.append("called")
            if nested["report_sha256"] != "nested-terminal":
                raise RuntimeError("nested report self-hash mismatch")

        hardening.verify_nested_authority(report, strict_verifier)
        self.assertEqual(calls, ["called"])

        report["dedup_v3"]["report_sha256"] = "attacker-recomputed-nested"
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report["outer_recomputed_sha256"] = hashlib.sha256(canonical).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "nested report self-hash mismatch"):
            hardening.verify_nested_authority(report, strict_verifier)
        self.assertEqual(calls, ["called", "called"])


if __name__ == "__main__":
    unittest.main()
