from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/diagnose_next100_063_balance_capacity_v5.py"
V4 = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
V5 = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"

spec = importlib.util.spec_from_file_location("next100_063_balance_v5", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_attrs_improves_code_raw_capacity_but_uk_still_limits_balance() -> None:
    v4 = json.loads(V4.read_text(encoding="utf-8"))
    raw_v5 = V5.read_bytes()
    v5 = json.loads(raw_v5.decode("utf-8"))
    report = module.build_report(
        v4,
        v5,
        v5_blob_sha1=module.v5_validator.git_blob_sha1(raw_v5),
    )

    assert report["raw_pre_successor_global_dedup_numeric_training_capacity_bytes"] == 2_215_615
    assert report["raw_capacity_by_stratum"] == {
        "uk": 100_856,
        "en": 1_838_293,
        "code": 276_466,
    }
    assert report["family_count_by_stratum"] == {"uk": 4, "en": 5, "code": 6}
    assert report["20m_raw_capacity_gap_by_stratum"] == {
        "uk": 8_899_144,
        "en": 5_161_707,
        "code": 3_723_534,
    }
    assert report["diagnostic_exact_mixture_family_capped_source_bytes"] == 61_440
    assert report["next_20_byte_increment_limiting_strata"] == ["uk"]
    assert report["truth_boundary"]["post_pack_unique_loss_positions"] == 0
    assert report["truth_boundary"]["training_authorized"] is False
    assert report["truth_boundary"]["paid_compute_authorized"] is False
