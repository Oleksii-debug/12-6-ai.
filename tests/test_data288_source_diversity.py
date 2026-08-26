from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "data288",
    ROOT / "tools" / "validate_data288_source_diversity.py",
)
assert SPEC and SPEC.loader
data288 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data288)


def _inputs() -> dict:
    return json.loads(
        (ROOT / "configs" / "data" / "data288_source_diversity_inputs_v2.json").read_text()
    )


def test_exact_terminal_inventory_metrics() -> None:
    report = data288.build_report(_inputs())
    assert report["object_count"] == 5
    assert report["independent_family_count"] == 4
    assert report["independent_family_count_by_stratum"] == {"uk": 1, "en": 1, "code": 2}
    assert report["raw_source_bytes"] == 448214
    assert report["normalized_unique_bytes"] == 183061
    assert report["normalized_unique_bytes_by_stratum"] == {
        "uk": 88565,
        "en": 84793,
        "code": 9703,
    }
    assert abs(report["top_family_share_normalized_unique_bytes"] - 0.4838004818066109) < 1e-15
    assert abs(report["shannon_effective_family_count"] - 2.4270101760218714) < 1e-15
    assert report["next_corpus_diversity_status"] == "BLOCKED_SOURCE_DIVERSITY"


def test_multiple_standardebooks_files_do_not_create_extra_family() -> None:
    report = data288.build_report(_inputs())
    row = next(x for x in report["family_rows"] if x["family_id"] == "en.standardebooks.manual")
    assert row["file_count"] == 2
    assert row["normalized_unique_bytes"] == 84793
    assert report["independent_family_count_by_stratum"]["en"] == 1


def test_failed_data228_candidates_cannot_enter_audit() -> None:
    data = _inputs()
    data["terminal_authorities"]["DATA-228"]["status"] = "TERMINAL_SUCCESS"
    try:
        data288.build_report(data)
    except data288.AuditError as exc:
        assert "DATA-228 must remain excluded" in str(exc)
    else:
        raise AssertionError("failed DATA-228 candidate was admitted")


def test_cross_family_duplicate_evidence_fails_closed() -> None:
    data = _inputs()
    data["independent_overlap_audit"]["cross_family_duplicate_or_mirror_edges"] = 1
    try:
        data288.build_report(data)
    except data288.AuditError as exc:
        assert "cross-family duplicate/mirror edge present" in str(exc)
    else:
        raise AssertionError("cross-family duplicate edge did not fail closed")
