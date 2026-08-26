from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/verify302/data300_corpus_v03_cleanroom_verification.json"
DATA300 = ROOT / "configs/data/data300_corpus_v03_frozen_build_contract_v2.json"
DATA301 = ROOT / "configs/data/data301_corpus_v03_terminal_build_v1.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_verify302_cleanroom_terminal_blocker_matches_frozen_contract_and_data301() -> None:
    report = _load(REPORT)
    data300 = _load(DATA300)
    data301 = _load(DATA301)

    assert report["execution_profile"] == "LOCAL_FREE"
    assert report["model_training_executed"] is False
    assert report["cleanroom"]["data301_bytes_used_as_build_input"] is False
    assert report["cleanroom"]["data300_contract_identity_sha256"] == data300["contract_identity_sha256"]
    assert report["cleanroom"]["data300_head_sha"] == data301["base_data300"]["head_sha"]

    rows = report["cleanroom"]["source_rows"]
    assert len(rows) == data301["candidate_inventory"]["source_count"] == 5
    assert len({row["source_id"] for row in rows}) == 5
    assert len({row["normalized_sha256"] for row in rows}) == 5
    assert sum(row["normalized_bytes"] for row in rows) == data301["candidate_inventory"]["normalized_unique_bytes_prebuild"] == 183061

    by_stratum = {
        stratum: sum(row["normalized_bytes"] for row in rows if row["stratum"] == stratum)
        for stratum in ("uk", "en", "code")
    }
    assert by_stratum == {"uk": 88565, "en": 84793, "code": 9703}
    assert report["cleanroom"]["by_stratum_bytes"] == by_stratum

    family_counts = {
        stratum: len({row["family"] for row in rows if row["stratum"] == stratum})
        for stratum in ("uk", "en", "code")
    }
    assert family_counts == {"uk": 1, "en": 1, "code": 2}
    assert report["cleanroom"]["family_counts"] == family_counts

    assert report["cleanroom"]["build_a_source_inventory_sha256"] == report["cleanroom"]["build_b_source_inventory_sha256"]
    assert report["cleanroom"]["build_a_source_rebuild_sha256"] == report["cleanroom"]["build_b_source_rebuild_sha256"]
    assert report["cleanroom"]["source_prebuild_trees_identical"] is True

    assert data301["terminal_verdict"]["status"] == "TERMINAL_BLOCKED"
    assert data301["terminal_verdict"]["corpus_identity"] is None
    assert data301["terminal_verdict"]["shard_identity"] is None
    assert report["data301_comparison"]["status"] == "TERMINAL_BLOCKED"
    assert report["data301_comparison"]["corpus_identity"]["comparison"] == "MATCH_NULL"
    assert report["data301_comparison"]["split_membership"]["comparison"] == "MATCH_NO_SPLITS_MATERIALIZED"
    assert report["data301_comparison"]["shard_sha256"]["comparison"] == "MATCH_EMPTY_SET"
    assert report["reproducibility"]["unexplained_differences"] == []
    assert report["reproducibility"]["terminal_blocker_reproducible"] is True
    assert report["reproducibility"]["corpus_reproducibility_established"] is False


if __name__ == "__main__":
    test_verify302_cleanroom_terminal_blocker_matches_frozen_contract_and_data301()
    print("VERIFY-302 clean-room evidence: PASS")
