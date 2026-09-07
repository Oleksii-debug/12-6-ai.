from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.snapshot_d03_hplt3_ukr_cyrl_controls import (
    CONFIG,
    HpltControlError,
    build_report,
    load_config,
    parse_map,
    parse_md5,
)


def _config() -> dict:
    return json.loads(Path(CONFIG).read_text(encoding="utf-8"))


def _map(*filenames: str) -> bytes:
    return (
        "\n".join(
            "https://data.hplt-project.org/three/sorted/ukr_Cyrl/" + filename
            for filename in filenames
        )
        + "\n"
    ).encode()


def _md5(rows: dict[str, str]) -> bytes:
    return ("\n".join(f"{digest}  ukr_Cyrl/{filename}" for filename, digest in rows.items()) + "\n").encode()


def test_control_snapshot_is_deterministic_zero_credit_and_selects_highest_bin() -> None:
    mapping = _map("9_2.jsonl.zst", "10_2.jsonl.zst", "10_1.jsonl.zst")
    checksums = _md5(
        {
            "9_2.jsonl.zst": "1" * 32,
            "10_2.jsonl.zst": "2" * 32,
            "10_1.jsonl.zst": "3" * 32,
        }
    )
    first = build_report(mapping, checksums, _config())
    second = build_report(mapping, checksums, _config())

    assert first == second
    assert first["shard_inventory"]["shard_count"] == 3
    assert first["bounded_selection"]["highest_wds_bin"] == 10
    assert first["bounded_selection"]["selected"]["filename"] == "10_1.jsonl.zst"
    assert first["bounded_selection"]["selected"]["upstream_md5"] == "3" * 32
    assert first["bounded_selection"]["selection_uses_evaluation_results"] is False
    assert first["gates"]["control_object_snapshot"] == "PASS"
    assert first["gates"]["selected_shard_sha256"] == "NOT_RUN"
    assert first["claim_boundary"]["immutable_control_snapshot"] is True
    assert first["claim_boundary"]["immutable_shard_sha256_identity"] is False
    assert first["claim_boundary"]["shards_downloaded"] == 0
    assert first["claim_boundary"]["training_authorized_bytes"] == 0
    assert first["claim_boundary"]["tokenizer_fit_authorized"] is False
    assert first["claim_boundary"]["model_training_executed"] is False
    assert first["claim_boundary"]["paid_compute_used"] is False
    assert len(first["control_snapshot"]["map_sha256"]) == 64
    assert len(first["control_snapshot"]["md5_sha256"]) == 64
    assert len(first["shard_inventory"]["inventory_identity_sha256"]) == 64
    assert len(first["report_sha256"]) == 64


def test_map_and_md5_inventory_must_match_exactly() -> None:
    mapping = _map("10_1.jsonl.zst", "9_1.jsonl.zst")
    checksums = _md5({"10_1.jsonl.zst": "a" * 32})

    with pytest.raises(HpltControlError, match="inventory mismatch"):
        build_report(mapping, checksums, _config())


def test_map_rejects_wrong_host_and_query() -> None:
    config = _config()
    wrong_host = b"https://example.invalid/three/sorted/ukr_Cyrl/10_1.jsonl.zst\n"
    with pytest.raises(HpltControlError, match="host rejected"):
        parse_map(wrong_host, config)

    query = b"https://data.hplt-project.org/three/sorted/ukr_Cyrl/10_1.jsonl.zst?x=1\n"
    with pytest.raises(HpltControlError, match="query/fragment rejected"):
        parse_map(query, config)


def test_map_rejects_duplicate_filename_even_when_url_text_differs() -> None:
    mapping = (
        b"https://data.hplt-project.org/three/sorted/ukr_Cyrl/10_1.jsonl.zst\n"
        b"https://data.hplt-project.org:443/three/sorted/ukr_Cyrl/10_1.jsonl.zst\n"
    )
    with pytest.raises(HpltControlError, match="duplicate map filename"):
        parse_map(mapping, _config())


def test_md5_accepts_md5sum_star_and_nested_path_but_rejects_duplicates() -> None:
    config = _config()
    parsed = parse_md5(
        (
            "a" * 32
            + "  ukr_Cyrl/10_1.jsonl.zst\n"
            + "b" * 32
            + " *9_1.jsonl.zst\n"
        ).encode(),
        config,
    )
    assert parsed == {"10_1.jsonl.zst": "a" * 32, "9_1.jsonl.zst": "b" * 32}

    duplicate = (
        "a" * 32
        + "  10_1.jsonl.zst\n"
        + "b" * 32
        + "  ukr_Cyrl/10_1.jsonl.zst\n"
    ).encode()
    with pytest.raises(HpltControlError, match="duplicate MD5 filename"):
        parse_md5(duplicate, config)


def test_invalid_shard_name_and_md5_path_traversal_fail_closed() -> None:
    config = _config()
    with pytest.raises(HpltControlError, match="filename rejected"):
        parse_map(
            b"https://data.hplt-project.org/three/sorted/ukr_Cyrl/4_1.jsonl.zst\n",
            config,
        )

    with pytest.raises(HpltControlError, match="path traversal"):
        parse_md5(("a" * 32 + "  ../10_1.jsonl.zst\n").encode(), config)


def test_control_objects_respect_frozen_size_cap() -> None:
    config = _config()
    config["control_objects"]["max_each_bytes"] = 3
    with pytest.raises(HpltControlError, match="map control exceeds size cap"):
        build_report(_map("10_1.jsonl.zst"), _md5({"10_1.jsonl.zst": "a" * 32}), config)


def test_source_audit_config_authority_drift_is_rejected(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["bounded_successor"]["bulk_download_forbidden"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(HpltControlError, match="bulk-download boundary weakened"):
        load_config(path)


def test_production_control_config_loads_with_zero_credit_boundary() -> None:
    config = load_config()
    assert config["parent"]["source_id"] == "HPLT-3.0-ukr_Cyrl"
    assert config["claim_boundary"]["training_authorized_bytes"] == 0
    assert config["claim_boundary"]["model_training_executed"] is False
