from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_data526_v8_reserved_decontamination as runner


@pytest.mark.parametrize(
    "payload",
    (
        '{"outer":{"a":1,"a":2}}',
        '{"a":1,"\\u0061":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ),
)
def test_load_json_rejects_ambiguous_authority_input(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid strict JSON"):
        runner._load_json(path)


@pytest.mark.parametrize(
    "bad_row",
    (
        '{"outer":{"a":1,"a":2}}',
        '{"a":1,"\\u0061":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ),
)
def test_load_jsonl_rejects_ambiguous_record_before_returning_rows(
    tmp_path: Path,
    bad_row: str,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"record_id":"ok","score":1.25}\n' + bad_row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"records\.jsonl:2"):
        runner._load_jsonl(path)


def test_strict_loaders_preserve_valid_finite_objects(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text(
        '{"schema":"v1","finite":1.25,"nested":{"count":2}}',
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"record_id":"r1","score":0.5}\n'
        '{"record_id":"r2","score":2}\n',
        encoding="utf-8",
    )

    assert runner._load_json(authority) == {
        "schema": "v1",
        "finite": 1.25,
        "nested": {"count": 2},
    }
    assert runner._load_jsonl(records) == [
        {"record_id": "r1", "score": 0.5},
        {"record_id": "r2", "score": 2},
    ]


def test_strict_loaders_keep_existing_object_root_requirements(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("[]", encoding="utf-8")
    records = tmp_path / "records.jsonl"
    records.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="JSON root must be an object"):
        runner._load_json(authority)
    with pytest.raises(TypeError, match="JSONL line 1 must be an object"):
        runner._load_jsonl(records)
