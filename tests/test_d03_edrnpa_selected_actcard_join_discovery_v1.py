from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import tools.run_d03_edrnpa_selected_actcard_join_discovery_v1 as mod


def _nested(xml: bytes) -> tuple[zipfile.ZipFile, zipfile.ZipInfo]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("edrnpa.xml", xml)
    archive = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    return archive, archive.getinfo("edrnpa.xml")


def _xml(*documents: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<rna><database>"
        + "".join(documents)
        + "</database></rna>"
    ).encode()


def _document(*items: str) -> str:
    return "<document>" + "".join(items) + "</document>"


def _text_item(name: str, value: str) -> str:
    return f'<item name="{name}"><text>{value}</text></item>'


def _rich_item(name: str, value: str) -> str:
    return (
        f'<item name="{name}"><richtext>'
        f'<par def="1">{value}</par>'
        "</richtext></item>"
    )


def _discover(
    xml: bytes,
    selected: dict[int, str],
) -> dict:
    archive, info = _nested(xml)
    try:
        return mod._discover_card_join(
            nested=archive,
            info=info,
            selected_codes_by_ordinal=selected,
        )
    finally:
        archive.close()


def test_complete_unique_join_is_aggregate_and_text_free() -> None:
    result = _discover(
        _xml(
            _document(
                _text_item("reestr_kod", "A-001"),
                _text_item("act-type", "Закон"),
                _text_item("issuer", "Орган A"),
            ),
            _document(
                _text_item("reestr_kod", "B-002"),
                _text_item("act-type", "Постанова"),
                _rich_item("summary", "Секретне значення"),
            ),
            _document(_text_item("reestr_kod", "UNSELECTED")),
        ),
        {11: "A-001", 27: "B-002"},
    )

    assert result["join_status"] == "COMPLETE_UNIQUE_SELECTED_TEXT_TO_CARD_JOIN"
    assert result["selected_text_documents"] == 2
    assert result["selected_unique_codes"] == 2
    assert result["matched_selected_unique_codes"] == 2
    assert result["unmatched_selected_unique_codes"] == 0
    assert result["ambiguous_selected_unique_codes"] == 0
    assert result["unique_join_card_documents"] == 2
    assert result["join_field_names"] == [
        {"item_name": "reestr_kod", "unique_selected_code_matches": 2}
    ]

    serialized = json.dumps(result, ensure_ascii=False)
    assert "A-001" not in serialized
    assert "B-002" not in serialized
    assert "Закон" not in serialized
    assert "Постанова" not in serialized
    assert "Орган A" not in serialized
    assert "Секретне значення" not in serialized


def test_unmatched_selected_code_is_explicitly_blocked() -> None:
    result = _discover(
        _xml(_document(_text_item("reestr_kod", "A-001"))),
        {1: "A-001", 2: "MISSING"},
    )

    assert result["join_status"] == "BLOCKED_UNMATCHED_SELECTED_CODES"
    assert result["matched_selected_unique_codes"] == 1
    assert result["unmatched_selected_unique_codes"] == 1


def test_duplicate_selected_text_code_is_explicitly_blocked() -> None:
    result = _discover(
        _xml(_document(_text_item("reestr_kod", "A-001"))),
        {1: "A-001", 2: "A-001"},
    )

    assert result["join_status"] == "BLOCKED_DUPLICATE_SELECTED_TEXT_CODE"
    assert result["duplicate_selected_code_groups"] == 1


def test_duplicate_card_occurrence_is_ambiguous_even_in_same_document() -> None:
    result = _discover(
        _xml(
            _document(
                _text_item("reestr_kod", "A-001"),
                _text_item("reestr_kod-copy", "A-001"),
            )
        ),
        {1: "A-001"},
    )

    assert result["join_status"] == "BLOCKED_AMBIGUOUS_CARD_MATCH"
    assert result["ambiguous_selected_unique_codes"] == 1
    assert result["uniquely_matched_selected_codes"] == 0


def test_empty_nonjoin_card_metadata_is_allowed_and_remains_structural() -> None:
    result = _discover(
        _xml(
            _document(
                _text_item("reestr_kod", "A-001"),
                _text_item("optional-empty", ""),
            )
        ),
        {1: "A-001"},
    )

    assert result["join_status"] == "COMPLETE_UNIQUE_SELECTED_TEXT_TO_CARD_JOIN"
    names = [
        row["item_name"]
        for row in result["matched_card_structure"]["fields"]
    ]
    assert names == ["optional-empty", "reestr_kod"]


def test_selected_text_code_collection_requires_exactly_one_code_per_selected_doc() -> None:
    missing = _xml(
        _document(_text_item("TEXT_ED", "one")),
        _document(
            _text_item("reestr_kod", "B-002"),
            _text_item("TEXT_ED", "two"),
        ),
    )
    archive, info = _nested(missing)
    try:
        with pytest.raises(mod.JoinDiscoveryError, match="missing reestr_kod"):
            mod._collect_selected_text_codes(
                nested=archive,
                info=info,
                selected_ordinals={1, 2},
            )
    finally:
        archive.close()

    duplicate = _xml(
        _document(
            _text_item("reestr_kod", "A-001"),
            _text_item("reestr_kod", "A-002"),
            _text_item("TEXT_ED", "one"),
        )
    )
    archive, info = _nested(duplicate)
    try:
        with pytest.raises(mod.JoinDiscoveryError, match="duplicate reestr_kod"):
            mod._collect_selected_text_codes(
                nested=archive,
                info=info,
                selected_ordinals={1},
            )
    finally:
        archive.close()


def test_selected_text_codes_are_collected_only_in_memory() -> None:
    xml = _xml(
        _document(
            _text_item("reestr_kod", "A-001"),
            _text_item("TEXT_ED", "one"),
        ),
        _document(
            _text_item("reestr_kod", "B-002"),
            _text_item("TEXT_ED", "two"),
        ),
    )
    archive, info = _nested(xml)
    try:
        codes = mod._collect_selected_text_codes(
            nested=archive,
            info=info,
            selected_ordinals={1, 2},
        )
    finally:
        archive.close()

    assert codes == {1: "A-001", 2: "B-002"}


def test_extract_named_nested_zip_rejects_missing_and_duplicate_members(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("25.2-edrnpa_text.zip", b"text")
    with pytest.raises(mod.JoinDiscoveryError, match="pinned nested ZIP member missing"):
        mod._extract_named_nested_zip(
            missing,
            tmp_path / "missing-out.zip",
            member_name=mod.CARD_ZIP_MEMBER,
        )

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(mod.CARD_ZIP_MEMBER, b"first")
        archive.writestr(mod.CARD_ZIP_MEMBER, b"second")
    with pytest.raises(mod.JoinDiscoveryError, match="duplicate nested ZIP member"):
        mod._extract_named_nested_zip(
            duplicate,
            tmp_path / "duplicate-out.zip",
            member_name=mod.CARD_ZIP_MEMBER,
        )


def test_control_character_in_join_value_fails_closed() -> None:
    xml = _xml(_document(_text_item("reestr_kod", "A&#1;001")))
    archive, info = _nested(xml)
    try:
        with pytest.raises(mod.JoinDiscoveryError):
            mod._discover_card_join(
                nested=archive,
                info=info,
                selected_codes_by_ordinal={1: "A001"},
            )
    finally:
        archive.close()


def test_selected_payload_identity_matches_historical_row_hash_contract() -> None:
    rows = [
        {
            "record_id": "edrnpa:a",
            "text": "перший",
            "training_eligible": False,
        },
        {
            "record_id": "edrnpa:b",
            "text": "другий",
            "training_eligible": False,
        },
    ]
    manual = __import__("hashlib").sha256()
    for row in rows:
        manual.update(mod.probe.cjson(row))

    observed = mod._selected_payload_identity(rows)

    assert observed == manual.hexdigest()
    tampered = [dict(row) for row in rows]
    tampered[1]["text"] = "інший"
    assert mod._selected_payload_identity(tampered) != observed
