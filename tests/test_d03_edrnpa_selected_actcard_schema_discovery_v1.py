from __future__ import annotations

import io
import json
import zipfile

import pytest

import tools.run_d03_edrnpa_selected_actcard_schema_discovery_v1 as mod


def _nested(xml: bytes) -> tuple[zipfile.ZipFile, zipfile.ZipInfo]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("edrnpa.xml", xml)
    archive = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    info = archive.getinfo("edrnpa.xml")
    return archive, info


def _xml(*documents: str) -> bytes:
    return (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        "<rna><database>"
        + "".join(documents)
        + "</database></rna>"
    ).encode()


def _document(*items: str) -> str:
    return "<document>" + "".join(items) + "</document>"


def _value_item(name: str, value: str) -> str:
    return f'<item name="{name}"><text>{value}</text></item>'


def _legal_item(name: str, value: str) -> str:
    return (
        f'<item name="{name}"><richtext>'
        f'<par def="1">{value}</par>'
        "</richtext></item>"
    )


def test_discovers_only_selected_document_structure_without_values() -> None:
    xml = _xml(
        _document(
            _value_item("act-type", "НЕ ВИНОСИТИ У ДОКАЗ"),
            _value_item("issuer", "Міністерство A"),
            _legal_item("legal-text", "Перший нормативний текст"),
        ),
        _document(
            _value_item("act-type", "Закон"),
            _value_item("registration", "123"),
            _legal_item("legal-text", "Другий нормативний текст"),
        ),
        _document(
            _value_item("secret-only-field", "НЕ МАЄ ПОТРАПИТИ"),
            _legal_item("legal-text", "Третій текст"),
        ),
    )
    archive, info = _nested(xml)
    try:
        result = mod._discover_schema(
            nested=archive,
            info=info,
            selected_ordinals={2},
        )
    finally:
        archive.close()

    assert result["selected_documents_seen"] == 1
    assert [row["item_name"] for row in result["fields"]] == [
        "act-type",
        "legal-text",
        "registration",
    ]
    assert result["richtext_item_names"] == [
        {"item_name": "legal-text", "occurrence_count": 1}
    ]
    serialized = json.dumps(result, ensure_ascii=False)
    assert "Закон" not in serialized
    assert "123" not in serialized
    assert "НЕ ВИНОСИТИ У ДОКАЗ" not in serialized
    assert "secret-only-field" not in serialized


def test_structural_signatures_are_deterministic_and_text_free() -> None:
    xml = _xml(
        _document(
            _value_item("issuer", "Орган"),
            _legal_item("legal-text", "Текст"),
        )
    )
    archive, info = _nested(xml)
    try:
        first = mod._discover_schema(nested=archive, info=info, selected_ordinals={1})
    finally:
        archive.close()

    archive, info = _nested(xml)
    try:
        second = mod._discover_schema(nested=archive, info=info, selected_ordinals={1})
    finally:
        archive.close()

    assert first == second
    fields = {row["item_name"]: row for row in first["fields"]}
    assert fields["issuer"]["structural_signatures"] == ["text"]
    assert fields["legal-text"]["structural_signatures"] == ["richtext|richtext/par"]
    assert fields["issuer"]["richtext_occurrence_count"] == 0
    assert fields["legal-text"]["richtext_occurrence_count"] == 1


def test_same_field_across_selected_documents_counts_documents_and_occurrences() -> None:
    xml = _xml(
        _document(
            _value_item("issuer", "A"),
            _value_item("issuer", "B"),
            _legal_item("legal-text", "One"),
        ),
        _document(
            _value_item("issuer", "C"),
            _legal_item("legal-text", "Two"),
        ),
    )
    archive, info = _nested(xml)
    try:
        result = mod._discover_schema(
            nested=archive,
            info=info,
            selected_ordinals={1, 2},
        )
    finally:
        archive.close()

    issuer = next(row for row in result["fields"] if row["item_name"] == "issuer")
    assert issuer["selected_document_count"] == 2
    assert issuer["occurrence_count"] == 3


def test_missing_selected_ordinal_fails_closed() -> None:
    xml = _xml(_document(_legal_item("legal-text", "One")))
    archive, info = _nested(xml)
    try:
        with pytest.raises(mod.DiscoveryError, match="not all selected ordinals"):
            mod._discover_schema(
                nested=archive,
                info=info,
                selected_ordinals={2},
            )
    finally:
        archive.close()


@pytest.mark.parametrize("name", ["", "bad\x7fname", "bad\x01name"])
def test_unsafe_item_name_fails_closed(name: str) -> None:
    xml = _xml(_document(_value_item(name, "value"), _legal_item("legal-text", "One")))
    archive, info = _nested(xml)
    try:
        with pytest.raises(mod.DiscoveryError, match="item name"):
            mod._discover_schema(
                nested=archive,
                info=info,
                selected_ordinals={1},
            )
    finally:
        archive.close()


def test_selected_ordinals_reject_duplicate_identity_and_bad_format() -> None:
    duplicate = [
        {"source_path": "x#document:7"},
        {"source_path": "y#document:7"},
    ]
    with pytest.raises(mod.DiscoveryError, match="invalid/duplicate"):
        mod._selected_ordinals(duplicate)

    with pytest.raises(mod.DiscoveryError, match="ordinal"):
        mod._selected_ordinals([{"source_path": "x#document:+7"}])
