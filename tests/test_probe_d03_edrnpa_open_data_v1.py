from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "probe_d03_edrnpa_open_data_v1.py"
SPEC = importlib.util.spec_from_file_location("edrnpa_probe_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _uk(label: str, repeats: int = 50) -> str:
    return ((f"Верховенство права та нормативний акт {label}. " * repeats).strip())


def _xml(texts: list[str]) -> bytes:
    documents = []
    for index, text in enumerate(texts, start=1):
        documents.append(
            "<document>"
            f'<item name="id">{index}</item>'
            f'<item name="title">Документ {index}</item>'
            "<text><richtext>"
            f"<par>{text}</par>"
            "</richtext></text>"
            "</document>"
        )
    body = "".join(documents)
    return (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        f"<rna><database>{body}</database></rna>"
    ).encode()


def _zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, payload in files.items():
            archive.writestr(path, payload)
    return output.getvalue()


def _archive_from_xml(xml: bytes, *, xml_name: str = "edrnpa_texts_test.xml") -> bytes:
    nested = _zip({xml_name: xml})
    return _zip({mod.NESTED_TEXT_ZIP: nested, "25.1-metadata.txt": b"metadata"})


def _materialize_texts(texts: list[str], **kwargs):
    data = _archive_from_xml(_xml(texts))
    return mod.materialize_archive_bytes(data, expected_md5=mod.md5(data), **kwargs)


def test_materializes_observed_nested_document_shape_and_keeps_zero_credit() -> None:
    rows, report = _materialize_texts([_uk("A"), _uk("B")])
    assert len(rows) == 2
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["evaluation_eligible"] is False for row in rows)
    assert report["source"]["format"] == "XML-in-ZIP-in-ZIP"
    assert report["selection"]["source_documents"] == 2
    assert report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0
    assert report["truth_boundary"]["training_authorized_bytes"] == 0
    assert report["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert report["rights"]["training_rights_admitted_by_this_probe"] is False
    assert report["normalization"]["generic_xml_recovery_enabled"] is False


def test_deterministic_replay_of_identical_source() -> None:
    data = _archive_from_xml(_xml([_uk("A"), _uk("B"), _uk("C")]))
    rows_a, report_a = mod.materialize_archive_bytes(
        data, expected_md5=mod.md5(data), byte_cap=5_000
    )
    rows_b, report_b = mod.materialize_archive_bytes(
        data, expected_md5=mod.md5(data), byte_cap=5_000
    )
    assert rows_a == rows_b
    assert report_a == report_b


def test_removes_only_xml10_forbidden_c0_byte_and_binds_offset() -> None:
    xml = _xml([_uk("A")])
    marker = "нормативний".encode("utf-8")
    offset = xml.index(marker) + len(marker)
    dirty = xml[:offset] + b"\x0c" + xml[offset:]
    data = _archive_from_xml(dirty)
    rows, report = mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))
    assert len(rows) == 1
    assert report["selection"]["xml10_control_bytes_removed"] == 1
    expected = mod.sha256(f"{offset}:0c\n".encode("ascii"))
    assert report["selection"]["xml10_control_removal_identity_sha256"] == expected
    assert report["normalization"]["xml10_control_removal_identity_sha256"] == expected


def test_does_not_recover_malformed_markup_after_control_cleaning() -> None:
    text = _uk("A") + " & незаконний"
    data = _archive_from_xml(_xml([text]))
    with pytest.raises(mod.ProbeError, match="malformed nested EDRNPA XML"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_resource_hash_mismatch() -> None:
    data = _archive_from_xml(_xml([_uk("A")]))
    with pytest.raises(mod.ProbeError, match="MD5 mismatch"):
        mod.materialize_archive_bytes(data, expected_md5="0" * 32)


def test_rejects_outer_zip_path_traversal_even_if_target_exists() -> None:
    nested = _zip({"edrnpa.xml": _xml([_uk("A")])})
    data = _zip({mod.NESTED_TEXT_ZIP: nested, "../escape.txt": b"bad"})
    with pytest.raises(mod.ProbeError, match="unsafe zip member"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_nested_zip_path_traversal() -> None:
    data = _archive_from_xml(_xml([_uk("A")]), xml_name="../edrnpa.xml")
    with pytest.raises(mod.ProbeError, match="unsafe zip member"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_dtd_or_entity_even_when_control_byte_interrupts_marker() -> None:
    xml = (
        b'<!DOC\x0cTYPE rna [<!ENTITY x "boom">]>'
        b'<rna><database><document><text>&x;</text></document></database></rna>'
    )
    data = _archive_from_xml(xml)
    with pytest.raises(mod.ProbeError, match="DTD/entity"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_malformed_nested_xml() -> None:
    data = _archive_from_xml(b"<rna><database><document><text>broken")
    with pytest.raises(mod.ProbeError, match="malformed nested"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_quarantines_email_and_phone_documents() -> None:
    rows, report = _materialize_texts(
        [
            _uk("clean"),
            _uk("email") + " contact@example.org",
            _uk("phone") + " +380501234567",
        ]
    )
    assert len(rows) == 1
    assert report["selection"]["quarantined_email_or_phone"] == 2


def test_deduplicates_normalized_text_deterministically() -> None:
    same = _uk("same")
    rows, report = _materialize_texts([same, same])
    assert len(rows) == 1
    assert report["selection"]["duplicate_normalized_objects"] == 1
    assert report["selection"]["candidate_objects_after_quality"] == 1


def test_filters_non_ukrainian_document() -> None:
    english = ("This is a long English legal document with regulations. " * 80).strip()
    rows, report = _materialize_texts([_uk("UA"), english])
    assert len(rows) == 1
    assert rows[0]["quality"]["ukrainian_alpha_ratio"] >= mod.MIN_UKRAINIAN_ALPHA_RATIO
    assert report["selection"]["documents_non_ukrainian"] == 1


def test_metadata_items_are_not_mixed_into_candidate_text() -> None:
    rows, _ = _materialize_texts([_uk("BODY")])
    assert "Документ 1" not in rows[0]["text"]
    assert "Верховенство права" in rows[0]["text"]


def test_rejects_unexpected_root_and_document_outside_database() -> None:
    bad_root = _archive_from_xml(b"<root><database/></root>")
    with pytest.raises(mod.ProbeError, match="unexpected EDRNPA XML root"):
        mod.materialize_archive_bytes(bad_root, expected_md5=mod.md5(bad_root))
    outside_xml = ("<rna><document><text>" + _uk("A") + "</text></document></rna>").encode()
    outside = _archive_from_xml(outside_xml)
    with pytest.raises(mod.ProbeError, match="outside database"):
        mod.materialize_archive_bytes(outside, expected_md5=mod.md5(outside))


def test_rejects_missing_or_multiple_nested_xml_payloads() -> None:
    missing = _zip({"other.zip": _zip({"x.xml": _xml([_uk("A")])})})
    with pytest.raises(mod.ProbeError, match="nested text ZIP member missing"):
        mod.materialize_archive_bytes(missing, expected_md5=mod.md5(missing))

    nested = _zip({"a.xml": _xml([_uk("A")]), "b.xml": _xml([_uk("B")])})
    multiple = _zip({mod.NESTED_TEXT_ZIP: nested})
    with pytest.raises(mod.ProbeError, match="exactly one XML"):
        mod.materialize_archive_bytes(multiple, expected_md5=mod.md5(multiple))


def test_rejects_symlink_member() -> None:
    nested = _zip({"edrnpa.xml": _xml([_uk("A")])})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(mod.NESTED_TEXT_ZIP, nested)
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"target")
    data = output.getvalue()
    with pytest.raises(mod.ProbeError, match="symlink"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_invalid_cap() -> None:
    with pytest.raises(mod.ProbeError, match="byte_cap"):
        _materialize_texts([_uk("A")], byte_cap=mod.MAX_SELECTED_BYTES + 1)


def test_report_self_hash_is_stable() -> None:
    _, report = _materialize_texts([_uk("A")])
    claimed = report.pop("report_identity_sha256")
    assert mod.sha256(mod.cjson(report)) == claimed
