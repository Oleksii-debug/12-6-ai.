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
            f'<item name="metadata"><text>Документ {index}</text></item>'
            '<item name="legal-text"><richtext>'
            f'<par def="1">{text}</par>'
            "</richtext></item>"
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
    return mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data), **kwargs)


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
        data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data), byte_cap=5_000
    )
    rows_b, report_b = mod.materialize_archive_bytes(
        data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data), byte_cap=5_000
    )
    assert rows_a == rows_b
    assert report_a == report_b


def test_removes_only_xml10_forbidden_c0_byte_and_binds_offset() -> None:
    xml = _xml([_uk("A")])
    marker = "нормативний".encode()
    offset = xml.index(marker) + len(marker)
    dirty = xml[:offset] + b"\x0c" + xml[offset:]
    data = _archive_from_xml(dirty)
    rows, report = mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))
    assert len(rows) == 1
    assert report["selection"]["xml10_control_bytes_removed"] == 1
    expected = mod.sha256(f"{offset}:0c\n".encode("ascii"))
    assert report["selection"]["xml10_control_removal_identity_sha256"] == expected
    assert report["normalization"]["xml10_control_removal_identity_sha256"] == expected


def test_does_not_recover_malformed_markup_after_control_cleaning() -> None:
    text = _uk("A") + " & незаконний"
    data = _archive_from_xml(_xml([text]))
    with pytest.raises(mod.ProbeError, match="malformed nested EDRNPA XML"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_rejects_resource_hash_mismatch() -> None:
    data = _archive_from_xml(_xml([_uk("A")]))
    with pytest.raises(mod.ProbeError, match="pinned resource digest mismatch"):
        mod.materialize_archive_bytes(data, expected_md5="0" * 32)


def test_rejects_outer_zip_path_traversal_even_if_target_exists() -> None:
    nested = _zip({"edrnpa.xml": _xml([_uk("A")])})
    data = _zip({mod.NESTED_TEXT_ZIP: nested, "../escape.txt": b"bad"})
    with pytest.raises(mod.ProbeError, match="unsafe zip member"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_rejects_nested_zip_path_traversal() -> None:
    data = _archive_from_xml(_xml([_uk("A")]), xml_name="../edrnpa.xml")
    with pytest.raises(mod.ProbeError, match="unsafe zip member"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_rejects_dtd_or_entity_even_when_control_byte_interrupts_marker() -> None:
    xml = (
        b'<!DOC\x0cTYPE rna [<!ENTITY x "boom">]>'
        b'<rna><database><document><text>&x;</text></document></database></rna>'
    )
    data = _archive_from_xml(xml)
    with pytest.raises(mod.ProbeError, match="DTD/entity"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_rejects_malformed_nested_xml() -> None:
    data = _archive_from_xml(b"<rna><database><document><text>broken")
    with pytest.raises(mod.ProbeError, match="malformed nested"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


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
        mod.materialize_archive_bytes(bad_root, expected_md5=mod.md5(bad_root),
        expected_sha256=mod.sha256(bad_root))
    outside_xml = ("<rna><document><text>" + _uk("A") + "</text></document></rna>").encode()
    outside = _archive_from_xml(outside_xml)
    with pytest.raises(mod.ProbeError, match="outside database"):
        mod.materialize_archive_bytes(outside, expected_md5=mod.md5(outside),
        expected_sha256=mod.sha256(outside))


def test_rejects_missing_or_multiple_nested_xml_payloads() -> None:
    missing = _zip({"other.zip": _zip({"x.xml": _xml([_uk("A")])})})
    with pytest.raises(mod.ProbeError, match="nested text ZIP member missing"):
        mod.materialize_archive_bytes(missing, expected_md5=mod.md5(missing),
        expected_sha256=mod.sha256(missing))

    nested = _zip({"a.xml": _xml([_uk("A")]), "b.xml": _xml([_uk("B")])})
    multiple = _zip({mod.NESTED_TEXT_ZIP: nested})
    with pytest.raises(mod.ProbeError, match="exactly one XML"):
        mod.materialize_archive_bytes(multiple, expected_md5=mod.md5(multiple),
        expected_sha256=mod.sha256(multiple))


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
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_rejects_invalid_cap() -> None:
    with pytest.raises(mod.ProbeError, match="byte_cap"):
        _materialize_texts([_uk("A")], byte_cap=mod.MAX_SELECTED_BYTES + 1)


def test_report_self_hash_is_stable() -> None:
    _, report = _materialize_texts([_uk("A")])
    claimed = report.pop("report_identity_sha256")
    assert mod.sha256(mod.cjson(report)) == claimed


def test_rejects_legacy_guessed_text_richtext_path() -> None:
    xml = (
        "<rna><database><document><text><richtext>"
        f'<par def="1">{_uk("WRONG")}</par>'
        "</richtext></text></document></database></rna>"
    ).encode()
    data = _archive_from_xml(xml)
    with pytest.raises(mod.ProbeError, match="unexpected EDRNPA richtext path"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data),
        expected_sha256=mod.sha256(data))


def test_requires_exactly_one_observed_richtext_item() -> None:
    missing_xml = (
        "<rna><database><document>"
        '<item name="metadata"><text>Документ</text></item>'
        "</document></database></rna>"
    ).encode()
    missing = _archive_from_xml(missing_xml)
    with pytest.raises(mod.ProbeError, match="exactly one richtext"):
        mod.materialize_archive_bytes(missing, expected_md5=mod.md5(missing),
        expected_sha256=mod.sha256(missing))

    text = _uk("TWO")
    multiple_xml = (
        "<rna><database><document>"
        '<item name="one"><richtext>'
        f'<par def="1">{text}</par></richtext></item>'
        '<item name="two"><richtext>'
        f'<par def="1">{text}</par></richtext></item>'
        "</document></database></rna>"
    ).encode()
    multiple = _archive_from_xml(multiple_xml)
    with pytest.raises(mod.ProbeError, match="exactly one richtext"):
        mod.materialize_archive_bytes(multiple, expected_md5=mod.md5(multiple),
        expected_sha256=mod.sha256(multiple))


def test_rejects_observed_schema_attribute_drift() -> None:
    text = _uk("ATTR")
    bad_par_xml = (
        "<rna><database><document>"
        '<item name="legal-text"><richtext>'
        f'<par wrong="1">{text}</par></richtext></item>'
        "</document></database></rna>"
    ).encode()
    bad_par = _archive_from_xml(bad_par_xml)
    with pytest.raises(mod.ProbeError, match="paragraph attributes"):
        mod.materialize_archive_bytes(bad_par, expected_md5=mod.md5(bad_par),
        expected_sha256=mod.sha256(bad_par))

    bad_item_xml = (
        "<rna><database><document>"
        '<item wrong="legal-text"><richtext>'
        f'<par def="1">{text}</par></richtext></item>'
        "</document></database></rna>"
    ).encode()
    bad_item = _archive_from_xml(bad_item_xml)
    with pytest.raises(mod.ProbeError, match="item attributes"):
        mod.materialize_archive_bytes(bad_item, expected_md5=mod.md5(bad_item),
        expected_sha256=mod.sha256(bad_item))


def test_known_historical_resource_has_exact_sha256_pin() -> None:
    assert mod.RESOURCE_SHA256 == (
        "a87b23eff3aadd2ff2dbed24cdeaaaf37862ecebe65b5f1edad02d5a95cde431"
    )


def test_correct_md5_wrong_sha256_rejects_before_zip_processing() -> None:
    payload = b"not a ZIP archive"
    with pytest.raises(mod.ProbeError, match="pinned resource digest mismatch"):
        mod.materialize_archive_bytes(
            payload,
            expected_md5=mod.md5(payload),
            expected_sha256="0" * 64,
        )


def test_download_wrong_sha256_with_matching_md5_unlinks_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"bounded independent test payload"

    class Response(io.BytesIO):
        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.headers: dict[str, str] = {}

    monkeypatch.setattr(mod, "RESOURCE_MD5", mod.md5(payload))
    monkeypatch.setattr(mod, "RESOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_kw: Response(payload))
    destination = tmp_path / "not-published.zip"
    with pytest.raises(mod.ProbeError, match="pinned resource digest mismatch"):
        mod.download_archive_to(destination)
    assert not destination.exists()


def test_download_never_truncates_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "source.zip"
    existing = b"UNRELATED USER FILE MUST SURVIVE"
    destination.write_bytes(existing)

    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("network must not be requested for an existing target")

    monkeypatch.setattr(mod.urllib.request, "urlopen", forbidden_network)
    with pytest.raises(mod.ProbeError, match="refusing to overwrite"):
        mod.download_archive_to(destination)
    assert destination.read_bytes() == existing
    assert not list(tmp_path.glob(".source.zip.partial-*"))


def test_download_publishes_only_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic pinned source bytes"

    class Response(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.headers: dict[str, str] = {}

    monkeypatch.setattr(mod, "RESOURCE_MD5", mod.md5(payload))
    monkeypatch.setattr(mod, "RESOURCE_SHA256", mod.sha256(payload))
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_kw: Response())
    destination = tmp_path / "source.zip"
    result = mod.download_archive_to(destination)
    assert destination.read_bytes() == payload
    assert result == {
        "bytes": len(payload),
        "md5": mod.md5(payload),
        "sha256": mod.sha256(payload),
    }
    assert not list(tmp_path.glob(".source.zip.partial-*"))


def test_interrupted_download_removes_only_its_private_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Interrupted(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"part of a source")
            self.headers: dict[str, str] = {}

        def read(self, size: int = -1) -> bytes:
            if self.tell() >= 4:
                raise OSError("interrupted source stream")
            return super().read(4)

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_kw: Interrupted())
    destination = tmp_path / "source.zip"
    with pytest.raises(OSError, match="interrupted source stream"):
        mod.download_archive_to(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".source.zip.partial-*"))


def test_oversized_download_header_never_creates_final_or_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Oversized(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"")
            self.headers = {"Content-Length": str(mod.MAX_ARCHIVE_BYTES + 1)}

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_kw: Oversized())
    destination = tmp_path / "source.zip"
    with pytest.raises(mod.ProbeError, match="exceeds compressed-byte"):
        mod.download_archive_to(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".source.zip.partial-*"))


def test_competing_target_created_at_publish_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified fixture payload"
    unrelated = b"UNRELATED CONCURRENT USER FILE"
    destination = tmp_path / "source.zip"

    class Competitor(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.headers: dict[str, str] = {}

        def read(self, size: int = -1) -> bytes:
            chunk = super().read(size)
            if not chunk:
                destination.write_bytes(unrelated)
            return chunk

    monkeypatch.setattr(mod, "RESOURCE_MD5", mod.md5(payload))
    monkeypatch.setattr(mod, "RESOURCE_SHA256", mod.sha256(payload))
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_kw: Competitor())
    with pytest.raises(mod.ProbeError, match="refusing to overwrite"):
        mod.download_archive_to(destination)
    assert destination.read_bytes() == unrelated
    assert not list(tmp_path.glob(".source.zip.partial-*"))


def test_explicit_preexisting_work_dir_is_never_mutated(tmp_path: Path) -> None:
    data = _archive_from_xml(_xml([_uk("WORK")]))
    source = tmp_path / "source.zip"
    source.write_bytes(data)
    work_dir = tmp_path / "existing workspace"
    work_dir.mkdir()
    existing_nested = work_dir / "nested-text.zip"
    existing_nested.write_bytes(b"UNRELATED EXISTING FILE")
    with pytest.raises(mod.ProbeError, match="refusing pre-existing work_dir"):
        mod.materialize_archive_path(
            source,
            expected_md5=mod.md5(data),
            expected_sha256=mod.sha256(data),
            work_dir=work_dir,
        )
    assert existing_nested.read_bytes() == b"UNRELATED EXISTING FILE"
    assert source.read_bytes() == data
    assert not (work_dir / "candidates.sqlite3").exists()


@pytest.mark.parametrize("existing_kind", ("records", "report"))
def test_output_publication_never_overwrites_user_files(
    tmp_path: Path, existing_kind: str
) -> None:
    rows, report = _materialize_texts([_uk("OUTPUT")])
    records = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    selected = {"records": records, "report": report_path}[existing_kind]
    selected.write_bytes(b"UNRELATED USER EVIDENCE")
    with pytest.raises(mod.ProbeError, match="refusing to overwrite"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    assert selected.read_bytes() == b"UNRELATED USER EVIDENCE"
    assert not (tmp_path / "records.jsonl.incomplete").exists()
    other = report_path if existing_kind == "records" else records
    assert not other.exists()


def test_output_publication_rejects_same_final_path(tmp_path: Path) -> None:
    rows, report = _materialize_texts([_uk("ALIAS")])
    shared = tmp_path / "one-file.json"
    with pytest.raises(mod.ProbeError, match="must be distinct"):
        mod._write_outputs(rows, report, records_path=shared, report_path=shared)
    assert not shared.exists()


def test_output_publication_preserves_preexisting_foreign_marker(
    tmp_path: Path,
) -> None:
    rows, report = _materialize_texts([_uk("MARKER")])
    records = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    marker = tmp_path / "records.jsonl.incomplete"
    marker.write_bytes(b"UNRELATED EXISTING MARKER")
    with pytest.raises(mod.ProbeError, match="refusing to overwrite"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    assert marker.read_bytes() == b"UNRELATED EXISTING MARKER"
    assert not records.exists()
    assert not report_path.exists()


def test_output_publication_roundtrips_unicode_paths_and_bytes(
    tmp_path: Path,
) -> None:
    rows, report = _materialize_texts([_uk("УКРАЇНА")])
    directory = tmp_path / "каталог зі пробілами"
    records = directory / "записи.jsonl"
    report_path = directory / "звіт.json"
    mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    expected_records = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    assert records.read_bytes() == expected_records
    assert report_path.read_bytes() == mod.cjson(report)
    assert not (directory / "записи.jsonl.incomplete").exists()
    assert not list(directory.glob(".*.stage-*"))


def test_competing_report_preserves_user_bytes_and_incomplete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, report = _materialize_texts([_uk("COMPETE")])
    records = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    marker = tmp_path / "records.jsonl.incomplete"
    unexpected = b"COMPETING UNRELATED REPORT"
    original_link = mod.os.link

    def compete(stage: Path, destination: Path) -> None:
        if destination == report_path:
            report_path.write_bytes(unexpected)
        original_link(stage, destination)

    monkeypatch.setattr(mod.os, "link", compete)
    with pytest.raises(mod.ProbeError, match="incomplete publication marker"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    assert records.is_file()
    assert report_path.read_bytes() == unexpected
    assert marker.read_bytes() == b"INCOMPLETE_NOT_TERMINAL\n"
    assert not list(tmp_path.glob(".*.stage-*"))
    with pytest.raises(mod.ProbeError, match="refusing to overwrite"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)


def test_failed_report_staging_retains_marker_without_final_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, report = _materialize_texts([_uk("FAIL")])
    records = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    marker = tmp_path / "records.jsonl.incomplete"

    def fail_report(_value):
        raise mod.ProbeError("injected report serialization failure")

    monkeypatch.setattr(mod, "cjson", fail_report)
    with pytest.raises(mod.ProbeError, match="injected report serialization failure"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    assert marker.read_bytes() == b"INCOMPLETE_NOT_TERMINAL\n"
    assert not records.exists()
    assert not report_path.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_substituted_control_marker_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, report = _materialize_texts([_uk("CONTROL")])
    records = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"
    marker = tmp_path / "records.jsonl.incomplete"
    displaced_marker = tmp_path / "retained original marker"
    original_link = mod.os.link

    def substitute_marker(stage: Path, destination: Path) -> None:
        if destination == report_path:
            marker.rename(displaced_marker)
            marker.write_bytes(b"UNRELATED REPLACEMENT MARKER")
        original_link(stage, destination)

    monkeypatch.setattr(mod.os, "link", substitute_marker)
    with pytest.raises(mod.ProbeError, match="marker identity changed"):
        mod._write_outputs(rows, report, records_path=records, report_path=report_path)
    assert marker.read_bytes() == b"UNRELATED REPLACEMENT MARKER"
    assert displaced_marker.read_bytes() == b"INCOMPLETE_NOT_TERMINAL\n"
    assert records.is_file() and report_path.is_file()
    assert not list(tmp_path.glob(".*.stage-*"))
