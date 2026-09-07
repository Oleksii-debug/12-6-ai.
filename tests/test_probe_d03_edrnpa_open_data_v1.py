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


def _xml(text: str) -> bytes:
    return f'<?xml version="1.0" encoding="utf-8"?><act><text>{text}</text></act>'.encode()


def _uk(label: str, repeats: int = 50) -> str:
    return ((f"Верховенство права та нормативний акт {label}. " * repeats).strip())


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in files.items():
            archive.writestr(path, payload)
    return output.getvalue()


def _materialize(files: dict[str, bytes], **kwargs):
    data = _archive(files)
    return mod.materialize_archive_bytes(data, expected_md5=mod.md5(data), **kwargs)


def test_materializes_ukrainian_xml_and_keeps_zero_credit() -> None:
    rows, report = _materialize(
        {
            "texts/a.xml": _xml(_uk("A")),
            "texts/b.xml": _xml(_uk("B")),
            "schema/main.xsd": b"<schema/>",
            "readme.txt": b"metadata",
        }
    )
    assert len(rows) == 2
    assert all(row["training_eligible"] is False for row in rows)
    assert report["selection"]["selected_bytes"] > 0
    assert report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0
    assert report["truth_boundary"]["training_authorized_bytes"] == 0
    assert report["truth_boundary"]["unique_causal_loss_positions"] == 0
    assert report["rights"]["training_rights_admitted_by_this_probe"] is False


def test_deterministic_selection_independent_of_zip_order() -> None:
    files = {
        "texts/a.xml": _xml(_uk("A")),
        "texts/b.xml": _xml(_uk("B")),
        "texts/c.xml": _xml(_uk("C")),
    }
    rows_a, report_a = _materialize(files, byte_cap=4_000)
    rows_b, report_b = _materialize(
        dict(reversed(list(files.items()))),
        byte_cap=4_000,
    )
    assert rows_a == rows_b
    assert report_a["selection"]["inventory_identity_sha256"] == (
        report_b["selection"]["inventory_identity_sha256"]
    )


def test_rejects_resource_hash_mismatch() -> None:
    data = _archive({"texts/a.xml": _xml(_uk("A"))})
    with pytest.raises(mod.ProbeError, match="MD5 mismatch"):
        mod.materialize_archive_bytes(data, expected_md5="0" * 32)


def test_rejects_zip_path_traversal() -> None:
    data = _archive({"../escape.xml": _xml(_uk("A"))})
    with pytest.raises(mod.ProbeError, match="unsafe zip member"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_dtd_or_entity_xml() -> None:
    payload = b'<!DOCTYPE foo [<!ENTITY x "boom">]><act><text>&x;</text></act>'
    data = _archive({"texts/a.xml": payload})
    with pytest.raises(mod.ProbeError, match="DTD/entity"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_rejects_malformed_xml() -> None:
    data = _archive({"texts/a.xml": b"<act><text>broken"})
    with pytest.raises(mod.ProbeError, match="malformed XML"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_quarantines_email_and_phone_records() -> None:
    clean = _xml(_uk("clean"))
    email = _xml(_uk("email") + " contact@example.org")
    phone = _xml(_uk("phone") + " +380501234567")
    rows, report = _materialize(
        {
            "texts/clean.xml": clean,
            "texts/email.xml": email,
            "texts/phone.xml": phone,
        }
    )
    assert len(rows) == 1
    assert report["selection"]["quarantined_email_or_phone"] == 2


def test_rejects_duplicate_normalized_text() -> None:
    payload = _xml(_uk("same"))
    rows_data = {
        "texts/a.xml": payload,
        "texts/b.xml": payload.replace(b"<act>", b"<act >"),
    }
    data = _archive(rows_data)
    with pytest.raises(mod.ProbeError, match="duplicate normalized"):
        mod.materialize_archive_bytes(data, expected_md5=mod.md5(data))


def test_filters_non_ukrainian_text() -> None:
    english = ("This is a long English legal document with regulations. " * 80).strip()
    rows, report = _materialize(
        {
            "texts/ua.xml": _xml(_uk("UA")),
            "texts/en.xml": _xml(english),
        }
    )
    assert len(rows) == 1
    assert rows[0]["quality"]["ukrainian_alpha_ratio"] >= mod.MIN_UKRAINIAN_ALPHA_RATIO
    assert report["selection"]["candidate_objects_after_quality"] == 1


def test_rejects_invalid_cap() -> None:
    files = {"texts/a.xml": _xml(_uk("A"))}
    with pytest.raises(mod.ProbeError, match="byte_cap"):
        _materialize(files, byte_cap=mod.MAX_SELECTED_BYTES + 1)


def test_report_self_hash_is_stable() -> None:
    _, report = _materialize({"texts/a.xml": _xml(_uk("A"))})
    claimed = report.pop("report_identity_sha256")
    assert mod.sha256(mod.cjson(report)) == claimed
