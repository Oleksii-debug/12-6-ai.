from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "extract_d03_ecfr_sections.py"
spec = importlib.util.spec_from_file_location("ecfr_sections", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

XML = b"""<DLPSTEXTCLASS>
<DIV1 N="Title 1" NODE="1" TYPE="TITLE"><HEAD>General Provisions</HEAD>
  <DIV3 N="Chapter I" NODE="1:1" TYPE="CHAPTER"><HEAD>Administrative Committee</HEAD>
    <DIV8 N="1.1" NODE="1:1:1" TYPE="SECTION">
      <HEAD>1.1 Purpose.</HEAD><P>This   is the first section.</P>
    </DIV8>
    <DIV8 N="1.2" NODE="1:1:2" TYPE="SECTION">
      <HEAD>1.2 Scope.</HEAD><P>Second section.</P>
    </DIV8>
  </DIV3>
</DIV1>
</DLPSTEXTCLASS>"""


def config() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "configs/data/d03_ecfr_section_extraction_v1.json"
        ).read_text(encoding="utf-8")
    )


def validate_local(candidate: dict) -> dict:
    return mod.validate_config(candidate, verify_request=False)


def extract_local(raw: bytes, candidate: dict | None = None) -> tuple[bytes, dict]:
    candidate = config() if candidate is None else candidate
    original = mod.validate_config
    try:
        mod.validate_config = lambda value, verify_request=True: original(
            value, verify_request=False
        )
        return mod.extract_sections(raw, candidate)
    finally:
        mod.validate_config = original


def test_config_is_closed_world_and_zero_credit() -> None:
    candidate = validate_local(config())
    assert candidate["source_binding"]["request_identity_sha256"] == mod.REQUEST_ID
    assert candidate["truth_boundary"]["training_authorized_bytes"] == 0
    assert candidate["truth_boundary"]["source_rights_decision_made"] is False


def test_extracts_sections_in_document_order_with_provenance() -> None:
    payload, summary = extract_local(XML)
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    assert [row["section_n"] for row in rows] == ["1.1", "1.2"]
    assert [row["node"] for row in rows] == ["1:1:1", "1:1:2"]
    assert rows[0]["head"] == "1.1 Purpose."
    assert rows[0]["normalized_text"] == "1.1 Purpose.\nThis is the first section."
    assert [a["tag"] for a in rows[0]["ancestors"]] == ["DIV1", "DIV3"]
    assert rows[0]["embedded_in_ecfr_xml"] is True
    assert rows[0]["source_rights_decision"] == "NOT_MADE"
    assert rows[0]["training_eligible"] is False
    assert summary["record_count"] == 2
    assert summary["training_authorized_bytes"] == 0


def test_two_clean_extractions_are_byte_identical() -> None:
    first_payload, first_summary = extract_local(XML)
    second_payload, second_summary = extract_local(XML)
    assert first_payload == second_payload
    assert first_summary == second_summary


def test_utf8_xml_declaration_is_accepted() -> None:
    raw = b"""<?xml version="1.0" encoding="UTF-8"?>
<DLPSTEXTCLASS><DIV1 N="Title 1" NODE="1" TYPE="TITLE">
<DIV8 N="1.1" NODE="1:1" TYPE="SECTION"><P>text</P></DIV8>
</DIV1></DLPSTEXTCLASS>"""
    payload, summary = extract_local(raw)
    assert summary["record_count"] == 1
    assert b'"section_n":"1.1"' in payload


@pytest.mark.parametrize(
    "raw",
    [
        b"<OTHER><DIV1><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV1></OTHER>",
        b"<DLPSTEXTCLASS><DIV1><DIV8 N='1.1' NODE='x' TYPE='OTHER'><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS><DIV1><DIV8 NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS><DIV1><DIV8 N='1.1' TYPE='SECTION'><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS><DIV1><DIV8 N='1.1' NODE='x' TYPE='SECTION'></DIV8></DIV1></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS>",
        b"",
    ],
)
def test_malformed_or_structurally_invalid_xml_fails_closed(raw: bytes) -> None:
    with pytest.raises(mod.ExtractionError):
        extract_local(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"<!DOCTYPE DLPSTEXTCLASS><DLPSTEXTCLASS><DIV1><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>",
        b"<!ENTITY x 'y'><DLPSTEXTCLASS><DIV1><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>",
    ],
)
def test_utf8_dtd_and_entity_declarations_fail_closed(raw: bytes) -> None:
    with pytest.raises(mod.ExtractionError, match="DTD/entity"):
        extract_local(raw)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-be", "utf-32"])
def test_alternate_encoding_dtd_entity_bypass_fails_closed(encoding: str) -> None:
    text = (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<!DOCTYPE DLPSTEXTCLASS [<!ENTITY x "EXPANDED">]>'
        '<DLPSTEXTCLASS><DIV1>'
        '<DIV8 N="1.1" NODE="x" TYPE="SECTION"><P>&x;</P></DIV8>'
        '</DIV1></DLPSTEXTCLASS>'
    )
    with pytest.raises(mod.ExtractionError, match="canonical UTF-8"):
        extract_local(text.encode(encoding))


def test_non_utf8_xml_declaration_fails_closed() -> None:
    raw = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<DLPSTEXTCLASS><DIV1><DIV8 N="1.1" NODE="x" TYPE="SECTION"><P>x</P></DIV8></DIV1></DLPSTEXTCLASS>"""
    with pytest.raises(mod.ExtractionError, match="declaration must specify UTF-8"):
        extract_local(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"<DLPSTEXTCLASS><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>orphan</P></DIV8></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS><DIV1><WRAP><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>wrapped</P></DIV8></WRAP></DIV1></DLPSTEXTCLASS>",
    ],
)
def test_section_direct_parent_must_be_div1_through_div7(raw: bytes) -> None:
    with pytest.raises(mod.ExtractionError, match="direct parent"):
        extract_local(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"<DLPSTEXTCLASS><DIV4><DIV2><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV2></DIV4></DLPSTEXTCLASS>",
        b"<DLPSTEXTCLASS><DIV3><DIV3><DIV8 N='1.1' NODE='x' TYPE='SECTION'><P>x</P></DIV8></DIV3></DIV3></DLPSTEXTCLASS>",
    ],
)
def test_section_ancestor_levels_must_strictly_increase(raw: bytes) -> None:
    with pytest.raises(mod.ExtractionError, match="strictly increasing"):
        extract_local(raw)


def test_non_div_wrapper_in_ancestor_chain_fails_closed() -> None:
    raw = b"""<DLPSTEXTCLASS><WRAP><DIV3>
    <DIV8 N="1.1" NODE="x" TYPE="SECTION"><P>x</P></DIV8>
    </DIV3></WRAP></DLPSTEXTCLASS>"""
    with pytest.raises(mod.ExtractionError, match="only DIV1..DIV7"):
        extract_local(raw)


def test_duplicate_section_node_fails_closed() -> None:
    raw = b"""<DLPSTEXTCLASS><DIV1>
    <DIV8 N="1.1" NODE="same" TYPE="SECTION"><P>a</P></DIV8>
    <DIV8 N="1.2" NODE="same" TYPE="SECTION"><P>b</P></DIV8>
    </DIV1></DLPSTEXTCLASS>"""
    with pytest.raises(mod.ExtractionError, match="duplicate section NODE"):
        extract_local(raw)


def test_multiple_direct_heads_fail_closed() -> None:
    raw = b"""<DLPSTEXTCLASS><DIV1>
    <DIV8 N="1.1" NODE="x" TYPE="SECTION"><HEAD>a</HEAD><HEAD>b</HEAD><P>c</P></DIV8>
    </DIV1></DLPSTEXTCLASS>"""
    with pytest.raises(mod.ExtractionError, match="multiple direct HEAD"):
        extract_local(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("project_authority", "request_path", "configs/data/forged.json"),
        ("project_authority", "request_git_blob_sha1", "0" * 40),
        ("source_binding", "request_identity_sha256", "0" * 64),
        ("source_binding", "title", False),
        ("xml_contract", "max_input_bytes", False),
        ("xml_contract", "root_tag", "OTHER"),
        ("xml_contract", "forbid_dtd_or_entity_declarations", False),
        ("truth_boundary", "training_authorized_bytes", False),
        ("truth_boundary", "training_authorized_bytes", 1),
        ("truth_boundary", "paid_compute_used", True),
        ("truth_boundary", "foreign_pretrained_weights", True),
    ],
)
def test_authority_drift_and_bool_aliases_fail_closed(
    section: str, key: str, value: object
) -> None:
    candidate = config()
    candidate[section][key] = value
    with pytest.raises(mod.ExtractionError):
        validate_local(candidate)


@pytest.mark.parametrize(
    "section",
    [
        "project_authority",
        "source_binding",
        "xml_contract",
        "output_contract",
        "truth_boundary",
    ],
)
def test_unknown_config_key_fails_closed(section: str) -> None:
    candidate = config()
    candidate[section]["extra"] = "drift"
    with pytest.raises(mod.ExtractionError, match="keys must be exact"):
        validate_local(candidate)


def test_root_unknown_config_key_fails_closed() -> None:
    candidate = config()
    candidate["extra"] = {}
    with pytest.raises(mod.ExtractionError, match="config root keys must be exact"):
        validate_local(candidate)


def test_request_path_substitution_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = tmp_path / "request.json"
    forged.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "REQUEST_PATH", forged)
    with pytest.raises(mod.ExtractionError, match="request path substitution"):
        mod.load_config()


def test_input_byte_bound_fails_closed() -> None:
    raw = b"x" * (mod.MAX_INPUT_BYTES + 1)
    with pytest.raises(mod.ExtractionError, match="byte bound"):
        extract_local(raw)
