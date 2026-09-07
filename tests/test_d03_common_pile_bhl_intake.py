from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from twelve_six.data.common_pile_bhl_intake import (
    BhlIntakeError,
    iter_gzip_jsonl_bytes,
    iter_gzip_jsonl_path,
    materialize,
    materialize_twice,
    self_identity,
    validate_config,
    verify_report,
)

CONFIG_PATH = Path("configs/data/d03_common_pile_bhl_intake_v1.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def reseal(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["contract_identity_sha256"] = self_identity(value, "contract_identity_sha256")
    return value


def good_text(seed: str = "biodiversity") -> str:
    sentence = (
        f"This {seed} page describes plants, animals, specimens, habitats, "
        "collections, observations, and natural history in continuous prose. "
    )
    return sentence * 18


def record(
    *,
    item_id: str = "123456",
    page_id: str = "87654321",
    page_num: str = "0001",
    text: str | None = None,
) -> dict:
    record_id = f"{item_id}-{page_id}-{page_num}"
    return {
        "id": record_id,
        "page_id": page_id,
        "item_id": item_id,
        "page_num": page_num,
        "text": good_text(record_id) if text is None else text,
        "source": "biodiversity-heritage-library",
        "added": "2023-12-23T13:54:36.000000",
        "metadata": {
            "license": "Public Domain",
            "url": f"https://www.biodiversitylibrary.org/page/{page_id}",
        },
    }


def test_checked_in_config_is_valid_and_zero_credit() -> None:
    config = load_config()
    validate_config(config)
    assert config["training_authorized_bytes"] == 0
    assert config["authorized_unique_loss_positions"] == 0
    assert config["corpus_admitted"] is False
    assert config["rights_policy"]["per_record_language_authority_available"] is False


def test_truth_boundary_and_gate_mutations_fail_even_when_resealed() -> None:
    config = load_config()
    config["training_authorized_bytes"] = 1
    with pytest.raises(BhlIntakeError, match="training_authorized_bytes drift"):
        validate_config(reseal(config))

    config = load_config()
    config["required_downstream_gates"].remove("language_quality")
    with pytest.raises(BhlIntakeError, match="required gates drift"):
        validate_config(reseal(config))


def test_resealed_upstream_and_audit_substitutions_fail_closed() -> None:
    config = load_config()
    config["upstream"]["revision"] = "0" * 40
    config["upstream"]["download_url"] = (
        "https://huggingface.co/datasets/common-pile/biodiversity_heritage_library/"
        "resolve/" + "0" * 40 + "/v0/00000_bhl.jsonl.gz?download=true"
    )
    with pytest.raises(BhlIntakeError, match="upstream revision drift"):
        validate_config(reseal(config))

    config = load_config()
    config["common_pile_audit"]["license_whitelist_blob_sha1"] = "0" * 40
    with pytest.raises(BhlIntakeError, match="license_whitelist_blob_sha1 drift"):
        validate_config(reseal(config))


def test_machine_family_ceiling_cannot_expand_when_resealed() -> None:
    config = load_config()
    config["selection_policy"]["max_total_normalized_utf8_bytes"] = 32_000_000
    with pytest.raises(BhlIntakeError, match="max_total_normalized_utf8_bytes drift"):
        validate_config(reseal(config))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(source="other"), "source drift"),
        (lambda row: row["metadata"].update(license="CC-BY-4.0"), "license"),
        (
            lambda row: row["metadata"].update(
                url="https://www.biodiversitylibrary.org/page/99999999"
            ),
            "page URL",
        ),
        (lambda row: row.update(id="123456-99999999-0001"), "bind BHL components"),
        (lambda row: row.update(page_id="not-numeric"), "page_id must be numeric"),
    ],
)
def test_record_authority_mutations_fail_closed(mutator, message: str) -> None:
    config = load_config()
    row = record()
    mutator(row)
    with pytest.raises(BhlIntakeError, match=message):
        materialize(config, [row])


def test_extra_schema_fields_fail_closed() -> None:
    config = load_config()
    row = record()
    row["metadata"]["language"] = "english"
    with pytest.raises(BhlIntakeError, match="metadata schema drift"):
        materialize(config, [row])

    row = record()
    row["unexpected"] = True
    with pytest.raises(BhlIntakeError, match="top-level schema drift"):
        materialize(config, [row])


def test_contact_secret_noise_are_quarantined_and_clean_row_survives() -> None:
    config = load_config()
    rows = [
        record(page_id="10000001", text=good_text() + " contact person@example.com"),
        record(page_id="10000002", text=good_text() + " password=do-not-keep"),
        record(page_id="10000003", text="1234 !!! ??? " * 100),
        record(page_id="10000004", text=good_text("accepted")),
    ]
    candidates, report = materialize(config, rows)
    assert [row["page_id"] for row in candidates] == ["10000004"]
    assert report["rejection_counts"] == {
        "email_like_contact": 1,
        "low_alpha_content": 1,
        "secret_like_text": 1,
    }


def test_arbitrary_control_character_fails_closed() -> None:
    config = load_config()
    with pytest.raises(BhlIntakeError, match="control"):
        materialize(config, [record(text=good_text() + "\x00")])


def test_duplicate_record_id_fails_closed() -> None:
    config = load_config()
    first = record(text=good_text("first"))
    second = record(text=good_text("different"))
    with pytest.raises(BhlIntakeError, match="duplicate record id"):
        materialize(config, [first, second])


def test_exact_normalized_duplicate_is_quarantined() -> None:
    config = load_config()
    same = good_text("same")
    first = record(page_id="10000011", text=same)
    second = record(page_id="10000012", text=same)
    rows, report = materialize(config, [first, second])
    assert [row["page_id"] for row in rows] == ["10000011"]
    assert report["rejection_counts"] == {"exact_normalized_duplicate": 1}


def test_metadata_never_enters_candidate_text_and_language_is_not_overclaimed() -> None:
    config = load_config()
    rows, report = materialize(config, [record(text=good_text("payload"))])
    assert len(rows) == 1
    assert "Public Domain" not in rows[0]["text"]
    assert "biodiversitylibrary.org" not in rows[0]["text"]
    assert report["per_record_language_authority_available"] is False
    assert report["language_quality_gate"] == "NOT_RUN"
    assert rows[0]["training_eligible"] is False
    assert rows[0]["evaluation_eligible"] is False


def test_two_independent_materializations_must_be_identical() -> None:
    config = load_config()
    rows = [
        record(page_id="10000021", text=good_text("one")),
        record(page_id="10000022", text=good_text("two")),
    ]
    candidates, report = materialize_twice(config, rows, copy.deepcopy(rows))
    assert len(candidates) == 2
    assert report["two_independent_materializations_verified"] is True
    verify_report(config, report)

    changed = copy.deepcopy(rows)
    changed[1]["text"] = good_text("changed")
    with pytest.raises(BhlIntakeError, match="materializations differ"):
        materialize_twice(config, rows, changed)


def test_report_tamper_fails_closed() -> None:
    config = load_config()
    _, report = materialize_twice(config, [record()], [record()])
    tampered = copy.deepcopy(report)
    tampered["accepted_documents"] += 1
    with pytest.raises(BhlIntakeError, match="report identity mismatch"):
        verify_report(config, tampered)


def test_family_byte_cap_is_deterministic() -> None:
    config = load_config()
    rows = []
    for index in range(19):
        page_id = str(20000000 + index)
        text = ("A" * 248_900) + f" page-{index}"
        rows.append(record(page_id=page_id, text=text))
    candidates, report = materialize(config, rows)
    assert len(candidates) == 18
    assert report["rejection_counts"] == {"family_byte_cap": 1}
    assert report["accepted_normalized_utf8_bytes"] <= 4_500_000


def test_gzip_jsonl_parser_is_strict_and_bounded() -> None:
    rows = [record(page_id="10000031"), record(page_id="10000032")]
    payload = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
    raw = gzip.compress(payload, mtime=0)
    parsed = list(iter_gzip_jsonl_bytes(raw, max_json_line_bytes=100_000))
    assert parsed == rows

    bad = gzip.compress(b"not-json\n", mtime=0)
    with pytest.raises(BhlIntakeError, match="malformed JSONL"):
        list(iter_gzip_jsonl_bytes(bad, max_json_line_bytes=100))

    long_line = gzip.compress(b'{"x":"' + b"a" * 200 + b'"}\n', mtime=0)
    with pytest.raises(BhlIntakeError, match="line exceeds"):
        list(iter_gzip_jsonl_bytes(long_line, max_json_line_bytes=50))


def test_gzip_jsonl_path_parser_uses_same_strict_contract(tmp_path: Path) -> None:
    rows = [record(page_id="10000041"), record(page_id="10000042")]
    payload = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
    path = tmp_path / "sample.jsonl.gz"
    path.write_bytes(gzip.compress(payload, mtime=0))
    parsed = list(iter_gzip_jsonl_path(path, max_json_line_bytes=100_000))
    assert parsed == rows
