from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "materialize_d03_common_pile_ubuntu_irc_v1.py"
)
spec = importlib.util.spec_from_file_location("common_pile_ubuntu_irc", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONFIG = mod.load_config()


def english_text(suffix: str = "") -> str:
    return (
        "[10:00] <alice> I am debugging an Ubuntu package installation and the service "
        "fails only after a deterministic restart. [10:01] <bob> Check the journal logs, "
        "compare the package version, and reproduce the command in a clean environment. "
        "[10:02] <alice> That explains the failure; the configuration file was stale and "
        "the corrected setting now survives a reboot. "
        f"{suffix}"
    )


def make_row(record_id: str = "2015-04-17-#ubuntu-test", suffix: str = "") -> dict:
    return {
        "id": record_id,
        "text": english_text(suffix),
        "source": "ubuntu-chat",
        "added": "2024-05-13T22:10:59.032680",
        "created": "2015-04-17T00:00:00",
        "metadata": {
            "license": "Public Domain",
            "authors": ["alice", "bob"],
            "url": "https://irclogs.ubuntu.com/2015/04/17/%23ubuntu-test.txt",
            "channel": "#ubuntu-test",
        },
    }


def test_config_binds_claim_rights_and_immutable_sha() -> None:
    cfg = mod.load_config()
    assert cfg["claim_issue"] == 905
    assert cfg["parent_rights"]["source_key"] == "ubuntu_irc"
    assert cfg["source"]["revision"] == mod.SOURCE_REVISION
    assert cfg["source"]["sha256"] == mod.SOURCE_SHA256
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["training_eligible"] is False


def test_source_identity_mutation_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["source"]["sha256"] = "0" * 64
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.UbuntuIrcIntakeError, match="immutable source contract drift"):
        mod.load_config(path)


def test_rights_binding_mutation_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["parent_rights"]["required_rights_signal"] = "CC-BY"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.UbuntuIrcIntakeError, match="parent rights binding drift"):
        mod.load_config(path)


def test_valid_row_is_normalized_and_retained() -> None:
    row = make_row()
    row["text"] = "  " + row["text"].replace(" ", "  ") + "\r\n"
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "\r" not in text
    assert "  " not in text


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda row: row.update(source="other-source"), "source_label_mismatch"),
        (
            lambda row: row["metadata"].update(license="CC BY"),
            "metadata_rights_or_origin_mismatch",
        ),
        (
            lambda row: row["metadata"].update(url="https://example.org/chat.txt"),
            "metadata_rights_or_origin_mismatch",
        ),
        (lambda row: row.update(text=row["text"] + " dev@example.org"), "email"),
        (lambda row: row.update(text=row["text"] + " +1 415 555 0123"), "phone"),
        (lambda row: row.update(text=row["text"] + " 192.168.1.44"), "ipv4"),
        (lambda row: row.update(text=row["text"] + " token=abcdef123456"), "secret_marker"),
        (lambda row: row.update(text=row["text"] + "\x00"), "control_character"),
        (lambda row: row.update(text="Short chat."), "too_short"),
        (
            lambda row: row.update(
                text=("[10:00] <u> Привіт, це український тест повідомлення. " * 20)
            ),
            "low_latin_alpha_ratio",
        ),
    ],
)
def test_row_level_filters_quarantine_without_credit(mutator, reason: str) -> None:
    row = make_row()
    mutator(row)
    ok, actual, text = mod.assess_row(row, CONFIG)
    assert ok is False
    assert actual == reason
    assert text == ""


def test_structural_schema_drift_is_source_fatal() -> None:
    row = make_row()
    row["unexpected"] = "x"
    with pytest.raises(mod.UbuntuIrcIntakeError, match="source row field drift"):
        mod.assess_row(row, CONFIG)


def test_duplicate_source_id_is_source_fatal() -> None:
    first = make_row()
    second = copy.deepcopy(first)
    second["text"] = english_text("different text that remains long enough")
    with pytest.raises(mod.UbuntuIrcIntakeError, match="duplicate source record id"):
        mod.select_rows([first, second], CONFIG)


def test_selection_is_deterministic_deduplicated_and_record_bounded() -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["selection"]["max_scanned_records"] = 5
    cfg["selection"]["max_retained_records"] = 2

    first = make_row("2015-04-17-#one", "first unique record")
    duplicate = make_row("2015-04-17-#two", "first unique record")
    third = make_row("2015-04-17-#three", "third unique record")
    fourth = make_row("2015-04-17-#four", "fourth unique record")
    fifth = make_row("2015-04-17-#five", "fifth unique record")
    sixth = make_row("2015-04-17-#six", "sixth unscanned record")

    accepted, reasons, scanned = mod.select_rows(
        [first, duplicate, third, fourth, fifth, sixth],
        cfg,
    )
    assert scanned == 5
    assert [row["record_id"] for row in accepted] == ["2015-04-17-#one", "2015-04-17-#three"]
    assert reasons["accepted"] == 2
    assert reasons["exact_normalized_duplicate"] == 1
    assert reasons["retained_record_cap_reached"] == 2
    assert sum(reasons.values()) == 5


def test_selection_honors_retained_byte_cap() -> None:
    cfg = copy.deepcopy(CONFIG)
    first = make_row("2015-04-17-#one", "first")
    second = make_row("2015-04-17-#two", "second distinct")
    first_bytes = len(mod.normalize(first["text"]).encode("utf-8"))
    cfg["selection"]["max_retained_normalized_utf8_bytes"] = first_bytes
    accepted, reasons, scanned = mod.select_rows([first, second], cfg)
    assert scanned == 2
    assert len(accepted) == 1
    assert reasons["retained_byte_cap_reached"] == 1


def test_gzip_jsonl_reader_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(make_row()).encode() + b"\n")
        handle.write(b"not-json\n")
    iterator = iter(mod._iter_gzip_jsonl(path))
    assert next(iterator)["source"] == "ubuntu-chat"
    with pytest.raises(mod.UbuntuIrcIntakeError, match="source JSONL is malformed"):
        next(iterator)


def test_verify_exact_source_rejects_wrong_payload(tmp_path: Path) -> None:
    path = tmp_path / "wrong.gz"
    path.write_bytes(b"not the immutable source")
    with pytest.raises(mod.UbuntuIrcIntakeError, match="source SHA-256 mismatch"):
        mod.verify_exact_source(path)


def test_candidate_payload_excludes_all_source_metadata() -> None:
    accepted, _, _ = mod.select_rows([make_row()], CONFIG)
    payload = mod.candidate_payload(accepted).decode("utf-8")
    assert "authors" not in payload
    assert "irclogs.ubuntu.com" not in payload
    assert "#ubuntu-test.txt" not in payload
    assert '"training_eligible":false' in payload
    assert '"evaluation_eligible":false' in payload


def test_report_is_text_free_zero_credit_evidence() -> None:
    accepted = [
        {
            "record_id": "2015-04-17-#test",
            "source_key": "ubuntu_irc",
            "source_label": "ubuntu-chat",
            "normalized_sha256": "a" * 64,
            "normalized_bytes": 321,
            "training_eligible": False,
            "evaluation_eligible": False,
            "text": "secret source text not included in durable report",
        }
    ]
    candidate = b"ephemeral candidate bytes\n"
    report = mod.build_report(
        CONFIG,
        rights_registry_identity="b" * 64,
        accepted=accepted,
        reasons=Counter({"accepted": 1}),
        scanned=1,
        candidate=candidate,
        observed_source_bytes=123,
    )
    encoded = json.dumps(report)
    assert "secret source text" not in encoded
    assert report["retained_records"] == 1
    assert report["retained_normalized_bytes"] == 321
    assert report["training_authorized_bytes"] == 0
    assert report["canonical_capacity_credited"] == 0
    assert report["family_credit_added"] == 0
    assert report["unique_causal_loss_positions_authorized"] == 0
    assert report["training_eligible"] is False
    assert report["evaluation_eligible"] is False
    assert report["model_training_executed"] is False
    assert report["final_test_accessed"] is False
    assert report["paid_compute_used"] is False
    assert report["universal_pii_absence_claimed"] is False
