from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "materialize_d03_rada_trees_rights_scoped_handoff.py"
)
spec = importlib.util.spec_from_file_location("rada_handoff", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def synthetic_config() -> dict:
    return {
        "path_provenance_policy": {
            "dated_plenary_path_regex": r"^texts/(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})__.+\.txt$",
            "minimum_year": 1990,
            "maximum_year": 2024,
            "explicit_holds": [
                {
                    "path": "texts/out-of-scope.txt",
                    "sha256": "c" * 64,
                    "size_bytes": 13,
                    "reason": "synthetic hold",
                }
            ],
        },
        "expected_result": {},
    }


def classified_fixture() -> list[dict]:
    return [
        {
            "path": "texts/2020-01-01__b.txt",
            "size_bytes": 11,
            "sha256": "a" * 64,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        },
        {
            "path": "texts/2020-01-01__a.txt",
            "size_bytes": 11,
            "sha256": "a" * 64,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        },
        {
            "path": "texts/2021-02-03__c.txt",
            "size_bytes": 12,
            "sha256": "b" * 64,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "windows-1251",
        },
        {
            "path": "texts/out-of-scope.txt",
            "size_bytes": 13,
            "sha256": "c" * 64,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        },
        {
            "path": "texts/2022-01-01__held-by-classifier.txt",
            "size_bytes": 14,
            "sha256": "d" * 64,
            "classification": "BINARY_OR_NUL_HOLD",
            "decoded_encoding": None,
        },
    ]


def bind_expected(cfg: dict, classified: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted, held = mod.rights_scoped_survivors(classified, cfg, verify_expected=False)
    cfg["expected_result"] = {
        "accepted_exact_unique_members": len(accepted),
        "accepted_exact_unique_bytes": sum(row["size_bytes"] for row in accepted),
        "held_exact_unique_members": len(held),
        "held_exact_unique_bytes": sum(row["size_bytes"] for row in held),
        "accepted_path_inventory_sha256": mod.rights.inventory_identity(accepted),
        "held_path_inventory_sha256": mod.rights.inventory_identity(held),
    }
    return accepted, held


def test_checked_in_parent_authority_is_exact_and_zero_credit() -> None:
    cfg, evidence = mod.load_parent_authority()
    assert cfg["expected_result"]["accepted_exact_unique_members"] == 4_384
    assert cfg["expected_result"]["accepted_exact_unique_bytes"] == 877_899_128
    assert evidence["claim_boundary"]["training_authorized_bytes"] == 0
    assert evidence["claim_boundary"]["model_training_executed"] is False
    assert evidence["claim_boundary"]["final_test_payload_accessed"] is False


def test_exact_duplicates_choose_lexical_survivor_and_hold_is_not_accepted() -> None:
    cfg = synthetic_config()
    classified = classified_fixture()
    expected_accepted, expected_held = bind_expected(cfg, classified)
    accepted, held = mod.rights_scoped_survivors(classified, cfg)
    assert accepted == expected_accepted
    assert held == expected_held
    assert [row["path"] for row in accepted] == [
        "texts/2020-01-01__a.txt",
        "texts/2021-02-03__c.txt",
    ]
    assert held == [
        {
            "path": "texts/out-of-scope.txt",
            "size_bytes": 13,
            "sha256": "c" * 64,
        }
    ]


def test_unrecognized_exact_unique_path_fails_closed() -> None:
    cfg = synthetic_config()
    classified = classified_fixture()
    classified.append(
        {
            "path": "texts/not-dated-or-held.txt",
            "size_bytes": 15,
            "sha256": "e" * 64,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        }
    )
    with pytest.raises(mod.HandoffError, match="not held"):
        mod.rights_scoped_survivors(classified, cfg, verify_expected=False)


def test_expected_inventory_substitution_fails_closed() -> None:
    cfg = synthetic_config()
    classified = classified_fixture()
    bind_expected(cfg, classified)
    cfg["expected_result"]["accepted_path_inventory_sha256"] = "0" * 64
    with pytest.raises(mod.HandoffError, match="accepted inventory drift"):
        mod.rights_scoped_survivors(classified, cfg)


def test_candidate_record_preserves_source_identity_and_zero_credit() -> None:
    payload = "Засідання Верховної Ради України".encode("utf-8-sig")
    row = {
        "path": "texts/2020-01-01__fixture.txt",
        "size_bytes": len(payload),
        "sha256": mod.sha256_bytes(payload),
        "date": "2020-01-01",
    }
    record = mod._candidate_record(
        row,
        payload,
        decoded_encoding="utf-8-sig",
        decode_order=["utf-8-sig", "windows-1251"],
    )
    assert record["source_family"] == mod.SOURCE_FAMILY
    assert record["source_payload_sha256"] == row["sha256"]
    assert record["session_date"] == "2020-01-01"
    assert record["text"] == "Засідання Верховної Ради України"
    assert record["training_eligible"] is False
    assert record["evaluation_eligible"] is False
    assert record["language_quality_privacy_complete"] is False


def test_candidate_record_hash_drift_fails_closed() -> None:
    payload = b"plain text"
    row = {
        "path": "texts/2020-01-01__fixture.txt",
        "size_bytes": len(payload),
        "sha256": "0" * 64,
        "date": "2020-01-01",
    }
    with pytest.raises(mod.HandoffError, match="payload SHA drift"):
        mod._candidate_record(
            row,
            payload,
            decoded_encoding="utf-8-sig",
            decode_order=["utf-8-sig", "windows-1251"],
        )


def test_candidate_record_encoding_drift_fails_closed() -> None:
    payload = "Україна".encode("utf-8")
    row = {
        "path": "texts/2020-01-01__fixture.txt",
        "size_bytes": len(payload),
        "sha256": mod.sha256_bytes(payload),
        "date": "2020-01-01",
    }
    with pytest.raises(mod.HandoffError, match="decoded encoding drift"):
        mod._candidate_record(
            row,
            payload,
            decoded_encoding="windows-1251",
            decode_order=["utf-8-sig", "windows-1251"],
        )


def test_emit_candidate_jsonl_is_deterministic_and_report_metadata_is_text_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first_path = "texts/2020-01-01__a.txt"
    second_path = "texts/2021-02-03__b.txt"
    first_payload = "Перше пленарне засідання".encode("utf-8-sig")
    second_payload = "Друге пленарне засідання".encode("windows-1251")
    for path, payload in ((first_path, first_payload), (second_path, second_payload)):
        disk = root / path
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_bytes(payload)
    accepted = [
        {
            "path": first_path,
            "size_bytes": len(first_payload),
            "sha256": mod.sha256_bytes(first_payload),
            "date": "2020-01-01",
        },
        {
            "path": second_path,
            "size_bytes": len(second_payload),
            "sha256": mod.sha256_bytes(second_payload),
            "date": "2021-02-03",
        },
    ]
    classified = [
        {
            "path": first_path,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        },
        {
            "path": second_path,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "windows-1251",
        },
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    result_a = mod.emit_candidate_jsonl(
        root,
        classified,
        accepted,
        first,
        ["utf-8-sig", "windows-1251"],
    )
    result_b = mod.emit_candidate_jsonl(
        root,
        classified,
        accepted,
        second,
        ["utf-8-sig", "windows-1251"],
    )
    assert result_a == result_b
    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line) for line in first.read_bytes().splitlines()]
    assert len(rows) == 2
    assert all(row["training_eligible"] is False for row in rows)
    assert result_a["candidate_records"] == 2
    assert result_a["accepted_source_payload_bytes"] == len(first_payload) + len(second_payload)
    assert "text" not in result_a
    assert "source_path" not in result_a


def test_emit_candidate_jsonl_removes_partial_on_failure(tmp_path: Path) -> None:
    root = tmp_path / "root"
    path = "texts/2020-01-01__a.txt"
    disk = root / path
    disk.parent.mkdir(parents=True, exist_ok=True)
    disk.write_bytes(b"wrong")
    accepted = [
        {
            "path": path,
            "size_bytes": 5,
            "sha256": "0" * 64,
            "date": "2020-01-01",
        }
    ]
    classified = [
        {
            "path": path,
            "classification": "PLAIN_TEXT_CANDIDATE",
            "decoded_encoding": "utf-8-sig",
        }
    ]
    output = tmp_path / "candidate.jsonl"
    with pytest.raises(mod.HandoffError):
        mod.emit_candidate_jsonl(
            root,
            classified,
            accepted,
            output,
            ["utf-8-sig", "windows-1251"],
        )
    assert not output.exists()
    assert not output.with_suffix(".jsonl.partial").exists()


def test_required_downstream_gates_keep_training_blocked() -> None:
    assert "LANGUAGE_QUALITY_PRIVACY" in mod.REQUIRED_DOWNSTREAM
    assert "CURRENT_GLOBAL_EXACT_NEAR_LINEAGE_DEDUP" in mod.REQUIRED_DOWNSTREAM
    assert "FRESH_RESERVED_EVALUATION_DECONTAMINATION" in mod.REQUIRED_DOWNSTREAM
    assert "POSITIVE_UNIQUE_LOSS_LEDGER" in mod.REQUIRED_DOWNSTREAM
