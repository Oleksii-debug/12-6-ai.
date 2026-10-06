from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "run_d03_selected_raw_assembly_v1.py"
SPEC = importlib.util.spec_from_file_location("run_d03_selected_raw_assembly_v1", MODULE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes | Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _raw(
    record_id: str,
    source_id: str,
    family: str,
    payload: str,
    *,
    modality: str = "text",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": modality,
        "normalized_payload": payload,
    }


def _authority(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["normalized_payload"].encode("utf-8")
    return {
        "record_id": row["record_id"],
        "source_id": row["source_id"],
        "family": row["family"],
        "modality": row["modality"],
        "payload_sha256": _sha(payload),
        "payload_bytes": len(payload),
    }


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(RUNNER, "verify_plan", lambda _plan: None)
    rows = [
        _raw("base-a", "source-a", "family-a", "A"),
        _raw("missing-b", "source-b", "family-b", "BB"),
        _raw("missing-c", "source-c", "family-b", "CCC"),
    ]
    selected = [_authority(row) for row in rows]
    projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in selected
    ]
    family_bytes: dict[str, int] = {}
    for row in selected:
        family = row["family"]
        family_bytes[family] = family_bytes.get(family, 0) + row["payload_bytes"]
    plan = {
        "selected_record_count": 3,
        "selected_source_bytes": 6,
        "selected_membership_sha256": _sha(sorted(row["record_id"] for row in selected)),
        "selected_payload_projection_sha256": _sha(projection),
        "selected_family_bytes_sha256": _sha(dict(sorted(family_bytes.items()))),
        "missing_record_count": 2,
        "missing_payload_bytes": 5,
        "family_materialization_plan": [
            {
                "family": "family-b",
                "missing_record_ids": ["missing-b", "missing-c"],
                "missing_record_count": 2,
                "missing_payload_bytes": 5,
            }
        ],
        "base_raw_authority": {
            "selected_record_count": 1,
            "selected_payload_bytes": 1,
            "selected_record_ids_sha256": _sha(["base-a"]),
        },
        "composition_identity_sha256": "1" * 64,
        "balance_result_identity_sha256": "2" * 64,
    }
    return {"rows": rows, "selected": selected, "plan": plan}


def test_assemble_complete_exact_cohort_is_deterministic_and_text_free_receipt(
    synthetic: dict[str, Any],
) -> None:
    rows = synthetic["rows"]
    output, receipt = RUNNER.assemble(
        selected_rows=synthetic["selected"],
        plan=synthetic["plan"],
        base_rows=[rows[0]],
        rematerialized_rows=[rows[2], rows[1]],
    )
    assert [row["record_id"] for row in output] == [
        "base-a",
        "missing-b",
        "missing-c",
    ]
    assert receipt["selected_record_count"] == 3
    assert receipt["selected_payload_bytes"] == 6
    assert receipt["base_selected_record_count"] == 1
    assert receipt["rematerialized_record_count"] == 2
    assert receipt["raw_output_is_ephemeral"] is True
    assert receipt["durable_output_contains_raw_payload"] is False
    encoded = _canonical(receipt)
    assert b"normalized_payload" not in encoded
    assert b"BB" not in encoded
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["training_executed"] is False


def test_payload_tamper_fails_closed(synthetic: dict[str, Any]) -> None:
    rows = synthetic["rows"]
    bad = dict(rows[1])
    bad["normalized_payload"] = "BX"
    with pytest.raises(RUNNER.AssemblyError, match="payload SHA drift"):
        RUNNER.assemble(
            selected_rows=synthetic["selected"],
            plan=synthetic["plan"],
            base_rows=[rows[0]],
            rematerialized_rows=[bad, rows[2]],
        )


def test_metadata_substitution_fails_closed(synthetic: dict[str, Any]) -> None:
    rows = synthetic["rows"]
    bad = dict(rows[1])
    bad["source_id"] = "replacement-source"
    with pytest.raises(RUNNER.AssemblyError, match="source_id authority drift"):
        RUNNER.assemble(
            selected_rows=synthetic["selected"],
            plan=synthetic["plan"],
            base_rows=[rows[0]],
            rematerialized_rows=[bad, rows[2]],
        )


def test_rematerialized_replay_fails_closed(synthetic: dict[str, Any]) -> None:
    rows = synthetic["rows"]
    with pytest.raises(RUNNER.AssemblyError, match="rematerialized record replay"):
        RUNNER.assemble(
            selected_rows=synthetic["selected"],
            plan=synthetic["plan"],
            base_rows=[rows[0]],
            rematerialized_rows=[rows[1], rows[1], rows[2]],
        )


def test_missing_rematerialized_record_fails_closed(synthetic: dict[str, Any]) -> None:
    rows = synthetic["rows"]
    with pytest.raises(RUNNER.AssemblyError, match="does not exactly cover"):
        RUNNER.assemble(
            selected_rows=synthetic["selected"],
            plan=synthetic["plan"],
            base_rows=[rows[0]],
            rematerialized_rows=[rows[1]],
        )


def test_rematerialized_lane_cannot_replace_base_authority(
    synthetic: dict[str, Any],
) -> None:
    rows = synthetic["rows"]
    with pytest.raises(RUNNER.AssemblyError, match="not planned missing ID"):
        RUNNER.assemble(
            selected_rows=synthetic["selected"],
            plan=synthetic["plan"],
            base_rows=[rows[0]],
            rematerialized_rows=[rows[0], rows[1], rows[2]],
        )


def test_extra_unselected_base_rows_are_ignored_only_at_base_boundary(
    synthetic: dict[str, Any],
) -> None:
    rows = synthetic["rows"]
    unrelated = _raw("not-selected", "source-z", "family-z", "ignored")
    output, _receipt = RUNNER.assemble(
        selected_rows=synthetic["selected"],
        plan=synthetic["plan"],
        base_rows=[unrelated, rows[0]],
        rematerialized_rows=[rows[1], rows[2]],
    )
    assert [row["record_id"] for row in output] == [
        "base-a",
        "missing-b",
        "missing-c",
    ]


def test_strict_jsonl_rejects_duplicate_key_blank_line_and_nonfinite(
    tmp_path: Path,
) -> None:
    cases = {
        "duplicate": b'{"record_id":"a","record_id":"b"}\n',
        "blank": b'{"record_id":"a"}\n\n',
        "nonfinite": b'{"record_id":"a","value":NaN}\n',
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(payload)
        with pytest.raises((RUNNER.AssemblyError, json.JSONDecodeError)):
            RUNNER.load_jsonl(path, label=name)


def test_create_only_publication_refuses_existing_output(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"sentinel\n")
    with pytest.raises(RUNNER.AssemblyError, match="already exists"):
        RUNNER._write_create_only(target, b"replacement\n")
    assert target.read_bytes() == b"sentinel\n"


def test_parser_requires_rematerialized_input_and_hides_science_knobs() -> None:
    help_text = RUNNER.parser().format_help()
    assert "--rematerialized-jsonl" in help_text
    assert "--variant-seed" not in help_text
    assert "--validation-fraction" not in help_text
    assert "--tokenizer" not in help_text
