from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.execute_current_survivor_privacy_g06 as runner


def _root_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _lf_root_bytes(value: object) -> bytes:
    return _root_bytes(value) + b"\n"


def _rows() -> list[dict[str, object]]:
    return [
        {
            "record_id": "r-b",
            "source_id": "source-b",
            "family": "fixture",
            "modality": "en",
            "payload_sha256": "b" * 64,
            "payload_bytes": 7,
        },
        {
            "record_id": "r-a",
            "source_id": "source-a",
            "family": "fixture",
            "modality": "uk",
            "payload_sha256": "a" * 64,
            "payload_bytes": 5,
        },
    ]


def _payload_projection(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]


def _inventory() -> tuple[dict[str, object], str, str]:
    rows = sorted(_rows(), key=lambda row: str(row["record_id"]))
    full_root = hashlib.sha256(_root_bytes(rows)).hexdigest()
    payload_root = hashlib.sha256(_root_bytes(_payload_projection(rows))).hexdigest()
    return (
        {
            "schema_version": "12-6.data526-record-inventory.v1",
            "record_count": len(rows),
            "total_payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
            "record_inventory_digest_sha256": full_root,
            "payload_inventory_digest_sha256": payload_root,
            "records": list(reversed(rows)),
        },
        full_root,
        payload_root,
    )


def _bind_fixture_constants(
    monkeypatch: pytest.MonkeyPatch,
    full_root: str,
    payload_root: str,
) -> None:
    monkeypatch.setattr(runner, "SURVIVOR_RECORD_COUNT", 2)
    monkeypatch.setattr(runner, "SURVIVOR_SOURCE_OBJECT_COUNT", 2)
    monkeypatch.setattr(runner, "SURVIVOR_TOTAL_PAYLOAD_BYTES", 12)
    monkeypatch.setattr(runner, "SURVIVOR_RECORD_INVENTORY_SHA256", full_root)
    monkeypatch.setattr(runner, "SURVIVOR_PAYLOAD_INVENTORY_SHA256", payload_root)


def _write_inventory(tmp_path: Path, inventory: dict[str, object]) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return path


def test_validate_inventory_matches_data526_no_lf_root_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, full_root, payload_root = _inventory()
    _bind_fixture_constants(monkeypatch, full_root, payload_root)

    validated, expected_g06_root = runner._validate_inventory(
        _write_inventory(tmp_path, inventory)
    )

    assert validated["record_inventory_digest_sha256"] == full_root
    assert validated["payload_inventory_digest_sha256"] == payload_root
    assert isinstance(expected_g06_root, str)
    assert len(expected_g06_root) == 64


def test_validate_inventory_rejects_lf_mutated_record_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, full_root, payload_root = _inventory()
    _bind_fixture_constants(monkeypatch, full_root, payload_root)
    normalized = sorted(_rows(), key=lambda row: str(row["record_id"]))
    inventory["record_inventory_digest_sha256"] = hashlib.sha256(
        _lf_root_bytes(normalized)
    ).hexdigest()

    with pytest.raises(runner.CurrentSurvivorG06Error, match="self-root mismatch"):
        runner._validate_inventory(_write_inventory(tmp_path, inventory))


def test_validate_inventory_rejects_tampered_row_under_original_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, full_root, payload_root = _inventory()
    _bind_fixture_constants(monkeypatch, full_root, payload_root)
    tampered = copy.deepcopy(inventory)
    tampered["records"][0]["payload_sha256"] = "c" * 64

    with pytest.raises(runner.CurrentSurvivorG06Error, match="self-root mismatch"):
        runner._validate_inventory(_write_inventory(tmp_path, tampered))
