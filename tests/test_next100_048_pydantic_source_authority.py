from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from tools import qualify_next100_048_pydantic as qualifier

ROOT = Path(__file__).resolve().parents[1]


def test_committed_pydantic_terminal_evidence_is_self_consistent() -> None:
    path = ROOT / "evidence/next100-048/pydantic-source-admission-v1.json"
    value = qualifier._load_json_bytes(path.read_bytes(), context=str(path))

    qualifier.verify_historical_evidence(value)

    predecessor = value["predecessor_code_authority"]
    assert predecessor["data227_head_sha"] == qualifier.DATA227_HEAD
    assert predecessor["rights_policy_git_blob_sha1"] == qualifier.DATA227_POLICY_BLOB
    assert value["upstream"]["commit"] == qualifier.UPSTREAM_COMMIT
    assert value["license"]["git_blob_sha1"] == qualifier.LICENSE_BLOB
    assert value["source_family_accounting"]["new_source_family"] == "github:pydantic/pydantic"
    assert value["source_family_accounting"]["selected_authored_capacity_bytes"] == 235_204


def test_data227_policy_fallback_is_pinned_and_bounded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    raw = b'{"decisions":[]}\n'
    expected_blob = qualifier.git_blob_sha1(raw)
    observed: list[tuple[str, int]] = []

    def fake_download(url: str, max_bytes: int = 300_000) -> bytes:
        observed.append((url, max_bytes))
        return raw

    monkeypatch.setattr(qualifier, "DATA227_POLICY_BLOB", expected_blob)
    monkeypatch.setattr(qualifier, "download", fake_download)

    value = qualifier.load_data227_policy(tmp_path)

    assert value == {"decisions": []}
    assert observed == [
        (
            "https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./"
            + qualifier.DATA227_HEAD
            + "/configs/data/data227_code_rights_policy_v1.json",
            100_000,
        )
    ]



@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"x","schema_version":"y"}',
        b'{"schema_version":NaN}',
        b'{"schema_version":Infinity}',
        b'{"schema_version":1e400}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(raw: bytes) -> None:
    with pytest.raises(
        qualifier.QualificationError,
        match="duplicate JSON key|non-finite JSON number",
    ):
        qualifier._load_json_bytes(raw, context="adversarial")


def test_generated_and_historical_evidence_use_distinct_schema_versions() -> None:
    assert qualifier.HISTORICAL_SCHEMA == (
        "12-6.next100-048-pydantic-source-admission.v1"
    )
    assert qualifier.SCHEMA == "12-6.next100-048-pydantic-source-admission.v2"
