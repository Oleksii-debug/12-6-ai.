from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from tools import qualify_next100_048_pydantic as qualifier

ROOT = Path(__file__).resolve().parents[1]


def test_committed_pydantic_terminal_evidence_is_self_consistent() -> None:
    value = json.loads(
        (ROOT / "evidence/next100-048/pydantic-source-admission-v1.json").read_text(
            encoding="utf-8"
        )
    )

    qualifier.verify_evidence(value)

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
