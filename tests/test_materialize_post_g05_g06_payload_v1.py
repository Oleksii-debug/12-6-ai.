from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools import materialize_post_g05_g06_payload_v1 as materializer


def _argv(tmp_path: Path, outputs: tuple[Path, Path, Path]) -> list[str]:
    return [
        "materialize_post_g05_g06_payload_v1.py",
        "--input-jsonl",
        str(tmp_path / "missing-input.jsonl"),
        "--composition-preflight",
        str(tmp_path / "missing-composition.json"),
        "--expected-composition-preflight-identity",
        "unused",
        "--g05-authority",
        str(tmp_path / "missing-g05.json"),
        "--expected-g05-execution-identity",
        "unused",
        "--g06-execution-envelope",
        str(tmp_path / "missing-g06-envelope.json"),
        "--expected-g06-envelope-identity",
        "unused",
        "--expected-g06-execution-identity",
        "unused",
        "--g06-terminal-qualification",
        str(tmp_path / "missing-g06-terminal.json"),
        "--expected-g06-terminal-qualification-identity",
        "unused",
        "--privacy-source",
        str(tmp_path / "missing-privacy.json"),
        "--provenance-quarantine",
        str(tmp_path / "missing-quarantine.json"),
        "--execution-head-sha",
        "unused",
        "--expected-materializer-implementation-git-blob-sha1",
        "unused",
        "--expected-materializer-v2-implementation-git-blob-sha1",
        "unused",
        "--output-jsonl",
        str(outputs[0]),
        "--output-inventory",
        str(outputs[1]),
        "--output-evidence",
        str(outputs[2]),
    ]


def _alias_outputs(root: Path, case: str) -> tuple[Path, Path, Path]:
    payload = root / "payload.jsonl"
    inventory = root / "inventory.json"
    evidence = root / "evidence.json"
    if case == "payload_inventory":
        return payload, payload, evidence
    if case == "payload_evidence":
        return payload, inventory, payload
    if case == "inventory_evidence":
        return payload, inventory, inventory
    if case == "all":
        return payload, payload, payload
    if case == "lexical":
        return root / "subdir" / ".." / "same.json", root / "same.json", evidence
    raise AssertionError(f"unknown alias case: {case}")


@pytest.mark.parametrize(
    "case",
    ["payload_inventory", "payload_evidence", "inventory_evidence", "all", "lexical"],
)
def test_output_aliases_fail_before_payload_read_or_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    output_root = tmp_path / "fresh-outputs"
    outputs = _alias_outputs(output_root, case)

    def unexpected_payload_read(_: Path) -> list[dict[str, object]]:
        pytest.fail("payload read occurred before aliased outputs were rejected")

    monkeypatch.setattr(materializer, "_load_jsonl", unexpected_payload_read)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, outputs))

    with pytest.raises(ValueError, match="output paths must be distinct"):
        materializer.main()

    assert not output_root.exists()
    assert all(not path.exists() for path in outputs)


def test_distinct_fresh_outputs_are_accepted_without_side_effects(tmp_path: Path) -> None:
    output_root = tmp_path / "fresh-outputs"
    outputs = (
        output_root / "payload.jsonl",
        output_root / "inventory.json",
        output_root / "evidence.json",
    )

    materializer._ensure_new(outputs)

    assert not output_root.exists()
    assert all(not path.exists() for path in outputs)


def test_preexisting_output_remains_fail_closed_and_unchanged(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    existing = output_root / "inventory.json"
    existing.write_text("sentinel\n", encoding="utf-8")
    outputs = (
        output_root / "payload.jsonl",
        existing,
        output_root / "evidence.json",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite output"):
        materializer._ensure_new(outputs)

    assert existing.read_text(encoding="utf-8") == "sentinel\n"
    assert not outputs[0].exists()
    assert not outputs[2].exists()
