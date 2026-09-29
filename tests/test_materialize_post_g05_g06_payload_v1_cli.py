from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_cli() -> ModuleType:
    script = (
        Path(__file__).parents[1] / "tools" / "materialize_post_g05_g06_payload_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_post_g05_g06_payload_v1_cli",
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, outputs: tuple[Path, Path, Path]) -> list[str]:
    missing = tmp_path / "missing-inputs"
    return [
        "materialize_post_g05_g06_payload_v1.py",
        "--input-jsonl",
        str(missing / "input.jsonl"),
        "--composition-preflight",
        str(missing / "composition.json"),
        "--expected-composition-preflight-identity",
        "composition-id",
        "--g05-authority",
        str(missing / "g05.json"),
        "--expected-g05-execution-identity",
        "g05-id",
        "--g06-execution-envelope",
        str(missing / "g06-envelope.json"),
        "--expected-g06-envelope-identity",
        "g06-envelope-id",
        "--expected-g06-execution-identity",
        "g06-execution-id",
        "--g06-terminal-qualification",
        str(missing / "g06-terminal.json"),
        "--expected-g06-terminal-qualification-identity",
        "g06-terminal-id",
        "--privacy-source",
        str(missing / "privacy.py"),
        "--execution-head-sha",
        "0" * 40,
        "--expected-materializer-implementation-git-blob-sha1",
        "1" * 40,
        "--expected-materializer-v2-implementation-git-blob-sha1",
        "2" * 40,
        "--output-jsonl",
        str(outputs[0]),
        "--output-inventory",
        str(outputs[1]),
        "--output-evidence",
        str(outputs[2]),
    ]


def _outputs_for_case(tmp_path: Path, case: str) -> tuple[Path, Path, Path]:
    root = tmp_path / "outputs"
    payload = root / "payload.jsonl"
    inventory = root / "inventory.json"
    evidence = root / "evidence.json"
    if case == "payload_inventory":
        return payload, payload, evidence
    if case == "payload_evidence":
        return payload, inventory, payload
    if case == "inventory_evidence":
        return payload, inventory, inventory
    if case == "all_equal":
        return payload, payload, payload
    if case == "lexical_alias":
        alias = root / "unused" / ".." / "payload.jsonl"
        return payload, alias, evidence
    raise AssertionError(f"unknown case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "payload_inventory",
        "payload_evidence",
        "inventory_evidence",
        "all_equal",
        "lexical_alias",
    ],
)
def test_aliases_fail_before_payload_access_or_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    cli = _load_cli()
    outputs = _outputs_for_case(tmp_path, case)

    def fail_if_payload_is_read(_path: Path) -> list[dict[str, object]]:
        raise AssertionError("payload input was read before output alias rejection")

    monkeypatch.setattr(cli, "_load_jsonl", fail_if_payload_is_read)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, outputs))

    with pytest.raises(FileExistsError, match="outputs must be distinct"):
        cli.main()

    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "missing-inputs").exists()


def test_three_distinct_fresh_outputs_keep_existing_preflight_behavior(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    root = tmp_path / "fresh"
    outputs = (
        root / "payload.jsonl",
        root / "inventory.json",
        root / "evidence.json",
    )

    cli._ensure_new(outputs)

    assert not root.exists()


def test_preexisting_output_is_still_rejected(tmp_path: Path) -> None:
    cli = _load_cli()
    existing = tmp_path / "existing.json"
    existing.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite output"):
        cli._ensure_new(
            (
                tmp_path / "payload.jsonl",
                existing,
                tmp_path / "evidence.json",
            )
        )

    assert existing.read_text(encoding="utf-8") == "sentinel"
